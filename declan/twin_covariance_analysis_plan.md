Below is a coding-agent handoff you can paste directly into Claude Code or another implementation agent.

---

# Coding-agent handoff: digital-twin reafferent covariance structure analyses

## Objective

Implement a focused set of **digital-twin covariance-structure analyses** for the foveal V1 FEM paper.

The purpose is **not** to test whether FEMs improve coding, enhance information, optimize sampling, or help discrimination. The purpose is to explain the structure of the recorded reafferent covariance using a deterministic, noiseless image-computable population model.

The core claim to support is:

> A substantial component of foveal V1 shared variability during fixation is reafferent: the population footprint of retinal image translation under the animal’s own eye movements. The digital twin explains why this reafferent covariance is low-dimensional, translation-structured, stimulus-specific, and sometimes aligned with stimulus-driven response dimensions.

The prescription document defines the twin’s role clearly: the recording shows that reafferent covariance exists and dominates, while the twin explains why that covariance has the structure it has. In the deterministic twin, the only source of across-trial variance is eye position, so its across-trial covariance is the reafferent covariance. 

Do **not** implement functional or performance analyses in this handoff. The prescription explicitly excludes claims about whether covariance helps or hurts coding, whether it is information-limiting in magnitude, how much an eye-aware decoder recovers, or whether FEMs improve discrimination, because those questions require a noise model. 

---

## Required outputs

Create a new analysis module and runnable script, preferably:

```text
VisionCore/
  declan/
    twin_covariance_structure/
      run_twin_covariance_structure.py
      covariance_core.py
      subspace_metrics.py
      eye_controls.py
      plotting.py
      README.md
```

The final run should write:

```text
outputs/twin_covariance_structure/
  config.json
  summary.csv
  per_image_metrics.csv
  per_condition_metrics.csv
  cfem_cache.npz or cfem_cache.pkl
  figures/
    A1_signal_alignment.png/pdf/svg
    A2_rank_mechanism.png/pdf/svg
    A3_image_specificity.png/pdf/svg
    A4_translation_tangent_alignment.png/pdf/svg
    A5_occupancy_vs_dynamics.png/pdf/svg
    A6_single_unit_to_population_bridge.png/pdf/svg
  README.md
```

The README must end with an interpretation block:

```text
Final structural status:
- A1_signal_alignment: supported / mixed / failed / not_run
- A2_low_rank_translation_dof: supported / mixed / failed / not_run
- A3_image_specificity: supported / mixed / failed / not_run
- A4_translation_tangent_alignment: supported / mixed / failed / not_run
- A5_occupancy_not_dynamics: supported / mixed / failed / not_run
- A6_single_unit_population_bridge: supported / mixed / failed / not_run

Scope statement:
These analyses test deterministic covariance geometry only. They do not test whether FEMs improve coding, optimize sampling, increase information, or help discrimination.
```

---

## Scientific guardrails

Hard constraints:

1. Use the term **reafferent covariance**, not “active-sensing benefit,” “efficient sampling,” or “information gain.”
2. Report **covariance geometry** only: rank, eigenspectrum, subspace overlap, translation-tangent alignment, image specificity, occupancy dependence.
3. For Jacobian/tangent analyses, report **alignment only**. Do not resurrect the failed magnitude identity `C_FEM ≈ JΣ_eyeJᵀ`. The prescription explicitly says any return to the magnitude comparison is the old error returning. 
4. Do not implement decoders, bits/spike, accuracy, Fisher information, or ideal observers in this task.
5. If bridging to recorded data, run a coordinate-frame audit first. A y-sign mismatch can suppress alignment, and the prescription states that this is prerequisite for quantitative model-data bridging. 

---

# Core definitions

For each image or stimulus condition `I`, define the deterministic model response:

```text
r(I, e, t) : population response vector for image I, eye position e, time t
```

Define twin FEM covariance:

```text
C_FEM(I) = E_t[ Cov_e( r(I, e, t) | t ) ]
```

This mirrors the recorded FEM covariance, but in the twin it is noise-free.

Operationally:

1. Choose a fixed image/stimulus `I`.
2. Replay a set of eye positions or eye traces over that image.
3. For each time point `t`, compute model responses across eye samples.
4. Compute covariance across eye samples at matched `t`.
5. Average those covariance matrices over `t`.

The prescription’s key distinction is that this is a **second-moment occupancy object**: it depends on where the eye dwells, not on temporal trajectory order. Two eye ensembles with the same per-time or marginal occupancy should produce the same covariance geometry, even if their temporal dynamics differ. 

---

# Infrastructure tasks

## Task 0.1: Load model, dataset, eye traces, and images

Reuse existing VisionCore infrastructure. The repo bundle shows the current pattern:

```python
from utils import get_model_and_dataset_configs
model, dataset_configs = get_model_and_dataset_configs(mode)
model = model.to(device)
```

and the existing analysis stack already uses `prepare_data`, `run_mcfarland_on_dataset`, `extract_metrics`, and fixRSVP dataset extraction patterns. 

Implement a loader that returns:

```python
model
device
dataset_config
stimuli_or_images
eye_traces
unit_ids
metadata
```

Requirements:

* Run model in `eval()` mode.
* Use deterministic inference, no dropout/no noise.
* Keep unit subset fixed across all analyses.
* Cache model responses aggressively. These analyses will repeatedly evaluate the same images under different eye-position ensembles.

Acceptance checks:

```text
[PASS] model loads on selected device
[PASS] N images selected
[PASS] N eye traces selected
[PASS] response array has shape (n_images, n_eye_samples, n_time, n_units) or equivalent
[PASS] no nonfinite model responses
[PASS] nonzero response variance across eye positions for at least most images
```

---

## Task 0.2: Implement covariance primitives

Create `covariance_core.py`.

Required functions:

```python
def center_response(R, axis):
    """
    Mean-center response array along specified axis.
    """

def compute_cfem_for_image(R):
    """
    Parameters
    ----------
    R : array, shape (n_eye, n_time, n_units)
        Model responses for one image under multiple eye positions/traces.

    Returns
    -------
    C : array, shape (n_units, n_units)
        E_t[Cov_eye(r | t)]
    per_t_covs : optional, shape (n_time, n_units, n_units)
    """

def compute_signal_covariance(mu_images):
    """
    Parameters
    ----------
    mu_images : array, shape (n_images, n_units)
        Mean response for each image.

    Returns
    -------
    C_signal : array, shape (n_units, n_units)
    """

def eigensystem(C, eps=1e-12):
    """
    Symmetrize C, eigen-decompose descending, clip tiny negative eigenvalues if numerical.
    """

def participation_ratio(evals):
    """
    PR = (sum lambda)^2 / sum(lambda^2)
    """

def top_subspace(evecs, k):
    """
    Return first k eigenvectors as orthonormal basis.
    """
```

Use symmetric covariance:

```python
C = 0.5 * (C + C.T)
```

Acceptance checks:

```text
[PASS] C_FEM is symmetric within tolerance
[PASS] eigenvalues are mostly nonnegative, tiny negatives only numerical
[PASS] PR is finite
[PASS] shuffled or constant-eye control reduces C_FEM trace strongly
```

---

## Task 0.3: Implement subspace metrics

Create `subspace_metrics.py`.

Required metrics:

```python
def projection_matrix(U):
    return U @ U.T

def subspace_overlap(U, V):
    """
    Return normalized overlap between subspaces.
    Suggested: trace(P_U P_V) / min(dim(U), dim(V)).
    Range [0, 1].
    """

def variance_captured(C, U):
    """
    Fraction of covariance C captured by subspace U:
    trace(U.T @ C @ U) / trace(C)
    """

def directional_variance_capture(C_source, U_target):
    """
    Alias for variance_captured, but explicit naming for figure labels.
    """

def principal_angles(U, V):
    """
    Return principal angles and cosines.
    """
```

Include dimensional sweeps:

```python
for k in [1, 2, 3, 5, 10]:
    overlap_k = subspace_overlap(U_cfem[:, :k], U_signal[:, :k])
    fem_by_signal_k = variance_captured(C_FEM, U_signal[:, :k])
    signal_by_fem_k = variance_captured(C_signal, U_FEM[:, :k])
```

The current recorded draft reports low-dimensional FEM covariance and stimulus/FEM subspace alignment: FEM participation ratio around 2.1, and directional variance capture values around X ≈ 0.67 and Y ≈ 0.75. These are recorded-data anchors, not exact model targets. 

---

# Analysis A1: Signal alignment as expected image-motion covariance

## Aim

Show that alignment between reafferent covariance and stimulus-driven response structure is expected if covariance is generated by retinal image motion. This addresses the reviewer-sensitive point that signal-aligned variability is not automatically “bad internal noise”; it can be the natural footprint of image translation.

The prescription marks A1 as the lead result and says it should adjudicate the mismatch between earlier ~0.77 and recent ~0.29 alignment values by producing the expected pure-reafference alignment under each metric. 

## Implementation

For a set of images `I = 1..N`:

1. Compute `C_FEM(I)` for each image using real or sampled eye-position cloud.
2. Compute a stimulus-driven covariance `C_signal`.

Primary definition:

```text
C_signal = Cov_I( mean_e,t r(I, e, t) )
```

Robustness definition:

```text
C_signal_local(I) = Cov over nearby images / transformations / stimulus variants
```

3. Compute:

```python
U_fem_I = top eigenvectors of C_FEM(I)
U_signal = top eigenvectors of C_signal

overlap_k[I, k] = subspace_overlap(U_fem_I[:, :k], U_signal[:, :k])
fem_var_by_signal[I, k] = variance_captured(C_FEM(I), U_signal[:, :k])
signal_var_by_fem[I, k] = variance_captured(C_signal, U_fem_I[:, :k])
```

4. Generate nulls:

```text
unit-shuffle null: shuffle neuron labels independently across images or covariance matrices
random-subspace null: random orthonormal subspaces with same k and n_units
image-label null: permute image identities before constructing C_signal, if applicable
```

## Outputs

`per_image_metrics.csv` columns:

```text
image_id
cfem_trace
cfem_pr
k
overlap_fem_signal
fem_variance_captured_by_signal
signal_variance_captured_by_fem
null_mean
null_lo
null_hi
```

Figure:

* Panel 1: overlap vs k, real vs null.
* Panel 2: FEM variance captured by signal subspace vs k.
* Panel 3: signal variance captured by FEM subspace vs k.
* Panel 4: per-image scatter of PR vs alignment.

## Acceptance criteria

A1 is **supported** if FEM-signal alignment is reliably above random-subspace/unit-shuffle null for at least the leading 1–5 dimensions.

A1 is **mixed** if alignment depends strongly on the definition of `S`.

A1 is **failed** if alignment is indistinguishable from null under all sensible definitions.

---

# Analysis A2: Low rank and what sets the rank

## Aim

Show that low rank arises because retinal translation has two spatial degrees of freedom, and quantify the excess rank above 2 as finite-cloud curvature.

The prescription frames this as essential: prior work asserted “low-rank because 2D translation,” while this analysis tests it by manipulating eye-motion degrees of freedom and cloud radius. 

## Implementation

For each image:

Compute `C_FEM(I)` under eye clouds:

```text
condition real_2d: real eye positions or real trace occupancy
condition x_only: vary x, hold y fixed
condition y_only: vary y, hold x fixed
condition line_random_angle: vary eye position along one random 1D axis
condition isotropic_2d_small: Gaussian/dither cloud with small radius
condition isotropic_2d_medium
condition isotropic_2d_large
```

For each condition:

```python
evals, evecs = eigensystem(C)
pr = participation_ratio(evals)
trace = np.sum(evals)
frac_top1 = evals[0] / trace
frac_top2 = evals[:2].sum() / trace
frac_top5 = evals[:5].sum() / trace
```

Radius sweep:

```text
radii_arcmin = [1, 2, 4, 8, 12, 16]  # adjust to actual eye-cloud scale
```

Use the same number of eye samples per condition.

## Outputs

`per_condition_metrics.csv`:

```text
image_id
condition
radius_arcmin
eye_dof
cfem_trace
pr
frac_top1
frac_top2
frac_top5
lambda1
lambda2
lambda3
```

Figure:

* PR by condition: 1D vs 2D.
* Eigenspectrum for representative images.
* PR vs cloud radius.
* Fraction variance in top 2 vs cloud radius.

## Acceptance criteria

A2 is **supported** if:

```text
1D eye variation gives PR near 1
2D small-cloud variation gives PR near 2
larger cloud radii increase PR modestly above 2
top two modes explain most covariance for real FEM-scale clouds
```

A2 is **mixed** if PR is low-rank but not DOF-sensitive.

A2 is **failed** if covariance rank is high and not controlled by eye-motion dimensionality.

---

# Analysis A3: Stimulus specificity, not global state

## Aim

Show that the reafferent covariance subspace rotates with image content, ruling out a simple global gain, arousal, or image-invariant state mode explanation.

The prescription notes that this reuses the same logic as the prior cross-image displacement-decoding result: within-image displacement structure generalizes strongly within an image but fails or anti-generalizes across images. The new framing is positive: image-specificity is expected for reafferent covariance and distinguishes it from global state. 

## Implementation

For each image:

```python
C_i = C_FEM(I_i)
U_i = top_subspace(C_i, k=2 or 3)
```

Compute all pairwise overlaps:

```python
overlap_ij = subspace_overlap(U_i, U_j)
```

Compare to:

```text
within-image split-half overlap:
    compute C_i from half eye samples A and half eye samples B

cross-image overlap:
    U_i vs U_j, i != j

global-state null:
    construct rank-1 or rank-2 global gain covariance vector, compare image invariance
```

Optional: group by image similarity if image metadata are available.

## Outputs

Figure:

* Heatmap of image × image FEM subspace overlap.
* Histogram of within-image split-half overlap vs cross-image overlap.
* Example top FEM eigenvectors projected into a shared low-D visualization.

## Acceptance criteria

A3 is **supported** if:

```text
within-image split-half overlap >> cross-image overlap
cross-image overlap is low or broad
```

A3 is **mixed** if some image classes share covariance subspaces.

A3 is **failed** if a single global subspace explains most `C_FEM(I)` across images.

---

# Analysis A4: Covariance subspace is the translation tangent

## Aim

Show that the dominant `C_FEM(I)` directions align with the local retinal-translation tangent:

```text
T_I = span{∂r(I,e)/∂x, ∂r(I,e)/∂y}
```

This is the safe surviving version of the Jacobian idea. The prescription is explicit: use `J` only to define directions, not magnitudes. 

## Implementation

For each image:

1. Define cloud center `e0`, usually mean eye position or zero.
2. Compute finite-difference translation derivatives:

```python
dr_dx = (r(I, e0 + [delta, 0]) - r(I, e0 - [delta, 0])) / (2 * delta)
dr_dy = (r(I, e0 + [0, delta]) - r(I, e0 - [0, delta])) / (2 * delta)
J = np.stack([dr_dx, dr_dy], axis=1)  # n_units x 2
T = orthonormalize(J)
```

3. Compute:

```python
U_fem = top_subspace(C_FEM(I), k=2)
tangent_overlap = subspace_overlap(U_fem, T)
fem_var_by_tangent = variance_captured(C_FEM(I), T)
```

4. Radius sweep:

Compute `C_FEM(I, radius)` for increasing eye-cloud radii, then measure alignment to the local tangent at center.

## Outputs

Figure:

* Tangent overlap distribution across images.
* FEM variance captured by tangent.
* Tangent alignment vs eye-cloud radius.
* Example schematic: top FEM covariance plane vs local translation tangent.

## Acceptance criteria

A4 is **supported** if:

```text
top FEM subspace aligns above null with T_I
alignment is strongest at small cloud radius
alignment decays gradually with radius rather than collapsing immediately
```

A4 is **mixed** if alignment is image-dependent.

A4 is **failed** if FEM subspace is not aligned with the local translation tangent above null.

Hard stop:

```text
Do not compute or report norm(C_FEM - JΣJᵀ), matrix reconstruction error, or magnitude identity.
```

---

# Analysis A5: Occupancy, not trajectory dynamics

## Aim

Demonstrate that the structure of `C_FEM` is governed by eye-position occupancy, not temporal order.

The prescription states the expected comparison clearly: real eye traces should match random dither with matched occupancy, but differ from amplitude-matched dither that does not match occupancy shape. 

## Implementation

Construct three eye ensembles:

```text
real_trace:
    actual eye positions in actual order

occupancy_matched_shuffle:
    same set of eye positions, randomly permuted in time or resampled iid from empirical occupancy

amplitude_matched_gaussian:
    random cloud with matched RMS amplitude but Gaussian/isotropic occupancy

amplitude_matched_ring or uniform:
    optional stronger mismatch control
```

For each image and ensemble:

```python
C = compute_cfem_for_image(R_ensemble)
PR
top eigenvectors
subspace overlap to real_trace C_FEM
signal alignment metrics from A1
tangent alignment metrics from A4
```

Primary comparisons:

```python
overlap_real_vs_occupancy_matched
overlap_real_vs_amplitude_matched
abs(PR_real - PR_occupancy_matched)
abs(PR_real - PR_amplitude_matched)
```

## Outputs

Figure:

* Real vs occupancy-matched vs amplitude-matched PR.
* Real-vs-control subspace overlap.
* Alignment metrics under each control.
* Optional example eye-position clouds.

## Acceptance criteria

A5 is **supported** if:

```text
real_trace ≈ occupancy_matched_shuffle
real_trace differs from amplitude_matched_gaussian/uniform
```

A5 is **mixed** if temporal order has small effects due to model temporal history but occupancy still dominates.

A5 is **failed** if temporal order strongly changes covariance geometry even after matching occupancy.

Important nuance:

The model has temporal processing, so exact equality may not hold if the response depends on history. If differences occur, include an additional static/history-controlled mode if possible:

```text
history_reset or single-frame mode:
    evaluate responses with matched instantaneous eye position but controlled temporal context
```

---

# Analysis A6: Single-neuron translation tuning to population covariance

## Aim

Bridge single-neuron FEM sensitivity to population reafferent covariance.

The prescription frames this as a novelty-positioning analysis: show how single-neuron translation gains aggregate into low-rank population covariance. 

## Implementation

For each image, define finite-cloud translation gain per unit.

Option 1, derivative gain:

```python
gain_x[n] = dr_dx[n]
gain_y[n] = dr_dy[n]
gain_mag[n] = sqrt(gain_x[n]^2 + gain_y[n]^2)
gain_angle[n] = atan2(gain_y[n], gain_x[n])
```

Option 2, finite-cloud sensitivity, preferred because it matches the covariance object:

```python
R_eye = responses across eye positions, shape (n_eye, n_units)
gain_mag[n] = std_eye(R_eye[:, n])
gain_x/gain_y = regression coefficients predicting response from eye x/y
```

Then relate to top covariance eigenvectors:

```python
u1 = top eigenvector of C_FEM
u2 = second eigenvector

corr(abs(u1), gain_mag)
corr(abs(u2), gain_mag)
regress u1 weights from gain_x/gain_y
regress unit FEM variance diag(C_FEM) from gain_mag
```

Also compute whether units with high translation gain dominate the FEM covariance trace:

```python
sort units by gain_mag
cumulative fraction of diag(C_FEM)
```

## Outputs

Figure:

* Unit gain magnitude vs FEM covariance diagonal.
* Top eigenvector loading vs gain components.
* Cumulative covariance contribution by gain-ranked units.
* Optional polar plot of gain directions.

## Acceptance criteria

A6 is **supported** if:

```text
unit translation sensitivity predicts unit FEM variance
top FEM eigenvector loadings are organized by translation gain
high-gain units dominate covariance contribution
```

A6 is **mixed** if only diagonal variance is explained, not shared eigenstructure.

A6 is **failed** if unit translation gains do not relate to `C_FEM`.

---

# Optional packaging only: A7 mimicry geometry

Do not recompute unless existing outputs are easy to load.

A7 should be packaged as geometry, not recoverability magnitude. The prescription says mimicry maps where reafferent covariance could be confusable with stimulus identity, but it should not be interpreted as information loss without a noise model. 

If existing mimicry outputs are available, add a supplemental CSV and figure:

```text
mimicry_alignment_by_image_pair
mimicry_alignment_by_phase
occupancy_weighted_mimicry
```

Use safe wording:

```text
translation-identity geometric overlap
confusability geometry
recoverability precondition
```

Avoid:

```text
information loss
decoder error
coding benefit/cost
```

---

# Suggested run sequence

Implement in this order:

1. **Core response cache and `C_FEM` computation**
2. **A2 low-rank and DOF manipulation**
3. **A3 image specificity**
4. **A1 signal alignment**
5. **A4 tangent alignment**
6. **A5 occupancy controls**
7. **A6 unit-to-population bridge**
8. **README and final summary**

The prescription prioritizes A1–A3 as the essential core, A4 and A6 as strong mechanistic support, A5 as a consistency/control analysis, and A7 as supplemental packaging. 

I would slightly reorder implementation to run A2/A3 before A1 because A2/A3 are simpler diagnostics of whether `C_FEM` is behaving sensibly.

---

# Shared utility pseudocode

```python
def run_model_on_image_eye_cloud(model, image, eye_positions, times=None, batch_size=64):
    """
    Return R with shape (n_eye, n_time, n_units).

    Implementation details depend on existing VisionCore stimulus/eye-shift pipeline.
    Keep model deterministic and eval-mode.
    Cache outputs by image_id, eye_condition, unit_subset, model_checkpoint.
    """
    pass


def compute_cfem_dataset(model, images, eye_ensembles, unit_ids):
    records = []
    cfem = {}

    for image_id, image in enumerate(images):
        for ensemble_name, eye_positions in eye_ensembles.items():
            R = run_model_on_image_eye_cloud(model, image, eye_positions)
            C = compute_cfem_for_image(R)
            evals, evecs = eigensystem(C)
            pr = participation_ratio(evals)

            cfem[(image_id, ensemble_name)] = dict(
                C=C,
                evals=evals,
                evecs=evecs,
            )

            records.append(dict(
                image_id=image_id,
                ensemble=ensemble_name,
                trace=np.sum(evals),
                pr=pr,
                frac_top1=evals[:1].sum() / evals.sum(),
                frac_top2=evals[:2].sum() / evals.sum(),
                frac_top5=evals[:5].sum() / evals.sum(),
            ))

    return cfem, pd.DataFrame(records)
```

---

# Sanity checks and failure modes

## Coordinate-frame audit

Before any model-data quantitative bridge:

```text
Check x sign.
Check y sign.
Check whether “eye right” means image shifts left on retina.
Check pixel/degree conversion.
Check whether model shift convention matches data eye-position convention.
```

If unresolved, keep all claims twin-internal.

## Covariance sanity checks

For each image:

```text
C_FEM trace > 0
PR finite
top eigenvalues not dominated by numerical artifacts
split-half C_FEM overlap above null
constant-eye control gives near-zero C_FEM
unit-shuffle null reduces structured alignment
```

## Model-history confound

Because the model includes temporal processing, A5 may show history effects. If so, report:

```text
The covariance is mostly occupancy-governed but not perfectly trajectory-invariant under the model’s temporal frontend/ConvGRU.
```

Do not reinterpret this as temporal coding or active-sensing function.

## Sample-size sensitivity

For all major metrics, rerun at multiple eye-sample counts:

```text
n_eye = 32, 64, 128, 256
```

Report stability.

---

# Final README interpretation template

Use this exact style:

```text
Summary

We computed deterministic reafferent covariance in the digital twin:
C_FEM(I) = E_t[Cov_e(r(I,e,t)|t)].

Because the model is deterministic, this covariance isolates the population response structure induced by retinal pose variation. These analyses therefore test attribution and geometry, not performance.

Main findings:
1. C_FEM was low-dimensional, with rank controlled by the dimensionality and radius of retinal translation.
2. C_FEM was image-specific, inconsistent with a single global state/gain mode.
3. C_FEM showed [strong/mixed/weak] alignment with stimulus-driven response dimensions.
4. The dominant C_FEM subspace showed [strong/mixed/weak] alignment with the local retinal-translation tangent.
5. Occupancy-matched eye-position controls [did/did not] reproduce the real-trace covariance geometry.
6. Single-unit translation sensitivity [did/did not] predict population covariance structure.

Interpretation:
These results support the structural claim that the recorded FEM-related covariance has the expected geometry of reafference: the image translating across the retina under the animal's own eye movements. They do not establish whether FEMs improve visual coding, optimize sampling, or increase information.
```

---

# Stop rule

After A1–A6 and README are complete, stop.

Do not branch into:

```text
decoder analyses
accuracy
bits/spike
Fisher information
ideal observer
E-optotype benefit
active-sensing efficiency
```

unless explicitly requested in a separate task with a noise model.
