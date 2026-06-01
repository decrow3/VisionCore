# FixRSVP neural trajectory analysis plan

**Version:** revised handoff plan  
**Goal:** identify whether recorded V1 activity during fixRSVP traces eye-linked, image-conditioned neural trajectories during inter-microsaccadic drift, and determine whether microsaccades are large translations, transient events, or boundaries between local trajectory regimes.

---

## 0. Executive summary for the coding agent

This analysis is not primarily a decoder project. The main question is geometric:

> During fixation, does V1 population activity evolve along structured trajectories that are coupled to the animal's measured eye trajectory through the image, and does this coupling depend on image context and microsaccade segmentation?

The central discriminator is:

> **Within-image / within-presentation trajectory geometry should be stronger than cross-image or image-shuffled geometry.**

Eye-position decoding alone is not a success criterion. A global eye-position or gain signal could produce decodable eye position without demonstrating image-specific retinal-translation geometry.

The first-pass manuscript-relevant analysis is **Level 2**:

1. Characterize perisaccadic transients.
2. Segment fixRSVP into inter-microsaccadic drift epochs.
3. Use scalar neural-distance vs eye-distance tests only as diagnostic QC for representation viability.
4. Prioritize directional increment geometry, i.e. whether predicted neural increment direction aligns with observed increment direction under matched image context.
5. Connect trajectory geometry back to FEM covariance structure.

Level 3 functional decoding can be added later, but do not let it delay the core Level 2 trajectory analysis.

---

## 1. Conceptual hypotheses

### 1.1 Core hypothesis

During stable intersaccadic drift, recorded V1 population activity should follow locally coherent trajectories. These trajectories should covary with measured retinal drift and should be more repeatable within matched image/time contexts than under image-shuffled or eye-shuffled controls.

### 1.2 Microsaccade hypotheses

Microsaccades may be:

1. **Large translations along the same geometry:** the neural jump is displacement-dependent and aligned with the same local drift geometry.
2. **Transient events:** microsaccades evoke a stereotyped population transient that temporarily dominates the trajectory.
3. **Coordinate-boundary events:** microsaccades move the population between local drift regimes, so pre- and post-microsaccadic drift segments have different local eye-to-neural mappings.

Do not claim coordinate reset until the transient mode has been characterized and controlled.

### 1.3 What is trivial

The following observations are expected and are not sufficient for the main claim:

- Eye position can be decoded from V1 rates.
- Microsaccades evoke population responses.
- Neural covariance is low-dimensional.
- Neural distance correlates with eye distance without image/time controls.

### 1.4 What is strong

The following would support the trajectory framework:

- Drift-only neural trajectories are smoother and more eye-linked than shuffled controls.
- Neural distance scales with eye distance within matched image/time contexts.
- Local eye-to-neural maps repeat across trials within the same image/window.
- Within-image trajectory geometry is stronger than cross-image or image-shuffled geometry.
- Microsaccades introduce a separable transient mode, or mark transitions between local drift geometries.

### 1.5 What would be game-changing

The strongest result would be:

> Recorded V1 during fixRSVP traces piecewise-coherent, image-conditioned trajectories: intersaccadic drift exposes local transformation coordinates, while microsaccades introduce transient modes or transitions between local coordinate charts.

A later Level 3 result would connect trajectory geometry to stimulus identity or fine-position recoverability. That is not required for the first-pass manuscript-strengthening analysis.

---

## 2. Essential stages and scope control

### 2.1 Essential stages for first-pass manuscript result

Run these first, ideally on one pilot session before scaling:

- **Stage 0:** QC and data alignment.
- **Stage 1:** Perisaccadic transient characterization.
- **Stage 2:** Drift/microsaccade segmentation.
- **Stage 3:** Distance-distance trajectory geometry (diagnostic/QC only, not primary endpoint).
- **Stage 3R:** Representation reliability sweep across binning/smoothing/residual/unit-set choices.
- **Stage 4:** Population-level increment geometry.
- **Stage 5:** Image-conditioned generalization tests.
- **Stage 7:** Covariance decomposition and model-basis comparison, if model bases are available.

Stage gating rule:
Run Stage 4 only on representations whose reliability clears a predefined threshold (recommended >= 0.5). Do not treat Stage 3 scalar-distance failures as a falsification of directional geometry.

### 2.2 Conditional stage

- **Stage 6:** Microsaccade boundary/reset tests. Run only after Stage 1 shows the transient duration and Stage 3-5 show interpretable drift geometry.

### 2.3 Optional later stages

- **Stage 8:** Functional identity/fine-position readout.
- **Stage 9:** Model-neural bridge overlays using matched digital twin trajectories.

### 2.4 Reduced-scope fallback

If Stage 0 shows that drift segments are too short, residuals are too noisy, or microsaccade counts are too low:

1. Keep Stage 3 distance-distance geometry as QC only.
2. Run Stage 3R reliability sweep and select the most reliable representation.
3. Run Stage 4 directional increment tests only if representation reliability is adequate.
4. Keep Stage 7 covariance decomposition.
5. Defer Stage 6 microsaccade boundary claims unless Stage 4 directional tests are interpretable.
6. Report covariance-level bridge results if directional increment tests remain underpowered.

---

## 3. Required inputs

For each session:

1. Spike times and cluster IDs.
2. Unit metadata:
   - V1/V2 assignment if available,
   - depth,
   - quality metrics,
   - inclusion in previous FEM/covariance analyses.
3. fixRSVP trial table:
   - trial ID,
   - image ID or stimulus ID,
   - frame/presentation start and stop times,
   - repeat identity,
   - valid analysis windows.
4. Eye traces aligned to neural time:
   - x position,
   - y position,
   - timestamps,
   - validity mask,
   - optional pupil/vergence/state variables.
5. Existing PSTH or image/time mean estimates, if available.
6. Optional model predictions or bases:
   - `B_model`,
   - `FEM_PCs`,
   - model FEM covariance,
   - local Jacobian-derived basis.

---

## 4. Time alignment and binning

### 4.1 Primary binning

Use 10 ms neural bins for the first-pass trajectory analysis.

Sensitivity checks:

- 5 ms bins if spike counts and smoothing support it.
- 20 ms bins if 10 ms residuals are too noisy.

### 4.2 Eye-neural alignment

Commit to a primary alignment method:

> **Primary:** linearly interpolate eye x/y traces to the neural bin centers.

For each neural bin center `t_k`, compute:

- `eye_x[k]`,
- `eye_y[k]`,
- validity mask.

Compute eye increments as:

```text
dp[k] = p[k+1] - p[k]
```

where both bins must have valid eye samples.

Sensitivity checks:

1. Use median eye position within the neural bin.
2. Use eye position at stimulus frame centers if the stimulus is locked to frame times.
3. Compare 10 ms neural bins to 120 Hz eye/frame sampling where possible.

Document which time convention is used for every output.

### 4.3 Neural variables

For each bin, store:

- `R[k, unit]`: spike count or rate vector,
- `R_smooth[k, unit]`: optional smoothed rate for visualization only,
- `R_resid[k, unit]`: stimulus-time residual,
- `dR[k, unit] = R_resid[k+1] - R_resid[k]`,
- `latent[k, dim]`: optional PCA/FA latent state,
- `dlatent[k, dim] = latent[k+1] - latent[k]`.

Increment analyses should be performed on population-level residuals or latent states, not per-neuron increments in isolation.

---

## 5. Image/window definition

The phrase “image-conditioned” must be explicit.

### 5.1 Primary image context

Primary unit of stimulus context:

> **single image presentation window**, i.e. one fixRSVP image epoch with fixed image ID and time-within-presentation.

This most directly tests whether drift trajectory geometry exists within a stable stimulus context.

### 5.2 Pooled image context

Secondary/sensitivity unit:

> all repeated presentations of the same image, aligned by time-within-presentation.

This increases sample size but mixes presentation-to-presentation variability.

### 5.3 Residual construction

Primary residual:

```text
R_resid(trial, image_id, time_bin) =
    R(trial, image_id, time_bin) - mean_R_excluding_this_trial(image_id, time_bin)
```

Use leave-one-trial-out or split-half means to avoid circularity.

Before running downstream analyses, report:

- number of repeats per `(image_id, time_bin)` cell,
- fraction of cells passing a minimum repeat threshold,
- reliability of image/time mean estimates,
- number of valid bins per image presentation.

If there are too few repeats at 10 ms, use 20 ms bins or coarser time-within-image bins.

---

## 6. Unit selection

Define a primary unit set before the trajectory tests.

Suggested primary criteria:

- V1 units only,
- included in previous FEM/covariance analyses if possible,
- minimum mean firing rate,
- stable across the session,
- valid spike sorting quality metrics,
- valid during fixRSVP.

Sensitivity checks:

- all V1 units,
- visually responsive units,
- high firing-rate subset,
- V2 if available.

Do not select units based on trajectory metrics.

---

## 7. Microsaccade detection and segmentation

### 7.1 Detection

Use calibrated eye traces. Store both onset and peak-velocity time if possible.

For each candidate microsaccade, save:

- trial ID,
- image ID,
- onset time,
- peak velocity time,
- offset time if available,
- amplitude,
- direction,
- duration,
- pre-event eye position,
- post-event eye position,
- validity flags.

Detection should use robust session-specific velocity thresholds or an established microsaccade detector, plus amplitude and refractory-period criteria.

### 7.2 Perisaccadic exclusion windows

Define a peri-event window for transient characterization:

```text
[-50, +150] ms relative to onset or peak velocity
```

Sensitivity checks:

- onset-aligned versus peak-velocity-aligned,
- shorter and longer post-event windows.

### 7.3 Drift segments

Define intersaccadic drift segments as intervals between exclusion windows.

For each segment require:

- valid eye trace throughout,
- stable image assignment for analyses requiring image matching,
- minimum duration.

Recommended first pass:

- minimum duration: 100 ms,
- sensitivity: 80 ms and 150 ms.

Store per segment:

- segment ID,
- trial ID,
- image/window ID,
- start/end time,
- duration,
- eye path length,
- net eye displacement,
- mean drift speed,
- neural path length,
- start/end neural state,
- distance to nearest microsaccade.

### 7.4 Pre-check displacement overlap

Before comparing drift events to microsaccades, plot:

- drift-step displacement distribution,
- microsaccade displacement distribution,
- overlap region.

If overlap is minimal, do not overclaim matched-displacement controls. Use closest-magnitude comparisons with caveats.

---

## Stage 0: QC

### Outputs

For each session:

1. Eye position histogram.
2. Eye velocity distribution.
3. Microsaccade amplitude/velocity/duration distributions.
4. Number of microsaccades per session/trial/image.
5. Drift segment duration distribution.
6. Number of valid drift segments per image/window.
7. Spike count/rate summary by unit.
8. PSTH reliability by image/time cell.
9. Fraction of valid bins after eye and neural masking.

### Pilot decision criteria

Continue to Stage 3 if:

- enough valid drift segments exist,
- enough repeated image/time samples exist for residual estimation,
- microsaccade detection produces plausible event distributions,
- population residuals are not dominated by a small number of units.

If not, switch to the reduced-scope fallback.

---

## Stage 1: Perisaccadic transient characterization

### Purpose

Characterize microsaccade-related transients before interpreting microsaccades as coordinate resets.

### Analysis

For each microsaccade:

1. Extract population activity in a peri-event window.
2. Compute raw and residual population PSTHs.
3. Estimate dominant transient axes using PCA/SVD on peri-event residual responses.
4. Estimate transient duration:
   - time to peak,
   - time to return to baseline,
   - time when projection onto transient axis returns below threshold.
5. Compare transient axes across:
   - images,
   - microsaccade amplitudes,
   - microsaccade directions,
   - sessions.

### Outputs

- Perisaccadic population PSTH.
- Perisaccadic residual PSTH.
- Top transient axes.
- Variance explained by transient axes.
- Transient time course.
- Recommended exclusion window for drift analyses.

### Interpretation

If a strong transient axis exists, Stage 6 must control for it before testing coordinate reset.

---

## Stage 2: Drift trajectory visualization

### Purpose

Visualize whether drift epochs look like coherent neural trajectories.

### Analysis

Within each session:

1. Build residual population matrix.
2. Fit PCA or FA on training data.
3. Project individual drift segments into low-dimensional state space.
4. Plot trajectories colored by:
   - time,
   - eye x/y,
   - image ID,
   - drift speed,
   - distance to microsaccade.

### Outputs

- Example single-image trajectories.
- All-segment trajectory clouds.
- Drift versus peri-microsaccade trajectories.
- Scree plots and explained variance.

### Interpretation

This stage is exploratory and qualitative. It should guide but not replace quantitative tests.

---

## Stage 3: Distance-distance trajectory geometry

### Purpose

Test whether neural trajectory distances scale with eye trajectory distances, without using a decoder.

### Primary neural representation

Use Euclidean distance in residual PCA space.

Recommended primary PCA dimensions:

- 10 dims,
- sensitivity: 5, 20 dims.

Rationale: PCA-Euclidean is more stable than full high-dimensional Euclidean and less dependent on noisy covariance estimates than Mahalanobis distance.

### Supplementary metrics

- raw residual Euclidean distance,
- cosine distance in residual PCA space,
- Mahalanobis distance if covariance estimates are stable,
- FA/GPFA latent distance if available.

### Pair classes

Compute pairwise neural and eye distances for:

1. within the same drift segment,
2. across drift segments within the same image presentation,
3. across microsaccade boundary, excluding transient windows,
4. same image across different repeats,
5. different images,
6. image-shuffled control,
7. eye-trace permutation control,
8. within-segment circular-shift control.

### Primary metric

For each class:

```text
corr(D_neural, D_eye)
```

and/or slope from robust regression:

```text
D_neural ~ beta * D_eye + nuisance covariates
```

Nuisance covariates may include:

- time separation,
- mean firing rate,
- image/time bin,
- trial identity,
- distance to microsaccade.

### Expected pattern

Strong result:

```text
within drift + matched image context
    > across microsaccade
    > image-shuffled / eye-shuffled controls
```

Critical discriminator:

```text
within-image geometry > cross-image geometry
```

This is the key test distinguishing image-specific retinal-translation geometry from generic eye-position modulation.

### Outputs

- Distance-distance scatter plots.
- Correlation/slope summary by pair class.
- Bootstrap confidence intervals.
- Session-level and pooled summaries.

---

## Stage 4: Population-level increment geometry

### Purpose

Ask whether neural increments during drift align with eye increments.

### Important caution

Do not analyze per-neuron 10 ms increments in isolation. They will be dominated by spike-count noise. Use population residuals or low-dimensional latent states.

### Primary representation

Use residual PCA state `z[k]`.

Then:

```text
dz[k] = z[k+1] - z[k]
dp[k] = p[k+1] - p[k]
```

### Analysis options

#### 4.1 Increment magnitude coupling

Test whether:

```text
||dz|| ~ ||dp||
```

within drift segments, with image/time controls.

#### 4.2 Local linear map

Fit:

```text
dz = A dp + error
```

Use ridge regression or reduced-rank regression.

Primary fit:

- pooled within image/window or image group, not per tiny segment.

Secondary fit:

- pre/post microsaccade drift segments for reset tests, using strong regularization.

#### 4.3 Prediction quality

Use held-out data to compute:

- predicted fraction of `dz`,
- cosine between predicted and actual `dz`,
- R²,
- shuffle-corrected effect.

### Controls

- eye-trace permutation,
- time-within-image circular shift,
- image shuffle,
- within-segment random sign/rotation of `dp` if appropriate.

### Expected pattern

A strong result is:

- above-shuffle increment alignment within drift,
- stronger within matched image contexts than across images,
- reduced or altered alignment near microsaccades.

---

## Stage 5: Image-conditioned generalization

### Purpose

This is the load-bearing test for the framework.

Distinguish:

1. generic eye-position modulation,
2. image-specific translation geometry.

### Primary analyses

#### 5.1 Within-image repeatability

For each image/window with enough repeats:

1. Fit eye-to-neural geometry on one split.
2. Test on held-out repeats of the same image/window.
3. Compare to eye permutation within image.

#### 5.2 Cross-image generalization

Train on one image or image group and test on held-out images.

Compare:

```text
within-image performance
vs
cross-image performance
vs
image-shuffled control
```

Interpretation:

- within-image > shuffled: eye-linked geometry exists.
- within-image > cross-image: geometry is image-conditioned.
- cross-image above shuffled: possible shared/global eye-position component.

#### 5.3 Mixed model

Fit a model with both shared and image-specific terms:

```text
neural_state ~ shared_eye_map(p) + image_specific_eye_map(image, p)
```

Estimate how much variance is explained by the shared component versus the image-specific component.

### Outputs

- within-image and cross-image prediction metrics,
- shared versus image-specific variance fractions,
- session-level summaries,
- image-level reliability ceilings.

---

## Stage 6: Microsaccade boundary tests

Run only after Stage 1 and Stages 3-5.

### 6.1 Large-translation versus transient versus reset

For each microsaccade, compare:

- pre-drift local geometry,
- peri-event transient,
- post-drift local geometry.

### 6.2 Pre/post local map comparison

Fit local drift maps before and after microsaccades:

```text
dz = A_pre dp
dz = A_post dp
```

Compare:

- subspace angle between `A_pre` and `A_post`,
- held-out prediction within pre and post,
- cross-prediction pre-to-post and post-to-pre.

### 6.3 Transient-axis subtraction

If Stage 1 identifies a strong transient axis:

1. Estimate the transient subspace from peri-event data.
2. Project it out or regress it out.
3. Re-run pre/post drift geometry tests.

This distinguishes:

- apparent reset caused by transient contamination,
- true change in local drift geometry.

### 6.4 Microsaccade displacement comparison

Ask whether microsaccades behave like large retinal translations.

Compare neural jump direction/size to:

- microsaccade amplitude/direction,
- drift events of nearest available displacement magnitude,
- model-predicted finite-displacement response if available.

If displacement distributions do not overlap, report this explicitly.

### Outputs

- transient-subtracted and non-subtracted pre/post comparisons,
- pre/post map angles,
- cross-prediction scores,
- microsaccade jump versus displacement plots.

---

## Stage 7: Covariance decomposition link

### Purpose

Connect trajectory analysis to existing FEM covariance work.

### Analyses

Estimate covariance for:

1. drift-only residual activity,
2. peri-microsaccade residual activity,
3. post-microsaccade recovery windows,
4. all fixation residual activity.

Compare:

- dimensionality,
- top covariance axes,
- alignment with previous `C_FEM`,
- alignment with model `FEM_PCs`,
- alignment with `B_emp`,
- alignment with perisaccadic transient axes.

### Metrics

- variance explained by top `k` dimensions,
- subspace overlap,
- projection of neural covariance into model bases,
- split-half reliability-normalized covariance alignment.

### Expected pattern

A strong result is:

- drift-only covariance is more translation-like,
- microsaccade covariance has a distinct transient/event axis,
- all-fixation covariance mixes drift and microsaccade components.

---

## Stage 8: Optional functional readout

Do not run this before the Level 2 trajectory result unless specifically requested.

### Possible analyses

1. Compare stimulus identity decoding from:
   - single time points,
   - drift-segment trajectories,
   - pooled pre/post microsaccade windows.
2. Test whether identity decoding improves in segments with stronger eye-linked trajectory geometry.
3. Ask whether nonlinear/energy-like features outperform signed linear averages.

### Caution

Natural images may not reproduce the clean E-optotype mechanism. Interpret functional results as generality tests, not as a requirement for the trajectory claim.

---

## Stage 9: Optional model-neural bridge

If model predictions are available for matched images and eye traces:

1. Generate model trajectories for the same image/eye windows.
2. Compute model residual trajectories and FEM covariance.
3. Compare neural trajectory axes to:
   - model `FEM_PCs`,
   - `B_model`,
   - local Jacobian basis,
   - finite-displacement model covariance.
4. Test whether drift-only neural geometry is better predicted by finite-displacement model geometry than by single local Jacobians.

This should be a bridge analysis, not the starting point.

---

## Nulls and controls

Use several nulls because each rules out a different trivial explanation.

### Primary nulls

1. **Eye permutation across trials:** preserves stimulus/time structure, breaks eye-neural coupling.
2. **Image shuffle:** preserves eye and neural marginal structure, breaks image-specific matching.
3. **Within-segment circular shift:** preserves smoothness and autocorrelation, breaks precise alignment.
4. **Time-within-image shuffle:** tests whether effects are just PSTH/time dynamics.
5. **Microsaccade label shuffle:** tests boundary specificity.

### Secondary nulls

1. Random low-dimensional subspaces matched for dimensionality.
2. Trial-label shuffle within image.
3. Matched mean-rate controls.
4. Unit subsampling controls.

---

## Outputs for coding agent

Create a session-level output directory with:

```text
outputs/
  session_name/
    qc/
    transient/
    trajectories/
    distance_distance/
    increments/
    image_conditioning/
    microsaccade_boundaries/
    covariance/
    summaries/
```

### Required machine-readable outputs

1. `session_qc.csv`
2. `microsaccades.csv`
3. `drift_segments.csv`
4. `distance_distance_metrics.csv`
5. `increment_metrics.csv`
6. `image_conditioning_metrics.csv`
7. `transient_metrics.csv`
8. `covariance_alignment_metrics.csv`
9. `analysis_config.yaml`

### Required figures

1. Eye position and velocity QC.
2. Microsaccade distributions.
3. Perisaccadic transient PSTH and transient-axis projection.
4. Drift trajectory examples in PCA/FA space.
5. Distance-distance plots by pair class.
6. Within-image versus cross-image geometry summary.
7. Increment alignment summary.
8. Drift versus microsaccade covariance comparison.
9. If Stage 6 is run: pre/post microsaccade geometry comparison.

---

## Pilot workflow

Run one high-quality V1 session first.

### Pilot steps

1. Stage 0 QC.
2. Stage 1 transient characterization.
3. Stage 3 distance-distance analysis.
4. Stage 5 within-image versus cross-image comparison.
5. Decide whether to proceed to Stage 4/6/7.

### Pilot success criteria

Proceed if:

- within-drift image-matched distance-distance effects exceed eye-shuffled controls,
- within-image effects exceed cross-image or image-shuffled controls,
- residual PCA trajectories show interpretable structure,
- microsaccade transient duration can be estimated well enough to define exclusion windows.

If these fail, switch to the reduced-scope fallback.

---

## Interpretation guide

### Strong first-pass conclusion

Use this if Stages 3 and 5 succeed:

> During fixRSVP, recorded V1 activity contains eye-linked, image-conditioned trajectory geometry during intersaccadic drift. This supports the idea that FEM-induced variability is not generic noise, but the population footprint of active sampling through stimulus-conditioned response geometry.

### Strong microsaccade conclusion

Use this only if Stage 6 succeeds after transient control:

> Microsaccades are not merely larger drift steps. They introduce a distinct transient mode and/or mark transitions between local drift geometries.

### Avoid these claims unless directly supported

- V1 has a content-general translation coordinate.
- Decoding eye position proves transformation geometry.
- Microsaccades reset coordinates before transient effects are removed.
- Drift geometry proves an optimized active-sensing policy.
- Natural-image results prove the same mechanism as the E-optotype sign-flip result.

---

## Agent implementation notes

1. Write modular code. Each stage should be independently runnable.
2. Save intermediate arrays and masks so analyses can be audited.
3. Do not silently discard images/windows with low repeat counts; report them.
4. Keep all time bases explicit.
5. Use held-out data for all regression/prediction metrics.
6. Primary headline metric should not be a decoder.
7. Treat image-conditioned generalization as the key controlled comparison.
8. Use population-level or latent-state increments, not noisy per-neuron increments alone.
9. Keep raw and residual analyses separate.
10. Make one pilot-session report before scaling.

---

## Suggested minimal first deliverable

A single-session pilot report containing:

1. QC summary.
2. Microsaccade detection summary.
3. Perisaccadic transient characterization.
4. Drift trajectory examples.
5. Distance-distance analysis:
   - within drift,
   - across drift,
   - across microsaccade,
   - image-shuffled,
   - eye-shuffled.
6. Within-image versus cross-image geometry comparison.
7. Clear recommendation:
   - proceed to full analysis,
   - modify preprocessing/binning,
   - or reduce scope.

This pilot should determine whether the full Level 2 trajectory analysis is worth running across all sessions.
