# Vernier Active-Sensing Analysis for the V1 Digital Twin

## One-sentence goal

Test how FEM-like retinal motion changes the V1 digital twin's recoverable sensitivity to a tiny Vernier offset, using a simple, interpretable hyperacuity task with fair static controls, motion-axis controls, explicit pose-aware and pose-blind readouts, and strict rendering/provenance audits.

## Core scientific question

A Vernier stimulus consists of two nearly aligned line segments separated by a small gap. The lower segment is shifted slightly left or right relative to the upper segment. The task is to estimate the sign or magnitude of this offset.

The analysis asks:

> Does small retinal motion, in the range generated during fixation, improve or impair the V1 population's recoverable sensitivity to a fine spatial misalignment, and how does that depend on retinal-pose knowledge?

This is intended to be a clean active-sensing test. It should not be framed as a strong temporal-coding analysis. The main hypothesis is that FEM-like motion samples multiple nearby retinal phases, which can expose fine-position information in the population response when pose is known but can also create nuisance covariance when pose is hidden. Any temporal accumulation should be performed by the analysis/readout, not assumed to occur inside the ConvGRU.

## Why Vernier rather than the previous E-optotype tests?

The E-optotype task is useful but compound. Orientation decoding depends on stroke width, gap position, global object identity, rendering scale, retinal phase, and possible object-identity confusions. Vernier acuity isolates one continuous spatial variable:

\[
\delta = \text{horizontal offset between two line segments}.
\]

This gives a simple endpoint:

\[
\text{Vernier threshold proxy} \propto \frac{1}{\sqrt{F_\delta}}.
\]

Lower threshold proxy means better sensitivity under a stated observer/noise model. Absolute thresholds are not identified because the assumed noise scale is free; the paper-facing quantities should be condition ratios, reliability changes, and sign consistency across sessions.

## Main claim this analysis can support

The desired positive result is not:

> Real FEM trajectories are uniquely optimal.

Nor is the useful first-pass claim simply:

> Real FEM improves Vernier acuity.

The useful first-pass claim is a coordinate-frame boundary:

> Vernier information is available in the phase-dependent V1 response, but it is only useful if the observer has sufficiently good retinal-pose information.

The bounded active-sensing claim is therefore:

> Motion helps a pose-aware observer and strongly burdens a pose-blind observer.

The load-bearing comparison is motion through a phase cloud versus appropriate static phase-cloud baselines, not only motion versus one repeated phase. If the repeated phase is a random cloud draw, the main benefit may appear as reduced across-trial variability or improved reliability rather than a higher mean Fisher value. A real-FEM advantage should not be claimed unless it beats a phase-cloud baseline matched to the same retinal-position distribution and remains recoverable under a stated pose-knowledge assumption.

A stronger secondary result would be:

> The benefit is axis-specific, motion-scale dependent, robust to modest pose uncertainty, and bounded by optimized and adversarial trajectory controls.

## Important claim boundaries

Do not claim:

- The model proves behavioral Vernier acuity.
- The absolute threshold proxy is a calibrated human acuity estimate.
- The exact biological trace is uniquely optimal.
- V1 implements a long-memory temporal code.
- The ConvGRU itself accumulates evidence over a full fixation.
- Any motion is useful.
- A pose-aware Fisher gain alone proves pose-blind downstream recoverability.

Do claim, if supported:

- Motion provides multiple short-latency retinal samples.
- The downstream observer can pool V1 responses across samples under explicitly stated pose assumptions.
- The benefit is about phase diversity, reliability, and fine spatial sensitivity, unless order-shuffle controls prove otherwise.
- Real or FEM-like traces can sit within a useful operating range without being uniquely optimal.
- Real, random amplitude-matched, order-shuffled, and phase-cloud-like outcomes can be positive if they corroborate the broader trajectory-agnostic claim.

---

# Current first-pass interpretation

The key first-pass result is not a naive real-FEM Vernier-acuity improvement. It is the coordinate-frame boundary:

> Vernier information is available in the phase-dependent V1 response, but it is only useful if the observer has sufficiently good retinal-pose information.

Pose-aware readout says phase diversity contains Vernier information. Pose-blind readout says that same phase diversity becomes a large nuisance covariance when pose is hidden. The bounded claim is therefore:

> Motion helps a pose-aware observer and strongly burdens a pose-blind observer.

Using `static_center` as a diagnostic single-phase reference:

| condition | FD step | PA/static | PB/static |
|---|---:|---:|---:|
| `matched_phase_cloud` | 0.25 | 1.56 | 0.11 |
| `real_fem` | 0.25 | 1.53 | 0.09 |
| `scaled_real_0.5` | 0.25 | 1.88 | 0.13 |
| `matched_phase_cloud` | 0.5 | 1.40 | 0.09 |
| `real_fem` | 0.5 | 1.44 | 0.08 |
| `scaled_real_0.5` | 0.5 | 1.81 | 0.13 |

Interpretation hierarchy:

1. `static_center`: diagnostic single phase, not the primary biological baseline.
2. `matched_phase_cloud`: primary static biological-position baseline.
3. `real_fem`: biological untuned fixation motion.
4. `scaled_real_0.5`: candidate task-tuned motion scale.

Current conclusion:

- Real FEM does not improve mean Fisher over matched phase cloud.
- `scaled_real_0.5` does improve mean Fisher over matched phase cloud in this first pass.
- All phase-variable conditions are heavily penalized by pose-blind readout.
- Before treating `scaled_real_0.5` as a true motion advantage, verify that its matched phase-cloud baseline is matched to the half-scaled positions, not to the full real-FEM cloud.

The first drift/microsaccade component-scale pass with Jake's microsaccade labels is a useful follow-up diagnostic, not yet the primary claim. It suggests reduced drift scale and microsaccade-containing components can be informative, but it should be folded into the next full scale-specific matched-control pass before any biological interpretation.

## Next-pass priorities

1. Run a full motion-scale sweep: `D = [0, 0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3]`, with scale-specific matched phase-cloud and order-shuffled controls.
2. Add a pose-uncertainty sweep between pose-aware and pose-blind:
   \[
   \sigma_{\mathrm{pose}} = [0, 0.25, 0.5, 1, 2, 4] \text{ arcmin}.
   \]
   The main question is: how accurate must retinal-pose knowledge be for FEM-like motion, especially reduced-amplitude motion, to beat static or matched phase-cloud baselines?
3. Add a secondary compact-aware pose-blind analysis to ask whether low-rank FEM/phase covariance makes pose-hidden Vernier information partially recoverable.
4. Add FD steps `[0.125, 0.25, 0.5, 1.0] arcmin` and decide whether the endpoint is local Fisher or finite-offset sign discrimination. Do not mix these interpretations.
5. Run axis and rotated-stimulus controls before making any task-axis claim.
6. Report reliability and lower-tail Fisher relative to matched phase-cloud baselines.
7. Finish rendering controls: pixel-level Fisher, centroid/template baselines, and renderer-resolution sweep.
8. Run or extend the drift/microsaccade component-scale analysis when GPU is available, using Jake-label-compatible event definitions and scale-specific matched controls.
9. Only then run optimized/adversarial trajectories; expect Vernier-optimized motion to be smaller/slower or axis-biased, not identical to natural fixation traces.

Suggested paper-facing interpretation if replicated:

> In a virtual Vernier task, retinal phase strongly shaped the V1 twin's sensitivity to fine spatial offset. When retinal pose was known, FEM-like phase sampling exposed substantially more Vernier information than a single centered static phase. However, the same phase variability became strongly information-limiting for a pose-blind observer. Real fixation traces did not outperform matched phase-cloud controls, consistent with the view that exact steady-fixation trajectories are not optimized for this task. Instead, reduced-amplitude real trajectories produced the highest first-pass Vernier information, suggesting that fine-acuity tasks may favor smaller, task-tuned motion within the broader FEM regime.

---

# Analysis overview

## Analysis levels

Run the analysis in increasing order of ambition:

1. **Rendering and provenance validation**
   - Confirm that subpixel offsets are represented faithfully at the model input.
   - Characterize the model stimulus grid and effective spatial cutoff.
   - Compute pixel-level Vernier information and simple luminance-centroid baselines.
   - Route all finite-difference model responses through the canonical validated twin inference path.

2. **Instantaneous phase sensitivity**
   - Compute Vernier information at single static retinal phases.
   - Establish that sensitivity to \(\delta\) varies with retinal phase.
   - Estimate both the mean and variance of instantaneous information over the fixation cloud.

3. **Phase-cloud and phase-diversity accumulation**
   - Compare FEM-like motion to static phase-cloud controls over the same duration.
   - Compare repeated static sampling only after specifying the repeated phase \(p_0\).
   - Test whether visiting multiple nearby phases increases mean information, reduces information variance, or improves worst-case/reliability summaries.

4. **Motion-axis specificity**
   - For a vertical Vernier stimulus, compare horizontal-dominant, vertical-dominant, and isotropic motion.
   - Prediction: horizontal motion should help more than vertical motion for horizontal offset discrimination.

5. **Motion-scale dependence**
   - Sweep motion amplitude/diffusion scale.
   - Prediction: too little motion helps less, FEM-scale motion lies in a useful regime, too much motion eventually hurts or saturates.

6. **Trajectory controls**
   - Compare real, random matched, order-shuffled, optimized, and adversarial trajectories.
   - Determine whether exact biological order matters, whether random FEM-like motion is sufficient, and whether bad motion exists under the same movement budget.

7. **Noise/readout robustness**
   - Repeat key metrics under several observation models.
   - Run pose-aware and pose-blind readouts in parallel from the first pass.
   - Because the twin is noiseless, all information results must be reported as observer-model-dependent.

---

# Stimulus design

## Canonical Vernier stimulus

Use a grayscale stimulus on a neutral gray background. The canonical stimulus contains:

- Upper vertical bar centered at \(x=0\).
- Lower vertical bar shifted horizontally by \(\delta\).
- A small vertical gap between the bars.
- Fixed line width.
- Fixed bar length.
- Fixed contrast polarity.

Suggested parameters should be expressed in both pixels and arcmin. Use the existing VisionCore stimulus geometry to convert units. Save the final conversion in the manifest.

### Starting parameter grid

Use a modest grid for the first run:

- `offset_arcmin`: symmetric around zero, for example `[-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0]`
- `bar_width_arcmin`: `[1.0, 2.0, 4.0]`
- `gap_arcmin`: `[2.0, 4.0]`
- `bar_length_arcmin`: `[8.0, 12.0, 16.0]`
- `contrast`: `[0.25, 0.5, 1.0]`
- `polarity`: bright-on-gray and dark-on-gray, if easy

For the first smoke test, use a single canonical condition:

- bar width: 2 arcmin
- gap: 4 arcmin
- bar length: 12 arcmin
- contrast: 0.5
- offsets: small symmetric set around zero, including `0` for rendering and response-symmetry audits
- multiple finite-difference step sizes around zero to verify local linearity

## Stimulus rendering requirements

The renderer must support subpixel offsets. Avoid integer-pixel-only rendering.

Recommended:

- Render at high spatial supersampling resolution.
- Draw bars with anti-aliased edges.
- Downsample to the model's stimulus grid using area interpolation or equivalent.
- Keep mean luminance matched across offsets.

Audit:

- Save stimulus images for all offsets, including `delta = 0`.
- Save difference images for `+delta - -delta`.
- Save line profiles through the Vernier feature.
- Confirm that positive and negative offsets are symmetric in pixel space.
- Characterize the model input pixel pitch, stimulus size, and effective spatial cutoff relative to bar width, gap, and offset.
- Compute pixel-level Fisher information about `delta` at the rendered stimulus level under a simple pixel-noise model.
- Compute luminance-centroid and total-luminance baselines to rule out trivial rendering cues.
- Verify that the finite-difference derivative is stable across at least two small `delta` step sizes.

## Optional naturalistic extension

Do not include this in the first run. If canonical Vernier works, add:

- Vernier-like breaks in natural contours.
- Natural-image patches with an inserted local gap and offset.
- High-frequency edge fragments embedded in natural texture.

The first-pass analysis should stay simple and schematic.

---

# Retinal motion conditions

## Core conditions

For each stimulus and offset, render retinal movies under these conditions:

1. `static_center`
   - Stimulus held fixed at the nominal center.

2. `static_repeated_phase`
   - Same retinal phase repeated for the full duration.
   - Specify the repeated phase \(p_0\) before looking at results.
   - Include at least `p0_center` and cloud-derived choices such as random draws or cloud quantiles.
   - This is a useful repeated-read baseline, but not the only load-bearing active-sensing baseline.

3. `static_phase_cloud_single`
   - Choose one static phase sampled from the empirical fixation cloud.
   - Each trial uses one phase, repeated over time.
   - This estimates the distribution of fixed-phase sensitivity and the reliability cost of being stuck at one phase.

4. `static_phase_cloud_matched_positions`
   - Use the same phase set or occupancy distribution as a motion condition, but without within-trial temporal ordering.
   - This is the primary baseline for asking whether real motion beats phase-cloud sampling itself.
   - Under independent pose-aware pooling, this should match FEM-like motion in expectation; deviations indicate temporal covariance, model history, pose-blind effects, or finite-sample/occupancy differences.

5. `real_fem`
   - Use measured FEM traces.

6. `real_drift_only`
   - Use drift segments, excluding microsaccades if available.

7. `real_microsaccade_containing`
   - Use traces containing microsaccades or microsaccade-like events.

8. `random_amp_matched`
   - Synthetic random traces matched to empirical step amplitude or RMS displacement.

9. `random_cloud_matched`
   - Synthetic traces matched to empirical fixation cloud occupancy.

10. `order_shuffled_positions`
   - Same visited retinal positions as a real trace, shuffled in time.
   - Tests whether sequence order matters beyond phase coverage.

11. `axis_horizontal`
   - Motion constrained or biased horizontally.

12. `axis_vertical`
   - Motion constrained or biased vertically.

13. `scaled_real`
   - Real traces scaled by diffusion/amplitude factors, for example:
     - `D = [0, 0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3]`

## Optional gradient-derived conditions

Add after the baseline run works:

14. `optimized`
   - Trajectory optimized to maximize Vernier Fisher information or pair discriminability under motor constraints.

15. `adversarial`
   - Trajectory optimized to minimize Vernier information under the same motor constraints.

These are high-value controls, but they should not block the first pass.

---

# Model outputs

For each condition, generate model-predicted rates:

\[
\mu_{\delta}(t, n)
\]

where:

- \(\delta\) is Vernier offset,
- \(t\) is time bin,
- \(n\) is neuron index.

Save:

- rates for all offsets and conditions,
- expected spike counts per condition,
- trial metadata,
- motion metadata,
- model/session metadata,
- pose/readout metadata,
- finite-difference provenance metadata.

Use the same dataset/session/model handling conventions as the current VisionCore analysis scripts. The `+delta` and `-delta` members of every finite-difference pair must use the same trajectory, phase samples, and model/session path so that motion variability does not leak into the derivative. The FD Jacobian should be generated through the canonical validated twin inference path, not from a re-converted or unaudited cache.

---

# Primary metrics

## 1. Fisher information for Vernier offset

For a continuous offset parameter \(\delta\), compute:

\[
F_\delta =
\left(\frac{\partial \mu}{\partial \delta}\right)^T
\Sigma^{-1}
\left(\frac{\partial \mu}{\partial \delta}\right).
\]

Estimate the derivative using symmetric finite differences:

\[
\frac{\partial \mu}{\partial \delta}
\approx
\frac{\mu_{+\delta} - \mu_{-\delta}}{2\delta}.
\]

For Poisson-style observers, compute Fisher on expected spike counts per bin, not on unscaled rates unless the bin width is explicitly included. Recompute the derivative at multiple small finite-difference steps to verify local linearity.

Compute this for:

- each time bin,
- cumulative windows \(1:t\),
- each motion condition,
- each stimulus parameter setting,
- each noise model.

## 2. Vernier threshold proxy

Convert Fisher information to a threshold proxy for within-analysis comparisons:

\[
\theta_{\mathrm{Vernier}} = \frac{1}{\sqrt{F_\delta + \epsilon}}.
\]

Report threshold ratios:

\[
\text{threshold ratio} =
\frac{\theta_{\mathrm{motion}}}{\theta_{\mathrm{static}}}
=
\sqrt{\frac{F_{\mathrm{static}}}{F_{\mathrm{motion}}}}.
\]

A ratio below 1 means motion improves the threshold proxy relative to the stated baseline. Do not report the absolute threshold proxy as calibrated model acuity in arcmin unless the observation noise scale has been independently calibrated.

## 3. Pairwise discriminability

For sign discrimination, compute:

\[
d'^2 =
(\mu_{+\delta} - \mu_{-\delta})^T
\Sigma^{-1}
(\mu_{+\delta} - \mu_{-\delta}).
\]

This is easier to interpret for binary left/right Vernier judgment.

Use both metrics:

- Fisher information for continuous sensitivity.
- \(d'^2\) for simple sign discrimination.

Under the same covariance and symmetric finite difference, \(d'^2 = (2\delta)^2 F_\delta\). Report both for interpretability, but do not treat agreement between them as independent confirmation.

## 4. Cumulative information over time

For each condition, compute cumulative Fisher or \(d'^2\) as a function of movie time.

Key summaries:

- final Fisher,
- final threshold proxy,
- time to reach 50 percent of final information,
- early slope,
- late slope,
- area under the motion-minus-baseline information curve.

## 5. Spike-count normalized metric

Compute:

- raw Fisher,
- Fisher per expected spike,
- \(d'^2\) per expected spike.

This guards against a trivial motion causes more spikes interpretation, but it does not identify an absolute threshold because the noise scale remains an observer assumption.

## 6. Reliability and variance metrics

For each baseline and motion condition, report:

- mean and median Fisher across phase draws or trajectories,
- across-trial variance or interquartile range of Fisher,
- lower-tail summaries such as the 10th percentile or worst quartile,
- probability that a motion trajectory exceeds its paired static phase-cloud baseline,
- threshold-proxy ratios computed against both repeated-phase and static phase-cloud baselines.

These are co-primary because FEM-like motion may stabilize recoverable information across nearby retinal phases without increasing the cloud-average Fisher.

---

# Noise and readout models

Because the digital twin is deterministic, all information estimates need an assumed observation model.

Run at least these, with pose-aware and pose-blind variants in the first pass. Then add an intermediate pose-uncertainty sweep so the result is not only a two-point comparison between perfect pose knowledge and no pose knowledge:

\[
\sigma_{\mathrm{pose}} = [0, 0.25, 0.5, 1, 2, 4] \text{ arcmin}.
\]

## 1. Pose-aware readout

The observer is given the retinal phase or eye state for each sample. Under independent Poisson/count noise, this is the block-diagonal additive-Fisher calculation:

\[
F_{1:T} = \sum_t F_t.
\]

This readout tests whether the information is present in the sampled responses if pose is known.

## 2. Pose-blind readout

The observer does not get the exact retinal phase or trajectory label. Motion-induced response variability is included in the covariance or marginal response distribution. This readout tests whether the information remains recoverable when eye-position variation is not factored out.

The pose-aware/pose-blind contrast is not optional: it is the main guard against mistaking phase diversity for downstream recoverability.

Current implementation note:

- `pose_blind_diagonal_count_plus_marginal` is the current first-pass pose-blind metric.
- It uses expected count noise plus the diagonal of the across-phase/across-trajectory marginal variance.
- It is not a full covariance inverse and should not be labeled `pose_blind_full_cov_optimal`.
- If a full covariance inverse is added, label it `pose_blind_full_cov_optimal`; this is the optimal linear covariance-aware pose-blind observer under the estimated covariance and shrinkage.

## 3. Compact-aware pose-blind readouts

Secondary goal:

> Test whether low-rank FEM/phase covariance makes pose-hidden Vernier information partially recoverable.

This is a bridge between the pose-aware and ordinary pose-blind endpoints. The question is whether the nuisance covariance induced by hidden retinal phase is compact enough that an observer can discount or remove it while preserving part of the Vernier signal direction.

Estimate leading nuisance modes \(U_k\) from phase/trajectory covariance. Candidate subspace sources:

- Vernier phase covariance.
- Real FEM trajectory covariance.
- Natural-image/translation covariance, if available from the compact translation-geometry analyses.
- Random orthonormal subspace control.

Use at least:

- \(k = [1, 2, 5, 10]\)
- \(\alpha = [0, 0.25, 0.5, 0.75, 1]\)

Hard projection:

\[
P_\perp = I - U_k U_k^\top
\]

Compute Fisher after projecting the response derivative and covariance into the orthogonal complement of the top nuisance modes. Report as `pose_blind_compact_project_k`.

Soft discounting:

Reduce the weight of the top nuisance modes rather than deleting them. Let \(\alpha = 1\) recover ordinary pose-blind covariance and \(\alpha = 0\) strongly discount the top nuisance subspace. Report as `pose_blind_compact_discount_k_alpha`.

Required plots:

- Fisher/static-center ratio versus \(k\).
- Fisher/static-center ratio versus \(\alpha\).
- Pose-aware and ordinary pose-blind reference lines.
- Separate curves or facets for Vernier phase covariance, real FEM trajectory covariance, natural-image/translation covariance if available, and random orthonormal controls.

Interpretation:

- If compact-aware sits between pose-aware and ordinary pose-blind, compact geometry partially rescues pose-hidden Vernier information.
- If compact-aware \(\approx\) ordinary pose-blind, compactness identifies the nuisance but does not recover the active-sampling benefit.
- If results depend strongly on arbitrary \(k\), shrinkage, or subspace source, keep this as an internal control rather than a paper claim.

## 4. Diagonal Poisson

\[
\Sigma = \operatorname{diag}(\mu + \epsilon).
\]

Use the predicted rates/counts for the corresponding condition.

## 5. Overdispersed diagonal

\[
\Sigma = \phi \operatorname{diag}(\mu + \epsilon)
\]

with `phi` values such as `[1, 2, 4]`.

## 6. Fixed diagonal from baseline

Use a fixed diagonal covariance estimated from the average response over offsets or from a reference condition. This prevents differences in \(\Sigma\) from dominating.

## 7. Recorded residual covariance, optional

If available and easy, use a shrinkage version of recorded residual covariance:

\[
\Sigma = \Sigma_{\mathrm{int}} + \lambda I
\]

or diagonal plus low-rank residual covariance.

This is optional for first pass. Do not block on it.

---

# Cumulative pooling implementation

Avoid requiring the ConvGRU to carry long memory.

Treat the model as producing short-latency V1 responses at each time. The downstream observer pools the sequence.

Practical implementation options for the pose-aware calculation:

1. Concatenate time bins into one response vector:
   \[
   \mu = [\mu_1, \mu_2, ..., \mu_T].
   \]

2. Assume block-diagonal noise over time for first pass:
   \[
   \Sigma = \mathrm{blockdiag}(\Sigma_1, ..., \Sigma_T).
   \]

3. Equivalent computation:
   \[
   F_{1:T} = \sum_t F_t
   \]
   if time bins are treated as conditionally independent.

This block-diagonal calculation is explicitly pose-aware. Under this assumption, motion through a cloud of phases and static sampling from the same phase cloud should match in expectation if responses are memoryless and time bins are conditionally independent. Any apparent FEM advantage over one repeated phase should therefore be interpreted as phase-cloud sampling or reliability unless it also beats scale-specific phase-cloud baselines and remains recoverable under a stated pose-uncertainty model.

Also run a response-history control if feasible:

- `movie_mode`: use the normal ConvGRU movie inference path.
- `framewise_or_reset_mode`: reset or otherwise control hidden state for each short-latency retinal sample.

A difference between these modes suggests a model-history contribution and should be reported separately from the main phase-diversity claim.

---

# Required controls

## 1. Static phase-cloud controls

Primary comparison:

\[
F_{\mathrm{motion}} \text{ vs } F_{\mathrm{static\_phase\_cloud}}.
\]

Use both `static_phase_cloud_single` and `static_phase_cloud_matched_positions`. This is the load-bearing baseline for the active-sampling claim because it asks whether a motion trajectory does more than sample from the same useful phase distribution.

Under independent pose-aware pooling, real FEM, random cloud-matched motion, order-shuffled positions, and matched static phase-cloud sampling may be similar in expectation. That result is not a failure if it supports the broader claim that phase coverage, not exact biological order, is the useful ingredient.

## 2. Repeated-static control

Secondary comparison:

\[
F_{\mathrm{motion}} > F_{\mathrm{static\_repeated\_phase}(p_0)}.
\]

This tests whether moving to multiple phases helps relative to rereading a specified phase. It is sensitive to the choice of \(p_0\), so report central, random-cloud, and cloud-quantile repeated phases separately. If \(p_0\) is a random cloud draw, a real effect may appear as reduced variance or improved lower-tail reliability rather than a higher mean.

## 3. Axis specificity

For vertical Vernier, horizontal motion should help more than vertical motion.

Primary contrast:

\[
F_{\mathrm{axis\_horizontal}} > F_{\mathrm{axis\_vertical}}.
\]

If this fails, interpret cautiously. The result may be generic modulation rather than task-specific spatial sampling.

## 4. Motion scale curve

Use scaled real traces or synthetic diffusion sweeps.

Expected qualitative result:

- no motion: lower information,
- small FEM-like motion: higher information,
- very large motion: saturation or decline.

If information increases monotonically over all tested scales, the tested range may be too narrow or the noise/readout model may be too permissive.

## 5. Order-shuffle control

Compare:

- real ordered trace,
- same positions shuffled in time.

If similar:

- mechanism is phase coverage, not temporal order.

This is a positive, manuscript-consistent outcome if other figures already argue for trajectory-agnostic effects.

If ordered real is stronger:

- possible sequence or temporal-filter contribution, but interpret with GRU-history caution and check `movie_mode` versus `framewise_or_reset_mode`.

## 6. Spike-count audit

For every motion condition, report:

- final expected spikes,
- raw Fisher,
- Fisher per expected spike,
- threshold ratio.

A positive result should not depend only on larger spike counts.

## 7. Offset symmetry, rendering, and provenance audit

Verify:

- responses to `+delta` and `-delta` are symmetric in expectation,
- the same trajectory and inference path are used for matched `+delta` and `-delta` finite-difference pairs,
- no rendering bias,
- no systematic luminance/contrast difference between signs,
- pixel-level Fisher and luminance-centroid baselines do not trivially explain the model effect,
- model-input feature sizes and offsets sit in a defensible spatial-frequency regime,
- FD Jacobians come from the canonical validated twin inference path.

---

# Optional trajectory optimization

## Optimized trajectory

Define:

\[
e^* =
\arg\max_e F_\delta(e) - C(e)
\]

with a motor cost:

\[
C(e) =
\lambda_1 \sum_t \|e_t\|^2
+
\lambda_2 \sum_t \|e_t-e_{t-1}\|^2
+
\lambda_3 \sum_t \|e_t-2e_{t-1}+e_{t-2}\|^2.
\]

Constraints:

- fixation radius bound,
- velocity bound,
- acceleration bound,
- optional RMS displacement matched to empirical traces.

## Adversarial trajectory

Define:

\[
e^- =
\arg\min_e F_\delta(e) + C(e)
\]

under the same constraints.

## Interpretation

If:

- optimized > real/random > adversarial,

then motion statistics matter and real FEMs lie in a useful range.

If:

- real \(\approx\) random matched,

then exact biological order is not necessary. This is a positive result if it agrees with trajectory-agnostic analyses elsewhere in the paper.

If:

- real/random \(\approx\) adversarial,

then the metric is not capturing a useful effect or constraints are too weak.

Run optimization under both pose-aware and pose-blind objectives if this section is implemented. A pure pose-aware optimizer may degenerate into camping on the best phase; the informative quantity is the gap between pose-aware-optimal and pose-blind-optimal trajectories under the same motor constraints.
---

# Suggested implementation plan

## Script names

Create:

```text
declan/active_sensing_vernier/run_vernier_fem_information.py
declan/active_sensing_vernier/summarize_vernier_fem_information.py
declan/active_sensing_vernier/render_vernier_stimuli.py
```

Optional later:

```text
declan/active_sensing_vernier/run_vernier_trajectory_optimization.py
```

## Output directory

```text
outputs/active_sensing_vernier/vernier_fem_information/<run_name>/
```

## Required output files

### Metadata and manifests

```text
manifest.json
README.md
config_resolved.yaml
model_inventory.csv
stimulus_inventory.csv
motion_condition_inventory.csv
```

### Stimulus audits

```text
stimulus_contact_sheet.png
stimulus_difference_contact_sheet.png
stimulus_line_profiles.csv
stimulus_symmetry_audit.csv
stimulus_rendering_cutoff_audit.csv
stimulus_pixel_fisher_audit.csv
stimulus_luminance_centroid_audit.csv
```

### Motion audits

```text
motion_trace_summary.csv
motion_trace_examples.png
motion_scale_summary.csv
motion_axis_summary.csv
```

### Model outputs

Use compact array storage:

```text
rates_by_condition.zarr
```

or:

```text
rates_by_condition.h5
```

Include dimensions:

- session/model
- stimulus parameter index
- offset
- motion condition
- trace index
- time
- neuron

### Primary metric tables

```text
vernier_fisher_timecourse.csv
vernier_fisher_summary.csv
vernier_threshold_summary.csv
vernier_dprime_summary.csv
spike_count_audit.csv
noise_model_sensitivity.csv
pose_readout_sensitivity.csv
pose_blind_full_cov_summary.csv
compact_aware_pose_blind_summary.csv
nuisance_subspace_summary.csv
reliability_variance_summary.csv
axis_control_summary.csv
motion_scale_summary.csv
order_shuffle_summary.csv
static_phase_cloud_summary.csv
```

### Bootstrap summaries

```text
bootstrap_summary.csv
decision_table.csv
```

### Figures

```text
fig_vernier_stimulus_schematic.png
fig_fisher_timecourse.png
fig_threshold_by_condition.png
fig_axis_control.png
fig_motion_scale_curve.png
fig_static_phase_cloud_control.png
fig_spike_count_audit.png
fig_order_shuffle_control.png
fig_compact_aware_pose_blind_k_sweep.png
fig_compact_aware_pose_blind_alpha_sweep.png
```

Optional later:

```text
fig_optimized_vs_adversarial.png
fig_optimized_trace_statistics.png
```

---

# Recommended first-pass configuration

Use a smoke test before full production.

## Smoke test

- one session/model
- one canonical Vernier stimulus
- offsets: `[-1, -0.5, -0.25, 0, 0.25, 0.5, 1] arcmin`
- conditions:
  - `static_repeated_phase`, with explicit `p0_center` and cloud-drawn/quantile phases
  - `static_phase_cloud_single`
  - `static_phase_cloud_matched_positions`
  - `real_fem`
  - `random_amp_matched`
  - `axis_horizontal`
  - `axis_vertical`
  - `scaled_real`, with `D = [0, 0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3]`
- readout/noise model:
  - pose-aware diagonal Poisson
  - pose-blind diagonal or marginal covariance
  - pose-blind full covariance with shrinkage, if sample support is adequate
  - compact-aware pose-blind hard projection and soft discounting
  - pose-aware fixed diagonal
  - pose-blind fixed or marginal covariance
- duration:
  - match current model movie duration, or use 1 second if consistent with prior analyses
- summary:
  - Fisher timecourse
  - threshold-proxy ratios versus repeated-phase and phase-cloud baselines
  - reliability and lower-tail Fisher summaries
  - spike-count audit
  - rendering cutoff, pixel-Fisher, and luminance-centroid audits

## Production run

- all relevant trained model sessions/readouts
- multiple Vernier parameter settings
- real, random, shuffled, and scaled trajectories
- all noise models
- bootstrap over:
  - stimulus settings,
  - traces,
  - sessions/readouts,
  - optionally neurons if computationally feasible

---

# Statistical aggregation

Use hierarchical summaries where possible.

Recommended bootstrap levels:

1. session/model
2. stimulus parameter condition
3. trajectory instance

Do not treat all time bins or all traces as fully independent if they share the same model and stimulus.

Primary aggregate:

- median or mean across stimulus/trace conditions within session,
- then bootstrap across sessions/models.

Report:

- mean effect,
- median effect,
- variance and lower-tail effects,
- 95 percent bootstrap CI,
- sign count across sessions,
- sensitivity across noise models and pose readouts,
- the number of sessions/readouts included in each panel or table.

Pair comparisons within session, stimulus condition, finite-difference step, trajectory family, and readout. Do not let time bins or trace instances inflate the effective sample size.

Primary effect sizes:

\[
\Delta F_{\mathrm{cloud}} =
F_{\mathrm{motion}} - F_{\mathrm{static\_phase\_cloud}}
\]

\[
\Delta F_{\mathrm{repeat}} =
F_{\mathrm{motion}} - F_{\mathrm{static\_repeated}(p_0)}
\]

and normalized threshold-proxy ratios:

\[
R_{\theta,\mathrm{cloud}} =
\sqrt{
\frac{F_{\mathrm{static\_phase\_cloud}}}
{F_{\mathrm{motion}}}
}
\]

\[
R_{\theta,\mathrm{repeat}} =
\sqrt{
\frac{F_{\mathrm{static\_repeated}(p_0)}}
{F_{\mathrm{motion}}}
}.
\]

Also report per-spike versions and reliability/lower-tail versions.

---

# Decision table

## Strong phase-cloud positive

Criteria:

- real or FEM-like motion improves Fisher or \(d'^2\), or improves lower-tail reliability, versus static phase-cloud baselines,
- threshold-proxy ratio versus phase cloud is below 1 or reliability improves,
- improvement survives per-spike normalization,
- recoverability survives a specified, biologically plausible level of pose uncertainty,
- horizontal motion beats vertical motion for vertical Vernier,
- effect strongest near small offsets,
- rendering and pixel-level controls do not explain the effect.

Interpretation:

> FEM-like motion improves fine positional sensitivity in the V1 twin beyond matched static phase-cloud sampling, with recoverable information under explicitly stated pose-knowledge assumptions.

## Corroborating phase-coverage positive

Criteria:

- real \(\approx\) random matched \(\approx\) order-shuffled \(\approx\) matched static phase-cloud under pose-aware pooling,
- these phase-cloud conditions beat unlucky or central repeated phases, or reduce lower-tail variance across repeated-phase draws,
- optimized improves further and adversarial is worse, if optimization is run.

Interpretation:

> Exact biological trajectory order is not uniquely optimal; FEM-like motion lies in a broad useful phase-coverage regime. This is a positive result if it agrees with trajectory-agnostic analyses elsewhere in the paper.

## Pose-aware-only benefit

Criteria:

- motion improves under pose-aware Fisher,
- benefit weakens or disappears under pose-blind readout,
- static phase-cloud controls explain most of the mean improvement.

Interpretation:

> The sampled phases contain Vernier information, but downstream recoverability depends on knowing or estimating retinal pose. Frame this as a pose-aware phase-sampling result, not a full active-sensing gain.

## Coordinate-frame boundary result

Criteria:

- phase-variable conditions beat `static_center` under pose-aware Fisher,
- the same phase-variable conditions are strongly penalized under pose-blind readout,
- real FEM does not beat a matched phase-cloud baseline, or only does so after changing the motion scale,
- scale-specific matched phase-cloud controls are needed before claiming a tuned-motion advantage.

Interpretation:

> Motion helps a pose-aware observer and strongly burdens a pose-blind observer. The result identifies a retinal-pose requirement for using phase-dependent Vernier information rather than proving that natural fixation traces are optimized for Vernier acuity.

## Compact-aware pose-blind rescue

Criteria:

- ordinary pose-blind readout is strongly below pose-aware readout,
- leading nuisance modes explain a compact share of phase/trajectory covariance,
- hard projection or soft discounting raises Fisher/static-center ratios above ordinary pose-blind without approaching a pose-aware oracle,
- the improvement is stronger for Vernier/FEM-derived nuisance subspaces than for random orthonormal controls,
- conclusions are stable over reasonable \(k\), \(\alpha\), and shrinkage choices.

Interpretation:

> Compact geometry partially rescues pose-hidden Vernier information: the nuisance covariance is structured enough that a covariance-aware observer can discount it, but explicit or estimated retinal pose remains valuable.

If compact-aware \(\approx\) ordinary pose-blind, interpret the compact subspace as identifying the nuisance without recovering the active-sampling benefit. If results are fragile to \(k\), shrinkage, or subspace source, keep the analysis as an internal control rather than a paper claim.

## Trajectory-specific or temporal-order effect

Criteria:

- real ordered traces beat random matched and order-shuffled controls,
- `movie_mode` beats `framewise_or_reset_mode`,
- effect survives the stated pose-uncertainty analysis.

Interpretation:

> There may be a sequence-specific or model-history contribution. This is interesting but must be reconciled with any trajectory-agnostic claims elsewhere in the manuscript.

## Weak or generic modulation

Criteria:

- real improves raw Fisher versus repeated static only,
- improvement disappears per spike,
- horizontal and vertical motion are similar,
- static phase-cloud matches real motion,
- pose-blind readout removes the effect.

Interpretation:

> Motion changes response magnitude or phase selection, but evidence for active-sensing hyperacuity is weak.

## Negative

Criteria:

- real does not beat phase-cloud or repeated-static baselines under reasonable noise models,
- no reliability benefit,
- no axis specificity,
- no motion-scale structure.

Interpretation:

> The V1 twin does not support a Vernier active-sampling benefit under this setup. Revisit stimulus scale, model spatial resolution, noise assumptions, pose treatment, and rendering.

---

# Practical cautions

## Avoid fixed-center oracle overinterpretation

`static_center` is useful, but it should not be the only baseline. A perfectly centered static stimulus can be an unrealistic oracle phase. The main load-bearing baselines are the static phase-cloud controls. `static_repeated_phase` is still valuable, but its interpretation depends on the specified repeated phase \(p_0\).

## Avoid long-memory claims

Do not say the model integrates over the whole fixation internally. The analysis integrates the predicted short-latency V1 responses over time.

## Check rendering scale carefully

This analysis is only meaningful if subpixel offsets are faithfully rendered and if the model's stimulus grid can represent the Vernier feature. The stimulus audit is mandatory and must include model-grid/cutoff characterization, pixel-level Fisher, luminance-centroid baselines, and finite-difference step-size stability.

## Do not use an overly flexible decoder as the first endpoint

Start with Fisher and \(d'^2\). A learned decoder can be added later, but Fisher/d-prime are more transparent and less likely to exploit model idiosyncrasies.

## Match spike and time budgets

Motion and static conditions must use the same duration and should be summarized with expected spike counts. Matched `+delta` and `-delta` finite-difference pairs must share the same trajectory or phase draw.

---

# Minimal coding checklist

1. Implement anti-aliased Vernier renderer.
2. Validate offset symmetry, line profiles, pixel-level Fisher, luminance-centroid baselines, and model-grid/cutoff regime.
3. Implement retinal motion renderer for repeated static, static phase-cloud, real, random, axis, shuffled, and scaled conditions.
4. Run the twin through the canonical validated inference path to obtain predicted rates/counts.
5. Compute finite-difference derivatives with respect to Vernier offset using paired trajectories and multiple step sizes.
6. Compute Fisher, threshold-proxy ratios, and \(d'^2\) under pose-aware and pose-blind diagonal Poisson and fixed/marginal covariance models.
7. Summarize cumulative timecourses, endpoint effects, and reliability/lower-tail effects.
8. Run static phase-cloud, repeated-phase, and axis controls.
9. Run spike-count audit.
10. Generate decision table.
11. Only then add optimized/adversarial trajectories under both pose-aware and pose-blind objectives.

---

# Plain-language expected result

If the analysis works, the paper-facing result should read something like:

> In a virtual Vernier acuity task, retinal phase strongly shaped the V1 twin's sensitivity to tiny spatial misalignments. When retinal pose was known, phase-variable sampling exposed more Vernier information than a single centered static phase. When retinal pose was hidden, the same phase variability became a nuisance covariance that strongly reduced recoverable information. The result should therefore be framed as a pose-dependent phase-sampling effect, not as a naive claim that natural fixation traces improve Vernier acuity. If reduced-amplitude or otherwise task-tuned trajectories beat scale-specific matched phase-cloud baselines after rendering, axis, reliability, and pose-uncertainty controls, they can support the narrower claim that fine-acuity tasks favor a smaller or more structured subset of the broader FEM regime.
