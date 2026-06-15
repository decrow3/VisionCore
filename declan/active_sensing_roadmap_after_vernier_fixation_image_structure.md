# Active-Sensing Roadmap After Vernier, Fixation-Regime, and Image-Structure Results

Last curated: 2026-06-15.

This note updates the active-sensing branch after the Vernier hyperacuity work,
fixation-statistics-by-stimulus analysis, BackImage local image-structure
analysis, scaled BackImage twin drift-geometry adjudication, and the
input-whitening negative result.

Companion response-space ledger:

```text
active_sensing_unit_space_provenance.md
```

Use that ledger before comparing 16-channel matched/session results,
sampled-population objectives, and full 756-channel canonical results.

## Core Update

The active-sensing branch has become more coherent, but also more constrained.
The strongest current claim is not that real fixational eye movements are
uniquely optimal, nor that any retinal motion improves information. The better
supported claim is:

```text
Fixational eye movements create structured, phase-dependent input to foveal V1.
This structure can be useful when retinal pose is known, costly when pose is
hidden, and shaped by behavioral regime and local image geometry. Drift axes
during BackImage viewing are robustly aligned with raw local edge geometry, but
the current V1-twin PA/PB/Pareto axis objectives do not add explanatory power
beyond that edge baseline. Drift also whitens natural-image input relative to
stabilization, but unconstrained input whitening alone does not determine
biological FEM amplitude.
```

This reframes the digital twin. The twin should not primarily ask whether
measured eye traces beat stabilized controls. It should be used as a virtual
preparation for asking:

1. What small retinal motions would be useful for a given local image?
2. What motions would be harmful under the same motor budget?
3. Which observer assumptions make motion beneficial versus costly?
4. Do real fixation statistics move in the directions predicted by the model?
5. Which tradeoff, not which single objective, places biological FEM statistics
   in a useful part of the landscape?

The key empirical loop is:

```text
local image patch
-> model-predicted useful motion geometry
-> observed FEM statistics
```

This is the most direct current test of whether FEMs are part of an adaptive
sampling strategy.

## What We Now Know

### 1. Vernier Acuity: Phase Sampling Helps Only Under the Right Coordinate Frame

The Vernier sweep showed that retinal phase diversity can substantially
increase fine-position information in the V1 twin for a pose-aware observer.
Pose-blind readouts are strongly penalized by the same phase variability, and
increasing pose uncertainty smoothly degrades the pose-aware benefit.

Important details:

- Unit-space audit: the first-pass and component-scale Vernier summaries use
  `756` canonical units. The scale/pose sweep's main pose-aware rows also use
  `756` units, while its compact-aware controls use a `256`-unit `top_abs_fd`
  subset from the same original 756-unit space.
- Noise/readout audit: the Vernier result did not rely on simulated noisy spike
  draws or an empirically fitted trial-noise model. It used deterministic twin
  rates and explicit Fisher/readout assumptions, especially pose-aware
  diagonal-Poisson Fisher. The pose-blind diagonal count-plus-marginal readout
  behaved very differently, so the absolute numbers remain observer-model
  dependent.
- Full real FEM is not optimal for the Vernier task.
- Reduced motion, especially `D ~= 0.125` to `0.25`, gives the strongest
  pose-aware Vernier information.
- Much of the advantage over static center is explained by sampled phase
  distribution, not exact real trajectory order.
- Pose-blind information remains far below static center for phase-variable
  conditions.
- Compact-aware pose-blind controls do not rescue the pose-aware benefit.
- Vertical-only motion outperformed horizontal-only motion in the current
  Vernier setup; this needs a rotated-stimulus control before interpretation.

Current interpretation:

```text
Fine-acuity tasks may prefer smaller, more controlled retinal motion than
generic fixation traces. Phase sampling can provide useful fine-position
information, but only if the observer has sufficiently accurate retinal-pose
information.
```

This supports the coordinate-frame asymmetry. Motion-induced structure is not
intrinsically signal or noise. It becomes signal when interpreted in the right
chart and nuisance covariance when pose is ignored.

### 2. Input Whitening: Old Temporal-PSD Metric Superseded by Rucci-Style Audit

The input-whitening analysis was designed as a non-circular ecological anchor:
it uses natural images and drift kinematics rather than twin responses to ask
whether biological FEM scale is predicted by retinal input whitening.

Old pooled temporal-PSD result:

- Estimated biological fixation drift was `D = 0.00110667 deg^2/s`
  (`3.984 arcmin^2/s`), with drift-fit `R2 = 0.916`.
- The run evaluated `1458` retinal movies and `157464` metric rows.
- Stabilization produced a steep, low-entropy temporal spectrum in the primary
  `4-40 cpd`, `1-30 Hz` passband.
- Measured biological drift strongly whitened the input relative to
  stabilization: PSD slope improved from `-4.207` to `-1.047`, spectral entropy
  from `0.194` to `0.579`, and spectral flatness from `0.006` to `0.345`.
- However, the unconstrained whitening objective usually kept improving up to
  the largest tested scale. Across all passband-metric rows, `956 / 972`
  optima chose `D_scale = 3`; all entropy and flatness optima chose
  `D_scale = 3`.
- The few exceptions were slope-only measured-drift cases at `D_scale = 0.125`
  for higher temporal lower-bound passbands.
- The paired image/crop bootstrap is not implemented yet in this runner, so the
  whitening result should be treated as a completed deterministic/SEM summary
  with passband sensitivity, not as a bootstrap-resampled uncertainty claim.

Correction:

```text
The old metric showed that larger retinal motion spreads pooled temporal power.
It was not a faithful Rucci-style spatial power-law whitening test.
```

The newer Rucci-style audit uses the spatial-frequency spectrum of
frame-to-frame retinal modulation. It asks whether motion flattens the spatial
dependence of modulation power and reports amplitude scale separately from the
approximate diffusion/power scale (`diffusion_scale = amplitude_scale^2`).
Early smoke runs show a cleaner dissociation:

- total frame-to-frame modulation power still increases with motion and peaks
  at the largest tested scale;
- spatial power-law flattening and derivative-transfer sanity checks peak at
  small nonzero motion (`0.125x` in the ungated smoke, `0.25x` after excluding
  scales below `5%` of biological-scale modulation power);
- at small motion, the transfer slope is close to the derivative prediction
  that modulation/source power should gain about `+2` log-log slope units.

Current interpretation:

```text
FEMs reformat natural-image input, but "whitening" is not one scalar objective.
Pooled temporal-power spreading, Rucci-style spatial power-law flattening, and
task/feature information have different scale optima.
```

This is a useful negative result. It closes off the simplest ecological story:

```text
Biological FEM amplitude is not explained by the old pooled temporal-PSD
whitening metric, and the Rucci-style spatial flattening audit has not yet made
whitening a standalone scale-setting answer.
```

The result should not be read as "biology should use 9x more diffusion." That
was an artifact of using pooled temporal entropy/flatness as if it were
Rucci-style whitening. The current conclusion is more disciplined: a small
amount of motion may best compensate the natural-image power law, larger motion
adds more temporal drive, and biological drift scale likely reflects additional
downstream and motor constraints such as pose uncertainty, fine acuity,
covariance cost, perceptual stability, fixation-window limits, temporal
sensitivity, and motor cost.

### 3. Compact Geometry: Useful Structure, Not a Complete Denoising Mechanism

The compact reafferent geometry remains important. Translation-induced
variability is structured and low-dimensional rather than arbitrary. However,
the denoising and compact-aware Vernier results bound the claim.

Supported:

- FEM-linked covariance is compact.
- Translation effects are routed through a shared low-dimensional population
  geometry.
- Compactness makes FEM-induced variability identifiable as structured
  reafference.

Not established:

- A downstream observer can subtract the actual trial-specific FEM fluctuation
  without pose information.
- A fixed compact projection recovers the pose-aware fine-position benefit.
- Compactness alone solves the pose problem.

Updated interpretation:

```text
Compactness makes the consequences of FEMs structured and potentially
manageable, but it does not replace retinal-pose information when the task
depends on fine spatial position.
```

### 4. Fixation Statistics Differ by Behavioral and Stimulus Regime

The fixation-statistics analysis showed that eye movements during fixation are
not fixed motor noise. They differ substantially across stimulus regimes.

Reviewed summaries indicate that BackImage and forage-like stimuli produce
broader fixation-window spatial spread than fixRSVP, even after within-window
centering. BackImage fixations are especially broad and somewhat faster. This
difference persists beyond immediate event transients.

Current interpretation:

```text
The eye movement regime used during fixRSVP should be treated as a tight
fixation-maintenance regime, not as a universal FEM policy. Natural image
viewing uses a broader and more structured fixation regime.
```

This matters for Vernier. Real traces from steady fixation or free viewing
should not be expected to be optimized for a Vernier task the animal was not
doing. A task-specific acuity regime could plausibly be smaller and more
controlled.

### 5. Local Image Statistics Weakly Predict Scalar FEM Metrics, but Orientation Alignment Survives

The local BackImage analysis does not support a broad scalar tuning story.
Local image features do not robustly improve prediction of RMS radius,
diffusion, speed, path length, anisotropy, return-to-center strength, or
high-frequency FEM fraction over controls.

The surviving result is directional:

- Drift or fixation-cloud orientation tends to align with local edge and
  spectral axes.
- The effect is modest, but stronger in reliable-axis subsets.
- The alignment is roughly parallel to local edge/spectral structure, not along
  the image gradient axis.

Current interpretation:

```text
Within selected natural-image fixation locations, local image structure appears
to shape the geometry of fixational motion more than its scalar amplitude.
```

The next active-sensing tests should focus less on "does high local information
make the eye move more?" and more on:

```text
Does local image geometry predict the direction, anisotropy, or scale of fine
retinal sampling?
```

### 6. Fixation Locations Are Modestly Contrast-Biased

Same-image random-location controls show that actual fixations are somewhat
higher contrast than random patches from the same image, but they are not
uniformly higher across all information metrics.

Current interpretation:

```text
Larger eye movements select somewhat more structured locations. Fine eye
movements may then adjust local sampling geometry within those selected
locations.
```

This suggests a two-stage oculomotor story:

1. Saccades choose informative regions.
2. FEMs tune local sampling geometry within those regions.

### 7. Scaled Twin Drift-Geometry Adjudication: Raw Edge Geometry Wins

The first CUDA pilot suggested a possible stability-like result, but the scaled
run resolved that ambiguity against the current V1-twin axis objectives. The
scaled run used corrected eye-coordinate order, a `270 px` full-image support
margin for `540 px` BackImage patches, an axis-only grid, `256` windows across
`29` sessions, `5000` candidate-grid axis nulls, `5000` predicted-axis shuffles,
and `5000` session bootstraps.

Unit-space audit: the folder's `n256` label refers to `max_windows=256`, not
population size. Saved `run_metadata.json` reports `twin_population_n=64`, so
this is a sampled digital-twin-population diagnostic rather than a full
canonical 756-unit free-viewing objective.

Scaled outcome:

- `raw_edge_axis`: session mean cos2 `+0.182`, weighted `+0.218`, `23/29`
  positive sessions, random-axis `p_ge = 0.0004`.
- `optimized_PB`: session mean cos2 `-0.019`, weighted `+0.008`, not above
  raw edge. Paired delta versus raw edge was `-0.201`, CI
  `[-0.348, -0.064]`, with `5/29` positive sessions.
- `optimized_PA`: session mean cos2 `-0.008`, weighted `-0.002`. Paired delta
  versus raw edge was `-0.190`, CI `[-0.389, +0.007]`.
- `optimized_Pareto_lambda_0.5`: session mean cos2 `-0.010`, weighted
  `+0.004`. Paired delta versus raw edge was `-0.193`, CI
  `[-0.357, -0.020]`.
- `adversarial_Pareto_lambda_0.5`: session mean cos2 `+0.167`, weighted
  `+0.193`, but predicted-axis shuffle nulls were also high. Treat this as
  objective-landscape or image-geometry structure, not a biological optimized
  axis result.

Current interpretation:

```text
Observed BackImage drift is modestly and robustly aligned with local edge
geometry. The current V1-twin PA/PB/Pareto axis objectives do not outperform raw
edge orientation.
```

This is a useful negative/constraint. It argues against a naive "drift maximizes
instantaneous response modulation" objective, and against the current
64-sampled-unit pose-blind/stability objective as an explanation beyond raw edge
geometry. The local image-contingent coupling is real; the present model
objective is not yet the mechanism.

### 8. Post-Fix Gabor/Pyramid and Edge-Parallel Stability Audits

After reviewing the latent-feature implementation, the BackImage
latent-information screen was patched so that Gabor local fields include even,
odd, and amplitude maps on the local grid; pyramid local fields use the expanded
local grid; model outputs are cropped back to the requested trace length before
observer construction; and delta observers subtract the matched static response.

The first post-fix latent pathfinder was:

`outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_latent_information_pathfinder_fixall_n64_rel0125-05_rand8_delta`

At `0.125x` observed RMS, canonical 756-unit twin, `k=4`, and `8` random axes,
Gabor pose-blind delta produced a small-scale positive:

- real-minus-random `+9.02`, CI `[+1.56, +18.15]`;
- real-minus-edge `+11.58`, CI `[+1.59, +24.55]`.

Pyramid pose-blind delta pointed in the same direction but was noisier:

- real-minus-random `+10.28`, CI `[-0.53, +26.07]`;
- real-minus-edge `+25.04`, CI `[-3.78, +73.38]`.

Important provenance correction: the apparent larger Gabor-only check,
`backimage_latent_information_pathfinder_gabor_realrand_n128_rel0125-05_rand8_nogrd`,
is not a clean post-fix replication. Its saved Gabor local field has shape
`(128, 384)`, while the fixed 8x8 even/odd/amplitude Gabor local field has
shape `(N, 4608)`. It also used absolute observers rather than the delta
observers that produced the positive result. The larger stage2 run included
pyramid, but it was still n=64, used the older 4x4 feature dimensionality, and
used absolute observers.

A clean n=128 canonical replication was followed by a locked n=256 scale sweep.
The n=128 run was:

`outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_latent_information_cleanrep_n128_rel0125-05_rand8_delta`

This run used the canonical 756-unit twin, fixed Gabor local field
`(128, 4608)`, fixed pyramid local field `(128, 3072)`,
`pose_blind_delta_mean`, rand8, and scales `0.125x`, `0.25x`, and `0.5x`
observed RMS.

Primary Gabor `k=4`, `0.125x` replicated real-vs-random directionally but did
not beat raw edge:

- real-minus-random `+3.31`, CI `[-0.14, +8.57]`, p(delta<=0) `0.0396`;
- real-minus-edge `-0.36`, CI `[-2.25, +1.09]`;
- edge-minus-random `+3.67`, CI `[+0.08, +9.28]`.

The larger relative scales were not just nuisance positives. Several
real-vs-edge effects became strong at `0.5x` or nearby scales:

- Gabor `k=4`, `0.5x`: real-minus-edge `+6.60`,
  CI `[+1.53, +11.83]`;
- pyramid `k=4`, `0.5x`: real-minus-edge `+7.26`,
  CI `[+2.35, +12.08]`;
- pyramid `k=8`, `0.25x`: real-minus-edge `+8.57`,
  CI `[+1.62, +20.85]`, with real-minus-random `+2.60`,
  CI `[-0.05, +6.27]`.

The completed n=256 scale sweep is:

`outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_latent_information_scalesweep_n256_rel0125-2_rand8_delta`

It used canonical 756 units, fixed Gabor/pyramid local fields,
`pose_blind_delta_mean`, rand8, and scales `0.125x`, `0.25x`, `0.5x`, `1x`,
and `2x` observed RMS. A cheap real-vs-random audit was added at
`posthoc_real_random_audit_summary.md`.

Key n=256 results:

- Gabor `k=4`, `0.25x`: real-minus-random `+3.48`,
  CI `[+0.75, +6.87]`; unclipped subset `+2.85`,
  CI `[+0.21, +6.25]`.
- Pyramid `k=8`, `0.25x`: real-minus-random `+2.19`,
  CI `[+0.62, +4.18]`; unclipped subset `+1.86`,
  CI `[+0.45, +3.67]`.
- Gabor `k=4`, `1x`: real-minus-random `+2.59`, but the session CI crosses
  zero; the unclipped subset remains positive but guarded.
- Pyramid `k=8`, `1x`: global real-minus-random is weak, but the unclipped
  subset is `+1.40`, CI `[+0.05, +2.71]`.
- Clipping increases with scale: `1.2%`, `3.9%`, `9.4%`, `18.8%`, and
  `40.2%` for `0.125x` through `2x`.
- Subsampling explains the earlier mixed pathfinders: n=64 Gabor `k=4`, `1x`
  subsamples have a negative 5th percentile, while n=128 Gabor `k=4`, `0.25x`
  subsamples are usually positive.
- Leave-session-out is reassuring at `0.25x`: Gabor `k=4` LSO
  `+2.17/+3.52/+4.09`; pyramid `k=8` LSO `+1.41/+2.25/+2.38`.
- Regime stratification suggests stronger effects in high observed-RMS and high
  drift-anisotropy windows, with additional edge density/coherence support for
  pyramid `k=8`.

An optimized seed-dependence replication has been launched:

`outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_latent_information_scalesweep_n256_rel0125-2_rand8_delta_seed1_manifest_optimized_tb2`

This replays the same 256 windows from the completed run's `analysis_windows.csv`,
changes the random-axis seed to `1`, uses the patched manifest/source-row path,
canonical trace batching, and `--check-trace-batch-equivalence`. The first
trace-batch-8 launch exceeded GPU memory during the equivalence preflight, so
the live run uses `--twin-trace-batch-size 2` and `--twin-batch-size 48`.

The more stable BackImage result is now the explicit edge-parallel versus
edge-orthogonal stability audit. The endpoint cache and cheap synthesis live at:

`outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_twin_stability_metric_audit`

Signed edge-parallel stability was positive across pixel and twin metrics:

- pixel session mean `+300.5`, CI `[+172.8, +408.8]`;
- twin raw MSE `+0.0004545`, CI `[+0.0003716, +0.0005432]`;
- response-norm `+0.02456`, CI `[+0.01993, +0.02931]`;
- per-rate `+0.003688`, CI `[+0.002902, +0.004501]`;
- full-cov whitened `+0.1706`, CI `[+0.1511, +0.1890]`.

Pixel and twin signed advantages agree across windows: full-cov whitened
window-within-session `r = +0.277`, CI `[+0.168, +0.417]`; diagonal-whitened
`r = +0.287`, CI `[+0.139, +0.419]`. Session-mean correlations are noisy.

Current interpretation:

```text
BackImage drift is not yet explained by global feature-information
maximization. The credible BackImage result remains local preservation:
edge-parallel motion disrupts pixels and V1-twin responses less than
edge-orthogonal motion. The `I_z` branch is still alive, but its strongest
current support is a regime-dependent small-scale feature-information effect,
especially near `0.25x` observed RMS. The `1x` result is suggestive but not
clean enough to carry a figure-level infomax claim by itself.
```

This keeps the branch alive but narrows its claim. The next local-screen gate is
seed dependence: if the optimized same-window seed-1 replication preserves the
`0.25x` Gabor/pyramid positives, the local `I_z` branch can be reported as
exploratory regime-dependent support. If it does not, demote it behind the
edge-parallel preservation result.

### 9. Aggregate Natural-Image FEM Information Is the Next Figure-Level Candidate

The local per-fixation `I_z` screen asks a difficult question: whether each
measured drift axis is optimal for its exact local patch. The new aggregate
analysis asks a broader and probably more biologically plausible question:

```text
Across natural-image patches I ~ p(I), does the empirical FEM distribution
q_real(tau) produce a better V1-twin representation of image information than
matched non-biological motion controls q_control(tau)?
```

The plan is captured in:

`declan/backimage_aggregate_fem_information_plan.md`

The key shift is from local policy matching to ensemble distributional
adaptation. Instead of asking whether fixation `i` uses the best axis for patch
`I_i`, compare motion distributions across many image samples:

- static;
- empirical FEM traces;
- scaled empirical traces (`0.125x`, `0.25x`, `0.5x`, `1x`, optional `1.5x`);
- OU matched to RMS/autocorrelation/confinement as the primary synthetic null;
- Brownian matched to effective RMS/diffusion as a secondary generic-diffusion
  null;
- shuffled, rotated, or phase-randomized empirical controls.

Primary readouts should include ensemble image-feature decoding and a
signal-versus-motion-nuisance covariance decomposition, with temporal PCs as
the primary response summary and mean-over-trajectory as a diagnostic. Promote
signal-motion subspace overlap because it connects directly to the covariance
story. Every nominal-scale summary must report effective RMS, clipping, path
length, and motion-energy matching. Figure-level claims should be twin-scoped.
Real tying OU is not automatically a failure; it means broad FEM-like
confinement/autocorrelation may be sufficient. If all motion helps and the
largest motion always wins, the metric is generic modulation unless the
signal/motion overlap shows a useful frontier.

## What Should Be Demoted

### Real FEM Versus Stabilized as the Main Active-Sensing Test

This comparison is too blunt. It does not distinguish pose-aware signal from
pose-blind nuisance, phase distribution from trajectory order, or task-tuned
motion from generic fixation statistics.

### Unconstrained Whitening as the FEM Scale Setter

Input whitening is a benefit of drift, but it is not the objective that sets
biological amplitude in the current grid. The old pooled temporal-PSD metric
showed that larger motion spreads temporal power, but the Rucci-style
spatial-power audit shows that spatial flattening peaks at small nonzero motion
in smoke runs. This should be framed as an ecological constraint, not as a
standalone optimality account.

### Exact Trajectory Optimality

Exact real trace order is not uniquely special in the current results. Real,
matched phase cloud, random matched, and order-shuffled controls are often
similar. This is not a failure. It means the relevant object is likely movement
statistics and phase distribution, not exact trace identity.

### Scalar Local Image Information Predicts Drift Amplitude

The audited BackImage analysis does not support this. Local scalar features do
not robustly predict speed, RMS radius, diffusion, or path length.

### Compactness-Aware Pose-Blind Rescue

Compact-aware projection or discounting does not recover the pose-aware Vernier
benefit. This should remain a boundary result, not a main positive claim.

### Strong Temporal-Code Framing

The current evidence does not require precise temporal order or long-memory
integration. A better framing is phase sampling and coordinate-frame
dependence.

## What Should Be Promoted

### Coordinate-Frame Dependence

The central active-sensing conclusion is:

```text
The same retinal motion can create useful information for a pose-aware observer
and nuisance covariance for a pose-blind observer.
```

This is supported by the Vernier pose-aware versus pose-blind sweep, the
pose-uncertainty results, and the compact-aware negative control.

### Task-Specific Motion Scale

Vernier prefers reduced motion, around `D=0.125` to `0.25`, rather than full
real FEM. The useful claim is:

```text
Real fixation traces are not expected to be Vernier-optimal. The twin predicts
that a fine-acuity regime would favor smaller retinal phase sampling.
```

Input whitening is no longer a single arrow. Pooled temporal-power spreading
favors larger motion, while Rucci-style spatial power-law flattening favors
small nonzero motion in smoke runs. This makes biological scale look like a
compromise between tasks and constraints rather than the peak of one scalar
objective.

### Multi-Objective Tradeoff Frontier

The active-sensing question should now be:

```text
Which tradeoff places biological FEM statistics near a useful part of the
landscape?
```

not:

```text
Which single objective produces biological FEMs?
```

The qualitative pressure table is now:

| Pressure | Direction it pushes motion scale |
| --- | --- |
| avoid stabilization / whiten natural input | larger than zero |
| maximize pooled temporal power spreading | larger than biological |
| Rucci-style spatial power-law flattening | small nonzero motion |
| fine spatial acuity / Vernier | smaller than biological |
| reduce pose-blind covariance burden | smaller or more constrained |
| motor and fixation stability | smaller or constrained |
| natural viewing exploration | broader and more structured |

The first cache-only tradeoff summary should be treated as a directional
diagnostic, not an explanatory model. Its main conclusion is:

```text
The ingredients point in sensible directions, but a generic scalar tradeoff does
not yet explain biological scale.
```

The old pooled temporal-power metric wants larger motion, while the newer
Rucci-style spatial flattening audit and Vernier acuity both push toward small
nonzero/reduced motion. Pose-blind covariance and generic diffusion costs also
push toward smaller or more constrained motion. The one-sided
above-biological window penalty can recover `D_scale = 1`, but partly by
construction because it explicitly says that motion above the biological range
is costly. That may point toward the right biological class of constraints, but
it is not yet a measured cost.

The next tradeoff work should replace this placeholder with measurable terms:

- probability of leaving a fixation window;
- expected displacement outside a foveal region;
- pose-uncertainty or pose-precision cost;
- output covariance cost;
- loss of Vernier or fine-position information;
- temporal power outside V1 usable bands;
- motor energy, speed, or acceleration cost inferred from real regimes.

### Rucci-Style and V1-Weighted Whitening

The old pooled temporal metric asked:

```text
Is the retinal temporal spectrum flat?
```

The Rucci-style question is:

```text
Does frame-to-frame retinal modulation flatten the spatial-frequency power law?
```

A complementary biological question is:

```text
Does drift place natural-image temporal modulations into temporal frequencies
that foveal V1 can actually encode?
```

This is the next non-circular input-statistics test. Define a weighting function
`W(f_t, f_s)`, where `f_t` is temporal frequency and `f_s` is spatial
frequency. Candidate implementations:

- model-derived temporal sensitivity from drifting gratings or filtered natural
  movies across temporal frequencies;
- output modulation spectra for each movie and scale;
- derivative-weighted sensitivity, `W(f) proportional to |d mu / d s_f|^2`;
- noise-normalized sensitivity, `W(f) proportional to response_gain(f)^2 /
  noise(f)`.

Then replace raw input flatness with either:

```text
weighted_whitening(D) = flatness(W(f) * P_D(f))
```

or a more biologically meaningful usable-power objective:

```text
usable_power(D) = sum_f W(f) P_D(f)
                - lambda * sum_f P_D(f) 1[f outside usable band]
```

Interesting outcomes:

- If V1-weighted whitening peaks closer to `D=1`, raw input whitening was too
  broad and the V1 temporal transfer function supplies a real non-circular
  constraint.
- If it still peaks at the upper boundary, even V1-weighted input statistics do
  not determine biological scale.
- If it peaks below biological scale, it aligns with Vernier and suggests
  natural viewing uses larger motion for reasons beyond early V1 encoding.

### Regime-Dependent FEM Statistics

FixRSVP, BackImage, Gaborium, and gratings produce different fixation-window
dynamics. This anchors the idea that FEMs are flexible and context-dependent.

### Local Image Geometry, Not Scalar Feature Magnitude

The current BackImage result points to orientation alignment. This should become
a central target for model-based prediction:

```text
Does the V1 twin predict that useful or stable motion should run parallel to
local edge/spectral structure?
```

### Conditional Fixation Objective

The better free-viewing objective is conditional on the already-selected local
patch:

```text
move enough to refresh the retinal input, but preferentially along directions
that minimally disrupt the selected foveal structure.
```

The first implemented pass adds this objective family to
`run_backimage_twin_drift_geometry.py` as `conditional_proxy`, with pixel
isophote/stability costs, saturating path-refresh benefit, and refresh-plus-
stability tradeoffs. The `n=256` BackImage proxy run found that the explicit
pixel-isophote objective was robustly above a random-axis null and essentially
tied raw edge orientation, but did not clearly beat it:

- `raw_edge_axis`: session mean cos2 `+0.182`, `23/29` positive sessions.
- `optimized_pixel_isophote`: session mean cos2 `+0.200`, `22/29` positive
  sessions.
- `optimized_pixel_isophote - raw_edge_axis`: paired session delta `+0.018`,
  bootstrap CI `[-0.044, +0.084]`.

This supports the stability/isophote framing as a good image-geometry baseline,
but not yet as added mechanistic explanation beyond raw edge geometry. The next
version should run the same conditional objective with true V1 response-
stability costs and ask whether it explains deviations from the raw edge axis,
anisotropy, or scale.

The true V1 response-stability run and cache-first residual/confidence summary
tighten this boundary. On the comparable `n=256`, axis-only twin grid,
`optimized_response_stability` had session mean cos2 `-0.019` and a paired
delta versus raw edge of `-0.201`, CI `[-0.349, -0.069]`. High-confidence
response-stability landscapes did not rescue the result: high-confidence
windows had session delta `-0.255`, while low-confidence windows had `-0.180`.
The response-stability margin had near-zero Spearman correlation with residual
improvement over raw edge (`rho ~= -0.05`) and a modest positive correlation
with drift anisotropy (`rho ~= +0.20`), which is hypothesis-generating only.

Current conclusion:

```text
BackImage drift-axis geometry is best treated as a raw local image-geometry and
local-preservation effect. The current V1 twin response-stability objective does
not explain the axis itself as an optimizer, but signed edge-parallel stability
is robust across pixel and twin metrics at the tested small endpoint.
```

### Optimized Versus Adversarial Local Motion Geometry

Trajectory optimization should define useful and harmful local motion directions
for each image patch, not attempt to recover exact biological traces.

## Updated Main Hypothesis

```text
FEMs are a context-dependent retinal sampling process. Their utility depends on
local image geometry, movement scale, and observer coordinate frame. During
natural viewing, FEMs may not maximize raw response modulation or raw input
whitening. Instead, their local geometry and scale may balance input whitening,
fine sampling, pose uncertainty, stability, and pose-induced covariance.
```

This predicts that observed drift geometry should be better matched to a Pareto
or pose-blind/stability objective than to a pure pose-aware modulation objective.

## Updated Trajectory-Optimization Program

### Main Goal

Use the V1 twin to predict which small motion axes and scales are useful or
harmful for each natural image patch, then test whether real drift statistics
align with those predictions.

Primary question:

```text
Given a local image patch, does the observed drift axis resemble the
model-predicted useful axis more than an adversarial or shuffled axis?
```

### Why Use Axes Before Full Traces?

Full trajectory optimization is powerful but likely to produce degenerate or
model-exploiting solutions. The next analysis should first optimize over a
low-dimensional family:

- motion axis;
- scale;
- anisotropy;
- temporal smoothness or path template.

This is more interpretable and directly connected to the observed edge-axis
alignment result.

### Candidate Motion Families

For each local patch, define candidate trajectories with:

- axis angle `theta`;
- major-axis scale `D_parallel`;
- minor-axis scale `D_perp`;
- anisotropy ratio;
- fixed duration matching the observed fixation window;
- matched RMS displacement or path length.

Simple first grid:

```text
theta = 0, 15, 30, ..., 165 deg
D = 0, 0.125, 0.25, 0.5, 1.0
anisotropy ratio = 1:1, 2:1, 4:1
```

For each candidate, render the patch through the twin and compute objective
values.

## Objectives to Compare

### Objective 1: Pose-Aware Sampling

This objective rewards motion that creates recoverable information when retinal
pose is known:

```text
U_PA(tau) = sum_t J(t; tau_t)^T Sigma_t^-1 J(t; tau_t)
```

It is expected to favor movements that produce strong phase-dependent changes.
It may favor gradient-axis motion or high-modulation directions.

### Objective 2: Pose-Blind Stability

This objective penalizes trajectories whose phase-dependent response variation
becomes nuisance covariance when pose is unknown:

```text
U_PB(tau) = Jbar^T (Sigma_noise + Cov_tau[mu(tau)])^-1 Jbar
```

This may favor smaller, more stable, or edge-parallel motions.

### Objective 3: Pareto Objective

The most biologically plausible family is:

```text
U_lambda(tau) = (1 - lambda) U_PA(tau)
              + lambda U_PB(tau)
              + beta U_whiten(tau)
              - gamma C_motor(tau)
```

Sweep `lambda`, rather than choosing one value in advance. If observed drift axes
are better predicted by intermediate or pose-blind-weighted objectives than by
pure pose-aware sampling, that supports the idea that FEMs balance sampling and
stability.

Do not try to rescue input whitening as a standalone objective by extending the
scale grid. Since the no-cost whitening optimum usually sat at the upper edge of
the tested grid, the useful next question is whether adding covariance, acuity,
temporal-sensitivity, and motor/fixation costs moves the preferred range back
toward biological scale.

### Objective 4: Adversarial Motion

For every objective and movement budget, identify the worst candidate
trajectory:

```text
tau_minus = argmin_tau U(tau)
```

The adversarial condition asks whether plausible eye movements with the same
scale and duration would be bad for the local patch. This shows that useful
motion is not arbitrary.

## Main Biological Tests

### Test 1: Does Model-Predicted Axis Match Real Drift Axis?

For each fixation window:

1. Extract the local image patch.
2. Compute observed drift or fixation-cloud axis.
3. Compute the model-predicted optimal axis under each objective.
4. Compute axis alignment, `cos(2 * (theta_real - theta_pred))`.
5. Compare against within-session shuffled patch-fixation pairings.

Primary result:

```text
Which objective best predicts real drift geometry?
```

### Test 2: Is Real Drift Closer to Optimized Than Adversarial?

Compare:

```text
cos(2 * (theta_real - theta_opt))
```

against:

```text
cos(2 * (theta_real - theta_adv))
```

If real drift is closer to optimized than adversarial, the objective has
biological relevance.

### Test 3: Does the Winning Objective Outperform Raw Image Orientation?

Compare model-derived predicted axes against:

- local edge axis;
- gradient axis;
- spectral axis.

Strong result:

```text
The twin-derived Pareto axis predicts real drift better than raw edge
orientation alone.
```

Weaker but useful result:

```text
Raw edge orientation predicts real drift as well as the twin, suggesting
image-geometry coupling but not a model-specific mechanism.
```

### Test 4: Does Preferred Motion Scale Depend on Image Structure?

For each patch, identify the scale `D*` that optimizes each objective. Then ask
whether `D*` depends on:

- contrast;
- high-spatial-frequency power;
- orientation coherence;
- spectrum anisotropy;
- model response modulation;
- pose-blind covariance cost.

This tests whether motion scale is locally content-dependent, even if measured
scalar FEM metrics are weakly predicted by raw image features.

## Rotated Vernier Bridge Experiment

The Vernier axis result should be clarified before being overinterpreted. Run
Vernier stimuli at:

```text
0, 45, 90, 135 deg
```

For each stimulus, compare motion axes:

- parallel to the Vernier edge axis;
- orthogonal to the Vernier edge axis;
- isotropic;
- reduced-scale real motion.

Key question:

```text
Does the preferred motion axis rotate with the stimulus?
```

If yes, this bridges Vernier and natural-image drift alignment. It would suggest
that the vertical-motion result was not a screen-axis artifact, but a
stimulus-geometry effect.

## Expected Outcomes

### Outcome A: Pareto Objective Predicts Edge-Parallel Drift

Best outcome:

```text
Real FEM geometry may reflect a compromise between extracting useful local
information and limiting pose-induced response variability.
```

This would align Vernier, compact geometry, and natural-image drift orientation.

### Outcome B: Pose-Aware Objective Predicts Gradient-Axis Motion, Real Drift Is Edge-Parallel

Interpretation:

```text
Real drift is not simply maximizing response modulation. The biological policy
likely includes stability, pose uncertainty, or downstream-readout constraints.
```

This rejects a naive active-sensing account while preserving the broader
coordinate-frame story.

### Outcome C: Raw Edge Axis Predicts Real Drift as Well as the Twin

Interpretation:

```text
The image-contingent coupling is real, but the current model objective does not
explain more than simple image geometry.
```

This is weaker but still biologically meaningful.

### Outcome D: No Model or Image Axis Predicts Real Drift

Interpretation:

```text
The current orientation alignment may be weak, confounded, or not captured by
these objectives.
```

This would argue against putting local image-contingent FEM tuning in the main
story.

## Practical Next Steps

1. Treat the scaled BackImage twin drift-geometry result as a negative for the
   current PA/PB/Pareto axis objectives in the audited 64-sampled-unit setting,
   not as a full canonical-population failure.
2. Keep the corrected coordinate order and `270 px` full-image patch margin in
   any follow-up.
3. Promote raw edge geometry to the baseline that any model objective must beat.
4. Build explicit edge-parallel versus edge-orthogonal candidate traces and ask
   whether raw image geometry and signed twin/pixel preservation predict the
   observed drift-axis bias.
5. Add rotated Vernier axis controls before interpreting Vernier motion-axis
   results as stimulus-geometry-specific.
6. Use the twin next for revised objectives, not blind repetition of the
   already-failed axis objectives: sliding along edges, minimizing retinal
   change, V1 temporal-band whitening, pose precision, or constrained
   stability. Run promising revised objectives in a larger sampled population or
   full canonical space before making V1-population claims.
7. Replicate the post-fix small-scale Gabor/pyramid pathfinder with stronger
   random baselines and scale-aware readouts before treating it as an optimizer
   target.
8. Only after an objective beats raw edge geometry should continuous
   optimized/adversarial trace generation become a main biological claim.

## Ultimate Claim Boundary

A successful result would not say:

```text
The real eye trace is optimal.
```

It would say:

```text
The V1 input-output geometry defines image-dependent motion axes and scales
that are useful or harmful for local sampling. Real fixational drift during
natural viewing is biased toward the useful side of this landscape, consistent
with an adaptive retinal-sampling role for FEMs.
```

That is the version of active sensing most consistent with the data so far.
