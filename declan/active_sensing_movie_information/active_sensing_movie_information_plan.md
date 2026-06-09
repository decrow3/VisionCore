# Active-Sensing Movie Information Plan

## One-Sentence Goal

Test whether real fixational eye-movement movies improve the efficiency with which the deterministic V1-model rate movie represents spatial information, relative to stabilized and matched counterfactual movies, and determine whether that efficiency gain is explained by retinal movie modulation, spectral power, higher-order image structure, or trajectory timing.

## Canonical Implementation

Build the analysis from Jake's existing production pipeline:

```text
jake/twininfo/
```

This is now the source of truth for trace selection, model-aligned movie
rendering, phase/pyramid controls, cumulative Fisher information, and cumulative
spatial single-spike information. The local
`declan/active_sensing_movie_information/run_active_sensing_movie_information.py`
runner should be treated as an exploratory prototype only.

Core files:

- `jake/twininfo/pipeline.py`
  - production entry point;
  - writes final per-movie summaries and time-resolved cumulative traces.
- `jake/twininfo/trace_selection.py`
  - selects fixation-only and one-microsaccade windows from real traces.
- `jake/twininfo/eye_controls.py`
  - operational robust-speed microsaccade detector.
- `jake/twininfo/retinal_examples.py`
  - model-aligned retinal rendering and lag-cube construction.
- `jake/twininfo/lagcube_information.py`
  - cumulative pattern Fisher and spatial SSI wrappers.
- `jake/twininfo/information.py`
  - Poisson Fisher and spatial single-spike information math.

First validation command:

```bash
.venv/bin/python -m jake.twininfo.pipeline \
  --run-name active_sensing_validation_small \
  --image-indices 24 29 30 \
  --n-crops-per-image 1 \
  --n-examples-per-kind 2 \
  --population-size 16 \
  --shift-grid-mode cross \
  --recompute
```

The main figure should be generated from `outputs/twininfo/<run_name>/`, not
from the temporary pilot output directory.

## Key Hypotheses

1. **Efficiency gain:** real FEM trajectories improve cumulative spatial
   information per expected spike relative to stabilized movies from the same
   image/trace pair.
2. **Time structure:** the efficiency gain emerges over movie time, rather than
   appearing only as a final endpoint fluctuation.
3. **Trajectory specificity:** a full active-sensing claim requires real FEMs to
   outperform the canonical matched-motion controls, not only stabilized
   controls.
4. **Event structure:** fixation-only and one-microsaccade windows have
   different efficiency profiles, consistent with drift and microsaccades
   playing different computational roles.
5. **Spectral mechanism:** the real-minus-control efficiency gain depends on
   spatial-frequency band in a way compatible with eye movements converting
   natural-image spatial spectra into temporal modulations.
6. **Residual phase structure:** intact-vs-phase effects are tested as
   real-minus-control interactions after spectral checks, not as the primary
   claim.
7. **Spike-count audit:** raw cumulative bits, bits/sec, expected spikes, and
   spike rate explain the result but do not define the primary claim.

## Why This Figure Is Separate From Tangent Geometry

The tangent-geometry analysis asks:

> Do image patches route finite translation-induced response change through a compact tangent basis?

The active-sensing movie analysis should ask:

> Do real FEM trajectories improve cumulative spatial information efficiency in the V1 model rate movie over time, and what image/trajectory properties explain that improvement?

The second question keeps the original movie/ecology mechanism intact while avoiding the boring result that larger motion simply drives more spikes. Raw response drive still matters as a diagnostic, because spectral power and motion-induced response gain may be part of the mechanism, but the load-bearing endpoint should be normalized by expected spike count.

## Primary Endpoint

The canonical first pass should be organized around one endpoint:

```text
paired real-vs-control cumulative spatial-information efficiency gain over matched image/trace movies
```

In the canonical `jake.twininfo` outputs, the first primary endpoint should be:

```text
final_cumulative_spatial_ssi_bits_per_spike
```

and its time-resolved counterpart:

```text
cumulative_spatial_ssi_bits_per_spike
```

Define the primary quantity explicitly:

```text
cumulative_bits(T) = sum_{t <= T} spatial_bits_t
expected_spikes(T) = sum_{t <= T} expected_spikes_t
cumulative_bits_per_spike(T) = cumulative_bits(T) / expected_spikes(T)
```

The phrase "bits/spike" should refer to this ratio. Avoid ambiguous phrases
such as "bits/spike per sec" unless a separate instantaneous rate quantity is
being defined.

For each image/trace pair, render real and control retinal movies from the same source image and the same trajectory seed whenever possible. Compute a cumulative information curve for each condition and summarize the paired gain at prespecified time windows:

```text
gain_efficiency(T) = bits_per_spike_real(image, trace, T) - bits_per_spike_control(image, trace, T)
relative_efficiency(T) = bits_per_spike_real(image, trace, T) / max(bits_per_spike_control(image, trace, T), eps)
```

For the bits/spike endpoint, the paired difference should be the primary statistic. Relative gain is a useful scale-normalized companion, but it should not replace the paired difference.

Raw cumulative spatial bits and bits/second should be reported as secondary
diagnostics, because they answer a different question: how much total
information is produced when response drive is allowed to vary.

Useful diagnostics:

```text
bits_per_sec(T) = cumulative_bits(T) / elapsed_time(T)
expected_spikes_per_sec(T) = expected_spikes(T) / elapsed_time(T)
instantaneous_bits_per_spike ~= dI / dN in a local/sliding window
```

Because the twin is deterministic, all model-information language should remain
rate-model language: "spatial information available from the deterministic
V1-model rate movie under the assumed readout model," not "biological V1 spike
trains encode more information."

All other analyses should explain this endpoint:

- pre-model retinal diagnostics ask what temporal modulations the eye trace creates;
- spectral and phase controls ask which image statistics support the gain;
- trajectory controls ask whether real FEM timing/statistics matter;
- drift/microsaccade decomposition asks which components of the trace contribute.

## Conceptual Sequence

The strongest version of the figure should follow this chain:

1. **Retinal movie transform:** real FEMs convert natural-image spatial spectra into temporal modulations.
2. **Efficiency gain:** real FEM movies improve cumulative spatial SSI bits/expected spike relative to stabilized controls.
3. **Time structure:** the gain emerges over time, not only as a final scalar.
4. **Trajectory specificity:** real FEM movies outperform the canonical matched-motion controls.
5. **Event structure:** fixation-only and one-microsaccade windows have different efficiency profiles.
6. **Spectral mechanism:** real-minus-control efficiency gain depends on spatial-frequency band in a Rucci-compatible way.
7. **Residual phase structure:** intact-vs-phase differences are interpreted only after spectral controls.
8. **Spike-count audit:** raw bits, bits/sec, expected spikes, and spike rate explain the efficiency result but do not define it.

## Primary Contrasts

### Trajectory Conditions

Use matched stimulus movies whenever possible.

- `real`: measured fixation trajectories in `jake.twininfo`.
- `stabilized`: retinal image held fixed at the fixation or trial-mean position.
- `random_amp`: measured step amplitudes with randomized directions.
- `random_cov`: Gaussian step sequence with measured step covariance.
- `trajectory_order_shuffle`: same sampled eye positions in shuffled temporal order.
- `pyramid_phase_scrambled`: local visual phase/pyramid control image rendered with the same trace.
- `sf_low`, `sf_mid_low`, `sf_mid_high`, `sf_high`: spatial-frequency control images in Jake's four-band decomposition.
- `fixation`: trace windows with no detected microsaccade.
- `microsaccade`: trace windows with exactly one detected microsaccade.

The first key comparison is `real` versus `stabilized`. Stabilized controls ask whether measured retinal motion improves spatial-information efficiency at all. A stronger active-sensing claim requires matched-motion controls:

```text
(real - stabilized) > (matched_motion - stabilized)
```

or equivalently:

```text
real > random_amp / random_cov / trajectory_order_shuffle
```

The canonical Jake pipeline now writes these matched-motion controls by default
and audits them in `metadata/03_trajectory_control_qc.csv`.

### Stimulus Conditions

- intact natural images through `real`;
- stabilized renderings of the same image/trace pair;
- local pyramid phase-scrambled controls;
- spatial-frequency band-limited reconstructions;
- optional radial-spectrum or trajectory-randomized controls after the canonical run is stable.

The first-pass claim should be spectral and temporal, not higher-order phase specific. Higher-order natural-image structure should become a residual question after spectral controls are credible.

For the direct Rucci-style image-content interaction, augment a standard run with
stabilized versions of the visual controls:

```text
stabilized_sf_low
stabilized_sf_high
stabilized_pyramid_phase_scrambled
```

These form direct paired contrasts within each visual-control family:

```text
sf_low - stabilized_sf_low
sf_high - stabilized_sf_high
pyramid_phase_scrambled - stabilized_pyramid_phase_scrambled
```

Incremental command:

```bash
conda run --no-capture-output -n yatesfv python -m jake.twininfo.pipeline \
  --run-name <existing_twininfo_run_name> \
  --augment-existing \
  --conditions stabilized_sf_low stabilized_sf_high stabilized_pyramid_phase_scrambled
```

`--augment-existing` reuses the existing `metadata/run_config.json`, skips rows
already present, computes only the missing stabilized visual controls, and
merges them into the standard summary CSV and cumulative-series NPZ.

Naming distinction:

- `trajectory_order_shuffle` is an eye-trajectory control. It preserves the
  sampled eye positions and shuffles their temporal order.
- `pyramid_phase_scrambled` is a visual stimulus control. It scrambles local
  image phase while rendering the movie with the same trajectory.
- Older outputs may contain `phase_order_shuffle`; interpret that legacy name as
  `trajectory_order_shuffle`, not as visual phase scrambling.

## Candidate Metrics

### 1. Pre-Model Retinal Movie Diagnostics

Before measuring model information, quantify what the trajectory does to the stimulus movie itself.

Candidate diagnostics:

```text
temporal_contrast_t = RMS(movie_t - movie_{t-1})
movie_power_t = RMS(movie_t)^2
motion_power_t = RMS(movie_t - stabilized_reference)^2
```

Frequency-domain diagnostics:

- temporal power spectrum of retinal movie pixels or local contrast;
- spatial-frequency-to-temporal-frequency conversion under real and control trajectories;
- band-specific temporal modulation for low/mid/high spatial-frequency image bands;
- event-triggered temporal modulation around microsaccades.

The retinal diagnostic should answer:

> Do real FEM traces create more or differently structured temporal modulation from natural images than stabilized or matched random traces?

This layer connects directly to the Rucci-style mechanism: eye movements transform spatial image structure into temporal input structure.

### 2. Cumulative Spatial SSI

Use Jake's `cumulative_spatial_ssi` / `spatial_single_spike_information` path on full convolutional rate maps to ask how much spatial information is available in the population response movie:

```text
SSI_T = spatial_ssi_population(responses up to time T)
```

Use cumulative bits per expected spike as the default paired endpoint:

```text
cumulative_bits_per_expected_spike(T) =
    sum_{t <= T} spatial_bits_t / sum_{t <= T} expected_spikes_t
```

Raw spatial bits, bits per second, and expected spikes should still be reported as companion diagnostics. The reason is important: raw information can increase monotonically with larger eye movements or higher firing rates. Bits per expected spike keeps the active-sensing claim from collapsing into "more motion drove more spikes."

Relevant output columns:

- final table: `metadata/05_lagcube_information_summary.csv`;
- primary final metric: `final_cumulative_spatial_ssi_bits_per_spike`;
- companion final metrics:
  - `final_cumulative_spatial_ssi_bits`;
  - `final_cumulative_spatial_ssi_bits_per_second`;
  - `final_cumulative_spatial_ssi_expected_spikes`;
  - Fisher pattern metrics.
- time-series file: `cache/cumulative_information_series.npz`.

### 3. Cumulative Fisher Information

For a scalar or low-dimensional stimulus parameter `theta`, estimate the model rate vector `lambda_t(theta)` over time and compute an accumulated Poisson Fisher trace:

```text
FI_T(theta) = sum_{t <= T} sum_i (d lambda_i,t / d theta)^2 / max(lambda_i,t, eps)
```

Pin down `theta` before using this as a primary metric. For natural images, displacement FI should be framed as retinal-pose sensitivity, not generic image information. Less circular first-pass choices may be spatial SSI for natural movies or identity separability for controlled stimuli.

Useful parameters include local retinal x/y pose, phase along a trajectory, or identity-relevant coordinates for controlled stimuli.

Report:

```text
gain_FI(T) = FI_real(T) - FI_control(T)
relative_gain_FI(T) = FI_real(T) / max(FI_control(T), eps)
```

The absolute gain is important because magnitude can be part of the mechanism. Relative gain is useful as a control-normalized summary but should not be the only metric.

### 4. Identity Separability Over Time

For optotypes or controlled image identities, compute cumulative separability:

```text
D_T(a, b) = ||mu_a,<=T - mu_b,<=T||^2
```

or a noise-normalized variant using diagonal Poisson variance. This tests whether a path through retinal positions provides complementary samples for identity readout.

### 5. Drift And Microsaccade Decomposition

Segment real traces into:

- fixation-only windows selected by `jake.twininfo.trace_selection`;
- one-microsaccade windows selected by `jake.twininfo.trace_selection`;
- event onset/offset samples from `metadata/01_trace_examples_used.csv`.

For the first pass, compare the final and time-resolved real-versus-stabilized gain by `kind == fixation` and `kind == microsaccade`. Only after that should we add finer peri-event decomposition, post-microsaccade settling windows, or drift-only masks inside microsaccade-containing traces.

## Proposed Main Figure

### Panel A: Retinal Movie Transform

Show how one natural image and one real eye trajectory become a retinal movie. Include temporal contrast or band-specific temporal modulation for real and stabilized renderings, plus phase or spatial-frequency image controls where useful. The goal is to make clear this is a time/movie question before it is a model-response question.

### Panel B: Example Real Versus Stabilized Movie Trace

For one image and trace, plot:

- eye position over time;
- retinal movie displacement;
- population response magnitude or low-dimensional response trajectory;
- cumulative information curve.

This panel should include drift and microsaccade annotations if event detection is stable.

### Panel C: Paired Cumulative Information Gain

Plot cumulative spatial SSI bits/expected spike over time for real FEM, stabilized, and matched-motion controls. The headline statistic should be the paired real-minus-control efficiency gain at prespecified time windows, with bootstrap over image identities and/or traces.

For the canonical first pass, use `cumulative_spatial_ssi_bits_per_spike` from
`cache/cumulative_information_series.npz`, grouped by `condition`, with final
paired statistics from `metadata/05_lagcube_information_summary.csv`.

### Panel D: Spectral Dependence

Ask whether the real-minus-control efficiency gain depends on spatial-frequency band:

```text
gain_efficiency_sf =
    bits_per_spike(real, sf_band)
    - bits_per_spike(control, sf_band)
```

The important result is not simply that one band has more bits/spike. The
mechanistic test is whether eye movements generate larger efficiency gains for
the spatial frequencies expected to be converted into useful temporal
modulations by measured drift and microsaccade scales.

### Panel E: Phase/Spectrum Controls

Compare intact natural images to spectrum-preserving phase-scrambled controls as an interaction:

```text
(real - control)_intact
vs
(real - control)_phase_scrambled
```

This should be labeled cautiously:

> phase controls test whether gains survive removal of natural spatial phase relationships, but may also perturb temporal consistency or model distribution depending on how movies are rendered.

Only if the intact gain is larger after spectral checks should the text discuss
higher-order natural-image structure.

### Panel F: Drift Versus Microsaccade Roles

Separate drift-only accumulation from microsaccade/repositioning windows. A useful outcome would distinguish:

- drift as local information accumulation;
- microsaccades as movement into new informative neighborhoods;
- motion energy without real trajectory timing as a weaker control.

## Supplemental Or Diagnostic Panels

- trajectory QC: displacement, velocity, and microsaccade distributions;
- stabilized and matched-control validation;
- image spectrum audit;
- response-magnitude audit;
- power/effect-size checks for image-level and trace-level bootstrap;
- tangent-geometry bridge panel only after the movie-information result is understood.

## Statistical Unit And Bootstrap

Primary uncertainty should respect both image and trajectory dependence.

Preferred hierarchy:

```text
image identity -> trajectory/fixation trace -> time window
```

Use image-level bootstrap for image-content claims, trace-level bootstrap for trajectory claims, and paired bootstrap for real-versus-control movie comparisons whenever each control is generated from the same image/trace seed.

Implementation detail for Jake outputs:

```text
pair id = image_index + crop_rank + example_id + kind
```

Use paired differences between `condition == real` and the control condition
within each pair. Bootstrap pairs for the main validation result; then split by
`kind` for fixation-only versus one-microsaccade summaries.

## Decision Criteria

The figure earns a main-result role if:

1. real movies show a reproducible positive paired cumulative spatial SSI bits/expected-spike gain over stabilized movies;
2. the effect is visible in time-resolved cumulative traces, not only in a noisy final scalar;
3. real movies outperform matched-motion controls;
4. fixation-only and one-microsaccade windows have interpretable efficiency profiles;
5. real-minus-control efficiency gain depends on spatial-frequency band in a Rucci-compatible way;
6. phase/pyramid controls are interpreted as residual interaction tests, not as primary evidence;
7. raw bits, bits/sec, expected spikes, and spike rate are audited so the primary result does not collapse into a spike-count effect.

If only item 1 is strong, the result is still useful but should be framed as a model diagnostic of motion-induced spatial-information efficiency, not a full active-sensing mechanism. Matched random-trajectory controls can then be added as the next disambiguating analysis.

## Naming

Use `active_sensing_movie_information` for this new figure family.

Keep `spatial_content_modulation` for the tangent-geometry tie-in figure.
