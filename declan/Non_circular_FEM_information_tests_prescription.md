# Non-Circular FEM Information Tests Prescription

## Goal

Build a compact, defensible analysis stack for the claim:

> FEMs are consistent with being tuned to improve cortical information during
> natural fixation.

This prescription deliberately avoids relying on a fitted digital twin for the
main optimality claim. The two load-bearing tests are:

1. an input-level whitening optimum computed from natural-image statistics and
   drift kinematics only;
2. a recorded-cortex pose-aware information test using actual V1 spike counts.

Two supporting analyses then make the benefit mechanistically specific:

3. spatial-frequency localization of the motion benefit;
4. sustained information accumulation over fixation.

None of these analyses proves optimization. The intended final language is:

> These results are consistent with FEMs being tuned to improve cortical
> information: biological drift lies near an input-whitening regime for natural
> images, recorded V1 information is more recoverable when retinal pose is
> accounted for, and the model benefit is concentrated in fine spatial structure
> and sustained accumulation over time.

## Repo Context To Reuse

Use existing `jake.twininfo` and recorded covariance infrastructure where
possible.

Relevant production model/movie code:

- `jake/twininfo/pipeline.py`
  - selected images, crops, trace examples, conditions, summaries;
  - writes `metadata/05_lagcube_information_summary.csv`;
  - writes `cache/cumulative_information_series.npz`.
- `jake/twininfo/retinal_examples.py`
  - exact retinal crop rendering;
  - `model_lag_cubes_from_image_trace`;
  - crop-center conventions.
- `jake/twininfo/retinal_movies.py`
  - stimulus movie helpers.
- `jake/twininfo/image_selection.py`
  - selected natural images and spatial-frequency controls.
- `jake/twininfo/trace_selection.py`
  - fixation and microsaccade trace windows.
- `declan/active_sensing_movie_information/summarize_figure5_additional_checks.py`
  - paired Figure 5 summaries.
- `declan/active_sensing_movie_information/generate_active_sensing_movie_information_figure.py`
  - final plotting conventions.

Relevant recorded-data/covariance code:

- `VisionCore/covariance.py`
  - covariance utilities, PSD projection, participation ratio.
- `scripts/figure_fixrsvp_mcfarland_covariance_declan.py`
- `scripts/figure_fixrsvp_mcfarland_covariance_declan2.py`
- `scripts/run_phase1_fem_covariance.py`
- `outputs/phase1_fem_covariance/`
- `outputs/active_sensing_movie_information/reafferent_variance_accounting/`

The exact recorded information implementation may need to locate the current
Fig 2/Fig 4 aligned spike-count cache. Do not recompute raw alignment unless the
cache is missing.

## Analysis 1: Input-Whitening Optimum

### Scientific question

Does biological drift amplitude sit near the movement scale that whitens natural
retinal input over a plausible V1-sensitive band?

This is the clean optimality-style test because the prediction comes from:

```text
natural image statistics + eye-motion kinematics
```

and does not use the fitted twin response model.

### Rationale

Natural images have approximately:

```text
P_spatial(f) ~ 1 / f^2
```

The current project already measured a slope near:

```text
radial spatial PSD slope ~= -2.01
```

Fixational drift turns spatial structure into temporal modulation at the retina.
As diffusion increases, low-dimensional spatial power is redistributed through
time. The efficient-coding prediction is that a biologically plausible drift
scale should flatten the retinal temporal spectrum across the band V1 can use.

### Primary output

For each motion scale, compute retinal temporal power and whitening metrics:

```text
loglog_temporal_psd_slope
abs_loglog_temporal_psd_slope
spectral_entropy
spectral_flatness
autocorrelation_time
```

Define:

```text
D_whiten_slope    = argmin abs(loglog_temporal_psd_slope)
D_whiten_entropy  = argmax spectral_entropy
D_whiten_flatness = argmax spectral_flatness
```

Overlay biological drift scale.

### Inputs

Use the same natural-image/crop/trace selections as the current Figure 5 run:

```text
outputs/twininfo/<figure5-run>/
metadata/run_config.json
metadata/01_trace_examples.csv or metadata/01_trace_examples_used.csv
metadata/02_image_crop_hotspots.csv
metadata/05_lagcube_information_summary.csv
```

If practical, also use all available natural-image crops, not only selected
high-energy crops, as a sensitivity check.

### Biological drift estimate

Estimate drift diffusion from fixation-only windows, excluding microsaccades.

For each trace:

```text
MSD(tau) = E_t[||e(t + tau) - e(t)||^2]
```

For 2D Brownian drift:

```text
MSD(tau) ~= 4 D_eye tau
```

Fit over short lags before confinement dominates. Suggested first-pass lag
range:

```text
tau = 8-80 ms
```

but write the code with a configurable lag range and report sensitivity.

Save:

```text
D_eye_deg2_per_s
D_eye_arcmin2_per_s
fit_lag_min_ms
fit_lag_max_ms
fit_r2
n_trace_windows
```

### Movie generation

Generate retinal movies without running the twin.

Motion families:

```text
stabilized
scaled_measured_drift_D
synthetic_brownian_D
synthetic_ou_D
```

Scale grid:

```text
D_scale = [0, 0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3]
```

For measured drift:

```text
e_D(t) = mean(e) + D_scale * (e(t) - mean(e))
```

For synthetic Brownian/O-U controls, match diffusion constants to:

```text
D_eye * D_scale^2
```

because scaling displacement by `D_scale` scales covariance/diffusion by
`D_scale^2`.

### Spectral computation

For each movie:

1. render retinal luminance or contrast frames;
2. subtract each pixel's temporal mean;
3. optionally apply a spatial passband before temporal analysis;
4. compute temporal PSD per pixel or over spatially averaged contrast energy;
5. average temporal PSD over valid pixels/crops/images.

Recommended first-pass temporal signal:

```text
contrast_movie = movie - temporal_mean(movie)
temporal_psd(f) = mean_pixels |FFT_t(contrast_movie)|^2
```

Use a window such as Hann before FFT and document it.

### Passband

The passband must be selected independently of the outcome.

Implement a configurable passband grid:

```text
spatial_passband_cpd:
  low:  [2, 4, 8]
  high: [20, 30, 40, 60]

temporal_passband_hz:
  low:  [0.5, 1, 2]
  high: [20, 30, 60]
```

Primary passband should be justified by one of:

- recorded RF size / spatial-frequency sensitivity;
- known V1 temporal sensitivity;
- existing Figure 5 spatial-frequency bands;
- a conservative literature-motivated band.

Do not choose the passband by maximizing the result.

### Outputs

Create:

```text
declan/active_sensing_movie_information/run_input_whitening_optimum.py
declan/active_sensing_movie_information/summarize_input_whitening_optimum.py
```

Write outputs to:

```text
outputs/active_sensing_movie_information/input_whitening/
```

Required files:

```text
input_whitening_manifest.json
whitening_movie_manifest.csv
drift_diffusion_estimates.csv
retinal_temporal_psd_by_movie.csv
whitening_scale_summary.csv
whitening_passband_sensitivity.csv
whitening_paired_bootstrap.csv
figures/whitening_scale_curves.pdf
figures/retinal_temporal_psd_examples.pdf
figures/whitening_passband_sensitivity.pdf
whitening_summary.md
```

### Acceptance criteria

- Stabilized movies have much lower temporal power except stimulus/rendering
  transients.
- `D_scale = 0` matches stabilized.
- Synthetic Brownian diffusion scales as expected.
- Temporal PSD estimates are stable across image/crop bootstrap.
- Biological `D_scale = 1` is explicitly compared to `D_whiten_*`.
- Passband sensitivity is reported even if the result weakens.

### Interpretation

Optimistic claim:

> Fixational drift amplitude sits near the value that decorrelates natural
> retinal input across a V1-sensitive band, an efficient-coding signature that
> FEMs are tuned to natural-scene statistics.

Caveat:

> Whitening is necessary but not sufficient for cortical utility, and the
> numerical optimum depends on the independently chosen passband.

## Analysis 2: Recorded-Cortex Pose-Aware Information

### Scientific question

In recorded V1, does accounting for measured eye position increase the stimulus
information recoverable from the population?

This is the direct cortex anchor. It says whether the eye-linked variability is
usable signal in actual V1 recordings, not only in the twin.

### Primary comparison

Compare two observers on held-out data:

```text
pose_blind:  p(y | stimulus)
pose_aware:  p(y | stimulus, eye_position_or_recent_eye_history)
```

or discriminative versions:

```text
pose_blind:  spikes -> stimulus
pose_aware:  spikes + eye covariates -> stimulus
```

The cleanest interpretation is from a generative or cross-validated likelihood
comparison, but balanced decoding accuracy is acceptable as a first pass.

### Guardrail

Do not define pose-aware information as "remove eye-position effects and decode
the residual." That can throw away useful reafferent signal.

The right question is:

> Are the same spikes more informative about the stimulus when the observer also
> knows retinal pose?

### Stimulus labels

Choose the richest stimulus label with enough repeats.

Candidate labels:

```text
image identity
stimulus frame/time bin
coarse stimulus segment
spatial-frequency condition, if available
```

Start with the label that gives stable cross-validation by session. If repeated
natural-image identity is too sparse, use coarse time/frame labels or binary
high/low image-content labels.

### Eye covariates

Use measured eye state at the relevant neural latency.

Candidate covariates:

```text
x(t-lag), y(t-lag)
recent displacement dx, dy
speed
history PCA over the last 25-80 ms
```

Start simple:

```text
x, y, dx, dy
```

with one fixed latency from the existing covariance analysis.

### First-pass decoders

Implement paired cross-validated decoders:

1. Blind linear decoder:

```text
X = spike counts
y = stimulus label
```

2. Pose-aware linear decoder:

```text
X = spike counts plus eye covariates
y = stimulus label
```

3. Better generative version if feasible:

```text
blind:  Poisson/negative-binomial GLM with stimulus terms
aware:  same plus eye-state terms and/or stimulus x eye low-rank terms
metric: held-out log likelihood, converted to bits/spike
```

The generative version is more faithful to "information recoverable given
pose," but the discriminative version is faster and easier to sanity-check.

### Cross-validation

Use splits that prevent leakage:

- split by trial, not by adjacent time bins;
- keep paired blind/aware train/test splits identical;
- report session-level metrics before pooling;
- use subject/session bootstrap for aggregate CIs.

### Metrics

For each session:

```text
balanced_accuracy
confusion_MI_bits
heldout_log_likelihood
bits_per_spike
expected_spikes
aware_minus_blind
```

If using classifiers, compute confusion-matrix mutual information as a bounded
and interpretable support metric.

### Controls

Required:

- eye covariates shuffled across trials within stimulus/time labels;
- pose-aware model with eye covariates but no spikes, to quantify pure pose
  leakage about stimulus labels;
- matched model complexity control or nested cross-validation if using richer
  pose-aware models.

Optional:

- pose-blind decoder trained on FEM-corrected residuals, clearly labeled as a
  diagnostic rather than the primary comparison;
- lag sweep around the expected visual latency.

### Outputs

Create:

```text
declan/active_sensing_movie_information/run_recorded_pose_information.py
declan/active_sensing_movie_information/summarize_recorded_pose_information.py
```

Write outputs to:

```text
outputs/active_sensing_movie_information/recorded_pose_information/
```

Required files:

```text
recorded_pose_info_manifest.json
recorded_pose_info_session_metrics.csv
recorded_pose_info_unit_counts.csv
recorded_pose_info_paired_contrasts.csv
recorded_pose_info_decoder_qc.csv
recorded_pose_info_shuffle_controls.csv
figures/recorded_pose_info_session_pairs.pdf
figures/recorded_pose_info_by_subject.pdf
recorded_pose_info_summary.md
```

### Acceptance criteria

- Pose-aware and pose-blind use identical train/test splits.
- Pose-aware improvement survives eye-shuffle controls.
- Eye-only decoder is not sufficient to explain the result.
- Results are reported at session and subject level.
- Decoder labels and spike windows match the existing covariance analysis
  windows or explicitly document deviations.

### Interpretation

Optimistic claim:

> In recorded V1, accounting for self-generated retinal motion increased the
> stimulus information recoverable from the population, so eye-linked
> variability is usable pose-conditioned signal rather than only nuisance noise.

Caveat:

> This shows the information is recoverable given pose; it does not show that
> downstream areas actually recover pose.

## Analysis 3: Spatial-Frequency Localization

### Scientific question

Is the motion benefit concentrated where the static code is resolution-limited?

This reframes the existing Figure 5 spatial-frequency result as mechanistic
specificity rather than just another control.

### Inputs

Use the production `jake.twininfo` outputs:

```text
metadata/05_lagcube_information_summary.csv
cache/cumulative_information_series.npz
```

Conditions:

```text
sf_low
sf_mid_low
sf_mid_high
sf_high
stabilized_sf_low
stabilized_sf_mid_low
stabilized_sf_mid_high
stabilized_sf_high
```

If stabilized counterparts are missing, augment the existing run first using
the documented `jake.twininfo.pipeline --augment-existing` path.

### Metrics

Primary:

```text
Delta_E_sf = final_cumulative_spatial_ssi_bits_per_spike(condition)
             - final_cumulative_spatial_ssi_bits_per_spike(stabilized_condition)
```

Companions:

```text
raw cumulative bits
expected spikes
bits/second
temporal modulation / retinal movie transform metrics
```

### Outputs

Add or reuse:

```text
outputs/active_sensing_movie_information/active_sensing_movie_information_figure/
```

Required summary:

```text
sf_localization_summary.csv
sf_localization_paired_contrasts.csv
figures/sf_localization_gain.pdf
sf_localization_summary.md
```

### Interpretation

Optimistic claim:

> The benefit appears where the static code fails, at fine spatial detail below
> instantaneous resolution, consistent with motion converting sub-RF structure
> into a temporal signal.

Why this matters:

```text
generic "any variance helps" -> SF-flat benefit
hyperacuity/spectral-temporal mechanism -> high-SF-localized benefit
```

Caveat:

> Phase-scrambled gains similar to intact gains bound the result to spectral
> content, not natural phase structure.

## Analysis 4: Information Accumulation Slope

### Scientific question

Does self-motion sustain information accumulation across fixation while
stabilized input saturates earlier?

This reframes the existing cumulative Figure 5 curves in terms of temporal
dynamics.

### Inputs

Use:

```text
outputs/twininfo/<figure5-run>/cache/cumulative_information_series.npz
outputs/twininfo/<figure5-run>/metadata/05_information_series_records.csv
```

Primary metric:

```text
cumulative_spatial_ssi_bits_per_spike
```

Companions:

```text
cumulative_spatial_ssi_bits
cumulative_expected_spikes
cumulative_fisher_pattern
```

### Metrics

For each paired real/stabilized movie:

```text
early_slope = slope over first 25% of fixation
mid_slope = slope over middle 50%
late_slope = slope over last 25%
late_minus_early_slope
time_to_50pct_final_information
time_to_80pct_final_information
real_minus_stabilized cumulative gain at each time
area_under_gain_curve
```

Use robust linear fits or simple endpoint differences over predefined windows.

### Outputs

Create:

```text
declan/active_sensing_movie_information/summarize_information_accumulation.py
```

Write:

```text
outputs/active_sensing_movie_information/information_accumulation/
accumulation_slope_metrics.csv
accumulation_paired_contrasts.csv
accumulation_time_to_threshold.csv
figures/accumulation_slope_pairs.pdf
figures/accumulation_gain_over_time.pdf
accumulation_summary.md
```

### Interpretation

Optimistic claim:

> Self-motion sustains information accumulation across the fixation while the
> stabilized code saturates earlier, consistent with motion refreshing the
> cortical representation rather than rereading a static image.

Caveat:

> This remains a pose-aware upper-bound metric, although the near-zero
> conditional correlations in the recorded covariance analysis justify it as a
> useful upper-bound comparison.

## Stacked Interpretation

If all four analyses land, use a stacked claim:

> The recorded data show that retinal pose makes stimulus information more
> recoverable in V1. Independently, natural-image statistics predict a
> whitening-favorable drift scale near biological fixation. The model-side
> information gain is concentrated at fine spatial frequencies and accumulates
> over fixation time. Together, these results support the view that FEMs are
> tuned to improve cortical information without requiring a claim of proven
> global optimality.

Do not say:

> FEMs are optimal.

Safer language:

```text
consistent with efficient coding
near an input-statistics whitening regime
tuned to natural-scene statistics
recoverable given retinal pose
mechanistically concentrated in high spatial frequencies
sustains accumulation over time
```

## Minimum Deliverable For A Coding Agent

The first useful implementation pass should do the following in order:

1. Build `run_input_whitening_optimum.py` and run a smoke test on the existing
   Figure 5 images/traces.
2. Write `whitening_summary.md` with the biological drift estimate, whitening
   scale, and passband sensitivity.
3. Build `summarize_information_accumulation.py` from existing
   `cumulative_information_series.npz`.
4. Build `sf_localization_summary.csv` from existing Figure 5 summary rows.
5. Locate the recorded spike/eye aligned cache and draft
   `run_recorded_pose_information.py`; if the cache is not obvious, write a
   short cache inventory before implementing the decoder.

Do not spend GPU time on the twin before the whitening and existing-output
summaries have run.

