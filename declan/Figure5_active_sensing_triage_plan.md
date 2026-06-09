# Figure 5 active-sensing triage plan

Working plan

## Core Goal

Rescue the headline:

> Active retinal sampling explains shared variability in foveal V1.

This does **not** require proving that real FEM trajectories are globally optimal. It requires showing:

1. measured retinal sampling explains shared variability;
2. the same retinal sampling has a plausible visual function;
3. that function is tied to real movement statistics, image statistics, or visual-channel structure rather than being an arbitrary model artifact.

Figures 1-4 mostly address item 1. Figure 5 must address items 2-3.

## Central Reframe

The question is not:

> Do real FEMs beat every synthetic trajectory?

The better question is:

> How do FEM statistics transform natural-image structure into V1-model information, and are drift/microsaccade regimes matched to different visual sampling functions?

This lets us incorporate three active-sensing mechanisms:

1. **Spectral-temporal conversion:** FEMs convert spatial frequencies into temporal modulations.
2. **Scale/channel matching:** drift and microsaccades occupy different displacement/speed regimes that may recruit parvo-like and magno-like channels differently.
3. **Image-contingent sampling:** local eye-movement statistics may depend on image structure where the animal is looking.

## Tier 0: Required Audits Before Interpreting Figure 5

These are non-negotiable. Without them, Figure 5 is too easy to dismiss.

### 0.1 Metric and spike-count audit

Status: first pass completed.

Existing outputs:

```text
active_sensing_movie_information_figure/active_sensing_spike_count_audit.pdf
active_sensing_movie_information_figure/spike_count_audit_paired_deltas.csv
active_sensing_movie_information_figure/spike_count_audit_condition_summary.csv
```

Current readout:

```text
intact real - stabilized
bits/expected spike = +0.035
cumulative bits     = +0.119
bits/sec            = +0.112
expected spikes     = +1445
spikes/sec          = +1365
```

Interpretation:

> The efficiency gain survives normalization by expected spike count, but expected spikes also increase. The claim should be "efficiency gain survives spike normalization," not "there is no drive effect."

For every movie condition, output:

```text
cumulative_spatial_ssi_bits
final_cumulative_spatial_ssi_bits_per_spike
expected_spikes
cumulative_spatial_ssi_bits_per_sec
expected_spikes_per_sec
movie_duration
n_valid_frames
```

Required checks:

- `bits_per_spike = bits / expected_spikes`.
- Paired real/control movies share image, trace, seed, duration, and valid-frame mask.
- Real-vs-stabilized gain survives in bits per expected spike, not only raw bits.

Decision:

- If bits/spike gain fails, Figure 5 becomes a motion-drive diagnostic, not an active-sensing result.
- If bits/spike gain survives, continue.

### 0.2 Matched-motion control validation

Status: not yet complete; this is now the next critical task.

Current issue:

```text
real       = 0.145 bits / expected spike
random_amp = 0.164
random_cov = 0.162
```

The current controls show that motion-like sampling can exceed real FEMs, but this is not yet interpretable because `random_amp` preserves step scale/path-length-like structure without necessarily matching occupancy cloud, fixation mean, sampled local image structure, or displacement covariance tightly enough.

For each control (`random_amp`, `random_cov`, step shuffle, position shuffle, phase-randomized trace), quantify:

- path length;
- step-size distribution;
- displacement covariance;
- occupancy radius;
- trajectory temporal power spectrum;
- temporal autocorrelation;
- number of valid rendered frames;
- local image gradient/highpass energy sampled along the path.

Decision:

- If random controls beat real because they move more or sample more high-frequency structure, they are not fair controls.
- If validated random controls still beat real, do not claim trajectory optimality.

Add a new control:

```text
random_amp_cloud_matched
```

Goal:

- preserve step-amplitude distribution/path length;
- match RMS displacement or position covariance;
- match mean fixation position;
- reject/resample paths until occupancy statistics fall within tolerance;
- use the same image/trace seed and valid-frame mask where possible.

This control determines whether the current `random_amp` advantage is a true generic-motion effect or a control-construction artifact.

### 0.3 Direct time-resolved gain curves

Plot:

```text
Delta E_real-stabilized(T)
Delta E_real-random_amp(T)
Delta E_real-random_cov(T)
```

where:

```text
E(T) = cumulative spatial SSI bits(T) / expected spikes(T)
```

Decision:

- If the endpoint is driven by a transient artifact or frame-count difference, demote.
- If the gain accumulates coherently, keep as main Figure 5 evidence.

## Tier 1: Must-Run Mechanism For Figure 5

These analyses are the best route to a defensible active-sensing claim.

### 1.1 Retinal movie transform QC

Status: first pass completed.

Existing outputs:

```text
active_sensing_movie_information_figure/retinal_movie_transform_qc.pdf
active_sensing_movie_information_figure/retinal_movie_transform_qc.csv
active_sensing_movie_information_figure/retinal_movie_transform_qc_summary.csv
```

Current readout:

- real and `random_amp` generate similar retinal temporal contrast;
- lowpass produces much weaker temporal modulation than highpass/intact;
- highpass has a larger FEM-minus-stabilized bits/spike gain than lowpass.

This already supports a spectral-temporal mechanism, but the next pass should directly relate movie-transform metrics to model bits/spike gain across matched movies.

Before model responses, quantify how real and control trajectories transform the image.

Compute:

- temporal contrast generated by each trajectory;
- temporal power spectrum of the retinal movie;
- spatial-frequency-to-temporal-frequency conversion;
- local highpass/gradient energy sampled along the path;
- drift versus microsaccade movie statistics.

Key chain:

```text
spatial image content
-> eye movement trajectory
-> retinal temporal modulation
-> V1-model bits/spike gain
```

Success:

> Real or FEM-like motion converts high spatial frequencies into temporal modulations that predict model information-efficiency gain.

### 1.2 Spatial-frequency mechanism

Status: partially complete and currently the strongest Figure 5 mechanism.

Current result:

```text
lowpass  real - stabilized = +0.010 bits/expected spike
highpass real - stabilized = +0.051 bits/expected spike
intact   real - stabilized = +0.035 bits/expected spike
phase    real - stabilized = +0.033 bits/expected spike
```

Expand from binary low/high into a graded SF analysis:

- `sf_low`
- `sf_mid_low`
- `sf_mid_high`
- `sf_high`
- lowpass/highpass if already clean
- intact natural images

For each:

```text
Delta E_real-stabilized(sf, T)
retinal temporal modulation(sf, T)
expected spikes(sf, T)
```

Success:

> The FEM efficiency gain is strongest for spatial-frequency content that FEMs transform into useful temporal modulation.

This is probably the most Rucci-aligned mechanism.

Immediate next step:

Add stabilized counterparts for intermediate spatial-frequency controls:

```text
sf_mid_low
sf_mid_high
```

so the mechanism is a graded spectral profile rather than only a binary lowpass/highpass result.

### 1.3 Displacement-scale tuning with empirical FEM bands

Use imposed displacement sweeps and/or existing tangent/information sweeps to ask:

```text
information efficiency gain as a function of displacement scale
```

Overlay empirical:

- drift displacement band;
- microsaccade displacement band.

Important controls:

- intact natural images;
- lowpass/highpass;
- phase-scrambled if cheap;
- random subspace/tangent diagnostic only as supplement.

Success:

> Model information efficiency or local translation sensitivity peaks or shoulders near empirical drift-scale displacements, especially for high spatial-frequency content.

Safe wording:

> Drift-scale displacements overlap the model's most informative local sampling regime.

Avoid:

> Drift is optimized.

## Tier 2: High-Reward Active-Sensing Upgrades

These are not required for the basic Figure 5 claim, but they can make the result feel genuinely active rather than merely motion-driven.

### 2.1 Image-contingent FEM statistics

Question:

> Do FEM statistics change depending on local image structure where the animal is looking?

Compute local image statistics along each real path:

- RMS contrast;
- gradient magnitude;
- high-frequency power;
- edge density;
- orientation coherence;
- phase congruency / multiscale structure if available;
- model tangent norm;
- local model information potential.

Relate these to FEM statistics:

- drift speed;
- drift direction;
- drift direction relative to local edge orientation;
- path curvature;
- microsaccade probability;
- microsaccade amplitude;
- microsaccade direction;
- post-microsaccade landing structure.

Nulls:

- same eye trace on shuffled image identities;
- same image with temporally shuffled eye trace;
- rotated/reflected traces;
- random paths matched for amplitude/covariance;
- fixation locations sampled from marginal eye-position distribution.

Success:

> Real FEM statistics are coupled to local image statistics, and this coupling predicts model information-efficiency gain.

This would make the paper much more actively "active."

### 2.2 Drift versus microsaccade division of labor

Current event split:

```text
fixation real - stabilized     = +0.028
microsaccade real - stabilized = +0.042
```

Expand into:

- pre-microsaccade;
- microsaccade;
- post-microsaccade settling;
- following drift segment.

For each window:

```text
Delta E_real-stabilized
raw bits
expected spikes
bits/spike
local highpass energy
local gradient energy
retinal temporal contrast
```

Test stronger repositioning claim:

- Do microsaccades land in higher local structure?
- Is post-microsaccade drift information rate higher?
- Does a microsaccade reset local redundancy or move to a new tangent neighborhood?

Success:

> Drift and microsaccades contribute differently: drift supports local accumulation; microsaccades reposition into new informative neighborhoods.

Safe fallback:

> Drift-only and microsaccade-containing windows differ in model information efficiency.

### 2.3 Parvo/magno-style channel stratification

Question:

> Are FEM displacement/speed regimes matched to different early visual channels?

Use model/unit proxies:

- RF/readout size;
- spatial-frequency preference;
- temporal-filter preference;
- sustained versus transient response index;
- highpass versus lowpass sensitivity;
- small-RF/high-SF units as parvo-like;
- large-RF/transient/low-SF units as magno-like.

Analyses:

- drift-scale information gain by channel class;
- microsaccade-scale response/information by channel class;
- temporal modulation spectrum by channel class;
- expected spikes and bits/spike by channel.

Success:

> Drift preferentially recruits high-SF/sustained/small-RF channels, while microsaccades preferentially recruit transient/low-SF/large-RF channels.

This would be a strong general sensing result even without real trajectory optimality.

## Tier 3: Conditional / Supplemental Analyses

These are useful but should not drive the main paper unless they land cleanly.

### 3.1 Phase and pyramid controls

Current result:

```text
intact gain          ~= +0.035
phase-scrambled gain ~= +0.033
```

Interpretation:

> Phase scrambling does not abolish the FEM-vs-stabilized gain.

Keep in supplement unless spectral matching reveals a clean residual.

Useful reporting:

- absolute information reduction under phase scramble;
- intact-minus-phase interaction;
- spectrum-matched residual controls.

### 3.2 Real-FEM trajectory optimality

Only claim if validated controls are beaten.

Current result:

```text
real       = 0.145
random_amp = 0.164
random_cov = 0.162
```

Current interpretation:

> Real trajectory optimality is not supported.

Do not make this a central success/failure condition.

### 3.3 Tangent-subspace recruitment

Useful as a bridge to Figure 4, not as the main Figure 5 mechanism.

Keep if it helps show:

- drift-scale displacement overlaps local tangent/information peak;
- high-SF content recruits translation geometry;
- compact reafferent modes carry part of the information gain.

Do not use as the primary active-sensing endpoint.

## Revised Figure 5 Layout

### Panel A: Retinal motion regimes and movie transform

Show:

- empirical drift/microsaccade displacement and speed distributions;
- retinal movie temporal modulation from real motion;
- optional spatial-frequency-to-temporal-frequency schematic.

Purpose:

> FEMs create distinct retinal movie regimes.

### Panel B: Time-resolved information-efficiency gain

Show:

```text
Delta E_real-stabilized(T)
```

with endpoint:

```text
+0.035 bits / expected spike
```

Purpose:

> Real retinal motion improves model information efficiency over stabilization.

### Panel C: Spike-count audit

Show:

- raw bits;
- expected spikes;
- bits/spike;
- maybe bits/sec and spikes/sec in supplement/table.

Purpose:

> The primary result is not simply more expected spikes.

### Panel D: Matched-motion controls

Show:

- stabilized;
- real;
- random_amp;
- random_cov;
- step/position/phase-shuffle if available.

Frame as:

> Boundary condition on trajectory specificity.

Do not frame as failed optimality unless controls are validated and the question is explicitly optimality.

### Panel E: Spatial-frequency mechanism

Show:

```text
Delta E_real-stabilized(sf band)
```

plus retinal temporal modulation by SF if possible.

Purpose:

> The effect is spectral-temporal and strongest for high spatial frequencies.

### Panel F: Drift/microsaccade and image-contingent sampling

Show either:

Option 1, if event analysis lands:

- drift;
- microsaccade;
- post-microsaccade;
- local image structure / information rate.

Option 2, if image-contingent sampling lands:

- local image statistic predicts FEM statistic;
- sampled local information potential exceeds matched null.

Purpose:

> FEM statistics are not arbitrary; they interact with visual scene structure and/or visual-channel regimes.

## Decision Tree

### Minimum Figure 5 success

Requirements:

- real > stabilized in bits/spike;
- spike audit passes;
- time-resolved delta is coherent;
- spatial-frequency dependence remains.

Claim:

> Real retinal motion improves V1-model information efficiency through a spectral-temporal mechanism.

### Strong active-retinal-sampling success

Additional requirements:

- retinal movie transform predicts model gain;
- empirical drift band overlaps displacement-scale information peak;
- drift/microsaccade regimes show distinct information profiles.

Claim:

> FEM statistics are matched to the sensory consequences of retinal sampling.

### Very strong active-sensing success

Additional requirements:

- FEM statistics depend on local image structure;
- real sampling paths sample higher local information potential than matched nulls;
- validated matched-motion controls cannot fully reproduce the image-contingent effect.

Claim:

> Real fixational eye movements implement image-contingent active retinal sampling.

## Immediate Execution Order

1. **Direct delta curves** for real-stabilized, real-random_amp, real-random_cov, and highpass/lowpass FEM-stabilized.
2. **Matched-motion validation** for current random controls, especially occupancy radius and sampled local gradient/highpass energy.
3. **Add `random_amp_cloud_matched`** and rerun Panel D controls.
4. **Spatial-frequency expansion** with stabilized `sf_mid_low` and `sf_mid_high` counterparts.
5. **Retinal movie transform-to-model gain regression**, linking SF-to-temporal modulation to bits/spike gain.
6. **Displacement-scale tuning** with empirical drift/microsaccade bands.
7. **Event-context expansion** around microsaccades.
8. **Image-contingent FEM statistics**.
9. **Channel stratification** into parvo/magno-like proxies.
10. Phase/pyramid residual controls only after the spectral story is stable.

## Bottom Line

The active-sensing paper does not need to prove real FEM trajectory optimality. It needs to show that measured retinal sampling explains shared variability and that the same sampling has a lawful sensory function.

The best current path is:

```text
retinal motion explains shared variability
+
retinal motion improves bits/spike over stabilization
concentrated in high spatial frequencies
+
FEM movement regimes align with displacement/channel/image statistics
```

That is enough to support:

> Active retinal sampling explains shared variability in foveal V1.
