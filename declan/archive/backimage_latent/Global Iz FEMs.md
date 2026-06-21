# Coding note: aggregate FEM active-sensing analysis across natural-image statistics

## Goal

We want to step back from the exact per-fixation local-patch matching analyses and test a broader active-sensing hypothesis:

**Do empirical fixational eye movements improve the aggregate representation of natural-image information in the V1 digital twin, compared with matched non-biological motion controls?**

The current per-fixation `I_z` analyses have been useful, but they are probably not the right level for the main Figure 4 active-sensing claim. They ask whether each measured drift axis is optimal for its exact local patch, which creates a weak and heterogeneous signal. We now want to look for larger effects by pooling across many image samples and many FEM traces, asking whether the empirical FEM distribution is broadly matched to natural-image statistics.

The target is not:

```text
For fixation i and patch I_i, does the exact measured drift axis tau_i maximize I_z?
```

The target is:

```text
Across natural-image samples I ~ p(I), does the empirical FEM distribution q_real(tau)
produce a better V1 representation of image information than matched control motion
distributions q_control(tau)?
```

In other words, test whether FEMs are adapted to the ensemble statistics of natural images, not whether each individual fixation is locally optimal.

## Conceptual framing

Let `I` be a natural-image patch or crop, `tau` be a retinal trajectory, and `R = F_theta(I, tau)` be the V1 digital twin response.

Define an aggregate utility:

```text
U(q) = I(z(I); R(I, tau)),    I ~ p(I), tau ~ q(tau)
```

where `q(tau)` is a motion distribution and `z(I)` is an image latent, feature vector, or image identity surrogate.

We want to compare:

```text
q_real        empirical FEM traces
q_static      no motion
q_scaled      scaled empirical traces
q_brownian    matched diffusion/RMS control
q_ou          matched RMS/autocorrelation control
q_shuffle     time-shuffled or trace-shuffled empirical controls
q_phase       phase-randomized controls, if practical
q_large       over-large empirical scale, as a "more motion" diagnostic
```

A positive result should not just be “any motion beats static.” That would be too generic. The stronger result is:

```text
empirical FEMs lie near the top of an information/cost curve,
or outperform matched synthetic motion controls,
or occupy a Pareto-efficient regime of image information versus movement-induced nuisance.
```

## Why this analysis is needed

Previous work has left several gaps.

1. The local per-fixation `I_z` analyses are heterogeneous. Positive effects appear in some scales and feature families, especially small-scale Gabor/pyramid readouts, but they are not a clean global claim.
2. The old whitening analyses were not sufficient as representation tests. Some metrics rewarded larger temporal modulation or power spreading, which does not directly answer whether V1 responses better encode image structure.
3. Exact matching of each fixation trajectory to each local patch may be the wrong biological expectation. FEM statistics may be optimized at the distributional level, not per sample.
4. We need a figure-level result with large effects across many samples. Aggregate image representation is a better candidate than weak per-window axis matching.

## Primary analysis design

### Inputs

Use the canonical 756-unit V1 digital twin.

Use many natural-image patches or crops from BackImage/natural-image stimulus sources. Do not restrict the analysis to only the original fixation-patch pairing unless this is used as one optional condition. The main analysis should draw broadly from:

```text
image patches/crops: I_j
empirical FEM traces: tau_k
motion families: q_m
```

Each motion family should be applied across many image samples so that the output estimates an ensemble-level representation.

### Motion families

Implement the following, in priority order.

#### 1. Static

```text
tau(t) = 0
```

Baseline image representation without retinal motion.

#### 2. Empirical FEM

Use measured FEM traces, preferably centered within fixation windows. These can be sampled independently from images to estimate the aggregate effect of the empirical motion distribution.

Important: preserve realistic temporal sampling and duration.

#### 3. Scaled empirical FEM

Scale the same empirical traces:

```text
scale = 0, 0.125x, 0.25x, 0.5x, 1x, maybe 1.5x
```

Avoid or de-emphasize 2x unless it is explicitly treated as an over-large or clipped condition.

Record effective RMS and clipping fraction for every generated trajectory. Do not group only by nominal scale.

#### 4. Brownian matched control

Synthetic Brownian motion matched to empirical RMS or diffusion. This tests whether any diffusion-like motion is sufficient.

#### 5. OU matched control

Synthetic Ornstein-Uhlenbeck motion matched to empirical RMS, temporal autocorrelation, and/or confinement. This is a stricter fixation-like control.

#### 6. Shuffled empirical controls

At least one of:

```text
time-shuffled increments
trace-shuffled image pairing
rotated traces
phase-randomized trace with matched temporal spectrum
```

The goal is to preserve some low-level motion statistics while disrupting biological trajectory structure.

### Matching constraints

For each motion family, record:

```text
nominal scale
effective RMS
path length
velocity distribution
autocorrelation
fraction clipped
duration
number of traces
number of image samples
```

The main comparison must not be confounded by “more motion.” When possible, compare motion families at matched effective RMS or matched motion energy.

## Primary representation metrics

We want at least two complementary aggregate scores.

### Metric 1: image-feature decoding across the ensemble

Define image latent vectors `z(I)` using external transforms. Start with simple robust feature families:

```text
Gabor or steerable-pyramid coefficients
DCT coefficients
spatial-frequency band energies
possibly low/mid/high SF grouped features
```

For each image and motion condition, generate response summaries from the twin.

Possible response summaries:

```text
mean response over trajectory
time-concatenated response, if dimensionality manageable
response temporal PCs
delta response from static
multi-bin response features
```

Train cross-validated ridge decoders:

```text
z(I) ~ R(I, tau)
```

The key comparison is decoder performance across motion families:

```text
U_real - U_static
U_real - U_brownian
U_real - U_ou
U_real - U_scaled
```

Report feature-family-specific and spatial-frequency-specific results.

Important: do not let candidate-specific ridge hyperparameter selection create misleading advantages. Use one or more of:

```text
shared alpha per feature family and response summary
fixed alpha sensitivity
candidate-specific alpha as a secondary permissive estimate
```

Primary figure results should survive shared or fixed alpha.

### Metric 2: signal versus motion-nuisance covariance

This should tie the analysis back to the covariance story.

For each image `i` and motion sample `k`, compute a response vector:

```text
R_ik = response_summary(F_theta(I_i, tau_k))
```

Then estimate:

```text
mu_i = E_k[R_ik]
Sigma_signal = Cov_i(mu_i)
Sigma_motion = E_i[Cov_k(R_ik)]
```

Optionally include a baseline residual/noise covariance or diagonal Poisson-like noise model.

Compute an aggregate information-like score:

```text
Score = logdet(I + Sigma_noise^{-1} Sigma_signal)
```

where `Sigma_noise` can be one of:

```text
Sigma_motion + lambda I
diag(Sigma_motion) + lambda I
pooled residual covariance, if available
Poisson/mean-rate diagonal proxy
```

Also report simpler stable summaries:

```text
trace(Sigma_signal)
trace(Sigma_motion)
signal / motion ratio
top-k signal variance
top-k motion variance
subspace overlap between signal and motion
participation ratio of signal and motion
```

This metric asks whether FEMs improve image-driven structure more than they add nuisance variation.

### Metric 3: efficiency or Pareto score

For each motion family and scale, compute information versus cost:

```text
information score
effective RMS
path length
velocity
motion-nuisance covariance
```

Then ask whether empirical FEMs lie on or near a Pareto frontier.

This may be more biologically plausible than requiring empirical FEMs to maximize a single scalar score.

## Core comparisons

The main tables should include:

```text
real - static
real - Brownian matched
real - OU matched
real - time-shuffled
real - phase-randomized, if available
scaled empirical - real
large motion - real
```

Also report whether increasing empirical scale monotonically increases the metric. If the best score is always at the largest tested motion, then the metric is probably still rewarding generic modulation rather than biological utility.

## Important diagnostics

### 1. Effective-scale and clipping audit

Every summary grouped by nominal scale must also report:

```text
fraction clipped
median effective RMS
IQR effective RMS
effective / target RMS
```

If clipping is high, label that condition as capped rather than true scale.

### 2. Motion-energy matched controls

Before interpreting `real > control`, verify that controls are matched in effective RMS/path length or report differences explicitly.

### 3. Image-trace pairing

Run at least two pairing modes if practical:

```text
paired/original: original trace with original image patch
unpaired/ensemble: traces sampled independently from image patches
```

The main aggregate hypothesis can use unpaired ensemble sampling. If paired is stronger, that suggests local policy matching. If unpaired is strong, that supports distributional adaptation.

### 4. Feature scale breakdown

For feature decoding, report low/mid/high spatial-frequency feature groups separately. The most plausible active-sensing result may be concentrated in mid/high SF features.

### 5. Subsampling stability

Bootstrap or subsample over:

```text
image samples
trace samples
sessions
```

The output should include uncertainty intervals and leave-session-out or session-clustered bootstraps where feasible.

## Suggested script structure

Create a new script rather than overloading the per-fixation local latent screen.

Suggested name:

```text
declan/fixation_statistics_by_stimulus/run_backimage_aggregate_fem_information.py
```

Suggested companion posthoc:

```text
declan/fixation_statistics_by_stimulus/summarize_backimage_aggregate_fem_information.py
```

Suggested output folder:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
    backimage_aggregate_fem_information_n<...>_<date_or_tag>/
```

## Minimum viable run

If compute is limited, start with this reduced version:

```text
images: 256 or 512 patches
traces: 256 empirical traces
population: canonical 756 units
motion families:
    static
    real empirical
    empirical 0.25x
    empirical 0.5x
    empirical 1x
    Brownian matched RMS
    OU matched RMS
features:
    Gabor k=4 or grouped Gabor features
    pyramid k=8 or grouped pyramid features
response summary:
    mean response over trajectory
    delta from static
metrics:
    feature decoding R2
    signal/motion covariance ratio
controls:
    shared-alpha ridge
    clipping/effective-RMS report
```

This MVP should be enough to answer:

```text
Does the empirical FEM distribution improve aggregate representation of natural-image features compared with static and matched synthetic motion?
Does biological or sub-biological scale sit near a useful regime rather than the largest tested motion?
```

## Stronger production run

If the MVP lands, scale to:

```text
images: 1024 or more
traces: 512 or more
motion families:
    static
    empirical scales 0.125, 0.25, 0.5, 1
    Brownian matched
    OU matched
    shuffled empirical
    phase-randomized if cheap
features:
    Gabor grouped by orientation/SF
    steerable pyramid grouped by scale/orientation
    DCT grouped by frequency band
response summaries:
    mean
    temporal PCs
    delta from static
metrics:
    decoding R2
    linear-Gaussian information/logdet
    signal versus motion covariance decomposition
    information/cost Pareto frontier
```

## Decision criteria

### Strong positive

A strong Figure 4 result would be:

```text
empirical FEMs improve aggregate natural-image feature representation over static
and outperform matched Brownian/OU/shuffled controls at comparable motion energy,
especially for mid/high spatial-frequency features.
```

Even better:

```text
biological or sub-biological empirical scale lies near the peak or Pareto frontier,
while over-large motion adds nuisance or fails to improve representation efficiency.
```

### Moderate positive

A moderate but useful result would be:

```text
empirical FEMs beat static and lie on a sensible information/cost tradeoff,
but matched OU/Brownian controls are similar.
```

Interpretation:

```text
FEM-like motion statistics are useful, but the exact biological trajectory structure
is not uniquely required.
```

### Negative or diagnostic

If all motion families improve similarly and the largest motion always wins:

```text
the metric is probably measuring generic feature modulation, not active sensing.
```

If empirical FEMs are worse than matched controls:

```text
this specific image-feature objective is not the right aggregate utility.
```

## Figure 4 target if successful

A successful aggregate analysis would support a Figure 4 with this logic:

```text
A. Ensemble active-sensing schematic: natural images x FEM distribution -> V1 response movies.
B. Motion families: static, empirical FEM, matched Brownian/OU, scaled empirical.
C. Aggregate image-feature information across scale/family.
D. Signal versus motion-nuisance covariance decomposition.
E. Feature breakdown by spatial-frequency band or latent family.
```

Potential claim:

```text
Fixational eye movements improve the ensemble representation of natural-image structure in foveal V1, placing empirical drift near a useful information/cost regime rather than simply maximizing retinal motion.
```

## Guardrails

Do not claim real FEMs are globally optimal unless they beat matched motion controls and do not simply track motion amplitude.

Do not rely on nominal scale alone. Always report effective scale and clipping.

Do not make the result hinge on exact original fixation-patch pairing. The main hypothesis is aggregate distributional adaptation.

Do not use candidate-specific decoder regularization as the only figure-level estimate.

Do not overinterpret 2x or heavily clipped conditions.

Do not bury null controls. If Brownian/OU performs similarly, that is still informative: it suggests the useful factor is broad FEM-like motion statistics rather than exact biological trace structure.
