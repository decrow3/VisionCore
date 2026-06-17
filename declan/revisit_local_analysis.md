# Local BackImage `I_z` Revisit After Aggregate FEM Result

Last curated: 2026-06-17

## Purpose

Revisit the local per-image/per-trace BackImage `I_z` analysis, but with a sharper question than the original local-axis screen.

The aggregate BackImage analysis now supports a distribution-level claim:

```text
Across many natural-image patches, empirical drift-like trajectory statistics add feature-decodable temporal structure beyond the static V1-twin response and beat OU-like confined controls.
```

That result does not require each measured trace to be optimally paired with its exact image patch. The local revisit should therefore ask a stronger but narrower follow-up:

```text
Does the actual image-trace pairing provide additional local benefit beyond the aggregate usefulness of empirical drift statistics?
```

The goal is not to prove that every real trace is the exact local optimum. The goal is to test whether real image-trace pairings are better than matched unpaired empirical traces, rotated traces, and synthetic controls, especially in image regimes where the local geometry predicts a benefit.

## Core Question

For each BackImage fixation window `i`, with image patch `I_i`, measured drift trace `tau_i`, and image feature latent `z_i = phi(I_i)`:

```text
Does F_theta(I_i, tau_i) provide more feature-decodable information about z_i
than F_theta(I_i, tau_j), where tau_j is an empirical trace from another matched fixation?
```

Primary contrast:

```text
actual paired empirical trace - matched unpaired empirical trace
```

Secondary contrasts:

```text
actual paired empirical trace - rotated actual trace
actual paired empirical trace - OU matched trace
actual paired empirical trace - Brownian matched trace
actual paired empirical trace - raw edge axis trace
actual paired empirical trace - edge-orthogonal trace
```

The local-pairing test is only meaningful because the aggregate distributional result has already shown that empirical traces are useful in general. The local test asks whether there is image-specific matching beyond that.

## Conceptual Distinction

Do not confuse these two claims:

### Aggregate distributional claim

```text
I ~ p(I), tau ~ q_empirical(tau)
```

Empirical motion statistics are useful across natural images.

### Local pairing claim

```text
I_i paired with its own tau_i
```

The actual trace used during fixation `i` is better matched to the actual local image patch than comparable empirical traces drawn from other windows.

The local claim is stronger and may be weaker or absent. A null result does not undermine the aggregate result. It only says the useful object is the empirical distribution, not exact image-trace pairing.

## Existing Evidence and Guardrails

### What is already positive

1. Aggregate n=256/K=4 drift-only run:

   * empirical drift adds feature-decodable signal beyond static;
   * empirical beats OU across scale;
   * Brownian/generic motion narrows the advantage at larger scales;
   * effect is scale-, readout-, and twin-scoped, not global optimality.

2. Local n=256 Gabor/pyramid screen:

   * stable real-vs-random positives at small scale, strongest near `0.25x`;
   * Gabor k=4 and pyramid k=8 are the main local feature candidates;
   * 1x is alive but guarded;
   * large nominal-scale results are sensitive to clipping/effective RMS.

3. Local geometry/stability:

   * observed drift is edge-aligned;
   * edge-parallel motion preserves pixels and V1-twin responses better than edge-orthogonal motion;
   * raw edge is the baseline that any local objective must beat.

### Guardrails

Do not claim:

```text
real traces are globally optimal
actual trace pairing explains all drift behavior
local I_z is the main figure-level active-sensing proof
```

unless the paired-vs-unpaired and paired-vs-rotated tests land cleanly.

Do claim, if supported:

```text
the aggregate usefulness of empirical drift statistics is complemented by local image-trace matching in specific image regimes
```

or, if local pairing is null:

```text
empirical drift statistics are useful distributionally, while local image-specific trajectory matching remains weak or unresolved
```

## Primary Analysis Design

### Unit space

Use:

```text
canonical 756-unit V1 twin
```

Do not use 16-channel matched space for discovery. If compute requires a smaller population, label it as smoke/pathfinder only.

### Image/window manifest

Use a fixed manifest for all local comparisons.

Preferred:

```text
same n=256 BackImage manifest used in the latest local and aggregate analyses
```

Manifest must include:

```text
session
source_row / window_id
image_id
time/fixation window
duration
observed RMS
drift anisotropy
image orientation coherence
edge density/coherence
real drift axis
raw edge axis
raw spectrum axis
edge-parallel stability metrics if available
```

Use `--window-manifest` style replay. Do not resample windows when changing controls.

### Trace bank

Use Jake-defined drift-only traces.

Required detector/filters:

```text
--max-trace-source-microsaccade-events 0
--max-trace-source-rms-deg
--max-trace-source-radius-deg
--max-trace-source-path-length-deg
--max-trace-source-speed-p95-deg-s
```

Record for every source trace:

```text
microsaccade_threshold_dps
n_microsaccade_events
fraction_microsaccade_samples
peak_microsaccade_speed_dps
source RMS
max radius
path length
speed p95
```

The actual paired trace must also pass the drift-only source criteria. If too many actual traces fail, report both:

```text
strict drift-only actual-pair subset
natural fixation subset with microsaccade-containing windows flagged
```

but make the strict drift-only subset primary.

### Motion scales

Primary:

```text
0.25x
0.5x
1x
```

Optional diagnostic:

```text
1.5x
2x
```

Only include 1.5x/2x if there is no clipping/capping or if the result is plotted by effective RMS. The local screen previously had high clipping at large nominal scales, so large-scale local claims must be effective-RMS-aware.

### Candidate trajectory families

For each image/window `i`, generate responses under:

#### 1. Actual paired empirical trace

```text
I_i + tau_i
```

This is the real local pairing.

#### 2. Matched unpaired empirical traces

```text
I_i + tau_j, j != i
```

Match unpaired traces as closely as practical by:

```text
session if possible
duration
effective RMS
path length
drift anisotropy
microsaccade-free status
```

If within-session matching leaves too few candidates, use cross-session matching but report it.

Use multiple unpaired samples per image:

```text
K_unpaired = 4 minimum
K_unpaired = 8 preferred if compute allows
```

#### 3. Rotated actual trace

```text
I_i + rotate(tau_i, theta)
```

Use at least:

```text
90-degree rotation
random rotation preserving radius/time structure
```

Purpose:

```text
Tests whether the actual trace orientation relative to the image matters.
```

If actual ≈ rotated, useful structure is probably trace kinematics rather than local orientation matching.

#### 4. OU matched trace

OU matched to actual trace statistics:

```text
effective RMS
duration
lag-1 autocorrelation
confinement/covariance shape
approximate path length if implemented
```

Purpose:

```text
Tests whether empirical trace statistics beat a confined synthetic drift model.
```

#### 5. Brownian matched trace

Brownian matched by effective RMS and duration. Optional but useful as a loose generic-motion control.

#### 6. Raw edge-axis / edge-orthogonal local candidate traces

Use axis-constrained small-motion traces along:

```text
raw edge axis
edge-orthogonal axis
```

Use matched scale and duration. These are not empirical trace controls; they are image-geometry baselines.

Purpose:

```text
Tests whether actual drift is better than a simple image-geometry baseline.
```

Raw edge is the known strong baseline. Any model/information objective earns explanatory value only if it beats raw edge or explains residuals/deviations from raw edge.

## Feature Latents

Use only the feature families that are already supported.

Primary:

```text
Gabor local field, k=4
Pyramid local field, k=8
```

Secondary if cheap:

```text
Gabor k=8
Pyramid k=4
DCT k=8
```

Do not broaden into a full feature-family screen until the paired-vs-unpaired test is interpretable.

Feature preprocessing must match the corrected post-fix implementation:

```text
Gabor local fields include even, odd, and amplitude maps
Pyramid local fields use expanded local grid
feature arrays are standardized within training folds
PCA/feature reduction fit only on training data if cross-validated
```

## Response Summaries

Use the same family of readouts as the aggregate analysis so results are comparable.

Primary:

```text
temporal PCA
```

Secondary:

```text
temporal DCT
delta_mean
mean
```

Interpret readouts separately:

```text
temporal PCA/DCT = temporal response structure
delta_mean = low-dimensional motion-induced feature remapping
mean = time-integrated response change
```

Do not collapse across readouts unless explicitly labeled.

## Decoder and Scoring

Use grouped cross-validation by image/window.

Required:

```text
decode_group_mode = image or source_row
all rows from the same image/window remain in the same fold
```

Primary metric:

```text
negative held-out MSE for z decoding
```

Higher is better.

Primary incremental metric:

```text
gain_family = score(R_static + R_motion_family) - score(R_static)
```

Primary local-pairing contrast:

```text
paired_advantage = gain_actual_paired_empirical - mean(gain_matched_unpaired_empirical)
```

Secondary contrasts:

```text
actual_minus_rotated = gain_actual_paired_empirical - gain_rotated_actual
actual_minus_OU = gain_actual_paired_empirical - gain_OU
actual_minus_Brownian = gain_actual_paired_empirical - gain_Brownian
actual_minus_edge = gain_actual_paired_empirical - gain_edge_axis
actual_minus_edge_orthogonal = gain_actual_paired_empirical - gain_edge_orthogonal
edge_minus_edge_orthogonal = gain_edge_axis - gain_edge_orthogonal
```

Regularization:

Use one of these primary policies:

```text
shared alpha per latent/readout/scale across all families
nested alpha with same alpha grid and identical fold structure
```

Candidate-specific alpha is allowed as a sensitivity analysis but should not be the only figure-level result.

Save:

```text
alpha selected per fold/family
feature standardization statistics
decoder fold assignments
group IDs
```

## Primary Hypotheses

### H1: Local pairing beyond aggregate statistics

```text
actual paired empirical > matched unpaired empirical
```

Interpretation:

Actual real drift traces are locally matched to their image patches beyond the usefulness of the empirical trace distribution.

### H2: Orientation-specific local matching

```text
actual paired empirical > rotated actual trace
```

Interpretation:

The actual trace orientation relative to local image geometry matters.

If actual ≈ rotated:

```text
local pairing benefit, if present, likely reflects empirical kinematics/coverage rather than local orientation.
```

### H3: Image-stable axis constraint

```text
edge axis > edge-orthogonal axis
```

and/or

```text
actual paired empirical advantage is larger when real drift is edge-parallel.
```

Interpretation:

Real drift may be useful partly because it follows local image-stable axes.

### H4: Regime dependence

The local pairing advantage should be stronger in windows with:

```text
high edge coherence
high edge density
high drift anisotropy
high real-edge alignment
high pixel edge-parallel stability advantage
high V1-twin edge-parallel stability advantage
```

This is important. The global mean may be small even if the local policy is real in the right image regimes.

## Stratification Plan

Predefine a small number of regime splits. Do not explore many bins.

Primary strata:

```text
high vs low image_orientation_coherence
high vs low drift anisotropy
high vs low real-edge alignment
high vs low edge-parallel pixel stability advantage
```

Optional:

```text
high vs low observed RMS
high vs low edge density
high vs low V1-twin edge-parallel stability advantage
```

For each stratum, report:

```text
paired_advantage
actual_minus_rotated
actual_minus_OU
edge_minus_edge_orthogonal
n windows
session distribution
```

Use within-session or session-clustered bootstrap when feasible.

## Output Tables

Create an output folder:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_local_pairing_Iz_revisit_<tag>/
```

Required files:

```text
run_metadata.json
analysis_windows.csv
trace_bank_qc.csv
candidate_metadata.csv
motion_sanity_by_family_scale.csv
decode_scores_by_fold.csv
incremental_gains_by_window.csv
paired_contrasts_by_window.csv
session_summary.csv
bootstrap_summary.csv
regime_stratification_summary.csv
alpha_policy_summary.csv
```

Required metadata fields:

```text
unit_space
checkpoint path
population_n
feature family
feature k
response summary
scale
candidate family
actual/unpaired/rotated/OU/Brownian/edge
trace source id
image/window id
session
effective RMS
target RMS
effective/requested RMS
clipping/capping flag
microsaccade event count
path length
speed p95
fold id
decode group id
selected alpha
```

## Output Figures

### Figure 1: Local pairing contrast

Plot:

```text
actual paired - matched unpaired empirical
```

Across scales for:

```text
Gabor k=4 temporal PCA
Pyramid k=8 temporal PCA
```

Show CIs and zero line.

### Figure 2: Control contrast panel

For the primary scale/readout, plot:

```text
actual - unpaired
actual - rotated
actual - OU
actual - Brownian
actual - edge
edge - edge_orthogonal
```

Use signed bars with CIs.

### Figure 3: Regime dependence

Plot paired advantage for high vs low:

```text
image coherence
drift anisotropy
real-edge alignment
edge-parallel stability advantage
```

### Figure 4: Local geometry link

Scatter or binned plot:

```text
x = real-edge alignment or edge-parallel stability advantage
y = paired advantage
```

This is the key bridge to the behavioral geometry result.

### Figure S1: Trace/QC

Show:

```text
accepted vs rejected traces
RMS distribution
speed p95 distribution
microsaccade events
effective/requested RMS
clipping fraction
```

### Figure S2: Decoder QC

Show:

```text
score_static
score_static_plus_actual
score_static_plus_unpaired
score_static_plus_rotated
score_static_plus_OU
alpha distributions
fold/group counts
```

## Minimal Pathfinder

If compute is limited, run:

```text
n = 128 windows
K_unpaired = 4
canonical 756 units
scales = 0.25x, 0.5x, 1x
latents = Gabor k=4, Pyramid k=8
summary = temporal PCA only
families = actual paired, matched unpaired empirical, rotated actual, OU, edge, edge-orthogonal
alpha = shared or nested
CV = grouped by image/window
trace bank = drift-only
```

Decision criteria:

```text
If actual paired > unpaired at any scale and especially in high-coherence/high-alignment strata:
    promote local pairing as a candidate Figure 4 behavioral-payoff panel.

If actual paired ≈ unpaired but edge > edge_orthogonal and real drift is edge-aligned:
    keep local result as image-stability constraint; aggregate distributional result remains main functional panel.

If actual paired < unpaired or rotated beats actual:
    do not claim image-specific trace matching; interpret aggregate result as distributional and keep behavior result as geometry/stability only.
```

## Full Run

If the pathfinder is promising:

```text
n = 256 windows
K_unpaired = 8
canonical 756 units
scales = 0.25x, 0.5x, 1x, optional 1.5x
latents = Gabor k=4, Pyramid k=8
summaries = temporal PCA, temporal DCT, delta_mean
families = actual paired, matched unpaired empirical, rotated actual, OU, Brownian, edge, edge-orthogonal
```

Use session-clustered or session-aware bootstrap.

## Interpretation Outcomes

### Outcome A: actual paired > unpaired and > rotated

Strong local matching result.

Claim:

```text
Empirical drift statistics are useful in aggregate, and actual image-trace pairings provide additional local benefit.
```

This would let Figure 4 connect aggregate active sensing to real behavior more directly.

### Outcome B: actual paired > unpaired but actual ≈ rotated

Partial local matching.

Claim:

```text
Local pairings may benefit from empirical trace kinematics or displacement coverage, but original trace orientation is not the key ingredient.
```

### Outcome C: actual paired ≈ unpaired, but edge > edge-orthogonal

Distributional plus stability result.

Claim:

```text
Empirical drift statistics are useful at the ensemble level, while local behavior is better understood as an image-stability constraint than exact feature-information optimization.
```

### Outcome D: actual paired ≈ unpaired and edge ≈ edge-orthogonal

Local revisit does not support a local `I_z` mechanism.

Claim:

```text
The active-sensing signal is distributional in the aggregate assay; local per-image/per-trace matching remains unsupported under these readouts.
```

### Outcome E: unpaired or rotated > actual

Important negative.

Claim:

```text
The actual trace-image pairing is not optimized for this feature-decoding endpoint; the aggregate benefit likely arises from broad empirical motion statistics rather than local trajectory selection.
```

## Figure 4 Integration

The local revisit should only enter the main figure if it adds one of these payoffs:

1. actual paired empirical beats matched unpaired empirical;
2. actual paired empirical beats rotated actual trace;
3. actual-pairing benefit is concentrated in image-stable/high-coherence regimes;
4. actual-pairing benefit correlates with real-edge alignment or edge-parallel stability advantage.

Otherwise, keep current Figure 4 structure:

```text
aggregate empirical drift statistics add feature-decodable temporal structure
+
real drift follows image-stable axes
```

and report the local revisit as a boundary/supplement.

## Implementation Notes

Suggested new script:

```text
declan/fixation_statistics_by_stimulus/run_backimage_local_pairing_Iz_revisit.py
```

Suggested posthoc:

```text
declan/fixation_statistics_by_stimulus/summarize_backimage_local_pairing_Iz_revisit.py
```

Reuse modules from:

```text
run_backimage_latent_information_screen.py
run_backimage_aggregate_fem_information.py
audit_backimage_latent_real_random.py
jake/twininfo/eye_controls.py
declan/vernier_active_sensing/trajectories.py
```

Prefer the aggregate runner’s corrected trace-bank and grouped-CV machinery over the older local runner’s per-axis path wherever possible.

Hard requirements before interpreting:

```text
CV grouped by image/window
same physical windows across all candidate families
same trace sources reused across scales where appropriate
effective RMS reported, not only nominal scale
no hidden clipping/capping
microsaccade filtering reported
feature arrays have corrected dimensions
response traces are aligned to static baseline
alpha policy reported
```

## Claim Boundary

This analysis is a local-pairing test, not a new broad optimizer screen.

The safe wording, even if positive, is:

```text
actual image-trace pairings provide additional feature-decodable temporal structure beyond matched unpaired empirical traces under a V1-twin readout
```

Do not write:

```text
the animal selects optimal FEM trajectories for each image
```

unless substantially stronger optimizer and behavioral controls are added.
