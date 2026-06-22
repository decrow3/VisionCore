# Companion: Local I_z Pairing Model

Date: 2026-06-21
Status: provisional methods/logic companion for the local mechanistic branch

## Summary

The local `I_z` pairing model asks a narrower question than the aggregate FEM
model. It does not ask whether empirical-like trajectories are useful as a
distribution. It asks whether the actual local image and the actual measured
trace carry paired sensitivity: does the response change from `I_i` under
`tau_i` help recover the local feature of `I_i` more than matched but unpaired
trace controls?

The simplifying assumption it breaks is exchangeability between image content
and trajectory. If all empirical traces are equally useful for all images, then
actual image-trace pairing should not matter after matching RMS and source
policy. If local image geometry matters, the actual pairing may have a small
advantage in a motion-delta readout.

The current claim is deliberately bounded. The `delta_mean` readout supports a
local mechanistic-sensitivity interpretation, especially at small scales and
against OU/Brownian controls. The paired advantage over matched-unpaired traces
is not yet stable enough to carry a main headline.

## Motivation

The aggregate model can be positive even if the exact pairing between image and
trace is irrelevant. A distribution of biological-like motion might add useful
temporal response structure on average, while the particular trace recorded on
a particular image could be interchangeable with another empirical trace. The
local pairing model tests that stronger pairing claim.

This is why the local branch belongs in the companion set but not as the main
Figure 4B endpoint. It is closer to a sensitivity analysis: it probes whether
the motion-induced response delta carries local image-contingent information
under strict matched controls.

## Notation And Estimator Contract

Shared notation:

```text
I_i: image/window i
tau_i: measured trajectory paired with image/window i
tau_j: matched trajectory from another image/window
y_i(tau) = f_theta(I_i, tau): V1-twin response movie
phi(I_i): local image feature target
s(y): response summary
D(s(y), phi(I_i)): grouped-by-image feature-decoding score
```

For local pairing, define a static-plus-motion gain for a trajectory assignment
`A`:

```text
Delta(A; phi, s) =
  CV_D([R_static(I_i), R_motion(I_i, tau_{A(i)})], phi(I_i))
  - CV_D(R_static(I_i), phi(I_i))
```

The pairing contrast is:

```text
P(A, B; phi, s) = Delta(A; phi, s) - Delta(B; phi, s)
```

with the main assignments:

```text
A = actual_paired_empirical
B = matched_unpaired_empirical, rotated_actual_90, ou_matched_actual,
    brownian_matched_actual
```

The candidate local readout is:

```text
s(y) = delta_mean(y)
phi(I) = pyramid_local_field(I), k = 16
```

This readout is intentionally different from the aggregate `temporal_pca`
candidate. The local question is about motion-induced response displacement,
not broad ensemble temporal decodability.

## Assumptions

A1. The actual image-trace pairing is meaningful after the reviewed manifest,
trace-bank, and coordinate fixes.

A2. Matched-unpaired controls draw from the full valid trace pool while
avoiding same-trial matches, so the comparison is not just a reduced-pool
artifact.

A3. Motion-family controls are matched in effective RMS and clipping, so
pairing contrasts are not explained by simple motion amplitude.

A4. `delta_mean` is a legitimate local sensitivity readout even if temporal PCA
is stronger for aggregate ensemble decoding.

A5. The grouped-by-image decoder and fixed 128-image manifest are adequate for
a local sensitivity test, but they are not a replacement for the higher-power
canonical rerun.

## Controls

Matched-unpaired empirical:

```text
Uses empirical traces from other windows/images. This is the hardest control
for the exact-pairing claim because it keeps the empirical trace distribution.
```

Rotated actual trace:

```text
Keeps the actual path magnitude but changes the image-relative direction.
```

OU and Brownian matched controls:

```text
Test whether confined drift or generic diffusion can explain the same local
motion-delta gain.
```

Rel2 sentinel:

```text
Checks whether any small-scale pairing read turns into a generic large-motion
effect.
```

Trace-bank and manifest QC:

```text
The clean runs use a fixed 128-image manifest, the full 3013-row trace pool,
zero same-trial matched-unpaired controls, corrected feature geometry, and zero
clipping in the inspected motion summaries.
```

## Existing Evidence

Primary local source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_local_pairing_Iz_power_pyramid_k16_rel025_0p5_1_seed7_k64_v1/
  backimage_local_pairing_Iz_power_pyramid_k16_rel025_0p5_1_seed11_k64_v1/
```

Primary local posthocs:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_local_pairing_Iz_power_pyramid_k16_rel025_0p5_1_seed7_k64_v1/
    incremental_static_plus_motion_tworeadout_v1/
  backimage_local_pairing_Iz_power_pyramid_k16_rel025_0p5_1_seed11_k64_v1/
    incremental_static_plus_motion_tworeadout_v1/
```

Run scope:

```text
128 images
28 sessions
families = actual_paired_empirical, matched_unpaired_empirical,
  rotated_actual_90, ou_matched_actual, brownian_matched_actual
scales = 0.25x, 0.5x, 1x
matched-unpaired samples per image = 64
median effective/requested RMS = 1.0
clipped fraction = 0.0
```

Power local `delta_mean` seed-mean contrasts:

```text
actual - matched_unpaired:
  0.25x +4.24
  0.5x  +3.74
  1x    +3.44

actual - Brownian:
  0.25x +9.95
  0.5x  +7.56
  1x    +7.15

actual - OU:
  0.25x +2.88
  0.5x  +2.99
  1x    +3.70

actual - rotated:
  0.25x +1.35
  0.5x  +2.69
  1x    +1.91
```

This pattern is now much more stable as a local sensitivity result. It supports
the idea that the local motion-delta readout is sensitive to actual empirical
motion at primary scales, including paired-vs-matched-unpaired separation
across both power seeds. It remains a mechanistic sensitivity result rather
than the whole aggregate headline. The corrected aggregate branch now separates
readout roles: `mean`/`delta_mean` are absolute aggregate candidates, temporal
PCA/DCT variants are order-sensitive diagnostics, and rotated controls remain a
necessary guardrail.

## Diagnostics And Failure Modes

The local branch is especially vulnerable to these failure modes:

```text
matched-unpaired empirical traces are as good as actual traces;
rotated actual traces remain competitive;
temporal_pca and DCT summaries do not show the same local pairing result;
large-scale sentinel rows turn the effect into generic motion;
the trace bank or manifest restriction accidentally makes the controls unfair;
local feature geometry differs from the aggregate convention;
the result depends on one seed, manifest, or K_unpaired sample.
```

Current handling:

```text
Keep this branch as the local mechanistic sensitivity readout. It can support a
companion/supplement panel, and it should be coordinated with the aggregate
mean/delta-mean absolute-gain panel rather than forced to replace or validate
the temporal PCA/DCT diagnostic branch.
```

## Current Claim Boundary

Supported:

```text
The local delta_mean readout provides mechanistic sensitivity evidence that
motion-induced V1-twin response changes can carry local image feature structure.
Actual paired traces separate from matched-unpaired and Brownian controls
across the primary 0.25x, 0.5x, and 1x scales in the seed7/seed11 power
summaries.
```

Not yet supported:

```text
Actual image-trace pairing is cleanly better than matched-unpaired empirical
traces across all readouts and all possible scales.
Temporal PCA is the right local pairing readout.
Recorded traces are uniquely optimal for their images.
The local pairing branch should replace the aggregate model as the Figure 4B
headline.
```

## Production Rerun Implications

The final local branch should use:

```text
local mechanistic sensitivity: pyramid_local_field k16 delta_mean
production targets:
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_local_pairing_Iz_power_pyramid_k16_rel025_0p5_1_seed7_k64_v1
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_local_pairing_Iz_power_pyramid_k16_rel025_0p5_1_seed11_k64_v1
```

Before promotion, the figure or supplement should report:

```text
actual gain over static
actual minus matched-unpaired empirical
actual minus rotated actual
actual minus OU and Brownian
motion QC and trace-bank source counts
seed/K_unpaired sensitivity
2x sentinel behavior only if later rerun specifically adds it
```
