# Companion: Retinal Movie Premise

Date: 2026-06-22
Status: provisional methods/logic companion for Figure 4A

## Panel Claim Under Test

```text
One image becomes a retinal movie.
```

This is the claim Panel 4A is there to show. During fixation, the image on the
screen is fixed, but the eye is not fixed. The image therefore moves across the
retina. Panel 4A is the physical premise for the rest of Figure 4: before we
ask whether motion helps the model, supports joint decoding, follows edges, or
matches behavior, we first show how a recorded fixation trace turns one
BackImage screen image into a sequence of retinal samples.

## Summary

Panel 4A is not a decoder result. It is a rendering and quality-control panel.
The selected example uses a real BackImage image canvas and a recorded eye
trace. The figure marks the screen image, the eye path, and several retinal
crops sampled along that path. The companion QC panels check that the real
movie has temporal contrast and motion power, while the stabilized control
removes retinal translation.

The panel supports a narrow statement: measured fixational motion creates a
structured retinal movie from a static screen image. It does not by itself show
that the motion is optimal, that the animal chose the path to encode features,
or that downstream observers decode the movie correctly.

## Motivation

The later panels depend on one simple fact that can otherwise get hidden inside
model code: every response movie starts with a movie on the retina. If the
screen image is `I` and the eye position changes over time, the model does not
see the same retinal crop at every time point. It sees a translated crop of the
same screen image at each sampled eye position.

Panel 4A makes that transformation visible. It also records the checks that
the transformation is real in the data products used by the rest of the figure,
not just a cartoon.

## Plain-English Methods

The 4A panel was built in four steps.

First, choose a real BackImage fixation window. The current selected example is
candidate 3 from the single-panel candidate sheet. It comes from session
`Logan_2020-01-10`, trial `407`, using eye samples `397359:397487`. The window
was selected because the local image patch has high contrast, a clear local
orientation, and strong positive drift-edge alignment metadata. These features
make the visual transformation easy to see at print scale.

Second, recover the screen image and eye trace for that window. The screen
image is loaded from the BackImage canvas for the session and trial. The eye
positions are the recorded `backimage.dset` eye-position samples for the same
global time range. The conversion from degrees of eye position to image pixels
uses the stored pixels-per-degree value for the BackImage data.

Third, render retinal samples. For each selected time point in the fixation
trace, the code shifts the screen image by the corresponding eye position and
takes the crop that would fall on the model retina. In plain terms, the screen
does not move, but the crop window moves across it. The figure shows several
of these crops so the reader can see that one screen image has become a short
movie.

Fourth, check that the movie transformation behaves as expected. The QC summary
compares real retinal movies with matched stabilized movies. The real movies
have nonzero temporal contrast and motion power. The stabilized movies remove
retinal translation, so their temporal contrast and motion power are near zero
by construction. Movie power is reported as a guardrail because the goal is not
to create a different image, but to change how the same image is sampled over
time.

## Notation And Estimator Contract

For a fixed screen image `I` and an eye-position trace `tau(t)`:

```text
I_screen: fixed image on the monitor
tau(t): measured eye position over time
I_retina(t): retinal crop produced by shifting I_screen by tau(t)
```

The transformation is:

```text
I_retina(t) = crop(I_screen shifted by tau(t))
```

The stabilized control uses the same screen image but removes the time-varying
shift:

```text
I_stabilized(t) = crop(I_screen shifted by a fixed reference position)
```

The QC values summarize temporal contrast, motion power, and total movie power
across rendered movies. These are rendering checks, not neural decoding scores.

## Assumptions

A1. The selected BackImage canvas and eye-position slice are correctly matched
by session, trial, and global sample index.

A2. The pixel-per-degree conversion is correct enough for figure-scale retinal
crop rendering and for the downstream model inputs.

A3. Stabilization is a useful control for the physical transformation because
it keeps the image fixed on the retina while preserving the same source image.

A4. The selected example is illustrative. It is not meant to estimate the full
population distribution of all BackImage fixations.

A5. The covariance bridge panel is supporting context only. Its denominators
are not identical across all source analyses, so it should not be read as a
single unified variance-fraction estimate.

## Controls

Stabilized movie:

```text
Uses the same screen image but removes retinal translation. This shows which
movie structure comes from eye motion.
```

Movie transform QC:

```text
Reports temporal contrast and motion power for real and stabilized movies.
This checks that the rendering code creates the intended retinal movie.
```

BackImage pipeline bridge:

```text
Shows that the same image windows and fixation traces feed the downstream
V1-twin analyses in Panels 4B-4D and the behavior geometry in 4E.
```

Covariance bridge guardrail:

```text
Connects the premise to earlier recorded-covariance evidence, while keeping
the denominator caveat visible.
```

## Existing Evidence

Selected single-panel example:

```text
declan/figure4_active_sensing_atlas/figures/panel_A/promotion_candidates/
  4A_single_panel_candidate_values.csv
```

Current selected candidate:

```text
candidate = 4A_candidate_3_real_high_contrast_positive
session = Logan_2020-01-10
trial_idx = 407
global_start:global_stop = 397359:397487
image_orientation_coherence = 0.773
anisotropy = 0.987
path_length_deg = 5.653
drift_edge_cos2 = 0.986
ppd = 37.505
```

Movie transform QC:

```text
declan/figure4_active_sensing_atlas/figures/panel_A/
  panel_A_movie_transform_qc_values.csv

temporal contrast RMS:
  real = 11.245
  stabilized = 0.000

motion power:
  real = 1462.431
  stabilized = 0.000

movie power:
  real = 15178.177
  stabilized = 15185.182
```

Pipeline bridge values:

```text
declan/figure4_active_sensing_atlas/figures/panel_A/
  panel_A_backimage_pipeline_values.csv

images = 256
sessions = 29
trace samples per condition = 4
trace sources = 151
RMS ratio = 1.0
max clipped fraction = 0.0
```

## Diagnostics And Failure Modes

The panel can be overread in these ways:

```text
one vivid example is mistaken for a population estimate;
retinal movie QC is mistaken for downstream feature decoding;
the stabilized control is treated as a biological no-motion condition rather
  than a rendering control;
the covariance bridge is treated as one common-denominator variance fraction.
```

Current handling:

```text
Keep Panel 4A framed as the physical premise.
Use the real candidate as an example, not a statistical claim.
Keep movie-transform QC visible.
Route downstream model and behavior claims to Panels 4B-4E.
```

## Current Claim Boundary

Supported:

```text
During fixation, recorded eye motion converts a fixed BackImage screen image
into a structured retinal movie, and the rendering/QC outputs show the expected
temporal contrast and motion power relative to stabilization.
```

Not yet supported by Panel 4A alone:

```text
The selected trajectory is optimal.
The animal chose the trajectory to improve feature encoding.
The V1 twin or behavior results follow automatically from the retinal movie.
The covariance bridge values form one single denominator-matched estimate.
```

## Production Rerun Implications

The production figure should report:

```text
session/trial/global sample provenance for the selected example
pixels-per-degree conversion
retinal crop construction from the recorded trace
real-vs-stabilized movie QC values
clear statement that Panel 4A is a premise/QC panel
```
