# Why SF×orientation×TF routing must be evaluated locally

Date: 2026-08-13

Status: methodological rationale and implementation contract

Scope: RR100 fixed-retina grating fits, corrected BackImage retinal movies,
digital-twin activation maps, and spatial SSI

## Executive conclusion

The existing fixed-retina SF×TF and SF×orientation tuning measurements are
**local unit measurements**. Each fit describes how an RR100 unit responds when
a windowed grating is presented through that unit's native spatial receptive
field. When the same digital-twin readout is applied to a larger retinal movie,
it is convolved across spatial positions and produces an activation map.

The corresponding spectral surrogate must preserve that construction. It
should apply translated, overlapping copies of the native local measurement
aperture across the retinal movie, estimate local SF×orientation×TF power at
each activation-map position, and weight each local spectrum by the unit's
measured tuning. The result is a **predicted spatial drive map**, not one scalar
for the entire movie.

This step is necessary before using spectral routing to explain activation-map
changes or spatial SSI. A whole-movie spectral scalar discards the spatial
organization that those outcomes depend on.

## What the fixed-retina tuning fit measures

The SF×TF probe was not fit to an abstract feature channel detached from the
unit's receptive field. It was measured by presenting a temporally modulated,
spatially windowed grating in the native 51×51 stimulus field and passing that
movie through the fitted unit's normal model pathway:

1. the shared convolutional core;
2. the unit-specific feature weights;
3. the unit-specific learned spatial readout;
4. the model output nonlinearity.

For the 51×51 input, the learned spatial readout returns a 1×1 output. The
measured response therefore belongs to the unit at its native spatial
location. The fitted surface

\[
H_u(sf,\theta,tf)
\]

summarizes the F0 response of unit \(u\) to gratings with different spatial
frequency, orientation, and temporal frequency through that local pathway.
It does not say that the same unit has pooled the corresponding spectral power
uniformly over a much larger image.

Relevant implementations are:

- `declan/active_sensing_movie_information/run_backimage_rr100_frequency_tuning_probe.py`
  for construction of the 51×51 windowed grating probe;
- `scripts/spatial_info.py` for the learned feature and spatial readout;
- `declan/active_sensing_movie_information/run_backimage_rr100_dense_sf_tf_grating_probe.py`
  for the dense fixed-retina tuning work.

## What changes for a larger retinal movie

On a larger input, the core produces a larger spatial feature map. The fitted
unit's learned spatial readout is then applied convolutionally across that
feature map. A single readout value becomes a two-dimensional activation map:

\[
R_u(p,t),
\]

where \(p\) is an activation-map position. In the audited RR100 geometry, one
activation-map step corresponds to approximately two stimulus pixels.

The full twin is therefore performing translated copies of the same local
unit operation. A comparable spectral model must also be spatially
translation-aware. Repeating one whole-movie scalar at every map position
would not be equivalent: it would predict a spatially constant map and could
not explain where activation becomes concentrated or dispersed.

## Correct local spectral construction

For each activation-map position \(p\), define a translated local aperture
\(A_p(x,y)\). For retinal movie \(I(x,y,t)\), estimate the local
spatiotemporal power

\[
P_p(sf,\theta,tf)
=
\left|
\mathcal{F}_{x,y,t}
\left\{A_p(x,y)I(x,y,t)\right\}
\right|^2.
\]

Then combine this power with the independently measured unit tuning:

\[
D_u(p)
=
\sum_{sf,\theta,tf}
P_p(sf,\theta,tf)H_u(sf,\theta,tf).
\]

Under the current direct-F0 convention, \(H_u\) is the nonnegative measured
F0 sensitivity itself; it should not be squared again without introducing and
validating a different response model.

The collection of \(D_u(p)\) over all valid positions is the predicted local
spectral-drive map for unit \(u\). Computationally, this can be implemented as
a sliding-window spatiotemporal Fourier transform or an equivalent complex
filter bank. It does not require materializing independent, non-overlapping
image tiles.

"Tiling" therefore means:

- translated, overlapping local measurements;
- centres aligned to the digital twin's activation-map coordinates;
- the same spatial stride, valid support, padding, and edge conventions as the
  twin;
- unit-specific SF×orientation×TF weights at every location.

It does **not** mean dividing the movie into disjoint 51×51 blocks or copying
one global prediction over the map.

## Why the aperture must be calibrated rather than guessed

The Gaussian envelope used to present the grating and the learned Gaussian
spatial readout are not individually identical to the composite input-space
receptive field.

In particular, the learned spatial mask lives on the core feature map. Its
width does not include the spatial support accumulated within the convolutional
core. Using that mask alone as the input-space aperture would underestimate the
composite receptive field. Conversely, applying the learned mask again after
using a tuning surface that already contains its effects could double-count
spatial weighting.

The native 51×51 measurement is therefore the correct starting basis, but its
equivalence on the larger canvas should be verified empirically:

1. Embed the exact 51×51 grating movie in the centre of the larger production
   canvas.
2. Confirm that the central activation-map value reproduces the native 1×1
   grating response for several units, SFs, orientations, and TFs.
3. Translate the embedded probe by known input-pixel offsets.
4. Verify that its response translates to the expected activation-map
   positions and measure the exact input-pixel-to-map-position relation.
5. Test central and edge locations to establish valid support and padding
   behavior.

This validation defines the effective local aperture and coordinate contract
without pretending that one architectural component is the entire receptive
field.

## Why whole-movie power is insufficient for activation maps

The current orientation-aware routing analysis first collapses each retinal
movie to one SF×orientation×TF power tensor and then produces one scalar per
unit and condition. Schematically, it evaluates

\[
X_{u,c}
=
\sum_{sf,\theta,tf}
P_c(sf,\theta,tf)H_u(sf,\theta,tf).
\]

That is a legitimate test of whether a **global condition-level spectral
summary** predicts a scalar response summary. It is not a prediction of the
digital twin's activation map.

Two movies can have almost identical global spectral power while placing that
power in different parts of the image. They can therefore produce different
activation maps. Likewise, FEM can redistribute local power differently near
different edges, textures, and receptive-field locations even when the total
power is unchanged.

A whole-movie collapse removes:

- the location of image structure;
- local contrast differences;
- local interactions between image orientation and eye trajectory;
- the spatial concentration or dispersion of predicted unit drive;
- boundary and valid-support effects.

Those are precisely the quantities needed to compare with activation maps.

## Why this is especially necessary for SSI

Spatial SSI is calculated from the distribution of activity over an activation
map. It is nonlinear: calculating SSI from a spatially or temporally averaged
map is not generally equivalent to averaging SSI calculated from instantaneous
maps.

Consequently, a scalar routed-power predictor cannot mechanistically explain
SSI. It contains no predicted spatial distribution from which SSI could be
calculated. A regression from one global spectral scalar to observed SSI may
find an association, but it cannot establish that local spectral routing
produced the observed map sharpening or broadening.

The required causal chain is instead:

\[
\text{retinal movie}
\rightarrow
P_p(sf,\theta,tf)
\rightarrow
D_u(p)
\rightarrow
\text{predicted activation map}
\rightarrow
\text{predicted SSI}.
\]

Predicted and twin SSI must then use the same map support, temporal definition,
normalization, and spike weighting. Instantaneous SSI, time-averaged
instantaneous SSI, and SSI of a mean map must remain separately labeled.

## What F0 can and cannot predict

The tuning surfaces primarily characterize F0: the phase-averaged mean response
to each grating condition. This makes them natural candidates for predicting a
local energy or mean-drive map over a matched temporal interval.

F0 alone does not retain stimulus phase. It therefore should not initially be
claimed to reconstruct exact frame-by-frame activations. The first comparison
should use map summaries compatible with phase-averaged tuning, such as:

- the twin's time-averaged activation map;
- a local RMS or energy map;
- an FEM-minus-stabilized map summarized over the same temporal window.

An instantaneous prediction would require a time-resolved local filter-bank
construction aligned to the twin's 32-frame temporal history. Even then,
magnitude-only F0 tuning may be insufficient because it omits temporal phase
and nonlinear contextual effects. This is a testable extension, not an
assumption.

## Revised scope of the existing negative result

The existing population result should be stated narrowly:

> Across the tested conditions, whole-movie spectral summaries weighted by
> direct-F0 SF×orientation×TF tuning did not predict the selected scalar neural
> outcomes better than the corresponding simpler global summaries.

It does not establish that:

- local spectral routing fails;
- the unit's tuning is irrelevant to its spatial activation pattern;
- FEM-induced changes in activation-map structure are globally uniform;
- spectral routing cannot explain SSI.

The activation-map hypothesis has not yet received the matched spatial test.
This is not merely an additional refinement: it restores the spatial operation
present in both the fitted twin and the outcome being explained.

For spatially averaged rate, global power may approximate mean local energy
under restrictive assumptions such as spatial stationarity, linearity, and
negligible boundary effects. That approximation must be evaluated rather than
silently extended to spatial maps or SSI.

## Map-first validation sequence

The analysis should proceed from the smallest visible equivalence test to
population summaries.

### 1. Native-to-large-canvas equivalence

Use exact embedded grating probes to establish the aperture, stride, response
scale, coordinate mapping, and edge behavior. Failure here invalidates the
surrogate before any natural-image test.

### 2. One movie and several units

For one corrected FEM movie and its stabilized counterpart, show:

- the image and eye trajectory;
- local SF×orientation×TF power at selected positions;
- each selected unit's measured tuning surface;
- the predicted local-drive map;
- the twin activation map summarized over the matched interval;
- FEM-minus-stabilized difference maps for both predictions and observations.

Use at least a positive example, a dissociation, and a negative/control example.
The selection rules and values should be saved to a table.

### 3. Time-resolved diagnostic

Only after the mean/energy maps are interpretable, test sliding temporal
windows. Compare predicted and observed map timecourses and identify failures
caused by phase, history, normalization, or nonlinear response effects.

### 4. SSI derived from maps

Calculate predicted SSI from the predicted maps using the same estimator as
the twin analysis. Preserve rate, expected-spike, raw spatial-information, and
bits-per-spike quantities. Prefer condition differences over unstable ratios.

### 5. Held-out population evaluation

After the map-level examples are understood, test generalization across held-out
images and traces. Compare at least:

- local unit-specific SF×orientation×TF routing;
- local total dynamic power without unit tuning;
- the previous whole-movie predictors;
- a simple local-contrast or energy control.

Evaluate activation-map prediction before SSI prediction. Report positive
examples and dissociations alongside aggregate performance.

## Decision criterion

The local routing account is supported only if independently measured tuning
improves prediction of the **location and magnitude** of twin activation over
appropriate local controls, and if the resulting predicted maps reproduce a
meaningful part of the FEM-induced SSI change on held-out data.

If local total power performs as well as unit-specific routing, that would
support a more spatially local but spectrally broad drive account. If neither
predicts activation maps, the failure would point toward missing phase,
nonlinear context, incomplete tuning, or an incorrect local-aperture model.
Those outcomes remain scientifically informative because the comparison is
made at the same spatial level as the twin computation.

## Provenance and current claim boundary

This document defines the next analysis contract; it is not evidence that the
local spectral predictor will succeed. Existing fixed-retina tuning can be
reused, but retinal movies must follow the corrected `dpi_pix`, 120-Hz,
explicit-history construction documented in
`CORRECTED_BACKIMAGE_ANALYSIS_PLAN_GPU_DEFERRED.md`.

The initial work should remain a targeted map-first validation. It should not
be presented as a production population result until the equivalence tests,
example maps, temporal definitions, and held-out evaluation have passed.
