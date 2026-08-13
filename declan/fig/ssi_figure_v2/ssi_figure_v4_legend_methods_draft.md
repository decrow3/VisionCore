# ssi_figure_v4 legend and methods draft

This draft is written to match the July 29 manuscript tone and terminology.
It assumes the main Methods already contain the experimental preparation, eye
tracking, stimulus presentation, spike sorting, covariance decomposition, and
digital twin model architecture/training sections.

## Insertion Notes

The current v4 layout uses A-H lettering. If updating the existing Results
text from the July 29 draft, use the following mapping: old C -> B, old D -> C,
old E/F -> D, old G/H -> E, old I -> F, and old J -> G. New panel H is the
patch-radius sensitivity panel and should be referenced in the Results where
the scale of local contour measurement is interpreted.

Panel A's 267 ms label denotes the temporal depth of the model input cube
(32 input frames at 120 Hz) for an individual model prediction. It is
intentionally different from the 40 scored output samples used for the SSI
movie bank and behavior bridge, whose sample centers span 0.325 s. The legend
and Methods below make this distinction explicit.

## Figure Legend Draft

Figure 4: Fixational eye movements sharpen spatial population codes according
to neuronal tuning and local image structure. (A) Schematic of the model-based
single-spike information (SSI) analysis. A trained digital twin was used to
predict the spatial activation map of each model unit as a natural image was
translated by a measured FEM trajectory. SSI quantifies how far the resulting
activation map departs from a spatially uniform response, in bits/spike. The
267 ms label marks the model's 32-frame visual input history, whereas the
population SSI analyses aggregate 40 scored output samples whose centers span
0.325 s. In the example shown, the FEM-jittered movie produced a more
spatially structured response map than the counterfactually stabilized movie
(0.14 vs. 0.10 bits/spike). The displayed heat maps show predicted firing rate
after subtracting each map's spatial mean for visualization; SSI was computed
from the original nonnegative predicted rates. (B) SSI change relative to a
unit-matched stabilized baseline as a function of total FEM path length for
units split by preferred spatial frequency. Low-SF units (blue; 71 units, 7100 unit-image
pairs) gained progressively with path length, whereas high-SF units (orange;
29 units, 2900 unit-image pairs) showed little additional benefit at long path
lengths. Open markers indicate drift-only traces and filled markers indicate
traces containing microsaccades. (C) Local image structure was estimated around
each gaze position. A Sobel structure-tensor analysis of a gaze-centered
natural-image patch defined the local contour axis and an orientation coherence
score. Example patches illustrate the coherence bins used in later panels.
(D) The effect of path length depended on whether unit tuning was aligned with
the local contour. Among contour-aligned unit-image pairs, low-SF units
(57 units, 977 pairs) retained a positive path-length dependence, whereas
high-SF aligned units (22 units, 356 pairs) declined with longer paths,
especially for microsaccade-containing traces. (E) For high-SF aligned units,
SSI depended on the contour-relative direction of the eye movement. The x-axis
groups original two-dimensional drift-only trajectories by their component RMS
excursion projected either normal to the local contour (across, solid) or
parallel to it (along, dashed). The model always received the original
two-dimensional trajectory; the contour-relative projections were used only to
summarize and bin trajectories.
The gray vertical band marks the interquartile range of component RMS values
in the drift-only real-trace bank (1.22-1.72 arcmin), pooled across
contour-normal and contour-parallel projections, and is shown as a behavioral
dose reference rather than a confidence interval. The near-zero bracket tests
the first nonzero RMS bin against the stabilized anchor separately for the
across and along component groupings (two-sided image-bootstrap p-values:
across p = 0.0028, along p = 0.0882). SSI declined more strongly across bins
of across-contour RMS than across bins of along-contour RMS; in the last
displayed bin, the across-minus-along contrast was -5.1 percentage points
(two-sided image-bootstrap p = 0.0004). The far tail of the RMS distribution
(>3.8 arcmin) is omitted from the display. (F) Real FEM
position spread was anisotropic around local contours. For each reviewed
BackImage fixation window, eye positions were projected onto axes at different
angles relative to the local contour. Curves show the fixed-animal,
equal-weight hierarchical estimate in three local edge-coherence bands;
shading denotes 95% session/trial bootstrap confidence intervals. As coherence
increased, spread became larger parallel to the contour and smaller in the
orthogonal direction. (G) The observed contour-relative
orientation of real FEMs was compared with a random-rotation null by mapping
measured contour-relative RMS values through the model dose curves. Positive
values indicate that the real pairing between eye trajectory and local image
axis yielded higher predicted SSI than the rotation null. The predicted
advantage increased with coherence for aligned high-SF units and was largest
in the highest coherence bin (0.155 percentage points, 95% CI
[0.044, 0.265]). All high-SF and low-SF populations showed weaker or less
selective effects. Open markers indicate points whose 95% CI includes zero.
(H) The relationship between local edge
coherence and edge-following behavior depended on the scale at which local
image structure was measured. For each patch radius, the slope of the
edge-following alignment index versus local coherence was fit over windows
with coherence > 0.3. The estimated slope rose from small patches, reached its
maximum at a 1.25 degree radius, and remained positive at larger radii,
indicating that FEM anisotropy is coupled to local contour structure measured
over degree-scale image neighborhoods. Error bars denote 95% CIs: image
bootstrap in B, D, and E; hierarchical session/trial bootstrap with the two
fixed animals equally weighted in F; paired-window bootstrap in G; and
regression CIs in H.

## Results Replacement Draft

### Fixational eye movements sharpen spatial population codes

Sensor motion is ordinarily associated with a loss of spatial precision. A
conventional camera exposure averages together shifted images generated by
motion, blurring edges and reducing spatial contrast. Leveraging our model of
V1, we therefore asked whether the jittered retinal input generated by
fixational eye movements blurs cortical activity or, counterintuitively,
sharpens it. Compared with counterfactually stabilized input, natural retinal
motion produced activation maps with stronger and more spatially concentrated
responses (Fig. 4A). Thus, the same movements that introduce temporal
variation into the retinal input can sharpen its spatial representation in V1.

To quantify this effect, we used the digital twin to construct model-derived
populations of co-tuned neurons tiled across visual space. For each fitted
unit, we replicated its spatial readout at every position within the modeled
visual field, producing an activation map whose values represent the responses
of neurons with the same tuning but different receptive-field locations. We
quantified the information contained in each population map by computing
single-spike information across space (SSI; see Methods), defined as the
divergence of the activation map from a uniform distribution. SSI is
insensitive to overall response magnitude and instead captures how selectively
activity is distributed across the simulated population. A diffuse map, in
which activity is distributed broadly across many locations, has low SSI.
Conversely, a non-uniform map with localized spatial structure has high SSI,
such that the identity of an active neuron carries more information for a
downstream decoder. We therefore refer to increases in SSI as sharpening of
the spatial population code.

FEMs increased SSI across two broad classes of spatial tuning, but the
dependence on movement amplitude differed substantially between them. Across
low-spatial-frequency units (71 units; 7100 unit-image pairs), SSI increased
progressively with trajectory path length, and trajectories containing
microsaccades produced larger gains than drift-only trajectories of
comparable length (Fig. 4B). High-spatial-frequency units (29 units; 2900
unit-image pairs) also benefited from retinal motion, but over a more
restricted range. In this population, SSI showed little additional benefit at
long path lengths and tended to decline for the largest drift-only and
microsaccade-containing movements (Fig. 4B). Retinal stabilization was
therefore not the most informative condition for either population, but
neither was greater motion universally better. The movement that benefits a
neuron depends on the spatial scale to which it is tuned.

This dependence on neuronal tuning became more pronounced when we considered
the local structure of the image. We identified image regions containing a
coherent local contour and compared each unit's preferred orientation with
that contour axis (Fig. 4C). Low-spatial-frequency aligned units retained the
population-wide pattern: longer trajectories and microsaccades produced
progressively larger increases in SSI (57 units; 977 unit-image pairs;
Fig. 4D). High-spatial-frequency aligned units behaved differently. Small
movements produced modest benefits, but longer paths, particularly those
containing microsaccades, reduced SSI and could drive it below the stabilized
baseline (22 units; 356 unit-image pairs; Fig. 4D). The average benefit
observed across all high-spatial-frequency units therefore conceals a specific
cost to the neurons carrying fine information about the local contour.

This interaction follows from the geometry of retinal motion. Retinal
displacement converts spatial variation into temporal modulation, and the
modulation experienced by an oriented neuron depends on the component of
motion across its preferred spatial structure. Motion across a contour sweeps
its luminance profile through the receptive field, whereas motion along the
contour produces less change in the feature encoded by a contour-aligned
unit. When the SSI of the original two-dimensional trajectories was grouped by
contour-relative position spread, high-SF aligned units showed a stronger
dependence on across-contour than along-contour RMS (Fig. 4E). The first
nonzero across-contour bin lay above the stabilized baseline (p = 0.0028),
whereas the corresponding along-contour estimate did not (p = 0.0882). At the
largest displayed matched dose, SSI change was 5.1 percentage points lower for
trajectories grouped by across-contour RMS than for trajectories grouped by
along-contour RMS (p = 0.0004). Thus, contour-relative position spread,
particularly its across-contour component, was more informative about the
coding consequence of a trajectory than total path length alone.

Natural FEM position clouds showed this directional structure. Around coherent
contours, gaze positions were distributed more broadly parallel to the local
contour and more narrowly in the orthogonal direction, with the anisotropy
increasing as local edge coherence increased (Fig. 4F). In the equal-animal
summary, the parallel-minus-orthogonal difference in the 0.5-1 coherence band
was 0.224 arcmin (95% CI [0.090, 0.596]), although the effect was stronger in
Allen than Logan. Such geometry would permit relatively large overall spread
while limiting the across-contour displacement associated with lower
high-spatial-frequency aligned SSI. To
relate this behavioral anisotropy to the model results, we mapped the measured
contour-relative RMS values through the one-dimensional SSI dose curves and
compared the resulting predictions with random rotations of the same
eye-position clouds. The observed contour-trajectory pairing predicted greater
SSI than the rotation null for aligned high-SF unit-image pairs, with the
largest difference in the highest coherence bin (0.155 percentage points, 95%
CI [0.044, 0.265]; Fig. 4G). This correspondence is consistent with natural
FEM geometry modestly preserving high-SF spatial information near coherent
contours. Finally, this coupling between FEM geometry and local image
structure was scale dependent: the relationship between local edge coherence
and edge-following behavior reached its maximum when contour structure was
measured over a 1.25 degree radius and remained positive at larger radii
(Fig. 4H). FEM statistics therefore appear to distribute motion-dependent
information selectively across cortical subpopulations rather than maximizing
the response of every neuron simultaneously.

## Methods Draft

### Single-spike information analysis

We quantified how retinal image motion altered the spatial specificity of
model responses using the trained digital twin described above. For each unit,
image, FEM trajectory, and time bin, the model produced a two-dimensional
activation map over spatial position. Single-spike information (SSI) was
computed from each nonnegative activation map as the divergence of that map
from a uniform spatial response. Let \(r_t(x)\) denote the predicted response
at position \(x\) in time bin \(t\), and let
\(\bar r_t = \langle r_t(x) \rangle_x\). We computed

\[
SSI_t = \left\langle
\frac{r_t(x)}{\bar r_t}
\log_2 \frac{r_t(x)}{\bar r_t}
\right\rangle_x .
\]

This quantity is reported in bits/spike. It is insensitive to an overall
multiplicative change in firing rate and instead measures how selectively the
activity identifies spatial position. A broad activation map has low SSI,
whereas a localized activation map has high SSI.

Population SSI was computed by pooling over selected units, images, movie
rows, and time bins using the model's expected spike count in each time bin as
the weight. For a set of movie rows \(m\), units \(u\), and time bins \(t\),
we computed

\[
SSI_{\mathrm{pop}} =
\frac{\sum_{m,u,t} \hat n_{m,u,t} SSI_{m,u,t}}
{\sum_{m,u,t} \hat n_{m,u,t}},
\]

where \(\hat n_{m,u,t}\) is the predicted expected spike count for that time
bin. Thus, the population value is the amount of spatial information carried
per predicted spike by the selected population. All plotted SSI effects are
expressed as a percent change relative to a matched stabilized baseline,

\[
100 \times
\frac{SSI_{\mathrm{FEM}} - SSI_{\mathrm{stable}}}
{SSI_{\mathrm{stable}}}.
\]

### Natural-image FEM movie bank

The SSI analyses used a BackImage movie bank generated from real image patches
and measured fixational eye movements. We selected 100 BackImage image windows
with valid local image measurements and local orientation coherence >= 0.20.
These were crossed with 1000 real FEM snippets sampled from the reviewed
BackImage fixation windows, yielding 100,000 image-by-trajectory movies. The
trace bank contained both drift-only snippets and snippets containing
microsaccades; 200 of the selected traces contained microsaccades. Traces were
sampled over the empirical range of path lengths and were restricted to path
lengths <= 350 arcmin. Movies were evaluated for 40 scored output samples at
120 Hz using the RR100 model population view; the sample centers span 0.325 s
from first to last sample. The visual input cube shown in Panel A denotes the
model's 32-frame temporal history for a single prediction (267 ms at 120 Hz),
whereas the SSI summaries aggregate the 40 scored output samples.

For each selected image we also generated a counterfactually stabilized movie.
In this condition the same BackImage patch was held fixed for the same number
of time bins with zero retinal displacement. The stabilized responses were
scored with the same model, spatial map, and SSI procedure as the FEM-jittered
movies. For every plotted movement bin, the stabilized baseline was matched to
the same image composition and the same selected unit or unit-image population.
This unit-matched baseline prevents differences in image identity or unit
selection from being interpreted as movement effects.

### Local image structure and unit-image selections

Local image structure was measured in gaze-centered BackImage patches. Unless
otherwise noted, the patch radius was 1 degree. Patches were excluded if less
than 98% of the patch fell inside the image or if more than 5% of the patch was
background. The local contour axis was estimated from the Sobel structure
tensor. If \(g_x\) and \(g_y\) are horizontal and vertical image gradients, we
computed \(J_{xx}=\langle g_x^2\rangle\), \(J_{yy}=\langle g_y^2\rangle\), and
\(J_{xy}=\langle g_x g_y\rangle\) within the patch. The orientation coherence
was

\[
\frac{\sqrt{(J_{xx}-J_{yy})^2 + 4J_{xy}^2}}{J_{xx}+J_{yy}},
\]

and the local edge axis was defined as the axis orthogonal to the dominant
gradient axis. Image-derived axes were reported in gaze coordinates, with
positive x rightward and positive y upward.

Units were divided by the spatial-frequency metric from the model tuning
analysis. Low-SF units had `sf_split_metric < 0.5` cycles/degree, and high-SF
units had `sf_split_metric >= 0.5` cycles/degree. For contour-aligned
analyses, we further required a valid preferred orientation and orientation
selectivity index >= 0.05. Unit-image pairs were classified by the acute
axis-angle difference between the unit's preferred orientation and the local
image contour axis. Aligned pairs had a difference <= 15 degrees. Orthogonal
pairs, used in supporting diagnostics, had a difference >= 67.5 degrees, and
oblique pairs fell between these cutoffs. Because the contour axis varies from
image to image, the aligned population is naturally defined as a set of
unit-image pairs rather than as a fixed set of units alone.

### Path-length and contour-relative model dose curves

For the path-length analyses, total FEM path length was computed as the sum of
Euclidean sample-to-sample eye displacements, converted to arcmin. Traces were
separated into drift-only and microsaccade-containing contexts. Drift-only
traces were divided into eight equal-count path bins; microsaccade-containing
traces were divided into five equal-count path bins. A zero-motion stabilized
point was plotted at path length zero. Panels B and D show the unit-baselined
SSI change in each path bin for low- and high-SF populations, with and without
the unit-contour alignment restriction.

For contour-relative analyses, each two-dimensional FEM trajectory was
projected onto two axes defined by the local image: the contour-parallel axis
and the contour-normal axis. The original, unmodified two-dimensional
trajectory was supplied to the model in every condition; the contour-parallel
and contour-normal projections were used only to summarize and bin
trajectories, not to construct one-dimensional stimulus movies. For a
trajectory \(e_t=(x_t,y_t)\), contour axis \(u\), and normal axis \(v\), we
computed both accumulated component path and position-spread metrics.
Component path was the sum of the absolute projected sample-to-sample
displacements,

\[
P_u = 60 \sum_t |(e_{t+1}-e_t) \cdot u|,
\qquad
P_v = 60 \sum_t |(e_{t+1}-e_t) \cdot v|.
\]

Component RMS excursion was the standard deviation of centered projected eye
position,

\[
R_u = 60 \sqrt{\left\langle ((e_t-\bar e)\cdot u)^2 \right\rangle_t},
\qquad
R_v = 60 \sqrt{\left\langle ((e_t-\bar e)\cdot v)^2 \right\rangle_t}.
\]

The main contour-relative SSI panel uses component RMS excursion because the
behavior-model bridge showed that real behavior was better described by
position spread than by accumulated path. The RMS dose curves used original
two-dimensional drift-only movies and the aligned high-SF unit-image
population. Bins were constructed from pooled contour-normal and
contour-parallel RMS values so that the two component summaries were compared
at matched dose ranges. Body bins were defined by pooled quantiles, with an
additional tail bin. The gray vertical band in Panel E marks the interquartile
range of drift-only component RMS values in the real trace bank
(1.22-1.72 arcmin), pooled across contour-normal and contour-parallel
projections and independent of unit tuning. It is a reference for where real
drift samples lie on the model dose axis, not a confidence interval. The
near-zero bracket in Panel E compares the first nonzero RMS bin with the
zero-motion stabilized anchor using two-sided image-bootstrap p-values for
each component grouping separately. The across grouping was positive in this
first bin (p = 0.0028), whereas the along grouping was weaker (p = 0.0882).
The far tail above 3.8 arcmin was omitted from the displayed panel. The
reported across-minus-along contrast was computed in the last displayed bin by
bootstrapping over images.

### Statistical uncertainty for model SSI curves

For each movement bin and selected population, the point estimate was computed
from expected-spike-weighted sums over all contributing unit-image-trajectory
rows. Uncertainty was estimated by resampling images with replacement. For
each bootstrap sample, the numerator and denominator of the moving SSI and the
unit-matched stabilized SSI were recomputed from the resampled image totals,
and the percent change was recomputed from those ratios. Unless otherwise
specified, model SSI confidence intervals in Panels B, D, and E used 10,000
image bootstrap resamples. P-values for displayed across-versus-along
contrasts were computed from the bootstrap distribution of the paired residual
difference between the two contour-relative component groupings. Panel F used
hierarchical session/trial bootstrap confidence intervals with the two fixed
animals equally weighted, Panel G used paired-window bootstrap confidence
intervals for behavior-model predictions, and Panel H used regression
confidence intervals as described below.

### Real FEM anisotropy around local contours

To quantify the association between natural eye-position geometry and local
image structure, we analyzed 11,749 reviewed BackImage fixation windows from
1,962 trials and 30 recording sessions (Allen, 14 sessions; Logan, 16
sessions). Windows were retained after requiring valid gaze samples, valid
local image features, and the patch-contamination criteria described above,
and came from the mid- and late-fixation phases of the BackImage condition.
For each window, the local contour axis was measured at the mean gaze position
and the eye-position cloud was centered. Let \(\Sigma\) denote the resulting
two-dimensional position covariance and \(\mathbf{u}_\theta\) the unit vector
at relative angle \(\theta\) from the local contour. Projected position spread
was calculated in arcmin as

\[
s(\theta) = 60\sqrt{\mathbf{u}_\theta^T\Sigma\mathbf{u}_\theta}.
\]

Profiles were evaluated from 0 to 180 degrees in 3.75-degree increments, where
0 and 180 degrees are contour-parallel and 90 degrees is contour-orthogonal.
Individual windows were assigned directly to three local edge-coherence bands:
0-0.2, 0.2-0.5, and 0.5-1.0. Within each trial and coherence band, the median
profile was taken across contributing windows. Trial profiles were then
aggregated by taking the median across trials within each session and the
median across sessions within each animal. The displayed population profile
is the arithmetic mean of the Allen and Logan profiles, giving the two fixed
animals equal weight.

Confidence intervals were computed from 1,000 hierarchical bootstrap draws.
Sessions were resampled with replacement separately within each animal, and
trials were resampled with replacement within each selected session. The full
aggregation procedure was recomputed for every draw, and Allen and Logan were
then averaged equally. Shading shows the 2.5th and 97.5th percentiles. These
intervals quantify session- and trial-level uncertainty for the two fixed
animals and are not an animal-population inference. For the reported
parallel-minus-orthogonal contrast, parallel spread was the mean of the 0- and
180-degree profile values, and the contrast was recomputed within every
bootstrap draw before its interval was calculated.

### Behavior-model bridge and random-rotation control

The behavior-model bridge asked whether the contour-relative structure of real
FEMs predicted higher values on the same model SSI curves measured in the
movie bank. For each reviewed BackImage fixation window, we extracted the central
40-sample snippet, whose sample centers span 0.325 s, and projected the
centered eye positions onto the local contour-parallel and contour-normal
axes. We then converted the observed component RMS values into predicted SSI
changes by piecewise-linear interpolation through the model dose curves.
Predictions outside the model curve range were marked invalid and excluded
from the corresponding summaries.

To construct the random-rotation null, each behavior snippet was rotated by an
independent angle drawn uniformly from [0, pi), while keeping the same local
image coherence bin and the same eye-position cloud. This preserves the
movement amplitude and temporal structure of each real trace but breaks the
specific alignment between that trace and the local image contour. We generated
256 random rotations per behavior window. The plotted match advantage is the
observed prediction minus the mean random-rotation prediction, in percentage
points of SSI change relative to the model unit baseline. The displayed panel
uses the component-mean marginal prediction for RMS excursion, defined as the
average of the contour-normal and contour-parallel one-dimensional marginal
predictions. This is not a full two-dimensional SSI surface; it is a compact
summary of how the observed contour-relative distribution samples the two
model dose axes. Confidence intervals were computed by bootstrap resampling of
paired window predictions, and open markers indicate bins in which the 95%
confidence interval included zero.

### Patch-radius sensitivity of local contour measurements

Finally, we tested whether the relationship between local image coherence and
edge-following behavior depended on the spatial scale used to define local
image structure. We recomputed Sobel structure-tensor edge axes and coherence
values for gaze-centered patches with radii from 0.25 to 3.0 degrees. To make
this sweep efficient, full-image gradient fields were computed once per trial,
and gradient products were averaged over each gaze-centered patch with
integral images. The same image-contamination criteria were applied at every
radius.

For each window and radius, we estimated the principal axis of the
eye-position covariance and compared it with the local edge axis. The
edge-following alignment index was

\[
\cos(2\Delta\theta),
\]

where \(\Delta\theta\) is the circular axis-angle difference between the drift
axis and the local edge axis. Values near 1 indicate motion parallel to the
local contour, values near -1 indicate motion normal to the contour, and
values near 0 indicate no consistent axis relationship. For each patch radius,
we fit a window-level ordinary least-squares regression of this alignment
index against local orientation coherence, restricted to windows with
coherence > 0.3. Panel H plots the fitted slope as a function of patch radius.
Confidence intervals are the 95% intervals from the regression standard error
using the appropriate Student-t critical value.
