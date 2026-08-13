# Figure 4 Panel G: Interpolation Bridge Audit and Replacement Plan

Status: methodological audit and proposed validation sequence, 2026-08-08.

This note concerns the **displayed Figure 4 Panel G**, titled:

```text
Contour-matched FEMs beat rotations for aligned high-SF units
```

The source module is historically named `panel_j_match_advantage.py`. This
panel must not be confused with displayed Panel E, the directly evaluated
model dose-response panel titled `Across-contour spread limits high-SF
benefit`.

## Executive assessment

Panel G is a meaningful exploratory **behavior-to-model prediction**, provided
that contour-parallel and contour-normal RMS are approximately sufficient and
separable summaries of the model response. It is not an exact counterfactual
model evaluation of each recorded image/trajectory pair. The current title and
y-axis omit that distinction and therefore state the result too strongly.

The best current interpretation is:

```text
Separately computed model dose-response curves predict that the recorded
contour-relative RMS geometry is more favorable than uniformly rotated
versions of the same trajectories for aligned high-SF units.
```

The panel does **not** currently establish:

```text
For the exact local image content, units, and measured trajectory, directly
computed SSI is higher than for counterfactual rotations or nonlocal image
content.
```

Until the latter is evaluated directly, Panel G should either be labeled as a
prediction or withheld from a claim about image-specific trajectory matching.

## What the current panel actually computes

The calculation has two stages.

### Stage 1: directly evaluated model dose curves

The model trace bank is evaluated directly across images, traces, and units.
For each trace, the central model-duration trajectory is decomposed relative to
the image edge axis into contour-parallel and contour-normal components. Model
outputs are then binned separately by component RMS. Within each bin the code
accumulates:

- information numerator;
- expected spikes;
- population SSI in bits/spike;
- SSI change relative to the cell-matched stabilized baseline;
- image-bootstrap uncertainty.

This is the source of displayed Panel E. Within the provenance of that cache,
the plotted dose curves are directly derived from model outputs rather than
predicted from behavior.

### Stage 2: behavioral doses interpolated onto those curves

For every reviewed behavioral window, the bridge:

1. takes the central 40 native samples (0.325 s; it does not compress all 128
   samples into 40 timepoints);
2. projects the trace parallel and normal to the measured local edge axis;
3. calculates parallel and normal RMS doses;
4. generates 256 uniform axial rotations of that same trace geometry;
5. recalculates the two component doses for every rotation;
6. interpolates each dose onto its corresponding one-dimensional model curve;
7. averages the two marginal predictions into a
   `component_mean_marginal` score;
8. plots the paired observed-minus-mean-rotation prediction by local edge
   coherence.

Thus the word `SSI` in Panel G refers to a behavior-weighted prediction from
precomputed SSI curves. No new activation maps, rates, spike counts, or SSI
values are evaluated for the exact behavioral image/trajectory/rotation
combinations.

## The strongest case for retaining the panel

The current construction is not arbitrary. Its strongest defense has five
parts.

### 1. It is a calibrated forward prediction

The mapping from component dose to population SSI comes from separately
evaluated model responses, not from fitting SSI to the same coherence-binned
behavioral outcome shown in Panel G. Panel G asks where measured behavior lands
on an already constructed model sensitivity curve.

### 2. The rotation null preserves much of the measured trajectory

A rotation retains the trace's temporal order, total radial geometry,
reversals, pauses, and sample-to-sample idiosyncrasies. It primarily changes
how the same trace decomposes relative to the local contour. This is a more
specific geometry control than substituting a generic synthetic trajectory.

### 3. Interpolation error is partly paired

The same model curves and interpolation rule are applied to the observed and
rotated versions of a window. If approximation error depends mainly on dose,
not condition identity, part of that error should cancel in the paired
contrast. Approximately 90% of component predictions are inside the modeled
dose range in the current coherence analysis.

### 4. The observed pattern is structured rather than merely positive

For the aligned high-SF population and the RMS metric, the current predicted
advantage grows with local edge coherence:

| local edge coherence | predicted real-minus-rotation advantage (pp SSI) | flat-window 95% CI |
|---|---:|---:|
| 0-0.2 | -0.001 | [-0.039, +0.035] |
| 0.2-0.5 | +0.034 | [+0.004, +0.065] |
| 0.5-0.8 | +0.062 | [+0.014, +0.111] |
| 0.8-1 | +0.155 | [+0.044, +0.265] |

The near-zero low-coherence result and monotonic increase are qualitatively
consistent with a contour-dependent mechanism rather than an undifferentiated
benefit of any rotation convention.

### 5. Population averaging can isolate general geometry

An image-averaged dose curve suppresses idiosyncratic texture effects. If the
scientific question is deliberately about the expected consequence of
contour-relative trajectory geometry, this can be a feature: the panel asks
whether behavior occupies model-favorable geometric doses on average, not
whether one particular texture produces an unusually favorable response.

Under the assumptions below, the panel is therefore a useful surrogate:

```text
SSI response is approximately determined by parallel and normal RMS;
the two component effects are approximately separable;
the model dose curves transfer to the behavioral windows;
and interpolation error is not condition-dependent.
```

## Problems and unsupported assumptions

### 1. The counterfactual is not evaluated by the model

The primary limitation is categorical, not cosmetic. The network never sees
the exact behavioral image with the exact real and rotated trajectories. The
analysis therefore cannot capture image-specific interactions, spatial phase,
local contrast, boundary crossings, nonlinear temporal integration, or
trajectory-dependent activation suppression specific to that pairing.

### 2. Two one-dimensional marginals are not a joint response surface

The bridge predicts parallel and normal components independently and then
averages their predicted SSI residuals. SSI is not additive across projected
motion components, and equal weighting has no direct probabilistic
interpretation. A trace with moderate dose in both components need not equal
the average of two traces in which one component is varied while the other is
implicitly marginalized over the trace bank.

At minimum, a surrogate should use and validate a joint surface:

```text
f(normal RMS, parallel RMS)
```

with the remaining trace geometry either controlled or explicitly modeled.

### 3. RMS may not be a sufficient behavioral or model variable

RMS discards temporal order, velocity, lag structure, reversals, net
displacement, projected range, and path tortuosity. Checkpoint 1 showed that
short-lag displacement alignment can differ from the 128-sample covariance
axis. Two traces with equal component RMS can therefore generate different
instantaneous activation maps, rates, and information.

### 4. The current rotation null does not test local image specificity

Uniform rotation shows that orientation relative to a contour matters to the
surrogate. It does not show that the animal selected its trajectory for that
particular local contour. A shared egocentric trajectory prior and a shared
absolute image-axis prior can produce the same observed-versus-rotation
contrast.

The code also computes an image-axis scramble, but that result is not displayed
in Figure 4. For aligned high-SF RMS, the current observed-minus-scrambled
confidence intervals include zero in every coherence bin. That scramble is not
the final answer—it is not adequately matched within session, phase, movement
scale, or image—but its failure means the present evidence is specific to the
rotation null, not to local image/trajectory pairing.

### 5. Metric selection was informed by the result

RMS was promoted after comparing path, RMS, range, and tortuosity-style axes;
the source comments explicitly cite the stronger RMS rotation-null result as
the reason for the switch. This is legitimate exploratory model development,
but the displayed confidence intervals and permutation p-values do not account
for selecting the metric after viewing those alternatives. A confirmatory
version must pre-specify the metric or validate it on held-out images,
sessions, or animals while reporting the full metric family.

### 6. The uncertainty unit is too optimistic for overlapping windows

The plotted confidence intervals use a flat bootstrap that treats fixation
windows as exchangeable. The production behavioral extraction uses
128-sample windows with 16-sample stride, so adjacent rows overlap by 87.5%.
They are not independent observations. The permutation p-value uses a different
session-aggregated construction, meaning the plotted CI and p-value target
different sampling hierarchies.

A production analysis should use non-overlapping fixation-level summaries or a
hierarchical resampling scheme over animal, session, trial/fixation, and then
window where appropriate.

### 7. Common support and missingness complicate the summary

About 10% of component doses are outside the modeled x-range and are set to
missing rather than extrapolated. The separately reported observed and rotated
means are calculated over their respective finite cases, while the displayed
observed-minus-rotated value uses a paired finite subset. Consequently:

```text
reported observed mean - reported rotated mean
```

does not generally equal:

```text
reported paired observed-minus-rotated mean
```

The paired contrast is the more defensible object, but all conditions should be
restricted to a common-support set and the exclusion pattern should be shown.

### 8. Normalized SSI can hide rate effects

The figure displays percentage-point SSI residuals without adjacent rates,
expected spikes, or information-per-sample values for the behavior-weighted
conditions. A condition can appear favorable in bits/spike while suppressing
the number of spikes carrying that information. Exact counterfactual
evaluations must preserve both normalized and absolute response quantities.

### 9. The title obscures the evidential level

`Contour-matched FEMs beat rotations` reads as a direct model result. The
actual computation is closer to `model dose curves predict an advantage for
the observed contour-relative RMS decomposition`. This distinction belongs in
the panel itself, not only in provenance.

## A more principled staged approach

The replacement should separate three questions:

1. Is the behavioral trajectory locally related to image content?
2. Does that exact relationship change neural-network response maps and SSI?
3. Can a low-dimensional dose surrogate accurately summarize the direct
   counterfactual result?

### Stage A: run the same-image 5-deg behavioral control first

For each non-overlapping selected window:

- keep the measured trajectory fixed;
- measure the edge axis at the true gaze-centered patch;
- measure patches at the pre-specified eight directions on a 5-deg annulus in
  the same original image;
- retain every complete, uncontaminated offset patch;
- apply the same uniform rotation draws to the trajectory for local and offset
  patches;
- retain every offset separately and average all valid directions for the
  primary per-window offset value;
- label orientation-preserving offsets (axial difference <=10 deg) and
  orientation-changing offsets (>=30 deg) as diagnostics, not primary filters.

For a behavior score `S`, calculate:

```text
D_local    = S(local patch, real trace)
             - mean_rotation S(local patch, rotated trace)

D_offset   = mean_valid_offsets [
               S(offset patch, real trace)
               - mean_rotation S(offset patch, rotated trace)
             ]

D_locality = D_local - D_offset
```

The same rotation angles must be reused for local and offset patches so the
difference-in-differences is paired. This stage does not require the neural
model and should be understood before model evaluation begins.

### Stage B: directly evaluate selected image/trace/unit examples

Start with targeted renders, not a population curve. For a small auditable set
of positive, dissociation, and control windows, construct:

1. local image content + recorded trajectory;
2. local image content + several rotations of that same trajectory;
3. same-image 5-deg offset content + the recorded trajectory;
4. the same offset content + the identical rotation draws.

Use the central 40 native samples required by the model. Do not time-compress a
128-sample trace into 40 samples. Retain the original image, patch center,
trajectory center, coordinate transform, and every offset direction in the
manifest.

Select unit roles before viewing the counterfactual outcome:

- independently tuned high-SF unit aligned with the **local** contour;
- high-SF orthogonal control;
- low-SF control;
- optional oblique or off-axis dissociation.

For the primary causal comparison, freeze the same units across local, offset,
real, and rotated conditions. Do not reselect the population after the offset
axis changes. A secondary analysis may describe units aligned to each offset
axis, but it answers a different composition question.

For each unit, condition, and timepoint, save:

- the raw activation map on a matched spatial extent and color scale;
- real-minus-rotation and local-minus-offset difference maps;
- mean rate and expected spikes;
- information numerator and information per sample;
- instantaneous SSI and its expected-spike-weighted trajectory summary;
- a separately labeled SSI of the trajectory-averaged map, if computed at all.

The map sheet is the next human checkpoint. The key question is whether the
direct SSI differences correspond to visible, temporally coherent response-map
changes rather than normalization or rate suppression.

### Stage C: validate any surrogate against direct evaluations

If the exact example-level model result is interpretable, build a held-out
validation set spanning parallel and normal RMS jointly. Fit or tabulate:

```text
direct SSI = f(normal RMS, parallel RMS)
```

Then test, on held-out image/trace/unit combinations:

- prediction error and calibration;
- whether residual error depends on image coherence, contrast, spatial
  frequency, trace curvature, range, or temporal structure;
- whether predicted condition differences agree in sign and magnitude with
  directly evaluated differences;
- whether the one-dimensional component-mean surrogate adds material error
  relative to the two-dimensional surface.

Only a surrogate that predicts direct counterfactual differences on held-out
conditions should be promoted to a scalable population analysis.

### Stage D: production direct-counterfactual summary

After the example maps pass inspection, evaluate a pre-specified,
non-overlapping set of windows across both animals. Use common random rotations
and the same valid offset directions for behavior and model comparisons.

Primary direct model contrasts should include:

```text
A_rotation = SSI(local, real) - mean_rotation SSI(local, rotated)

A_offset   = mean_offsets [
               SSI(offset, real) - mean_rotation SSI(offset, rotated)
             ]

A_locality = A_rotation - A_offset
```

Repeat these contrasts for information per sample, expected spikes, and rate.
Use absolute differences as primary outcomes; retain bits/spike but do not use
a percentage or ratio when the baseline can approach zero.

Inference should resample the experimental hierarchy:

```text
animal -> session -> trial/fixation
```

and report animal-specific results, leave-one-session-out sensitivity,
leave-one-image-out sensitivity, common-support exclusions, and
orientation-preserving versus orientation-changing offset diagnostics.

### Stage E: decide the fate of Panel G

Possible outcomes and figure consequences are:

| result | defensible panel |
|---|---|
| direct local-real SSI exceeds rotations and same-image offsets | direct counterfactual local-matching panel |
| direct results agree with a validated held-out surrogate | explicitly labeled predicted-SSI panel may be retained |
| rotation advantage survives but locality does not | trajectory-geometry/model-sensitivity panel, not local control |
| behavior locality exists but direct SSI does not change | behavioral adaptation panel without neural-benefit claim |
| neither behavior nor direct SSI supports locality | remove matching claim; retain descriptive shared-axis results |

## Immediate figure-language recommendation

If the current panel remains temporarily, change its title and axis to:

```text
Title: Model curves predict a benefit of recorded contour-relative RMS
Y-axis: predicted SSI advantage vs rotations (pp)
```

The caption should state that the result averages two one-dimensional marginal
interpolations and is exploratory. The rotation result must not be described as
evidence for image-specific control unless it exceeds a matched local-pairing
or same-image spatial-offset null.

## Provenance pointers

Current displayed matching panel:

```text
declan/fig/ssi_figure_v2/panels/panel_j_match_advantage.py
```

Behavior-to-model interpolation and rotation/scramble summaries:

```text
declan/fig/ssi_figure_v2/behavior_model_bridge/
  run_behavior_model_bridge.py
  run_random_rotation_match_null.py
  run_random_rotation_prediction_by_coherence.py
```

Directly evaluated model dose curves:

```text
declan/fig/ssi_figure_v2/panels/
  panel_g_alternative_x_axes_diagnostic.py
  panel_g_rms_excursion.py
```

Behavioral confound and same-image offset plan:

```text
declan/fig/ssi_figure_v2/
  figure4_behavior_confounds_map_first_debugging_plan.md
```

## Checkpoint 2B first attempt: selected-window same-image offsets

The pre-specified selected-window diagnostic was completed on 2026-08-08 with:

```text
declan/fig/ssi_figure_v2/behavior_confounds/
  build_checkpoint2b_local_offset_examples.py
outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1/
  checkpoint2b_local_vs_offset_patch_examples.png
  checkpoint2b_offset_patch_manifest.csv
  checkpoint2b_local_vs_offset_patch_values.csv
  checkpoint2b_local_offset_axis_relationship.csv
  checkpoint2b_locality_summary.csv
  checkpoint2b_run_metadata.json
```

This is a targeted map-first checkpoint, not a population estimate and not a
direct neural-model evaluation. It retained 46 of 48 pre-specified 5-deg
offsets; two failed the existing background-contamination threshold. The local
feature recomputation reproduced every stored local edge axis to numerical
precision. Both RMS components were inside the modeled interpolation support
for every retained local/offset condition and every rotation draw.

The visible result is heterogeneous. The oblique-local-positive example had a
large local-minus-offset surrogate contrast (`+2.351 pp`), but the
shared-horizontal-positive and image-dominant-dissociation scenes retained
nearly the same edge axis around most or all of the annulus and had locality
contrasts near zero (`-0.048 pp` and `-0.004 pp`). The latter cases are support
failures for an orientation-disrupting offset control rather than evidence for
or against local matching. The motor-prior dissociation was also near zero in
the surrogate (`+0.054 pp`).

The controls exposed two additional caveats. The low-coherence example showed
a positive apparent surrogate locality contrast (`+0.694 pp`) even though its
local axis is unreliable (`coherence = 0.048`). The low-anisotropy trace had a
near-maximal raw cos2 alignment value despite its unstable cloud axis, while
its surrogate locality contrast was near zero (`+0.009 pp`). Therefore, raw
axis agreement must be gated or weighted by both image coherence and trace
anisotropy; it cannot serve as an unqualified behavioral locality endpoint.

This interpolation-based attempt remains below Gate A. Before a direct neural
model render, the behavior endpoint itself must be expressed in absolute units
that remain meaningful for low-anisotropy trajectories.

### Checkpoint 2B behavior-only absolute-RMS correction

The reusable parts of the first attempt were retained: original-BackImage
sampling, the pre-specified 5-deg annulus, the 1-deg feature radius, patch QC,
the shared per-window rotations, the manifest, and the auditable rules for
displaying axis-preserving and axis-changing examples. The `cos(2 delta)`
behavior endpoint and the Panel-G interpolation were removed from the corrected
sheet.

The corrected implementation and artifacts are:

```text
declan/fig/ssi_figure_v2/behavior_confounds/
  build_checkpoint2b_behavior_absolute_rms.py
outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1/
  checkpoint2b_behavior_absolute_rms_v1/
    checkpoint2b_behavior_absolute_rms_examples.png
    checkpoint2b_behavior_absolute_rms_values.csv
    checkpoint2b_behavior_absolute_rms_rotation_values.csv
    checkpoint2b_behavior_absolute_rms_offset_manifest.csv
    checkpoint2b_behavior_absolute_rms_display_selection.csv
    checkpoint2b_behavior_absolute_rms_axis_relationship_summary.csv
    checkpoint2b_behavior_absolute_rms_locality_summary.csv
    checkpoint2b_behavior_absolute_rms_run_metadata.json
```

For each patch axis, the behavior score is the full-window absolute positional
spread difference

```text
S = RMS_parallel - RMS_normal  [arcmin]
D_location = S_measured - mean_rotation(S_rotated)
D_locality = D_local - mean_valid_offsets(D_offset)
```

The same measured 128-sample trace and the same 256 rotation angles are used at
the local patch and every valid offset in a window. No SSI interpolation or
neural-model response enters this calculation. The full per-rotation table is
saved so every null mean and interval in the sheet can be reconstructed.

The six selected-window `D_locality` values were:

| role | `D_locality` (arcmin) | visible interpretation |
|---|---:|---|
| shared horizontal positive | +0.228 | real-versus-rotation advantage persists around the annulus; no offset reaches the >=30-deg axis-change criterion |
| oblique local positive | +5.216 | strong local advantage; orientation-changing offsets reverse the sign |
| motor-prior dissociation | -2.410 | the local trajectory is more normal- than parallel-spread; changing the remote axis can reverse the sign |
| image-dominant dissociation | +0.005 | essentially identical advantage at local and offset patches; every retained offset preserves the image axis |
| low-coherence control | +2.976 | apparent locality despite local coherence 0.048, so the local axis is not a reliable endpoint |
| low-anisotropy control | +0.004 | absolute effect is essentially zero, resolving the misleading near-maximal raw `cos2` value |

These are example-level observations, not a population estimate. The sheet
therefore stops at the map-first human checkpoint. The key questions are
whether the oblique example is a credible positive, whether the
low-coherence false positive is sufficiently disqualifying, and whether the
shared-horizontal/image-dominant rows should be read as extended/global scene
structure or simply as failures to obtain an orientation-disrupting offset.

## Direct exact-pair SSI checkpoint

The targeted direct evaluation was completed on 2026-08-08 with:

```text
declan/fig/ssi_figure_v2/behavior_model_bridge/
  run_direct_exact_pair_ssi.py
outputs/fig/ssi_figure_v2/behavior_model_bridge/
  panel_g_direct_exact_pair_ssi_targeted_v1/
    direct_exact_pair_selected_unit_maps.png
    direct_exact_pair_selected_unit_maps_four_frames.pdf
    direct_location_rotation_contrasts.csv
    direct_locality_summary.csv
    direct_population_metrics.csv
    direct_unit_metrics.csv
    direct_selected_example_units.csv
    direct_unit_selection.csv
    run_metadata.json
```

This run contains no dose-curve interpolation. It freshly rendered and scored
52 retained image patches crossed with the recorded trajectory and eight
deterministic full-circle rotations, for 468 exact model movies. Each trace was
the native central 40 samples, mean centered without temporal compression.
Rotations used a full-circle midpoint grid with antipodal pairs and were
applied around the trace centroid. Unit sets were selected from the local
contour before viewing any counterfactual response and then frozen across all
offsets and rotations.

For the locally aligned high-SF population, the selected-example direct
results were:

| role | direct local real-minus-rotation SSI (bits/spike) | direct locality SSI (bits/spike) | local information/sample difference | local expected-spikes/sample difference |
|---|---:|---:|---:|---:|
| shared-horizontal positive | +0.001788 | +0.002904 | +0.000022 | +0.000368 |
| oblique-local positive | +0.111633 | +0.084663 | +0.000185 | -0.000788 |
| motor-prior dissociation | +0.064012 | +0.020815 | +0.000068 | -0.000069 |
| image-dominant dissociation | +0.000950 | -0.000879 | -0.000007 | -0.000065 |
| low-coherence control | +0.000751 | -0.001709 | +0.000006 | +0.000026 |
| low-anisotropy control | +0.001407 | +0.000903 | +0.000009 | +0.000103 |

Here `locality` is the local real-minus-rotation contrast minus the mean of the
same contrast over every valid pre-specified offset. These are targeted
examples and have no population confidence interval.

The strongest direct result is the independently selected oblique positive.
Its local aligned-population SSI exceeded every one of the eight directly
evaluated rotations. The selected aligned unit's framewise SSI advantage was
positive at 39 of 40 frames, and the raw map sheets show sustained vertical
response concentrations in the recorded condition that are weakened in the
rotation-mean maps. Information per sample increased even while expected
spikes decreased, so this example is not a bits/spike increase produced only
by spike suppression.

The motor-prior dissociation is a second visible positive, but it is less
rotation-general: the recorded aligned-population SSI exceeded six of eight
rotations, and the selected aligned unit's framewise advantage was positive at
32 of 40 frames. The horizontally shared example had only a small population
effect and its selected unit was positive at 14 of 40 frames. The
image-dominant, low-coherence, and low-anisotropy locality contrasts were near
zero or negative at the scale of the two positive examples.

The direct results also show why the marginal bridge cannot substitute for
fresh evaluation. The shared-horizontal and motor-prior examples had negative
surrogate local predictions but positive direct local SSI differences. The
low-coherence control had a strongly positive surrogate locality prediction
but a slightly negative direct locality result. The image-dominant surrogate
predicted a large local benefit while the direct bits/spike effect was small
and direct information per sample decreased. Surrogate and direct units are
not numerically comparable, but these sign and ordering failures are enough to
reject the one-dimensional interpolation as an example-level estimator.

The displayed rotation-mean activation map is a visualization average of
fresh response maps. Its annotated instantaneous map SSI is a diagnostic; all
reported trajectory SSI contrasts use the mean of the separately calculated
condition-wise direct SSI values, not SSI computed from the averaged map.

This checkpoint establishes that direct exact-pair effects can be large and
map-visible in selected positive examples. It does not yet establish a
population Panel G. Before expanding, the next human decision is whether the
oblique and motor-prior map changes are the intended notion of sharpening and
whether the eight-angle rotation grid is sufficiently converged. A small
4/8/16/32-angle convergence rerun on these two examples is the smallest useful
technical follow-up.

## Direct rotation-grid convergence checkpoint

The proposed 4/8/16/32-angle check was completed on 2026-08-09 for the local
patch and preselected orientation-changing representative offset in the
oblique-local-positive and motor-prior-dissociation examples. The 4-, 16-, and
32-angle conditions were fresh model evaluations; the 8-angle values were
joined from the prior direct exact-pair run for the identical patch IDs. The
implementation and auditable outputs are:

```text
declan/fig/ssi_figure_v2/behavior_model_bridge/
  run_direct_exact_pair_ssi.py
  summarize_direct_rotation_convergence.py
outputs/fig/ssi_figure_v2/behavior_model_bridge/
  panel_g_direct_rotation_convergence/
    direct_rotation_grid_convergence.png
    direct_rotation_grid_convergence.pdf
    direct_rotation_angle_curves_k32.png
    direct_rotation_angle_curves_k32.pdf
    direct_rotation_convergence_values.csv
    direct_rotation_convergence_summary.csv
    direct_rotation_angular_exceedance_k32.csv
    convergence_metadata.json
    k04/
    k16/
    k32/
```

For the locally aligned high-SF population, the exact real-minus-rotation-mean
SSI values were:

| role and effect | 4 angles | 8 angles | 16 angles | 32 angles |
|---|---:|---:|---:|---:|
| oblique local | 0.136589 | 0.111633 | 0.111120 | 0.111338 |
| oblique representative offset | 0.030471 | 0.026559 | 0.028140 | 0.027455 |
| oblique local minus representative offset | 0.106118 | 0.085074 | 0.082980 | 0.083883 |
| motor-prior local | 0.067945 | 0.064012 | 0.064052 | 0.064717 |
| motor-prior representative offset | 0.020712 | 0.024459 | 0.022874 | 0.023824 |
| motor-prior local minus representative offset | 0.047233 | 0.039553 | 0.041178 | 0.040893 |

All entries are bits/spike. Against the predeclared adequacy rule—absolute
error from the 32-angle value no larger than the greater of 0.005 bits/spike
or 5% of the 32-angle magnitude—the 8-angle grid passed for all six effects.
Its absolute errors were 0.000295 and 0.000705 for the two local effects and
0.001192 and 0.001340 for the two representative-locality contrasts. The
4-angle grid failed for oblique local SSI and both locality contrasts. Eight
angles are therefore adequate for estimating the rotation mean in these two
examples; four are not.

The dense angular curves also narrow the correct scientific claim. Although
the oblique real condition exceeded all eight rotations in the original grid,
four of 32 dense-grid rotations had SSI at or above the real condition. The
real condition exceeded 28/32 rotations for oblique local, 30/32 for its
representative offset, 24/32 for motor-prior local, and 22/32 for its
representative offset. The relevant estimand is consequently the direct
real-minus-rotation **mean**, not a claim that the recorded image orientation
is the unique or global SSI maximum. The 32-angle curves are smooth and
approximately antipodally structured, which explains both the poor 4-angle
quadrature and the stability from 8 through 32 angles.

This validates an eight-angle deterministic full-circle rotation mean for the
next direct Panel G expansion on these exemplars. It does not validate every
new image patch automatically: the production analysis should retain exact
per-pair angle values so convergence or angular multimodality can be audited,
and any population claim should remain traceable to patch-level direct
contrasts rather than an interpolated dose curve.

## Prepared full exact-pair production cohort

The earlier Figure 4 neural-model bank used 100 selected image windows crossed
with 1,000 independently selected fixation traces. Repeating that 100 x 1,000
Cartesian product would not answer the direct-pair question because almost all
movies pair an image with another window's trajectory. The compatible reuse is
therefore the same 1,000 Figure 4 fixation traces, each restored to its own
reviewed source image window and local contour axis.

The production runner and runbook are:

```text
declan/fig/ssi_figure_v2/behavior_model_bridge/
  run_panel_g_exact_pair_production.py
  merge_panel_g_exact_pair_production.py
  panel_g_exact_pair_production_runbook.md
outputs/fig/ssi_figure_v2/behavior_model_bridge/
  panel_g_exact_pair_fig4_trace_bank_n1000_v1/
    exact_pair_cohort_manifest.csv
    exact_pair_unit_population_membership.csv
    shards/pairs_000000_001000/run_plan.json
```

The preflight resolves 1,000 unique native pairs from all 30 sessions: 621
Logan and 379 Allen windows, 443 mid-fixation and 557 late-fixation windows,
including the original 200 microsaccade traces. All pairs pass the saved image
QC, have finite local axes, and retain at least one predeclared aligned high-SF
unit. Every saved trace matches a fresh native central-40-sample reconstruction
exactly (maximum absolute discrepancy zero).

With eight validated rotations, the run comprises 9,000 fresh model movies.
The runner caches and identity-checks every pair independently, supports
disjoint pair-index shards, and writes raw unit, population-condition, and
pair-contrast outputs. A four-pair timing smoke completed 36 movies in 1.2
minutes after model initialization, giving an operational estimate of roughly
five GPU-hours on one GPU. The runbook specifies four 250-pair shards and a
merge that rejects missing or overlapping shards by default.

The timing smoke is an implementation check, not a scientific sample. Its
first four aligned-population contrasts happened to be small and negative;
that is a useful warning against treating the selected positive examples as a
population forecast. The full production run has been prepared but not
launched.

## Full exact-pair production Checkpoint 1

The GPU-0 production run completed all 1,000 native pairs and 9,000 fresh
movies in 102.4 minutes. The first descriptive, example-traceable readout is:

```text
declan/fig/ssi_figure_v2/behavior_model_bridge/
  analyze_panel_g_exact_pair_checkpoint1.py
outputs/fig/ssi_figure_v2/behavior_model_bridge/
  panel_g_exact_pair_fig4_trace_bank_n1000_v1/checkpoint1_production_readout/
    checkpoint1_production_overview.png
    checkpoint1_production_overview.pdf
    checkpoint1_selected_pair_inputs_and_angle_curves.png
    checkpoint1_selected_pair_inputs_and_angle_curves.pdf
    checkpoint1_aligned_pair_metrics.csv
    checkpoint1_selected_pairs.csv
    checkpoint1_metadata.json
```

For locally aligned high-SF units, the direct real-minus-eight-rotation-mean
effect had mean -0.000460 bits/spike, median +0.000028 bits/spike, and 50.5% of
pairs above zero. Its Pearson and Spearman correlations with local orientation
coherence were -0.019 and -0.020. Thus the full paired cohort does not show a
broad coherence-linked SSI advantage of the kind implied by the interpolated
bridge. All-high-SF, orthogonal-high-SF, and low-SF population means were also
near zero at the population scale.

The 0.5-0.8 coherence bin had a small positive aligned mean (+0.000481) and
median (+0.000262) across 181 pairs, but the coherence >=0.8 tail contained
only 22 pairs and had a positive median (+0.000627) with a negative mean
(-0.001253), showing sensitivity to tail cases. The continuous coherence
correlation remains the more direct warning against a monotonic population
claim.

Bits/spike and information/sample effects disagreed in sign for 20.3% of
pairs. The largest information-supported positive was pair 32 (+0.050070
bits/spike, +0.000044 information bits/sample); a nearly equal bits/spike gain
in pair 13 (+0.050034) accompanied an information loss (-0.000083 bits/sample)
and was frozen as a normalization dissociation. Pair 940 was the strongest
information-supported negative (-0.062977 bits/spike, -0.000225 bits/sample).

Direct SSI effect had a weak positive relationship with the pair's
parallel-minus-normal positional RMS (Pearson 0.173, Spearman 0.190). The
Spearman value remained 0.190 in the 800 drift-only traces, but their mean
effect was negative (-0.000803 bits/spike); the 200 microsaccade traces had a
positive mean (+0.000914). Pair 920, selected for the largest positive-aligned
versus negative-orthogonal population dissociation, visibly contains an
exceptionally large parallel event and has +37.4 arcmin parallel-minus-normal
RMS. This is a useful mechanistic example but not evidence for a typical drift
effect.

Six pair roles were frozen before targeted map inspection: information-
supported positive, information-supported negative, normalization
dissociation, aligned-versus-orthogonal population dissociation,
high-coherence near-null control, and high-coherence positive. Checkpoint 1
shows their native patches, contour-coordinate trajectories, and all eight
fresh angle evaluations. It stops before a targeted activation-map rerender or
session-aware population inference.

## Contour-axis overlay audit

Human inspection of the Checkpoint-1 selected-pair sheet identified an axis
mismatch. The original plotting code placed gaze-coordinate axes and traces
(+y upward) directly on image-array coordinates (+row downward). This
vertically reflected every non-horizontal contour arrow and the trace overlay.
The bug was confined to `analyze_panel_g_exact_pair_checkpoint1.py`'s image
overlay; contour-coordinate trajectory projections, pair-specific unit masks,
and model conditions all remained in the internally consistent gaze frame.
The overlay now reflects y when entering screen coordinates.

The corrected sheet and an explicit pixel audit are:

```text
checkpoint1_selected_pair_inputs_and_angle_curves.{png,pdf}
checkpoint1_contour_axis_overlay_audit.{png,pdf,csv}
declan/fig/ssi_figure_v2/behavior_model_bridge/
  diagnose_panel_g_contour_axis_overlay.py
```

For every selected pair, a Sobel structure tensor recomputed from the exact
displayed 1-degree aperture reproduced the stored coherence, gradient axis,
and edge/tangent axis to numerical precision. The stored image-array edge axis
is exactly the vertical reflection of the stored gaze-coordinate edge axis,
and the gradient and edge axes are exactly orthogonal. Thus there is no stale
metadata, wrong source row, or patch-centering discrepancy.

The pixel audit nevertheless exposes a semantic QC problem. Most selected
axes plausibly follow visible structure after correcting the coordinate frame,
but pair 32's high-coherence tensor tangent does not describe the visually
salient truncated wire fragment. Its oriented gradient energy is confined
near the aperture boundary, so a high global structure-tensor coherence can
still yield a poor local-contour label. This matters scientifically because
aligned/orthogonal unit membership depends on the stored tangent even though
the direct model evaluations themselves do not.

The production coherence result should therefore remain paused. The smallest
next input-level checkpoint is a blind, coherence-stratified gallery of native
1-degree apertures with undirected tensor axes and a boundary-energy diagnostic.
That gallery should determine whether to add a central spatial weighting or an
explicit contour-validity gate before any targeted activation-map rerender or
session-aware population summary.

## Why the previous Panel G effect disappears under direct evaluation

The old-to-new change was decomposed by applying the historical aligned-high-SF
RMS surrogate to the exact 1,000 source rows used in the production run. The
historical 256 random angles were regenerated for their original full-table row
indices; a second surrogate calculation used the same eight fixed angles as the
direct run. Outputs and the reproducible comparison script are:

```text
declan/fig/ssi_figure_v2/behavior_model_bridge/
  compare_panel_g_surrogate_direct_same_pairs.py
outputs/fig/ssi_figure_v2/behavior_model_bridge/
  panel_g_exact_pair_fig4_trace_bank_n1000_v1/old_surrogate_vs_direct_audit/
    old_surrogate_vs_direct_decomposition.{png,pdf}
    old_surrogate_vs_direct_pair_table.csv
    old_surrogate_vs_direct_coherence_bins.csv
    old_surrogate_vs_direct_metrics.json
```

The decomposition rules out cohort change as the primary explanation. The old
surrogate retains its positive coherence-bin pattern on the exact-pair cohort:

| coherence bin | old surrogate, full 11,749-window cohort (pp) | old surrogate, exact-pair cohort (pp) | fresh direct effect (bits/spike) |
|---|---:|---:|---:|
| 0-0.2 | -0.001 | -0.008 | -0.000096 |
| 0.2-0.5 | +0.034 | +0.020 | -0.001040 |
| 0.5-0.8 | +0.062 | +0.272 | +0.000481 |
| 0.8-1 | +0.155 | +0.287 | -0.001253 |

The units in the two right-hand columns are deliberately not equated. The
surrogate is percentage change relative to its stabilized population baseline;
the direct estimand is absolute real-minus-rotation-mean bits/spike. The valid
comparisons are the coherence pattern, pair ordering, and sign—not magnitude.
On the identical pairs and identical eight rotations, surrogate versus direct
effects had Pearson 0.109, Spearman 0.105, and only 54.7% sign agreement among
finite pairs. Changing from 256 random rotations to the eight fixed rotations
did not explain the failure: the two surrogate versions had Spearman 0.968.

The old Panel G therefore did not estimate the counterfactual now of interest.
It treated pooled one-dimensional conditional dose curves as if they were
pair-specific causal response functions. For each fixation it reduced the
trajectory to contour-parallel and contour-normal RMS, interpolated those two
marginal curves independently, and averaged the predictions. It never passed
that image-trajectory pair—or its rotated twins—through the model. The
surrogate is consequently strongly coupled to the imposed low-dimensional
geometry: its Spearman correlation with parallel-minus-normal RMS was 0.514.
The fresh model effect had only 0.190 correlation with that same geometry.

This is an ecological transport failure. The pooled dose curve averages across
other images, trajectories, units, firing rates, and spatiotemporal response
histories. A real rotation on one fixed image changes the complete retinal
movie; its response depends on spatial phase, local image content, temporal
ordering, nonlinear activation, and rate normalization, none of which is
identified by the two marginal RMS coordinates. Averaging the marginal curves
also discards their joint response surface. The clean old trend was therefore a
real property of the surrogate construction, but it was not evidence that the
model itself gives each native coherent image-trajectory match higher SSI than
rotated versions.

The contour-axis semantic QC can attenuate or misassign the locally aligned
population, but it cannot account for this discrepancy by itself. The direct
all-high-SF population, which does not use the local axis for membership, also
had near-zero mean (+0.000054 bits/spike) and near-zero coherence association
(Spearman -0.008). Finally, the top-coherence direct bin has only 22 pairs and a
positive median despite a negative mean, so the current result does not prove
that no exact-pair effect exists. It shows that the previous broad, monotonic
Panel G claim is unsupported by the principled direct estimator and that the
old interpolation cannot be used as evidence for it.
