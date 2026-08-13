# Figure 4 Behavior Confounds: Map-First Debugging Plan

Status: active human-guided audit; selected-example Checkpoints 1-4 rendered,
2026-08-09.

Scope: current Figure 4 bottom-row behavior panels F-H in
`compose_ssi_figure_v4.py`:

- F: `Real FEM spread is contour-aligned`;
- G: `Contour-matched FEMs beat rotations for aligned high-SF units`;
- H: `Edge following saturates near foveal scale`.

The construction, strongest defense, limitations, and proposed direct-model
replacement for displayed Panel G are audited separately in:

```text
declan/fig/ssi_figure_v2/behavior_model_bridge/
  figure4_panel_g_interpolation_audit.md
```

This note records a claim-threatening reference-frame confound and translates
the follow-up into the repository's `map-first-analysis` workflow. It is a
debugging plan, not evidence that the proposed alternatives are true. Each
checkpoint should be rendered, reviewed, and interpreted before the next one
is run.

## Current Status Snapshot

1. **Reference-frame concern established.** Both image axes and position-cloud
   axes have shared screen-horizontal biases, and a within-session image-axis
   shuffle reproduces the aggregate raw alignment. Raw alignment alone does
   not establish local image/trajectory pairing.
2. **Selected local-pair examples are mixed.** Real-pair, rotation,
   reassignment, and same-image 5-deg-offset sheets contain positive examples,
   dissociations, and cases where local and offset patches behave similarly.
   They motivate the pairing/locality tests but do not yet supply a population
   result.
3. **The behavioral object is clearer.** The most defensible description is
   contour-relative position-cloud confinement. One-sample velocity and
   unsigned path do not consistently support generic `edge-following motion`
   language; longer-lag displacement can recover the position relationship.
4. **Displayed Panel G is not an exact-pair model test.** Its SSI values are
   interpolated from one-dimensional population dose curves. That remains a
   useful exploratory prediction, but the model must be evaluated directly on
   exact local and reassigned image/trajectory pairs for a mechanistic claim.
5. **The Panel-H radius manipulation is conceptually appropriate.** Holding
   fixation and trajectory fixed while adding surrounding image content asks
   how spatially local the predictive image-behavior relationship is. Axis or
   coherence changes across radius are not automatically confounds; undefined
   weak axes are a measurement limitation. The population `D_pair(r)` and
   same-image-offset `D_locality(r)` curves have not yet been computed.

The next load-bearing behavior analysis is therefore a hierarchical radius
curve with matched real-pair reassignment at every radius, paired with the
same-image spatial-offset curve. Movement amplitude, gaze eccentricity, and
event class should be retained as secondary modifiers. The exact-pair neural
model evaluation comes after the behavior-side locality curve is understood.

## Executive Decision

The current behavior panels do not yet distinguish local image-contingent FEM
control from two independent orientation anisotropies expressed in the same
screen/head reference frame.

The reviewed BackImage table contains a screen-horizontal bias in both the FEM
spread axes and the local image-edge axes. The already-implemented
within-session x phase image-axis shuffle reproduces the raw alignment almost
exactly:

| subset | observed edge cos2 | shuffled mean edge cos2 | observed - shuffled | shuffle p |
|---|---:|---:|---:|---:|
| all windows | 0.097299 | 0.097243 | +0.000056 | 0.493 |
| reliable axes | 0.135464 | 0.136123 | -0.000659 | 0.537 |

The absolute axial marginals make the mechanism concrete:

| quantity | all-window resultant R | all-window preferred axis |
|---|---:|---:|
| FEM position-cloud axis | 0.248 | -14.6 deg |
| local image-edge axis | 0.405 | -2.2 deg |

For image coherence >= 0.5, the FEM marginal has `R = 0.344` at `-10.5 deg`
and the edge marginal has `R = 0.676` at `-3.3 deg`. Therefore, stronger raw
relative alignment at high coherence can arise because high-coherence image
patches are increasingly horizontal while the animal or measurement system
already has a stable near-horizontal prior.

This is the behavior-side version of the reference-frame distinction in the
Otero-Millan literature:

- small/fixational movements can be dominated by an egocentric motor prior;
- large exploratory movements can be more influenced by allocentric scene
  structure;
- movement amplitude/spatial scale is not the same variable as instantaneous
  gaze eccentricity;
- a 2-D eye-position trace does not include RF-eccentricity-dependent retinal
  displacement caused by torsion;
- the direction of the marmoset prior must be measured rather than imported
  from the human horizontal or rhesus vertical result.

### Supplemental gaze-position anisotropy: map-first checkpoint 1 (2026-08-09)

The first supplemental checkpoint maps where gaze-position dependence could
enter before fitting a population model. It uses the same 11,749 reviewed
128-sample drift windows as the updated Panel 4F (30 sessions, 1,962
session-trials). These are contiguous clean windows extracted after detected
high-speed samples were removed; the trial-level `events_in_trial` field is
not a label for event-containing windows.

The analysis keeps three quantities separate: horizontal-minus-vertical RMS
in screen coordinates; tangential-minus-radial RMS relative to each window's
mean gaze position; and axis-free major-minus-minor RMS. Total drift-cloud RMS
is plotted alongside them because movement scale itself changes with gaze
eccentricity.

Artifacts:

```text
declan/fig/ssi_figure_v2/behavior_confounds/
  build_supp_gaze_position_anisotropy_checkpoint1.py
outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1/
  supp_gaze_position_anisotropy_checkpoint1_v1/
    gaze_position_mechanism_maps.{png,pdf,svg}
    gaze_eccentricity_descriptive_curves.{png,pdf,svg}
    gaze_position_effect_size_comparison.{png,pdf,svg}
    gaze_position_window_values.csv.gz
    gaze_position_grid_values.csv
    gaze_eccentricity_descriptive_values.csv
    preliminary_effect_size_reference.csv
    summary_report.md
    run_metadata.json
```

The visible pooled-window result is a larger and more anisotropic drift cloud
at eccentric gaze, expressed mainly as a screen-horizontal allocation rather
than a consistent gaze-tangential allocation. Comparing peripheral (>=8 deg)
with central (<4 deg) windows gives raw median changes of `+0.656` arcmin in
screen horizontal-minus-vertical RMS, `-0.168` arcmin in gaze
tangential-minus-radial RMS, `+0.713` arcmin in axis-free anisotropy, and
`+1.079` arcmin in total drift RMS radius. All displayed RMS components use
the same sample-covariance convention as Panel 4F. For scale only, these are
respectively `+2.93x`, `-0.75x`, `+3.18x`, and `+4.81x` the updated Panel 4F
high-coherence contour-relative estimate (`+0.224` arcmin, 95% CI
`[+0.090, +0.596]`). These ratios share units but do not represent equivalent
biological contrasts.

The subject split is the main checkpoint warning. Both animals show a positive
peripheral change in screen horizontal-minus-vertical RMS, but their
gaze-frame changes have opposite signs (Allen `+0.342` arcmin; Logan `-0.283`
arcmin). Peripheral windows also have larger drift scale, and screen position,
gaze polar angle, subject composition, and movement scale are not separated by
these pooled medians. The maps therefore establish a candidate screen-frame
gaze-position effect, not an eccentricity mechanism or a confound-adjusted
effect size. Two-dimensional eye position also cannot test the torsional,
RF-eccentricity-dependent displacement described in the literature.

The smallest next checkpoint is a session/animal-hierarchical central-versus-
peripheral contrast that matches or stratifies movement scale and gaze polar
angle, followed by a direct draw-wise comparison with the Panel 4F contrast.
Detected microsaccades should remain a separate raw-event analysis because
the Panel 4F window table contains event-free drift windows.

### Supplemental gaze-position anisotropy: broad-model checkpoint 2 (2026-08-09)

The broad-model checkpoint was completed with:

```text
declan/fig/ssi_figure_v2/behavior_confounds/
  build_supp_gaze_position_anisotropy_broad_model.py
  build_supp_gaze_position_anisotropy_report_pdf.py
outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1/
  supp_gaze_position_anisotropy_broad_model_checkpoint2_v1/
    broad_model_design_and_normalization_audit.{png,pdf,svg}
    broad_model_adjusted_eccentricity_curves.{png,pdf,svg}
    broad_model_incremental_specification_effects.{png,pdf,svg}
    broad_model_residual_spatial_maps.{png,pdf,svg}
    broad_model_window_values.csv.gz
    descriptive_normalized_eccentricity_curves.csv
    adjusted_effect_size_comparison.csv
    broad_model_adjusted_eccentricity_curves.csv
    broad_model_coefficients.csv
    broad_model_diagnostics.csv
    summary_report.md
    supplemental_gaze_position_anisotropy_report_v1.{pdf,md}
    run_metadata.json
```

The design has strong within-session support: all 30 sessions contain both
central (<4 deg) and peripheral (>=8 deg) windows. Separate animal models use
weights that give equal mass to sessions and trials, session-clustered
covariance, and equal animal weight for the grand estimate. The primary
outcomes divide each anisotropy component by total sample-covariance RMS and
translate the fitted fraction back to arcmin at the high-coherence Panel 4F
median movement scale (`2.706` arcmin). This is necessary because `10.7%` of
windows exceed 8 arcmin RMS and the maximum is `309.4` arcmin, making direct
arcmin-scale least squares highly tail-sensitive.

The primary broad additive model includes nonlinear eccentricity and movement
scale, first- and second-harmonic gaze polar angle, local image coherence and
coherence-weighted edge axis, image gradient energy/background fraction,
fixation phase, time since the last detected event, and session fixed effects.
It estimates the change from the median central eccentricity (`2.741` deg) to
the median peripheral eccentricity (`9.580` deg) as:

| outcome | Allen | Logan | equal-animal estimate (95% CI) | relative to Panel 4F |
|---|---:|---:|---:|---:|
| screen H-V | +0.426 | +0.373 | +0.399 [+0.261, +0.538] | +1.78x |
| gaze-frame T-R | -0.181 | -0.201 | -0.191 [-0.368, -0.014] | -0.85x |
| axis-free major-minor | -0.007 | -0.066 | -0.036 [-0.084, +0.011] | -0.16x |

Thus the raw rise in axis-free anisotropy is explained by the accompanying
increase in total drift scale under this model. What remains is primarily a
rotation or reallocation toward the screen-horizontal axis, not a larger
normalized ellipse. The adjusted gaze-frame effect is radial rather than
tangential, opposite to what would be expected from importing a tangential
torsion account into these two-dimensional measurements.

This estimate is specification-sensitive. Allowing eccentricity to interact
flexibly with movement scale and gaze polar angle reduces the equal-animal
screen effect to `+0.180 [-0.044, +0.404]` arcmin, the gaze-frame effect to
`-0.074 [-0.290, +0.142]`, and the axis-free effect to
`-0.058 [-0.172, +0.057]`. The residual maps also retain structured local
patches. The additive estimate should therefore not yet be promoted as the
final supplemental effect size. The next map-first checkpoint should show the
screen H-V eccentricity effect separately over movement-scale and gaze-angle
strata for both animals, with explicit support, to determine where the
additive and interaction-rich models disagree.

## Immediate Claim Audit

### Panel F

The visible contour-relative position-spread profiles are real descriptive
objects. The unresolved issue is their reference. A uniform or randomly
rotated orientation null erases the empirical egocentric FEM prior, whereas a
decisive null must preserve both absolute FEM and image-axis marginals and
destroy only their local pairing.

#### Current-state implementation audit (2026-08-09)

The displayed panel has several strengths that should be retained. It uses the
position-cloud covariance rather than accumulated path or one-sample velocity,
shows the complete angular spread profile rather than only a ratio, and uses
local edge coherence as an explicit conditioning variable. Its source contains
11,749 reviewed windows from 30 sessions and both animals. The existing
session-level progression table also shows a positive high-coherence
parallel-minus-orthogonal RMS difference in both animals (Allen approximately
+0.29 arcmin and Logan approximately +0.12 arcmin for the original broad high
coherence bracket), so the descriptive effect is not confined to one animal.

The current compact renderer nevertheless has three statistical/presentation
problems:

1. The observed angular curves are window medians with no sampling interval;
   the legend counts windows even though windows within trials and sessions are
   not independent.
2. The four displayed wide coherence bands are constructed by taking a
   window-count-weighted RMS of the saved 0.1-wide-bin medians, rather than by
   recomputing the median from all windows in each wide band. Direct
   recomputation changes the high-coherence curves by as much as 0.053 arcmin.
   It does not reverse the qualitative pattern: the exact
   parallel-minus-orthogonal difference is approximately +0.10, +0.09, +0.23,
   and +0.34 arcmin in the 0-0.2, 0.2-0.5, 0.5-0.8, and 0.8-1 bands.
3. The dashed orientation-scrambled reference draws an independent uniform
   axis for every window. Its band measures randomization variability, not
   sampling uncertainty, and it does not preserve the empirical image-axis
   marginal. It therefore does not answer the shared-reference-frame concern.

For a descriptive main-figure role, the closest principled update is to retain
the real contour-relative angular profiles, recompute them directly from their
constituent windows, and add hierarchical session/trial uncertainty with Allen
and Logan equally weighted. The uniform-orientation reference need not be shown
in the main panel. The resulting claim is an association claim:

```text
Fixation position spread becomes more contour-anisotropic as local edge
coherence increases.
```

Matched real-pair reassignment remains the appropriate robustness analysis for
whether the effect is specific to the exact local image/trajectory pairing.
Shared scene and movement statistics can be unpacked in the Discussion rather
than being subtracted from the descriptive panel.

#### Updated descriptive candidate execution (2026-08-09)

The production candidate is saved under:

```text
outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1/
  panel_f_descriptive_hierarchical_profiles_v1/
```

Its point estimator takes the median profile across windows within a trial,
the median across trials within a session, the median across sessions within
each fixed animal, and finally the equal-weight mean of Allen and Logan. The
1,000-draw bootstrap resamples sessions within animal and trials within the
selected session.

The initially retained 0.8-1.0 band proved too sparse for a stable hierarchical
profile: it contained 108 trials, with several sessions contributing only one
to five trials and very large between-session estimates. The displayed
candidate therefore merges 0.5-1.0 into one high-coherence band, restoring 584
trials and all 30 sessions. The equal-animal parallel-minus-orthogonal effects
are:

| coherence | difference (arcmin) | 95% hierarchical CI |
|---|---:|---|
| 0-0.2 | +0.1469 | [-0.0432, +0.3110] |
| 0.2-0.5 | +0.1689 | [-0.0197, +0.3292] |
| 0.5-1.0 | +0.2241 | [+0.0897, +0.5963] |

The high-coherence point estimate is positive in both animals but heterogeneous:
Allen is +0.4030 arcmin [0.1290, 1.0041], whereas Logan is +0.0451 arcmin
[-0.1009, 0.4116]. Thus the updated panel supports a pooled descriptive
coherence-dependent anisotropy while making clear that the magnitude is not
equally expressed in both animals.

#### Panel 4F visual-variation checkpoint

Polar, normalized-shape, endpoint, subject-resolved, and session-resolved views
are saved under:

```text
outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1/
  panel_f_descriptive_hierarchical_profiles_v1/visual_variations_v1/
```

The zero-origin polar plots show that the anisotropy is modest relative to the
absolute position-spread scale. The paired zoomed polar view makes the profile
shape legible but uses a nonzero radial origin and must remain explicitly
labeled as such. The orthogonal-centered curve likewise isolates shape but is
a derived point-estimate view without its own confidence ribbon.

The most important new visual result is animal heterogeneity. Allen develops a
deep contour-normal trough at intermediate and high coherence, whereas Logan
is nearly flat at intermediate coherence and only weakly anisotropic in the
high band. Session-level dots show overlapping coherence distributions with
both positive and negative sessions. The equal-animal high-coherence summary
is positive, but the main-panel wording and Discussion should not imply equal
expression in the two animals.

Without the matched-pair test, the strongest pairing-specific wording remains:

```text
FEM spread and local contours share an orientation bias.
```

### Panel G

Showing that a recorded trajectory outperforms uniformly rotated versions in
the response model proves that trajectory orientation matters to that model.
It does not prove that the animal selected that trajectory for the particular
local contour. If both real traces and coherent edges are near-horizontal, the
same model advantage can be produced without local image-contingent control.

The decisive contrast is:

```text
real local image/trajectory pair
minus
matched real trajectory reassigned from another local image window
```

Uniform rotations remain a useful secondary geometry control, not the primary
behavioral pairing null.

### Panel H

The x-axis is the radius of image content integrated around the fixation point
chosen by the animal. Changing that radius is the intended manipulation, not
a nuisance to remove. A small radius measures the contour local to fixation;
as radius grows, the estimate incorporates contours farther from fixation and
eventually approaches the image's broader orientation statistics. A loss of
image-behavior correspondence with increasing radius is therefore the
candidate behavioral signal: it would localize the spatial support over which
fixation-centered image content predicts the measured position cloud.

The target curve is not raw alignment alone but pairing-null-subtracted local
prediction at every radius:

```text
D_pair(r) = A_real_local_pair(r) - E[A_matched_reassignment(r)]
```

The same-image offset control provides the complementary locality curve:

```text
D_locality(r) = D_local(r) - D_offset(r)
```

At small radii, locally contingent behavior predicts a positive locality
advantage. At larger radii, local and offset patches should converge as both
incorporate more shared/global image content. Global image orientation is not
assumed to be wholly unrelated to the fixation region; this convergence is
why the offset comparison is needed.

Image-axis coherence remains a measurement-quality variable. A weak axis can
make a radius point uninterpretable, but coherence and axis changes should not
be matched away across radii: adding surrounding, potentially unrelated image
content is exactly what the analysis is designed to test. Reliability should
be displayed and handled through estimator sensitivity or uncertainty, not by
selecting only images whose orientation stays constant with radius.

Patch radius is still not movement amplitude, gaze eccentricity, or RF
eccentricity. Those variables can modify the curve but do not define its
estimand. Until the pairing and locality curves are measured, the strongest
defensible wording is:

```text
Raw contour-relative association is concentrated within small image support
around animal-selected fixation points.
```

Even if supported, this is a behavioral image-prediction scale. Calling it a
neural integration window or a causal scale of contour guidance would require
additional evidence.

## Metric Contract to Freeze Before New Analysis

Keep these objects separate in every artifact and table:

1. `theta_cloud`: principal axis of the centered eye-position covariance.
2. `theta_step_lag`: axial direction distribution of displacements at an
   explicitly named lag.
3. `theta_net`: start-to-end displacement direction, with magnitude retained.
4. `rms_parallel` and `rms_orthogonal`: position spread relative to the local
   edge axis.
5. `path_parallel` and `path_orthogonal`: unsigned projected path length.
6. `theta_edge`: local Sobel structure-tensor edge axis at an explicitly named
   patch center and patch radius.
7. `coherence`: reliability of that local image-axis estimate.
8. `A_raw = mean(cos(2 * (theta_cloud - theta_edge)))`.
9. `A_pair = A_raw - E_permutation[A_reassigned]`, where reassignment preserves
   the empirical absolute-axis marginals.

Do not use `edge following`, `movement direction`, `position spread`, and
`trajectory orientation` interchangeably. Prefer absolute differences over
ratios, and keep raw component magnitudes beside any normalized index.

## Interaction Contract

For each checkpoint:

1. save the rendered artifact and all plotted values;
2. show the artifact rather than only reporting a statistic;
3. state what is visibly happening without a population generalization;
4. identify surprises, failures, and ambiguous examples;
5. state what remains unsupported;
6. propose the smallest next step;
7. pause for human interpretation.

Planned output root:

```text
outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1/
```

Every output should record source-table identity, trace provenance, command,
configuration, selection rule, whether it is a targeted visualization or a
production summary, and the git revision when available.

## Checkpoint 1: Show the Reference-Frame Confound Itself

### Smallest contrast

Hold session, fixation phase, movement scale, image coherence, and gaze
eccentricity approximately fixed. Compare two high-coherence windows:

1. a near-horizontal edge with a near-horizontal FEM cloud;
2. an oblique edge with a near-horizontal FEM cloud.

Hypothesis under local image-contingent control:

```text
The FEM axis should rotate with the local edge, including for oblique edges.
```

Prediction under a stable egocentric prior:

```text
The FEM axis should remain near its animal/session preferred screen axis even
when the local edge is oblique.
```

Evidence against both simple accounts would include unstable axes, dependence
on estimator choice, or direction changes restricted to a few samples.

### Required artifact

```text
checkpoint1_reference_frame_examples.png
checkpoint1_reference_frame_example_values.csv
checkpoint1_selected_windows.csv
checkpoint1_run_metadata.json
```

Each example row should show, using matched scales:

- the gaze-centered image patch and Sobel edge axis;
- the raw screen-space eye path;
- the centered position cloud and covariance ellipse;
- absolute FEM and edge axes in the screen/head frame;
- contour-relative FEM coordinates;
- displacement-direction histograms at several named lags;
- movement RMS/range, image coherence, gaze eccentricity, time since the last
  detected event, and detector threshold.

Observed inputs and derived axes must be placed in separate, labeled panels.

### Auditable example roles

Define roles before rendering and save the selection table:

| role | selection intent |
|---|---|
| shared-horizontal-positive | coherent horizontal edge and horizontal FEM axis |
| oblique-local-positive | coherent oblique edge and matched oblique FEM axis |
| motor-prior-dissociation | coherent oblique edge but FEM stays near the session prior |
| image-dominant-dissociation | weak session-prior prediction but strong local pairing |
| low-coherence-control | unreliable image axis at matched movement scale |
| low-anisotropy-control | unreliable FEM cloud axis at matched image coherence |

Minimum columns for `checkpoint1_selected_windows.csv`:

```text
session, subject, trial_idx, global_start, global_stop, selection_role,
selection_rule, selection_value, theta_cloud_deg, theta_edge_deg,
drift_edge_cos2, image_orientation_coherence, anisotropy, rms_radius_deg,
projected_range_deg, abs_mean_radius_deg, gaze_polar_angle_deg,
phase, samples_since_event, patch_radius_deg, trace_provenance
```

### Human checkpoint

Stop after this render and ask:

- Do the oblique examples visibly rotate with the contour?
- Is the covariance-cloud axis a faithful description of the path?
- Which dissociation deserves the time-resolved drill-down?
- Are the example roles and nuisance matching adequate?

## Checkpoint 2: Make the Pairing Null Concrete

Do not begin with a population permutation p-value. For the selected examples,
render four side-by-side constructions:

1. the real local image/trajectory pair;
2. the same real trajectory uniformly rotated;
3. the same image paired with a real trajectory reassigned from a matched
   window;
4. the same trajectory paired with a matched image/edge axis from another
   window.

Matching variables should initially include:

```text
subject, session, phase, coherence bin, FEM RMS/range bin,
FEM anisotropy bin, gaze-eccentricity bin, and time-since-event bin
```

Where possible, add a stricter within-image or within-trial reassignment.

Required artifacts:

```text
checkpoint2_pairing_null_examples.png
checkpoint2_pairing_null_example_values.csv
checkpoint2_pairing_manifest.csv
```

For Panel F, show the full contour-relative spread profile for all four
constructions. For Panel G, show the model dose and predicted SSI for all four
constructions, with rate/spike quantities retained beside bits per spike. This
is a targeted diagnostic render, not a population result.

### Checkpoint 2B: Same-Image 5-Degree Spatial-Offset Patch Control

Add a spatial control that preserves the original image and its global scene
statistics while disrupting the correspondence between the eye trajectory and
the image content at the actual gaze location.

For every selected window, retain the eye trajectory but replace the local
gaze-centered patch with image content centered 5 deg away on the same original
BackImage. Use the same patch radius, image-axis estimator, coordinate
convention, contamination thresholds, and coherence calculation as for the
local patch. Repeat the identical trajectory-randomization procedure for both
patch locations.

The smallest factorial contrast is:

| patch content | real trajectory | randomized trajectory |
|---|---|---|
| actual local patch | `S_local_real` | `S_local_random` |
| same-image patch 5 deg away | `S_offset_real` | `S_offset_random` |

Here `S` should be evaluated separately for the behavior alignment/spread
metric and the Panel-G model prediction. Preserve absolute component values
beside any contrasts.

Define:

```text
D_local = S_local_real - E[S_local_random]
D_offset = S_offset_real - E[S_offset_random]
D_locality = D_local - D_offset
```

Interpretation hypotheses:

- local image-contingent control predicts `D_local > D_offset`, with a positive
  `D_locality` effect;
- a global image-statistic or shared egocentric-axis account predicts similar
  real-versus-randomized effects for local and offset patches;
- no effect for either patch argues against both the local and global versions
  of the proposed relationship;
- an offset effect that depends on whether the remote axis matches the local
  axis suggests extended contour structure rather than an image-wide global
  statistic.

Do not choose one favorable offset direction after looking at the result.
Pre-specify a 5-deg annulus with candidate directions at, for example, 0, 45,
90, 135, 180, 225, 270, and 315 deg in screen coordinates. Retain every
candidate whose complete patch lies within the original image and passes the
same background/contamination criteria as the local patch. Average over valid
directions for the primary per-window offset control, while preserving every
offset as a row in the audit table.

A 5-deg displacement does not guarantee a different image axis: a long contour
or global scene orientation can extend through both patches. Therefore record
the axial local-offset difference and include two diagnostic roles:

```text
orientation-preserving offset: abs axial delta <= 10 deg
orientation-changing offset: abs axial delta >= 30 deg
```

These thresholds are diagnostics, not filters for the primary all-valid-offset
estimate. Also report coherence, contrast, edge energy, background fraction,
and distance to the image boundary for both patch centers. If too few valid
offsets survive at exactly 5 deg, report that support failure rather than
silently changing the distance; a later distance curve at 2.5, 5, and 10 deg
can be a separately labeled follow-up.

Required map-first artifacts:

```text
checkpoint2b_local_vs_offset_patch_examples.png
checkpoint2b_local_vs_offset_patch_values.csv
checkpoint2b_offset_patch_manifest.csv
checkpoint2b_local_offset_axis_relationship.csv
```

For each selected window, the example sheet should show:

- the original image with the true gaze center, 5-deg annulus, and every valid
  offset center;
- the local patch and its edge axis;
- at least one orientation-preserving and one orientation-changing offset
  patch when available;
- the same real eye trajectory overlaid in local coordinates for all patches;
- real and randomized trajectory scores for the local and offset patches;
- `D_local`, `D_offset`, and `D_locality` beside the inputs from which they
  were calculated.

Observed image content, measured trajectory, derived axes, and model predictions
must remain visibly separated. Stop after the selected-window sheet before
computing the population difference-in-differences.

The human checkpoint is:

- Does a 5-deg offset visibly disrupt the relevant local contour?
- Does the real-trajectory advantage disappear only for orientation-changing
  offsets?
- Are apparent offset effects explained by long contours that span both
  patches?
- Is 5 deg sufficiently supported within the displayed image bounds?

### Gate A

Proceed to population pairing tests only if the real local pairing visibly and
consistently differs from the matched real-pair reassignment in the selected
positive examples while the dissociation/control roles behave sensibly. The
local result should also exceed the same-image 5-deg offset result when that
offset visibly changes the available contour axis. Uniform rotation alone is
not sufficient to pass this gate.

If it does not, preserve the failure and reframe F-H as shared-axis/model
geometry panels rather than evidence for locally image-contingent behavior.

## Checkpoint 3: Determine What Behavioral Object Is Aligned

The current main behavior object is a 128-sample position cloud. Existing
follow-ups suggest that parallel RMS/net displacement and unsigned projected
path can give different answers. Strong lag-1 reversal and cancellation also
make single-sample step direction vulnerable to measurement noise.

For the Checkpoint-1 selected windows, render:

- raw and lightly smoothed paths;
- position covariance axes;
- velocity covariance axes;
- displacement directions and magnitudes at 8, 25, 50, 100, and 250 ms where
  sampling permits;
- start-to-end displacement;
- parallel/across RMS, range, unsigned path, reversal fraction, and
  autocorrelation;
- versions under the primary, stricter, and looser event detector/padding
  settings.

Required artifacts:

```text
checkpoint3_behavior_object_multilag.png
checkpoint3_behavior_object_values.csv
checkpoint3_detector_sensitivity.csv
```

### Gate B

- If position spread, lagged displacement, and movement-direction objects all
  agree, `contour-aligned motion` is reasonable language.
- If only position spread/range agrees, use `reduced across-contour excursion`
  or `contour-relative confinement`.
- If the effect disappears after modest smoothing, downsampling, or detector
  changes, prioritize eye-tracker and event-segmentation diagnostics.

Stop for human interpretation before any broad regression.

### Checkpoint 3 execution: selected-window behavior-object diagnostic

The example-level checkpoint was completed on 2026-08-08 with:

```text
declan/fig/ssi_figure_v2/behavior_confounds/
  build_checkpoint3_behavior_object_multilag.py
outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1/
  checkpoint3_behavior_object_v1/
    checkpoint3_behavior_object_multilag.png
    checkpoint3_behavior_object_values.csv
    checkpoint3_multilag_values.csv
    checkpoint3_detector_sensitivity.csv
    checkpoint3_run_metadata.json
```

The same six Checkpoint-1 windows were retained. The sheet displays each local
image and measured trace, the raw and symmetric three-point-smoothed path,
position- and velocity-covariance axes, absolute parallel/normal displacement
RMS at 8.3, 25, 50, 100, and 250 ms, absolute component values for position
RMS, position range, unsigned path, and start-to-end displacement, and a fixed
detector sensitivity comparison. No population inference was performed.

The visible result distinguishes position confinement from instantaneous
movement direction:

- parallel-minus-normal position RMS was stable under light smoothing in the
  shared-horizontal (`+3.56` to `+4.26` arcmin), oblique (`+4.95` to `+5.28`),
  image-dominant (`+4.00` to `+4.72`), and low-coherence (`+3.98` to `+4.21`)
  examples. The motor-prior dissociation remained negative (`-3.49` to
  `-4.14`), and the low-anisotropy control remained near zero (`+0.062` to
  `-0.022`).
- one-sample velocity covariance did not reproduce the position axis. Its
  axial difference from the local edge was `83.8`, `50.2`, `41.1`, `48.8`,
  `0.6`, and `30.0` deg across the six displayed roles. The apparently aligned
  low-coherence case is not interpretable because the image axis itself is
  unreliable.
- displacement alignment in the positive examples emerged mainly at longer
  lags. For example, shared-horizontal parallel-minus-normal displacement RMS
  changed from `-0.52` arcmin at 8.3 ms to `+2.49` at 250 ms; oblique changed
  from approximately zero to `+4.78`; and image-dominant changed from `-0.06`
  to `+2.84`. The motor-prior example reached `-2.40` at 250 ms.
- raw unsigned path was not a stable surrogate for position spread. Its
  parallel-minus-normal contrast changed sign after light smoothing for the
  shared-horizontal (`-62.4` to `+6.1` arcmin) and oblique (`-25.8` to
  `+36.5`) examples, while the low-anisotropy control retained a nonzero path
  contrast despite essentially zero position anisotropy.
- projected one-sample steps showed high reversal fractions (approximately
  `0.71-0.84`) and strongly negative lag-1 autocorrelation (approximately
  `-0.58` to `-0.83`), consistent with high-frequency alternation strongly
  influencing the cumulative path and velocity objects.
- the primary detector threshold exactly reproduced the reviewed extraction.
  Aggressive, primary, and permissive settings flagged zero samples in five of
  six selected windows. The aggressive setting flagged seven samples in the
  oblique example, but its position-RMS contrast remained similar (`+5.04`
  versus `+4.95` arcmin). This makes the selected examples robust to the tested
  detector variation, but five rows provide little detector stress because no
  alternative mask changed them.

The provisional Gate-B reading is therefore **position-cloud confinement, not
instantaneous contour-aligned motion**. Longer-lag displacement can recover the
position-spread relationship, but one-sample velocity and unsigned path cannot
currently support a generic movement-direction claim. This remains a human
checkpoint rather than a population conclusion.

## Checkpoint 4: Establish the Fixation-Centered Image-Support Curve

The primary manipulation keeps the animal-selected fixation point and measured
trajectory fixed while increasing the image-patch radius. This deliberately
allows the integrated contour content to change with radius. Use selected
examples before estimating a population curve.

Movement and gaze variables are secondary modifiers. A small example
factorial can still compare:

- low versus high movement RMS/range;
- small versus large image patch radius;
- central versus eccentric gaze;
- drift-only versus microsaccade-containing epochs, plus detected
  microsaccades as a separate event analysis.

The central population quantities, if the examples support them, are:

```text
D_pair(r) = local alignment - matched real-pair reassignment
D_locality(r) = D_local(r) - D_same-image-offset(r)
```

Movement-amplitude, gaze-eccentricity, and event-regime stratification then ask
whether this primary radius curve changes across behavioral regimes. Keep
absolute FEM anisotropy in the screen/head frame as a separate outcome. This
tests the Otero-Millan movement-scale hypothesis without redefining patch
radius as movement scale.

Required artifacts:

```text
checkpoint4_scale_eccentricity_examples.png
checkpoint4_scale_eccentricity_example_values.csv
```

### Gate C

Interpret Panel H as a fixation-local behavioral image-support result only if:

1. `D_pair(r)` is positive at small radii and decays relative to a
   marginal-preserving real-pair reassignment at the same radius;
2. the local patch outperforms same-image spatial-offset patches at small
   radii and the curves converge as their support becomes more global;
3. the result replicates across subjects/sessions;
4. it is not created by undefined low-coherence axes and is qualitatively
   robust to reasonable image-axis estimators and patch-center definitions.

Movement amplitude, gaze eccentricity, and detected-event regime are important
effect modifiers, not prerequisites for defining the primary scale. Do not
require the estimated image axis or coherence to remain fixed across radii;
that would condition away the intended manipulation. Gate C does not by itself
license a neural-integration-window or causal-guidance claim.

### Checkpoint 4 execution: selected-window scale/eccentricity diagnostic

The example-level checkpoint was completed on 2026-08-09 with:

```text
declan/fig/ssi_figure_v2/behavior_confounds/
  build_checkpoint4_scale_eccentricity_examples.py
outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1/
  checkpoint4_scale_eccentricity_v1/
    checkpoint4_scale_eccentricity_examples.png
    checkpoint4_scale_eccentricity_example_values.csv
    checkpoint4_scale_eccentricity_selected_windows.csv
    checkpoint4_scale_eccentricity_session_support.csv
    checkpoint4_detected_event_examples.png
    checkpoint4_detected_event_values.csv
    checkpoint4_selected_events.csv
    checkpoint4_run_metadata.json
```

For each animal, one session was fixed and four mid-fixation drift windows
were selected as a 2 x 2 crossing of low/high movement RMS with lower/higher
session-relative gaze-eccentricity quartiles. Selection used movement RMS,
gaze eccentricity, position-cloud anisotropy, and time since a detected event;
it did not use image orientation, image coherence, or contour-relative
alignment. The identical measured path and gaze center were then evaluated at
0.5-, 1.25-, and 2-deg image-patch radii. A separate sheet retained one
approximately 0.5-deg and one approximately 5-deg detected event per animal
across those radii. No population inference was performed.

The labels `lower-eccentricity` and `higher-eccentricity` are deliberately
relative. These sessions do not supply a foveal-versus-peripheral comparison:
the selected lower-eccentricity Logan windows were still 4.30-4.52 deg from
screen center, and Allen's were 2.79-3.13 deg.

The sheet verifies the intended radius manipulation: within a row, the animal-
selected fixation point and measured path are identical, while progressively
more surrounding image content contributes to the edge-axis estimate. It also
shows where measurement quality limits interpretation:

- where coherence and the axial estimate remained reasonably stable, the
  same trace's contour-relative spread was often stable across radius. Allen's
  low-movement/higher-eccentricity example had coherence `0.35-0.44` and a
  rotation-null-subtracted parallel-minus-normal RMS advantage of
  `+0.75` to `+0.80` arcmin. Logan's corresponding example had coherence
  `0.43-0.58` and an advantage of `+0.80` to `+0.83` arcmin.
- radius dependence is ambiguous when the image axis is unreliable. Allen's
  low-movement/lower-eccentricity example changed from an approximately
  horizontal axis at 0.5 and 1.25 deg to an approximately vertical axis at
  2 deg while coherence remained only `0.05-0.18`; its signed advantage
  consequently flipped. Logan's high-movement/higher-eccentricity example had
  coherence `0.02-0.11`, so its across-radius score changes are not a useful
  scale measurement.
- some reasonably reliable axes showed the sought quantitative radius
  sensitivity. Allen's high-movement/lower-eccentricity rotation-null
  advantage was
  `+0.73`, `+0.80`, and `+0.34` arcmin as radius increased, while coherence
  declined from `0.72` to `0.50`.
- detected-event amplitude did not produce a simple image-relative ordering.
  Allen's coherent approximately 5-deg event remained 43-52 deg from the image
  axis across radii, whereas Logan's coherent approximately 0.5-deg event
  remained 61-67 deg away. For Logan's approximately 5-deg event, the apparent
  event-edge difference changed from 89 deg at 0.5 deg radius to 10-12 deg at
  larger radii as coherence increased from `0.16` to `0.44-0.50`. This is a
  concrete example of surrounding image content changing the orientation
  assigned to the same event. Because its 0.5-deg axis is weak, this particular
  row cannot establish a reliable local-to-global transition by itself.

The corrected provisional reading is that the across-radius construction is
conceptually appropriate: changing contour content outside the small
fixation-centered window is the signal of interest, not a confound. The sheet
does not yet evaluate Gate C, however. Its `D` uses uniform rotations rather
than marginal-preserving matched real-pair reassignment; it contains only one
window per example cell; and it does not compare the local radius curve with
the same-image offset-patch curve. It therefore cannot yet estimate the
population spatial-support scale or determine whether small-radius prediction
is specifically local rather than inherited from global orientation bias.

This example-sheet execution covers drift-only factorial examples and detected
events as a separate regime. It did not itself test mixed microsaccade-
containing epochs, the population `D_pair(r)` and `D_locality(r)` curves,
population replication, or patch-centering and image-axis-estimator
sensitivity. The population curves are addressed in the next subsection. A
coherence-matched
across-radius selection is not proposed because it would condition on a
quantity that can legitimately change as additional image content is
integrated; coherence should instead accompany every radius point as a
reliability diagnostic.

### Panel 4H production pairing/locality radius audit

The load-bearing population radius analysis was completed on 2026-08-09 with:

```text
declan/fig/ssi_figure_v2/behavior_confounds/
  build_panel_h_pairing_locality_radius_population.py
outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1/
  panel_h_pairing_locality_radius_population_v1/
    panel_h_pairing_locality_radius_population.png
    panel_h_window_values.csv.gz
    panel_h_trial_values.csv
    panel_h_hierarchical_mean_curves.csv
    panel_h_hierarchical_slope_curves.csv
    panel_h_complete_support_windows.csv
    panel_h_match_strata.csv
    panel_h_matching_quality.csv
    panel_h_matched_trajectory_reassignments.npz
    panel_h_offset_patch_radius_features.csv.gz
    summary_report.md
    run_metadata.json
```

The analysis retained 11,448 windows from 1,911 trials and 30 sessions that
passed local-patch QC at all 11 radii from 0.25 to 3 deg. For the offset
comparison, an offset direction contributed only if that same 5-deg direction
passed QC at every radius. Every window retained at least one such direction;
8,789 retained all eight. The primary null used 256 within-session and phase
real-trajectory reassignments. Each assignment preserved the trajectory
marginal exactly, prohibited same-trial donors, used adaptive strata over
movement RMS, position-cloud anisotropy, gaze eccentricity, and time since the
last event, and was held fixed across every radius and local/offset patch.

Window values were collapsed to trials before inference. The 1,000-draw
hierarchical bootstrap resampled sessions and trials while holding Allen and
Logan fixed and equally weighted; its intervals are therefore conditional on
these two animals, not animal-population confidence intervals.

The corrected result does not currently establish a spatial cutoff:

- raw local alignment and the matched real-pair null were very similar. Mean
  `D_pair` ranged from approximately `0.000` to `+0.012` axial cos2 and its
  hierarchical interval included zero at every radius.
- mean `D_locality` was descriptively positive (`+0.008` to `+0.022`), but its
  interval also included zero at every radius. Much of this difference came
  from slightly negative offset-pair scores rather than a reliably positive
  local `D_pair`, so it is not evidence for a local predictive scale by itself.
- the direct correction of the currently displayed alignment/coherence slope
  was more suggestive. At the pre-existing 1.25-deg peak, the raw slope was
  `+0.613`, the matched-pair null slope was `+0.353`, and the residual was
  `+0.260` with a hierarchical 95% interval of `[-0.050, +0.576]`. Both
  animals had positive point estimates at 1.25 deg and 21 of 30 session slopes
  were positive, but neither animal-specific interval excluded zero.
- the corrected slope fell immediately after 1.25 deg, but it did not remain
  near zero: Allen became weakly negative at larger radii while Logan became
  increasingly positive. Consequently, the across-animal curve does not show
  a replicated point where predictive image support stops.
- the parallel-minus-normal RMS counterpart was positive descriptively and
  declined toward 3 deg, but its hierarchical intervals also included zero.

The Gate-C reading is therefore **suggestive peak, gate not passed**. The
matched-pair null explains a substantial part of the displayed raw 4H curve.
The residual at 1.25 deg is compatible with a fixation-local image-support
effect but is too uncertain, and too different across animals at larger
radii, to identify a cutoff. Extending beyond 3 deg is not yet the immediate
priority: the smallest useful next diagnostic is the session-resolved
corrected slope curve and null-definition sensitivity, focused on why Allen
and Logan diverge after 1.25 deg.

## Checkpoint 5: Test Temporal Direction and Patch-Center Endogeneity

The current local image patch is centered at the mean gaze computed from the
same window used to estimate the FEM cloud. That is a descriptive pairing but
not a directional test of whether image structure guided subsequent motion.

For selected examples, compare edge axes measured at:

- fixation or saccade landing;
- the first valid sample in the window;
- the pre-window gaze position;
- the whole-window mean gaze;
- the window endpoint.

Ask whether an earlier image axis predicts later displacement better than a
later image axis predicts earlier displacement. Use non-overlapping windows
for this checkpoint.

Required artifacts:

```text
checkpoint5_lagged_image_behavior_examples.png
checkpoint5_lagged_image_behavior_values.csv
```

This is still observational. A decisive reference-frame test requires repeated
or experimentally rotated images.

## Checkpoint 6: RF-Eccentricity and Torsion Sensitivity

This checkpoint concerns the bridge from measured screen-space behavior to the
retinal trajectory used by the neural model; it is not a correction to the
screen-space behavioral association itself.

Select representative central, intermediate, and peripheral RFs at distinct
polar angles. For each selected image/trace/RF combination, render:

- the measured 2-D translational retinal path;
- paths after adding plausible torsional rotations;
- the tangential displacement contribution using `delta_s ~= e * delta_phi`;
- contour-relative motion components before and after torsion;
- instantaneous response maps and difference maps when model evaluation is
  warranted;
- rates, expected spikes, absolute SSI, and paired SSI differences.

Required artifacts:

```text
checkpoint6_torsion_rf_examples.png
checkpoint6_torsion_rf_example_values.csv
checkpoint6_selected_rfs.csv
```

### Gate D

If plausible torsion materially changes the sign, preferred axis, or RF-group
ordering of the neural bridge, the 2-D retinalization is not sufficient for an
RF-general mechanistic claim. Report the sensitivity and motivate direct 3-D
eye measurement rather than treating one assumed torsion value as ground truth.

## Population Summaries Come Last

Only after Gates A-B, and Gates C-D where relevant, run population analyses.

Primary behavior summary:

```text
D_pair(r) = observed local alignment(r)
            - matched real-pair reassignment alignment(r)

D_locality(r) = D_local(r) - D_same-image-offset(r)
```

Report the full radius curves rather than only a selected peak or threshold.
Any fitted decay scale should be secondary to the observed curve and should
retain uncertainty from the subject -> session -> fixation hierarchy.

Primary model bridge:

```text
SSI(real local pair, r) - SSI(matched real-trajectory reassignment, r)
```

Secondary controls:

- uniform trajectory rotation;
- within-session x phase edge-axis shuffle;
- within-image/trial reassignment;
- same-image 5-deg spatial-offset patches crossed with the identical trajectory
  randomization, summarized by `D_locality`;
- detector and smoothing sensitivity;
- alternate image-axis estimators and patch centers;
- movement-amplitude, gaze-eccentricity, and subject strata.

Inference must respect the sampling hierarchy. The production extraction uses
128-sample windows with 16-sample stride, so adjacent windows overlap by 87.5%
and are not independent observations. Use non-overlapping fixation-level
summaries or a hierarchical bootstrap over subject -> session -> trial/fixation.
Report leave-one-session, leave-one-image, and animal-specific results. Do not
use flat window SEM/bootstrap intervals as the primary inferential surface.

## Recommended Execution Order

1. Build Checkpoint 1 only: absolute reference-frame examples and auditable
   selection table.
2. Pause and select the most informative positive, dissociation, and control
   examples.
3. Build Checkpoint 2 only: make uniform rotation versus real-pair reassignment
   visually concrete.
4. Build Checkpoint 2B only: compare the true local patch with pre-specified
   same-image patches on a 5-deg annulus, using identical trajectory
   randomization.
5. Pause and decide whether local image-contingent pairing remains plausible.
6. Build Checkpoint 3: adjudicate spread, displacement, path, and measurement
   interpretations.
7. Run Checkpoints 4-5 only if the behavior-side claim survives.
8. Run Checkpoint 6 only for the neural/retinal bridge that remains supported.
9. Compute hierarchical population summaries and revise the figure last.

## Existing Provenance and Evidence Surfaces

Behavior extraction and event removal:

```text
declan/fixation_statistics_by_stimulus/extraction.py
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/window_features.csv
```

Reviewed image/FEM windows and existing marginal-preserving shuffle:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_image_structure_reviewed_v2_screenfiltered_yfix/
    backimage_image_fem_windows.csv
    orientation_alignment_summary.csv
    run_metadata.json
declan/fixation_statistics_by_stimulus/run_backimage_image_structure_analysis.py
```

Position-spread, displacement, path, reversal, and scale diagnostics:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_contour_motion_component_plots_v1/
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_patch_radius_sensitivity_v1/
```

Current Figure 4 behavior/model panels:

```text
declan/fig/ssi_figure_v2/panels/panel_h_unwrapped_edge_coherence.py
declan/fig/ssi_figure_v2/panels/panel_j_match_advantage.py
declan/fig/ssi_figure_v2/panels/panel_k_patch_radius_alignment_slope.py
declan/fig/ssi_figure_v2/behavior_model_bridge/
declan/fig/ssi_figure_v2/compose_ssi_figure_v4.py
```

Map-first SSI precedent and relevant provenance guardrails:

```text
.agents/skills/map-first-analysis/SKILL.md
.agents/skills/map-first-analysis/references/ssi-debugging-case-study.md
```

Important inherited guardrails:

- keep instantaneous and trajectory-averaged SSI distinct;
- retain rates, expected spikes, and absolute information quantities beside
  bits per spike;
- prefer differences over unstable ratios;
- treat the exact baseline/null construction as part of the claim;
- reject old trace caches with the 128-to-40 timepoint compression provenance;
- label targeted visualization renders separately from production summaries.

## Execution Log

### Checkpoint 1 completed: absolute reference-frame examples

Artifacts were generated under:

```text
outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1/
```

The six algorithmically selected roles were balanced across animals (three
Allen, three Logan). Native 128-sample traces reproduced the reviewed window
means to numerical precision and reproduced the stored RMS values exactly.

Visible result: the oblique-local-positive example had a cloud/edge difference
of 0.23 deg while both axes were about 82 deg from the session motor prior.
Conversely, the motor-prior dissociation had a cloud/prior difference of 0.36
deg and a cloud/edge difference of 87.3 deg. This weakens either simple account
as a complete description. Short-lag displacement axes were much less aligned
than the 250-ms displacement axes in the positive examples, so the covariance
effect may describe slower excursion geometry rather than instantaneous
movement direction.

Primary artifacts:

```text
checkpoint1_reference_frame_examples.png
checkpoint1_reference_frame_example_values.csv
checkpoint1_selected_windows.csv
checkpoint1_run_metadata.json
```

### Checkpoint 2 completed: concrete pairing-null constructions

For each Checkpoint-1 target, a partner was chosen from a different trial in
the same session and coherence bin by minimizing a session-standardized
distance over coherence, FEM RMS, central-snippet range, FEM anisotropy, gaze
eccentricity, and time since event, with a recorded phase penalty. The sheet
shows the real pair, one concrete uniform-rotation draw, trajectory
reassignment, and image-axis reassignment. Rotation profiles and model scores
summarize 256 uniform draws rather than the displayed draw alone.

The most important visible failure was that nearest matched partners often
preserved the absolute image axis: local/partner edge differences were 4.8,
1.8, 4.9, 87.8, 15.3, and 0.01 deg across the six roles. Therefore, simple
within-session image reassignment did not reliably disrupt the local axis. Only
the image-dominant dissociation supplied a clearly orientation-changing image
partner; there, image-axis reassignment visibly rotated the contour-relative
path and reduced the aligned-high-SF marginal model prediction relative to the
real pair. This is an example-level observation, not a population result.

Matching support was imperfect: two partners crossed fixation phase, and the
low-anisotropy control had a large matching distance. Those rows remain in the
manifest as failures rather than being silently replaced after viewing the
effect.

The checkpoint-local Panel-G curve export retains moving and baseline absolute
SSI, information per sample, and expected spikes per sample beside normalized
SSI. Predictions remain one-dimensional marginal dose-curve interpolations,
not a reconstructed 2-D response surface or a targeted model rerun.

Primary artifacts:

```text
checkpoint2_pairing_null_examples.png
checkpoint2_pairing_null_example_values.csv
checkpoint2_pairing_manifest.csv
checkpoint2_pairing_null_spread_profiles.csv
checkpoint2_run_metadata.json
```

Checkpoint 2 remains below Gate A. The next planned human-guided stage is
Checkpoint 2B, whose pre-specified same-image 5-deg annulus is designed to
preserve image-wide statistics while making the local-versus-nonlocal content
contrast explicit.

## First Human Decision

Before writing analysis code, decide whether Checkpoint 1 should be restricted
to one session for the cleanest within-session reference-frame contrast or
should deliberately include both animals from the start. The recommended first
pass is six role-selected windows spanning both animals, with matching performed
within session and the cross-animal comparison used only as a visible control.

## Gaze-position supplement: revised checkpoints 3 and 4

The gaze-position analysis now uses dimensionless covariance contrasts rather
than a directional RMS difference divided by total RMS. This avoids a
part-whole normalization and separates drift-cloud shape from total scale:

```text
A_screen = (cov_xx - cov_yy) / (cov_xx + cov_yy)
A_gaze   = (var_tangential - var_radial) / (cov_xx + cov_yy)
A_axis   = (lambda_1 - lambda_2) / (lambda_1 + lambda_2)
```

For an interpretable arcminute display only, model predictions are translated
at a fixed total drift-cloud RMS radius of 2.7064 arcmin. The central endpoint
is the observed median below 4 deg gaze eccentricity (2.741 deg), and the
peripheral endpoint is the observed median at or above 8 deg (9.580 deg).

Equal-animal central-to-peripheral estimates are:

| Outcome | additive model | interaction model |
|---|---:|---:|
| screen horizontal - vertical | +0.368 [+0.244, +0.493] arcmin | +0.174 [-0.026, +0.374] arcmin |
| gaze tangential - radial | -0.174 [-0.334, -0.013] arcmin | -0.069 [-0.268, +0.129] arcmin |
| axis-free major - minor | -0.022 [-0.067, +0.024] arcmin | -0.037 [-0.143, +0.068] arcmin |

Five-fold trial-held-out RMSE is slightly lower for the additive model for all
three outcomes in both animals. The advantage is small. The additive estimate
is therefore the compact main summary, while the interaction estimate remains
the sensitivity bound. The appropriate wording is “a plausible screen-frame
effect that is sensitive to model specification.”

Tracker-coordinate proxy checks do not rule out a position-dependent artifact.
Calibration residuals or stationary-target recordings would be needed for a
clean adjudication. The analysis contains only two animals, two-dimensional eye
position cannot test torsional retinal displacement, and the weak gaze-frame
result should not be presented as a stable radial law.

Checkpoint 4 directly re-estimates the Figure 4F contrast. The high-coherence
covariance reconstruction is +0.227 arcmin, essentially reproducing the
reported +0.224 arcmin. Giving four absolute contour-axis bins equal weight
reduces the contrast to +0.040 arcmin [-0.101, +0.142], an estimated 82.2%
attenuation. This was a provisional audit: its bins did not wrap 180/0 deg into
one horizontal axial bin, so the 82.2% value must not be retained as a final
estimate. It did replace the earlier numerical-ratio argument: numerical
similarity between the gaze-position effect and Figure 4F did not show that
gaze biased Figure 4F. Absolute screen-axis marginals account for much of the
displayed contrast. Several peripheral gaze-by-orientation cells are sparse,
so a precise gaze-specific attenuation coefficient is not yet supported.

Primary artifacts:

```text
outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1/
  supp_gaze_position_covariance_checkpoint3_v1/
    covariance_contrast_metric_contract.png
    covariance_contrast_adjusted_curves.png
    covariance_contrast_specification_effects.png
    model_specification_cross_validation.png
    tracker_proxy_diagnostics.png
  panel_f_gaze_attenuation_checkpoint4_v1/
    panel_f_gaze_attenuation_overview.png
    panel_f_nonparametric_attenuation.csv
    panel_f_orientation_gaze_cells.csv
    panel_f_model_attenuation.csv
  supp_gaze_position_drift_report_v2/
    supplemental_gaze_position_drift_report_v2.pdf
    supplemental_gaze_position_drift_report_v2.md
```

The earlier v1 report is retained as a provenance artifact. The v2 report
supersedes its wording and its “relative to Figure 4F” numerical-ratio framing.

## Checkpoint 5 completed: axial-orientation validation of Figure 4F

Checkpoint 5 corrects the contour-axis circularity. The four canonical bins are
centered at 0, 45, 90, and 135 deg; the horizontal bin joins 157.5-180 deg to
0-22.5 deg. The primary outcome is the exact high-coherence, within-window
parallel-minus-orthogonal RMS difference. Gaze and total-RMS conditioning are
kept in a separate sensitivity panel because they change the estimand.

The exact paired reconstruction is +0.204 arcmin [+0.105, +0.373], close to the
reported Figure 4F value of +0.224 arcmin. Equal weighting of the four wrapped
axial bins gives -0.028 arcmin [-0.498, +0.136]. A continuous doubled-angle
median regression gives +0.001 arcmin [-0.185, +0.187]. Modeling the paired
parallel-minus-orthogonal difference is algebraically the endpoint-relation by
orientation interaction. Median regression matches Figure 4F's hierarchical
median construction and avoids allowing the long-tailed window RMS differences
to dominate the continuous fit.

Across bin counts of 4, 6, 8, and 12 and four boundary phases, configurations
with complete support in both animals range from -0.243 to +0.104 arcmin.
Several 8-bin and all 12-bin configurations contain an empty Allen cell. They
are recorded but excluded from the uniform-bin summaries rather than averaged
over the remaining bins.

The prior checkpoint's empirical Panel C and regression Panel F were different
estimands. Panel C changed only the absolute-orientation weights. Panel F also
fixed or conditioned on gaze, total drift-cloud RMS radius, image variables,
phase, and event timing. Total RMS is part of the behavior being studied, so it
is not an automatic nuisance variable. The conditional series is retained as a
sensitivity analysis rather than a primary correction.

Leave-one-session-out prediction, with session fixed effects omitted, favors
the additive gaze-position model for every outcome and animal. Its RMSE is
lower by about 7-15%. This strengthens the parsimony argument relative to the
trial-held-out check, while the interaction estimate remains the sensitivity
bound.

The promoted interpretation is:

> The displayed pooled Figure 4F contrast is strongly dependent on the
> absolute contour-orientation distribution. After axial-orientation control,
> evidence for orientation-independent local contour-drift alignment is weak.

Primary artifacts:

```text
outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1/
  panel_f_axial_orientation_audit_checkpoint5_v1/
    panel_f_axial_orientation_audit.png
    axial_primary_estimates.csv
    axial_bin_count_phase_sensitivity.csv
    axial_continuous_uniform_estimates.csv
    axial_conditional_sensitivity.csv
    session_heldout_model_specification_cv.png
    session_heldout_model_specification_cv.csv
  supp_gaze_position_drift_report_v3/
    supplemental_gaze_position_drift_report_v3.pdf
    supplemental_gaze_position_drift_report_v3.md
```

Reports v1 and v2 remain provenance artifacts. Report v3 supersedes their
Figure 4F attenuation wording and removes the fixed 82% attenuation claim.
