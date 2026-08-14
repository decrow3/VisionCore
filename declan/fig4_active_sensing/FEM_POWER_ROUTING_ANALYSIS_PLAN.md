# Analysis plan: can FEM-driven neural changes be explained by redistribution of retinal image power?

Date: 2026-08-13

Status: working formal protocol following the recorded-grating model-selection
checkpoint; the contracts below must be frozen before confirmatory evaluation

Scope: corrected natural-image retinal movies, fixed-retina grating measurements,
RR100 digital-twin responses and activation maps, firing rates, and spatial
selectivity index (SSI)

## 1. Scientific objective

The established prediction in the field is that fixational eye movements
(FEMs) transform spatial structure in an image into temporal-frequency power.
Neurons with different spatial-frequency, orientation, and temporal-frequency
tuning should receive different amounts of this redistributed power.

The primary scientific question is:

> Are the FEM-driven changes in digital-twin activation maps, firing rates, and
> SSI explained by redistribution of retinal image power across temporal
> frequencies, after accounting for tuning estimated from fixed-retina
> digital-twin grating responses in recorded-spatial-frequency-validated units?

The analysis must distinguish two related but non-equivalent questions:

1. **Prediction:** Does a power-derived scalar predict firing rate, activation
   magnitude, or SSI on held-out stimuli?
2. **Mechanistic explanation:** Does a spatially local power-derived map predict
   where activation changes, and does SSI calculated from that predicted map
   reproduce the twin's SSI change?

A scalar power measure may predict SSI without explaining the spatial mechanism.
Conversely, a spatially local model may recover useful information that was
destroyed by collapsing an entire retinal movie into one power spectrum. These
outcomes will therefore be reported separately.

## 2. Current evidence and starting point

### 2.1 Completed recorded-grating model comparison

Competing power formulations have been evaluated using 61 visually responsive
units from 15 recording sessions. Predictions were evaluated on complete held-out
experimental trials using identical receptive-field apertures, response windows,
and trial folds.

The best single power predictor was local power weighted by the unit's
phase-averaged spatial-frequency, orientation, and temporal-frequency response.
For recorded firing rates, this model achieved mean held-out \(R^2=0.0316\),
compared with \(0.0173\) for orientation-collapsed phase-averaged tuning and
\(0.0183\) for squared spatial- and temporal-frequency tuning. The oriented
predictor improved over the orientation-collapsed predictor in 74% of units.

The absolute recorded-response values are expected to be low because the target
is a noisy single-trial 333-ms response window. The median window contains three
recorded spikes. The relevant population benchmark is a session-balanced
within-trial correlation of approximately \(r=0.34\) between the complete
digital twin and recorded responses. Its squared correlation is not a calibrated
held-out \(R^2\) and must not be described as variance explained. The grating
result is consequently used to select and rank spectral formulations, not as a
claim that retinal power fully explains recorded neural responses.

**Decision from this checkpoint:** use the orientation-aware phase-averaged
power model as the primary spectral formulation in the natural-image analysis.
Retain orientation-collapsed phase-averaged tuning, squared tuning, local total
power, and whole-image total power as controls.

### 2.2 Completed exploratory natural-image response checkpoint

A targeted natural-image checkpoint compared receptive-field-local
orientation-aware and orientation-collapsed power with exact digital-twin
responses. It contained three conditions selected using input quantities only
and all five available units in the inspected recording session. Neural outcomes
were not used to select the three conditions or the five-unit cohort, although
the later explanatory examples were selected using response comparisons.

Across these three conditions, orientation-aware power followed the magnitude
of the digital twin's moving-versus-stabilized response more closely than the
orientation-collapsed power measure for all five units. The clearest case was
unit 56, for which the descriptive correlation with response-modulation
magnitude changed from \(r=-0.89\) for orientation-collapsed power to
\(r=+0.98\) for orientation-aware power. Units 28, 65, and 64 also showed
substantial improvements, while unit 0 improved only slightly.

Unit 56 also provided an important dissociation: orientation-aware power closely
tracked the root-mean-square magnitude of the moving-versus-stabilized response,
while the signed mean-rate change reversed direction in the strongest condition.
This suggests that phase-averaged power is more naturally related to the
**magnitude of FEM-induced response modulation** than to whether FEM increases
or suppresses mean firing.

Response extraction and summary calculations were verified against raw response
shards and assembled arrays to a maximum discrepancy of
\(4.5\times10^{-9}\) Hz. This establishes the numerical response join but does
not validate the spectral aperture or the scientific inference.

The checkpoint remains exploratory for four reasons:

- each unit correlation contains only three conditions and is therefore highly
  unstable and descriptive rather than inferential;
- the outcome is a scalar temporal response summary, not a spatial activation
  map;
- the analysis uses the current approximate receptive-field aperture;
- the responses are digital-twin predictions, not recorded natural-image
  responses.

**Decision from this checkpoint:** treat response-modulation magnitude as the
primary scalar activation outcome in the expanded natural-image analysis. Keep
signed mean-rate change as a distinct secondary outcome. Use this result as
qualitative convergent support for expanding the orientation-aware model, not as
evidence that activation-map structure or SSI has been explained.

Relevant artifacts are:

- `declan/fig4_active_sensing/analyze_rr100_natural_image_rf_local_oriented_power_response_checkpoint.py`;
- `outputs/fig4_active_sensing/rr100_natural_image_rf_local_oriented_power_response_checkpoint_v1/unit_three_condition_summary.csv`;
- `outputs/fig4_active_sensing/rr100_natural_image_rf_local_oriented_power_response_checkpoint_v1/condition_unit_power_response_metrics.csv`;
- `outputs/fig4_active_sensing/rr100_natural_image_rf_local_oriented_power_response_checkpoint_v1/response_join_audit.csv`.

### 2.3 Invalidated natural-image spectral cache

The first 3,000-condition natural-image spectral cache stored spectra in
image-grouped order while saving condition identities in response-matrix order.
Only 20 of 3,000 rows were aligned. Any condition-level examples, response
comparisons, model fits, or cross-validation using that cache must be treated as
invalid and regenerated.

The affected outputs will remain preserved for provenance but must be labelled
as superseded. They must not be cited as evidence for or against the power
redistribution account.

At the time this protocol was revised, eight analysis scripts still named the
invalid cache directly, and the orientation-aware archive mixed valid
digital-twin grating weights with invalid natural-movie routing fields. Merely
documenting the error is therefore insufficient; Stage 0 quarantines these
artifacts before any new analysis runs.

### 2.4 Production response-cache status

The corrected response request contains 100 images, 1,000 traces, and 100
balanced rounds. Each complete round spans all 100 images and 100 distinct
traces exactly once, giving 1,000 moving movies per round with matched
stabilized responses. The earlier 3,000-condition analysis used only three
complete rounds and is a pilot/debugging checkpoint, not the production
endpoint.

On 2026-08-13 the live manifest contained 22 complete balanced rounds and
22,000 analyzable moving movies. This count is operational status, not a
stopping rule. The confirmatory production snapshot is predeclared as the first
50 complete balanced rounds (50,000 moving movies); any analysis before round 49
is complete is explicitly interim convergence evidence. The cache builder may
append later rounds incrementally, but the primary result must retain the frozen
first-50-round identity. Any analysis of all 100 rounds is a separately labelled
precision extension and cannot replace the primary result silently.

### 2.5 Unresolved spatial issue

The previous natural-image predictors produced one spectral scalar per unit and
movie. The digital twin, however, applies its spatial readout convolutionally on
a larger input and produces an activation map. A whole-image scalar contains no
representation of where image power lies relative to the receptive field and
cannot by itself reconstruct map sharpness or dispersion.

The existing receptive-field-local aperture is an architectural approximation:
the learned Gaussian readout is back-projected through the feedforward spatial
support. It deliberately excludes recurrent propagation and does not use the
learned core filters to estimate a signed or stimulus-dependent effective
receptive field. Its meaning must therefore be calibrated empirically before it
is used as the spatial basis of a mechanistic claim.

## 3. Analysis principles

1. **Correctness before interpretation.** Condition identities, image inputs,
   FEM traces, response rows, and spectra must be independently verified before
   model fitting.
2. **Prediction and mechanism remain separate.** A scalar association with SSI
   is a predictive result; a spatial explanation requires a predicted map.
3. **Maps before population summaries.** New spatial proxies must first be
   understood in concrete images, units, conditions, and difference maps.
4. **Independent controls.** Unit-specific spectral tuning must be compared with
   local total power, local contrast or energy, and whole-image predictors.
5. **Held-out evaluation.** Calibration and model selection must not use the
   images, FEM traces, trials, or response values used for evaluation.
6. **Metric transparency.** Rate, expected spikes, raw information numerator,
   bits per second where available, and bits per spike must remain separate.
7. **Auditable examples.** Positive examples, dissociations, and negative
   controls must be selected by saved criteria rather than informal inspection.
8. **Reader-facing figures.** Titles and labels must state the scientific
   object, prediction target, validation split, units, and metric meaning.
   Unapproved shorthand and internal formula identifiers belong only in
   machine-readable tables.
9. **Magnitude and sign are different claims.** Nonnegative dynamic power can
   predict response-modulation magnitude directly, but a signed response change
   requires a sign-generating response model. Correlation with signed change is
   exploratory unless moving and stabilized responses are predicted separately.
10. **Twin and biological conclusions remain separate.** Tuning derived from
    digital-twin gratings and outcomes generated by the twin test an internal
    mechanistic decomposition of the twin. Generalization to recorded V1
    requires a separately identified recorded-response test.
11. **Power invariance must be challenged directly.** Power-matched phase
    controls are required because phase is discarded by the proposed proxy.

## 4. Primary hypotheses and informative alternatives

### Hypothesis 1: scalar power predicts neural outcomes

Across held-out natural-image and FEM conditions, an orientation-aware
power-derived scalar predicts FEM-induced response-modulation magnitude better
than global power and orientation-collapsed spectral controls. Signed firing
rate or SSI change is predicted mechanistically only by fitting moving and
stabilized outcomes separately through a training-only response link and then
subtracting their predictions. Direct association between unsigned power and a
signed change remains an explicitly exploratory diagnostic.

### Hypothesis 2: local power predicts activation-map structure

A spatially translated local power calculation predicts the location and
magnitude of activation in the digital-twin map. Unit-specific tuning improves
map prediction beyond untuned local power and local contrast. A power-matched
phase manipulation bounds what such a phase-discarding model cannot explain.

### Hypothesis 3: power-derived maps account for SSI

SSI calculated from predicted activation maps explains a meaningful component
of the FEM-minus-stabilized SSI change on held-out conditions.

Informative alternatives include:

- scalar power predicts SSI but local maps do not, indicating prediction without
  a demonstrated spatial mechanism;
- local total power predicts maps as well as unit-specific tuning, supporting a
  spatially local but spectrally broad account;
- local maps predict activation but not SSI, indicating a mismatch in map
  sharpness, normalization, or temporal weighting;
- neither scalar nor local power succeeds, implicating missing phase, temporal
  history, nonlinear context, incomplete tuning, or an incorrect aperture.

## 5. Planned analysis stages

### Stage 0: quarantine invalid inputs and split reusable tuning from movie routing

**Objective:** make it impossible for a downstream analysis to consume the
misaligned three-round spectral cache or a mixed artifact derived from it.

**Motivation:** invalid outputs are still referenced by active scripts. A prose
warning does not protect against accidental reuse, and the valid grating-derived
tuning must be separable from invalid natural-movie fields.

**Method:**

- inventory every code, manifest, table, figure, and archive that reads or was
  derived from `rr100_corrected_three_round_spectral_cache_v1`;
- write machine-readable supersession manifests beside the invalid cache and
  each affected derived artifact, naming the defect and permitted use;
- create a clean grating-only tuning artifact containing fixed-retina
  digital-twin spatial-frequency, Fourier-orientation, and temporal-frequency
  weights plus recorded-spatial-frequency cohort validation, with no
  natural-movie-derived fields;
- centralize cache validation so downstream scripts reject superseded inputs,
  mixed-provenance tuning archives, incomplete round sets, or identity/hash
  mismatches before loading scientific values;
- update consumers to require an explicit corrected-cache path rather than a
  hard-coded three-round directory;
- retain invalid outputs in place for provenance without allowing them to be
  selected as defaults.

**Decision gate:** no corrected spectral construction or response analysis may
begin until the inventory is saved, the grating-only artifact passes provenance
checks, and all known consumers fail fast on a superseded cache.

**Expected runtime:** 1–2 hours.

### Stage 1: rebuild and validate the corrected natural-image spectral cache

**Objective:** incrementally produce a trustworthy condition-by-spectrum cache
over complete balanced rounds, and freeze the predeclared 50-round production
snapshot.

**Motivation:** all downstream scalar prediction depends on exact alignment
between retinal input, FEM trace, response outcome, and spectral predictor. The
identified row-order error is sufficient to invalidate the previous fits.

**Method:**

- treat the existing three-round, 3,000-movie analysis only as a pilot/debugging
  checkpoint;
- process only complete balanced rounds and append newly completed rounds
  without changing prior row identities;
- render the exact corrected lag-zero 40-frame retinal movie for every condition
  in each included round;
- write spectra directly into the declared response-matrix row rather than
  appending in rendering order;
- store radial and orientation-resolved positive-temporal-frequency power for
  moving and stabilized movies under the identical transform;
- retain separately labelled static-image predictors needed for stabilized
  activation, including the temporal-mean image's spatial spectrum, local mean,
  and contrast; do not mix these with dynamic positive-frequency power;
- preserve the exact spatial-frequency, Fourier-orientation, and
  temporal-frequency axes;
- independently rerender a random sample, round boundaries, image boundaries,
  and trace boundaries;
- compare rerendered spectra with cached spectra numerically;
- record source identities, hashes, code version, and rendering configuration.

**Stopping and snapshot contract:** the primary confirmatory snapshot is rounds
0–49 after all 50 are complete. Earlier complete-round snapshots may be used for
engineering validation and labelled convergence plots only. Round inclusion is
determined without neural outcomes. The manifest must record whether an artifact
is pilot, interim, primary-50-round, or later precision-extension output.

**Required controls:** orientation-resolved power must sum to radial power;
all declared rows must be populated exactly once; response and spectrum identity
arrays must agree exactly.

**Decision gate:** no response fitting may proceed unless every identity check
passes, independent rerenders reproduce the cache within numerical tolerance,
and the requested analysis tier has the declared complete rounds. Confirmatory
claims wait for the frozen 50-round snapshot.

**Expected runtime:** approximately 10–20 minutes per newly available complete
round, plus 20–40 minutes for final audits; actual throughput must be updated
from the first incremental build rather than inferred from the obsolete
three-round estimate.

### Stage 2: corrected whole-image scalar prediction

**Objective:** determine what the corrected global spectral summaries can
predict without interpreting them as activation-map mechanisms.

**Motivation:** scalar prediction is a valid and scientifically useful question.
It also provides the baseline against which spatially local models must improve.

**Primary tuning weights:** phase-averaged spatial-frequency, orientation, and
temporal-frequency responses estimated from fixed-retina digital-twin grating
responses in recorded-spatial-frequency-validated units. These are independent
of the held-out natural images and FEM traces, but they are not independently
measured biological orientation-by-spatial-frequency-by-temporal-frequency
tuning.

**Predictor constructions:**

1. **Magnitude model:** orientation-aware moving-movie dynamic power predicts
   temporal RMS and mean absolute moving-minus-stabilized response change. This
   is the primary scalar test.
2. **Signed mechanistic model:** calculate predictors for moving and stabilized
   movies separately, map each to its corresponding activation or firing rate
   with a monotonic or otherwise predeclared nonlinear link fit on training data
   only, and subtract the two predicted responses. Because temporal-mean
   subtraction makes stabilized dynamic power approximately zero, the
   stabilized model must include the declared static-image and baseline terms
   needed to predict its response; a zero dynamic-power value must not be
   equated with zero neural activation.
3. **Signed-change diagnostic:** direct regression or correlation between
   nonnegative moving dynamic power and signed response change is exploratory
   and cannot establish a sign-generating mechanism.

**Controls:**

- whole-image total supported dynamic power;
- orientation-collapsed phase-averaged spatial- and temporal-frequency response;
- squared spatial- and temporal-frequency tuning, retained for continuity with
  the earlier figure series;
- simple image contrast and dynamic-energy summaries;
- Fourier-phase-scrambled or Fourier-phase-shifted retinal movies matched for orientation-aware
  spatial- and temporal-frequency power. The primary construction randomizes or
  rotates Fourier phase while preserving the full three-dimensional Fourier
  magnitude array, conjugate symmetry, movie dimensions, and input scaling;
  numerical equality of the binned power predictor must be audited.

**Outcomes, analyzed separately:**

- temporal root-mean-square magnitude of the FEM-minus-stabilized response
  change, promoted as the primary scalar activation outcome;
- temporal mean absolute FEM-minus-stabilized response change;
- separately predicted moving and stabilized firing rates and their signed
  difference, retained as secondary mechanistic outcomes;
- direct signed FEM-minus-stabilized mean-rate change, retained only as an
  exploratory unsigned-power diagnostic;
- expected spikes;
- raw information numerator;
- SSI in bits per spike;
- FEM-minus-stabilized SSI change.

**Validation:** before any Stage 4 inspection, reserve final-test image and
FEM-trace identities. Use only the remaining development identities for model
selection and crossed validation. All transformations and calibration
parameters must be fit on training folds. Report held-out \(R^2\), correlation
where useful, and the uncertainty contract in Section 6.

**Justification:** this stage answers whether power is predictive. It does not
claim that a scalar explains the spatial origin of SSI. The three-condition
checkpoint motivates the response-modulation-magnitude target but is too small
to substitute for this held-out expansion.

**Decision gate:** proceed regardless of whether scalar prediction succeeds,
because the spatial diagnostic distinguishes a true failure from information
lost by whole-movie averaging.

**Expected runtime:** 15–30 minutes after the corrected cache exists.

### Stage 3: calibrate the spatial receptive-field and coordinate contract

**Objective:** determine which input-space aperture and spatial translation rule
best reproduce the digital twin's local operation.

**Motivation:** neither the learned Gaussian readout nor its feedforward support
back-projection is automatically equivalent to the composite input-space
receptive field. Applying an aperture in addition to tuning measured through the
native readout may also double-count spatial pooling.

**Method:**

1. Present the native full-field 51×51 grating movies through the normal unit
   pathway and record the native scalar response.
2. Embed identical probes in a larger canvas and compare the corresponding
   central activation-map values.
3. Translate the probe by known pixel offsets and measure activation-map
   displacement, stride, response magnitude, and edge effects.
4. Compare candidate apertures:
   - learned Gaussian readout alone;
   - current feedforward-support back-projection;
   - empirical gradient or finite-difference sensitivity footprint;
   - translated-probe response envelope.
5. Repeat across units with different readout centres, widths, spatial-frequency
   preferences, orientation preferences, and temporal-frequency preferences.

**Evaluation:** native-to-large-canvas response agreement, translated-map
agreement, aperture centre and extent, and stability across grating conditions.

**Decision gate:** select an aperture only if it reproduces native response scale
and spatial translation adequately. Otherwise revise the local surrogate and do
not describe the architectural back-projection as an effective receptive field.

**Expected runtime:** 1–3 hours, depending on the number of translated probes and
whether finite-difference or gradient footprints are required.

### Stage 4: targeted local power-derived activation maps

**Objective:** establish what the proposed local spectral mechanism predicts in
concrete natural-image examples.

**Motivation:** a new map proxy must be visually and quantitatively interpretable
before population statistics are meaningful. This stage diagnoses whether power
predicts the spatial activation pattern rather than merely a scalar outcome.
The completed three-condition checkpoint compared only scalar temporal response
magnitude and therefore does not satisfy this stage.

**Construction:** at each activation-map position, apply a translated copy of the
calibrated aperture, calculate local spatial-frequency, Fourier-orientation, and
temporal-frequency power, and weight it by tuning estimated from fixed-retina
digital-twin grating responses. Match the digital twin's spatial stride, valid
support, and edge convention. Construct moving and stabilized local predictors
separately; form a signed difference map only after applying the frozen
training-only activation calibration to each condition. Never assign a sign to
an unsigned power-difference map by interpretation alone.

Because phase-averaged tuning does not retain stimulus phase, the first promoted
targets will be:

- time-averaged activation maps;
- local root-mean-square or energy maps;
- FEM-minus-stabilized maps summarized over the matched interval.

Exact frame-by-frame reconstruction will remain a later diagnostic rather than
an initial claim.

**Development/test separation:** Stage 4 may inspect only the predeclared
development image and trace identities. Aperture choice, temporal summary,
support, calibration family, display scaling, and example-selection rules may be
revised using that development subset. The reserved final-test image and trace
identities must remain unrendered and unused until all choices are frozen for
Stage 6. If no untouched bank can be reserved, every such decision must instead
be nested within the training portion of each outer fold and the result must not
be called a single final test.

**Initial example set:** one corrected image/FEM contrast and several units,
including:

- strong predicted and observed map change;
- strong predicted change with weak observed change;
- weak predicted change with strong observed change;
- local-total-power control;
- orientation or spatial-frequency control.

Selection roles, criteria, values, unit identifiers, image identifiers, and trace
identifiers must be saved before detailed rendering.

**Required panels:** source image, FEM path, representative retinal frames,
local power at selected positions, digital-twin-grating-estimated tuning,
predicted map, twin map,
and direct FEM-minus-stabilized difference maps. Add a power-matched
Fourier-phase-scrambled or Fourier-phase-shifted movie and its twin map: the spectral proxy
should remain fixed while any resulting twin-map change quantifies information
that power alone cannot represent. Predicted quantities and twin outcomes must
be visually distinguished.

**Decision gate:** pause for human inspection. Advance only after the aperture,
target tensor, map support, color scaling, temporal summary, calibration family,
and visible successes or failures are understood and frozen without inspecting
the reserved final-test bank.

**Expected runtime:** 30–90 minutes for a targeted example set.

### Stage 5: SSI prediction and spatial-mechanism diagnostic

**Objective:** test direct SSI prediction and map-mediated SSI prediction as two
separate results.

**Motivation:** SSI is nonlinear and depends on the distribution of activation
over space. A scalar predictor can predict SSI statistically but cannot reveal
why the map sharpened or broadened.

**Frozen target-map contract:** the observed target is the post-output-activation
RR100 spatial firing-rate map returned by the canonical rate-map computation,
after applying the frozen RR100 population view and clamping numerical negative
values to zero. Its units are hertz. Moving and stabilized maps use identical
40-frame alignment, spatial stride, valid support, and edge convention. The
primary SSI support is the exact common valid-support mask shared by observed
and predicted maps; it may not change by condition. Pre-activation feature or
readout tensors are separate diagnostics and must not be called activation maps
in this analysis.

**Frozen calibration contract:** the primary power-to-rate calibration is a
nonnegative monotonic link fit on training identities only, with its intercept,
static-image terms, dynamic-power terms, regularization, and any unit pooling
saved explicitly. The link must output finite nonnegative rates in hertz; no
post hoc offset, clipping rule, normalization, or support change may be chosen
using validation or final-test SSI. Moving and stabilized predictors are
calibrated to their respective rate maps and subtracted only afterward. The
specific link family is chosen and frozen during Stage 4 development; a
softplus-affine link with nonnegative dynamic-power slope is the default.

**Analysis A — direct prediction:** evaluate whether scalar power predicts
held-out SSI. Treat direct unsigned-power prediction of signed
FEM-minus-stabilized SSI change as exploratory. The mechanistic signed SSI
prediction is the difference between SSI computed from separately predicted
moving and stabilized maps.

**Analysis B — map-mediated diagnostic:** apply the frozen calibration, then
calculate SSI from the predicted nonnegative maps with the same estimator,
spatial support, temporal definition, and spike weighting used for the twin.

Preserve and report separately:

- mean firing rate;
- expected spikes;
- raw information numerator;
- bits per second where defined;
- SSI in bits per spike;
- instantaneous SSI;
- expected-spike-weighted mean instantaneous SSI;
- SSI calculated from a time-averaged map.

Prefer FEM-minus-stabilized differences over ratios, particularly where the
stabilized value can approach zero. The stabilized baseline must retain its
declared construction and must not be silently replaced with a different static
or centred baseline.

The power-matched phase control must pass through the same frozen calibration.
Because its orientation-aware spatial- and temporal-frequency power is matched,
any reproducible observed map or SSI difference that the predictor cannot
express is a direct lower bound on the failure of a power-only account.

**Interpretation:**

- scalar and map-derived SSI succeed: stronger evidence for a spatial power
  mechanism;
- scalar succeeds but map-derived SSI fails: predictive association without a
  demonstrated spatial explanation;
- scalar fails but map-derived SSI succeeds: global averaging hid local
  information;
- both fail: current power representation is insufficient.

**Expected runtime:** 30–60 minutes after predicted maps have been generated.

### Stage 6: held-out population map analysis

**Objective:** determine whether the local spectral mechanism generalizes across
units, images, and FEM traces.

**Motivation:** targeted maps establish meaning but cannot support a population
claim. Population analysis is justified only after visible map-level successes,
dissociations, and controls have been characterized.

**Models to compare:**

1. whole-image total dynamic power;
2. whole-image orientation-aware tuning-weighted power;
3. local total dynamic power;
4. local orientation-collapsed tuning-weighted power;
5. local orientation-aware tuning-weighted power;
6. local contrast or energy;
7. power-matched Fourier-phase-scrambled or Fourier-phase-shifted input;
8. the complete digital twin as a reference, not as a power model.

**Primary map outcomes:**

- spatial correlation of predicted and twin activation maps;
- correlation of FEM-minus-stabilized difference maps constructed by separately
  predicting moving and stabilized maps and then subtracting;
- explained variance in map magnitude;
- map centre and spatial extent errors;
- activation-map sharpness;
- SSI and FEM-minus-stabilized SSI change derived from the maps.

**Validation:** use the image and FEM-trace identities reserved before Stage 4
as a single untouched final test after every aperture, proxy, calibration,
support, temporal summary, and example-selection rule has been frozen. Prevent
leakage through preprocessing, calibration, example selection, normalization,
or unit filtering. If nested cross-validation is used instead, label it as such
and keep every decision inside the corresponding training fold.

**Inference:** distinguish two targets explicitly:

- inference conditional on the fixed image and trace bank uses
  session-balanced resampling of sessions and units;
- generalization to new images and FEM trajectories uses a crossed multiway
  bootstrap or equivalent crossed random-effects procedure over sessions,
  images, and traces, preserving the unit-within-session structure and the
  image-by-trace crossing.

Report both when feasible, along with the fractions of units and sessions with
positive improvements. Do not treat condition rows sharing images or traces as
independent replicates.

**Reporting:** preserve positive examples, prediction-without-response
dissociations, response-without-prediction dissociations, and negative controls.
Every aggregate result must be traceable to the saved unit and condition rows.

**Expected runtime:** 4–12 hours, depending on condition-bank size, map stride,
cache reuse, and GPU batching.

## 6. Statistical and validation contracts

### Cross-validation

- Recorded-grating analyses hold out complete experimental trials.
- Before Stage 4, split natural-image data into a development bank and an
  untouched final-test bank with disjoint image identities and disjoint FEM
  trace identities. Save and hash this split.
- Natural-image development analyses use crossed folds that hold out both image
  identities and FEM-trace identities. Final evaluation occurs once on the
  untouched bank after all choices are frozen.
- Calibration from power to rate or activation is learned on training folds only.
- Example selection for explanatory figures must either be input-only or be
  performed within a declared exploratory subset that is not used for final
  evaluation.
- If the available bank cannot support an untouched test, use fully nested
  crossed validation and describe the result as nested cross-validated evidence,
  not as performance on a pristine final test.

### Uncertainty

Population point estimates will weight recording sessions equally. For claims
conditional on the fixed stimulus bank, confidence intervals use hierarchical
bootstrap sampling of sessions followed by units within sessions. For claims
that generalize to new stimuli, confidence intervals use crossed multiway
resampling of sessions, images, and traces, with units nested within sessions
and condition identities rebuilt from the sampled image-by-trace crossing.
Cluster-robust or crossed random-effects estimates may be reported as sensitivity
analyses. Unit-level distributions and the fraction of positive units will be
reported alongside session-balanced estimates. Every interval must state which
sampling dimensions it supports.

### Recorded-response noise

Raw 333-ms recorded-window \(R^2\) must be interpreted relative to spike-count
noise. Where repeated conditions permit, estimate split-half reliability and a
noise ceiling. Report Poisson deviance or predictive log likelihood as a
complement to ordinary \(R^2\), and provide repeat-averaged tuning results as a
separate target rather than conflating them with single-trial prediction.

### Model comparison

Primary comparisons are paired within unit and condition. Added complexity must
improve held-out prediction, not merely training fit. The orientation-aware
single predictor is the primary spectral formulation based on the completed
grating checkpoint; the separately fitted orientation-increment model remains a
diagnostic because its benefit was heterogeneous across units.

## 7. Figure and terminology standards

Every iterative and production figure must be interpretable without knowledge
of internal scripts or output directories.

Figures must include:

- a question or conclusion in the main title;
- the biological target: recorded firing rate, digital-twin firing rate,
  activation map, or SSI;
- sample size and validation split;
- physical units on axes;
- a plain-language definition of \(R^2\), SSI, or any less familiar metric;
- a clear distinction between observed input, derived power proxy, digital-twin
  outcome, and recorded response;
- an explanation of lines, points, intervals, and color scales.

Avoid unapproved shorthand such as `F0`, `SF×TF`, `RF`, `H²`, `radial`,
`oriented`, `full twin`, `CV`, or internal model keys in reader-facing panels.
Use, for example, “phase-averaged grating response,” “spatial- and
temporal-frequency tuning,” “receptive-field-local,” “digital-twin firing rate,”
and “held-out fraction of firing-rate variance explained.” Internal identifiers
may be retained in tables and manifests for reproducibility.

## 8. Provenance and artifact requirements

Each checkpoint must save:

- the exact command and configuration;
- source file paths and cryptographic hashes;
- code or repository identity;
- unit, session, image, trace, condition, and response-row identifiers;
- raw predictors, transformed predictors, outcomes, and held-out predictions;
- fold assignments and fitted calibration parameters;
- selection criteria and values for examples;
- machine-readable summaries;
- descriptive PNG and PDF figures;
- a manifest stating whether the result is a smoke test, targeted map checkpoint,
  or production population result;
- explicit supersession relationships when a correction is made.

Prior interpretable outputs must not be overwritten. Invalid outputs should be
preserved but clearly marked as unsuitable for scientific inference.

## 9. Decision criteria for the scientific conclusion

Evidence that power predicts an outcome requires improved held-out prediction
over appropriate global and local controls.

Evidence for a **spatial power-routing mechanism** requires more:

1. the receptive-field and coordinate equivalence tests pass;
2. local unit-specific tuning improves prediction of activation-map location and
   magnitude beyond local total power and contrast controls;
3. predicted maps reproduce a meaningful component of held-out FEM-induced SSI
   changes;
4. the effect generalizes across recording sessions and is not driven solely by
   a small number of units;
5. rate, expected spikes, raw information, and normalized SSI support a coherent
   interpretation;
6. the claimed signed FEM effect comes from separately predicted moving and
   stabilized maps rather than assigning sign to nonnegative power;
7. power-matched phase controls do not reveal response or map changes large
   enough to account for the proposed effect while leaving the proxy unchanged.

Failure at any stage narrows the conclusion rather than invalidating all prior
results. In particular, direct scalar prediction may remain valid even if the
spatial mechanism is unresolved.

## 10. Planned execution order and estimated runtime

| Order | Stage | Expected runtime |
|---:|---|---:|
| 0 | Quarantine invalid caches and export clean grating-only tuning | 1–2 hours |
| 1 | Incrementally build and audit corrected spectral cache | 10–20 minutes per complete round, plus 20–40 minutes final audit |
| 2 | Corrected whole-image scalar prediction | 15–30 minutes |
| 3 | Calibrate receptive-field and spatial-coordinate contract | 1–3 hours |
| 4 | Targeted local power-derived activation maps on development identities | 30–90 minutes |
| 5 | Direct and map-mediated SSI analysis | 30–60 minutes after map generation |
| 6 | Untouched-test population map analysis | 4–12 hours |

These estimates assume the current workstation, existing input and response
caches, and GPU batching where applicable. Each stage ends with an explicit
review checkpoint before the next stage begins.

The completed three-condition natural-image response checkpoint is a qualitative
bridge into Stages 2 and 4. It does not remove or shorten quarantine, the
50-round corrected-cache endpoint, held-out scalar expansion, receptive-field
calibration, spatial map comparison, phase controls, or SSI analysis.

## 11. Immediate next action

The next implementation step is Stage 0: inventory and quarantine every
consumer of the invalid three-round cache, export a clean grating-only tuning
artifact, and add fail-fast provenance checks. Stage 1 can then build spectra
incrementally over complete balanced rounds. Confirmatory scalar results must
wait for the frozen first-50-round snapshot and the pre-Stage-4 development/test
split.
