# Analysis plan: can FEM-driven neural changes be explained by redistribution of retinal image power?

Date: 2026-08-13; map-support phase factorial revised 2026-08-14

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

The phase-residual question also has two distinct spatial scales:

1. **within-position phase sensitivity:** phase within the effective receptive
   field of one translated readout position changes that position's response;
2. **cross-position phase organization:** phase relationships across the union
   of translated readout positions organize where activation falls across the
   complete activation map.

Because SSI is calculated from the complete spatial activation map, the second
question is the primary lightweight phase diagnostic. The canonical rate-map
implementation convolves each unit's learned spatial readout across the full
core feature map with valid padding. One scored map is therefore driven by the
union of input support across all translated positions, not by one median-sized
effective-RF window. RF-local metamer synthesis remains a possible follow-up,
but it is not required before testing whether global map-input power is
sufficient for map sharpening.

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

A targeted three-condition checkpoint compared receptive-field-local
orientation-aware and orientation-collapsed power with exact digital-twin
responses for five units. Its original version included trace 638, whose
32-frame model history crossed the selected fixation boundary. That original
result, including the claim that orientation-aware power improved agreement for
all five units, is superseded and must not be cited as clean-history evidence.

After the response-cache quarantine, the checkpoint was rerun twice using only
traces whose full model history remained within fixation. Conditions were
selected using input quantities only; neural outcomes were not used for
condition or unit selection.

The two clean selection policies produced different descriptive summaries:

- **Original-reference sensitivity:** retained the original all-condition input
  thresholds while excluding ineligible traces. Orientation-aware power improved
  response-modulation-magnitude agreement for all five units. Unit 56 changed
  from \(r=-0.90\) for orientation-collapsed power to \(r=+0.91\) for
  orientation-aware power; unit 65 changed from \(r=-0.13\) to \(r=+0.97\).
- **Clean-reselected checkpoint:** recalculated the input-distribution thresholds
  within the clean-history subset. Orientation-aware power improved agreement
  for units 0 and 56 but worsened it for units 28, 64, and 65. Unit 56 remained
  strongly positive for orientation-aware power (\(r=+1.00\)), but the
  orientation-collapsed predictor was also positive (\(r=+0.77\)).

This selection sensitivity shows that the earlier five-of-five statement is not
robust evidence. The result that survives both clean policies is narrower: unit
56 remains a positive example in which orientation-aware power tracks the
root-mean-square magnitude of the moving-versus-stabilized response. Its
orientation-aware correlation with signed mean-rate change is negative under
both clean policies, preserving the qualitative dissociation between modulation
magnitude and the direction of the firing-rate change.

A subsequent expanded checkpoint tested 100 clean-history conditions: one
input-only, deterministically selected condition for every image, with 100
distinct eye-movement traces. The dramatic three-condition effects did not
generalize. Orientation-aware versus orientation-collapsed correlations with
response-modulation magnitude were, respectively: unit 0, \(0.14\) versus
\(0.16\); unit 28, \(0.31\) versus \(0.27\); unit 56, \(0.17\) versus \(0.09\);
unit 64, \(0.02\) versus \(0.02\); and unit 65, \(-0.08\) versus \(-0.15\).
Thus orientation information produced small descriptive correlation increments
for units 28, 56, and 65, no meaningful change for unit 64, and a small decrement
for unit 0.

Absolute predictive performance remained weak. Five-fold held-out \(R^2\) was
positive only for unit 28, improving from \(0.016\) to \(0.040\). Both models
had negative held-out \(R^2\) for units 0, 56, 64, and 65. For unit 56, the
orientation-aware correlation with signed mean-rate change was \(r=0.19\),
similar to its \(r=0.17\) correlation with modulation magnitude. The earlier
unit-56 magnitude-versus-sign dissociation therefore does not generalize across
the expanded condition set.

The same 100 clean-history conditions were then evaluated for all 61
recorded-spatial-frequency-validated units from 15 sessions. Receptive-field-local
power had modest absolute predictive value for digital-twin response-modulation
magnitude: session-balanced mean correlation was \(r=0.266\) for
orientation-collapsed power and \(r=0.268\) for orientation-aware power.
Session-balanced mean five-fold held-out \(R^2\) was \(0.033\) and \(0.027\),
respectively; 40 of 61 units had positive held-out \(R^2\) for the collapsed
model and 37 of 61 for the orientation-aware model.

There was no supported population advantage from adding orientation specificity.
The session-balanced correlation difference was \(+0.0014\), with hierarchical
95% interval \([-0.0186,+0.0176]\), and 52.5% of units improved. The held-out
\(R^2\) difference was \(-0.0058\), with interval \([-0.0186,+0.0044]\), and
45.9% of units improved. Orientation-aware power reduced standardized absolute
error slightly on average, but its interval also included zero. Thus the current
evidence supports a modest relationship between local spectral power and
response-modulation magnitude, not a general benefit of the orientation-aware
power formulation on natural images.

Response extraction was verified against sanitized raw response shards and the
corresponding clean assembled rows to a maximum discrepancy below
\(3.8\times10^{-9}\) Hz. All selected traces pass the clean-history gate, and
exact rerendering reproduced the selected spectral arrays. This validates the
targeted joins but not the spectral aperture or a population inference.

The checkpoint remains provisional for six reasons:

- the original unit correlations contained only three conditions and were highly
  unstable; the expanded checkpoint corrects this but does not rescue those
  example-level claims;
- the conclusion changes with the input-only selection reference population;
- the outcome is a scalar temporal response summary, not a spatial activation
  map;
- the analysis uses the current approximate receptive-field aperture;
- the responses are digital-twin predictions, not recorded natural-image
  responses;
- the quarantined 577-trace subset is provisional pending a fresh stratified
  1,000-trace cohort.

**Decision from this checkpoint:** treat response-modulation magnitude as the
primary scalar activation outcome and signed mean-rate change as a distinct
secondary outcome, but do not treat any of the three-condition correlations or
the unit-56 dissociation as stable evidence. The 61-unit checkpoint supports
modest scalar prediction by receptive-field-local power, while providing no
population evidence that orientation-aware weighting improves upon the
orientation-collapsed model. The corrected production cache, prespecified
held-out evaluation, and spatial-map diagnostic remain necessary; no
activation-map or SSI mechanism has yet been explained.

Relevant artifacts are:

- `declan/fig4_active_sensing/analyze_rr100_natural_image_rf_local_oriented_power_response_checkpoint.py`;
- `outputs/fig4_active_sensing/rr100_natural_image_rf_local_oriented_power_response_checkpoint_v2_clean_history/`;
- `outputs/fig4_active_sensing/rr100_natural_image_rf_local_oriented_power_response_checkpoint_v2_clean_history_original_reference/`;
- `outputs/fig4_active_sensing/rr100_natural_image_rf_local_oriented_power_response_expanded_n100_clean_history_v1/`;
- `outputs/fig4_active_sensing/rr100_natural_image_rf_local_oriented_power_population_n100x61_clean_history_v1/`.

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

The initial corrected response request contained 100 images, 1,000 traces, and
100 balanced rounds. A subsequent audit found that 423 trace histories began
before the selected fixation because the validity gate checked trial, display
calibration, and finite samples but not the fixation-epoch boundary. Although
all scored frames were inside fixation, those response histories violated the
declared model-input contract.

The defect is quarantined in the live cache. Original shards containing affected
rows were moved to a recoverable quarantine tree, sanitized active shards retain
only valid rows, and resumed scoring skips all 423 flagged traces. The active
provisional target is therefore 100 images by 577 clean traces. The earlier
22-round count and any proposed first-50-round snapshot from the original trace
bank are not valid confirmatory endpoints.

The 577-trace subset may support explicitly provisional engineering checks and
the clean-history targeted checkpoint in Section 2.2. It does not preserve the
intended fully stratified 1,000-trace population. The confirmatory endpoint is a
fresh, frozen, fully stratified 100-image by 1,000-trace cohort, with all 100
balanced rounds (100,000 moving movies) completed and matched stabilized
responses. Earlier complete-round snapshots are convergence diagnostics only.

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
map prediction beyond untuned local power and local contrast. The Stage-2A
map-support phase manipulation bounds what global power cannot explain; a
stronger bound on this local model requires the conditional RF-local control.

### Hypothesis 3: power-derived maps account for SSI

SSI calculated from predicted activation maps explains a meaningful component
of the FEM-minus-stabilized SSI change on held-out conditions.

### Hypothesis 4: map-wide phase organization contributes beyond power

For the exact 32-frame by 151-by-151 history cube that generates one activation
map, preserve the complete three-dimensional Fourier magnitude while replacing
phase across the full map support. If global map-input power is sufficient, the
FEM-minus-stabilized activation-map and SSI contrast should survive under a
phase field shared between the paired conditions. If the contrast changes,
global power is insufficient and phase-dependent spatial localization across
the tiled map positions contributes to the result.

This is intentionally coarser than an RF-local sufficiency claim. Because local
power may move between map positions, a reduced SSI effect cannot by itself
distinguish a nonlinear within-RF phase computation from redistribution of
energy among tiled positions. Persistence under this harsher manipulation is
strong evidence for power sufficiency; loss motivates the complementary
power-reduced/phase-preserved arm and, only if still necessary, an RF-local
follow-up.

An apparent loss is not sufficient evidence when phase destruction also makes
the input structurally out of distribution. Exact spectrum, marginal histogram,
and contrast can coexist with noise-like higher-order structure unlike the
twin's natural-image training inputs. Hypothesis 4 is therefore evaluated only
after the structural-distribution and generic-degradation controls in Stage 2A;
otherwise a loss is labelled ambiguous rather than phase-dependent.

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

**Implementation status (2026-08-13): complete.** The gate records nine known
consumers rejecting the superseded cache in executable smoke tests, 16 affected
artifact directories inventoried and labelled, and an exact clean export of the
100-unit, 8-SF by 25-TF by 4-orientation grating tensor. Audit artifacts are in
`outputs/fig4_active_sensing/rr100_power_routing_stage0_quarantine_v1`; the clean
tuning artifact is in
`outputs/fig4_active_sensing/rr100_grating_only_orientation_tuning_v1`.

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
- create a clean grating-only tuning artifact containing the fixed-retina
  digital-twin SF-by-TF-by-orientation response tensor, its held-out tuning
  predictions, and recorded-spatial-frequency cohort validation, with no
  natural-movie-derived fields; interpolate analysis-grid weights from this
  artifact only after a corrected spectral grid has passed Stage 1;
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

### Stage 0B: freeze the joint spatial-frequency-by-temporal-frequency-by-direction tuning contract

**Targeted implementation checkpoint (2026-08-14): complete; population
parametric gate remains open.** The full fixed-retina signed-temporal-frequency
cache has been reorganized into a complete empirical tensor for all 100 RR100
units with shape 100 units by 8 spatial frequencies by 13 temporal-frequency
magnitudes by 8 motion directions. No twin inference was rerun. The primary
artifact is
`outputs/fig4_active_sensing/rr100_joint_sftf_direction_tuning_checkpoint_v2_smooth/rr100_empirical_joint_sftf_direction_tuning.npz`.

The empirical tensor, not one preferred-orientation label and not a parametric
fit, is the primary routing object. Preserve signed above-blank responses and
their nonnegative excitatory component separately. Do not average opposite
signed temporal frequencies before direction tuning has been characterized.
Any scalar orientation or direction preference used for unit annotation must
be explicitly derived from the spatial-frequency-by-temporal-frequency support.
No hard response cutoff is used. For each cell, normalize the positive angular
response into (q(\theta\mid sf,tf)), define direction-summed sensitivity
(S(sf,tf)), and average the conditional profiles with smooth weights
proportional to (S(sf,tf)^\alpha). The primary development definition uses
(\alpha=2); (\alpha=1) and (\alpha=3) are saved sensitivity controls.
The (\alpha=1) result is the ordinary response-weighted marginal, while
(\alpha>1) progressively emphasizes the unit's passband without creating a
discontinuity at an arbitrary half-maximum boundary. Single-peak estimates
remain saved diagnostics. Units without positive above-blank support receive
no angular preference. The earlier 50%-of-peak artifact remains preserved as
an interpretable superseded checkpoint and is not the promoted contract.

The first map-first examples show that this distinction is material. One unit
is well described as orientation selective without direction selectivity;
another has strong direction tuning locally that is diluted by full-support
marginalization; and another changes angular structure across frequency. The
independent 18-angle static BackImage probe is retained as a cross-check, not
pooled with the dynamic tensor, because its spatial-frequency, phase, canvas,
and readout contracts differ.

Two conventional parametric models have been fitted only to the five
predeclared example roles as diagnostics: a separable log-Gaussian
spatial-frequency and temporal-frequency envelope multiplied by first and
second circular harmonics, and an extension allowing both harmonics to vary
smoothly with log spatial and temporal frequency. Deterministic five-fold
condition-held-out prediction is required. The extension improved held-out
(R^2) by 0.134 for the strong-direction example, 0.091 for the
frequency-dependent example, 0.057 for the static/dynamic dissociation, and
0.002 for the orientation-only example. This supports a heterogeneous
contract: separability is adequate for some units but cannot be assumed for all
units. The weak-response control was correctly excluded from parametric
fitting. Artifacts and residual views are in
`outputs/fig4_active_sensing/rr100_joint_sftf_direction_tuning_checkpoint_v2_smooth`.

**Cartographer constraint:** Stage 3 may continue to diagnose canvas context,
translation, and aperture support, but it must not freeze an aperture-to-tuning
calibration or propagate the old four-bin preferred-orientation field into
Panels D/E or routing maps. Candidate localized probes should be selected from
the full empirical tensor, with the spatial-frequency, temporal-frequency,
bar-orientation, and drift-sign condition recorded. Units whose angular tuning
depends on frequency must remain visible as dissociations rather than being
collapsed into one global orientation. The observed 51-input-to-151-input
direct-core canvas effect remains a blocking implementation diagnostic, while
the separate native empirical-tuning-to-map transfer has not yet been tested
under a matched estimand.

**Decision gate:** inspect the selected empirical tensors and parametric
residuals before fitting the hierarchy to all units. Freeze the smooth-weight
exponent, interpolation rule, responsiveness gate, and treatment of frequency-
dependent angular tuning before replacing any Figure 4 orientation field or
generating production routing weights.

### Stage 1: rebuild and validate the corrected natural-image spectral cache

**Development implementation checkpoint (2026-08-13): complete, but the
confirmatory gate remains open.** A provisional cache now covers 27 complete
clean-history response rounds: 15,579 conditions, 100 images, and 577 traces.
All traces occur exactly once per round, all 32-frame histories lie within the
selected fixation, spectra are written directly to response matrix rows, and
86 boundary/random rerenders reproduce cached spectra exactly. Quarantining
423 traces breaks image balance: each round has 99 nonempty images and total
image degree ranges from 128 to 185. Therefore this cache is restricted to
engineering/development fits with explicit image control. It does not satisfy
the frozen replacement 100-by-1,000 confirmatory endpoint. Artifacts are in
`outputs/fig4_active_sensing/rr100_clean_history_spectral_cache_rounds000_026_n027_v1`.

**Objective:** incrementally produce a trustworthy condition-by-spectrum cache
for the replacement fully stratified cohort and freeze its complete 100-round
production snapshot.

**Motivation:** all downstream scalar prediction depends on exact alignment
between retinal input, FEM trace, response outcome, and spectral predictor. The
identified row-order error is sufficient to invalidate the previous fits.

**Method:**

- treat the existing three-round, 3,000-movie analysis only as a pilot/debugging
  checkpoint;
- require the fresh 1,000-trace cohort manifest to certify that every 32-frame
  model history and 40-frame scored segment lies within the selected fixation;
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

**Stopping and snapshot contract:** the primary confirmatory snapshot is the
complete 100-image by 1,000-trace crossing: rounds 0–99 after all 100 are
complete. Earlier complete-round snapshots may be used for engineering
validation and labelled convergence plots only. Round inclusion is determined
without neural outcomes. The manifest must record whether an artifact is pilot,
interim, or complete-production output.

**Required controls:** orientation-resolved power must sum to radial power;
all declared rows must be populated exactly once; response and spectrum identity
arrays must agree exactly.

**Decision gate:** no response fitting may proceed unless every identity check
passes, independent rerenders reproduce the cache within numerical tolerance,
and the requested analysis tier has the declared complete rounds. Confirmatory
claims wait for the frozen replacement 100-round snapshot.

**Expected runtime:** approximately 10–20 minutes per newly available complete
round, plus 20–40 minutes for final audits; actual throughput must be updated
from the first incremental build rather than inferred from the obsolete
three-round estimate.

### Stage 2: corrected whole-image scalar prediction

**Development implementation checkpoint (2026-08-13): complete; reserved-test
and confirmatory gates remain open.** The provisional 27-round cache was split
by identity before fitting: 20 images and 116 traces are frozen as an unopened
final-test bank, while 9,868 conditions crossing 80 development images with 461
development traces were evaluated. Three deterministic 5-by-5 crossed folds
held out both image and trace identities, and fitting and scoring gave every
image equal total weight to correct the provisional schedule's image imbalance.

For response-modulation RMS, session-balanced mean held-out \(R^2\) was 0.224
for total supported dynamic power, 0.204 for spatial-by-temporal direct-F0
power, and 0.197 for orientation-aware direct-F0 power. Simple dynamic-energy
and static-image controls reached 0.311. Adding radial or orientation-aware
tuned power to those controls changed \(R^2\) only to 0.313 and 0.314. The
paired orientation-aware-minus-radial difference was \(-0.007\), with
hierarchical 95% interval \([-0.023,+0.008]\). The orientation-aware increment
over image and energy controls was \(+0.003\), interval
\([-0.001,+0.010]\). Thus whole-movie input energy predicts response-modulation
magnitude, but this development result provides no evidence that detailed
unit-specific orientation tuning is the source of that prediction.

Direct SSI-change prediction was much weaker. Orientation-aware tuned power
alone achieved session-balanced mean held-out \(R^2=0.023\), interval
\([-0.010,+0.065]\). Image and energy controls achieved 0.089; adding radial
tuned power increased this to 0.105, a paired increment of 0.016 with interval
\([+0.008,+0.025]\), whereas adding orientation-aware power produced an
unsupported increment of 0.004, interval \([-0.002,+0.010]\). This is evidence
that scalar input summaries contain some information about SSI, not evidence
that power explains map sharpening. The spatial origin and map-mediated SSI
prediction remain entirely untested until Stages 3--5.

Artifacts are in
`outputs/fig4_active_sensing/rr100_clean_history_whole_movie_power_stage2_v1`.
The result is explicitly development-only because the replacement balanced
100-by-1,000 cache is unfinished and the frozen final-test identities remain
unopened.

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
- the Stage-2A map-support amplitude-by-phase factorial. Its exact contract is
  equality of the unwindowed three-dimensional Fourier magnitude of each
  32-frame history cube, with conjugate symmetry, dimensions, mean, RMS
  contrast, histogram, and input range audited separately. The later
  Hann-windowed or binned predictor is a phase-sensitive localization
  diagnostic and is not the definition of exact equality.

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

### Stage 2A: map-support amplitude-by-phase factorial smoke test

**Development checkpoint status (2026-08-14): complete; decision gate not
passed.** The exact raw-cube Fourier-amplitude contract passed, with maximum
relative amplitude error of $1.9\times10^{-8}$ and negligible imaginary and
Hermitian-symmetry errors. Across the six units selected before viewing their
natural-image responses, changing FEM amplitude to stabilized amplitude while
holding FEM phase fixed reproduced the original FEM-minus-stabilized effect
closely: $r=0.9999$ for mean rate (mean absolute discrepancy $0.00055$ Hz) and
$r=0.9961$ for SSI (mean absolute discrepancy $0.00146$ bits/spike). This is
evidence that the amplitude/power change is important in this concrete example;
it is not yet a population result or an explanation of the spatial map.

The shared-random-phase arms did not satisfy the input-distribution control.
Depending on seed, approximately 7--10% of reconstructed values fell outside
the twin's canonical input range, histogram Wasserstein distance reached about
0.065, tiled local energy was almost completely redistributed, and response
effects varied materially among the three predeclared seeds. Consequently,
these arms cannot support a clean conclusion about phase sufficiency or phase
necessity. The checkpoint artifacts are in
[`rr100_map_support_amplitude_phase_factorial_stage2a_v1`](/home/declan/VisionCore/outputs/fig4_active_sensing/rr100_map_support_amplitude_phase_factorial_stage2a_v1)
and the focused human-review artifacts are in
[`rr100_map_support_amplitude_phase_factorial_stage2a_review_v1`](/home/declan/VisionCore/outputs/fig4_active_sensing/rr100_map_support_amplitude_phase_factorial_stage2a_review_v1).

A post-checkpoint symmetric phase-support audit found no FEM-phase-invalid bins
at relative amplitude thresholds through $10^{-8}$ of the FEM spectral peak.
At the deliberately conservative $10^{-4}$ threshold, bins classified as weak
in the FEM source contained only $3.84\times10^{-4}$ of the stabilized target
spectral energy. Thus the implemented stabilized-amplitude/FEM-phase result is
not materially exposed to unsupported FEM phase in this example. This does not
remove the need for the symmetric validity guard in every future factorial. In
the reverse, deliberately unused pairing, stabilized-phase-invalid bins contain
6.55% of FEM target energy, confirming why the guard cannot be one-sided. The
post-checkpoint tables and figure are in
[`rr100_map_support_amplitude_phase_factorial_stage2a_method_audit_v1`](/home/declan/VisionCore/outputs/fig4_active_sensing/rr100_map_support_amplitude_phase_factorial_stage2a_method_audit_v1).

**Objective:** determine, before further RF-local engineering, whether complete
map-input power or map-wide phase organization is the more informative axis for
the FEM-induced activation-map and SSI effect.

**Unit of manipulation:** one exact model input history cube for one scored map,
with shape 32 frames by 151 by 151 pixels. This is a targeted spectral-control
input, not a claim to be a physically coherent source movie. The rate-map code
already evaluates scored history cubes as separate batch items, so the first
checkpoint can be exact at the model-input level without constructing one
surrogate source image or a coherent 72-frame surrogate movie.

For each condition $c\in\{F,S\}$, where $F$ is FEM and $S$ is the declared
stabilized baseline, write the raw history-cube transform as

\[
\mathcal F\{X_c\}=A_c e^{i\phi_c}.
\]

Use correctly Hermitian phase fields so inverse transforms remain real. The
primary first-pass factorial uses two power levels and two phase levels:

| Power spectrum | Phase field | Construction | Role |
|---|---|---|---|
| FEM, $A_F$ | FEM, $\phi_F$ | $\mathcal F^{-1}(A_F e^{i\phi_F})$ | original FEM input |
| FEM, $A_F$ | shared random, $\psi_s$ | $\mathcal F^{-1}(A_F e^{i\psi_s})$ | phase destroyed, FEM power retained |
| stabilized, $A_S$ | FEM, $\phi_F$ | $\mathcal F^{-1}(A_S e^{i\phi_F})$ | FEM phase retained, FEM power removed |
| stabilized, $A_S$ | shared random, $\psi_s$ | $\mathcal F^{-1}(A_S e^{i\psi_s})$ | both FEM phase and FEM power removed |

The original stabilized input
$\mathcal F^{-1}(A_S e^{i\phi_S})$ remains the biological/counterfactual
baseline but is not used as a source of phase in bins where stabilized temporal
amplitude is effectively zero. This avoids promoting numerically undefined
stabilized phases into nonzero FEM-frequency bins. Apply the same rule
symmetrically to every source-phase/target-amplitude pairing, including
$A_S e^{i\phi_F}$. Before rendering, define a predeclared source-phase validity
floor relative to the source spectrum and report both the fraction of target
bins and the fraction of target spectral energy assigned phase from invalid
source bins. The arm passes only when that target-energy fraction is negligible
under a documented threshold-sensitivity analysis. Otherwise reject the arm or
use a separately labelled construction; do not silently copy, zero, or
randomize unsupported phase. The same $\psi_s$ must be used for the FEM-power
and stabilized-power randomized arms within each seed, removing phase-seed
variation from their paired power contrast.

The two-by-two factorial reports, without assuming additivity:

- the phase effect at FEM power;
- the phase effect at stabilized power;
- the power effect under FEM phase;
- the power effect under shared random phase;
- their interaction;
- every condition's difference from the original stabilized map;
- the original FEM-minus-stabilized effect as the reference quantity.

After the first map checkpoint passes, a predeclared power interpolation may
replace the two-level comparison with

\[
A_\lambda=\sqrt{\lambda A_F^2+(1-\lambda)A_S^2},\qquad 0\leq\lambda\leq1,
\]

under both FEM and shared-random phase. This supplies a power dose-response
without changing the spectral shape by an arbitrary contrast scalar. Do not
run or tune this sweep before the endpoint factorial is visually understood.

Do not force these terms into percentages that sum to 100%. A descriptive
fraction of the original SSI effect may be shown only when the original paired
difference is stable and bounded away from zero; absolute paired differences,
rate, expected spikes, raw information, and bits per second remain primary.

**Power contract:** exact equality refers only to the unwindowed raw history
cube and its complete three-dimensional Fourier magnitude. This avoids the
earlier error of scrambling a 72-frame support and judging equality on a
different Hann-windowed 40-frame subset. A canonical Hann-windowed
spatial-by-temporal-frequency calculation may be reported as a secondary,
explicitly phase-sensitive localization diagnostic. Its mismatch is expected
after phase replacement and must not be described as failure of the exact raw
cube power contract.

**First checkpoint:** one development image, one clean-history trace, one
representative scored frame, an auditable predeclared unit set, and several
predeclared random phase seeds. Unit roles must be assigned from intact
development responses and already-frozen power-model predictions before any
surrogate responses are inspected. Include, where available: strong intact SSI
sharpening unexplained by the current power model; a power-consistent SSI or
activation-map example; high predicted power shift with weak SSI change; weak
predicted power shift with strong SSI change; a weak-effect or negative control;
and units spanning effective-RF scale and validated SF/orientation tuning. Require
an intact effect bounded away from zero for roles intended to test preservation
or loss. Save unit identity, role, criterion, criterion value, reference
condition/frame, RF scale, tuning metadata, mean rate, intact SSI effect, and
whether selection was algorithmic or user-requested. Save the input cubes,
current-frame views, raw activation maps, direct map differences, instantaneous
SSI, mean rate, expected spikes, and raw information quantities. Use shared map
color scales plus per-unit companion scales. Stop for inspection before
additional frames, units, or population summaries.

The completed checkpoint used six units selected only from fixed-retina grating
properties before natural-image responses. That is auditable and suitable for
an initial engineering smoke test, but it does not satisfy the full
outcome-role list above. Any accepted rerun must replace or augment it with the
predeclared role-based set before population expansion.

**Input audits:** numerical Fourier-magnitude equality; symmetric source-phase
validity for every target-amplitude pairing; conjugate symmetry and imaginary
reconstruction residual; mean and RMS contrast; histogram and pixel range;
global and local phase-retention measures; local-power redistribution across
tiled positions; global and patchwise higher-order statistics such as kurtosis;
and the fraction of input values outside the twin's training range. Where
feasible, also measure distance in an early frozen twin representation rather
than relying only on marginal pixel statistics. Begin with the raw exact
construction. Clipping, rank matching, IAAFT, or other histogram projection is
not allowed without rerunning every power and phase audit and labelling the
result as a different control.

Exact magnitude, histogram, and contrast matching cannot establish that a
random-phase input is in distribution: natural-image structure is carried
strongly by phase, and a noise-like reconstruction can remain structurally far
from the twin's training inputs. A loss of sharpening under such an input is
therefore ambiguous between phase-dependent computation and generic
out-of-distribution degradation. As a calibration—not a proof of natural-image
validity—apply the same accepted manipulation to the validated recorded-grating
input cubes and report whether rate-map scale, tuning rank, and map reliability
degrade comparably. Down-weight a loss result when it accompanies broad,
non-specific degradation on this control.

**Interpretation:** use the following table as the authoritative logic for this
primary experiment. Do not require units to share one mechanism or collapse a
structured dissociation into a single mean.

| Input outcome | Neural outcome | Interpretation |
|---|---|---|
| Global map-input power matched, phase destroyed, and distribution gate passed | SSI sharpening persists | Strong evidence that global map-input power is sufficient for that example |
| Global map-input power matched, phase destroyed, and distribution gate passed | SSI sharpening is reduced or lost without generic response degradation | Phase-dependent localization beyond global power matters; within-RF phase sensitivity remains unresolved |
| Random-phase input fails the range, structural-distribution, or generic-degradation control | SSI sharpening is reduced or lost | Non-diagnostic between phase dependence and out-of-distribution model behavior |
| Stabilized power, FEM phase retained, with symmetric phase support passed | SSI follows the power-reduced arm | Power reduction is sufficient to reproduce the change under preserved FEM phase |
| Factorial phase effect remains at both power levels | SSI follows phase rather than power | Evidence for a phase contribution, with the interaction reported separately |
| RF-local power also matched, phase destroyed | Residual SSI change remains | Conditional evidence for within-position phase sensitivity |
| Power mismatched | Any response result | Non-diagnostic because the temporal-power mechanism was not controlled |
| Phase relationships reconstructed | SSI sharpening persists | Non-diagnostic because the surrogate no longer isolates phase |
| Phase result follows contrast control | SSI changes with contrast in both branches | Contrast, not phase, is the parsimonious explanation |
| Different units show different outcomes | Structured dissociation by tuning, intact-effect role, or RF scale | Evidence for heterogeneous mechanisms; preserve and report it rather than collapsing it |

The power-reduced/phase-preserved arm determines whether lowering the FEM power
spectrum alone produces the same map/SSI change. A residual ambiguity after
the factorial is the trigger for an RF-local control, not a reason to assume
one in advance.

**Decision gate:** advance only after the input cubes and first activation maps
are visually interpretable, the exact raw-cube power contract passes, and
source-phase validity, pixel range, histogram, higher-order structural
statistics, early-representation distance where available, local-energy
redistribution, and the generic-degradation calibration do not plausibly
dominate the twin response. Marginal histogram/range agreement alone is not an
in-distribution certificate. This checkpoint uses development identities only.

**Expected runtime:** minutes for synthesis and a targeted GPU render; no
renderer-in-the-loop optimization is required.

### Stage 3: audit the architectural spatial contract and tuning compatibility

**Cartographer targeted checkpoint status (2026-08-14): complete; decision gate
not passed.** Six recorded-spatial-frequency-validated units were selected
algorithmically before Stage 3 responses to span learned-readout support,
preferred spatial frequency, and orientation selectivity. Each unit's strongest
measured fixed-retina grating defined a localized 32-lag probe. No reserved
natural-image or trace identity was opened.

The coordinate rule passed: translating a 151-by-151 input by two pixels
translated the 51-by-51 activation map by one bin with minimum map correlation
$r=0.999999$ and maximum normalized root-mean-square error $9.15\times10^{-4}$
for interior translations. Even 40-pixel edge translations retained minimum
$r=0.999997$ and maximum normalized error 0.00542.

Direct-core canvas-size transfer failed. Although the central 51-by-51 pixels
were identical, the 51-input direct-core scalar and central 151-input map value
differed by as much as 0.460 expected counts/frame (55.2 Hz after multiplying by
120; mean absolute difference 0.0514 counts/frame or 6.17 Hz; $r=0.966$ across
36 unit--probe pairs). Separately subtracting 51-input and 151-input direct-core
blank responses did not resolve the mismatch: blank differences were at most
$8.4\times10^{-5}$ counts/frame (0.0101 Hz), whereas modulation error still
reached 0.460 counts/frame. The exactly embedded and analytically extended
large-canvas probes agreed within $5.18\times10^{-5}$ counts/frame (0.00622 Hz),
localizing this particular difference to direct-core canvas context rather than
visible stimulus energy outside the crop. This was not the session-native
empirical-tuning pathway.

Candidate supports also disagreed. Median 90%-energy radii were 10.93 input
pixels for the learned-readout back-projection, 6.25 pixels for grating
input-gradient energy, and 48.71 pixels for the translated-probe response
envelope. The last is visibly probe- and response-dependent and is not a direct
receptive-field aperture. Therefore no aperture or native-tuning-to-large-map
calibration is frozen. The raw checkpoint is in
[`rr100_spatial_coordinate_contract_stage3_v1`](/home/declan/VisionCore/outputs/fig4_active_sensing/rr100_spatial_coordinate_contract_stage3_v1),
the baseline diagnostic is in
[`rr100_spatial_coordinate_contract_stage3_context_audit_v1`](/home/declan/VisionCore/outputs/fig4_active_sensing/rr100_spatial_coordinate_contract_stage3_context_audit_v1),
and the focused review is in
[`rr100_spatial_coordinate_contract_stage3_review_v1`](/home/declan/VisionCore/outputs/fig4_active_sensing/rr100_spatial_coordinate_contract_stage3_review_v1).

**Orientation/direction coordination contract (2026-08-14):** all subsequent
localized Stage 3 grating probes must be selected from the complete empirical
RR100 SF-by-absolute-TF-by-motion-direction tensor in
[`rr100_joint_sftf_direction_tuning_checkpoint_v1`](/home/declan/VisionCore/outputs/fig4_active_sensing/rr100_joint_sftf_direction_tuning_checkpoint_v1).
For every probe, save the exact spatial frequency, temporal-frequency
magnitude, motion direction in image coordinates, bar orientation in image
coordinates, drift sign, signed temporal frequency, empirical signed F0, and
empirical positive F0. Do not reconstruct these probes from the older four-bin
preferred-orientation field. Include a frequency-dependent angular-tuning role
and preserve changes in preferred direction across SF-by-TF slices as a real
dissociation rather than averaging them away. Parametric separable and
frequency-dependent-angle fits are diagnostics only; they cannot replace the
empirical tensor as routing weights or probe definitions.

This improved direction contract does not resolve the native-to-large-canvas
failure. Direct large-canvas calibration remains independently required, and
neither an aperture-to-tuning calibration nor any old four-bin orientation
summary may be frozen or propagated into Figure 4 Panels D/E while that gate is
open.

**Empirical directional-probe input checkpoint (2026-08-14): complete; neural
scoring not started.** Fourteen localized 32-lag probes were selected for five
predeclared unit roles: strong consistent direction, orientation without
direction, frequency-dependent direction, static--dynamic orientation
dissociation, and a weak-response control. Responsive units contribute their
empirical peak tensor cell, the opposite drift at the same SF and TF, and a
strong SF-by-TF slice with the largest supported change in preferred direction;
the weak control contributes its least-suppressed cell and opposite drift. The
frequency-dependent-direction example, RR100 unit 55, changes from 270-degree
motion at 1.414 cycles/degree and 32 Hz to 90-degree motion at 2.828
cycles/degree and 22.627 Hz. This difference is retained explicitly rather than
replaced by a fitted or averaged preferred angle.

The saved table records SF, absolute TF, motion direction, bar orientation,
drift sign, signed TF, signed F0, positive F0, tensor indices, and selection
criteria for every probe. The rendered histories and empirical tuning slices
are available in
[`rr100_spatial_coordinate_contract_stage3_directional_probe_design_v1`](/home/declan/VisionCore/outputs/fig4_active_sensing/rr100_spatial_coordinate_contract_stage3_directional_probe_design_v1).
This is an input-design checkpoint only: it does not validate transfer to the
large canvas, select an aperture, or authorize changes to Panels D/E.

**Empirical directional-probe transfer checkpoint (2026-08-14): complete;
decision gate not passed.** The 14 approved probes were scored through the
51-input direct-core scalar pathway, an identical probe embedded in a 151-by-151
canvas, and the same analytic localized grating rendered directly on the large
canvas. All 70 probe-by-selected-unit activation maps were saved, while the
primary visual comparison uses each probe owner's map. Native and large-canvas
blank responses were scored and subtracted separately.

The two large-canvas constructions agree closely: their central modulations
differ by at most $6.35\times10^{-5}$ expected counts/frame (0.00762 Hz) and
their owner-map correlations are at least 0.999974. The 51-input-to-151-input
direct-core transfer remains materially imperfect, however: across probes,
modulation correlation is 0.883, mean absolute error is 0.0658 counts/frame
(7.90 Hz), and maximum absolute error is 0.241 counts/frame (28.9 Hz). The
mismatch is structured across units and probes rather than a single
transferable gain. RR100 unit 67 preserves ordering but its peak modulation
increases from 0.106 to 0.347 counts/frame; RR100 unit 75 is compressed across
all three probes; and RR100 unit 55 reverses modulation sign for two
single-phase localized probes. The weak-response control is comparatively
stable. These are descriptive direct-core canvas diagnostics, not an
inferential population summary or a test of the phase-averaged empirical F0
tuning tensor.

The response traces and raw activation maps are in
[`rr100_spatial_coordinate_contract_stage3_directional_probe_transfer_v1`](/home/declan/VisionCore/outputs/fig4_active_sensing/rr100_spatial_coordinate_contract_stage3_directional_probe_transfer_v1).
Because single-phase response signs can change with direct-core canvas context,
no native-to-large tuning calibration, aperture, or Panel D/E routing weight is
authorized by this checkpoint.

**Motion-relative translation checkpoint (2026-08-14): complete; aperture gate
still open.** Every approved probe was placed at nine large-canvas locations:
the centre and positive/negative near and far offsets along its empirical
motion and bar axes. Near offsets requested 8 input pixels and far offsets
requested 40; diagonal components were rounded to even pixels so that the
validated two-input-pixel-to-one-map-bin stride remained exact. This produced
126 histories and 630 saved probe-position-by-selected-unit activation maps.

Within the 151-by-151 pathway, local responses are effectively invariant to
these translations. The largest position-induced modulation change was
$1.61\times10^{-6}$ expected counts/frame (0.000193 Hz) for near offsets and
$5.33\times10^{-5}$ counts/frame (0.00639 Hz) for far offsets. The minimum
shifted-map correlation was 0.999959 near and 0.999951 far;
the maximum far normalized root-mean-square map error was 0.00760. There was no
material difference between translations along the motion and bar axes. Thus
the structured native-to-large mismatch is not caused by probe position within
the large canvas or by a direction-relative translation interaction. It is
localized to the change between native and large-canvas pathways or boundary
contexts, while the large-canvas pathway itself remains strongly translation
equivariant.

The response-position traces and raw maps are in
[`rr100_spatial_coordinate_contract_stage3_directional_translation_v1`](/home/declan/VisionCore/outputs/fig4_active_sensing/rr100_spatial_coordinate_contract_stage3_directional_translation_v1).
The original raw difference sheet included zero-filled nonoverlap borders that
were never part of the numerical comparison; the corrected overlap-masked
review is in
[`rr100_spatial_coordinate_contract_stage3_directional_translation_review_v1`](/home/declan/VisionCore/outputs/fig4_active_sensing/rr100_spatial_coordinate_contract_stage3_directional_translation_review_v1).
This result validates spatial transport of a directly measured large-canvas
map but does not make the native scalar tuning amplitudes valid routing weights
and does not identify a unique aperture.

**Pathway-contract correctness audit (2026-08-14): complete; dense tuning
transfer not yet authorized.** The native empirical tensor and sparse Stage 3
probes were not like-for-like measurements. Native production uses 33 history
frames (current through $t-32$), the selected session's learned blur/scale
adapter, a long phase-consistent movie, temporal averaging, a carrier-phase
schedule, and matched blank subtraction. Sparse Stage 3 uses one 32-frame
localized history, bypasses all session adapters with `core_forward`, and
scores one instantaneous carrier phase. In addition, the sparse runners stored
post-activation expected counts per 1/120-second frame but labelled rate-like
columns and figures as Hz. All affected output directories now contain a
warning file; multiply rate-like quantities by 120 for Hz. Correlations,
translation stride, and normalized map errors are unchanged.

The model-defined feedforward readout spans 50 active input pixels within the
51-pixel native window; floor pooling leaves one edge pixel outside that
theoretical span. The 3-by-3 ConvGRU expands the maximum theoretical history
support to 98 input pixels for 32-frame histories and 102 pixels for 33-frame
histories. These are architectural upper support spans, not fitted effective
weights. The full audit is in
[`rr100_stage3_pathway_contract_audit_v1`](/home/declan/VisionCore/outputs/fig4_active_sensing/rr100_stage3_pathway_contract_audit_v1).
The sparse experiments remain valid direct-core canvas and translation tests
after unit correction, but they neither establish nor refute transfer of the
native empirical F0 tensor.

**Matched tuning-transfer input checkpoint (2026-08-14): complete; neural
scoring not started.** The corrected targeted design retains two auditable
SF-by-TF slices for each of the five predeclared roles and all eight empirical
motion directions within every slice. The responsive units use their primary
tensor slice plus the previously selected frequency-dependent angular
dissociation slice. The weak-response control uses its least-suppressed primary
slice plus a frequency-separated control slice selected without neural map
responses. This gives 80 directional routing cells, 88 phase-specific unit
conditions under the exact native phase schedule, and 21,120 valid response
histories before batching.

For each condition, the rerender will generate the original session-native
51-by-51 movie, construct 33-frame current-through-$t-32$ histories, apply the
learned session adapter once, and then branch the identical adapted history to
the 51-input scalar and central 151-input activation-map pathways. Both outputs
will be converted from expected counts/frame to Hz, scored over the exact native
valid-response duration, blank-subtracted within pathway, and averaged over the
native carrier-phase schedule. The scalar branch must first reproduce the
cached native condition means to numerical tolerance. The design is in
[`rr100_stage3_matched_tuning_transfer_design_v1`](/home/declan/VisionCore/outputs/fig4_active_sensing/rr100_stage3_matched_tuning_transfer_design_v1).
This checkpoint tests transfer of complete direction profiles under a matched
estimand; it does not fit an aperture or open any natural-image test identity.

**Revised objective after architectural review (2026-08-14):** audit the
compatibility of the model-defined spatial geometry, the native empirical
SF-by-TF-by-direction tensor, and the large-canvas activation-map pathway. Do
not reverse engineer spatial quantities that are fixed by construction.

**Known by design:** the trained unit has a learned Gaussian spatial readout on
the final core feature map; `PopulationReadout.forward` applies that same
readout convolutionally with valid padding to generate the activation map. The
model configuration fixes the valid 7-by-7, 9-by-9, and 5-by-5 feedforward
convolutions, the 2-by-2 stride-2 pool, and the 3-by-3 spatial ConvGRU with
same-size padding. The two-input-pixel map stride follows from this architecture
and has now been verified numerically. These quantities are definitions or
implementation checks, not empirical receptive-field parameters to fit.

**Important distinction:** back-projecting the learned readout through
all-ones feedforward kernels gives a prespecified input-support envelope. It is
not the learned signed weighting of the convolutional channels, and it
deliberately omits spatial propagation through recurrent ConvGRU state. The
translated-probe response envelope is likewise a visualization of
convolutional replication, not an aperture estimate. Stimulus-dependent
gradients or perturbations may diagnose effective weighting, but they must not
silently replace the model-defined geometry.

**Method:**

1. Freeze an auditable architecture table from the trained configuration and
   runtime tensors: core kernels, padding, pooling, stride, recurrent spatial
   kernel, learned readout size, and valid activation-map support.
2. Retain the feedforward back-projection of the learned readout as the primary
   prespecified geometric support envelope. Label recurrently expanded
   theoretical support separately; do not fit a narrower Gaussian aperture from
   translated responses.
3. Treat the completed translation experiments as implementation verification:
   the large-canvas pathway transports maps accurately. Do not use them to
   estimate an empirical receptive-field radius.
4. Audit tuning transfer directly. For the predeclared unit roles, measure a
   sufficiently dense set of SF-by-absolute-TF-by-direction cells through the
   central large-canvas map pathway and compare it with the native tensor using
   signed responses, positive responses, preferred direction, rank order, and
   held-out affine calibration. Preserve frequency-dependent angular changes.
5. If tensor shape is preserved, freeze a training-only native-to-large
   amplitude calibration. If shape or response sign changes in the corrected
   phase-averaged comparison, do not transfer native tensor weights; measure
   the required routing tensor directly in the large-canvas pathway. The sparse
   single-phase unit 55 sign reversal motivates this check but is not itself a
   tensor-transfer result.
6. Use gradients or localized perturbations only if the subsequent
   architecture-defined power map fails in a way that specifically implicates
   within-support weighting. Such analyses remain diagnostics of nonlinearity
   or nonseparability, not replacement definitions of the spatial readout.

**Evaluation:** native-to-large tensor-shape agreement, sign preservation,
preferred-direction preservation within SF-by-TF slices, held-out affine
calibration error, and exact activation-map transport. Aperture radii from
response sweeps are not primary endpoints.

**Decision gate:** advance when the model-defined spatial support and a
large-canvas-compatible tuning tensor are frozen separately. A scalar gain is
allowed only if tensor shape and sign are preserved. Otherwise use directly
measured large-canvas tuning and retain the failure of native transfer as a
documented dissociation. Do not fit a free empirical 2-D receptive field unless
the architecture-defined local-power analysis later demonstrates a specific
need.

**Expected runtime:** minutes for the architecture audit and approximately
1–3 hours for a targeted dense tuning-transfer checkpoint; a complete tensor
rerender is conditional on that result.

### Stage 4: targeted local power-derived activation maps

**Objective:** establish what the proposed local spectral mechanism predicts in
concrete natural-image examples.

**Motivation:** a new map proxy must be visually and quantitatively interpretable
before population statistics are meaningful. This stage diagnoses whether power
predicts the spatial activation pattern rather than merely a scalar outcome.
The completed three-condition checkpoint compared only scalar temporal response
magnitude and therefore does not satisfy this stage.

**Construction:** at each activation-map position, apply a translated copy of the
model-defined spatial support contract, calculate local spatial-frequency,
Fourier-orientation, and signed temporal-frequency power, and weight it with a
large-canvas-compatible complete empirical
SF-by-absolute-TF-by-motion-direction response tensor. Use the native tensor
only if Stage 3 demonstrates shape and sign transfer; otherwise use the tensor
measured directly through the activation-map pathway. Preserve bar
orientation and drift sign as distinct saved coordinates. Frequency-dependent
angular tuning must remain available to the router; do not replace it with one
preferred angle per unit, the old four-bin orientation field, a marginal
orientation curve, or a parametric fit. Match the digital twin's spatial stride,
valid support, and edge convention. Construct moving and stabilized local
predictors separately; form a signed difference map only after applying the
frozen training-only activation calibration to each condition. Never assign a
sign to an unsigned power-difference map by interpretation alone. This
construction remains blocked until Stage 3 freezes the model-defined spatial
support contract separately from the large-canvas-compatible empirical tuning
weights.

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
predicted map, twin map, and direct FEM-minus-stabilized difference maps. Add
the accepted Stage-2A map-support factorial and its twin maps. The global
map-input power representation remains fixed in the phase-destroyed arm, while
RF-local power is explicitly allowed to redistribute and must be plotted rather
than assumed fixed. Any resulting map change quantifies information absent from
the global power representation; it is not yet a lower bound on failure of the
later RF-local predictor. Predicted quantities and twin outcomes must be
visually distinguished.

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

The accepted map-support factorial must pass through the same frozen
calibration. Exact raw-cube Fourier-magnitude matching establishes the failure
of a global map-input power account when a phase-only change alters the twin
map. It does not establish failure of an RF-local power account unless the
relevant translated local power maps are also matched. Keep those conclusions
separate.

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
7. map-support amplitude-by-phase factorial input, with shared random phase in
   paired power contrasts;
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

1. the model-defined spatial support and coordinate contract are implemented
   exactly, and the tuning weights are validated in the large-canvas pathway;
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
7. map-support phase controls do not reveal response or map changes large
   enough to account for the proposed effect while leaving global map-input
   power unchanged; any stronger RF-local sufficiency claim additionally
   requires matched translated local power.

Failure at any stage narrows the conclusion rather than invalidating all prior
results. In particular, direct scalar prediction may remain valid even if the
spatial mechanism is unresolved.

## 9A. Parallel workstream ownership

**Cartographer** is the working name of the agent responsible for the spatial
mechanism branch. The name reflects its immediate task: establish how the
digital twin maps input coordinates and translated receptive-field support onto
activation-map coordinates before constructing any local power-derived map.

Cartographer owns:

- **Stage 3:** architectural spatial-contract audit, native-versus-large-canvas
  tuning compatibility, controlled probe translations, stride and edge
  conventions, and diagnostic comparison of model-defined support with
  stimulus-dependent sensitivity only if needed;
- **Stage 4, conditional on the Stage 3 human gate:** development-only local
  power-derived activation maps, auditable example selection, raw twin and
  predicted maps, and direct condition-difference maps;
- **Stage 5 map-mediated branch, conditional on the Stage 4 human gate:** frozen
  power-to-rate calibration and SSI calculated from predicted maps, retaining
  rate, expected spikes, raw information, and instantaneous versus mean-map SSI;
- **Stage 6 spatial implementation and audit preparation only:** final-test map
  code, provenance, and frozen configuration may be prepared, but the untouched
  identities may not be opened until both workstreams and the human reviewer
  agree that all choices are frozen.

The **phase-control workstream** owns the separately versioned,
distribution-constrained Stage 2A control, including symmetric phase support,
range, histogram, higher-order structure, local-energy redistribution,
phase-retention, early-representation distance where feasible, and
recorded-grating generic-degradation calibration. It must not overwrite the
completed Stage 2A v1 runner or artifacts.

The workstreams may proceed in parallel because Stage 3 calibrates the spatial
contract independently of whether the revised phase surrogate passes. Results
are integrated only at the Stage 4/5 interpretation checkpoint. Cartographer
will not edit or run the other workstream's versioned phase-control scripts or
outputs. The phase-control workstream should avoid Cartographer's versioned
Stage 3--5 files. Either workstream must inspect recent changes before editing
this shared plan; workstream-specific findings should otherwise remain in their
own README and manifest until integration.

## 10. Planned execution order and estimated runtime

| Order | Stage | Workstream owner | Expected runtime |
|---:|---|---|---:|
| 0 | Quarantine invalid caches and export clean grating-only tuning | shared; completed | 1–2 hours |
| 1 | Incrementally build and audit replacement-cohort spectral cache | shared; provisional stage completed | 10–20 minutes per complete round, plus 20–40 minutes final audit |
| 2 | Corrected whole-image scalar prediction | shared; development stage completed | 15–30 minutes |
| 2A | Distribution-constrained map-support phase control | phase-control workstream | minutes for a targeted checkpoint, excluding surrogate optimization |
| 3 | Audit architectural spatial contract and native-to-map tuning compatibility | **Cartographer** | 1–3 hours |
| 4 | Targeted local power-derived activation maps on development identities | **Cartographer**, after Stage 3 gate | 30–90 minutes |
| 5 | Direct and map-mediated SSI analysis | **Cartographer** owns map-mediated branch; direct branch shared | 30–60 minutes after map generation |
| 6 | Untouched-test population map analysis | joint final gate; Cartographer prepares spatial implementation | 4–12 hours |

These estimates assume the current workstation, existing input and response
caches, and GPU batching where applicable. Each stage ends with an explicit
review checkpoint before the next stage begins.

The completed three-condition natural-image response checkpoint is a qualitative
bridge into Stages 2 and 4. It does not remove or shorten quarantine, the
replacement 100-round corrected-cache endpoint, held-out scalar expansion, receptive-field
calibration, spatial map comparison, phase controls, or SSI analysis.

## 11. Immediate next action

Stages 0, provisional Stage 1, development Stage 2, and the Stage 2A raw-map
checkpoint are complete. The frozen identity split leaves 20 images and 116
traces unopened for the later final-test map analysis. Stage 2A's exact power
contract passed, and its fixed-FEM-phase amplitude comparison supports an
important role for the FEM-driven amplitude/power change in the inspected
example. Its random-phase control failed the input-distribution gate, however,
so no phase-sufficiency conclusion is authorized.

The two scientifically distinct routes are now authorized as parallel,
non-overlapping workstreams:

1. the phase-control workstream develops a separately labelled
   distribution-constrained phase control, reruns all spectral, phase-support,
   range, histogram, higher-order-structure, local-energy, generic-degradation,
   and response audits, and stops at a new Stage 2A human checkpoint; and
2. **Cartographer** proceeds through Stage 3 without claiming that the current
   factorial established phase sufficiency or necessity.

Cartographer's coordinate and translation checkpoints are complete. The
two-input-pixel to one-map-bin rule and large-canvas spatial transport passed.
The pathway audit subsequently showed that the sparse canvas probes used 32
frames, bypassed learned session adapters, scored one carrier phase, and stored
expected counts/frame under incorrect Hz labels. Those outputs are quarantined
for physical-unit and tensor-transfer claims but remain valid direct-core map
diagnostics after multiplying rate-like quantities by 120. Selection of a
response-fitted aperture was neither achieved nor required.

Cartographer must therefore not begin Stage 4. The matched tuning-transfer input
design is now frozen for human inspection: five predeclared unit roles, two
auditable SF-by-TF slices per unit, all eight directions, exact native phase and
duration schedules, 33-frame histories, learned session adapters applied before
branching, matched blanks, and explicit Hz conversion. The smallest next action
after approval is to score those 88 phase-specific unit conditions and test
direction-profile shape, sign, and held-out affine transfer at the central map
position. The learned Gaussian readout, feedforward architectural support
envelope, and recurrently expanded theoretical support are model-defined
quantities and must remain explicit. Operating-point gradients are conditional
diagnostics; the translated-probe response envelope is not an aperture. The old
four-bin preferred-orientation field must not enter Panels D/E. Neither
workstream may open the reserved final-test identities.
Confirmatory claims still wait for the replacement balanced 100-image by
1,000-trace cache and the joint Stage 6 freeze.
