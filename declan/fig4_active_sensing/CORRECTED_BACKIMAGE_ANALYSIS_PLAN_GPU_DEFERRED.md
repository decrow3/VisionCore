# Corrected BackImage/RR100 analysis plan with the large GPU rerun deferred

Date: 2026-08-12

Status: working analysis roadmap

Scope: BackImage retinal-motion, spatial-SSI, SF×TF power-routing, population-response, gain, orientation/geometry, and behavior-confound analyses

## Objective

Recover as much reliable information as possible from the existing data and caches while postponing the corrected 100-image × 1,000-trace neural-model run. All work before that run must serve at least one of four purposes:

1. preserve unaffected fixed-retina tuning results;
2. determine which structural findings in the legacy 100 × 1,000 response cache are likely to transfer;
3. construct and validate the corrected production input contract;
4. predefine the analyses, examples, and statistical tests that will be applied after the deferred GPU run.

The plan deliberately separates **input facts**, **legacy-cache structural observations**, **provisional corrected results**, and **calibrated production results**. Those categories must not be blended in figures or prose.

## Current evidence ledger

### Valid and reusable now

- RR100 unit identity and population mapping.
- Controlled fixed-retina SF tuning.
- Controlled fixed-retina joint SF×TF F0 tuning.
- Parametric SF and TF models and their grating fit-quality estimates.
- Recorded-grating validation and the 61-unit recorded-validated cohort:
  - 31 units in the current low-SF half;
  - 30 units in the current high-SF half.
- Static orientation probes.
- The corrected BackImage stimulus contract established by the eye-trace/Nyquist audit:
  - raw BackImage rate: 240 Hz;
  - model visual rate: 120 Hz;
  - global-even visual samples;
  - `dpi_pix` crop trajectory;
  - session `roi_src` offset;
  - retinal-translation sign;
  - one-sided FFT with the 60-Hz Nyquist bin retained.

### Completed input validation

- The corrected renderer was compared with exact saved BackImage inputs on 16 original pairs:
  - median pixel correlation: 0.924;
  - minimum pixel correlation: 0.815.
- Corrected 32–60 Hz retinal-power estimates remain substantial:
  - corrected crossed reconstructions: median 37.8%;
  - exact saved inputs: median 41.3%.
- The 1,000 identities in the legacy production trace bank have been reconstructed without a model rerun:
  - 973 have a valid, finite, same-trial, DPI-valid 32-frame recorded lead-in;
  - 27 fail the DPI-validity requirement;
  - all are in bounds and remain within their recorded trial;
  - old and corrected path and frequency descriptors can differ substantially.

### Provisional, not production-calibrated

- The corrected 2-image × 4-trace response-map smoke test.
- The corrected 16-image × 32-trace spatial-SSI validation bank.
- The low-SF path-length association observed in the 16 × 32 bank.

These analyses correct sampling, crop geometry, retinal sign, and the native helper's `T+1` response alignment. They do **not** use the 32 recorded frames immediately preceding the scored segment. The helper synthesizes lag history from the analyzed trace, so these results remain provisional.

### Superseded as calibrated evidence

- The legacy 100 × 1,000 response magnitudes and SSI estimates.
- Old power and SF×TF overlap maps generated from the defective reconstruction.
- Old global-versus-tuning-aware predictor comparisons.
- Old population common-ordering and additive/multiplicative gain conclusions.
- Old orientation-aware overlap magnitudes.
- Old microsaccade/drift labels, speeds, durations, phases, PSD axes, and other time-derived quantities computed by assigning 1/120 s to raw 240-Hz samples.

The superseded caches should be preserved for lineage and structural piloting, not deleted.

## Canonical contracts to freeze before further work

### Behavior contract

- Use raw `eyepos` for measured physical eye behavior.
- Treat raw BackImage `eyepos` as 240 Hz.
- Report explicitly whether behavior was measured at 240 Hz or transformed to a 120-Hz representation.
- Do not use the shifter correction as though it were a physical eye measurement.

### Visual-input contract

- Prefer exact saved `backimage.dset["stim"]` frames for original recorded image–trajectory pairs.
- For crossed or counterfactual movies, use the `dpi_pix` crop trajectory, global-even source indices, the session RF offset, and retinal-motion sign.
- Use 32 recorded 120-Hz lead-in frames followed by a 40-frame scored segment.
- Use the explicit-history renderer. Do not construct the temporal lags by repeating the scored trace.
- Normalize a native `T+1` response to the intended scored segment using the documented response alignment; assert exactly 40 retained responses.

### Stabilized baseline contract

- The primary corrected production baseline will be zero relative retinal translation through both the 32-frame history and 40-frame scored segment, on the same corrected target-image patch.
- If trial-mean stabilization is retained as a secondary historical bridge, label it separately. Do not equate it with the zero-translation baseline.

### Unit-tuning contract

- Use F0 SF×TF tuning measured with fixed gaze.
- Carry fit quality continuously and retain the recorded-grating validation gate.
- Use continuous preferred SF as the primary population coordinate where possible.
- Low/high SF halves may be used for visualization, but must not replace continuous analyses.
- Treat the present parametric TF support through 32 Hz as the measured core.
- Treat the 45.25-Hz probe as edge-assisted evidence, not a dense complete fit.
- Do not assign unit-specific weights above measured support until the TF grating probe has been extended.

## Phase 1 — finish the no-model trace identity audit

### Question

Does the ordering of the 1,000 trace identities survive correct sampling and visual calibration well enough for the legacy response cache to be useful as a structural pilot?

### Work

1. Use the completed concrete-example figure as the map-first input checkpoint.
2. Summarize old-versus-corrected agreement for:
   - path length;
   - RMS and maximum radius;
   - median step and speed;
   - covariance anisotropy and orientation;
   - temporal-power centroid;
   - power above 15 Hz and above 32 Hz.
3. Report Pearson and Spearman agreement, robust regression, rank changes, and quantile-confusion matrices.
4. Repeat summaries for:
   - all 1,000 identities;
   - the 973 explicit-history-valid identities;
   - old drift/microsaccade strata, labeled as legacy strata only.
5. Identify algorithmic roles:
   - rank preserved;
   - largest rank increase;
   - largest rank decrease;
   - largest high-TF correction;
   - largest `dpi_pix` calibration correction;
   - invalid-history control.
6. Save a trace crosswalk containing legacy descriptors, corrected descriptors, validity flags, and inclusion recommendations.

### Gate

- If corrected path/radius ranks are high, legacy path-ordered neural patterns may be examined as structural hypotheses.
- If ranks are only moderate, use the cache for convergence, low-rank structure, and example selection—but not corrected path-effect claims.
- If ranks are poor or highly stratum-dependent, do not reinterpret legacy neural responses using corrected trace descriptors.

### Outputs

- Population agreement figure.
- Quantile-transition heatmaps.
- Corrected trace crosswalk CSV.
- Invalid-history table.
- Updated manifest with a one-sentence reuse verdict for each descriptor.

## Phase 2 — audit and reconstruct the 100 image identities without a model rerun

### Question

Do the legacy 100 image identities preserve the local image content and ordering relevant to the model's corrected RF-centered visual input?

### Work

1. Reconstruct each legacy image identity using:
   - the old gaze-centered patch contract;
   - the corrected RF-centered patch contract;
   - exact saved stimulus frames for its original recorded condition when available.
2. Plot multiple examples spanning:
   - strongest old/corrected patch agreement;
   - weakest agreement;
   - stable contour axis;
   - changed contour axis;
   - stable versus changed spatial-frequency content.
3. Recompute corrected image descriptors:
   - RMS contrast;
   - gradient and oriented-gradient energy;
   - contour coherence and axis;
   - radial spatial-frequency power;
   - orientation-resolved spatial-frequency power;
   - distance from image/RF boundary and crop-validity diagnostics.
4. Quantify old-versus-corrected descriptor ranks and category transitions.
5. Produce a 100-image crosswalk and image-validity gate.

### Gate

Only image descriptors with demonstrated old/corrected agreement may be used to stratify the legacy response cache. Orientation-conditioned and local-contour claims require stronger agreement than contrast-only or broad-SF claims.

## Phase 3 — repair the BackImage window/timebase table

### Question

Which behavioral and image-window labels remain valid after treating the raw dataset at 240 Hz?

### Work

1. Modify the extractor so sampling rate is dataset-specific rather than globally fixed to 120 Hz.
2. Regenerate BackImage windows and event tables at the raw 240-Hz timebase.
3. Separately derive the model-aligned, global-even 120-Hz visual indices.
4. Recompute:
   - durations and fixation phase;
   - speeds and accelerations;
   - microsaccade/event labels;
   - PSD/frequency summaries;
   - diffusion and rate quantities;
   - all temporal selection gates.
5. Preserve time-independent spatial quantities and quantify whether they changed because window membership changed.
6. Create a row-level old/new window crosswalk using session, trial, and overlapping source intervals.

### Gate

No behavior-confound, microsaccade, phase, or speed analysis advances until its labels have been regenerated and its cohort change has been reported.

## Phase 4 — use the legacy 100 × 1,000 response cache as a structural pilot

This phase does not claim calibrated FEM effects. Every artifact must say:

> Neural responses were computed under the legacy reconstructed-motion contract; the analysis is used only to assess structural robustness and design the corrected production test.

### 4A. Cache integrity and estimand audit

- Verify the complete image × trace × unit grain and matrix/table alignment.
- Verify mean rate, expected spikes, information numerator, SSI, and population aggregation identities.
- Replace historical unit labels with current fixed-retina parametric and recorded-validated tuning metadata.
- Separate all-100, valid-parametric, and recorded-validated cohorts.
- Preserve mean rate and expected spikes beside bits/spike.

### 4B. Stability and convergence

- Estimate how effects stabilize as the number of images increases from small panels toward 100.
- Estimate how effects stabilize as traces increase toward 1,000.
- Use repeated image- and trace-disjoint subsampling.
- Determine whether the intended corrected production bank needs all 100 × 1,000 conditions or whether a smaller predeclared design would retain precision.

### 4C. Influence and robustness

- Compute image-, trace-, and unit-level influence.
- Repeat after removing:
  - the longest paths;
  - largest steps/speeds;
  - invalid-history identities;
  - legacy event-positive traces;
  - low-response or poor-tuning units.
- Report leave-one-image/trace and grouped tail sensitivity.

### 4D. Population structure

- Assess low-rank structure of mean-rate and SSI effects.
- Compare signed changes, magnitudes, information numerator, and expected spikes.
- Use split-half reliability across images, traces, and units.
- Do not call PC1 a biological latent factor or gain mechanism.

### 4E. Legacy-versus-corrected descriptor sensitivity

- Fit identical descriptive models using legacy descriptors and corrected identity descriptors.
- Treat this as a sensitivity analysis, not as a corrected neural-response model.
- Record whether conclusions depend on rank preservation or change under the corrected labels.

### 4F. Auditable selection for later corrected reruns

Select and save:

- stable positive examples;
- low/high-SF dissociations;
- high predicted retinal change with weak cached neural change;
- weak predicted change with large cached neural change;
- influential-image and influential-trace controls;
- history-invalid controls;
- common-component followers and dissociating units.

The selection table should drive the small corrected bridge run and later production map sheets.

### Forbidden interpretations in Phase 4

Do not use the legacy response cache to establish:

- absolute real-FEM SSI or rate effects;
- true speed, microsaccade, phase, or temporal-frequency effects;
- unit-specific spectral routing;
- additive or multiplicative gain under corrected retinal motion;
- exact image–trajectory pairing effects;
- corrected orientation alignment.

## Phase 5 — build corrected retinal-power caches without running the neural model

### 5A. Exact original-pair input audit

- Use exact saved model stimuli for each available original image–trajectory condition.
- Include the recorded 32-frame history and 40 scored frames.
- Compute static/DC and dynamic SF×TF power with the 60-Hz Nyquist bin retained.
- Keep radial and orientation-resolved power separately.

### 5B. Crossed/counterfactual power bank

- Use the explicit-history corrected renderer for crossed image × trace conditions.
- Start with the 973 matched valid trace identities and the image-valid cohort.
- Render and reduce movies on the fly; save spectral sufficient statistics rather than every full movie unless a selected example requires frames.
- Save:
  - total positive-TF power;
  - SF×TF radial power;
  - orientation-resolved SF×TF power for secondary analyses;
  - supported power through 32 Hz;
  - edge-assisted 32–45.25 Hz power;
  - currently unmeasured 45.25–60 Hz power;
  - explicit history and crop-validity flags.

### 5C. Input-only population questions

- Is between-condition spectral variation mostly amplitude-like or shape-like?
- How low-rank are normalized versus unnormalized power maps?
- How much variation belongs to images, traces, and interactions?
- How stable are power rankings under trace and image subsampling?
- Which conditions should be retained for a balanced corrected neural run?

These are retinal-input results and may be reported without a neural rerun, provided neural sensitivity or response is not inferred from them.

## Phase 6 — validate the explicit-history boundary with a small bridge run

This is optional while all GPU use is deferred, but it is mandatory before launching the large production job.

### Design

- Approximately 20 image identities × 50 trace identities = 1,000 movies.
- Use identities present in the legacy response cache.
- Balance image structure, corrected path length, corrected temporal power, legacy/corrected rank change, and history validity.
- Score all 100 RR100 units while promoting the 61 recorded-validated units for inference.

### Conditions

1. legacy renderer/trace contract;
2. corrected crop and sampling with repeated synthetic history;
3. corrected crop and sampling with 32 recorded lead-in frames;
4. stabilized explicit-history baseline.

### Questions

- How much does genuine prehistory change mean rate, SSI, and maps?
- Are legacy versus corrected neural condition rankings preserved?
- Does the provisional low-SF path association survive explicit history?
- Which units, images, and traces reverse or dissociate?
- Is a 32-frame lead-in sufficient, as checked against longer recorded lead-ins where available?

### Gate

The large GPU run is launched only if:

- retained responses are exactly the predeclared 40-frame scored segment;
- the 32-frame history agrees with a longer-history control within a predeclared tolerance;
- static and moving histories are matched except for relative retinal translation;
- selected raw maps and timecourses are interpretable;
- the production cohort and outputs are frozen in a manifest.

Approximate GPU time: tens of minutes, not the multi-hour production run. This estimate should be benchmarked rather than assumed.

## Phase 7 — prepare the deferred production bank

Preparation can be completed without scoring the model.

### Cohorts

Maintain two related cohorts:

1. **Matched legacy bridge cohort:** the 973 old trace identities with valid recorded history, used for old/new comparisons.
2. **Corrected production cohort:** 1,000 traces selected from the corrected eligible pool. Replace the 27 invalid histories rather than silently retaining them.

Freeze 100 corrected image identities. Replace an image only if the corrected crop or validity audit fails; document every replacement.

### Event composition

The old bank forced 200 legacy microsaccade-positive traces. That stratification cannot be inherited automatically. After correcting the event detector and timebase, explicitly choose one of:

- preserve a predeclared number of corrected event-positive traces for coverage;
- sample the natural corrected event prevalence;
- analyze drift-like and event-containing banks separately.

Record the choice before neural scoring.

### Production tensor contract

For each trace:

- 32 corrected global-even history frames;
- 40 corrected global-even scored frames;
- 72 total 120-Hz positions;
- target-relative mean-centering defined with respect to the scored segment;
- explicit-history temporal embedding;
- exactly 40 retained neural responses.

### Saved outputs

- Mean rate per movie/unit.
- Expected spikes per movie/unit.
- Instantaneous information numerator summed over the scored segment.
- Movie SSI in bits/spike.
- Information rate in bits/s or equivalent unnormalized information quantity.
- Population versions of each quantity.
- Per-frame values for a predeclared selected subset.
- Full spatial maps only for selected examples and validation conditions.
- Image, trace, unit, movie, and provenance tables.
- Input spectral sufficient statistics joined by immutable keys.

### Compute estimate

The legacy 100,000-movie run required approximately 20 GPU-hours total, split into two roughly 10-hour, 50,000-movie shards. The corrected run should be budgeted in the same order until the bridge benchmark provides a better estimate. Two GPUs could reduce wall time to roughly one long overnight run, subject to availability and I/O.

## Phase 8 — analyses after the deferred production run

Follow map-first order. Do not start with regression tables.

### 8A. Input and response maps

- Show corrected image patch, full recorded lead-in, scored path, speed, and SF×TF power.
- Show moving, stabilized, and difference maps across time for multiple units.
- Use explicit positive, dissociation, and negative/control roles.
- Save the selection criteria before drilling down.

### 8B. Spatial SSI and rate

- Keep instantaneous SSI, spike-weighted movie SSI, information numerator, mean rate, expected spikes, and information rate together.
- Prefer paired moving-minus-stabilized differences.
- Analyze continuous preferred SF first; show low/high halves second.
- Use image-, trace-, and unit-aware uncertainty.
- Repeat tail and influence checks.

### 8C. Retinal power and SF×TF overlap

- Rebuild Kuang-style static sensitivity, FEM-created power, and overlap panels.
- Keep measured-core, edge-assisted, and unsupported TF bands visibly distinct.
- Compare total supported power with unit-specific tuning-weighted power using identical support and preprocessing.
- Do not interpret power above fitted TF support as unit-specific routing.

### 8D. Predictor comparison

Compare, out of sample:

1. static image response only;
2. unweighted dynamic power;
3. SF-only weighted power;
4. TF-only weighted power;
5. separable SF×TF weighted power;
6. empirical nonseparable grating surface where support permits;
7. orientation-aware power as a secondary extension.

Use image- and trace-disjoint folds. Report per-unit performance, population distributions, fit-quality dependence, and failures—not only the median.

### 8E. Shared population structure and apparent gain

- Recompute signed and magnitude response-effect matrices.
- Test rank and split-half stability across independent images and traces.
- Compare shared neural image/condition scores with retinal power only after both are independently established.
- Compare additive, multiplicative, and combined models using matched held-out folds.
- Preserve response-floor and heteroscedasticity diagnostics.
- Use “gain” only if the multiplicative model improves prediction and reproduces map/timecourse behavior—not merely because a common component exists.

### 8F. Orientation and geometry

- Treat radial SF×TF results as primary.
- Add orientation-resolved power only after radial results are stable.
- State exactly what rotates in every control: image, trajectory, or unit tuning.
- Do not promote optimal image–trajectory pairing or optimal eye-movement direction without new evidence.
- Revisit contour-alignment and behavior/model-bridge analyses only with corrected image axes, corrected traces, and corrected response caches.

### 8G. Robustness and generality

- Compare the matched 973-trace bridge cohort with the new 1,000-trace production cohort.
- Repeat after excluding event-containing traces, invalid/edge crops, extreme paths, and weak grating fits.
- Assess convergence over images and traces.
- Separate statements about the frozen RR100 model from claims about experimental neurons or biological V1.

## Phase 9 — extend the fixed-retina TF probe

The corrected input audit shows enough power between 32 and 60 Hz to justify better unit-level TF coverage. This is a separate GPU task and may remain deferred alongside the large movie run.

### Design

- Extend dense F0 TF samples beyond 32 Hz through approximately 56 Hz.
- Retain 60 Hz as a separately labeled Nyquist-edge control because a sampled sinusoid is phase-degenerate there.
- Preserve multiple carrier phases and direction folding.
- Check model response relative to blank and reliability across phase.
- Update parametric and empirical SF×TF surfaces without overwriting the current through-32-Hz fits.

### Use

Repeat complete-support spectral-routing claims only after the extended probe passes its own map-first and response-quality checks.

## Phase 10 — refresh the figures and documents

### Main Figure-4/SSI products

Build the final narrative from corrected production artifacts:

1. fixed-retina SF×TF sensitivity;
2. FEM-created retinal power;
3. matched spectral drive;
4. concrete response-map/SSI examples;
5. population response and SF dependence;
6. out-of-sample mechanistic prediction;
7. additive/multiplicative interpretation, only if supported;
8. orientation/geometry as secondary or supplementary evidence.

### Documentation

- Update the top-level provenance warning when corrected replacements exist.
- Update `IMPACT_NOTES.md` with replacement paths and status.
- Maintain a machine-readable artifact ledger with:
  - source identity and hashes;
  - corrected contract;
  - cohort;
  - status: valid, provisional, legacy structural-only, superseded, or replaced;
  - replacement artifact where available.
- Clearly label checkpoint numbers as internal provenance labels, not a pre-established scientific framework.
- Preserve legacy figures but remove them from promoted/current figure paths.

## Immediate order of work while the large GPU run is deferred

1. Complete Phase 1 population trace-agreement summaries.
2. Complete the 100-image identity/crop audit.
3. Repair and regenerate the BackImage window/timebase tables.
4. Create the cross-stream artifact/status ledger.
5. Run the legacy-cache integrity, convergence, influence, and low-rank analyses.
6. Freeze auditable examples and the small bridge cohort.
7. Build input-only corrected spectral sufficient-statistics caches.
8. Implement and unit-test explicit-history rendering and response indexing.
9. Prepare—not launch—the small bridge and large production run manifests.
10. Pause for GPU scheduling and user approval before neural scoring.

## Estimated effort before the large GPU run

These are working estimates, not guarantees:

- Trace population agreement and crosswalk: less than one working day.
- 100-image crop/content audit: approximately one working day.
- BackImage timebase/window regeneration and validation: one to two working days.
- Legacy-cache structural analyses and selection tables: one to two working days.
- Input-only spectral bank and summaries: one to several working days, depending on CPU rendering throughput and retained spectral resolution.
- Explicit-history tests and frozen run manifests: approximately one working day.

Total no-large-GPU preparation: approximately four to eight working days of analysis time, much of which can run unattended once the contracts are frozen.

## Completion criteria

The pre-GPU phase is complete when:

- every promoted existing artifact has a status and provenance record;
- the trace and image identity crosswalks are complete;
- behavior/timebase labels are corrected;
- the legacy cache has a documented structural-only reuse verdict;
- explicit recorded history is implemented and unit-tested;
- the small bridge cohort and 100 × 1,000 production cohorts are frozen;
- all primary analyses have predeclared metrics, baselines, folds, and example-selection roles;
- no current figure or report silently consumes a superseded cache.

The full analysis is complete only after the corrected neural response bank, extended TF coverage where required, map-first response checks, population analyses, robustness checks, and updated report have all passed their respective gates.

## Key supporting artifacts

- Eye-trace and Nyquist audit: `outputs/fig4_active_sensing/rr100_eye_trace_conditioning_nyquist_audit_checkpoint_19_v1/`
- Analysis impact notes: `outputs/fig4_active_sensing/rr100_eye_trace_conditioning_nyquist_audit_checkpoint_19_v1/IMPACT_NOTES.md`
- Corrected 1,000-trace descriptor checkpoint: `outputs/fig4_active_sensing/rr100_legacy1000_corrected_trace_descriptors_v1/`
- Corrected response-map smoke test: `outputs/fig4_active_sensing/rr100_corrected_ssi_map_first_smoke_checkpoint_20_v2/`
- Provisional 16 × 32 SSI validation bank: `outputs/fig4_active_sensing/rr100_corrected_ssi_validated_halves_fullbank_checkpoint_21_v1/`
- Legacy 100 × 1,000 response cache: `outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/`
- Fixed-retina parametric models: `outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1/`
