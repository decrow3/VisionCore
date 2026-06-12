# Active-Sensing Movie Information Figure

This folder is a clean workspace for a broader active-sensing figure that is not organized around compact tangent geometry.

The canonical implementation is Jake's production movie-information pipeline:

```text
jake/twininfo/
```

This folder should now serve as the figure-planning, interpretation, and
figure-generation workspace built around `jake.twininfo` outputs. The local
`run_active_sensing_movie_information.py` script is a temporary exploratory
runner, not the source of truth for the final analysis.

## Core Question

Do real fixational eye-movement trajectories improve the efficiency with which the deterministic V1-model rate movie represents spatial information in natural-image movies, and does that efficiency gain follow spectral power, higher-order natural-image structure, or trajectory timing?

This is different from the spatial-content tangent diagnostic in `declan/fig4_cov_TFTS/`. That analysis asks whether finite translation response changes project into an image-disjoint tangent basis. The analysis here should instead preserve the movie-based logic:

1. choose real fixation, drift, and microsaccade traces;
2. render retinal stimulus movies;
3. run natural images and image controls through the model;
4. compute cumulative Fisher information, spatial SSI, or identity separability over time;
5. compare real trajectories against stabilized and matched trajectory controls;
6. stratify by spatial-frequency content, phase controls, and drift/microsaccade windows.

## Primary Endpoint

The first-pass figure should have one load-bearing endpoint:

> paired real-vs-control cumulative spatial information efficiency gain over matched image/trace movies, measured as cumulative bits per expected spike.

Everything else should explain that gain. Pre-model movie diagnostics ask what the eye trace does to the retinal stimulus. Model metrics ask whether the V1 twin uses the resulting temporal modulations. Trajectory, spectral, phase, and drift/microsaccade analyses are explanatory layers, not separate primary endpoints.

Use Jake's `final_cumulative_spatial_ssi_bits_per_spike` / `cumulative_spatial_ssi_bits_per_spike` outputs as the default endpoint. Raw cumulative bits and bits/second are important companion diagnostics, but they should not carry the primary claim because they can increase simply when larger eye movements or higher contrast drive more spikes.

Use disciplined language for the deterministic model: this is spatial
information available from the model rate movie under the assumed spatial-SSI
readout, normalized by expected spike count. It is not a direct measurement of
biological information in noisy V1 spike trains.

## Interpretation Guardrail

The Twininfo pipeline is valuable as a model-side retinal-motion diagnostic,
but it should not be used by itself as the backbone for the strong claim that
active sensing explains cortical variability. Its clean positive result is more
bounded: FEM-like retinal motion can increase a deterministic V1-model spatial
information-efficiency proxy relative to stabilization, especially for higher
spatial-frequency content.

Three claims must stay separate:

1. measured FEMs explain a component of recorded V1 shared variability;
2. retinal motion increases a V1-model information proxy relative to
   stabilization;
3. the animal's measured FEM statistics are uniquely or specially useful.

The current matched-motion controls weaken claim 3: random motion controls can
match or exceed real FEMs under the spatial-SSI bits/expected-spike endpoint.
So the safe wording is not "real FEM trajectories are optimal active-sensing
trajectories." The safer framing is:

> retinal image motion exposes structured, information-bearing response
> variation, and the recorded FEM-linked covariance is a reafferent component of
> V1 variability.

Stronger computational language requires additional constraints: validated
matched-motion controls, spike-count/frame-count audits, retinal movie
transform QC, and ideally a pose-aware versus pose-blind or
population-covariance metric where reafferent variability can help or hurt.

For the current implementation checklist and generated audit summaries, use:

```text
figure5_additional_checks_prep.md
summarize_figure5_additional_checks.py
```

For the current scientific priority order after the Checks 5-9 audit, use:

```text
figure5_reafferent_covariance_plan.md
```

First-pass variance-accounting implementation:

```text
summarize_reafferent_variance_accounting.py
figure5_reafferent_covariance_implementation_notes.md
outputs/active_sensing_movie_information/reafferent_variance_accounting/
```

This dashboard now includes a finite-difference trace-closure layer:

```text
variance_accounting_trace_closure.csv
variance_accounting_trace_closure_summary.csv
```

These tables put the saved finite-difference capture fractions into matched
target-covariance trace units. They are numerator accounting artifacts, not
yet the final `tr(C_reaff_explained) / tr(C_reliable_shared)` denominator.

First-pass constrained population-coding implementation:

```text
summarize_constrained_population_coding.py
outputs/active_sensing_movie_information/constrained_population_coding/
```

This summarizes the saved natural-image Check 6 pairwise rows into condition
means and paired real-minus-control contrasts for `dprime2_pop`,
`dprime2_indep`, and `eta = dprime2_pop / dprime2_indep`.

Pose-aware recoverability prep:

```text
run_figure5_natural_image_population_checks_5_to_9.py --export-pose-covariates
```

This optional runner mode exports `natural_image_condition_pose_summary.csv`
and `natural_image_condition_pose_frames.csv`, aligned to the cached
natural-image response records. Those files are the design-matrix bridge for a
future pose-aware decoder; they are not themselves a recoverability result.

Recorded pose-aware response-prediction bridge:

```text
declan/active_sensing_movie_information/run_recorded_pose_aware_prediction.py
outputs/active_sensing_movie_information/recorded_pose_aware_prediction/
```

This cache-first runner uses `outputs/cache/fig3_digitaltwin.pkl` recorded V1
spikes and measured eye position to compare trial-disjoint held-out penalized
Poisson prediction for a PSTH-only GLM baseline, eye-only, additive-eye, scalar
eye-gain, and coarse time-by-eye interaction models. It writes valid-aware
shuffled-eye controls, per-row penalty metadata, and session-bootstrap
model-ladder summaries. Treat additive-eye gains as pose-linked predictability;
translation-specific active-sensing language still requires the interaction
model to beat additive/gain/shuffled controls and a later geometry-enrichment
analysis.

Natural-image-only population Checks 5-9 now live here:

```text
declan/active_sensing_movie_information/run_figure5_natural_image_population_checks_5_to_9.py
```

This runner recomputes center-location biological-twin responses for the
natural-image movies described by the production `jake.twininfo` run, then
uses natural-image identity as the stimulus axis. It supersedes the earlier
e-optotype cached-rate scaffold for Figure 5 interpretation.

Default production invocation:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python declan/active_sensing_movie_information/run_figure5_natural_image_population_checks_5_to_9.py \
  --run-dir outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu \
  --out-dir outputs/active_sensing_movie_information/figure5_natural_image_population_checks_5_to_9
```

Completed natural-image run, 2026-06-09:

```text
outputs/active_sensing_movie_information/figure5_natural_image_population_checks_5_to_9/
```

Run scope:

```text
source: twininfo_natural_image_center_rates
stimulus axis: natural_image_identity
response space: 16 biological twin channels at the center readout location
inventory: 27 images x 4 selected trace windows x 16 conditions
```

Result summary:

- Check 5: real and stabilized have essentially identical reafference-signal
  alignment at k=2 (`alpha` 0.88549 versus 0.88550), so this run does not show
  a real-specific alignment advantage.
- Code audit note: the response cache and center-channel extraction were
  checked after the run. The cache has 1728 blocks with 27 images, 4 trace
  windows, 16 conditions, and no duplicate condition/image/example keys. The
  runner now also writes
  `check5_natural_image_covariance_spectrum_diagnostics.csv`, which shows why
  Check 5 saturates: in the 16-channel natural-image response space, the top 2
  signal PCs already explain about 91 percent of real signal variance and top
  10 explains about 99.9 percent.
- Check 6: real has lower full-covariance image-identity dprime than
  stabilized (`dprime2_pop` 9.60 versus 11.14), but higher covariance
  efficiency (`eta` 1.499 versus 1.117). Random controls are comparable to or
  above real on `eta`, so this is not trajectory-optimality evidence.
- Check 7: train-fold residual-PCA remove-out does not improve real
  image-identity decoding (`delta` -0.009 at k=2, 0.000 at k=10).
- Check 8: skipped because the old 756-unit Figure 4/TFTS basis is not
  compatible with this 16-channel natural-image response space.

Interpretation: the natural-image center-channel population run supports a
bounded covariance-efficiency claim for real retinal motion relative to
stabilization, but not the stronger e-optotype scaffold claim of
real-specific alignment or recoverability.

The natural-image and e-optotype runs are not yet matched analyses: this run
uses 16 center-channel responses, 27 image classes, and 4 trace-window repeats,
whereas the e-optotype scaffold used 756 response channels, 4 orientation
classes, and up to 128 trials per orientation. Treat differences between them
as a prompt for a matched response-space/repeat-count control, not as direct
stimulus-domain evidence. Also note that twininfo `stabilized` is
trial-mean-stabilized for each selected trace, not the e-optotype
`fixed_center` condition.

Historical e-optotype scaffold outputs are kept only as development artifacts:

```text
outputs/active_sensing_movie_information/figure5_cached_rate_checks_5_to_9_fixed_lm-020/
outputs/active_sensing_movie_information/figure5_cached_rate_checks_5_to_9_check8_tfts_delta025_lm-020/
```

Do not use those e-optotype outputs as Figure 5 evidence. They were useful for
debugging the population-coding machinery, but the active-sensing claim is
natural-image-only.

## Working Interpretation

The active-sensing hypothesis should not be forced to prove itself through tangent-subspace recruitment. Magnitude is not only a confound here: image power, edge energy, and motion-induced response gain may be part of the mechanism. A useful figure should therefore report both absolute information gain and control-normalized gains.

Drift and microsaccades should also be allowed to play different roles. Drift is naturally local and tangent-like. Microsaccades may instead act as repositioning events that move the retinal input into new local neighborhoods where drift can then accumulate information.

The first layer should be retinal, not neural: real FEMs should be tested for how they transform natural-image spatial spectra into temporal modulations before asking whether the twin converts those movies into cumulative information efficiency.

## Relationship To Tangent Geometry

The tangent/covariance result can be used later as a bridge:

> real eye movements generate structured, low-dimensional response modulations in V1.

It should not be the primary active-sensing claim. The primary claim should be about what real retinal movies do to information accumulation over time.

## Initial Deliverables

- `active_sensing_movie_information_plan.md`: detailed scientific and analysis plan.
- `data_and_code_inventory.md`: existing repo assets to inspect before implementation.
- `run_active_sensing_movie_information.py`: temporary exploratory runner for retinal movie diagnostics and optional twin spatial-SSI curves.
- Future generator: `generate_active_sensing_movie_information_figure.py`.

## Canonical Pipeline

Run a small real-data validation pass through Jake's pipeline:

```bash
.venv/bin/python -m jake.twininfo.pipeline \
  --run-name active_sensing_validation_small \
  --image-indices 24 25 26 \
  --n-crops-per-image 1 \
  --n-examples-per-kind 2 \
  --population-size 16 \
  --shift-grid-mode cross \
  --recompute
```

Optional movie QC:

```bash
.venv/bin/python -m jake.twininfo.pipeline \
  --run-name active_sensing_validation_small_movies \
  --image-indices 24 25 26 \
  --n-crops-per-image 1 \
  --n-examples-per-kind 2 \
  --population-size 16 \
  --shift-grid-mode cross \
  --make-stimulus-movies \
  --recompute
```

Key outputs are written under:

```text
outputs/twininfo/<run_name>/
```

Use these files for the active-sensing figure:

- `metadata/01_trace_examples_used.csv`
  - fixation-only versus one-microsaccade windows;
  - event onset and trace source metadata.
- `metadata/05_lagcube_information_summary.csv`
  - final per-movie model endpoints;
  - primary column: `final_cumulative_spatial_ssi_bits_per_spike`;
  - companion columns: raw spatial bits, bits/second, expected spikes, Fisher summaries;
  - default conditions include `real`, `stabilized`, `random_amp`,
    `random_cov`, `trajectory_order_shuffle`, visual phase/pyramid controls,
    and SF bands.
- `metadata/03_trajectory_control_qc.csv`
  - matched-motion control audits for path length, RMS displacement, step RMS,
    and step covariance.
- `cache/cumulative_information_series.npz`
  - time-resolved cumulative traces for figure panels.
- `metadata/02_pyramid_image_control_audit.csv`
  - image-control QC for phase/pyramid conditions.
- `figures/01_*trace_selection*.pdf`
  - audit plots for drift/microsaccade classification.

## Exploratory Runner

Dependency-light smoke test:

```bash
.venv/bin/python declan/active_sensing_movie_information/run_active_sensing_movie_information.py \
  --run-label smoke_test \
  --stimulus-conditions intact,phase_scrambled
```

The smoke path writes a clearly labeled `retinal_temporal_power_proxy` endpoint. It remains useful for quick plotting prototypes, but final model claims should come from `jake.twininfo`.

Digital-twin spatial SSI path:

```bash
.venv/bin/python declan/active_sensing_movie_information/run_active_sensing_movie_information.py \
  --run-label model_pilot \
  --stimulus-source nat_stack \
  --stimulus-conditions intact,phase_scrambled \
  --run-model
```

The `--run-model` path uses the cached fixRSVP fixation-bout pool by default:

```text
declan/fixrsvp_fixation_pool.pkl
```

The default model endpoint is:

```text
cumulative_spatial_bits_per_expected_spike
```

This is preferred over raw cumulative bits because it penalizes conditions that increase apparent information merely by driving the model to fire more. Companion raw and rate-normalized metrics are written to:

```text
metrics/model_efficiency_by_movie.csv
```

Available primary model metrics are:

- `cumulative_spatial_bits_per_expected_spike`
- `cumulative_spatial_bits`
- `mean_spatial_bits_per_sec_to_date`

Fisher information should be used as a supporting diagnostic unless the parameter `theta` is pinned down in the figure text.

## Naming Note

`trajectory_order_shuffle` and `pyramid_phase_scrambled` are different controls.

- `trajectory_order_shuffle` shuffles the temporal order of the measured eye
  positions while keeping the same sampled positions.
- `pyramid_phase_scrambled` changes the visual image content by scrambling
  local image phase in the pyramid control.

Older pilot outputs used `phase_order_shuffle` for the trajectory-order
control. Treat that legacy name as `trajectory_order_shuffle`, not as a visual
phase scramble.
