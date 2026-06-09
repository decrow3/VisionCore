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
