# Data And Code Inventory

This is a starting map of existing repo assets that may support the active-sensing movie-information figure.

## Existing Planning Documents

- `declan/e1_active_sensing_efficiency_revised_handoff.md`
  - Best prior handoff for spatial SSI and E-optotype identity efficiency.
  - Emphasizes that model-only results should not be overstated as biological active-sensing proof.

- `declan/fixrsvp_trajectory_implementation_readiness.md`
  - FixRSVP trajectory stages, loader choices, microsaccade detection, drift segmentation, and QC contracts.
  - Useful for real trace extraction and drift/microsaccade windowing.

- `declan/fem_path_integrated_separability_handoff.md`
  - Useful for path-integrated identity separability and phase sampling.
  - More tangent-adjacent than the current active-sensing goal, but contains useful cumulative/path logic.

## Existing Scripts To Inspect Before Implementation

- `jake/twininfo/`
  - Most relevant existing implementation of the movie-information analysis.
  - `pipeline.py` is the production entry point and writes cumulative Fisher,
    spatial SSI, prefix-normalized bits/spike, phase controls, SF controls, and
    gain summaries under `outputs/twininfo/<run_name>/`.
  - `trace_selection.py` and `retinal_examples.py` select real fixation-only
    and one-microsaccade windows using an operational speed-threshold event
    detector; this is a cleaner drift/microsaccade split than the temporary
    step-size marker in this folder's pilot runner.
  - `retinal_examples.py` renders model-aligned lag cubes with the same
    `make_counterfactual_stim` path as the digital twin, avoiding the approximate
    SciPy image-shift renderer used in the first pilot.
  - `lagcube_information.py` and `information.py` implement cumulative pattern
    Fisher information and spatial single-spike information from full
    convolutional rate maps. The saved `cumulative_spatial_ssi_bits_per_spike`
    endpoint is prefix-normalized by expected spikes and is therefore the most
    appropriate primary metric when guarding against larger eye movements simply
    increasing spike count.
  - Unit tests in `jake/twininfo/tests/test_lagcube_information.py` verify that
    scaling all rates scales raw cumulative bits but leaves cumulative
    bits/spike unchanged.

- `check_fixrsvp_model_spatialinfo.py`
  - Prior spatial SSI entry point, if present in the checkout.

- `spatial_info.py`
  - Expected location for `spatial_ssi_population` and counterfactual stimulus helpers, if present.

- `eval/fixrsvp.py`
  - Canonical fixRSVP loader referenced by existing handoffs.

- `scripts/run_fixrsvp_trajectory_pilot.py`
  - Existing trajectory pilot scaffold.

- `scripts/fixrsvp_eye_conventions.py`
  - Eye-position convention and px/deg conversion helpers.

- `check_fixrsvp_model_fisherinfo.py`
  - Secondary source for Fisher-information code.

Some paths may have moved. Use `rg` before implementation rather than assuming exact file locations.

## Existing Output Families To Inspect

- `outputs/Allen_*_fixrsvp_trajectory_*`
  - Existing fixRSVP trajectory QC, microsaccade tables, drift segments, response trajectory diagnostics, and covariance alignment outputs.

- `outputs/supp_eoptotype_phase_sampling*`
  - Existing phase-sampling summaries, real-minus-stationary gains, bootstrap summaries, pairwise separation, and condition metadata.

- `outputs/stats/fem_path_integrated_separability*`
  - Existing path-integrated identity separability outputs.

- `outputs/tangent_subspace_information/panelE_production_fisher*`
  - Useful only as a support/bridge to tangent geometry, not as the organizing active-sensing result.

## Expected New Output Root

Use a separate output family:

```text
outputs/active_sensing_movie_information/<run_label>/
```

Suggested subdirectories:

```text
movies/
responses/
metrics/
figures/
qc/
logs/
summaries/
```

## Minimum Machine-Readable Outputs

- `run_config.json`
- `trajectory_qc.csv`
- `stimulus_condition_manifest.csv`
- `movie_condition_manifest.csv`
- `retinal_movie_diagnostics.csv`
- `retinal_movie_frequency_summary.csv`
- `cumulative_information_by_movie.csv`
- `information_gain_summary.csv`
- `drift_microsaccade_decomposition.csv`
- `spectrum_control_summary.csv`
- `phase_control_summary.csv`
- `bootstrap_summary.csv`
- `decision_table.csv`

## First Implementation Pass

Start with one reproducible pilot:

```text
run_active_sensing_movie_information.py
```

Minimum pilot scope:

1. one session;
2. limited image set;
3. real FEM, stabilized, and one matched random-motion control;
4. pre-model retinal movie diagnostics for temporal contrast and band-specific temporal modulation;
5. intact natural images plus phase-scrambled controls if movie rendering is straightforward;
6. cumulative spatial SSI as the first natural-image model metric, with Fisher information used only after `theta` is pinned down;
7. paired bootstrap over image/trace movies.

Do not add tangent-basis projection metrics to the pilot. Add them only later as a bridge if the movie-information result is interpretable on its own.

## Recommended Next Pass

Treat `jake/twininfo` as the canonical implementation for the main active-sensing
analysis rather than extending the temporary pilot in parallel:

1. Run a small `jake.twininfo.pipeline` validation job with a fixed run name.
2. Use `metadata/01_trace_examples_used.csv` to report explicit fixation-only
   versus one-microsaccade windows.
3. Use `metadata/05_lagcube_information_summary.csv` for paired final endpoints,
   prioritizing `final_cumulative_spatial_ssi_bits_per_spike` over raw bits or
   bits/second.
4. Use `cache/cumulative_information_series.npz` for time-resolved curves in the
   figure.
5. Keep raw bits and bits/second as secondary diagnostics because they answer a
   different question: how much total information is generated when response
   drive is allowed to vary.
