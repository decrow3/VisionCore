# Active-Sensing Unit-Space Provenance

Last curated: 2026-06-13.

This table records which response space each active-sensing-adjacent analysis
actually used. Its purpose is to prevent mixing 16-channel matched/session
results, sampled canonical-population results, and full 756-channel canonical
results when interpreting pose-aware, pose-blind, and geometry-aware observers.

## Current Rule

Use the 16-channel matched/session-centered twin when the question is tied to
the current Figure 5 natural-image endpoint or recorded-session comparability.
Use the full canonical twin, or at least a much larger sampled population, when
the question is about compact geometry, covariance-aware observer hierarchy, or
V1 population-level task objectives.

More generally: the canonical large digital twin is the default model-observer
space when we are asking normative or mechanistic questions of the data that do
not require the twin to serve as an empirical model of a specific recorded
session or as a comparison to individual neuron responses. Use matched/session
readouts when neuron identity, session comparability, or recorded-response
alignment is the claim; use the canonical/shared readout when the claim is about
population geometry, task objectives, observer assumptions, compact
translation structure, or population-level information.

The proposed hierarchy:

```text
cov_pose_aware >= cov_geometry_aware >= cov_pose_blind
```

is a same-response-space hypothesis. Do not assemble it from the current
16-channel covariance curves plus the historical 756-channel compact addback
scaffold.

## Provenance Table

| Analysis | Output folder | Population space | Units/channels | Readout type | Matched to recorded units? | Canonical/shared readout? | Claim type supported |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Figure 5 natural-image spatial-SSI endpoint | `outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu/` | Active-sensing production twininfo run | `16` biological units over a `51 x 51` spatial grid in metadata (`41616` simulated spatial readout rows) | Session-matched biological twin units with spatial readout grid | Yes | No | Natural-image endpoint continuity: retinal motion improves model spatial information efficiency over stabilization. |
| Natural-image population Checks 5-9 | `outputs/active_sensing_movie_information/figure5_natural_image_population_checks_5_to_9/` | Center/sample subset of the active-sensing production run | `16` center biological twin channels | Center-location/session-matched response cache | Yes | No | Bounded constrained-population sanity checks; not a compact-geometry test. |
| Covariance-aware active-sensing curves | `outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu/covariance_optimality/covopt_full_gpu1/` and `outputs/active_sensing_movie_information/covariance_optimality/covopt_full_gpu1/` | Center/sample subset selected by `--population-mode sampled_units` | `16` channels | Session-matched sampled biological twin readout | Yes | No | Pose-aware versus pose-blind covariance cost in the tested space; not sufficient for compact-geometry-aware middle observer claims. |
| Historical e-optotype Checks 5-9 scaffold | `outputs/active_sensing_movie_information/figure5_cached_rate_checks_5_to_9_fixed_lm-020/` | E-optotype cached-rate scaffold | `756` response channels | Canonical shared readout scaffold | No | Yes | Development/debugging of population-coding machinery; not natural-image Figure 5 evidence. |
| Historical compact addback/removeout Check 8 | `outputs/active_sensing_movie_information/figure5_cached_rate_checks_5_to_9_check8_tfts_delta025_lm-020/` | Figure 4/TFTS compact-basis response space | `756` response channels | Canonical shared readout with external compact basis | No | Yes | Valid e-optotype scaffold for compact addback/removeout; not the middle rung for current 16-channel covariance curves. |
| Figure 4/TFTS compact geometry | `outputs/twin_feature_tangent_structure_prod_limited_synth/` and `outputs/active_sensing_movie_information/compact_basis_exports/figure4_tfts_compact_basis_delta025.npz` | Canonical Figure 4/TFTS shared readout | `756` response channels | Shared tangent-basis readout | No | Yes | Translation effects are compact and image-generalizing. This is the correct scale for compact-geometry claims. |
| Vernier first pass | `outputs/vernier_active_sensing_first_pass/` | Canonical twin response space | `756` units in `information_summary.csv` | Canonical validated twin inference path | No | Yes | Hyperacuity-style pose-aware versus pose-blind phase-sampling result. |
| Vernier component-scale run | `outputs/vernier_active_sensing_component_scale/` and `outputs/vernier_active_sensing_component_scale_jake_labels/` | Canonical twin response space | `756` units in `information_summary.csv` | Canonical validated twin inference path | No | Yes | Component scale/phase-sampling follow-up in the full canonical unit space. |
| Vernier scale/pose sweep | `outputs/vernier_active_sensing_scale_pose_sweep_gpu0/` | Mixed within one summary table | Full pose-aware rows use `756`; compact-aware rows use `256` `top_abs_fd` subset from `756` original units | Canonical full rows plus response-selected subset rows | No | Full rows yes; subset rows are sampled from canonical | Main pose-aware curves are 756-unit; compact-aware controls are 256-subset controls and should be labeled as such. |
| BackImage twin drift-geometry adjudication | `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_twin_drift_geometry_scaled_n256_twin_axis_only/` | Sampled digital-twin population | `64` twin units in saved `run_metadata.json`; folder `n256` refers to `max_windows=256`, not units | Random sampled twin readout from `common.build_population` | No | No, sampled | Useful diagnostic against raw edge geometry, but not a full canonical-population objective. Avoid over-reading the negative PA/PB/Pareto result as a 756-unit failure. |
| BackImage conditional twin objectives | `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_conditional_fixation_objectives_twin_axis_only_n256/` | Sampled digital-twin population | `64` twin units in saved `run_metadata.json`; folder `n256` refers to windows | Random sampled twin readout | No | No, sampled | Diagnostic revised-objective branch; needs larger/canonical rerun before strong V1-population claims. |
| BackImage image-structure analyses | `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure_reviewed_v2_screenfiltered/` | Image/eye-trace feature space | Not a twin-response analysis | Raw local image/trajectory features | Not applicable | Not applicable | Local image geometry and FEM-statistics evidence; independent of twin population size. |
| Recorded pose-aware prediction | `outputs/active_sensing_movie_information/recorded_pose_aware_prediction_multisession_6pilot/` and related smoke folders | Recorded V1 spike-count space | Session-dependent recorded units | Recorded-cortex GLM ladder | Recorded data itself | No | Recorded-session bridge/control; not a digital-twin population-geometry result. |
| Input whitening | `outputs/active_sensing_movie_information/input_whitening/` | Retinal movie/input statistics | Not a twin-response analysis | Image-space temporal spectrum | Not applicable | Not applicable | Non-circular input-statistics result; biological drift whitens relative to stabilization but unconstrained whitening favors larger scale. |

## Immediate Consequences

- The current 16-channel covariance-optimality result should stay bounded:
  pose-aware versus pose-blind differences are clear in the tested space, but
  the compact-geometry-aware middle observer has not yet been tested where
  compact geometry was established.
- The Vernier first-pass and component-scale model results are in the expected
  756-unit canonical space. The scale/pose sweep also contains 256-unit
  compact-aware subset controls; those rows should not be described as full
  756-unit compact rescue tests.
- The BackImage twin drift-geometry negative is more provisional than the folder
  name suggests: it used `64` sampled twin units and `256` windows. A revised
  free-viewing objective can be worth rerunning in a larger sampled population
  or the full 756-channel canonical space, but the old PA/PB/Pareto objective
  should not be blindly repeated.

## Recommended Next Production Test

For covariance/geometry-aware observers, run all three observers in the same
population, stimuli, traces, and noise model:

```text
F_pose_aware
F_geometry_aware
F_pose_blind
```

Use `N=100` or `N=256` as the scalable first production target and `N=756` as
the clean canonical target if compute permits. Report:

```text
signal retained
nuisance variance removed
net Fisher change
```

This decomposition is necessary because compact projection can fail either by
leaving nuisance variance or by removing task signal along with nuisance.

Implementation note, 2026-06-13:

- `jake/twininfo/run_covariance_optimality.py` now supports
  `--population-source analysis --analysis-population-size N` for same-run
  large sampled populations.
- It also writes `cov_geometry_aware_k*` Fisher rows. The current first-pass
  geometry-aware observer conditions on the top-k movement-covariance
  eigenspace and evaluates Fisher with the residual movement covariance. This
  gives the intended same-space ladder:

```text
cov_pose_aware
cov_geometry_aware_k*
cov_pose_blind
```

- The runner also writes `covopt_geometry_diagnostics.csv`, including nuisance
  variance removed, residual nuisance variance, coding overlap, and signal
  overlap for the geometry subspace.

Suggested smoke:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m jake.twininfo.run_covariance_optimality \
  --from-run-dir outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu \
  --run-name covopt_geometry_hierarchy_smoke_n16 \
  --population-source analysis \
  --analysis-population-size 16 \
  --analysis-grid-position-mode center \
  --no-analysis-deduplicate-units \
  --no-use-center-rate-cache \
  --condition-families scaled_real \
  --scales 0,1 \
  --max-pairs 1 \
  --geometry-k-list 2,10 \
  --batch-size 16 \
  --fisher-step-arcmin 0.5 \
  --skip-sensitivity
```

Suggested first production target:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m jake.twininfo.run_covariance_optimality \
  --from-run-dir outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu \
  --run-name covopt_geometry_hierarchy_n256 \
  --population-source analysis \
  --analysis-population-size 256 \
  --analysis-grid-position-mode center \
  --condition-families scaled_real,random_amp_scaled,random_amp_cloud_matched_scaled,trajectory_order_shuffle_scaled \
  --scales 0,0.125,0.25,0.5,0.75,1,1.5,2,3 \
  --geometry-k-list 2,5,10,20 \
  --batch-size 64 \
  --fisher-step-arcmin 0.5
```
