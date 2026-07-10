# Vernier Active-Sensing Analysis

This package implements the first-pass Vernier active-sensing analysis described
in `declan/vernier_active_sensing_analysis_plan.md`.

The runner first writes rendering/provenance audits, then optionally runs the
digital twin on finite-difference Vernier pairs with paired trajectories.

## Model Stimulus Normalization

Every stimulus tensor entering the twin must match the dataset `pixelnorm`
contract: `(raw_u8 - 127.0) / 255.0`, with the intended neutral background near
`0.0` and the full 8-bit range roughly `[-0.5, +0.5]`. Renderer-local values and
display-normalized movies such as `[0, 1]`, `raw / 127.0`, or `raw / max_raw`
are not valid model inputs by themselves.

For the Vernier renderer specifically, model-bound tensors are converted from
the renderer-local `0..max_raw` range into raw 8-bit values first, then
pixel-normalized. The expected metadata value for corrected Vernier outputs is
`pixelnorm_renderer_raw_scaled_to_u8_minus_127_div_255`.

See `../../docs/digital_twin_stimulus_normalization.md` before adding or
rerunning a model-facing Vernier sweep.

## SSI Interpretation And Ratio Guardrails

Spatial spiking information (SSI) is a task-agnostic rate-map specificity
measure. For a unit map `r_u(x)`, the internal normalization
`g_u(x) = r_u(x) / mean_x r_u(x)` is intentional: it makes SSI measure spatial
concentration relative to that unit's own mean response, rather than raw gain.
In weak-modulation regimes it behaves like a normalized spatial-variance
measure, and it is zero for a uniform map.

Do not confuse that internal SSI normalization with later plot normalizations.
Curves labeled `SSI / static` or `log2 SSI / static` are fold-change diagnostics:

```text
log2_ratio_u(s) = log2((I_u(s) + eps) / (I_u(static) + eps))
```

Averaging `log2_ratio_u` across units is a geometric mean in ratio space, not an
arithmetic mean of raw ratios. That is still sensitive to tiny static SSI
denominators. A unit with `I_static ~= 0` can show a large positive fold-change
after a small absolute SSI increase.

For interpretation, prefer this hierarchy:

- Primary unit quantity: absolute `I_u(s)` in bits/spike.
- Primary change quantity: `I_u(s) - I_u(b)`, where `b` is either static or a
  matched movie baseline such as `0x`.
- Population/budget quantity: spike-weighted SSI, approximated in the RR100
  diagnostics by `sum_u mean_rate_u(s) * I_u(s)`.
- Fold-change quantity: useful as a diagnostic, especially with a predeclared
  denominator floor or after excluding units below a static-SSI floor.

The RR100 along=0 diagnostics include both the raw fold-change plots and the
denominator checks:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.plot_rr100_along0_denominator_diagnostics --mode both
```

The filtered polarity-group plot keeps only units whose static SSI exceeds the
chosen floor and reports geometric-mean ratio curves for retained all/positive/
negative groups:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.plot_rr100_along0_polarity_group_averages \
  --mode both \
  --min-static-ssi-bits 0.01 \
  --include-all-group
```

Treat the unfiltered unit-wise fold-change plots as denominator diagnostics, not
as evidence that weak-baseline units dominate the absolute spatial information
budget.

## Negative Probe Maps Are Not Suppression Labels

The RR100 `polarity` tables classify units from the static Vernier activation
map shape: a unit is labeled `negative` when the low tail below the map median
is larger than the high tail above the median. This is a descriptive
negative-probe-map label for the particular probe geometry used to make the
map, not a ground-truth contrast-suppression label.

Use stricter language:

- `negative-probe-map unit`: the cached static map has a dominant negative tail.
- `blank-referenced suppressive unit`: both `bright - blank` and `dark - blank`
  maps have dominant negative local modulation.
- `real contrast-suppressed unit`: the unit is validated against real spike
  responses, such as Gaborium ground truth.

The blank-referenced check is reproducible with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.check_rr100_bright_dark_suppression \
  --out-dir outputs/notebook_vernier_walkthrough/rr100_bright_dark_suppression_check \
  --polarity-csv outputs/notebook_vernier_walkthrough/rr100_real_trace_scale_grid/unit_ssi_along0_diagnostics/rr100_real_trace_along0_polarity_unit_table.csv \
  --device cuda:0
```

Under the corrected pixelnorm input convention, the current RR100 static check
finds that 33 of 41 negative-probe-map units are blank-referenced suppressive
for both bright and dark Vernier bars. The remaining units should not be called
contrast suppressive from Vernier maps alone.

Probe geometry matters. A vertical Vernier bar can produce negative-looking maps
in units that are not real contrast-suppressed cells, likely because fixed probe
orientation interacts with each channel's preferred orientation. When using real
Gaborium ground-truth labels, report false-positive rates by probe orientation
and size, and prefer a full orientation sweep such as
`0,22.5,45,67.5,90,112.5,135,157.5` before claiming that a negative map reflects
contrast suppression.

## Endpoint-History Last-Frame Contract

The endpoint-history runners use the final `history_frames` samples from each
valid recorded trace, not the first samples. The model input history is:

```text
tau_tail = tau[-history_frames:]
tau_endpoint[t] = tau_tail[t] - tau_tail[-1]
```

Then Fisher and SSI are computed from only the terminal model response frame.
Current endpoint caches declare `history_window=last_n_valid_trace_frames` and
`endpoint_alignment=last_history_window_minus_final_position`; older caches
without those fields should be treated as stale first-window outputs.

Regenerate the RR100 endpoint scale grid after changes to this contract with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_rr100_endpoint_history_scale_grid \
  --out-dir outputs/vernier_endpoint_history_last_frame_tutorial/rr100_endpoint_history_scale_grid \
  --device cuda:1 \
  --batch-size 64 \
  --force
```

`run_rr100_endpoint_history_scale_grid.py` always followed this contract correctly. As of 2026-07-07,
`plot_rr100_endpoint_along0_unit_ssi.py` (the per-unit along=0 diagnostic that cross-checks against the
grid summary) did not: it truncated each raw trace to the *first* `history_frames` samples via
`valid_trace(..., max_frames=...)` instead of taking the terminal window via
`terminal_history_window(...)`, so it was scoring a different, non-terminal chunk of real eye motion per
trace than the grid it was meant to audit. This is fixed — the script now imports
`terminal_history_window` and matches the contract above, its cache `schema_version` was bumped to 4 so
any stale caches from the old behavior are automatically invalidated and recomputed (no `--force`
needed), and its `compare_to_summary()` cross-check now reports a `max_abs_population_ratio_delta` of
~1e-16 against `rr100_endpoint_history_scale_grid_summary.csv` (previously nonzero). The two downstream
postprocessors, `plot_rr100_along0_polarity_group_averages.py` and
`plot_rr100_along0_denominator_diagnostics.py`, now also verify `cache_identity_json` (presence, minimum
`schema_version`, and cross-condition consistency) before combining per-condition caches into one along=0
line, so a future contract change will fail loudly there instead of silently blending old and new caches.

## `static_center` Is A Deterministic Oracle, Not A Trial-Mean Baseline

`static_center` gives every trial the same canonical zero retinal phase, so it is a deterministic oracle
by construction (zero within-condition variance), the same role `fixed_center` played in the earlier
E-optotype work (`declan/archive/eoptotype/fem_eoptotype_hyperacuity_results.md`) and the same caveat
already flagged in `declan/vernier_active_sensing_analysis_plan.md` under "Avoid fixed-center oracle
overinterpretation." The load-bearing static baseline described there is trial-mean/phase-cloud
diversity, not the oracle.

The `across=0, along=0` grid point (`real_aniso_across_0_along_0`) was intended to be that trial-mean
baseline (each trace held at its own mean position), but the two scale-grid scripts differ in whether it
actually is one:

- In `run_rr100_endpoint_history_scale_grid.py`, it is **not**. `endpoint_aligned_trace` subtracts each
  trace's own final frame (`arr - arr[-1:]`), and at `across=along=0` the pre-alignment trace is constant,
  so this subtraction always yields exactly zero regardless of what that per-trial constant was. Verified
  empirically (`rr100_endpoint_history_scale_grid_motion_inventory.csv`): every one of the 16 traces has
  `trace_x_mean_deg = trace_y_mean_deg = 0.0` for this condition, byte-identical to `static_center`
  (`pose_aware_fisher_mean = 0.003522` for both). This is a mathematical consequence of endpoint-alignment
  (which forces every condition to share the same terminal retinal position), not a bug, and it cannot be
  fixed by picking a different grid point — `0x,0x` can never carry trial-mean diversity under this
  contract. `phase_cloud_endpoint_history` in `run_endpoint_history_last_frame_readout.py`'s
  `ENDPOINT_CONDITIONS` is the closest existing non-oracle reference for that pipeline, since it shuffles
  (rather than holds constant) each trace's own positions before alignment.
- In `run_rr100_real_trace_scale_grid.py` (no endpoint-alignment), it **is**: `trace_x_mean_deg` for
  `real_aniso_across_0_along_0` genuinely varies from -0.34 deg to +0.25 deg across the 16 sampled traces
  (`rr100_real_trace_scale_grid_motion_inventory.csv`).

That trial-mean condition is also unusually noisy at `n_traces=16`: per-trace `pose_aware_fisher`
(trajectory-integrated) has `CV=0.77`, `SEM/mean=19%` at `across=0,along=0`, versus `CV=0.19-0.51`,
`SEM/mean=5-13%` for real-motion conditions (`across>=0.5`). A purely static sub-pixel view is very
sensitive to which arbitrary phase a given trace happens to land at relative to the model's pixel grid
(`model_pixel_arcmin = 60/RETINA_PPD ~= 1.6`, versus `fd_step_arcmin=0.25`); real motion integrates over
many phases within a trial and converges to a more consistent per-trial estimate. The full eye-trace pool
has 1059 traces (`scripts/temporal_decoding/data/eye_traces.npz`) versus the 16 currently subsampled, so
there is plenty of room to tighten this with more traces. An `n_traces=128` rerun of
`run_rr100_real_trace_scale_grid.py` was started 2026-07-07 to check whether the `0x,0x` SEM tightens up,
writing to `outputs/notebook_vernier_walkthrough/rr100_real_trace_scale_grid_n128` (kept separate from the
existing `n=16` run under `outputs/notebook_vernier_walkthrough/rr100_real_trace_scale_grid` for direct
comparison). Check that directory for results before re-running.

## Render-only smoke

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_vernier_active_sensing \
  --skip-model \
  --fd-steps-arcmin 0.25,0.5 \
  --out-dir outputs/vernier_active_sensing_smoke
```

## Tiny model smoke

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_vernier_active_sensing \
  --out-dir outputs/vernier_active_sensing_model_smoke \
  --n-traces 1 \
  --max-frames 3 \
  --fd-steps-arcmin 0.5 \
  --conditions static_center \
  --device cpu \
  --batch-size 2
```

The current joint-geometry observer is an instantaneous local-chart pilot. The
Wu-style interpretation here is Bayesian nuisance marginalization, not joint
image/eye reconstruction: Vernier sign is the desired latent, and eye trajectory
is nuisance state. The primary joint observer score is therefore a Vernier
likelihood ratio, approximating `log sum_w p(response | theta, w) p(w)` for
`theta in {+delta, -delta}`. Pose recovery metrics are diagnostics, not the
success criterion.

The `residual` likelihood mode reports Mahalanobis residual scores for
deterministic expected-count comparisons, not normalized log probabilities. In
residual mode, the raw true-hypothesis gap closure is suppressed and the
margin-based closure is reported separately. The lag-aware diagnostic path in
`run_lag_geometry_diagnostic.py` estimates temporal lag-plane translation
kernels and compares them with exact known-trajectory responses before those
kernels are promoted into the production joint filter.

The lag diagnostic writes three row families in `lag_geometry_diagnostic.csv`:
`rate_fidelity` for motion-induced response deltas, `likelihood_fidelity` for
exact-vs-approximate known-trajectory residual scores, and
`decision_fidelity` for whether the approximation preserves the known-eye
Vernier decision margins. Use those diagnostics to choose a history length
before running or implementing an expensive trajectory-marginalized observer.

## Reusable synthetic trajectory priors

Artificial FEM priors live in `declan.vernier_active_sensing.synthetic_trajectory_priors`
and are stimulus-agnostic: pass source eye traces in degrees and receive newly
sampled traces plus metadata. The current recommended comparison prior is the
empirical confined-step scale mixture, which fits a continuous distribution of
anti-persistent, spatially confined dynamics from real fixation snippets and
then samples fresh traces.

```python
from declan.vernier_active_sensing import (
    generate_synthetic_trajectory_prior,
    recommended_empirical_confined_config,
)

result = generate_synthetic_trajectory_prior(
    source_traces_deg,
    n_traces=256,
    n_frames=60,
    seed=0,
    config=recommended_empirical_confined_config(kappa_weight_power=0.5),
)
synthetic_traces_deg = result.traces_deg
prior_metadata = result.metadata
```

Use this prior when a decoder or figure needs a non-catalog artificial FEM
comparison matched to real fixation scale, step amplitude, and anti-persistent
confinement.

## Joint geometry observer smoke

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_vernier_active_sensing \
  --out-dir outputs/vernier_joint_geometry_smoke \
  --n-traces 2 \
  --max-frames 5 \
  --fd-steps-arcmin 0.5 \
  --conditions real_fem,static_center,order_shuffled_positions \
  --run-joint-geometry-observer \
  --joint-observer enumerated \
  --joint-compact-k-list 2 \
  --joint-eye-step-max-arcmin 1 \
  --joint-eye-step-sigma-arcmin 1 \
  --joint-eye-step-arcmin 1 \
  --joint-max-particles 3000 \
  --joint-likelihood-normalization residual \
  --device cpu \
  --batch-size 2
```

## Exact trajectory-table observer

This cache-only observer reads saved `rates_*.npz` files and evaluates the
Vernier likelihood ratio after marginalizing over the empirical trajectory
catalog. By default it uses a Poisson count likelihood and includes
the observed trajectory in the empirical prior. Pass `--leave-one-out` for a
generalization diagnostic that excludes the true trace from the nuisance
catalog.

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_trajectory_table_observer \
  --source-dir outputs/vernier_joint_geometry_enumerated_gpu0_fixed \
  --out-dir outputs/vernier_trajectory_table_observer \
  --conditions real_fem,static_center \
  --prior-conditions real_fem,static_center \
  --fd-steps-arcmin 0.25,0.5 \
  --reference-condition static_center \
  --likelihood-normalization poisson
```

## Noisy retinal-trajectory catalog observers

These cache-only observers keep the exact cached response table, but replace the
uniform trajectory prior with Gaussian weights centered on a trajectory cue. See
`noisy_retinal_trajectory_observer.md` for the motivation, math, literature
basis, and promotion guardrails.

For the RR100 scaled-real-trajectory grid, this is a cache-only postprocess:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_rr100_noisy_trajectory_observer \
  --source-dir outputs/notebook_vernier_walkthrough/rr100_real_trace_scale_grid \
  --out-dir outputs/notebook_vernier_walkthrough/rr100_noisy_trajectory_observer \
  --trajectory-sigmas-arcmin 0,0.125,0.25,0.5,1,2,inf
```

In this include-self version, `sigma=0` recovers the trajectory-known endpoint
because the true trajectory is retained in the catalog. `sigma=inf` recovers the
uniform trajectory-unknown marginal. Intermediate values test finite trajectory
precision using the saved real scaled traces, not synthetic traces.

For the more principled stepping-stone analysis, first generate a larger
scaled-real cache with along-contour motion fixed at `1x`:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_rr100_real_trace_scale_grid \
  --out-dir outputs/notebook_vernier_walkthrough/rr100_real_trace_along1_mc \
  --across-scales 0,0.125,0.25,0.5,0.75,1,1.5,2,3 \
  --along-scales 1 \
  --n-traces 160 \
  --max-frames 60 \
  --fd-step-arcmin 0.25 \
  --device cuda:1 \
  --batch-size 64 \
  --force
```

Then evaluate a leave-one-trajectory-out empirical catalog marginal:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_rr100_heldout_trajectory_observer \
  --source-dir outputs/notebook_vernier_walkthrough/rr100_real_trace_along1_mc \
  --out-dir outputs/notebook_vernier_walkthrough/rr100_heldout_trajectory_observer_along1 \
  --trajectory-sigmas-arcmin 0,0.25,0.5,1,2,4,8,inf \
  --prior-k-list 32,64,128 \
  --n-observation-traces 32 \
  --n-prior-traces 128 \
  --split-seed 0
```

Here observation traces and nuisance-prior traces are disjoint. The separate
`known_*` columns are the pose-aware endpoint; `sigma=inf` is the pose-unaware
uniform empirical marginal over held-out trajectories. In this held-out version,
`sigma=0` is nearest retained catalog trajectory, finite sigma values are local
catalog marginals, and `sigma=inf` is the uniform catalog marginal. The sigma
sweep does not interpolate to the pose-aware endpoint because the true
trajectory is deliberately excluded from the prior catalog.

Run the catalog-density diagnostic before interpreting held-out accuracy:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_rr100_catalog_mismatch_diagnostic \
  --source-dir outputs/notebook_vernier_walkthrough/rr100_real_trace_along1_mc \
  --out-dir outputs/notebook_vernier_walkthrough/rr100_catalog_mismatch_diagnostic_along1 \
  --n-observation-traces 32 \
  --n-prior-traces 128 \
  --split-seed 0
```

This compares same-sign trajectory mismatch to same-trajectory Vernier sign
distance. If `D_traj >> D_sign`, the held-out catalog is too sparse to support a
sharp finite-catalog marginal without an interpolation/noise model.

For the Vernier tutorial, treat this held-out catalog observer as a negative
control rather than a production solution. The primary tutorial result is the
pose-aware versus pose-unaware Fisher scale sweep; the catalog observer
documents why sparse whole-trajectory lookup is not the right bridge.

## Larger first pass

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_vernier_active_sensing \
  --out-dir outputs/vernier_active_sensing_first_pass \
  --n-traces 16 \
  --max-frames 60 \
  --fd-steps-arcmin 0.25,0.5 \
  --conditions static_center,static_repeated_phase,static_phase_cloud_single,static_phase_cloud_matched_positions,real_fem,order_shuffled_positions,axis_horizontal,axis_vertical,scaled_real_0.5,scaled_real_1.5 \
  --device cuda:0 \
  --batch-size 16
```

## Drift/microsaccade component scale pass

Component conditions use Jake's `jake.twininfo.eye_controls.detect_microsaccade_events`
labeling path. By default, the per-trace speed threshold is the robust MAD
threshold (`z=6`) with `1` frame of padding on each side. Pass
`--microsaccade-speed-threshold-dps 30` to force the earlier fixed threshold.

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_vernier_active_sensing \
  --out-dir outputs/vernier_active_sensing_component_scale \
  --n-traces 16 \
  --max-frames 60 \
  --fd-steps-arcmin 0.25,0.5 \
  --conditions static_center,static_phase_cloud_matched_positions,real_fem,drift_only_scaled_0.5,drift_only_scaled_1.0,drift_only_scaled_1.5,microsaccade_only_scaled_0.5,microsaccade_only_scaled_1.0,microsaccade_only_scaled_1.5,drift_scaled_0.5,drift_scaled_1.5,microsaccade_scaled_0.5,microsaccade_scaled_1.5 \
  --device cuda:0 \
  --batch-size 16 \
  --microsaccade-threshold-z 6 \
  --microsaccade-pad-frames 1
```

## Next-pass scale and pose-readout sweep

Scale-specific controls use aliases such as `static_phase_cloud_matched_scaled_0.5`
and `order_shuffled_scaled_0.5`, so reduced-amplitude motion is compared against
reduced-amplitude phase clouds rather than the full real-FEM cloud.
When `--full-cov-max-units` is smaller than the readout dimensionality, full-covariance
rows are labeled as unit-subset diagnostics.

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_vernier_active_sensing \
  --out-dir outputs/vernier_active_sensing_scale_pose_sweep \
  --n-traces 16 \
  --max-frames 60 \
  --fd-steps-arcmin 0.125,0.25,0.5,1.0 \
  --conditions static_center,real_fem,scaled_real_0,scaled_real_0.125,scaled_real_0.25,scaled_real_0.5,scaled_real_0.75,scaled_real_1.5,scaled_real_2,scaled_real_3,static_phase_cloud_matched_scaled_0,static_phase_cloud_matched_scaled_0.125,static_phase_cloud_matched_scaled_0.25,static_phase_cloud_matched_scaled_0.5,static_phase_cloud_matched_scaled_0.75,static_phase_cloud_matched_scaled_1,static_phase_cloud_matched_scaled_1.5,static_phase_cloud_matched_scaled_2,static_phase_cloud_matched_scaled_3,order_shuffled_scaled_0,order_shuffled_scaled_0.125,order_shuffled_scaled_0.25,order_shuffled_scaled_0.5,order_shuffled_scaled_0.75,order_shuffled_scaled_1,order_shuffled_scaled_1.5,order_shuffled_scaled_2,order_shuffled_scaled_3,axis_horizontal,axis_vertical \
  --pose-sigmas-arcmin 0,0.25,0.5,1,2,4 \
  --run-full-cov-pose-blind \
  --run-compact-aware-pose-blind \
  --compact-k-list 1,2,5,10 \
  --compact-alphas 0,0.25,0.5,0.75,1 \
  --full-cov-max-units 256 \
  --device cuda:0 \
  --batch-size 16
```

Summarize the run with the same directory as both the source and figure target:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.summarize_vernier_active_sensing \
  --run-dir outputs/vernier_active_sensing_scale_pose_sweep \
  --out-dir outputs/vernier_active_sensing_scale_pose_sweep
```

For the rotated-stimulus axis control, rerun the same axis conditions with:

```bash
--stimulus-orientation-deg 90
```

## Scale-specific phase-cloud controls

Use these conditions to test whether a scaled-real advantage survives baselines
matched to the scaled retinal positions, rather than to the full real-FEM cloud:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_vernier_active_sensing \
  --out-dir outputs/vernier_active_sensing_scale_sweep \
  --n-traces 16 \
  --max-frames 60 \
  --fd-steps-arcmin 0.125,0.25,0.5,1.0 \
  --conditions static_center,real_fem,static_phase_cloud_matched_positions,order_shuffled_positions,scaled_real_0,scaled_phase_cloud_matched_positions_0,scaled_order_shuffled_positions_0,scaled_real_0.125,scaled_phase_cloud_matched_positions_0.125,scaled_order_shuffled_positions_0.125,scaled_real_0.25,scaled_phase_cloud_matched_positions_0.25,scaled_order_shuffled_positions_0.25,scaled_real_0.5,scaled_phase_cloud_matched_positions_0.5,scaled_order_shuffled_positions_0.5,scaled_real_0.75,scaled_phase_cloud_matched_positions_0.75,scaled_order_shuffled_positions_0.75,scaled_real_1.5,scaled_phase_cloud_matched_positions_1.5,scaled_order_shuffled_positions_1.5,scaled_real_2,scaled_phase_cloud_matched_positions_2,scaled_order_shuffled_positions_2,scaled_real_3,scaled_phase_cloud_matched_positions_3,scaled_order_shuffled_positions_3 \
  --device cuda:0 \
  --batch-size 16
```

## Summarize and plot

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.summarize_vernier_active_sensing \
  --run-dir outputs/vernier_active_sensing_first_pass
```

Outputs:

- `render_audit/pixel_audit.json`
- `render_audit/pixel_audit_fd_rows.csv`
- `render_audit/*.png`
- `cache/rates_<condition>_fd<step>arcmin.npz`
- `information_summary.csv`
- `condition_reliability_summary.csv`
- `paired_baseline_contrasts.csv`
- `paired_baseline_contrast_summary.csv`
- `motion_inventory.csv`
- `vernier_active_sensing_manifest.json`
- `figures/*.png`
- `figures/source_tables/*.csv`
