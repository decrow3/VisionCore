# Axis-conditioned BackImage trajectory observer implementation plan

Last updated: 2026-06-17

## Purpose

This note turns the FEM-V1 roadmap's proposed unifying analysis into an
implementation plan that uses the infrastructure already present in this repo.

The goal is to test whether the compact-geometry plus trajectory-observer branch
can explain the strongest BackImage active-sensing result:

```text
Observed drift is biased toward local oriented image structure, and
edge-parallel motion perturbs local pixels and V1-twin responses less than
edge-orthogonal motion.
```

The proposed bridge is an axis-conditioned BackImage trajectory-table observer.
For each local BackImage patch, compare trajectory catalogs constrained to local
edge-parallel and edge-orthogonal axes, alongside empirical and synthetic motion
families, using the same finite-table known-eye, zero-eye, and joint-eye
observer machinery already implemented.

The decisive question is:

```text
Do edge-parallel or real-like trajectories preserve image identity under
trajectory marginalization better than edge-orthogonal trajectories, while still
retaining enough motion-dependent response structure for posterior
concentration?
```

## Existing infrastructure to reuse

### BackImage trajectory-table observer

Use the existing finite observer stack:

- `declan/backimage_trajectory_observer/observer.py`
- `declan/backimage_trajectory_observer/candidate_sets.py`
- `declan/backimage_trajectory_observer/likelihood.py`
- `declan/fixation_statistics_by_stimulus/run_backimage_trajectory_table_observer.py`

This already provides:

- raw response-table schema:

```text
prior_lambda_counts[candidate, trajectory, time, unit]
known_lambda_counts[candidate, time, unit]
zero_lambda_counts[candidate, time, unit]
y_obs_counts[time, unit]
```

- expected-count Poisson likelihood scoring;
- known-eye, zero-eye, joint-eye, and best-single-trajectory diagnostics;
- posterior entropy and `N_eff / K`;
- nearest retained trajectory rank and distance;
- hard-negative candidate sets;
- `matched_static_response` candidate sets via stabilized static-response
  prepass;
- leave-one-out trajectory priors;
- response table caching and per-trial metadata.

This analysis should extend this runner or add a sibling runner that reuses the
same scoring functions. Do not create a second observer implementation.

### BackImage local image geometry and edge-parallel stability

Reuse the local image geometry columns from the reviewed BackImage manifest:

```text
image_edge_axis_deg
image_gradient_axis_deg
image_orientation_coherence
drift_orientation_deg
anisotropy
image_patch_distance_to_image_border_px
```

Use the existing edge-parallel stability machinery:

- `declan/fixation_statistics_by_stimulus/run_backimage_edge_parallel_stability_screen.py`
- `declan/fixation_statistics_by_stimulus/run_backimage_twin_drift_geometry.py`
- `declan/fixation_statistics_by_stimulus/summarize_backimage_twin_drift_geometry.py`

This already provides:

- edge-parallel and edge-orthogonal endpoint perturbation tests;
- pixel perturbation cost;
- optional V1-twin response perturbation cost;
- real drift versus edge-axis alignment;
- raw-edge baseline adjudication.

The new observer should consume or recompute these local axes, then push
axis-conditioned traces through the existing response-table observer.

### Compact retinal-translation geometry

Reuse the compact geometry outputs as context and optional diagnostics:

- `declan/compact_retinal_translation_geometry/`
- `outputs/compact_retinal_translation_geometry/`
- `outputs/twin_feature_tangent_structure_prod_v2`
- `outputs/matched_twin_covariance_closure_rf_null_step025_rfbacked_v2`

Current limitation:

```text
The promoted compact tangent-map cache supports cardinal +/-x and +/-y response
grids, but not arbitrary diagonal or edge-conditioned directions.
```

Therefore, the first axis-conditioned BackImage observer should use full twin
forward responses for edge-parallel and edge-orthogonal trajectory catalogs.
Compact-subspace projections can be added as a secondary diagnostic once the
full-response result is interpretable.

## Scientific design

### Primary trajectory families

For each selected BackImage window, construct matched trajectory catalogs:

```text
edge_parallel
edge_orthogonal
empirical
rotated
ou
brownian
static
```

Use the existing names `empirical`, `rotated`, `ou`, `brownian`, and `static`
where possible so current result readers remain compatible. Add explicit
axis-conditioned family labels for the new families:

```text
axis_edge_parallel
axis_edge_orthogonal
```

If useful later, add:

```text
axis_model_safe
axis_model_unsafe
axis_random_control
```

but do not add those until the raw-edge version is working.

### Axis-conditioned trace construction

For each observed trace source and local edge axis:

1. Compute the unit vector along the local edge:

```text
u_parallel = [cos(theta_edge), sin(theta_edge)]
u_orthogonal = [-sin(theta_edge), cos(theta_edge)]
```

2. Build an axis-constrained displacement path using the observed trace as the
   scalar template.

Recommended first implementation:

```text
centered_trace = trace - mean(trace)
scalar_parallel_template = centered_trace dot u_parallel
scalar_orthogonal_template = centered_trace dot u_orthogonal
```

For `axis_edge_parallel`, place the scalar template on `u_parallel`.
For `axis_edge_orthogonal`, use the same scalar template but place it on
`u_orthogonal`, so amplitude and temporal structure are matched while direction
changes.

A stricter matched version should use a common scalar template for both axes,
chosen from the real trace projection with larger variance or from arclength
increments. That avoids giving one axis more temporal energy because the real
trace happened to be aligned with it.

3. Match or record these quantities for every generated trace:

```text
rms_displacement_deg
path_length_deg
duration_s
n_timebins
max_radius_deg
clipping_fraction
speed_mean_deg_s
speed_p95_deg_s
source_trace_id
axis_deg
axis_relation
```

4. Enforce the same clipping rules used by the current trajectory observer:

```text
--max-rms-deg
--max-rendered-trace-path-length-deg
--max-source-trace-path-length-deg
--max-trace-source-speed-p95-deg-s
```

If clipping differs between axis families, the run should fail or mark the trial
as invalid for the primary comparison.

### Observer comparisons

For each observation condition and prior family, reuse the existing observers:

```text
known-eye: log p(y | I, true trajectory)
zero-eye:  log p(y | I, static zero trajectory)
joint-eye: log sum_tau p(y | I, tau) p(tau)
```

Primary comparisons:

```text
axis_edge_parallel prior vs zero-eye
axis_edge_orthogonal prior vs zero-eye
axis_edge_parallel joint-eye vs axis_edge_orthogonal joint-eye
empirical joint-eye vs axis_edge_parallel joint-eye
empirical joint-eye vs ou/brownian/rotated joint-eye
```

Important distinction:

```text
Observation family and prior family should be recorded separately.
```

The first diagnostic can use empirical observations and compare different prior
families. A stronger follow-up should generate observations from
axis_edge_parallel and axis_edge_orthogonal as well, so the catalog-support
question is separated from the real-motion question.

## Implementation stages

### Stage 0: summarize current matched-static observer run

Before adding new axis families, finish and summarize the running matched-static
confirmation:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
```

Expected output files:

```text
observer_trials.csv
observer_summary.csv
candidate_sets.csv
response_cache_index.csv
trajectory_metadata.csv or motion_metadata.csv
summary_report.md
```

Decision:

```text
If joint-eye improvement collapses under matched_static_response candidates,
pause the axis-conditioned observer. The current observer claim would need
debugging before it can carry the unification story.
```

### Stage 1: add axis-conditioned trace utilities

Add a small utility module rather than growing the runner further:

```text
declan/backimage_trajectory_observer/axis_conditioned_traces.py
```

Suggested functions:

```text
axis_unit(axis_deg) -> np.ndarray
axis_perp(axis_deg) -> np.ndarray
trace_metrics(trace, dt) -> dict
axis_conditioned_trace(source_trace, axis_deg, relation, template_mode, scale, max_rms_deg) -> tuple[trace, meta]
matched_axis_trace_pair(source_trace, edge_axis_deg, template_mode, scale, max_rms_deg) -> dict
```

Keep this module pure NumPy so it is easy to test quickly.

Initial `relation` values:

```text
parallel
orthogonal
```

Initial `template_mode` values:

```text
same_parallel_projection
same_orthogonal_projection
same_dominant_projection
arclength_signed
```

Use `same_dominant_projection` as the first default if it yields stable
matching. If not, use `arclength_signed` because it better preserves path
length independent of the original drift axis.

### Stage 2: extend or wrap the trajectory observer runner

Preferred path:

```text
Add axis-conditioned family support to
declan/fixation_statistics_by_stimulus/run_backimage_trajectory_table_observer.py
```

New arguments:

```text
--axis-conditioned-families axis_edge_parallel,axis_edge_orthogonal
--axis-source-column image_edge_axis_deg
--axis-template-mode same_dominant_projection
--axis-match-policy strict
--axis-include-observation-families
```

Alternative path:

```text
Create declan/fixation_statistics_by_stimulus/run_backimage_axis_conditioned_trajectory_observer.py
```

The alternative is cleaner if the runner starts needing many axis-specific
metadata fields. In either case, scoring should still call
`score_image_identity_table`.

Required metadata additions:

```text
axis_conditioned: bool
axis_source_column
axis_deg
axis_relation
axis_template_mode
axis_match_policy
axis_pair_id
source_trace_rms_deg
rendered_trace_rms_deg
source_trace_path_length_deg
rendered_trace_path_length_deg
clipping_fraction
axis_match_status
```

### Stage 3: add pixel and V1 perturbation side metrics

The primary observer output should be joined to the local stability metrics:

```text
pixel_parallel_cost
pixel_orthogonal_cost
pixel_relative_advantage
twin_parallel_cost
twin_orthogonal_cost
twin_relative_advantage
drift_edge_cos2
image_orientation_coherence
drift_anisotropy
```

Implementation options:

1. Read the existing `edge_parallel_stability_by_window.csv` from the canonical
   edge-parallel screen and join by stable `source_row` or selected-window id.
2. Recompute the cheap pixel metrics inside the axis observer and only join
   twin metrics when available.

The first implementation can recompute pixel metrics and join twin metrics if
the file is present. The run should not fail just because optional twin
stability metrics are absent.

### Stage 4: smoke test

Run a small CPU or CUDA smoke:

```text
n_images = 8
n_candidates = 4
K = 4
candidate_set_mode = hard_negative_structure
observation_family = empirical
prior_families = axis_edge_parallel,axis_edge_orthogonal,empirical,ou,static
scale = 0.5
likelihood_scale = 1.0
trajectory_prior_mode = leave_one_out
```

Smoke success criteria:

```text
all tables have expected shapes
axis metadata is present for every axis-conditioned trajectory
parallel and orthogonal catalogs have matched RMS/path/duration
known-eye remains high on easy candidates
joint-eye is finite
N_eff / K is finite and in [1/K, 1]
response cache index points to valid npz files
```

Tests to add:

```text
declan/backimage_trajectory_observer/tests/test_axis_conditioned_traces.py
```

Minimum tests:

```text
parallel and orthogonal traces preserve shape
RMS and path length are matched under the default template mode
axis metadata is deterministic under fixed inputs
zero source traces return finite zero traces
clipping is reported
```

### Stage 5: primary diagnostic run

Run a deliberately small but interpretable diagnostic:

```text
n_images = 32 or 64
n_candidates = 4 or 8
K = 4 or 8 per prior family
candidate_set_modes = hard_negative_structure,matched_static_response
observation_family = empirical
prior_families = axis_edge_parallel,axis_edge_orthogonal,empirical,ou,rotated,brownian
scales = 0.5,1.0,2.0
likelihood_scales = 0.5,1.0
trajectory_prior_mode = leave_one_out
```

Include the `2.0x` row as the above-natural sentinel, giving the sweep a clean
half/natural/double structure. Treat it as a guard against the trivial regime
where larger motion simply makes the observer task easier or increases pose
damage, and audit effective RMS and clipping across axis families.

Primary output directory:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_trajectory_observer_<tag>/
```

Primary summary tables:

```text
observer_summary.csv
observer_trials.csv
axis_family_contrasts.csv
axis_match_quality.csv
axis_stability_join.csv
posterior_gain_correlation.csv
run_metadata.json
summary_report.md
```

### Stage 6: model-derived residual prediction

Only after Stage 5 is stable, ask whether the observer objective explains real
drift beyond raw edge geometry.

Per window, compute a model-side utility contrast:

```text
axis_joint_advantage =
  joint_true_score(axis_edge_parallel) - joint_true_score(axis_edge_orthogonal)

axis_pose_cost_advantage =
  N_eff_fraction(axis_edge_orthogonal) - N_eff_fraction(axis_edge_parallel)

axis_identity_preservation =
  joint_margin(axis_edge_parallel) - joint_margin(axis_edge_orthogonal)
```

Then predict observed drift alignment:

```text
drift_edge_cos2 ~ image_orientation_coherence
drift_edge_cos2 ~ image_orientation_coherence + axis_joint_advantage
drift_edge_cos2 ~ image_orientation_coherence + pixel_relative_advantage
drift_edge_cos2 ~ image_orientation_coherence + pixel_relative_advantage + axis_joint_advantage
```

Use within-session demeaning or session bootstrap, matching the existing
edge-parallel stability screen's style. The main question is whether the
observer-derived metric adds residual explanatory power beyond raw edge
coherence and pixel stability.

## Promotion and stopping rules

### Promote into the main paper if all are true

1. The matched-static observer confirmation preserves `joint-eye > zero-eye`.
2. Axis-conditioned catalogs are well matched on RMS, path length, duration, and
   clipping.
3. Edge-parallel or real-like priors improve joint robustness relative to
   edge-orthogonal priors.
4. Posterior concentration or margin diagnostics explain trial-level gains.
5. Observer-derived axis utility predicts observed drift alignment beyond raw
   edge geometry or identifies image-dependent cases where edge-parallel motion
   helps most.
6. The result can be explained in one concise Results section.

### Keep as a separate paper or supplement if any are true

1. The observer result is robust but does not beat raw edge geometry as an
   explanation of behavior.
2. The result requires too much machinery for the main shared-variability paper.
3. The result supports a trajectory-aware observer story but not a real-FEM
   active-sensing story.

### Demote or pause if any are true

1. `joint-eye > zero-eye` collapses under `matched_static_response`.
2. Axis-conditioned traces cannot be matched without heavy clipping.
3. Edge-orthogonal trajectories outperform edge-parallel trajectories in the
   preservation or robustness metrics.
4. Posterior concentration is diffuse and unrelated to score gain.
5. The result reduces to generic trajectory marginalization with no edge or
   real-motion specificity.

## First concrete implementation checklist

1. Summarize the active matched-static observer run.
2. Add `axis_conditioned_traces.py`.
3. Add unit tests for trace matching and metadata.
4. Add axis-conditioned family support to the trajectory observer runner.
5. Run an `n=8, K=4` smoke.
6. Run an `n=32 or 64` diagnostic with `hard_negative_structure`.
7. Repeat with `matched_static_response`.
8. Write a posthoc summary that joins observer outcomes to edge-parallel
   stability and real drift-axis alignment.
9. Decide whether the axis-conditioned result earns a main-paper role or stays
   as the seed of the compact/Wu observer paper.

## Expected interpretation patterns

### Strong unifying result

```text
known-eye high
zero-eye impaired
joint-eye(axis_edge_parallel) > joint-eye(axis_edge_orthogonal)
joint-eye(empirical) >= joint-eye(axis_edge_parallel)
lower N_eff / K predicts larger joint-minus-zero gain
axis utility predicts drift_edge_cos2 beyond raw edge coherence
```

Interpretation:

```text
Local natural-image geometry and trajectory-aware inference explain why
edge-parallel drift is useful: it preserves image identity and reduces pose
nuisance while retaining enough reafferent structure for latent-trajectory
marginalization.
```

### Useful but separate observer result

```text
joint-eye > zero-eye
empirical ~= OU ~= Brownian
axis_edge_parallel ~= axis_edge_orthogonal
raw edge geometry predicts behavior as well as or better than observer utility
```

Interpretation:

```text
Natural-image responses support pose-marginalized image identification, but the
current observer does not explain the observed along-contour behavior beyond raw
image geometry.
```

### Negative result

```text
known-eye high
zero-eye impaired
joint-eye near zero-eye
N_eff / K diffuse
matched-static candidates erase the effect
```

Interpretation:

```text
The current finite trajectory catalog or likelihood convention is not sufficient
to support the mechanistic observer claim. Keep the main paper focused on
recorded reafferent covariance and simpler BackImage local-geometry evidence.
```
