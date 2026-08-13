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

## Current BackImage Contour-Axis Candidate

Important status update, 2026-07-16:

The previous BackImage contour-axis RR100 spatial-SSI caches are
interpretation-limited by a trace-construction bug. Selected BackImage source
windows were 128 samples, while the SSI runner used `n_timepoints=40`; the old
selected-window reconstruction path compressed each full 128-sample eye trace
into 40 model frames. This means nominal `scale=1` contour-axis traces were not
native 40-frame snippets, and diffusion/time-scale interpretations can be
substantially distorted.

Treat all prior contour-axis SSI outputs that could have used
`source_trace_contract = reconstructed_trace_bank_from_selected_windows` as
untrusted for calibrated motion-scale, diffusion-scale, aligned-vs-orthogonal,
or SF-group scale-curve claims until the run is checked or rerun. This includes
the long original/rot90 contour-axis caches and downstream unit-first,
population, orientation-stratified, and rotation-crossover summaries that
consume them.

Incident note and audit checklist:

```text
declan/active_sensing_movie_information/contour_axis_trace_resampling_bug_note.md
```

Corrected runs should use cache schema `3` and the source-trace contract:

```text
center_cropped_native_selected_window_trace_n_timepoints
```

Trace-bank runs should treat motion scale as a small predeclared metric bundle,
not as diffusion alone. The runner now writes `trace_bank_metric_summary.csv`
and `trace_bank_metric_summary_panel.{png,pdf}` next to
`trace_bank_metadata.csv`. The core bundle is:

- path length / path speed;
- RMS radius and BCEA68 spatial spread;
- MSD diffusion constant;
- covariance anisotropy, axis ratio, orientation, and normalized covariance
  shape entries;
- lag-1 autocorrelation;
- microsaccade event count, sample fraction, threshold, and peak speed.

Microsaccade contamination metrics are mandatory audit columns and should be
used either as exclusion criteria or as stratifiers before interpreting
diffusion- or scale-binned traces as drift-like snippets.

## Random Patch x Trace SSI Matrix

The collaborator-facing random-sampling object is generated by:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.active_sensing_movie_information.run_backimage_random_patch_trace_ssi_matrix \
  --n-patches 100 \
  --n-traces 1000 \
  --n-timepoints 32 \
  --trace-max-path-length-arcmin 350 \
  --out-dir outputs/active_sensing_movie_information/backimage_random_highcontrast_patch_trace_rr100_ssi_matrix_n100_p1000_t32_v1
```

This makes the full `100 x 1000` Cartesian set of movies from random
high-contrast BackImage patches and native center-cropped eye traces. It writes:

- `ssi_matrix.npy`: `movies x RR100 units` time-resolved spatial SSI,
  expected-spike weighted over frames;
- `expected_spikes_matrix.npy` and `mean_rate_matrix.npy`: companion unit
  response budgets for population summaries and leave-one-unit checks;
- `movie_feature_matrix.csv`: patch, trace, microsaccade-contamination, and
  trace-vs-image direction/axis conditioning variables;
- `patch_table.csv`, `trace_table.csv`, `unit_metadata.csv`, and
  `unit_tuning_matrix.csv`: the shareable source/unit tables. By default the
  tuning table is the current RR100 dynamic log-Gaussian marginal SF tuning
  table, joined into `unit_metadata.csv` and also copied as its own matrix.

Use `--dry-run` with small `--n-patches`, `--n-traces`, and `--max-movies` to
check manifests without loading the twin. The smoke output from the first code
check is:

```text
outputs/active_sensing_movie_information/backimage_random_patch_trace_ssi_matrix_dryrun_smoke_unitmeta/
```

The current live BackImage contour-axis result is documented in:

```text
backimage_spatial_ssi_revival_note.md
```

The current main diagnostic is the original-movies-only, contour-matched,
unit-first across-motion plot:

```text
outputs/active_sensing_movie_information/
  backimage_contour_axis_rr100_sf_contour_alignment_long_axis30_n576_low0p05_high0p5_v1/
    unit_first_original_only_alignment_split_v1/
      backimage_rr100_original_only_contour_matched_across_scale_setting_main.png
```

The measurement is unit-first and static-subtracted:

```text
Delta SSI_u(a,l) = SSI_u(a,l) - SSI_u(0,0)
plotted curve    = mean_u Delta SSI_u(a,l)
```

Only contour-matched unit-window pairs are included in the main plot. Low and
high SF are different unit groups, but each unit's motion effect is compared
against that same unit's fully static baseline before averaging.

Current interpretation:

> The contour-matched high-SF channel may set the upper useful scale of FEMs.

Low-SF contour-matched information continues increasing beyond the natural
`1x` motion scale. High-SF contour-matched information plateaus near `1x` for
pure across-contour motion and declines beyond `1x` when an along-contour
component is already present. This supports "scale-setting channel" language,
not a broad claim that high-SF units limit all information.

Keep the broader 2x4 original-only figure, orthogonal pairings, along-axis
sweeps, and original/rot90 crossover as controls. The pooled
aligned-minus-orthogonal curves are not the mechanistic alignment result
because fixed absolute-orientation anisotropy can dominate them.

Near-term microsaccade extension:

```text
declan/active_sensing_movie_information/run_backimage_contour_axis_rr100_spatial_ssi.py
  --trial-source-mode microsaccade_snippets
  --sweep-mode isotropic
```

This mode detects real high-speed microsaccade-like events in BackImage source
windows, cuts pre/event/post eye-trace snippets with a post-event tail for
transients, scales the centered snippet up and down, and sends the resulting
movies through the same RR100 twin/SSI path as the drift scale sweeps.

Microsaccade trace-mode controls now separate three questions:

- `full_snippet`: scale the full real pre/event/post trace, including drift.
- `padded_event_zero_rest`: isolate the detected event pulse and freeze all
  outside-event drift, useful as a pulse-only bound.
- `padded_event_scaled_full_snippet`: keep the original outside-event drift at
  `1x`, but scale only the detected microsaccade-event increments. In this mode
  `0x` is a drift-retained/event-removed reference, not a fully static movie,
  and is the preferred real-trace control before moving to synthetic
  Brownian-to-microsaccade continua.

Down the line, put drift windows and microsaccade snippets onto a shared
movement scale: raw arcmin, and unit-normalized phase scale
`movement_arcmin * preferred_SF_cycles_per_arcmin`. That would let the low-SF
and high-SF curves ask whether the same retinal displacement is large or small
relative to each unit's preferred spatial period.

## Temporal-Remapping Counterfactual Pilot

Fixed-geometry retiming and amplitude-duration counterfactuals are implemented
as a separate wrapper around the existing BackImage RR100 twin/SSI path:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.active_sensing_movie_information.run_backimage_temporal_remapping_pilot \
  --n-images 4 \
  --n-traces 8 \
  --n-timepoints 32 \
  --traversal-frames 8,12,16,24,32 \
  --timing-placements terminal,endpoint_hold,centered \
  --retiming-profiles uniform,natural_speed_profile \
  --dry-run \
  --out-dir outputs/active_sensing_movie_information/temporal_remapping/backimage_rr100_retiming_pilot_v1
```

By default, image patches are sampled from the saved Figure-4 matched-static
candidate table:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_matched_static_percandidate_gpu1_n128_c4_k16_scales_0p5_1_2_bconsistent_v1/candidate_sets.csv
```

That path pins the temporal-remapping image universe to the Figure-4 candidate
source rows and applies only basic source-window validity gates. Use
`--image-pool reviewed_windows` to recover the older behavior, which rebuilds a
larger reviewed-window pool with the contrast/coherence gates.

Drop `--dry-run` to score the generated counterfactual movies through the
frozen twin. The runner writes:

- `condition_table.csv`;
- `image_feature_table.csv`;
- `trace_feature_table.csv`;
- `retimed_trajectory_metrics.csv`;
- `summary.json`;
- `verification_01_selected_inputs.png`;
- `verification_02_timing_design.png`;
- `verification_03_trajectory_metrics.png`;
- `verification_04_geometry_invariance.png`;
- `qc/retiming_trace_*.png`;
- scored runs additionally write `retiming_ssi.npz`,
  `retiming_population_observations.csv`,
  `retiming_population_summary.csv`, `retiming_unit_observations.csv`,
  `verification_05_population_ssi_vs_velocity.png`, and
  `verification_06_unit_sf_group_curves.png`.

The discrete retiming convention is endpoint-inclusive: a traversal with
`D` model frames includes both \(\gamma(0)\) and \(\gamma(1)\), so the sampled
endpoint-to-endpoint interval is `(D - 1) / 120` seconds. The requested
duration `D / 120` and sampled endpoint interval are both written in the
metrics table. Continuous path geometry is checked separately from the
120-Hz model-sampled path, because short traversals necessarily sample fewer
points and can shortcut curved source paths.

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

Covariance-aware operating-regime implementation:

```text
jake/twininfo/run_covariance_optimality.py
summarize_covariance_optimality.py
outputs/active_sensing_movie_information/covariance_optimality/covopt_full_gpu1/
```

Post-audit status, 2026-06-13:

- `cov_pose_aware` now uses the same covariance-Fisher/ridge path as
  `cov_pose_blind` with no movement covariance, rather than aliasing the raw
  independent Fisher path.
- The former `D=0` ridge artifact is fixed: pose-aware and pose-blind rows
  match exactly when movement scale is zero.
- At empirical `D=1`, the corrected pose-aware minus pose-blind Fisher gaps
  remain positive but are slightly smaller than the pre-audit summary:
  scaled-real fixation `0.038 +/- 0.003`, scaled-real microsaccade
  `0.195 +/- 0.016` Fisher trace per expected spike.
- Random amplitude controls still match or exceed real in this metric, so the
  result supports a pose-relevant covariance-cost interpretation, not unique
  optimality of measured FEM trajectories.

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

Response-space accounting:

- See `../active_sensing_unit_space_provenance.md` for the cross-branch ledger
  of 16-channel, sampled-population, and 756-channel canonical results.
- General rule: use matched/session readouts when the claim depends on
  empirical-session comparability or individual recorded-neuron alignment. Use
  the canonical large digital twin when asking normative/mechanistic
  population-level questions that do not require individual-neuron matching.
- The production `twininfo` source run was configured with `population_size=16`
  and `population_grid_position_mode=full_grid`, so
  `metadata/00_population_units.csv` contains `16 x 51 x 51 = 41616`
  simulated spatial readout rows.
- The natural-image Checks 5-9 and current covariance-optimality summaries use
  the center/sample subset: 16 session-matched biological twin channels, not
  the full 41616-row spatial population.
- The Figure 4/TFTS compact-geometry basis is the canonical 756-response-channel
  shared readout. It is not dimension-compatible with the 16-channel
  natural-image response cache.
- The historical compact add-back/remove-out run therefore applies to the
  756-channel e-optotype/TFTS scaffold, not to the current 16-channel
  natural-image covariance curves.
- The plausible hierarchy
  `cov_pose_aware >= cov_geometry_aware >= cov_pose_blind` remains untested in
  this natural-image run. It requires all three curves in the same response
  coordinates: either rerun natural-image checks in the canonical 756-channel
  space, or build a matched compact basis for the exact 16 center channels
  while labeling the 16D ceiling/underpowering caveat.

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

## Covariance Hierarchy Update

The `covopt_geometry_hierarchy_n256` run should be labeled as:

> Strong low-rank covariance rescue, translation-geometry specificity untested.

At empirical `D=1`, the middle observer closes most of the pose-aware versus
pose-blind Fisher-efficiency gap (`k=2` about `96-98%`; `k=20` about
`99.7-99.9%`). That is a real result: the pose-blind cost is concentrated in a
low-dimensional movement-covariance subspace.

The important caveat is that the current `cov_geometry_aware_k` observer is not
an independent compact translation-tangent basis. It removes the top eigenspace
of the movement covariance being corrected, so it is effectively an oracle
top-PC covariance correction. In this run, `cov_geometry_aware_k` and
`cov_topPC_aware_k` are the same object.

Do not claim yet that the compact translation geometry discovered in the TFTS /
Figure 4 analyses is what closes the gap. The next implementation target is a
separate `cov_tangent_geometry_aware_k` observer whose basis is learned from
finite-difference translation tangents, ideally image-disjoint from the
covariance-optimality evaluation set. The decisive comparison is:

```text
oracle top movement-PC
independent translation-tangent basis
random subspace
unit-shuffled tangent basis
```

The signal-preservation diagnostic is also mixed: the top movement-covariance
subspace contains substantial coding/signal variance at larger `k`, so the safe
claim is covariance accounting in an overlapping signal/nuisance subspace, not
pure nuisance removal.

Exact tangent pathfinding, 2026-06-14:

```text
outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/tangent_geometry_pathfinding_d1/
```

This run projects the completed `N=256` covariance-optimality cache onto the
116 units that overlap the canonical 756-channel Figure 4/TFTS tangent basis,
then recomputes exact D=1 Fisher scores for `k = 2, 5, 10, 20`. It is a
partial-overlap pathfinding result, not a final matched 756-channel claim.

The independent tangent basis is consistently above random subspaces and
usually above unit-shuffled tangent controls, so there is some translation-
geometry-specific covariance structure. But it closes much less of the
pose-aware versus pose-blind gap than the oracle top movement-PC observer:
at `k=2`, tangent closure is about `0.18-0.25` while oracle top-PC closure is
about `0.95-0.98`; at `k=20`, tangent closure is about `0.33-0.41` while
oracle top-PC closure is about `0.998-0.999`. All 32 decision rows are labeled
`topPC_much_better_than_tangent`.

Safe interpretation: compact translation tangents capture a non-random slice
of the covariance penalty, but the large hierarchy rescue is still mostly a
generic low-rank movement-covariance/top-PC result in this response space.
The next decisive run should build the covariance cache directly in the full
canonical 756-channel tangent response space, or an explicitly matched tangent
and covariance response space, before making a strong geometry-rescue claim.

Same-cache cleanup sanity check:

```text
outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/tangent_geometry_pathfinding_d1_cache_tangent/
```

This variant builds a tangent basis directly from the cached Jacobian columns
in the same 116-unit subset. It is intentionally not independent, so it is only
a response-space sanity check. It does not rescue the functional readout
interpretation: mean gap closure for same-cache tangent is about `0.04`,
`0.08`, `0.09`, and `0.09` for `k = 2, 5, 10, 20`, compared with oracle
top-PC closure of about `0.97`, `0.99`, `1.00`, and `1.00`.

Status labels before the removal-semantics audit:

- Compact functional branch: useful negative. Independent translation tangents
  capture non-random covariance structure but do not account for the dominant
  covariance-aware information rescue.
- Compact structural branch: promotable as a structural mechanism for
  FEM-linked covariance, not as a demonstrated functional readout mechanism.

Removal-semantics pause, 2026-06-14:

The tangent-vs-oracle hierarchy interpretation is paused because the original
pathfinding used `cov - U(U.T cov U)U.T` plus PSD projection rather than the
matched noise-side-only residual `R Sigma_FEM R.T`. That difference matters for
non-eigenvector bases and can confound manifest/cache tangent comparisons.

Corrected audit implementation:

```text
declan/active_sensing_movie_information/run_noise_side_closure_audit.py
outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/noise_side_closure_audit_sampled/
```

This audit leaves responses, task derivatives, and signal terms untouched for
all bases. Every basis calls the same `covariance_residual_noise_side`
function, using `Sigma_k = R Sigma_FEM R.T` as the extra nuisance covariance.
Provenance for the sampled audit is `Sigma_FEM_pooled_residual`.

Sanity checks pass: `D=0` has zero covariance trace and `F_PA ~= F_PB ~= F_k`
to numerical precision, and a synthetic `Sigma_FEM = J Sigma_e J.T` closes with
residual trace near zero and closure `1.0`.

Sampled D=1 result, 12 rows per group and one random/unit-shuffled draw:

- Oracle top-PC still closes almost all of the gap: mean closure rises from
  `0.967` at `k=2` to `0.999` at `k=20`.
- Manifest tangent is stronger under corrected semantics than in the old audit:
  mean closure rises from `0.078` at `k=2` to `0.507` at `k=20`.
- Cache tangent does not span the pooled-residual `Sigma_FEM`: residual trace
  remains high, about `0.66` at `k=2` and `0.41` at `k=20`, with mean closure
  only `0.20-0.24`.

Current interpretation after correction: do not call this a settled useful
negative for compact tangents. The corrected sampled audit points first to a
`J`/`Sigma_FEM` provenance mismatch for cache tangents. A full all-row corrected
audit, ideally also checking `within_pair`, is needed before restoring a final
functional status label.

Covariance-target provenance audit, 2026-06-14:

```text
declan/active_sensing_movie_information/run_covariance_target_provenance_audit.py
outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/covariance_target_provenance_d1/
outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/covariance_target_provenance_d1_highk/
```

This audit was run as a trace-capture provenance check only. It intentionally
does not add more closure variants. The script compares three covariance
targets: exact reconstructed `J Sigma_e J.T` from the cached Jacobian columns,
the within-pair movement covariance, and the pooled-residual covariance target
used by the corrected closure audit.

Mean D=1 trace capture, 116 matched units:

- Exact `J Sigma_e J.T`: cache tangent captures `0.27`, `0.42`, `0.53`,
  and `0.63` at `k = 2, 5, 10, 20`; oracle top-PC captures `0.53`, `0.68`,
  `0.79`, and `0.89`; manifest tangent captures `0.03`, `0.09`, `0.14`,
  and `0.22`.
- Pooled-residual covariance: cache tangent captures `0.36`, `0.48`, `0.55`,
  and `0.59`; oracle top-PC captures `0.77`, `0.87`, `0.93`, and `0.97`;
  manifest tangent captures `0.03`, `0.10`, `0.15`, and `0.23`.
- Within-pair movement covariance is effectively identical to pooled residual
  for these trace-capture summaries, so broad pooling alone does not explain
  the mismatch.

The high-k check sharpens the provenance issue. At `k=116`, manifest tangent
spans the full 116D response space and captures all three targets, while the
current cache-tangent construction still captures only about `0.74` of the
exact `J Sigma_e J.T` trace and `0.64` of the pooled/within-pair trace.
Therefore the current cache basis should not yet be interpreted as "the" full
column space of the cached Jacobian target. A likely implementation-level
candidate is the unit-centering/rank handling used when constructing the cache
tangent basis, which can remove common-rate directions present in the
covariance target.

Updated status label: compact covariance functional branch is paused pending
covariance-provenance resolution. The corrected noise-side-only audit validates
the scoring semantics. Oracle top-PC closure remains near complete, but cache
tangents do not span the pooled-residual covariance target, and under the
current basis construction they also do not fully span the exact reconstructed
`J Sigma_e J.T` target. The previous useful-negative label should not be
interpreted until the covariance target and cache-basis definition are
decomposed.

Projection debug, 2026-06-14:

```text
declan/active_sensing_movie_information/run_covariance_projection_debug.py
outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/projection_debug_d1/
```

The minimal projection checks identify the source of the `k=116` surprise.
Identity projection captures all trace for exact `J Sigma_e J.T`,
pooled-residual, and within-pair targets. A basis built as `orth(J_exact)`
inside the same function that reconstructs `Sigma_J` also passes: trace capture
`1.0`, residual trace about `7e-31`, direct J residual about `7e-31`, closure
`1.0`, and rank `116`.

The existing cache tangent basis fails the full-basis sanity check because it
is built from unit-centered cached J columns. At `k=116`, it has rank `115`,
`||U U.T - I||_F = 1`, and it leaves about `0.257` of the uncentered J-column
Frobenius energy outside the subspace, while the centered-J residual is
numerically zero. Mean trace capture is therefore capped at about `0.744` for
exact `J Sigma_e J.T` and `0.638` for pooled/within-pair covariance. Unit hashes
match for the basis and covariance objects (`baac97be3d382ca4`), so this is a
centering/response-space definition issue, not a unit-order mismatch.

Uncentered exact-J closure follow-up, 2026-06-14:

```text
declan/active_sensing_movie_information/run_uncentered_j_tangent_closure_audit.py
outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/uncentered_j_tangent_closure_d1_sample6_k2_20_116/
```

This focused run uses corrected noise-side-only semantics and compares oracle
top-PC, uncentered exact-J tangent, centered exact-J tangent, manifest tangent,
random, and unit-shuffled manifest bases for exact `J Sigma_e J.T`,
within-pair, and pooled-residual covariance targets. Scope: D=1, 116 matched
units, six sampled rows per family/kind group, `k = 2, 20, 116`, one random and
one unit-shuffled draw. A full all-row closure pass was too slow for an
interactive run; trace/residual provenance remains available from the all-row
trace-only audit above.

Mean sampled closure and trace-capture result:

- Exact `J Sigma_e J.T`: uncentered exact-J nearly matches oracle top-PC at
  low/mid rank. At `k=2`, trace capture/closure are `0.517/0.892` for
  uncentered exact-J versus `0.531/0.893` for oracle. At `k=20`, they are
  `0.899/0.993` versus `0.915/0.994`. At `k=116`, both are exactly closed.
- Pooled-residual covariance: uncentered exact-J is also close to oracle. At
  `k=2`, trace capture/closure are `0.694/0.947` for uncentered exact-J versus
  `0.767/0.967` for oracle. At `k=20`, they are `0.943/0.998` versus
  `0.968/0.999`.
- Within-pair covariance is effectively identical to pooled residual in this
  run.
- Centered exact-J remains poor under the corrected semantics: pooled/within
  closure is about `0.21` at `k=2`, `0.24` at `k=20`, and only `0.23` even at
  `k=116`, because it omits the common-mode direction.
- Manifest tangent remains a separate, weaker independent-basis result: pooled
  closure is about `0.077` at `k=2` and `0.502` at `k=20`, with unit-shuffled
  manifest controls similar at `k=20`.

Updated coding conclusion: the previous same-cache tangent failure was a
basis-centering artifact. Under matched noise-side-only semantics, an
uncentered exact-J tangent basis strongly closes the exact linear covariance
target and, in this sampled pass, also closes most of the pooled/within
movement-covariance target. This revives the local-linear tangent mechanism as
a same-cache explanation, while leaving the independent manifest-basis
specificity question unresolved.

Current compact covariance branch status: positive matched-cache mechanism.
Uncentered local translation sensitivity explains the movement-covariance
rescue in the matched cache. The independent canonical/manifest tangent basis
remains partial and is not sufficient for the full rescue.

Final focused k=20 summary:

```text
declan/active_sensing_movie_information/summarize_uncentered_j_tangent_closure.py
outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/uncentered_j_tangent_closure_d1_sample6_k20_summary/
```

This summary reads the existing sampled closure audit and recomputes only the
signal fraction for the same sampled row/basis regime. At `k=20`, uncentered
exact-J is essentially oracle-like across all three targets:

| target | oracle closure | uncentered exact-J closure | centered exact-J closure | manifest closure | random closure | uncentered exact-J residual trace | uncentered exact-J signal frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact `J Sigma_e J.T` | `0.994` | `0.993` | `0.402` | `0.469` | `0.312` | `0.101` | `0.904` |
| pooled residual | `0.999` | `0.998` | `0.241` | `0.502` | `0.344` | `0.057` | `0.904` |
| within pair | `0.999` | `0.998` | `0.241` | `0.502` | `0.295` | `0.057` | `0.904` |

The compact figure is
`outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/uncentered_j_tangent_closure_d1_sample6_k20_summary/uncentered_j_tangent_k20_metrics.png`.
The main message is that the common-mode uncentered tangent is essential:
centered exact-J retains substantial signal fraction (`0.644`) but fails to
close the pooled/within covariance gap because it omits the common-mode
covariance direction. Manifest tangent remains an independent-basis bridge, not
the same-cache closure mechanism.

Canonical manifest convention check:

```text
declan/active_sensing_movie_information/rebuild_manifest_tangent_basis_conventions.py
outputs/active_sensing_movie_information/compact_basis_exports/manifest_tangent_basis_conventions/
```

The cached canonical TFTS tangent maps still contain raw per-object `bx/by`
vectors, so the canonical basis can be rebuilt in an uncentered convention
without rerunning the digital twin. This diagnostic exports both conventions:
raw uncentered `stack bx/by -> SVD` and the historical centered convention
`stack bx/by -> subtract per-unit mean tangent across source rows -> SVD`.

At the manifest/source level, however, the uncentered rebuild is almost the same
top-k subspace as the existing centered export. For the default `0.25` arcmin
delta, uncentered-vs-centered subspace overlap is `0.999` at `k=2`, `0.999` at
`k=20`, and `0.998` at `k=50`. The centered export is also exactly
reconstructed from the cache, confirming the provenance. Therefore this
particular convention change is unlikely by itself to make the manifest basis
behave like the same-cache uncentered exact-J basis. The manifest result should
still be treated as a weaker independent-basis companion unless a future
independent basis is rebuilt from source objects that better match the
natural-image covariance cache.

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
