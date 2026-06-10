# Figure 5 Additional Checks Prep

This note starts from `declan/Figure5_active_sensing_triage_plan.md` and turns
the next Figure 5 checks into an executable staging plan.

## Current Baseline

Use the production movie-information pipeline as the Figure 5 source of truth:

```text
jake/twininfo/pipeline.py
```

Best current run to treat as the baseline:

```text
outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu/
```

Current coverage in that run:

```text
conditions:
real
stabilized
random_amp
random_amp_cloud_matched
random_cov
phase_order_shuffle
pyramid_phase_scrambled
sf_low
sf_mid_low
sf_mid_high
sf_high
stabilized_pyramid_phase_scrambled
stabilized_sf_low
stabilized_sf_mid_low
stabilized_sf_mid_high
stabilized_sf_high

n rows:
1728 = 108 paired image/crop/trace units x 16 conditions
```

Primary endpoint:

```text
metadata/05_lagcube_information_summary.csv
final_cumulative_spatial_ssi_bits_per_spike
```

Time-resolved endpoint:

```text
cache/cumulative_information_series.npz
cumulative_spatial_ssi_bits_per_spike
```

Prepared utility:

```bash
.venv/bin/python declan/active_sensing_movie_information/summarize_figure5_additional_checks.py \
  --run-dir outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu
```

This writes:

```text
metadata/figure5_additional_checks_audit.csv
metadata/figure5_delta_curve_summary.csv
metadata/figure5_companion_metric_delta_summary.csv
metadata/figure5_trajectory_fairness_summary.csv
metadata/figure5_image_control_qc_summary.csv
metadata/figure5_trajectory_qc_gain_correlations.csv
metadata/figure5_retinal_transform_qc_summary.csv
metadata/figure5_retinal_transform_gain_regression.csv
```

Implemented checks:

- required-file, condition-coverage, metric-column, and series/summary
  consistency audit;
- primary bits/expected-spike final and time-window paired deltas;
- companion paired deltas for raw spatial bits, bits/sec, expected spikes,
  Fisher, and Fisher/spike, split by `all`, `fixation`, and `microsaccade`;
- trajectory-control fairness summaries from existing `03_trajectory_control_qc`
  fields;
- image-control QC summaries from `02_pyramid_image_control_audit`;
- correlations between trajectory QC mismatch metrics and random-control
  bits/spike gains.
- retinal movie transform QC summaries and transform-delta-to-gain
  correlations/regressions when `retinal_movie_transform_qc.csv` is available.

Observed all-image mean paired deltas from the baseline run:

```text
real - stabilized                              +0.03515
random_amp - stabilized                        +0.05360
random_amp_cloud_matched - stabilized          +0.05181
random_cov - stabilized                        +0.05154

sf_low - stabilized_sf_low                     +0.00997
sf_mid_low - stabilized_sf_mid_low             +0.03644
sf_mid_high - stabilized_sf_mid_high           +0.05331
sf_high - stabilized_sf_high                   +0.05065
pyramid_phase_scrambled - stabilized_pyramid_phase_scrambled  +0.03280
```

Interpretation at prep time:

- real FEM still beats stabilization in bits per expected spike;
- validated random-motion controls still beat real on the current metric;
- the strongest mechanism is the direct spatial-frequency profile under
  stabilized visual-control counterparts.

Therefore the main claim should remain:

> Real retinal motion improves model spatial-information efficiency over
> stabilization through a spectral-temporal mechanism.

Do not claim real-trajectory optimality unless a later validated control pass
changes the random-control result.

## Check 1: Baseline Output Audit

Goal:

> Confirm that the all-image run has the conditions, pair keys, and final/time
> series fields needed for every Figure 5 panel.

Required files:

```text
metadata/run_config.json
metadata/01_trace_examples_used.csv
metadata/03_trajectory_control_qc.csv
metadata/05_lagcube_information_summary.csv
metadata/05_information_series_records.csv
cache/cumulative_information_series.npz
```

Audit items:

- every non-stabilized condition has a matching stabilized row for each
  `(example_id, kind, image_index, crop_rank)` pair;
- the time-series records match the summary rows exactly;
- `random_amp_cloud_matched` provenance is mostly random-candidate, not
  fallback;
- final expected spikes, raw bits, bits/sec, and bits/spike are all present;
- `phase_order_shuffle` is treated as legacy `trajectory_order_shuffle`.

Deliverable:

```text
outputs/twininfo/<run>/metadata/figure5_additional_checks_audit.csv
```

Minimum decision:

- pass: proceed to Check 2;
- fail: patch the pipeline or regenerate the run before interpreting Figure 5.

## Check 2: Matched-Motion Fairness Extension

Goal:

> Decide whether random controls beat real because they are better active
> sampling, or because they sample a different retinal/image regime.

Already present in `metadata/03_trajectory_control_qc.csv`:

```text
path_length_deg
rms_displacement_deg
step_rms_deg
step_mean_deg
step_p95_deg
step_cov_*
pos_cov_*
*_rel_error fields
control_description
```

Add or summarize:

- temporal autocorrelation of x/y position and step amplitude;
- trajectory temporal power spectrum;
- valid rendered frame count;
- local image gradient and highpass energy sampled along each condition path;
- distance from crop hotspot or fixation center over time.

Decision:

- if random controls beat real while sampling higher gradient/highpass energy or
  larger occupancy, treat them as unfair for trajectory specificity;
- if random controls remain tightly matched and still beat real, demote
  trajectory-specific optimality.

## Check 3: Direct Delta Curves

Goal:

> Verify that real-minus-control bits/spike gains accumulate coherently over
> time.

Use:

```text
cache/cumulative_information_series.npz
metadata/05_information_series_records.csv
```

Primary curves:

```text
real - stabilized
real - random_amp
real - random_amp_cloud_matched
real - random_cov
sf_low - stabilized_sf_low
sf_mid_low - stabilized_sf_mid_low
sf_mid_high - stabilized_sf_mid_high
sf_high - stabilized_sf_high
```

Summary windows:

```text
early: 25 percent of analyzed samples
mid:   50 percent
late:  final sample
```

Deliverable:

```text
outputs/twininfo/<run>/metadata/figure5_delta_curve_summary.csv
outputs/twininfo/<run>/figures/figure5_delta_curve_checks.pdf
```

## Check 4: Retinal Transform To Gain Regression

Goal:

> Link the retinal movie transform directly to model bits/spike gain.

Predictors:

- temporal contrast;
- temporal power in slow/medium/fast bands;
- spatial-frequency-specific temporal modulation;
- local image gradient/highpass energy along the path;
- event kind: fixation-only versus one-microsaccade.

Response:

```text
paired final_cumulative_spatial_ssi_bits_per_spike gain
```

Preferred models:

```text
gain ~ temporal_modulation + expected_spikes + kind
gain ~ sf_band * temporal_modulation + kind
gain ~ condition + sampled_highpass_energy + trajectory_qc_terms
```

Keep these regressions descriptive. They explain the Figure 5 endpoint; they do
not replace the paired endpoint.

## Check 5: Recorded Reafference-Signal Alignment Fork

Central question:

> Is FEM-linked variability merely extra rate variance, or structured
> reafference whose relationship to coding axes makes it benign, limiting, or
> recoverable?

This fork uses recorded V1 and Figure 4 covariance estimates. It is a bridge,
not the primary Figure 5 movie endpoint.

Inputs:

```text
C_reaff:
    best controlled FEM-linked covariance estimate from Figure 4
    plus global-rate and PC1-residualized variants

C_signal:
    cross-validated covariance of stimulus mean responses

dmu_ij:
    pairwise stimulus mean difference vectors
```

Metrics:

```text
P_reaff = U_reaff U_reaff.T
alpha   = tr(P_reaff C_signal) / tr(C_signal)
L_ij    = dmu_ij.T C_reaff dmu_ij / max(||dmu_ij||^2, eps)
angles  = principal_angles(U_reaff, U_signal)
```

Nulls:

- unit shuffle;
- random k-dimensional subspaces;
- eigenvalue-matched random covariance;
- stimulus-label shuffle;
- train/test split-half estimates;
- global-rate and PC1 residualized controls.

Deliverable target:

```text
declan/active_sensing_movie_information/run_recorded_reafference_signal_alignment.py
outputs/active_sensing_movie_information/recorded_reafference_signal_alignment_<run>/
```

Interpretation:

- high alignment: reafference may be information-limiting for a pose-blind
  reader;
- low alignment: reafference is structured but mostly coding-orthogonal;
- null alignment: reafference is real, but not specially organized relative to
  the tested stimulus axes.

### Natural-Image-Only Implementation Decision: 2026-06-09

Use natural images only for Figure 5 Checks 5-9. The earlier cached
`lm-0.20` e-optotype checks are now treated as development scaffolding and
should not be used as Figure 5 evidence.

Natural-image-only runner:

```text
declan/active_sensing_movie_information/run_figure5_natural_image_population_checks_5_to_9.py
```

This runner recomputes center-location biological-twin responses for the
natural-image movies in a `jake.twininfo` run and uses natural-image identity
as the stimulus axis. For the current production run this means:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python declan/active_sensing_movie_information/run_figure5_natural_image_population_checks_5_to_9.py \
  --run-dir outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu \
  --out-dir outputs/active_sensing_movie_information/figure5_natural_image_population_checks_5_to_9
```

```text
run_dir:
outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu/

response space:
16 biological twin channels at the center readout location

stimulus classes:
natural image identities

repeats:
the selected fixation and microsaccade trace windows
```

Why center biological channels instead of the full spatial SSI population:

- the production twininfo spatial-SSI run evaluates 16 biological channels
  over a 51 x 51 spatial grid, yielding 41,616 simulated spatial readout units;
- full-covariance population coding in that full grid is not a well-conditioned
  or computationally appropriate Check 6 target;
- the center-channel response space gives a tractable natural-image
  population-coding diagnostic without returning to synthetic e-optotypes.

Check 8 status under this natural-image-only rule:

- the previous 756-unit Figure 4/TFTS basis is not dimension-compatible with
  the 16-channel center natural-image response space;
- compact add-back/remove-out should remain skipped until a compatible
  natural-image compact basis is generated for the same response channels.

### Completed Natural-Image Population Checks 5-9: 2026-06-09

Completed run:

```text
outputs/active_sensing_movie_information/figure5_natural_image_population_checks_5_to_9/
```

Output files:

```text
natural_image_center_rates.npz
natural_image_center_rate_records.csv
natural_image_center_rate_inventory.csv
check5_natural_image_reafference_signal_alignment.csv
check5_natural_image_covariance_spectrum_diagnostics.csv
check5_natural_image_pairwise_Lij.csv
check6_natural_image_constrained_dprime.csv
check6_natural_image_constrained_dprime_summary.csv
check7_natural_image_reafference_removeout.csv
check7_natural_image_reafference_removeout_summary.csv
check8_natural_image_compact_addback_removeout.csv
check9_natural_image_condition_sweep_alignment_summary.csv
manifest.json
```

Run counts:

```text
rate blocks: 1728
alignment rows: 32
dprime rows: 5616
removeout rows: 32
n images: 27
n repeats per image: 4
n units: 16
```

Main read:

- Check 5 does not show a real-specific alignment advantage. At k=2,
  `alpha_real = 0.88549` and `alpha_stabilized = 0.88550`; at k=10 both are
  near ceiling (`0.99728` and `0.99844`).
- Code audit after the run found no response-cache corruption: 1728 rows,
  27 images, 4 selected trace windows, 16 conditions, and no duplicate
  condition/image/example keys. Rate arrays are finite and shaped
  `(1728, 128, 16)`.
- One runner bug was fixed after audit: when a response cache already existed,
  rerunning with `--conditions ...` ignored the requested subset and analyzed
  every cached condition. The runner now filters cached records by requested
  conditions and validates that record rows match the NPZ rate rows.
- The new spectrum diagnostic explains why the natural-image alignment metric
  is prone to ceiling effects: in the 16-channel response space, real has
  signal top-2 variance fraction 0.906 and signal top-10 fraction 0.999; the
  corresponding residual fractions are 0.914 and 0.999.
- Check 6 gives real lower full-covariance image-identity dprime than
  stabilized (`dprime2_pop` 9.60 versus 11.14), but higher covariance
  efficiency (`eta` 1.499 versus 1.117).
- Random trajectory controls are comparable to or above real on `eta`
  (`random_amp` 1.469, `random_amp_cloud_matched` 1.503, `random_cov` 1.533),
  so this is not evidence for real-trajectory optimality.
- Check 7 does not show real recoverability under train-fold residual-PCA
  remove-out (`delta` -0.009 at k=2, 0.000 at k=10).
- Check 8 is correctly marked
  `skipped_missing_compatible_natural_image_basis`.

Interpretation:

> In the natural-image center-channel response space, real retinal motion has a
> more favorable covariance-efficiency ratio than stabilization under Check 6,
> but the stronger e-optotype scaffold result does not transfer: real is not
> uniquely better aligned with signal axes and is not recovered by residual
> subspace remove-out.

Important comparability caveat:

> The natural-image and e-optotype checks are not matched analyses. The
> natural-image run uses 16 center-channel responses, 27 image classes, and
> only 4 trace-window repeats per image. The e-optotype scaffold used 756
> response channels, 4 orientation classes, and up to 128 trials per
> orientation. Twininfo `stabilized` also holds each selected trace at its own
> mean eye position, whereas the e-optotype scaffold has a separate
> `fixed_center` condition. A direct stimulus-domain comparison requires either
> a 16-channel/downsampled e-optotype rerun or a richer natural-image response
> cache with a matched response space, repeat count, and stabilization
> definition.

Do not forget the tabled heavier comparison:

> Run natural-image Checks 5-9 in the canonical 756-response-channel space, or
> whichever exact response space is used by the Figure 4 compact basis, once
> the variance-accounting and constrained-coding priorities are stable.

### Historical E-Optotype Scaffold Status: 2026-06-09

Runnable scaffold:

```text
declan/active_sensing_movie_information/run_figure5_cached_rate_checks_5_to_9.py
```

Completed run:

```text
outputs/active_sensing_movie_information/figure5_cached_rate_checks_5_to_9_fixed_lm-020/
```

Run settings:

```text
source: model_cached_rates
logmar: -0.20
conditions:
real
stabilized
matched_null
fixed_center
scaled_0.05_current_n4_first
scaled_0.1_current_n4_first
k_list: 2,10
max_trials: 128
n_nulls: 100
n_splits: 5
```

Important scope limits:

- this is a cached deterministic model-rate scaffold, not the recorded-V1
  covariance fork described above;
- this scaffold uses synthetic e-optotype rates, not natural images;
- only the six cached `lm-0.20` conditions listed above were available under
  `scripts/temporal_decoding/data/rates`;
- scaled `0.05` and `0.1` conditions currently have only 4 trials per
  orientation, so their dprime and CV numbers are smoke-level diagnostics;
- `fixed_center` has near-zero residual variance and therefore enormous
  dprime values; treat it as a deterministic sanity check, not a fair active
  sensing control.

Current cached-rate read:

```text
Check 5 alpha, real vs stabilized:
    k=2:   0.710 vs 0.430
    k=10:  0.800 vs 0.650

Check 6 constrained dprime summary:
    dprime2_pop real       1.876
    dprime2_pop stabilized 2.068
    eta real               2.525
    eta stabilized         1.398

Check 7 train-fold remove-out:
    real k=2:        +0.041 accuracy
    real k=10:       +0.090 accuracy
    stabilized k=2:  -0.006 accuracy
    stabilized k=10: +0.057 accuracy
    matched_null k=10:+0.094 accuracy
```

Interpretation:

- real cached-rate residual structure is more aligned with stimulus axes than
  stabilized residual structure;
- real does not beat stabilized in absolute full-covariance dprime in this
  scaffold, but its covariance-efficiency ratio `eta` is higher;
- train-fold-fitted reafferent/tangent remove-out improves real decoding,
  consistent with recoverability, but `matched_null` also improves at k=10, so
  this is not evidence for real-trajectory optimality;
- Check 8 was intentionally skipped in this run because no external compact
  Figure 4 basis was supplied.

Decision:

- do not promote these e-optotype results into Figure 5;
- do not run additional e-optotype amplitude sweeps for Figure 5 unless they
  are explicitly labeled as synthetic-method controls outside the main claim.

## Check 6: Constrained Population Coding Metric

Goal:

> Recompute the active-sensing comparison with a metric where shared
> FEM-linked covariance can hurt.

Pairwise metric:

```text
dprime_pop^2   = dmu.T inv(Sigma) dmu
dprime_indep^2 = dmu.T inv(diag(Sigma)) dmu
eta            = dprime_pop^2 / max(dprime_indep^2, eps)
```

Run for:

```text
real
stabilized
random_amp
random_amp_cloud_matched
random_cov
sf_low / sf_mid_low / sf_mid_high / sf_high
pyramid_phase_scrambled
```

Decision:

- `eta_real >= eta_stabilized`: FEM-linked covariance is benign or useful under
  this metric;
- `eta_real < eta_stabilized`: retinal motion creates limiting variability
  unless pose is known;
- random approximately real: generic retinal-motion geometry is sufficient;
- random below real: biological trajectory statistics add specificity.

## Check 7: Pose-Aware Recoverability

Goal:

> Test whether eye-position or compact-reafferent conditioning recovers
> information that appears as shared variability to a pose-blind analysis.

Compare held-out readouts:

```text
pose-blind:
    stimulus/readout from responses alone

pose-aware:
    stimulus/readout with measured eye position, retinal pose, or trace
    coefficients

tangent-aware:
    stimulus/readout after explicit compact reafferent conditioning or removal
```

Implementation guardrails:

- fit pose/tangent terms on training data only;
- evaluate stimulus readout on held-out trials;
- report both accuracy/log-likelihood and calibration/residual covariance;
- keep wording as "recoverable by a pose-aware readout."

## Check 8: Compact Add-Back / Remove-Out

Goal:

> Test whether the compact Figure 4 geometry is mechanistic for Figure 5, rather
> than only descriptive of covariance.

For twin movie responses:

```text
delta_r(t)          = r_real(t) - r_stabilized(t)
delta_r_compact     = P_U10 delta_r(t)
delta_r_orthogonal  = (I - P_U10) delta_r(t)

r_compact_addback   = r_stabilized + delta_r_compact
r_orth_addback      = r_stabilized + delta_r_orthogonal
```

Ask:

- does compact addback recover FEM covariance closure?
- does compact addback recover constrained `eta` or pose-aware recoverability?
- does orthogonal addback fail?

Decision:

- compact addback succeeds: compact reafferent geometry carries the covariance
  and coding consequences of active retinal sampling;
- compact addback fails: compact geometry describes covariance, but functional
  information effects are distributed more broadly.

Natural-image implementation status:

- run
  `declan/active_sensing_movie_information/run_figure5_natural_image_population_checks_5_to_9.py`
  first for Checks 5-7 and Check 9 summaries;
- Check 8 should be skipped until a compact basis exists for the same
  natural-image center-channel response space;
- do not use the old 756-unit e-optotype/TFTS add-back run as a natural-image
  Check 8 result.

### Historical E-Optotype Check 8 Status: 2026-06-09

Exported external compact basis:

```text
outputs/active_sensing_movie_information/compact_basis_exports/figure4_tfts_compact_basis_delta025.npz
outputs/active_sensing_movie_information/compact_basis_exports/figure4_tfts_compact_basis_delta025_manifest.json
```

Source:

```text
outputs/twin_feature_tangent_structure_prod_limited_synth/tangent_maps/twin_tangent_maps.pkl
```

Construction:

```text
stack bx/by derivative vectors across 63 valid objects
unit-center columns
SVD over the 126 x 756 derivative matrix
save V.T as response-space basis, default delta = 0.25 arcmin
```

Basis compactness diagnostic:

```text
delta 0.25 arcmin:
    top2  variance fraction 0.378
    top10 variance fraction 0.687
    top20 variance fraction 0.828
```

Completed Check 8 run:

```text
outputs/active_sensing_movie_information/figure5_cached_rate_checks_5_to_9_check8_tfts_delta025_lm-020/
```

Mean add-back summary over six orientation pairs:

```text
compact_addback k=2:
    alpha              0.0745
    dprime2_pop        2.091
    eta                1.492
    remove-out gain   +0.0277

orthogonal_addback k=2:
    alpha              0.172
    dprime2_pop        1.882
    eta                2.613
    remove-out gain   -0.0020

compact_addback k=10:
    alpha              0.647
    dprime2_pop        2.152
    eta                1.590
    remove-out gain   +0.0477

orthogonal_addback k=10:
    alpha              0.777
    dprime2_pop        1.905
    eta                1.757
    remove-out gain   +0.0275
```

Interpretation:

- the external compact basis add-back was a valid non-self-fit e-optotype
  scaffold;
- compact k=10 add-back recovers stabilized-scale full-covariance dprime and
  a positive remove-out/recoverability effect;
- orthogonal add-back remains competitive and has higher alignment alpha in
  this cached-rate run, so the result does not yet show that the Figure 4
  compact basis uniquely carries the functional consequences of active retinal
  sampling;
- under the natural-image-only rule, this is historical/debugging context only,
  not a Figure 5 result.

## Optional Check 9: Amplitude / Diffusion Sweep

This is now optional and downstream. Use
`figure5_reafferent_covariance_plan.md` as the current priority order.

Run this only after the variance-accounting denominator, constrained-coding
metric, pose-aware recoverability test, and compact addback/removeout test are
stable.

Sweep FEM amplitude scale:

```text
s = 0, 0.25, 0.5, 1, 1.5, 2, 3
```

For each scale:

```text
independent-rate bits/spike
constrained eta or dprime/spike
raw bits
expected spikes
compact recruitment
linearity/local-tangent capture
```

Separate:

- intact natural images;
- four spatial-frequency bands;
- fixation-only and one-microsaccade windows.

## Updated Execution Order

Use the reframed plan:

```text
declan/active_sensing_movie_information/figure5_reafferent_covariance_plan.md
```

Priority order:

1. Variance-accounting denominator for reliable shared variability.
2. Constrained population coding with `eta = J_pop / J_indep` or equivalent
   covariance-aware dprime/Fisher metric.
3. Pose-aware or tangent-aware recoverability.
4. Compact addback/removeout in a matched response space.
5. Optional amplitude/diffusion sweep as a final dose-response check.

## Claim Discipline

Current safe Figure 5 wording:

> Real retinal motion improves V1-model spatial-information efficiency over
> stabilization, with the strongest direct evidence coming from a graded
> spatial-frequency mechanism.

Current unsafe wording:

> Real FEM trajectories are optimal.

Stronger wording becomes available only if validated random controls no longer
beat real, or if the recoverability/add-back forks show a trajectory-specific
computation that random controls cannot reproduce.
