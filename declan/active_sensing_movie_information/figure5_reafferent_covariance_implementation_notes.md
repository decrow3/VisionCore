# Figure 5 Reafferent Covariance Implementation Notes

These are running notes for the implementation pass that starts from
`figure5_reafferent_covariance_plan.md`.

## 2026-06-09: Start Priority 1

User request:

> "Alright lets go for implementation, keep good notes because I'm multitasking
> and will forget"

Current implementation target:

> Priority 1: variance-accounting denominator.

Reason:

- the Figure 5 story is now about reafferent covariance accounting and
  recoverability, not real-trajectory optimality;
- before building new heavy jobs, we should summarize what the existing Phase
  1 and derivative-geometry outputs already say about the denominator.

## Existing Output Families Inspected

Phase 1 FEM covariance:

```text
outputs/phase1_fem_covariance/
```

Useful tables:

```text
summaries/phase1_master_summary.csv
covariance_geometry/covariance_geometry_session_metrics.csv
covariance_geometry/model_alignment_metrics.csv
noise_correlations/noise_correlation_session_metrics.csv
aggregation_scaling/aggregation_scaling_session_metrics.csv
```

Direct recorded derivative / twin alignment:

```text
outputs/direct_recorded_derivative_twin_alignment_prod/
```

Useful tables:

```text
session_summary.csv
recorded_derivative_reliability.csv
tier1_compact_basis_capture.csv
tier2_matched_derivative_alignment.csv
null_summary.csv
```

Finite-difference closure:

```text
outputs/matched_twin_covariance_closure_fd_allen_step025/
```

Useful tables:

```text
finite_difference_session_summary.csv
finite_difference_metric_summary.csv
finite_difference_capture_metrics.csv
```

## Implementation Decision

First implementation should be a summarizer, not a raw-data recomputation.

Rationale:

- several expensive covariance jobs have already produced denominator-like and
  numerator-like summaries;
- a summarizer gives us a dashboard and exposes missing denominators;
- it is fast, safe to rerun, and does not disturb GPU jobs.

Planned script:

```text
declan/active_sensing_movie_information/summarize_reafferent_variance_accounting.py
```

Planned outputs:

```text
outputs/active_sensing_movie_information/reafferent_variance_accounting/
variance_accounting_session_rollup.csv
variance_accounting_component_candidates.csv
variance_accounting_aggregate_summary.csv
variance_accounting_summary.md
manifest.json
```

## Interpretation Guardrails

Not all fractions have the same denominator.

High-confidence denominator-ish fields:

- `aggregation_reliability_full`
- `aggregation_reliability_at_max_N`
- `eye_shuffle_reliability_at_max_N`
- `true_minus_eye_shuffle_reliability_at_max_N`
- `noise_corr_reduction_fraction`

Candidate numerator fields:

- direct-derivative compact capture;
- finite-difference tangent capture;
- model-alignment excess over nulls.

Do not collapse these into one headline number until the denominator is made
explicit for each row.

## 2026-06-09: First Summarizer Implemented

Script:

```text
declan/active_sensing_movie_information/summarize_reafferent_variance_accounting.py
```

Command:

```bash
.venv/bin/python declan/active_sensing_movie_information/summarize_reafferent_variance_accounting.py
```

Outputs:

```text
outputs/active_sensing_movie_information/reafferent_variance_accounting/
variance_accounting_component_candidates.csv
variance_accounting_session_rollup.csv
variance_accounting_aggregate_summary.csv
variance_accounting_trace_closure.csv
variance_accounting_trace_closure_summary.csv
variance_accounting_summary.md
manifest.json
```

Verification:

```bash
.venv/bin/python -m py_compile declan/active_sensing_movie_information/summarize_reafferent_variance_accounting.py
```

Run result:

```text
evidence rows: 1035
session rollup rows: 4
aggregate rows: 5
```

Current session-mean aggregate read:

```text
aggregation true-minus-eye-shuffle fraction: 0.848 +/- 0.040
noise-correlation reduction fraction:        0.333 +/- 0.037
model-alignment excess / reliability:        0.083 +/- 0.025
direct derivative compact capture:           0.484 +/- 0.030
finite-difference tangent capture:           0.436, one session
```

Important sign note:

- `phase1_master_summary.csv` stores `noise_corr_reduction` as
  corrected-minus-raw, so the raw value is negative when eye correction reduces
  correlations.
- The summarizer reports the positive numerator
  `raw_noise_corr_median - eye_corrected_corr_median`.

Current interpretation:

- There is strong denominator-like evidence that eye/FEM terms account for a
  large fraction of aggregation reliability above eye-shuffle controls.
- Noise-correlation evidence gives a more conservative but still meaningful
  fraction, about one third.
- Compact derivative and finite-difference tangent captures are promising
  numerator candidates, but they use target-covariance denominators rather
  than the raw reliable-shared-covariance denominator.

Next implementation need:

> Build or locate the raw covariance trace denominator:
> `tr(C_reaff_explained) / tr(C_reliable_shared)`.

## 2026-06-09: Trace-Closure Layer Added

Question:

> Can we move closer to a true variance-accounting denominator with existing
> saved outputs?

Search result:

- no saved full reliable-shared covariance matrices were found in
  `outputs/phase1_fem_covariance/`;
- the direct recorded-derivative production folder does not save target
  covariance traces at the row/context level;
- `outputs/matched_twin_covariance_closure_fd_allen_step025/finite_difference_capture_metrics.csv`
  does save `target_trace`, `capture`, null fractions, and excess fractions.

Implementation:

The summarizer now writes:

```text
variance_accounting_trace_closure.csv
variance_accounting_trace_closure_summary.csv
```

These tables convert finite-difference capture fractions into target
covariance trace units:

```text
captured_trace                 = capture * target_trace
unit_shuffle_null_trace        = unit_shuffle_null_median * target_trace
random_subspace_null_trace     = random_subspace_null_median * target_trace
excess_over_unit_shuffle_trace = effect_minus_unit_shuffle_median * target_trace
excess_over_random_trace       = effect_minus_random_subspace_median * target_trace
```

Verification:

```bash
.venv/bin/python -m py_compile declan/active_sensing_movie_information/summarize_reafferent_variance_accounting.py
.venv/bin/python declan/active_sensing_movie_information/summarize_reafferent_variance_accounting.py --help
.venv/bin/python declan/active_sensing_movie_information/summarize_reafferent_variance_accounting.py
```

Run result:

```text
evidence rows: 1035
session rollup rows: 4
aggregate rows: 5
trace closure rows: 15
trace closure summary rows: 15
```

Current finite-difference trace read, `Allen_2022-02-16` only:

```text
fd_sample_eye_trace_cov, no residualization:
  k=2  captured 26.116 / target 43.391 = 0.602
  k=10 captured 32.426 / target 43.391 = 0.747
  k=20 captured 35.527 / target 43.391 = 0.819

fd_sample_eye_trace_cov, global-rate residualized:
  k=2  captured 11.100 / target 27.624 = 0.402
  k=10 captured 16.954 / target 27.624 = 0.614
  k=20 captured 19.994 / target 27.624 = 0.724

fd_sample_eye_trace_cov, global-rate + target-PC1 residualized:
  k=2  captured 2.951 / target 15.656 = 0.188
  k=10 captured 6.677 / target 15.656 = 0.426
  k=20 captured 9.150 / target 15.656 = 0.584
```

Interpretation:

- this is a real improvement over a unitless capture fraction because it puts
  the numerator in covariance trace units;
- it is still a matched-target-covariance denominator, not the final
  reliable-shared covariance denominator;
- the full Priority 1 denominator needs producer-side outputs containing
  either covariance matrices or at least matched trace terms for
  `C_reliable_shared`, `C_reaff_explained`, and controls.

## 2026-06-09: Priority 2 Constrained-Coding Dashboard Added

Reason:

Priority 2 asks whether reafferent covariance is benign, limiting, or useful
under a covariance-aware population metric. The natural-image Check 6 run
already saved pairwise `dprime2_pop`, `dprime2_indep`, and `eta` rows, so this
can be summarized without rerunning the model.

Script:

```text
declan/active_sensing_movie_information/summarize_constrained_population_coding.py
```

Command:

```bash
.venv/bin/python declan/active_sensing_movie_information/summarize_constrained_population_coding.py
```

Outputs:

```text
outputs/active_sensing_movie_information/constrained_population_coding/
constrained_population_condition_summary.csv
constrained_population_real_contrasts.csv
constrained_population_summary.md
manifest.json
```

Verification:

```bash
.venv/bin/python -m py_compile declan/active_sensing_movie_information/summarize_constrained_population_coding.py
.venv/bin/python declan/active_sensing_movie_information/summarize_constrained_population_coding.py --help
.venv/bin/python declan/active_sensing_movie_information/summarize_constrained_population_coding.py
```

Run result:

```text
input rows: 5616
condition rows: 16
contrast rows: 5
```

Current paired real-minus-control read:

```text
real - stabilized:
  delta dprime2_pop   -1.548
  delta dprime2_indep -5.005
  delta eta           +0.382
  eta positive frac   0.724

real - random_cov:
  delta dprime2_pop   -0.270
  delta dprime2_indep -0.836
  delta eta           -0.035
  eta positive frac   0.476

real - random_amp_cloud_matched:
  delta dprime2_pop   -0.414
  delta dprime2_indep -3.314
  delta eta           -0.004
  eta positive frac   0.556

real - trajectory_order_shuffle:
  delta dprime2_pop   +0.143
  delta dprime2_indep -0.840
  delta eta           +0.194
  eta positive frac   0.613
```

Interpretation:

- real beats stabilized on covariance efficiency (`eta`) but loses absolute
  covariance-aware image-identity separability (`dprime2_pop`);
- random covariance and cloud-matched random controls match or slightly exceed
  real on `eta`;
- this supports the constrained-coding bridge but still does not support
  real-trajectory optimality;
- this remains the 16-channel natural-image center-response run. The tabled
  canonical 756-channel natural-image Check 6 remains the fair comparison.

## 2026-06-09: Priority 3 Pose-Covariate Export Hook Added

Reason:

Priority 3 needs a pose-aware or tangent-aware readout. The current
natural-image response cache has rates plus condition/image/example metadata,
but it does not save per-condition pose covariates. Without that design
matrix, Check 7 removeout should not be overinterpreted as pose-aware
recoverability.

Implementation:

`run_figure5_natural_image_population_checks_5_to_9.py` now has an explicit
flag:

```bash
.venv/bin/python declan/active_sensing_movie_information/run_figure5_natural_image_population_checks_5_to_9.py \
  --out-dir outputs/active_sensing_movie_information/figure5_natural_image_population_checks_5_to_9 \
  --export-pose-covariates
```

Expected outputs:

```text
outputs/active_sensing_movie_information/figure5_natural_image_population_checks_5_to_9/
natural_image_condition_pose_summary.csv
natural_image_condition_pose_frames.csv
```

## 2026-06-13: Covariance-Optimality Code Audit and Fix

Question:

> Are the covariance-aware results trustworthy if we inspect the implementation
> instead of the generated summaries?

Audit finding:

- `cov_pose_aware` in `jake/twininfo/run_covariance_optimality.py` was
  previously `f_ind.copy()`, so it was exactly equal to the raw independent
  Fisher path.
- `cov_pose_blind` used `covariance_fisher_by_time(...)`, which includes ridge
  regularization. This created a small nonzero pose-aware-minus-pose-blind gap
  even at `D=0`, where movement covariance is exactly zero.
- The sign of the covariance cost was not fabricated, but the pre-audit gaps
  mixed a real covariance penalty with a ridge-path mismatch.

Implementation fix:

- `cov_pose_aware` now calls `covariance_fisher_by_time(mu, J, None,
  ridge_frac=...)`, matching the pose-blind numerical path with zero extra
  movement covariance.
- The runner gained `--refresh-results` to regenerate result tables from the
  saved `mu/J` cache without rerendering model rates.
- The runner gained `--skip-sensitivity` to preserve the expensive gain/noise
  sensitivity table while refreshing core row metrics.
- `summarize_covariance_optimality.py` now reports explicit contrasts:
  `pose_gap`, `pose_gap_minus_D0`,
  `independent_minus_cov_pose_aware`, and
  `independent_minus_cov_pose_blind`.

Verification:

```text
D=0 cov_pose_aware_vs_blind max abs diff: 0
manual covariance-optimality tests: pass
py_compile: pass
```

Corrected empirical `D=1` pose-aware minus pose-blind Fisher gaps:

```text
scaled_real:
  fixation      0.0382 +/- 0.0032
  microsaccade  0.1952 +/- 0.0163

random_amp_scaled:
  fixation      0.0842 +/- 0.0063
  microsaccade  0.2472 +/- 0.0215

random_amp_cloud_matched_scaled:
  fixation      0.0541 +/- 0.0047
  microsaccade  0.2582 +/- 0.0185

trajectory_order_shuffle_scaled:
  fixation      0.0275 +/- 0.0024
  microsaccade  0.0925 +/- 0.0085
```

Interpretation after fix:

- real FEMs still create a meaningful pose-blind covariance cost, especially in
  microsaccade windows;
- knowing pose, under this model, recovers that cost;
- random amplitude controls still match or exceed real, so this remains
  evidence for pose-relevant reafferent covariance, not evidence for unique
  optimality of measured FEM trajectories.

What it does:

- reproduces the selected trace examples from the source twininfo run;
- reconstructs the exact trajectory-control trace for each cached response
  record with `_trajectory_for_condition`;
- writes per-record trace summaries and per-frame pose covariates aligned by
  `record_index`, `example_id`, `image_index`, `crop_rank`, and `condition`.

Verification run completed:

```bash
.venv/bin/python -m py_compile declan/active_sensing_movie_information/run_figure5_natural_image_population_checks_5_to_9.py
.venv/bin/python declan/active_sensing_movie_information/run_figure5_natural_image_population_checks_5_to_9.py --help
.venv/bin/python declan/active_sensing_movie_information/run_figure5_natural_image_population_checks_5_to_9.py \
  --out-dir outputs/active_sensing_movie_information/figure5_natural_image_population_checks_5_to_9
```

The cached no-export run still completed:

```text
alignment rows: 32
dprime rows: 5616
removeout rows: 32
```

Not yet run:

- `--export-pose-covariates`, because it explicitly loads the digital twin to
  reproduce selected traces and should be run outside the sandbox/CUDA-aware;
- the actual pose-aware decoder, which should be the next Priority 3 script
  once these covariates are exported.
