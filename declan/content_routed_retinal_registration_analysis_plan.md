# Content-Routed Retinal-Registration Analysis Plan

## Purpose

This is the canonical integrated plan for testing whether recorded V1 responses
reflect content-routed compact reafferent geometry.

It merges:

- `recorded_retinal_registration_contrastive_analysis.md`
- `Analysis Plan- Testing Content-Routed Compact Reafferent Geometry in V1.md`

The key correction is:

```text
U_trans is a shared transformation channel, not a shared pose coordinate system.
```

The next analysis should not ask whether a generic decoder can recover absolute
eye position. It should ask whether the correct image-specific retinal
registration chart explains recorded response differences better than wrong
charts, gain-only structure, random geometry, or unit-shuffled geometry.

## Core Model

For an image or stimulus-history object `I`, retinal pose `tau`, and recorded
response `y`:

```text
y ~ p(y | I, tau)
```

The fitted twin provides a mean-response surface:

```text
mu_I(tau)
```

Locally:

```text
mu_I(tau + Delta tau) ~= mu_I(tau) + J(I) Delta tau
```

Across many images/histories:

```text
J(I) ~= U_trans A(I)
```

where:

- `U_trans` is the shared compact translation channel.
- `A(I)` is the image/history-dependent routing matrix.
- the channel can be image-general while the coordinate meaning inside the
  channel remains image-specific.

## Current Evidence

### Strong Positive Structural Evidence

The compact geometry and covariance closure are the structural spine:

- local retinal translation tangents are image/history-specific,
- the pooled tangent family is compact,
- the compact tangent basis generalizes across image identity,
- fitted-twin translation covariance predicts a reliable component of recorded
  FEM covariance,
- forcing finite-difference sources through the cross-fit compact tangent basis
  retains covariance closure.

Safe statement:

> FEM-linked covariance is concentrated in an image-general compact translation
> geometry.

### Controlled Nulls That Should Not Be Overinterpreted

The content-blind pose-aware GLM failed:

```text
Y ~ t + e
```

This bounds additive or coarse eye-state regressors. It does not refute
content-routed retinal translation.

The gain-orthogonal structured displacement decoder failed. This bounds one
strict single-trial displacement-readout endpoint, especially under
gain-orthogonal constraints. It does not prove that image-conditioned pose
manifolds contain no recoverable retinal-registration structure.

The forward-twin denoising diagnostic showed compact corrections beating random
and unit-shuffled geometry, but not shuffled-eye compact. This supports a
compact-channel effect but does not establish trial-specific eye-trace
prediction.

### Current A2 Status After Implementation

The A2 chart-swap branch is now implemented and materially constrained by the
result.

What is established:

- The chart-swap machinery can detect chart-aligned retinal-displacement
  structure when it is present: pseudo-spike and split-aware linear chart
  injection controls pass clearly.
- The broad all-unit recorded effect is not robust under the tested split rules.
- The cleanest recorded positive lives in a targeted subset,
  `gain_bottom50`, under the Allen-dominated `drift_trial_disjoint n=5`
  baseline.

What is not established:

- A stable all-session recorded bridge from correct chart to held-out response
  differences.
- A preregistered targeted-subset claim that survives fold/session sensitivity.

The practical implication is that A2 should now be treated as a diagnostic
branch, not a main-claim rescue path, unless a single preregistered rerun
survives with both Allen and Logan included.

Post-patch decoder note, 2026-06-16:

- The relative-displacement decoder has now been audited and patched so that
  matched contexts are image-aware by default (`image_time_bin`), the
  eye-label-shuffle null must pass explicitly, and the `target_pc1` projection
  is derived from fold-train tangent covariance with session-target fallback.
- The refreshed six-session production run remains `diagnostic`, not promoted.
  It shows positive matched-context signal under weaker projections (`none`,
  `global_rate`), but the compact effect shrinks sharply under
  `target_pc1` and `global_rate+target_pc1`.
- Working interpretation: recorded responses carry some same-image
  displacement-related information, but the present decoder does not isolate a
  compact-specific content-routed bridge beyond broader low-dimensional response
  structure.

## Claim Discipline

### Safe Claim

Fixational eye movements introduce a reafferent component into V1 activity that
is compact and image-general: displacement-induced covariance is concentrated in
a shared low-dimensional subspace whose coordinate meaning is image-specific.

### Stronger Claim If A Preregistered Targeted Rerun Lands

The correct image-dependent translation chart explains recorded response
differences better than wrong charts and gain-only controls. This would show
that compact geometry is not merely a covariance channel, but a content-routed
translation channel.

### Claims To Avoid

Do not claim:

- V1 has a global eye-position code,
- all FEM covariance is explained,
- compact geometry proves downstream denoising,
- real FEM trajectories are optimal,
- raw denoising or raw decoding accuracy is sufficient evidence,
- failure of simple GLMs or generic decoders refutes the geometry.

## Analysis Ladder

Run the analysis as a ladder:

```text
A0: twin pseudo-spike positive control
A1: drift-restricted geometry audit
A2: correct-chart vs wrong-chart pairwise alignment
A3: conditional pose-ranking observer
A4: contrastive information lower bound
A5: task-conditioned denoising, only after a task is defined
```

The next concrete deliverable was `A2`, and it has now been implemented. `A0`
and `A1` remain required guardrails. `A3` and `A4` should stay stretch
analyses unless a preregistered targeted A2 rerun survives. `A5` should wait.

## A0: Twin Pseudo-Spike Positive Control

Before scoring recorded data, verify that the candidate construction and noise
model can recover pose in synthetic responses generated from the same twin
surface.

For each sampled context `I_i` and true pose `tau_i`:

1. Compute `mu_I_i(tau_i)`.
2. Generate pseudo-spikes using Poisson or calibrated negative-binomial noise.
3. Generate matched decoy poses or matched wrong charts.
4. Run the same pairwise/ranking analysis planned for recorded spikes.

Success criterion:

```text
true pose or true chart beats matched decoys in pseudo-spikes
performance degrades under wrong-chart and shuffled-chart controls
performance scales sensibly with SNR and pose-difference magnitude
```

If A0 fails, do not interpret a recorded null. Fix candidate construction,
latency, normalization, or noise scoring first.

## A1: Drift-Restricted Geometry Audit

### Goal

Test compact translation geometry in the regime where the local tangent model
should be most valid.

The local model:

```text
mu_I(tau + Delta tau) ~= mu_I(tau) + J(I) Delta tau
```

should be most appropriate during small drift-scale motion, not during
microsaccades, flicks, post-saccadic transients, or large finite shifts.

### Question

Does compact covariance closure and/or chart alignment become stronger,
cleaner, or more interpretable when restricted to drift-only fixation windows?

### Inputs

Use the same matched recorded/twin unit space and fixRSVP samples as the current
covariance closure.

Required per sample:

- session,
- trial,
- time bin,
- image/time identity,
- spike counts,
- eye position,
- eye displacement,
- FEM event labels or microsaccade/flick exclusion windows,
- fitted-twin Jacobians or full-forward translation responses.

### Drift Mask

Define a conservative drift-only mask:

- exclude microsaccade/flick windows,
- exclude a post-event buffer,
- optionally exclude high-speed samples,
- restrict displacement step size or eye velocity to a small range,
- keep only samples with valid eye tracking and sufficient repeats.

Write exact thresholds to:

```text
drift_mask_summary.csv
```

### Outputs

For all samples and drift-only samples, report:

- number of sessions,
- number of samples,
- number of image/time conditions,
- eye displacement distribution,
- compact tangent basis rank,
- compact covariance closure,
- effect over unit-shuffle,
- effect over RF/readout-preserving null if available,
- global-rate and target-PC1 controlled result.

### Interpretation

If drift-only strengthens the effect:

> The compact translation geometry is most apparent in the local retinal-motion
> regime where the tangent approximation should apply.

If drift-only weakens or does not change the effect:

> The compact covariance result is not obviously restricted to drift-only
> samples, but the local tangent interpretation should remain framed with
> caution.

## A2: Correct-Chart Versus Wrong-Chart Pairwise Alignment

This is the key new analysis.

### Goal

Test content-dependent routing directly.

The theory says that the correct image chart `A(I)` should matter. If a response
difference was caused by a retinal shift in image `I`, then the predicted
response direction from `A(I)` should align with the recorded response
difference better than a chart from another image `A(I')`.

### Core Construction

Use pairs of repeats at the same nominal image/time condition.

For two trials `i, j` at the same stimulus-history object `I`:

```text
Delta y_ij = y_i - y_j
Delta tau_ij = tau_i - tau_j
```

The correct chart prediction is:

```text
q_true = J(I) Delta tau_ij
```

Using the compact factorization:

```text
q_true ~= U_trans A(I) Delta tau_ij
```

The wrong-chart prediction is:

```text
q_wrong = U_trans A(I') Delta tau_ij
```

where `I'` is a different image/history object.

Critically, the baseline response `r0(I)` is held fixed. The analysis swaps only
the translation chart, not the full image response. This prevents the result
from being explained by image identity or baseline gain.

### Score

Preferred whitened alignment score:

```text
S(q) = Delta y^T C^{-1} q / sqrt(q^T C^{-1} q)
```

where `C` is a cross-fit residual covariance estimate or a regularized
diagonal/low-rank covariance.

Transparent first-pass scores:

```text
cos(Delta y, q)
Delta y^T q / ||q||
```

Primary contrast:

```text
Delta S = S(q_true) - S(q_wrong)
```

Prediction:

```text
Delta S > 0
```

### Chart Variants

Run three nested variants.

#### Variant A: Full Finite-Difference Chart

```text
q = J(I) Delta tau
```

This tests the full fitted-twin local chart.

#### Variant B: Compact-Projected Chart

```text
q = U_trans U_trans^T J(I) Delta tau
```

This asks whether the chart effect survives restriction to the compact tangent
geometry.

#### Variant C: Gain-Controlled Compact Chart

Project out global-rate and target-PC1 modes before scoring.

This is the skeptic-facing version.

### Controls

Run all primary contrasts under these controls:

1. Wrong chart matched on predictor norm.
2. Wrong chart matched on image statistics.
3. Wrong chart matched on predicted response norm or predicted global rate.
4. Same image identity but wrong time/history, if available.
5. Different image identity with same time bin, if available.
6. Gain-only chart.
7. Random dimension-matched subspace.
8. Unit-shuffled compact basis.
9. RF/readout-preserving compact-basis null.
10. Global-rate and target-PC1 projection.
11. Shuffled eye-pair control.
12. Chart-time shuffle.
13. Sign-flipped or axis-swapped chart as a convention check.

### Wrong-Chart Matching

The primary wrong-chart control should be norm-matched.

For each pair, compute:

```text
||q_true||
```

and select wrong charts with similar:

```text
||q_wrong||
```

This prevents the true chart from winning simply because it predicts a larger
response change.

Also match or bin by predicted gain:

```text
g(I) = <r0(I), 1>
```

or by condition mean population response norm.

### Whitening Covariance

Start with diagonal residual variance for robustness.

Then add:

- shrinkage covariance,
- low-rank residual covariance,
- FEM-corrected covariance if stable.

Whitening covariance must be estimated on training folds only.

### Projection Controls

Run all primary metrics under:

```text
none
global_rate
target_pc1
global_rate+target_pc1
```

The `global_rate+target_pc1` result is the skeptic-facing number.

### Split Design

Use trial-disjoint cross-validation.

Train-fold quantities:

- compact basis,
- whitening covariance,
- gain calibration,
- chart scaling,
- norm-matching bins if learned,
- RF/readout null bins if needed.

Test-fold quantities:

- pairwise recorded response differences,
- true chart scores,
- wrong chart scores,
- controls.

Leakage rules:

- no shared trials across train/test,
- no shared trial pairs across folds,
- image-disjoint chart training reported separately from trial-disjoint if used.

### Success Criterion

The result is supportive if:

```text
correct-chart alignment > wrong-chart alignment
correct-chart alignment > gain-only
correct-chart alignment > random/unit-shuffled/RF-null
effect survives global_rate+target_pc1
effect is stronger or cleaner in drift-only windows
leakage audits pass
```

For a paper-facing targeted positive, add two stricter requirements:

```text
subset rule is defined without using held-out response outcomes
result survives a split rule that includes both Allen and Logan
```

### Interpretation

Positive:

> Recorded response differences are better explained by the correct
> image-specific translation chart than by wrong charts or gain-only structure.
> This supports the view that the compact translation geometry is
> content-routed, not merely global gain or generic low-dimensional covariance.

Current status note:

> The implemented branch does not yet meet this standard. The closest result is
> a gain-bottom compact positive under `drift_trial_disjoint n=5`, but Logan
> drops out there and the effect becomes unstable under split rules that restore
> broader session coverage.

Null:

> The compact geometry predicts recorded covariance, but this pairwise test does
> not show that the correct image-specific routing matrix explains held-out
> response differences at the sample level.

Current status note:

> This is the present default interpretation for the all-unit recorded effect.

True chart beats random/unit-shuffle but not wrong chart:

> The compact translation channel is relevant, but image-specific routing is not
> resolved by this test.

True chart beats wrong chart only before gain controls:

> The effect may be dominated by gain-like or baseline response structure; do
> not promote as content-routing evidence.

Current status note:

> The implemented branch lands closest to a mix of this regime and the null
> regime once fold/session sensitivity is taken seriously.

## A3: Conditional Pose-Ranking Observer

This is a stretch analysis, not the first next step.

### Goal

Test whether the recorded response is more compatible with the true retinal
registration than with matched alternative registrations.

For each sample `i`, construct:

```text
{ tau_i_true, tau_i_1, ..., tau_i_K }
```

Candidate poses should be:

- from the same session,
- from the same block or nearby block,
- from the same time or matched time window,
- from the same fixation regime,
- matched on eye-position magnitude,
- matched on predicted global rate or predicted response norm.

### Score

For each candidate pose:

```text
mu_k = mu(I_i, tau_k)
score(tau_k) = log p(y_i | mu_k)
```

Possible likelihoods:

- diagonal Poisson,
- diagonal Gaussian on residuals,
- covariance-whitened Gaussian,
- projected likelihood in `U_trans`.

If full mean-model mismatch is large, use residualized scoring:

```text
y_res = y_i - PSTH_I
mu_res(tau) = mu_I(tau) - mean_tau mu_I(tau)
```

and score only the pose-dependent residual.

### Metrics

- true-pose rank percentile,
- top-1 accuracy,
- top-k accuracy,
- true-minus-decoy score,
- true-minus-max-decoy score.

### Required Controls

- rate/norm-matched decoys,
- wrong-chart decoys,
- gain-only model,
- random subspace,
- unit-shuffled compact,
- shuffled-eye candidates,
- image/time mismatched candidates,
- twin pseudo-spikes as positive control.

### Interpretation

If true pose ranks above decoys only with the correct image chart:

> Recorded V1 activity contains recoverable retinal-registration information
> when interpreted through the correct content-dependent chart.

If true pose ranks above decoys even with gain-only:

> The effect may reflect global rate matching, not content-routed translation
> geometry.

If no ranking effect is found:

> Trialwise retinal-pose likelihood is not recoverable at current SNR/model
> accuracy, even though covariance geometry remains valid.

## A4: Contrastive Information Lower Bound

Only run this if A3 works.

Use the candidate-ranking task as a contrastive lower bound on pose information
conditioned on image/history:

```text
I(Y; tau | I) >= log(K + 1) - L_contrastive
```

Safe wording:

> Recorded V1 activity contains recoverable information about retinal
> registration when interpreted through the correct image-conditioned chart.

Avoid presenting this as an absolute calibrated information estimate.

## A5: Task-Conditioned Denoising Analysis

This should wait until a specific task direction is defined.

The compact FEM subspace overlaps stimulus-driven variance. Therefore, a
generic "remove `U_trans`" operation is not guaranteed to help. It may remove
both nuisance and signal.

The relevant object is task-relevant signal direction:

```text
d = mu_a - mu_b
```

For a pose-blind observer:

```text
F_blind(d) = d^T (Sigma_int + Sigma_FEM)^-1 d
```

For a pose-aware or FEM-corrected observer:

```text
F_clean(d) = d^T Sigma_int^-1 d
```

The FEM cost is task-specific:

```text
Delta F(d) = F_clean(d) - F_blind(d)
```

Only run this after choosing a task:

- E-optotype orientation,
- high versus low spatial-frequency content,
- natural-image identity,
- coarse image segment decoding,
- local displacement discrimination.

This can support:

> The compact FEM geometry imposes a task-dependent reliability cost. It is
> nuisance for some readouts and useful signal for others.

It should not be used to claim generic denoising benefit.

## Candidate Construction And Filters

### Contexts

Prefer repeated fixRSVP contexts where:

- same nominal image/time/history appears across trials,
- eye traces differ naturally across repeats,
- fixation is stable and drift-only windows can be isolated,
- enough repeats exist to build matched decoys or pairs.

### Poses

Primary target:

```text
relative retinal registration within a known image/history
```

not absolute eye position.

### Latency And History

Do not assume instantaneous eye position is the relevant pose. Sweep:

```text
latency = 20, 40, 60, 80, 100, 120 ms
history = current pose, recent mean pose, short retinal trajectory basis
count window = 8, 16, 25, 50 ms
```

Use the same latency/history convention for recorded responses and twin
rendering.

### Unit And Context Filters

Run broad first, then stratify:

- all matched units,
- high twin-prediction units,
- high FEM-modulation units,
- high RF/STA reliability units,
- high local image-gradient / high spatial-frequency contexts,
- drift-only contexts,
- microsaccade/flick contexts as a separate analysis.

Do not mix drift and microsaccade regimes in the primary local-translation test.

## Main Deliverable

### Proposed Script

```text
declan/compact_retinal_translation_geometry/run_correct_chart_swap_alignment.py
```

### Proposed Summarizer

```text
declan/compact_retinal_translation_geometry/summarize_correct_chart_swap_alignment.py
```

### Output Directory

```text
outputs/compact_retinal_translation_geometry/correct_chart_swap_alignment/
```

### Required Output Files

```text
manifest.json
README.md
session_inventory.csv
drift_mask_summary.csv
pair_inventory.csv
fold_leakage_audit.csv
chart_alignment_pair_metrics.csv
chart_alignment_session_summary.csv
chart_alignment_bootstrap_summary.csv
chart_swap_control_summary.csv
gain_control_summary.csv
compact_vs_full_summary.csv
pseudo_spike_positive_control.csv
latency_history_sweep.csv
figures/chart_swap_alignment_pairs.pdf
figures/chart_swap_summary.pdf
figures/drift_vs_all_summary.pdf
figures/latency_history_sweep.pdf
figures/subspace_controls.pdf
```

### Key Columns For Pair-Level Table

```text
session
subject
fold
trial_i
trial_j
time_i
time_j
condition_id
image_id
drift_mask
delta_eye_x
delta_eye_y
delta_eye_norm
score_true_chart
score_wrong_chart
score_gain_only
score_random
score_unit_shuffle
score_rf_readout_null
score_shuffled_eye
true_minus_wrong
true_minus_gain
true_minus_random
true_minus_unit_shuffle
projection_control
chart_space
basis_k
prediction_norm_true
prediction_norm_wrong
rate_match_bin
```

### Summary Metrics

At session level:

```text
mean_true_minus_wrong
median_true_minus_wrong
bootstrap_CI
n_positive_sessions
sign_test
mean_true_minus_gain
mean_true_minus_random
mean_true_minus_unit_shuffle
mean_true_minus_rf_readout
drift_only_effect
all_sample_effect
```

## Figure Logic If A2 Works

### Panel 1: Conceptual Schematic

Same baseline image, same displacement vector, two possible charts:

```text
correct chart: A(I)
wrong chart:   A(I')
```

Only the routing matrix changes.

### Panel 2: Pairwise Alignment

Bars or paired dots:

```text
correct chart
wrong chart
gain-only
random
unit-shuffled compact
```

Metric:

```text
whitened alignment score
```

or:

```text
true-minus-control score
```

### Panel 3: Drift-Only Enrichment

All samples versus drift-only samples.

### Panel 4: Gain-Controlled Result

Same contrast after `global_rate+target_pc1` projection.

## Decision Table

### Strong Positive

```text
true chart > wrong chart
true chart > gain-only
true chart > random/unit-shuffled/RF-null
survives global_rate+target_pc1
stronger in drift-only windows
```

Interpretation:

> The compact translation geometry is content-routed. The image-specific chart
> matters for explaining recorded response differences.

Status after implementation:

> Not reached.

### Partial Positive

```text
true chart > random/unit-shuffled
true chart ~= wrong chart
```

Interpretation:

> Compact geometry matters, but this test does not resolve image-specific
> routing.

Status after implementation:

> This is the most generous read of the targeted gain-bottom hint, but it is
> not yet stable enough to promote.

### Gain-Dominated

```text
true chart > wrong chart before controls
effect disappears after gain/global-PC controls
gain-only ~= true chart
```

Interpretation:

> The apparent chart effect is likely dominated by gain or baseline response
> structure.

Status after implementation:

> Some early variants lived here before stricter controls and fold diagnostics
> were added.

### Null

```text
true chart does not beat wrong chart or controls
```

Interpretation:

> The compact geometry predicts covariance but does not explain held-out
> pairwise response directions at the current SNR/model precision.

Status after implementation:

> This is the conservative current read for the all-unit recorded branch, with
> the caveat that a targeted subset hint remains diagnostically interesting.

## What Not To Spend More Time On

Do not prioritize:

- more broad chart-swap split/variant sweeps,
- another content-blind pose-aware GLM,
- a 24-session production sweep of the failed GLM ladder,
- more unconstrained eye-position decoders,
- MLP rescue of the gain-orthogonal structured decoder,
- fixed-alpha forward denoising,
- generic "project out `U_trans`" denoising without a task direction,
- claims about real FEM optimality.

Do prioritize, if the branch continues at all:

- per-session effect atlases,
- pair-composition audits for the clean baseline,
- one preregistered targeted subset rerun with fixed fold rule and Allen+Logan
  coverage,
- stopping the branch cleanly if that rerun fails.

## Paper-Facing Interpretation

The analysis program should support this framing:

> Fixational eye movements inject a reafferent component into V1 activity that
> is compact and image-general. The compact channel is shared across images, but
> the coordinate meaning inside that channel depends on the current image. This
> makes the reafferent structure identifiable and potentially discountable,
> while making exact retinal-pose interpretation image-dependent.

The main new analysis tested the second sentence directly:

> Does the correct image-dependent chart explain recorded response differences
> better than the wrong chart?

That was the sharpest next question. The sharpest next follow-up is narrower:

> Is the gain-bottom positive a real biological subpopulation effect, or a
> fold/session-composition artifact?

If that question cannot be answered positively by one preregistered rerun, this
branch should remain a useful diagnostic constraint rather than a promoted
recorded-data bridge.
