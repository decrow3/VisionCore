# 4C Supplemental: True Joint Eye/Image Estimator

Date: 2026-06-24
Status: implemented continuous no-anchor observer plus next-step work plan

## Purpose

The current promoted Figure 4C continuous result asks whether image-feature
information can be recovered when the measured eye trace is hidden and the
observer must marginalize over latent eye motion. The primary diagnostic is
posterior expected feature recovery, not exact image identity.

This supplement records how we moved from the older finite trajectory-table
observer toward a true joint estimator that infers or marginalizes a continuous
eye trajectory, closer in spirit to the Wu et al. joint reconstruction model,
while staying tractable inside the current VisionCore cache structure.

The immediate goal is not full pixel reconstruction. The first goal is a
continuous latent-eye estimator for the existing 4C feature-posterior endpoint.

That first implementation now exists and has a verified full-cache artifact.
The promoted strict no-anchor continuous observer now uses a predeclared
scale-specific trajectory prior, reaching emitted posterior feature cosine
`0.9378` with exact image accuracy `0.7083`. The split-heldout promotion gate is
`0.9371` feature cosine at the same image accuracy. The finite-catalog
image-identity observer remains context and a hard stress-test baseline.

## Scientific Readout Boundary

The current endpoint is best described as feature reconstruction quality. It is
not a direct mutual-information or bits estimate.

```text
posterior feature estimate = sum_I p(I | response) feature(I)
feature score              = cosine(posterior feature estimate, true feature)
```

This score asks whether the observer recovers the local image-feature direction
we chose to analyze. A high score can occur even when exact image identity
remains uncertain, because multiple hard-negative images can share similar
local feature structure. Exact image accuracy, posterior true-candidate mass,
and posterior `N_eff` are therefore retained as guardrails, not replacements
for the feature endpoint.

This is close in spirit to Wu et al.'s reconstruction-quality logic, but not
identical in implementation. Wu et al. reconstructed images with a Bayesian
LNBRC likelihood plus dCNN natural-image prior and scored reconstruction
quality primarily with MS-SSIM, with LPIPS as a confirmation metric. They did
not make their main joint-eye claim with feature-vector cosine, nor with an
explicit Shannon-information/bits endpoint. Our feature cosine should therefore
be written as a compact feature-reconstruction quality metric, not as "bits
about the stimulus."

The most honest current claim is:

```text
Measured motion improves recoverable local image-feature structure relative to
the 0x stabilized counterfactual. A strict no-start joint observer can recover
much of that structure when eye position is hidden, and compact/shared response
geometry carries much of the recoverable signal.
```

The newest along-versus-across check adds a caveat to the Figure 4D bridge.
When the promoted strict no-start estimator is split by
`axis_edge_parallel` versus `axis_edge_orthogonal`, the paired all-scale
feature-cosine contrast is only `+0.0011` along-minus-across, and confidence
intervals cross zero at every scale. At 1x the feature cosine is `0.9407` along
versus `0.9366` across, with identical hard image accuracy (`0.7031`). Thus the
older matched-static 4D along-axis advantage should not be read as a strong
property of the strict continuous joint estimator.

The diagnostic also contains known-eye rows. Those are a ceiling/control, not
an axis-prior test: when the measured trace is supplied, the
`axis_edge_parallel` and `axis_edge_orthogonal` labels no longer affect the
trajectory inference, so the paired along-minus-across contrast is exactly
zero. The zero-eye rows are similarly identical across axis labels.

The matched-static feature-posterior setup and the strict continuous estimator
are different scientific objects:

```text
matched-static feature-posterior:
  Uses matched-static response candidates, applies axis-conditioned trajectory
  priors, and scores joint-minus-zero gain in local pyramid feature -MSE. This
  is the source of the positive 4D along-over-across result.

strict continuous joint estimator:
  Uses the promoted hard-negative continuous-joint cache, marginalizes a
  continuous latent eye trace without a start anchor, and scores posterior
  feature cosine plus hard image identity. This is the current 4C estimator.
```

So the clean interpretation is not "along contours always help the joint
decoder." It is: the matched-static feature-posterior branch shows an
along-axis benefit, while the promoted strict continuous observer currently
does not show a robust inherited along-versus-across advantage.

## Current Prototype Status

Implemented analyzer:

```text
declan/backimage_trajectory_observer/analyze_continuous_joint_trajectory.py
```

Implemented cache support:

```text
declan/fixation_statistics_by_stimulus/run_backimage_trajectory_table_observer.py
```

The trajectory-table cache now records trajectory coordinate sidecars for new
runs, and the existing full 4C cache was backfilled with non-destructive
sidecars. This lets the continuous estimator infer a latent trajectory from the
same response-table rows used by the finite catalog observer.

Implemented diagnostic plots:

```text
declan/figure4_active_sensing_atlas/scripts/build_panel_c_continuous_joint_checks.py
```

Generated outputs:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/continuous_joint/
```

Key figures:

![continuous joint overall accuracy](figures/panel_C/diagnostics/continuous_joint/continuous_joint_overall_accuracy.png)

![continuous joint accuracy by scale](figures/panel_C/diagnostics/continuous_joint/continuous_joint_accuracy_by_scale.png)

![continuous trajectory recovery](figures/panel_C/diagnostics/continuous_joint/continuous_joint_trajectory_recovery.png)

![no-anchor trajectory-prior checks](figures/panel_C/diagnostics/continuous_joint/continuous_joint_subset_prior_comparison.png)

![catalog residual anchor smoothing diagnostics](figures/panel_C/diagnostics/continuous_joint/catalog_residual_anchor_smoothing_diagnostics.png)

![continuous joint feature recovery diagnostics](figures/panel_C/diagnostics/continuous_joint/continuous_joint_feature_recovery.png)

![continuous joint endpoint metric comparison](figures/panel_C/diagnostics/continuous_joint/continuous_joint_endpoint_metric_comparison.png)

![continuous joint posterior temperature sweep](figures/panel_C/diagnostics/continuous_joint/continuous_joint_feature_temperature_sweep.png)

![trial-disjoint posterior temperature calibration](figures/panel_C/diagnostics/continuous_joint/continuous_joint_feature_temperature_cv.png)

![quadratic temperature calibration by scale and axis prior](figures/panel_C/diagnostics/continuous_joint/continuous_joint_quadratic_temperature_slice_cv.png)

![quadratic basis bottleneck comparison](figures/panel_C/diagnostics/continuous_joint/continuous_joint_quadratic_basis_bottleneck_comparison.png)

![trial-disjoint quadratic encoder selection](figures/panel_C/diagnostics/continuous_joint/continuous_joint_quadratic_encoder_selection_cv.png)

![trial-disjoint quadratic encoder axis selection](figures/panel_C/diagnostics/continuous_joint/continuous_joint_quadratic_encoder_axis_selection_cv.png)

![quadratic axis-interleaved basis smoke](figures/panel_C/diagnostics/continuous_joint/continuous_joint_quadratic_axis_interleaved_basis_smoke.png)

![quadratic encoder default-score stability](figures/panel_C/diagnostics/continuous_joint/continuous_joint_quadratic_encoder_default_stability.png)

![quadratic 0.5x ridge probe](figures/panel_C/diagnostics/continuous_joint/continuous_joint_quadratic_scale0p5_ridge_probe.png)

![quadratic 1.0x k15 probe](figures/panel_C/diagnostics/continuous_joint/continuous_joint_quadratic_scale1_k15_probe.png)

![quadratic 2.0x parallel k15 probe](figures/panel_C/diagnostics/continuous_joint/continuous_joint_quadratic_scale2_parallel_k15_probe.png)

![continuous joint along-vs-across trace diagnostic](figures/panel_C/diagnostics/continuous_joint/continuous_joint_axis_trace_diagnostic.png)

![quadratic optimizer feature attribution](figures/panel_C/diagnostics/continuous_joint/continuous_joint_quadratic_optimizer_feature_attribution.png)

![quadratic observation geometry feature attribution](figures/panel_C/diagnostics/continuous_joint/continuous_joint_quadratic_geometry_feature_attribution.png)

![quadratic geometry-conditioned calibration](figures/panel_C/diagnostics/continuous_joint/continuous_joint_quadratic_geometry_temperature_cv.png)

![lagged local observation diagnostic](figures/panel_C/diagnostics/continuous_joint/continuous_joint_lagged_observation_diagnostic_full.png)

![constrained temporal-filter diagnostic](figures/panel_C/diagnostics/continuous_joint/continuous_joint_temporal_filter_observation_diagnostic_full.png)

![polynomial local observation diagnostic](figures/panel_C/diagnostics/continuous_joint/continuous_joint_polynomial_observation_diagnostic_full_ridge1e2.png)

![quadratic no-anchor feature diagnostic](figures/panel_C/diagnostics/continuous_joint/continuous_joint_quadratic_feature_diagnostic_full.png)

## Current Result

There are now two useful continuous-estimator readouts. The primary development
readout should be feature recovery: posterior-weighted cosine to the true local
image feature vector. The hard image-ID task is still useful, but it is a
stress test. It turns a graded posterior into an all-or-nothing MAP decision
among deliberately hard negatives, so it can hide real improvement when the
posterior moves toward the right feature but not all the way to the exact
candidate ID.

The practical selection rule is therefore: choose model changes by heldout
posterior-weighted feature cosine, report image accuracy as the sharper
"became decisive" endpoint, and use trajectory RMSE, optimizer convergence,
and local observation geometry as mechanism diagnostics rather than primary
promotion criteria.

The first full-cache no-anchor feature result was the quadratic compact
observation diagnostic. It keeps the latent eye path two-dimensional, fits an
origin-constrained local quadratic response map in the compact basis, profiles
the latent path without choosing an anchor trajectory, and scores the recovered
path through the Poisson expected-count likelihood. Its uncalibrated endpoint
established that feature recovery gives a clearer progress signal than exact
image identity:

```text
known-eye feature cosine:              0.959
finite catalog joint feature cosine:   0.927
best single catalog feature cosine:    0.927
quadratic no-anchor feature cosine:    0.910
zero-eye feature cosine:               0.826
```

I added a focused representation diagnostic for the related but distinct
question: does the V1 twin population represent the image-feature target better
with measured motion than with the 0x stabilized counterfactual? This uses the
feature compact-mechanism cache and compares matched rows:

```text
0x stabilized:        zero_static
1x motion, eye known: known_eye
1x motion, eye hidden: full_exact
```

At the 1x scale, the oracle known-eye comparison is strongly above the 0x
stabilized counterfactual:

```text
0x stabilized feature cosine:        0.6678
1x motion, eye hidden feature cosine: 0.8721
1x motion, eye known feature cosine:  0.9358

oracle 1x gain over 0x:              0.2680
hidden-eye 1x gain over 0x:          0.2043
latent-eye penalty:                  0.0637
```

This answers the representation question more directly than the no-start joint
observer: with the eye trace supplied, the moving 1x response carries more
recoverable local image-feature information than the stabilized 0x response.
The hidden-eye joint decoder then asks how much of that advantage survives when
eye position is latent, and it preserves most of the oracle gain in this cache.
The diagnostic builder is:

```text
declan/figure4_active_sensing_atlas/scripts/build_panel_c_motion_vs_stabilized_representation.py
```

Outputs:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/panel_C_motion_vs_stabilized_representation.png
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/panel_C_motion_vs_stabilized_representation_summary.csv
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/panel_C_motion_vs_stabilized_representation_contrasts.csv
```

The current promoted continuous readout adds a scale-conditioned compact
encoder and heldout posterior-temperature calibration. It preserves the same
image-identity MAP choices but recalibrates posterior mass for the feature
readout:

```text
scale-calibrated no-anchor image accuracy:     0.7083
scale-calibrated no-anchor feature cosine:     0.9358
finite catalog joint feature cosine context:   0.927
```

The best current encoder-improvement lead is a guarded affine quadratic
observation model. The unguarded affine model helped the default feature score
but failed the heldout calibrated gate; the offset term was too large in the
2.0x slices to trust without a guardrail. Re-running with a heavy intercept
ridge (`quadratic_intercept_ridge_multiplier=1000`) gives a cleaner
feature-primary result:

```text
origin-constrained heldout feature cosine:   0.9343
guarded affine x1000 heldout feature cosine: 0.9374
origin-constrained image accuracy:           0.7083
guarded affine x1000 image accuracy:         0.6927
```

A split-swapped model-selection check chooses `affine_x1000` from each
opposite heldout half and recovers the same aggregate feature cosine `0.9374`.
That makes it a principled candidate under the feature-recovery objective, but
not a full replacement yet: exact image identity still favors the
origin-constrained observer, and the full-cache intercept ablation below shows
that the affine feature lead is intercept-dependent.

The first offset-burden guardrail now passes in the limited sense we can test
from existing QC. The `continuous_joint_affine_offset_guardrail` diagnostic
shows that the x1000 intercept penalty cuts the largest 2.0x median intercept
fractions from roughly `0.33-0.39` to `0.14-0.15` while preserving the feature
lead. That argues against promoting an unregularized affine offset, and it
keeps `affine_x1000` alive as a diagnostic feature-primary candidate. It does
not yet prove the intercept is causal-motion structure rather than a helpful
static offset; the direct full-cache ablation below shows that the feature lead
depends on retaining the intercept.

That ablation gate is now complete. The analyzer option
`--quadratic-affine-intercept-scale 0` fits the same affine model but zeros the
intercept during trajectory profiling and Poisson scoring. On the full cache,
normal x1000 beats the intercept-ablated x1000 control:

```text
origin-constrained heldout feature cosine: 0.9343
normal x1000 heldout feature cosine:       0.9374
intercept=0 x1000 heldout feature cosine:  0.9184
origin-constrained image accuracy:         0.7083
normal x1000 image accuracy:               0.6927
intercept=0 x1000 image accuracy:          0.6380
origin-constrained posterior true mass:    0.5990
normal x1000 posterior true mass:          0.5772
intercept=0 x1000 posterior true mass:     0.5412
```

This does not prove the intercept is safe; it proves the opposite useful fact:
the affine feature advantage is intercept-dependent. The x1000 model remains a
diagnostic feature-primary lead, but the full-cache ablation blocks promotion
unless the intercept can be justified as legitimate motion-conditioned
structure rather than static/candidate offset leakage.

I tried one constrained version of that justification:
`quadratic_prior_mean_poisson_profile`. It fits only deviations around the
heldout-safe trajectory-prior mean and uses the implied prior-mean response as
the offset, rather than fitting a free intercept. On a matched 64-table screen:

```text
origin heldout feature cosine:         0.9637
prior-mean centered feature cosine:    0.9551
free affine x1000 feature cosine:      0.9591
intercept=0 x1000 feature cosine:      0.9402
```

This is a principled negative control: prior-mean centering partly rescues the
intercept ablation, but it is not competitive with the origin-constrained
promoted model on the smoke screen. Do not run it full-cache unless a narrower
subset diagnostic gives a reason to expect a reversal.

The cleaner positive candidate is not affine; it is a trajectory-prior change.
Running the same origin-constrained scale-conditioned quadratic observer with a
soft known-start prior (`tau_0 ~ N(observed_tau_0, 0.0001 I)`) improves the
full-cache heldout feature gate:

```text
inferred-start heldout feature cosine: 0.9343
known-start heldout feature cosine:   0.9361
inferred-start image accuracy:        0.7083
known-start image accuracy:           0.7070
inferred-start posterior true mass:   0.5990
known-start posterior true mass:      0.6029
```

In the split-swapped selector restricted to inferred-start vs known-start,
known-start is selected on both heldout halves. It also increases trajectory
correlation in every scale/axis slice with similar RMSE. This is a principled
feature-primary candidate because it changes the trajectory prior rather than
adding a response offset, but it is not the same strict no-anchor endpoint: it
uses the measured initial eye position.

A calibrated known-start analyzer artifact now exists:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/
continuous_joint_quadratic_poisson_scale_conditioned_knownstart_calibrated_full
```

It uses the promoted fixed scale-temperature schedule
`0.5:0.125,1.0:0.125,2.0:0.5` and reaches emitted feature cosine `0.9374`.
Keep the split-swapped `0.9361` value as the principled selection gate; the
`0.9374` value is the fixed-temperature production readout.

I also fixed and tested the matched-Brownian path for the quadratic profiler:
the full 2x2 Brownian increment covariance now enters the final quadratic
trajectory objective, not just the linear initialization. This makes the
Brownian prior sweep meaningful for the compact quadratic observer. Provenance:

```text
declan/backimage_trajectory_observer/analyze_continuous_joint_trajectory.py
declan/backimage_trajectory_observer/tests/test_continuous_joint_trajectory.py
declan/figure4_active_sensing_atlas/scripts/run_panel_c_knownstart_brownian_continuous_joint_observer.py
declan/figure4_active_sensing_atlas/scripts/build_panel_c_knownstart_brownian_prior_sweep.py
```

The 64-table smoke sweep over heldout matched-Brownian covariance scale found a
real but non-final lead:

```text
figure: continuous_joint_knownstart_brownian_prior_sweep_smoke64.png
table:  continuous_joint_knownstart_brownian_prior_sweep_smoke64.csv

known-start AR(1) smoke64:       feature cosine 0.9647, image acc 0.7969
Brownian scale 4 smoke64:        feature cosine 0.9681, image acc 0.7969
Brownian scale 8 smoke64:        feature cosine 0.9688, image acc 0.8281
Brownian scale 16 smoke64:       feature cosine 0.9653, image acc 0.8125
```

That justified a full-cache scale-8 check, but the full gate blocked promotion:

```text
known-start AR(1) full:          feature cosine 0.9361, image acc 0.7070
known-start Brownian8 full:      feature cosine 0.9360, image acc 0.6992
strict inferred-start full:      feature cosine 0.9343, image acc 0.7083
```

Interpretation: matched Brownian is a principled trajectory-prior diagnostic and
was worth fixing, but it does not replace the simpler known-start AR(1)
candidate on the full-cache feature gate.

The by-scale full-cache audit showed why the global Brownian candidate failed:
AR(1) is better at 0.5x and 1.0x, while Brownian8 is better only on the hard
2.0x slice. I therefore built a diagnostic scale-specific prior hybrid:

```text
declan/figure4_active_sensing_atlas/scripts/build_panel_c_scale_specific_prior_hybrid.py
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/
continuous_joint_quadratic_poisson_scale_conditioned_knownstart_scale_specific_prior_hybrid_full
```

Rule:

```text
0.5x: known-start AR(1)
1.0x: known-start AR(1)
2.0x: known-start matched-Brownian covariance scale 8
```

I then encoded the same rule as a predeclared analyzer configuration and reran
from source:

```text
declan/figure4_active_sensing_atlas/scripts/run_panel_c_scale_prior_hybrid_continuous_joint_observer.py
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/
continuous_joint_quadratic_poisson_scale_conditioned_knownstart_scale_prior_hybrid_predeclared_full
```

Full-cache heldout gate for the predeclared rerun:

```text
known-start AR(1):           feature cosine 0.9361, image acc 0.7070
known-start Brownian8:       feature cosine 0.9360, image acc 0.6992
scale-specific prior hybrid: feature cosine 0.9367, image acc 0.7070
```

The predeclared source rerun matches the posthoc hybrid exactly. The
split-swapped selector chooses the hybrid on both halves. This is now the
leading less-strict feature-primary candidate because it improves the weak 2.0x
slice without sacrificing the easier scales or adding an affine response offset.
The strict inferred-start observer remains the no-start endpoint; this candidate
inherits the known-start caveat.

The same scale-specific prior rule also improves the strict inferred-start
endpoint when rerun without the first-sample prior:

```text
declan/figure4_active_sensing_atlas/scripts/run_panel_c_strict_scale_prior_continuous_joint_observer.py
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/
continuous_joint_quadratic_poisson_scale_conditioned_strict_scale_prior_predeclared_full

strict previous inferred-start: feature cosine 0.9343, image acc 0.7083
strict scale-prior rerun:       feature cosine 0.9371, image acc 0.7083
known-start scale-prior rerun:  feature cosine 0.9367, image acc 0.7070
```

This is the cleanest current improvement: it is still strict no-start,
predeclared, and offset-free. It is therefore the promoted strict feature
endpoint. The emitted fixed-temperature artifact reaches feature cosine
`0.9378`.

Verifier/provenance:

```text
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.figure4_active_sensing_atlas.scripts.verify_panel_c_scale_prior_hybrid_continuous_joint_observer --expect-full
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/continuous_joint/continuous_joint_scale_prior_hybrid_observer_manifest.json
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/continuous_joint/continuous_joint_scale_prior_hybrid_predeclared_full_summary.csv
```

Verification/provenance:

```text
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.figure4_active_sensing_atlas.scripts.verify_panel_c_knownstart_continuous_joint_observer --expect-full
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/continuous_joint/continuous_joint_knownstart_observer_manifest.json
```

The best older pure no-anchor continuous image-identity estimator is the
compact-basis `k=10` linear-Poisson profile model with a time-varying compact
observation matrix `A_I(t)` that is Gaussian-smoothed across time (`sigma=1.5`)
and shrunk halfway toward the time-constant fit (`time_shrinkage=0.5`), with a
tuned AR(1) prior (`alpha=0.70`, `process_var=0.01`). Giving the estimator only
the measured initial eye position as a soft prior
(`tau_0 ~ N(observed_tau_0, 0.01 I)`) is still no-anchor joint inference over
the drift path. On the full n=128, scale 0.5/1/2 hard-negative cache at
likelihood scale 1.0:

```text
known-eye accuracy:              1.000
zero-eye accuracy:               0.445
finite catalog joint accuracy:   0.770
best single catalog trajectory:  0.784
continuous joint accuracy:       0.561
```

The known-start prior mostly improves the trajectory estimate rather than image
identification:

```text
unknown-start AR(1) no-anchor:   accuracy 0.560, median RMSE 0.186, median corr 0.027
known-start AR(1) no-anchor:     accuracy 0.561, median RMSE 0.182, median corr 0.108
```

I also checked two practical routes suggested by the failure mode. First, I
tested whether the response basis itself was suppressing one eye axis. Removing
the compact `k=10` response basis and running the same known-start AR(1) profile
in the full unit basis on a 96-table subset did not rescue the model: subset
accuracy fell from 0.750 to 0.719 and median trajectory RMSE worsened from 0.141
to 0.259. I then built an image-disjoint axis-interleaved tangent basis from the
same source as the compact basis. This basis fits separate SVDs to horizontal
and vertical finite-difference tangents and orders the components as
`x1, y1, x2, y2, ...`, so `k=10` allocates five modes to each cardinal
displacement axis. It gave a small subset bump but did not generalize on the
full cache:

```text
pooled compact k=10, subset:             accuracy 0.750, median RMSE 0.141, median corr 0.156
axis-interleaved compact k=10, subset:   accuracy 0.760, median RMSE 0.143, median corr 0.181
pooled compact k=10, full cache:         accuracy 0.561, median RMSE 0.182, median corr 0.108
axis-interleaved compact k=10, full:     accuracy 0.559, median RMSE 0.183, median corr 0.105
full-unit known-start AR(1), subset:     accuracy 0.719, median RMSE 0.259, median corr 0.133
```

The fitted full-unit and axis-interleaved local maps remained nearly rank-1, so
the problem is not simply that the original pooled compact basis discarded an
otherwise available second displacement axis.

Second, I added a no-anchor empirical Gaussian trajectory prior. This estimates
the full path mean/covariance from the candidate's trajectory ensemble, excluding
the held-out true trace when available, then solves the same local linear profile
objective. This is a principled version of "use prior trajectory statistics"
without selecting one anchor trace. On the same 96-table subset it improved
trajectory RMSE but hurt image identification:

```text
known-start AR(1), k=10:           accuracy 0.750, median RMSE 0.141, median corr 0.156
axis-interleaved AR(1), k=10:      accuracy 0.760, median RMSE 0.143, median corr 0.181
full-unit known-start AR(1):       accuracy 0.719, median RMSE 0.259, median corr 0.133
catalog Gaussian prior, k=10:      accuracy 0.677, median RMSE 0.094, median corr 0.128
catalog Gaussian coarse-to-fine:   accuracy 0.729, median RMSE 0.168, median corr 0.170
```

This is informative even though it is not a new headline. The Gaussian prior
recovers a more plausible trace by using the trajectory ensemble covariance, but
that smoother trace is less useful for image discrimination under the current
local response likelihood. The coarse-to-fine hybrid partially restores image
accuracy but still does not beat the simple known-start AR(1) profile. The
working diagnosis is therefore: trajectory priors can regularize the under-
observed latent state, but the local neural residual still does not provide a
well-conditioned 2D image-specific likelihood.

I also tested a stricter no-anchor coarse-to-fine variant that uses only generic
low-frequency DCT temporal modes as the coarse trajectory, followed by AR(1) or
matched-Brownian refinement. This avoids selecting or initializing from any
finite catalog trace. On a 96-table subset the best matched-Brownian DCT setting
reached 0.771, but it did not generalize on the full cache:

```text
pure AR(1) no-anchor:                  accuracy 0.560, median RMSE 0.186
known-start AR(1) no-anchor:           accuracy 0.561, median RMSE 0.182
known-start residual CTF DCT no-anchor accuracy 0.562, median RMSE 0.200
matched-Brownian no-anchor:            accuracy 0.547, median RMSE 0.178
DCT coarse-to-fine AR(1), no-anchor:   accuracy 0.560, median RMSE 0.208
DCT coarse-to-fine Brownian, no-anchor accuracy 0.557, median RMSE 0.224
finite catalog joint reference:        accuracy 0.770
```

The fixed residual coarse-to-fine implementation now does what was intended:
fit a low-frequency DCT path first, subtract that response contribution, then
fit a fine AR(1) residual path. This gives a tiny image-accuracy increase over
plain known-start AR(1), but with worse median trajectory RMSE. So for the
no-anchor version, the most conservative full-cache model remains the compact
`k=10` AR(1) linear-Poisson profile with a soft known-start prior. The residual
DCT coarse-to-fine result is useful as a diagnostic: the full-cache failure is
not just caused by needing a smoother initialization. Known start helps the
gauge/trace estimate, but it does not close the image-decoding gap to the finite
trajectory catalog.

The current best explanation is that the local eye-response encoder is nearly
rank-1 in the compact response basis. For each image candidate, the fitted local
map is `z_t ~= A_I(t) tau_t`. On the active no-anchor known-start run, the
singular-value diagnostics for `A_I(t)` are:

```text
scale  median s1  median s2   median s2/s1  median log10 condition  rank1 frac s2/s1<0.2
0.5x   0.083      0.000043    0.00046       3.34                    0.99998
1.0x   0.062      0.000067    0.00100       3.00                    0.99990
2.0x   0.051      0.000102    0.00191       2.72                    0.99968
```

This means the compact tangent model usually carries one dominant displacement
axis and almost no independent sensitivity to the orthogonal axis. That makes
2D no-anchor trajectory inference ill-conditioned before the trajectory prior
even enters. This is why known-start and coarse-to-fine improve trace
correlation only modestly: they regularize an under-observed latent state. The
next principled improvement should therefore target the response model, either
by using a compact basis that preserves both eye-displacement axes or by fitting
a nonlinear/piecewise response manifold rather than a single zero-eye tangent.

The stronger diagnostic model is a catalog-residual hybrid. For each candidate
image it starts from every finite catalog response movie, infers a compact
continuous residual trajectory around that catalog anchor, reconstructs the
response, and scores the candidate from the strongest anchors. The current
headline uses a top-2 log-mean over anchor scores, then shrinks that score 19%
toward the all-anchor marginal. The all-anchor, unshrunk top-2, and tuned AR(1)
top-2 scores are retained as diagnostics. This is not a pure replacement for
catalog marginalization, and it is not the active no-anchor answer; it is finite
catalog support plus continuous residual inference. On the same full cache:

```text
known-eye accuracy:                                  1.000
zero-eye accuracy:                                   0.445
finite catalog joint accuracy:                       0.770
best single catalog trajectory:                      0.784
catalog-residual continuous, all anchors:            0.798
catalog-residual continuous, top-2 anchor log-mean:  0.841
catalog-residual continuous, tuned top-2 log-mean:   0.845
catalog-residual continuous, shrunk top-2 log-mean:  0.850
remaining gap to known-eye ceiling:                  0.150
```

A coarse-to-fine version of the same catalog-residual diagnostic is also implemented.
It first scores heavily smoothed anchor traces, keeps the best coarse basins,
then refines those anchors at the unsmoothed scale before applying the same
top-2/shrink scoring rule. With smoothing schedule `6,0`, keeping 16 coarse
anchors exactly recovers the current full-cache headline while pruning more
aggressive keep-4/keep-8 settings lose only a few trials:

```text
unpruned shrunk top-2 catalog residual:      accuracy 0.850, median RMSE 0.074
coarse-to-fine keep-4, schedule 6,0:         accuracy 0.842, median RMSE 0.074
coarse-to-fine keep-8, schedule 6,0:         accuracy 0.845, median RMSE 0.074
coarse-to-fine keep-16, schedule 6,0:        accuracy 0.850, median RMSE 0.074
single-stage smoothed anchors, sigma 6:      accuracy 0.844, median RMSE 0.070
```

I then checked what the smoothed anchor is actually doing. The single-stage
`sigma=6` anchor smoothing run changes the predicted image on only 8 of 768
tables and changes the true candidate's best anchor on only 10 of 768 tables.
It improves trajectory RMSE on 89.8% of tables, but the net image accuracy drops
slightly from 0.850 to 0.844. On the 96-table smoothing sweep, accuracy is flat
at 0.906 from `sigma=0` through `sigma=6`, while median RMSE improves from
0.0528 to 0.0509. The changed image decisions are almost all near-zero-margin
ties; smoothing usually keeps the same anchor but slightly changes the residual
score.

This supports a more specific local-minimum interpretation for anchor-based
diagnostics: low-frequency catalog geometry is useful for basin finding and
trace regularization, but it is not the source of the image-decision gain. The
best image score still wants the final unsmoothed residual scoring stage. Since
the current work should avoid anchors, these catalog-residual numbers should be
treated as an upper diagnostic for what local residual correction can do, not as
the model to promote.

I also added a feature-recovery version of the continuous-joint diagnostic:

```text
declan/figure4_active_sensing_atlas/scripts/build_panel_c_continuous_joint_feature_recovery.py
```

This is the better endpoint for deciding whether the continuous estimator is
making graded progress. Image ID is intentionally all-or-nothing; feature
recovery asks whether the posterior over candidate images recovers the true
candidate's local image feature vector by cosine. In this cache, feature
recovery is the main model-selection metric; image accuracy is a secondary
readout for whether the posterior has become sharp enough to choose the exact
hard-negative identity. The diagnostic uses the same
`pyramid_local_field` feature target used by the promoted Panel C feature
posterior machinery, then computes both posterior-weighted feature cosine and
MAP-feature cosine from each continuous-joint candidate score vector.

The same script now writes an explicit endpoint-comparison artifact:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/continuous_joint/continuous_joint_endpoint_metric_comparison.csv
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/continuous_joint/continuous_joint_endpoint_metric_comparison.png
```

That table ranks continuous-joint variants by posterior feature cosine while
keeping image-ID rank beside it. This makes the selection policy concrete:
feature recovery is the development endpoint, and image accuracy is the hard
MAP identity endpoint. In the current full-cache table, the scale-calibrated
quadratic model is the best no-anchor feature-recovery readout:

```text
run                                      feature rank  image rank  image acc  feature cosine  delta vs finite catalog
quadratic scale-calibrated no-anchor      1            6           0.708      0.9358          +0.0093
catalog residual smoothed anchor          2            3           0.844      0.9301          +0.0035
catalog residual top-2 shrink             4            1           0.850      0.9300          +0.0034
quadratic scale-conditioned no-anchor     6            6           0.708      0.9108          -0.0158
quadratic scale-conditioned iter160       7            5           0.710      0.9107          -0.0158
quadratic no-anchor                       8            7           0.691      0.9098          -0.0168
AR(1) no-anchor                          10            9           0.561      0.8712          -0.0553
```

So the feature-cosine endpoint supports a conservative split decision: keep the
native scale-conditioned quadratic compact encoder, add scale-specific posterior
calibration for the feature-recovery readout, do not promote the longer iter160
optimizer, and treat catalog-residual gains as anchor-assisted diagnostics
rather than no-anchor encoder improvements.

The feature-cosine readout softens the endpoint, but it does not reverse the
main conclusion. The no-anchor estimator is above zero-eye in feature space,
yet still well below the finite trajectory catalog. The catalog-residual runs
sit only slightly above the finite best-trajectory feature cosine, even when
their image-ID accuracy improves substantially:

```text
observer/run                         image acc  posterior feature cosine  MAP feature cosine
zero-eye baseline                    0.445      0.826                     0.790
finite joint catalog                 0.770      0.927                     0.946
best single catalog trajectory       0.784      0.927                     0.949
known-eye ceiling                    1.000      0.959                     1.000

no-anchor AR(1) continuous           0.561      0.871                     0.853
no-anchor residual CTF continuous    0.562      0.870                     0.853
no-anchor Brownian CTF continuous    0.557      0.871                     0.852

catalog-residual all anchors         0.798      0.930                     0.957
catalog-residual top-2 shrink        0.850      0.930                     0.969
catalog-residual CTF keep-8          0.845      0.930                     0.969
catalog-residual smoothed anchor     0.844      0.930                     0.969
```

Interpretation: feature recovery is indeed less all-or-nothing and should be
the preferred development metric for the no-anchor estimator. But the current
no-anchor local-linear model is not merely making near-miss image choices; its
posterior feature estimate is also substantially weaker than the finite catalog
observer. Conversely, the catalog-residual variants mostly improve the discrete
decision among already feature-similar hard negatives: image accuracy moves
from 0.798 to 0.850, while posterior feature cosine stays around 0.930.

The current candidate-decision artifact makes this the explicit policy:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/continuous_joint/continuous_joint_candidate_decision_table.csv
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/continuous_joint/continuous_joint_candidate_decision_table.png
```

Use heldout posterior-weighted feature cosine as the primary development and
model-selection task. Exact image ID should remain the harder MAP identity
readout, useful as a secondary check once a candidate already improves feature
recovery. In the current full-cache gate, this separates the candidates cleanly:

```text
candidate                    status                    feature cosine  image acc
strict scale-prior            promoted strict endpoint   0.9371          0.7083
strict inferred-start old     superseded strict          0.9343          0.7083
known-start candidate         less-strict candidate      0.9361          0.7070
known-start scale-prior       less-strict feature lead   0.9367          0.7070
affine x1000                  diagnostic, blocked        0.9374          0.6927
affine x1000 intercept=0      failed control             0.9184          0.6380
```

So yes: for the continuous estimator, feature recovery is the right thing to
optimize first. Image ID is too all-or-nothing for early development, and can
make small but real posterior improvements look like failures. The caveat is
that cosine alone is not sufficient for promotion: the affine x1000 variant
wins the feature gate but fails the intercept-ablation guardrail, so mechanism
controls still decide whether a feature gain is legitimate.

I then tested whether this is just posterior calibration by sweeping a
posthoc temperature on the continuous-joint candidate scores:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/continuous_joint/continuous_joint_feature_temperature_summary.csv
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/continuous_joint/continuous_joint_feature_temperature_best.csv
```

The answer is no for the no-anchor models. Their feature cosine is almost flat
around the default temperature, so score calibration alone cannot close the
gap:

```text
run                              best temp  default cosine  best cosine
no-anchor AR(1)                  2.0        0.87119         0.87125
no-anchor residual CTF           2.0        0.87001         0.87001
no-anchor Brownian CTF           1.0        0.87114         0.87114
catalog-residual top-2 shrink    0.125      0.92995         0.97253
```

This is a useful negative result. The no-anchor score vectors are not merely
too sharp or too soft; their candidate ordering/features are the limiting
factor. The catalog-residual score vectors, by contrast, benefit strongly from
sharpening because their MAP candidate is already feature-near the truth. For
improving the no-anchor joint encoder, the next lever should therefore be the
observation model or latent geometry, not only likelihood-temperature tuning.

I also folded the older full-cache pure-continuous runs into the same feature
diagnostic. This checks whether an already-computed encoder variant was hidden
by the image-ID endpoint. The best pure-continuous feature cosine is the older
smoothed time-varying `A_I(t)` Poisson profile, but it is effectively tied with
the current no-anchor AR(1) result and does not improve image accuracy:

```text
run                              image acc  posterior feature cosine
Poisson k=10 smooth A(t)         0.560      0.87129
no-anchor AR(1)                  0.561      0.87119
no-anchor Brownian CTF           0.557      0.87114
no-anchor residual CTF           0.562      0.87001
Poisson k=10 Brownian            0.547      0.86594
Poisson k=10 A(t)                0.540      0.86394
Poisson k=20 A(t)                0.535      0.86219
Poisson k=10 time-constant       0.497      0.84248
```

So there is no existing full-cache pure-continuous encoder variant that closes
the feature-recovery gap. The best current pure-continuous family remains a
plateau around cosine `0.871`, below the finite catalog at `0.927` and known
eye at `0.959`.

I next tested a more mechanistic observation-model idea: maybe the compact
response residual is not well described by instantaneous eye position
`tau_t`, because the response depends on recent retinal motion. I added a
trajectory-held-out diagnostic for causal lagged designs:

```text
declan/figure4_active_sensing_atlas/scripts/build_panel_c_lagged_observation_diagnostic.py
```

The diagnostic fits candidate-specific compact response predictors from the
finite trajectory table:

```text
instantaneous: z_t ~= A0 tau_t
lagged:        z_t ~= A0 tau_t + A1 tau_{t-1} + A2 tau_{t-2} + ...
```

The held-out unit is a trajectory fold inside each candidate image, not a
random time bin, so the test asks whether the extra lag structure generalizes
to unseen traces from the same trajectory prior. On the full 768-table cache:

```text
lags          CV R2       train R2    median s2/s1  median s3/s1
0             -0.0007     0.0051      0.006         n/a
0,1           -0.0048     0.0098      0.180         0.002
0,1,2         -0.0117     0.0157      0.164         0.078
0,1,2,4       -0.0260     0.0299      0.279         0.092
0,1,2,4,8     -0.0621     0.0644      0.260         0.120
```

This is another useful negative result. Lagged eye-position features make the
coefficient spectrum look much less rank-1, but they monotonically hurt
trajectory-held-out prediction while increasing training fit. Stronger ridge
regularization on a 96-table subset did not change the conclusion. So a naive
history-augmented linear encoder would likely add unstable degrees of freedom
rather than improve the no-anchor joint estimator. If history enters the model,
it probably needs to enter through a constrained temporal filter or a fitted
retinal encoding model, not by freely adding lagged position regressors.

I then ran the constrained version of that idea:

```text
declan/figure4_active_sensing_atlas/scripts/build_panel_c_temporal_filter_observation_diagnostic.py
```

This keeps the eye regressor two-dimensional and tests only causal filters:
pure delays, causal boxcar averages, and exponential moving averages. This is a
fairer latency/history test because it does not add extra latent dimensions.
On the full cache, the instantaneous model is still best overall:

```text
filter      CV R2      train R2    median s2/s1
instant     -0.0007    0.0051      0.0060
delay1      -0.0018    0.0078      0.0050
delay2      -0.0038    0.0111      0.0042
box2        -0.0013    0.0067      0.0047
box4        -0.0033    0.0102      0.0035
ema0.25     -0.0011    0.0061      0.0049
ema0.50     -0.0021    0.0084      0.0038
ema0.90     -0.0461    0.0598      0.0018
```

There is a tiny scale-specific exception at 2.0x where very weak smoothing
(`ema0.25` or `box2`) beats instantaneous by roughly `8e-5` CV R2, but the
effect is too small to justify changing the estimator. The more reliable
conclusion is that simple latency/filtering is not the missing ingredient.
The constrained filters also reduce the already-small second singular direction,
so they do not solve the rank-1 geometry problem.

The first positive encoder diagnostic is nonlinear local geometry. I added a
trajectory-held-out polynomial observation diagnostic:

```text
declan/figure4_active_sensing_atlas/scripts/build_panel_c_polynomial_observation_diagnostic.py
```

This tests whether the compact response residual is better modeled as a curved
function of the same 2D eye position:

```text
linear:     z_t ~= A [x_t, y_t]
quadratic:  z_t ~= B [x_t, y_t, x_t^2, x_t y_t, y_t^2]
cubic:      adds x^3, x^2 y, x y^2, y^3
```

The origin-constrained quadratic/cubic models are physically cleaner because
zero displacement still predicts zero residual. Affine variants are included
only as diagnostics for possible static response offsets, trajectory-prior mean
effects, or coordinate/zero-reference mismatch. With ridge `0.01`, the
full-cache trajectory-held-out result is:

```text
model              CV R2     train R2
linear             -0.0006   0.0051
quadratic           0.2040   0.2968
cubic               0.2114   0.3184
affine linear       0.6022   0.6372
affine quadratic    0.6434   0.6876
```

By scale, the origin-constrained quadratic model is positive everywhere:

```text
scale   linear CV R2   quadratic CV R2   cubic CV R2
0.5x    -0.0014        0.2189             0.2192
1.0x    -0.0010        0.2239             0.2292
2.0x     0.0005        0.1693             0.1857
```

This is the strongest evidence so far for how to improve the no-anchor joint
encoder. The failure is not just score calibration, temporal lag, or trajectory
prior selection; the current zero-eye tangent is too locally linear. A
quadratic 2D response manifold preserves the latent eye state and substantially
improves held-out compact response prediction. The next principled estimator
step should be a nonlinear profile observer over the same 2D AR(1) path, using
the origin-constrained quadratic map first. Cubic terms improve only slightly
over quadratic after ridge stabilization, so quadratic is the better first
implementation target. The large affine result should be investigated
separately before being used in an estimator, because an intercept could absorb
candidate-specific baseline mismatch or prior-mean response curvature rather
than eye-motion encoding.

Practical nonlinear-estimator plan:

```text
1. Fit candidate-specific quadratic compact maps with trajectory-held-out
   catalog rows:
   phi(tau) = [x, y, x^2, x y, y^2].

2. Keep the same AR(1) / matched-Brownian trajectory prior over the 2D path.

3. Replace the closed-form linear AR(1) profile solve with a small nonlinear
   least-squares/profile solve over tau[time, 2]. Use the current linear
   solution as initialization, and optionally compare zero/known-start
   initialization.

4. Reconstruct compact residuals from phi(tau_hat), lift through the compact
   basis, and score image identity with the existing Poisson expected-count
   likelihood.

5. Validate first on the 96-table subset with feature cosine, then full cache.
```

The first feature-cosine implementation of this path is now in:

```text
declan/figure4_active_sensing_atlas/scripts/build_panel_c_quadratic_joint_feature_diagnostic.py
```

This is still a diagnostic implementation, not the final production observer.
It fits a candidate-specific origin-constrained quadratic compact map from
trajectory-held-out catalog rows, then performs a no-anchor nonlinear MAP
profile solve over `tau[time, 2]`. The endpoint is posterior feature cosine,
because that is less all-or-nothing than exact eye-trace recovery and closer to
the scientific object: recovering the retinalized image feature implied by the
unknown eye path.

On the full 768-table cache, with compact basis `k=10`, ridge `0.01`, inferred
initial position, and the warm-started full-quadratic solve:

```text
observer             image acc   feature cosine   MAP feature cosine
zero                 0.445       0.826            0.790
finite joint         0.770       0.927            0.946
best catalog         0.784       0.927            0.949
linear profile       0.336       0.761            0.754
quadratic profile    0.514       0.857            0.844
quadratic Poisson    0.691       0.910            0.917
known eye            1.000       0.959            1.000
```

By scale, the quadratic-Poisson readout remains positive everywhere and is
especially valuable at larger motion scales, where the local linear profile is
weakest:

```text
scale   linear profile   quadratic profile   quadratic Poisson   finite joint   known eye
0.5x    0.832            0.904               0.918               0.927          0.953
1.0x    0.740            0.851               0.910               0.929          0.959
2.0x    0.712            0.816               0.902               0.924          0.967
```

This is now the first full-cache no-anchor result in this sequence that
substantially moves feature recovery beyond the previous pure-continuous
plateau around `0.871`, while staying below the finite-catalog and known-eye
ceilings. Image accuracy also moves in the right direction (`0.691`), but the
feature endpoint is the more appropriate readout because the posterior can be
near the right local feature even when the exact image ID is not MAP.

The earlier 96-table subset was somewhat optimistic, but it correctly predicted
the direction of the full-cache improvement:

```text
observer             image acc   feature cosine   MAP feature cosine
zero                 0.646       0.892            0.869
finite joint         0.865       0.947            0.977
best catalog         0.885       0.944            0.978
linear profile       0.490       0.825            0.825
quadratic profile    0.646       0.906            0.899
quadratic Poisson    0.833       0.937            0.954
known eye            1.000       0.973            1.000
```

This is the first result in this sequence where the nonlinear no-anchor model
meaningfully closes the feature-recovery gap without selecting an anchor trace.
The result is strongest when the profiled quadratic path is scored through the
existing Poisson expected-count likelihood, which is exactly the endpoint we
would want for the real observer.

Caveat: the nonlinear optimizer still hits the iteration cap in some candidate
fits even after warm-starting from the closed-form linear AR(1) path. On the
full cache, final-stage optimizer success is `0.81` with median iterations
`71/80`; failures are most common at `0.5x` scale and least common at `2.0x`.
This is good enough for a diagnostic promotion, but the full-cache result should
not yet be treated as a final ceiling. Immediate engineering next steps are:

```text
1. Keep the warm-started full-quadratic solve as the promoted subset setting.
2. Use continuation/coarse-to-fine solves as QC, not as the default, unless they
   improve feature cosine as well as convergence.
3. Report optimizer convergence and feature cosine together.
4. Add the production quadratic run to the standard feature-recovery plotting
   script once the desired full-cache analyzer output directory is selected.
```

The production analyzer port is now implemented behind:

```text
--continuous-score-mode quadratic_poisson_profile
```

This keeps the existing linear/Kalman/catalog modes unchanged. The analyzer now
fits both the original linear compact map and the origin-constrained quadratic
compact map, warm-starts the quadratic path solve from the linear AR(1) profile,
writes the usual `continuous_joint_feature_posterior.csv`, and records
quadratic map/optimizer QC in `continuous_joint_qc.csv`.

A 96-table production analyzer smoke/subset run matches the standalone
diagnostic image endpoint exactly:

```text
zero accuracy:                  0.646
finite catalog joint accuracy:  0.865
quadratic Poisson accuracy:     0.833
```

The production output directory used for that integration check is:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/
continuous_joint_quadratic_poisson_profile_subset96
```

The full production analyzer run is also complete:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/
continuous_joint_quadratic_poisson_profile_full
```

It has been added to the standard feature-recovery comparison as
`noanchor_quadratic_poisson`. The regenerated
`continuous_joint_feature_recovery_summary.csv` gives:

```text
run                         image acc   posterior feature cosine   MAP feature cosine
noanchor_quadratic_poisson  0.691       0.9098                     0.9167
noanchor_quadratic_scale    0.708       0.9108                     0.9160
noanchor_ar1                0.561       0.8712                     0.8527
finite catalog joint        0.770       0.9265                     0.9461
known eye                   1.000       0.9593                     1.0000
```

The updated `continuous_joint_feature_recovery.png` now includes the quadratic
no-anchor bar and its scale-by-scale curve.

Posterior-temperature calibration gives one more principled improvement. The
image scores are unchanged, but the posterior-weighted feature estimate improves
when the quadratic no-anchor posterior is sharpened:

```text
run                         best temp   default feature cosine   best feature cosine
noanchor_quadratic_poisson  0.25        0.9098                   0.9321
noanchor_quadratic_scale    0.25        0.9108                   0.9325
noanchor_ar1                2.00        0.8712                   0.8713
noanchor_residual_ctf       2.00        0.8700                   0.8700
catalog_residual_top2       0.125       0.9300                   0.9725
```

By scale, the quadratic optimum is sharper at small/intermediate scale and
slightly softer at 2.0x: `0.125` gives `0.958` at 0.5x, `0.25` gives `0.934`
at 1.0x, and `0.5` gives `0.910` at 2.0x. The all-scale temperature `0.25`
therefore makes the production quadratic no-anchor model essentially match the
finite-catalog uncalibrated feature cosine (`0.932` vs `0.927`) while still
remaining a continuous no-anchor estimator. This should be reported as a
posterior-calibrated feature recovery diagnostic, not as a change in image-ID
accuracy.

To avoid choosing temperature on the same tables used for evaluation, I added a
trial-disjoint split-half calibration diagnostic. Temperatures are selected on
one half of trials and evaluated on the other half, then the split is swapped.
This confirms that the quadratic calibration gain is not just all-cache
overfitting:

```text
run                         calibration       default cosine   heldout calibrated cosine
noanchor_quadratic_poisson  global temp       0.9098           0.9321
noanchor_quadratic_poisson  scale-specific    0.9098           0.9332
noanchor_quadratic_scale    global temp       0.9108           0.9325
noanchor_quadratic_scale    scale-specific    0.9108           0.9343
noanchor_ar1                scale-specific    0.8712           0.8835
noanchor_residual_ctf       scale-specific    0.8700           0.8818
catalog_residual_top2       global temp       0.9300           0.9725
```

For the quadratic run, the split-half scale-specific temperatures were stable:
0.5x selected `0.125` in both trial splits, 2.0x selected `0.25`/`0.5`, and
1.0x selected `0.25`/`0.125`. This makes the calibrated quadratic feature
result a principled heldout readout, not merely a decorative posterior
rescaling.

This calibration is now available directly in the production analyzer, rather
than only in the posthoc plotting script:

```text
--continuous-posterior-temperature 1.0
--continuous-posterior-temperature-by-scale 0.5:0.125,1.0:0.125,2.0:0.5
```

The flag only rescales the `continuous_joint` candidate posterior scores written
to `continuous_joint_feature_posterior.csv`. It does not alter the optimized
eye trace, the raw candidate scores, or the MAP image decision. The CSV keeps
`candidate_score_raw` beside the effective `candidate_score`, and records
`posterior_temperature` per observer row. Thus the recommended reporting split
is explicit: raw image accuracy remains `0.7083`, while the heldout
scale-specific feature-cosine readout is `0.9343`. The all-cache scale-specific
optimum uses `0.5` at 2.0x and gives feature cosine `0.9358`; the split-half CV
chooses `0.25` on one half and `0.5` on the other, so the 2.0x temperature
should be reported as the least stable part of the calibration recipe.

The feature-recovery diagnostics preserve this provenance explicitly:
`analyzer_posterior_temperature` is the temperature already emitted by analyzer
rows, while `posterior_temperature` is any additional posthoc temperature used
by the scorer. For the promoted calibrated run, `posterior_temperature` remains
`1.0` and `analyzer_posterior_temperature` carries `0.125,0.125,0.5`.

I validated the analyzer path on a 12-table real-cache smoke run:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/
continuous_joint_quadratic_poisson_scale_conditioned_calibrated_smoke12
```

The smoke confirms the output contract: only `continuous_joint` rows receive
non-unit posterior temperatures, posterior mass sums to 1 within numerical
precision, and scoring the effective `candidate_score` matches scoring
`candidate_score_raw` with the recorded `posterior_temperature` to better than
`3e-14` feature-cosine error. This is a structural validation of the calibrated
posterior path; the 12-table subset is not used as a performance estimate.

I then ran the full 768-table calibrated analyzer with the same native
scale-conditioned encoder:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/
continuous_joint_quadratic_poisson_scale_conditioned_calibrated_full

declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/continuous_joint/
continuous_joint_quadratic_scale_conditioned_calibrated_full_summary.csv
```

This direct analyzer artifact reproduces the all-cache posthoc calibration
number while keeping image accuracy fixed:

```text
scale  temp   image acc  feature cosine  mean true mass
all    mixed  0.7083     0.93584         0.5929
0.5x   0.125  0.8164     0.95802         0.6842
1.0x   0.125  0.7031     0.93867         0.6185
2.0x   0.5    0.6055     0.91082         0.4760
```

This was the strongest principled version before the scale-prior update. It is
now superseded by the strict scale-prior promoted artifact:

```text
continuous_joint_quadratic_poisson_scale_conditioned_strict_scale_prior_predeclared_full

heldout gate: feature cosine 0.9371, image acc 0.7083
emitted artifact: feature cosine 0.9378, image acc 0.7083
trajectory prior: 0.5x AR(1), 1.0x AR(1), 2.0x Brownian8
```

The encoder remains the native scale-conditioned compact quadratic model, while
the posterior readout uses a scale-specific temperature chosen by the feature
recovery diagnostic and the trajectory prior uses a predeclared scale-specific
rule. It should be described as calibrated feature recovery, not as an
image-identification improvement.

The promoted run is now executable through a single wrapper:

```text
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.figure4_active_sensing_atlas.scripts.run_panel_c_promoted_continuous_joint_observer
```

For smoke checks, use `--max-tables` with a separate `--out-dir`. The wrapper
does not introduce a new model; it freezes the selected encoder and posterior
calibration recipe so the full artifact above can be regenerated.

The promoted artifact can be verified with:

```text
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.figure4_active_sensing_atlas.scripts.verify_panel_c_promoted_continuous_joint_observer --expect-full
```

This verifier checks the scale-conditioned basis/ridge recipe, posterior
temperature recipe, posterior normalization, raw-vs-effective score relation,
and the expected full-cache feature cosine/image-accuracy values.

It can also write the machine-readable promotion manifest:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/continuous_joint/continuous_joint_promoted_observer_manifest.json
```

The manifest records the promoted observer slug, output artifact path,
scale-conditioned encoder recipe, posterior calibration recipe, verified
metrics, and verification status for downstream figure/build scripts.

For future encoder candidates, use the reusable heldout calibration audit as
the promotion gate:

```text
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.figure4_active_sensing_atlas.scripts.audit_panel_c_continuous_joint_feature_calibration --run 'candidate_slug|Candidate label|/path/to/run_dir'
```

With no `--run`, it audits the current promoted artifact. The promoted artifact
reproduces the trial-disjoint feature-calibration result:

```text
default feature cosine:          0.911565
heldout scale-calibrated cosine: 0.937074
image accuracy:                  0.708333
selected temps by split:         0.5:0.125,1.0:0.125,2.0:0.25;0.5:0.125,1.0:0.125,2.0:0.5
```

This script is meant to prevent accidental promotion by hard image accuracy
alone. A new encoder should improve heldout posterior-weighted feature cosine
under the same feature target before it replaces the current recipe.

I then tightened this one step further by selecting temperature within each
scale/axis-prior slice on one trial half and evaluating on the other. The lift
survives in every slice:

```text
scale  axis        default cosine  heldout calibrated cosine  delta
0.5x   orthogonal  0.9196          0.9585                     +0.0389
0.5x   parallel    0.9162          0.9575                     +0.0413
1.0x   orthogonal  0.9106          0.9330                     +0.0224
1.0x   parallel    0.9087          0.9335                     +0.0248
2.0x   orthogonal  0.9043          0.9140                     +0.0097
2.0x   parallel    0.8992          0.9034                     +0.0042
```

This makes the calibration result broad rather than slice-specific. It also
identifies the next principled encoder target: the quadratic local observation
model is already close to finite-catalog feature recovery at 0.5x/1.0x after
heldout calibration, while the 2.0x slices, especially the parallel prior,
still carry most of the remaining no-anchor feature gap.

I tested that target directly with a focused compact-basis/ridge sweep on the
2.0x parallel slice. This keeps the same no-anchor quadratic encoder and only
changes the compact response basis dimension and ridge used to fit the local
quadratic response map. On the first 64 filtered tables:

```text
encoder basis/ridge  image acc  feature cosine  optimizer success
k=10, ridge 0.01     0.625      0.8945          0.855
k=20, ridge 0.01     0.656      0.8992          0.695
k=20, ridge 0.1      0.672      0.8993          0.883
k=30, ridge 0.01     0.547      0.8963          0.531
```

The principled candidate is therefore not "more basis blindly"; it is a modest
increase to `k=20` with stronger ridge regularization. On the full 128-table
2.0x parallel slice, this improves the default quadratic-Poisson readout while
keeping optimizer behavior essentially unchanged:

```text
encoder              image acc  default cosine  best-temp cosine  optimizer success
k=10, ridge 0.01     0.578      0.8993          0.9056            0.889
k=20, ridge 0.1      0.617      0.9046          0.9114            0.891
```

This is a real encoder-side improvement on the identified bottleneck, not just
posterior-temperature calibration. It reduces the 2.0x parallel feature gap to
the finite trajectory catalog from `0.0191` to `0.0138` before temperature
calibration.

The all-scale check argues against blindly using the larger basis everywhere.
`k=20, ridge=0.1` improves convergence and the 1.0x/2.0x slices, but gives up
some 0.5x feature recovery. The best principled candidate is therefore
scale-conditioned: retain the original `k=10, ridge=0.01` encoder at 0.5x and
use `k=20, ridge=0.1` at 1.0x/2.0x. This uses the known prior scale only, not
the image identity, true trace, or anchor trajectory:

```text
encoder setting       image acc  default cosine  global-temp cosine  scale-temp cosine
k=10 all scales       0.691      0.9098          0.9323              0.9341
k=20/ridge0.1 all     0.699      0.9100          0.9300              0.9325
scale-conditioned     0.708      0.9108          0.9325              0.9358
```

I reran the stricter reusable heldout audit on the already-computed full-cache
encoder variants. This uses the predeclared scale-specific calibration gate,
not a posthoc best-mode selection:

```text
run                         default cosine  heldout scale-temp cosine  image acc
scale-conditioned            0.910764        0.934322                   0.708333
scale-calibrated artifact    0.910764        0.934322                   0.708333
scale-conditioned iter160    0.910730        0.934226                   0.709635
```

So the longer optimizer still should not be promoted. It slightly improves hard
image accuracy, but it loses a little on the heldout feature-cosine gate.

The same audit on the targeted full-slice probes gives:

```text
slice                       default cosine  heldout scale-temp cosine  image acc
0.5x k10 ridge0.1            0.917451        0.953121                   0.796875
1.0x k15 ridge0.1            0.910261        0.934069                   0.687500
2.0x parallel k15 ridge0.1   0.904501        0.909299                   0.593750
```

The `2.0x` parallel `k=15` probe confirms a local bottleneck improvement, but
it is not by itself a complete all-scale encoder. The trial-disjoint
scale+axis encoder-selection result below remains the fair test for using
axis-prior metadata, and that rule underperforms after posterior calibration.

I then tested the observation-model clue from the polynomial diagnostic. The
trajectory-held-out polynomial fit showed that affine quadratic maps predict
compact residuals much better than origin-constrained maps, but a free
intercept is not automatically safe: it could absorb candidate-specific mean
motion response or static offset rather than a true latent-eye geometry. I
therefore added a separate analyzer score mode,
`quadratic_affine_poisson_profile`, and tested it only on the identified
`2.0x` parallel bottleneck first.

Matched full-slice result, same `k=20, ridge=0.1`, same heldout
scale-specific feature gate:

```text
model                    default cosine  heldout scale-temp cosine  image acc
origin quadratic          0.904574        0.906558                   0.617188
affine quadratic          0.915898        0.921989                   0.625000
```

This is the first observation-model change that clearly improves the full
`2.0x` parallel bottleneck. It is still a lead rather than a promoted encoder:
the intercept must pass all-scale/axis and offset guardrails before it can
replace the origin-constrained production recipe.

The first all-scale/axis split is now complete, and it blocks unguarded affine
promotion:

```text
model                    default cosine  heldout scale-temp cosine  image acc
origin scale-conditioned  0.910764        0.934322                   0.708333
affine scale-conditioned  0.917065        0.933099                   0.669271
```

So the affine map improves the default feature readout, but it loses the
predeclared heldout calibrated feature gate and hurts hard image accuracy. This
is exactly the kind of case the feature-calibration audit was meant to catch:
the model is informative as an observation-geometry diagnostic, but it should
not replace the promoted origin-constrained encoder.

The guarded intercept-ridge sweep changes the feature-primary status. With
`quadratic_intercept_ridge_multiplier=1000`, split-swapped model selection
chooses `affine_x1000` on both heldout halves and yields heldout feature cosine
`0.937395`, above the origin-constrained `0.934322`. Hard image accuracy still
drops (`0.692708` vs `0.708333`), so this is a candidate, not a promoted
replacement.

The static-offset guardrail is also informative. The unguarded full affine run
emits intercept norm fractions in `continuous_joint_qc.csv`; median intercept
fraction grows strongly with scale:

```text
scale/axis              median intercept norm fraction
0.5x orthogonal         0.0155
0.5x parallel           0.0126
1.0x orthogonal         0.1260
1.0x parallel           0.1064
2.0x orthogonal         0.3943
2.0x parallel           0.3332
```

That pattern is plausible for a trajectory-prior mean response at larger
motion scale, but it is large enough to reject the unguarded affine offset as a
promotion candidate. The x1000 guardrail reduces the largest 2.0x median
fractions to `0.1518` and `0.1363` while retaining the feature lead. That is a
positive intercept-burden guardrail. The direct full-cache ablation now shows
that the feature lead depends on retaining the intercept, so this remains a
diagnostic lead rather than a promotion.

The scale-conditioned quadratic no-anchor observer is now also complete as a
native full-cache production analyzer run. It is included in the standard
feature-recovery comparison as `noanchor_quadratic_scale_conditioned`, with
image accuracy `0.708`, default feature cosine `0.9108`, global-temperature
feature cosine `0.9325`, and trial-disjoint scale-specific calibrated feature
cosine `0.9343`.

The native production output also agrees with the standalone posthoc hybrid
used to discover the rule, which is a useful implementation check:

```text
comparison                     image acc  default cosine  scale-temp cosine
native scale-conditioned        0.708333   0.910764       0.935835
standalone hybrid               0.708333   0.910768       0.935834
native minus standalone hybrid  0.000000  -0.000004       0.000001
```

I also added a paired default-score stability check across the three cached
full-cache encoder settings: all-scale `k=10`, all-scale `k=20/ridge=0.1`, and
native scale-conditioned. This keeps posterior temperature fixed at `1.0`, so
it is an encoder-side readout rather than a calibration readout:

```text
paired comparison       feature delta      95% bootstrap CI       image acc delta
k20 - k10               +0.00025          [-0.00162, +0.00202]   +0.0078
scale-cond - k10        +0.00099          [-0.00065, +0.00258]   +0.0169
scale-cond - k20        +0.00074          [-0.00005, +0.00158]   +0.0091
```

This reinforces the conservative interpretation. Scale-conditioning is the
right native encoder setting among the cached candidates, especially because it
recovers the better 0.5x behavior of `k=10` while matching the 1.0x/2.0x
behavior of `k=20/ridge=0.1`. But the paired feature-cosine confidence
intervals are close to zero. The encoder-side gain should be described as
modest and stabilizing, with posterior calibration carrying the larger feature
recovery improvement.

I then checked whether the 0.5x part of the rule could be improved with the
same `k=10` compact basis but stronger quadratic-map ridge. This was run on
the full 0.5x slice through the native analyzer path:

```text
0.5x encoder             image acc  feature cosine
k=10, ridge 0.01         0.8164     0.91793
k=10, ridge 0.10         0.7969     0.91745
paired delta             -0.0195    -0.00048
95% bootstrap CI         [-0.043, 0] [-0.00199, +0.00088]
```

This argues against changing the 0.5x encoder to the stronger ridge setting.
Together with the smaller `k=5` smoke checks, the current `0.5x: k=10,
ridge=0.01` choice remains the principled 0.5x setting.

I also probed the other side of the scale-conditioned rule: whether the 2.0x
parallel slice should use a slightly smaller compact basis than `k=20`. A
64-table smoke grid suggested `k=15, ridge=0.1` might improve feature cosine
over `k=20, ridge=0.1`, but the full 128-table 2.0x parallel validation did
not confirm that:

```text
2.0x parallel encoder     image acc  feature cosine
k=10, ridge 0.01          0.5781     0.89928
k=15, ridge 0.10          0.5938     0.90450
k=20, ridge 0.10          0.6172     0.90457
k15 - k20 paired delta   -0.0234    -0.00007
95% bootstrap CI         [-0.086, +0.039] [-0.00386, +0.00391]
```

So `k=15` is a useful sanity check showing that the improvement over `k=10`
does not require the full `k=20`, but it is not a better promoted setting than
`k=20, ridge=0.1` on the full 2.0x parallel slice.

The same midpoint check at 1.0x gives the same conclusion. A 64-table smoke
grid gave `k=15, ridge=0.1` a tiny feature advantage, but the full 256-table
1.0x validation again favors the current `k=20, ridge=0.1` setting:

```text
1.0x encoder             image acc  feature cosine
k=10, ridge 0.01         0.6719     0.90956
k=15, ridge 0.10         0.6875     0.91026
k=20, ridge 0.10         0.7031     0.91034
k15 - k20 paired delta  -0.0156    -0.00007
95% bootstrap CI        [-0.043, +0.012] [-0.00178, +0.00160]
```

This closes the local basis-size probe around the promoted 1.0x/2.0x setting:
`k=15` is better than the small basis but does not beat the current
`k=20/ridge=0.1` rule on full-slice validation.

Finally, I tested whether the encoder rule should condition on the known
axis-prior family as well as scale. This is principled metadata, but it adds
more degrees of freedom, so I evaluated it with the same trial-disjoint
selection protocol. The result is mixed and not worth promoting:

```text
selection granularity       heldout image acc  heldout feature cosine
scale, encoder only          0.7031             0.91044
scale+axis, encoder only     0.7031             0.91059
scale, encoder+temperature   0.7005             0.93319
scale+axis, encoder+temp     0.6966             0.93163
```

The scale+axis rule finds tiny default-temperature feature gains, but those do
not survive the calibrated endpoint. Since calibrated feature recovery is the
main heldout readout, axis-conditioned encoder selection remains a diagnostic
lead rather than a promoted rule.

I also tested the alternate axis-interleaved compact basis artifact with the
same scale-conditioned basis dimensions and ridge values. This changes the
response basis itself while keeping the inference, scale rule, and scoring
fixed. On a matched 96-table smoke:

```text
basis                     image acc  feature cosine
standard compact           0.8438     0.93756
axis-interleaved compact   0.8333     0.93739
axis - standard delta     -0.0104    -0.00016
```

The axis-interleaved basis gives a tiny 1.0x feature gain in this smoke, but
loses at 0.5x/2.0x and loses overall. This is not worth a full-cache promotion;
the standard image-disjoint compact basis remains the preferred encoder basis.

Native production optimizer QC is more conservative than the standalone hybrid
QC and should be reported from the production output:

```text
scale  basis/ridge       optimizer success  median iter
0.5x   k=10, ridge 0.01  0.729              72
1.0x   k=20, ridge 0.10  0.808              74
2.0x   k=20, ridge 0.10  0.826              74
```

I then joined optimizer status back to feature recovery. Tables where the true
candidate's quadratic profile optimizer reports success have better feature
cosine and image accuracy than tables where it reports failure:

```text
true-candidate optimizer status  image acc  feature cosine
failed                           0.663      0.8995
success                          0.723      0.9144
```

However, this appears to be mostly a difficulty marker rather than a causal
iteration-limit problem. Re-running the same native scale-conditioned observer
with `--quadratic-optimizer-max-iter 160` raises optimizer success to nearly
one in every scale, but barely changes the readouts:

```text
run                         image acc  default cosine  global-temp cosine  scale-temp cosine
scale-conditioned iter80     0.7083     0.9108          0.9325              0.9343
scale-conditioned iter160    0.7096     0.9107          0.9324              0.9342
```

So higher iteration budget is useful as convergence QC, but it is not the next
encoder improvement to promote. The remaining gap is more likely in the
observation geometry or candidate score calibration than in simple optimizer
termination.

To check that possibility, I joined the native scale-conditioned feature
recovery rows to the true-candidate quadratic observation fit geometry. The
feature-cosine endpoint is much less brittle than image ID here. The clearest
mechanistic signal is the residual variance quartile: the lowest-residual
quartile reaches feature cosine `0.9188`, while the upper three quartiles sit
near `0.907` to `0.909`. Image accuracy changes more sharply over the same
split, from `0.870` in the lowest-residual quartile to `0.542` in the highest.
Across scales, simple fit-energy and singular-spectrum correlations with
feature cosine are weak, so geometry helps identify easy/hard cases but does
not by itself explain the whole remaining feature gap.

That geometry signal is nevertheless useful for a very limited calibration
step, but the calibration rule must not use true-candidate geometry. I tested a
trial-disjoint posterior-temperature rule that uses only the known scale and a
candidate-blind residual-geometry quartile, where the residual metric is
aggregated across all candidate fits for the table. The quartile thresholds and
temperatures are selected on one trial split and evaluated on the other. This
is not a new trace prior and it does not use image identity, the true eye
trace, or anchors at evaluation time; it only asks whether response-geometry
diagnostics can choose a better posterior calibration for feature recovery.

```text
calibration rule             heldout image acc  heldout feature cosine
default temperature           0.7083             0.9108
scale-specific temperature    0.7083             0.9343
scale + residual quartile     0.7083             0.9350
training-selected geometry    0.7083             0.9315
```

The fixed residual-quartile rule is a small exploratory gain over scale-only
calibration (`+0.0006`), but a stricter rule-selection diagnostic is more
cautionary. When the training half chooses among scale-only temperature and
several candidate-blind geometry metrics/bin counts using an inner split, it
selects mean quadratic-fit energy terciles on both outer splits. That selected
rule drops to feature cosine `0.9315` on outer heldout, below the scale-only
temperature result. So the robust improvement to promote remains
scale-specific posterior calibration; geometry-conditioned calibration is a
diagnostic lead, not yet a principled replacement.

I also added a stricter trial-disjoint encoder-selection diagnostic. It chooses
between `k=10, ridge=0.01` and `k=20, ridge=0.1` within each scale on one half
of trials, then evaluates on the other half:

```text
selection rule              heldout image acc  heldout feature cosine
encoder only, temp=1        0.703              0.9104
encoder + posterior temp    0.701              0.9332
```

The selected encoders mostly recover the predeclared scale-conditioned rule,
but not perfectly: one split keeps `k=10` at 1.0x for the default-temperature
readout, and one split keeps `k=10` at 2.0x after temperature calibration. This
is a useful guardrail. The scale-conditioned production run is a real
encoder-side improvement for image accuracy and default feature cosine, but
the reliable heldout feature endpoint remains dominated by posterior
calibration; the native predeclared scale-conditioned rule should be presented
as a modest encoder improvement, not as a large independent jump.

That native production path is now implemented in:

```text
declan/backimage_trajectory_observer/analyze_continuous_joint_trajectory.py
```

New analyzer arguments:

```text
--basis-max-dim-by-scale 0.5:10,1.0:20,2.0:20
--ridge-by-scale 0.5:0.01,1.0:0.1,2.0:0.1
```

The overrides are keyed only by the known prior scale. The effective
`basis_max_dim_requested`, `basis_dim`, and `ridge` are written into
`continuous_joint_trials.csv` and included in the summary grouping, so mixed
scale-conditioned runs remain auditable.

A 6-table production smoke run first verified the native scale-conditioned
settings:

```text
prior family             scale  basis dim  ridge
axis_edge_orthogonal     0.5x   10         0.01
axis_edge_parallel       0.5x   10         0.01
axis_edge_orthogonal     1.0x   20         0.10
axis_edge_parallel       1.0x   20         0.10
axis_edge_orthogonal     2.0x   20         0.10
axis_edge_parallel       2.0x   20         0.10
```

Smoke output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/
continuous_joint_quadratic_poisson_scale_conditioned_smoke6
```

The full native production output is:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/
continuous_joint_quadratic_poisson_scale_conditioned_full
```

The native override path is covered by a synthetic analyzer test:

```text
declan/backimage_trajectory_observer/tests/test_continuous_joint_trajectory.py
test_analyzer_applies_scale_conditioned_basis_and_ridge
```

The test writes two cached response tables at different scales and verifies
that the analyzer records the intended effective `basis_max_dim_requested`,
`basis_dim`, and `ridge` in both trial and summary outputs.

Continuation has now been tested as a control. A four-stage schedule
(`quadratic scale 0,0.25,0.5,1`; observation variance scale `4,2,1,1`)
improves reported optimizer success from `0.83` to `0.87`, but slightly lowers
the quadratic-Poisson feature cosine (`0.937` to `0.936`) and lowers the
quadratic-profile feature cosine more (`0.906` to `0.891`). A lighter schedule
(`0.5,1`) behaves similarly. That suggests the current feature gain is not a
spurious failed-optimizer artifact, and that objective smoothing can pull the
path into a more converged but less useful basin.

This is now the most principled improvement path for the true no-anchor joint
encoder.

For the current image-identification endpoint, the practical oracle ceiling is
the known-eye observer: the response model is evaluated with the measured eye
trace available, while image identity is still decoded from the candidate set.
That ceiling is 1.000 in every scale/family cell of this cache. If both the
true image content and the true eye trace are given, image identification itself
is degenerate; the useful version of that check is the known-image/known-eye
likelihood or trajectory sanity check, not an image-ID accuracy.

By motion scale for the active no-anchor AR(1) continuous model, averaging over
the parallel and orthogonal axis priors:

```text
scale  zero-eye  finite joint  continuous joint  median tau RMSE  median tau corr
0.5x   0.609     0.797         0.711             0.062            0.095
1.0x   0.391     0.801         0.543             0.168            0.095
2.0x   0.336     0.711         0.430             0.241            0.128
```

Interpretation:

```text
pure no-anchor continuous joint > zero-eye overall
pure no-anchor continuous joint < finite catalog joint overall
known initial position improves trace correlation, but barely changes image accuracy
DCT coarse-to-fine no-anchor does not improve the full-cache no-anchor result
the compact local eye-response map is almost rank-1, so 2D trace inference is
  poorly conditioned in the current basis
catalog-residual diagnostics show that local residual corrections can help, but
  they rely on finite catalog anchors and should not be the active model here
trajectory correlation remains weak, especially at 2.0x, so image scoring and
  exact eye-trace recovery are still separable
```

So the answer to "is true joint estimation possible/tractable in this repo?" is
yes, in a cache-only form. The answer to "does the current no-anchor true joint
estimator match the finite catalog observer?" is no. The current limitation is
now more specific: pure continuous inference needs a better global observation
model or scoring rule, because neither a matched Brownian prior nor a generic
DCT coarse-to-fine trajectory basis closes the gap on the full cache.

This result replaced the earlier time-constant compact `k=10` headline:

```text
time-constant compact k=10 continuous accuracy:          0.497
time-varying compact k=10 continuous accuracy:           0.540
smoothed/shrunk time-varying compact k=10 accuracy:      0.553
tuned-prior smoothed/shrunk compact k=10 accuracy:       0.560
matched-Brownian no-anchor compact k=10 accuracy:        0.547
catalog-residual compact k=10 all-anchor accuracy:       0.798
catalog-residual compact k=10 top-2 accuracy:            0.841
catalog-residual compact k=10 tuned top-2 accuracy:      0.845
catalog-residual compact k=10 shrunk top-2 accuracy:     0.850
catalog-residual compact k=10 coarse-to-fine keep-16:    0.850
```

The matched-Brownian no-anchor prior estimates a full 2D covariance from the
finite catalog's frame-to-frame increments, then infers one continuous path
without selecting or averaging around any catalog anchor. It is better than the
catalog-mean prior as a trace regularizer, but not as an image scorer:

```text
zero-mean AR(1) no-anchor:       accuracy 0.560, median trace RMSE 0.186
matched-Brownian no-anchor:      accuracy 0.547, median trace RMSE 0.178
```

So the Brownian prior is a useful route for improving continuous trace recovery
without anchors, while the image-discrimination endpoint still prefers the
AR(1) pure model and, much more strongly, the catalog-residual hybrid.

The pure normalized Kalman marginal likelihood is not the right scoring rule
for the current cache. With a compact `k=50` basis, it scored below zero-eye:

```text
Kalman marginal continuous joint accuracy: 0.406
zero-eye accuracy:                         0.445
finite catalog joint accuracy:             0.770
```

The better-performing pure continuous model first infers an AR(1)-regularized
latent trajectory in compact coordinates under a regularized `A_I(t)`,
reconstructs the expected response movie, and then scores image identity with
the existing Poisson-style response likelihood. The catalog-residual model uses
the same compact residual inference, but around each finite catalog response
movie rather than around the zero-eye response. Both keep the estimator tied to
the image-discrimination endpoint rather than rewarding only trajectory-residual
plausibility.

## Current 4C Anchor

The current promoted 4C contracts should remain the anchor:

```text
known-eye > joint-eye > zero-eye
compact-only near full joint
compact-removed near zero-eye
feature-posterior score first
pixel reconstruction later
```

The current result is already interpretable because the finite-table observer
is explicit and auditable. The weakness is that the word `joint` currently
means marginalization over a fixed trajectory catalog, not continuous
trajectory inference. If the catalog under-covers the observed trace, joint-eye
performance can be artificially capped.

## Wu-Style Reference Point

The Wu Nature Communications 2024 repository describes natural-image
reconstruction from retinal ganglion cell spikes, including eye-movement
conditions:

```text
Known-LNBRC-dCNN: known eye movements
Zero-LNBRC-dCNN: zero-eye assumption on moved input
Joint-LNBRC-dCNN: joint estimation of image and eye movements
```

Their full reconstruction path is heavier than the current 4C observer. It
uses fitted retinal encoding likelihoods, reconstruction hyperparameter grid
searches, eye-movement probability weights, GPU reconstruction, and a
half-quadratic splitting reconstruction algorithm.

Practical implication:

```text
Borrow the observer logic first.
Do not immediately port the full pixel-reconstruction machinery.
```

## Phase 1: Cache-Only Continuous Trajectory Estimator

This phase is implemented by:

```text
declan/backimage_trajectory_observer/analyze_continuous_joint_trajectory.py
```

This should reuse existing response-table caches and avoid rerunning the V1
twin. For each candidate image, use a local linear observation model:

```text
y_obs(t) - y_zero(I,t) ≈ J_I(t) tau_t + noise
```

In compact or static-PC coordinates:

```text
z_t = U^T [y_obs(t) - y_zero(I,t)]
z_t ≈ A_I(t) tau_t + eps_t
```

Use an OU / AR(1) trajectory prior:

```text
tau_t = alpha tau_{t-1} + eta_t
eta_t ~ N(0, Q)
```

The first implementation includes a continuous marginal likelihood option with
a Kalman filter:

```text
log p(y_obs | I) = sum_t log p(z_t | z_{<t}, I)
```

The image posterior can then be fed into the existing image-identity and
feature-posterior scoring code.

The implementation also includes the stronger current option:

```text
linear_poisson_profile:
  infer tau_hat(t) under an AR(1) trajectory prior;
  reconstruct lambda_hat[I,t,u] from the compact response model;
  score candidates with the existing expected-count response likelihood.

catalog_residual_profile:
  for each finite catalog trajectory anchor tau_j;
  infer a continuous residual delta_tau_hat(t);
  reconstruct lambda_hat[I,j,t,u] around the cached finite response movie;
  aggregate anchor scores to obtain the candidate image score.

Supported anchor aggregations:

```text
logmeanexp:
  marginal-style average over all finite anchors;
max:
  profile score from the single best anchor;
topk_logmeanexp:
  marginal-style average over the best k anchors.
```

The current headline uses `topk_logmeanexp` with `k=2`, `alpha=0.70`,
`process_var=0.01`, and `catalog_residual_all_anchor_shrinkage=0.19`. On the
full cache this improves image identification from 0.798 for the all-anchor
marginal to 0.850, while preserving the catalog-residual interpretation.
The shrinkage score is a one-run calibration:

```text
score = 0.81 * top2_logmeanexp(anchor_scores)
      + 0.19 * all_anchor_logmeanexp(anchor_scores)
```

The separate AR(1) retune (`alpha=0.85`, `process_var=0.01`) was selected from
a targeted 2.0x parallel sweep and then validated on the full cache; it lifts
that weak condition from 0.727 to 0.750, but the shrunk top-2 score gives the
best overall full-cache image accuracy.

Anchor smoothing diagnostic:

```text
unsmoothed shrunk top-2 anchors: accuracy 0.850, median trace RMSE 0.074
sigma=6 smoothed anchors:        accuracy 0.844, median trace RMSE 0.070
```

This does not mean smoothed anchors are the better image model. Smooth6 changes
only 8/768 image predictions and usually keeps the same true-candidate best
anchor; its reliable effect is lower trajectory RMSE. The stronger lesson is
that low-frequency anchor geometry can identify/prune the right basin, while
the final image score should still be computed after unsmoothed residual
refinement.
```

## Phase 2: Estimate Local Observation Matrices From Existing Tables

For a first smoke test, estimate `A_I(t)` from existing cached trajectory
tables rather than finite-difference rerendering:

```text
Delta[I,tau,t,u] = lambda_full[I,tau,t,u] - lambda_zero[I,t,u]
z[I,tau,t] = U^T Delta[I,tau,t,:]
fit z[I,tau,t] ~ A_I(t) tau[t]
```

This fit must be held out with respect to the evaluated trajectory. The
continuous observer needs the same anti-leakage discipline as the finite
catalog observer's leave-one-out trajectory prior. Otherwise `A_I` can be tuned
on the same `(I, tau)` row it is later scoring, and an apparent
`continuous joint > catalog joint` win can become a circular fitting artifact.

Allowed Phase 2 fitting contracts:

```text
trajectory-held-out:
  fit A_I from all prior trajectories except the observed/evaluated trajectory;
  evaluate the Kalman likelihood on the left-out trace.

image/trajectory cross-fit:
  fit A_I on disjoint images and disjoint trajectories when support allows.

finite-difference Jacobian:
  compute or load J_I at tau=0 and use it as A_I, avoiding regression from the
  same trajectory catalog being scored.
```

Disallowed:

```text
fit A_I on all cached trajectories and evaluate on one of those same traces
without a leave-one-trajectory-out or cross-fit guard.
```

Run this for:

```text
compact basis, k=10
static-PC basis, k=10
random basis
unit-shuffled compact basis
```

The static-PC basis is required, not optional, because the current 4C companion
evidence indicates that compact translation geometry is largely shared with the
static image-response manifold.

The initial implemented version started with a time-constant observation matrix
as the Phase 1/2 simplification:

```text
z_t ≈ A_I tau_t + eps_t
```

The time-varying version has now been implemented and improves the continuous
compact estimator. Smoothing/shrinkage improves image scoring further, and
tuning the AR(1) prior gives a small additional full-cache gain, mostly at 0.5x
and 1.0x. The earlier `alpha=0.92`, `process_var=0.001` smoothed model remains
slightly better at 2.0x and has lower trajectory RMSE, so this is an image-score
headline rather than a trajectory-recovery headline. The next step should tune
regularization and trajectory prior against both image accuracy and trajectory
recovery, not one metric alone.

## Phase 3: Compare Against Existing Catalog Observer

For the same source rows, compare:

```text
zero-eye
finite catalog joint-eye
continuous Kalman joint-eye
known-eye
```

Interpretation:

```text
Kalman ≈ catalog:
  The finite catalog was covering the relevant trajectory support.

Kalman > catalog:
  The current 4C catalog observer was underestimating latent-eye recovery.

Kalman < catalog and nearest catalog entry is far from tau*:
  The local linear-Gaussian assumption is likely failing, or the noise model is
  wrong. Move to EKF/UKF or a small particle filter before interpreting the
  result.

Kalman < catalog and nearest catalog entry is very close to tau*:
  The finite catalog may have been unfairly easy through over-coverage or a
  near-duplicate trajectory. Treat catalog joint-eye as potentially inflated
  rather than treating Kalman as a mechanistic failure.
```

Therefore every Kalman-vs-catalog comparison must include the existing catalog
coverage diagnostics:

```text
nearest trajectory distance to tau*
nearest trajectory rank
whether the exact observed trace was excluded
catalog N_eff / K and posterior entropy
```

Primary output tables:

```text
continuous_joint_trials.csv
continuous_joint_summary.csv
continuous_joint_feature_posterior.csv
continuous_joint_trajectory_recovery.csv
continuous_joint_qc.csv
continuous_joint_report.md
```

## Phase 4: Add Trajectory-Recovery Diagnostics

The finite catalog posterior gives posterior concentration, but not a clean
continuous trajectory estimate. A Kalman smoother returns `tau_hat(t)`, so the
supplement should report trajectory recovery as a co-primary deliverable, not
as a late QC step:

```text
trajectory RMSE
x/y correlation with true trace
path RMS error
posterior/smoother uncertainty
relationship between feature recovery and trajectory recovery
failure cases by image and motion scale
```

This is the key attribution read. It asks whether feature recovery improves
because the observer inferred eye motion better, because the response itself
contained a better sensory signal, or both.

The `tau_hat`-vs-feature-recovery relationship is the scientific payoff of the
continuous estimator. The finite catalog observer cannot cleanly provide this
attribution, so schedule pressure should not demote trajectory recovery below
feature-posterior scoring.

## Phase 5: Only Then Prototype Pixel Reconstruction

Full Wu-style reconstruction should be a later branch, not the first
implementation target.

If the continuous feature-posterior estimator works, prototype pixel
reconstruction with small image sets and local crops. Useful `plenoptic`
building blocks already available in the environment:

```text
plenoptic.synthesize.Metamer:
  optimization scaffold for matching a model representation.

plenoptic.simulate.PortillaSimoncelli:
  differentiable texture-statistics prior; useful as a tractable prior
  prototype, but not equivalent to a modern diffusion/denoiser prior.

plenoptic.metric.ms_ssim, ssim, nlpd:
  reconstruction-quality metrics.

plenoptic.simulate.SteerablePyramidFreq:
  phase-aware and multiscale image representations already used elsewhere in
  this repo.
```

This branch should be scored as reconstruction quality, not as response
information or bits. A natural-image prior contributes information that is not
contained in the neural response, so it belongs with 4C's recovery-quality
logic, not 4B's information-lower-bound logic.

Portilla-Simoncelli is only a scaffold for getting the reconstruction pipeline
running. Because it is a texture-statistics prior, it can produce
texture-faithful but phase- and position-loose reconstructions. It should not
be used as evidence for a phase-completeness claim. A phase-completeness claim
requires a stronger image prior, such as a denoiser or diffusion prior, and a
metric that preserves phase/position sensitivity.

## Completed Smoke Test

The initial smoke test was expanded to the full cache rather than stopping at
16 or 32 images:

```text
source: existing BackImage trajectory-table response caches
n_rows: 768 response tables
candidate_set_mode: hard_negative_structure
motion scales: 0.5x, 1.0x, 2.0x
basis: image-disjoint compact basis, k=5/10/20 plus Kalman k=50
endpoint: image identity for this prototype; feature posterior remains the 4C endpoint
```

Original success criterion:

```text
known-eye > continuous joint-eye > zero-eye
continuous joint-eye is competitive with or better than finite catalog joint-eye
compact and static-PC versions are both reported
trajectory RMSE is finite and interpretable
```

Observed:

```text
known-eye > finite catalog joint-eye > continuous joint-eye > zero-eye overall
continuous joint-eye is not competitive with finite catalog joint-eye
trajectory RMSE is finite but trajectory correlation is weak
```

This is a partial success. The estimator is tractable and beats zero-eye
overall, but it does not yet remove the need for the finite catalog observer.
The next model change should target the observation model before adding a more
expensive image prior or pixel reconstruction stage.

Stop criterion retained for next iterations:

```text
continuous joint-eye fails below zero-eye
trajectory estimates diverge or saturate
linearization residuals grow with scale enough to explain the failure
compact-only success appears without matching static-PC controls
tau_hat collapses toward zero variance while feature recovery sits at zero-eye
  level, suggesting the noise/whitening model ate the displacement signal
catalog joint-eye beats Kalman only when nearest catalog entries are near
  duplicates of tau*, suggesting catalog over-coverage rather than Kalman failure
```

The signal/nuisance-collapse diagnostic is especially important. If the residual
covariance used for `eps_t` is estimated from motion-containing residuals, the
whitening model can suppress the same compact subspace that carries
`A_I tau_t`. This can yield a quiet, stable, useless Kalman filter: flat
trajectory estimates and zero-eye-level feature recovery. That should be read
as a noise-model failure mode before being read as absence of displacement
information.

## Claim Boundaries

Supported if successful:

```text
A continuous latent-eye observer can recover image features from the V1 twin
without knowing the measured eye trace, and this recovery is not merely an
artifact of finite trajectory-catalog coverage.
```

Still not supported by this supplement alone:

```text
The animal computes this estimator.
The compact basis is unique over the static image-response manifold.
Empirical FEMs are optimal over all plausible motion priors.
Pixel reconstruction quality is a bits or information measure.
```

## Recommended Implementation Order

```text
1. Implement cache-only Kalman likelihood for one response table.
2. Add tiny synthetic tests for known/zero/joint ordering.
3. Fit held-out or finite-difference A_I from cached table deltas in compact coordinates.
4. Score image identity, feature posterior, and trajectory recovery for n=16 or n=32.
5. Repeat with static-PC and random bases.
6. Add catalog-coverage and signal/nuisance-collapse diagnostics.
7. Scale to the current 4C n128 feature-posterior cache.
8. Only after that, prototype pixel reconstruction with a plenoptic prior.
```
