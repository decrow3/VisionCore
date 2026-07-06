# Companion: Joint Posterior / Trajectory-Table Observer Model

Date: 2026-06-30
Status: provisional methods/logic companion for Figure 4C; inherited-decoder audit complete with guarded interpretation

## Panel Claim Under Test

```text
Joint eye/image decoding can recover feature information when eye position is
hidden, and the compact translation component is important for that recovery.
```

This is the result Panel 4C is there to show if the evidence supports it. The
claim has two parts: joint image/trajectory inference can recover feature
information when eye position is latent, and compact response structure carries
enough of that recovery that compact removal should damage the endpoint. The
analysis motivations, assumptions, estimator contracts, controls, results, and
caveats below are all there to decide whether that sentence is strong,
qualified, or should be softened.

The current evidence supports that claim, but with an important boundary. It
does not show that the compact translation component is the unique response
structure that makes the observer work. A matched control based on static image
responses is nearly as strong.

The endpoint should also be described carefully. The promoted score is
posterior expected feature recovery, reported as cosine similarity in a chosen
local feature space. That is a reconstruction-quality metric for a selected
feature target, not a Shannon-information or bits estimate. This matches the
spirit of Wu et al.'s reconstruction-quality analysis more than an infomax
analysis: Wu used Bayesian MAP image reconstruction and evaluated image quality
mainly with MS-SSIM, with LPIPS as a confirmation metric.

The current candidate-posterior version has been preserved before starting the
next branch:

```text
declan/figure4_active_sensing_atlas/archive/
4c_candidate_posterior_snapshot_2026_06_24/
```

## 2026-06-30 Inherited-Decoder Audit Gate

The corrected aggregate and local feature-information audits found that several
decoder branches inherited subtle but claim-changing issues: train/test grouping
by image instead of source trial, mismatched static-versus-motion baseline
contracts, non-commensurate metric axes, and stale cache outputs whose point
estimates no longer matched their intervals. The joint decoder/observer branch
must now be checked for the same failure class before any stronger 4C wording is
promoted.

Audit checklist:

```text
1. Split/provenance: verify that calibration, encoder fitting, target-space PCA,
   posterior temperature selection, and final scoring are source-image/source-
   trial disjoint where the claim requires held-out generalization. Any image-
   grouped or candidate-table grouped result should be labeled as optimistic
   provenance context.

2. Baseline contract: verify that known-eye, zero-eye, hidden/joint-eye, 0x
   stabilized, matched-static, compact-only, compact-removed, and static-PC
   controls are scored against the intended baseline. Do not mix a motion
   response baseline with a zero-motion/static response baseline unless the
   row explicitly says so.

3. Metric axis: keep 4C feature recovery/cosine/negative-MSE endpoints separate
   from the 4B diagonal-information-in-bits axis. If a joint-decoder result is
   compared with 4B, recompute or translate the comparison onto a same-axis
   metric rather than mixing posterior feature recovery with information bits.

4. Cache integrity: add the same validator pattern used for the aggregate
   information posthoc: reported point estimates must lie inside their own CIs,
   manifest paths and code versions must be written, and stale finite-cache
   tables must not feed current promoted figures.

5. Response-table leakage: audit whether response tables, trajectory catalogs,
   hard negatives, compact/static-PC bases, and feature targets share source
   images or source trials across training, calibration, and evaluation folds.

6. Interpretation: require a before/after audit note. If the current 4C result
   survives, report it as source-disjoint/provenance-validated. If it weakens,
   keep 4C as a diagnostic observer story rather than a promoted mechanism
   claim.
```

Implementation update: the inherited-decoder gate is now implemented and run.
The reusable audit is:

```text
declan/figure4_active_sensing_atlas/scripts/audit_panel_c_joint_decoder_inherited_contracts.py
```

The current report is:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/continuous_joint/
inherited_decoder_audit_20260630/panel_c_joint_inherited_decoder_audit.csv
```

The audit has 17 PASS and 2 WARN checks, with no FAIL checks. The promoted
posterior math is internally consistent, posterior mass sums to one, all
validated feature-cosine contrast point estimates lie inside their own CIs, and
the supervised continuous feature-embedding / tau-MLP diagnostics are
source-row disjoint. The promoted calibration audit now includes `source_row`
splits and selects best rows from that stricter source-row-heldout gate.

The remaining warnings are provenance notes rather than blockers: a true source
row can appear under multiple trial ids in the n=128 response-table cache, so
source-row CV is stricter than trial-id CV; and one older axis-trace diagnostic
has no CI columns for the cache-integrity validator to inspect. The already
stated static-PC caveat remains the main scientific guardrail.

## Summary

The current 4C claim is no longer only the older matched-static image-identity
rescue. It now asks whether the compact translation-response component is
important for the feature-posterior endpoint used in the panel. The simplifying
assumption it breaks is still the zero-eye approximation: treating the response
to a moving retinal image as if the image stayed fixed on the retina. But the
promoted result adds a mechanism test: compact-only recovery should retain much
of the joint feature recovery, while compact removal should push recovery back
toward the zero-eye curve.

The older matched-static image-identity observer remains important historical
support because it established that marginalizing over possible trajectories
can rescue image information when eye position is latent. The current panel,
however, is supported by the newer feature-posterior compact-only /
compact-removed / compact-addback analysis. That is the analysis that matches
the title.

The newest static-response-PC tests change the mechanistic interpretation. In
plain language: the compact translation component is real and useful, but it
mostly overlaps with the main response patterns the model already uses for
static images. This means the safe claim is not "there is a special compact
translation-only channel." The safer and better claim is "eye movements push V1
responses along a compact part of the normal image-response manifold."

There is now a promoted continuous no-anchor observer for the 4C feature
recovery readout; the finite trajectory-table result remains context and a hard
image-identity guardrail. The continuous observer uses a native
scale-conditioned compact quadratic observation model, infers a continuous
trajectory for each candidate image, and applies scale-specific posterior
calibration for the feature-recovery readout. The verified full-cache artifact
is:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/continuous_joint/
continuous_joint_promoted_observer_manifest.json
```

This continuous observer is the strongest current no-anchor feature-recovery
readout: emitted posterior-weighted feature cosine `0.9378`, with
source-row-heldout promotion gate `0.9371`, at image accuracy `0.7083`.
That should be read as calibrated feature recovery, not as a hard image-ID
improvement. The encoder/prior choice is conservative: keep the native
scale-conditioned compact quadratic model, with a predeclared scale-specific
trajectory prior; do not promote the longer iter160 optimizer or anchor-assisted
catalog-residual variants as no-anchor encoder improvements.

A new continuous feature-embedding reconstruction branch now exists as the
next, more Wu-style step away from finite candidate choice. It uses the same
response-table cache but replaces `sum_I p(I | y) phi(I)` with a linear-Gaussian
posterior over a compact whitened feature embedding `z`:

```text
response_features = A z + noise
z ~ N(0, I)
z_hat = E[z | response]
```

The full-cache artifact is:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/
continuous_feature_embedding/continuous_feature_embedding_reconstruction_manifest.json
```

This branch is not promoted over the candidate-posterior endpoint yet. It is
important because it removes finite candidate selection at test time. The
runner now evaluates a small target-space sweep around a 32D
`pyramid_local_field` embedding and a 20D image-disjoint compact response
basis. All variants are cross-fit by source image; fold-local variants fit the
feature PCA after excluding the held-out source rows. All-scale feature cosine:

```text
feature-space mode              known   hidden  zero-model  0x stabilized
global centered whitened PCA    0.1261  0.1036  0.0471      0.1860
fold centered whitened PCA      0.1203  0.0994  0.0567      0.1896
fold z-scored whitened PCA      0.1505  0.1271  0.0618      0.2330
fold z-scored PCA               0.1372  0.1346  0.1368      0.3218
```

The fold z-scored whitened option is the most useful Gaussian-prior target
space so far: it improves the known-eye and hidden-eye feature readout relative
to the first global PCA baseline while retaining normalized coordinates for
`z ~ N(0, I)`. The unwhitened z-scored option is a scale diagnostic; it raises
the hidden score but also raises the zero-eye and stabilized curves. No option
yet shows moving-response recovery above the 0x stabilized counterfactual in
this compact-basis linear-Gaussian form. That makes the branch a principled
next-step diagnostic rather than the main 4C result. The next method step is to
replace the nuisance-trained hidden-eye decoder with a true joint
optimization/posterior over `(z, tau)`.

The nonlinear upper-bound pass changes that conclusion for the information
question. A Tejas-style ReLU MLP decoder was added to the same runner and fit
only on held-out-source-safe training folds. It uses the preferred
`fold_zscore_whitened_pca` target, the same 20D compact response basis, and
maps compact response movies directly to `z`. The full-cache artifact is:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/
continuous_feature_embedding_mlp_hammer/continuous_feature_embedding_reconstruction_manifest.json
```

All-scale MLP feature cosine:

```text
known eye:              0.2329
hidden eye tau-nuisance: 0.3482
zero-eye model:         0.1824
0x stabilized:          0.2819
```

Paired contrasts:

```text
hidden - zero-eye model:       +0.1657  CI [+0.1422, +0.1912]
hidden motion - 0x stabilized: +0.0662  CI [+0.0450, +0.0887]
known motion - 0x stabilized:  -0.0491  CI [-0.0684, -0.0302]
```

Scale breakdown for hidden-eye MLP is also above the stabilized counterfactual:

```text
0.5x: hidden 0.3481 vs 0x 0.2819
1.0x: hidden 0.3441 vs 0x 0.2819
2.0x: hidden 0.3523 vs 0x 0.2819
```

This should be framed as an information upper-bound result, not a downstream
linear-decodability claim. The odd known-eye < hidden-eye ordering is likely a
training-set-size effect: the hidden/tau-nuisance bank sees many trajectory
draws per candidate and acts as data augmentation, while the known-eye bank has
one measured-trajectory response per candidate. The important result is that a
strong nonlinear decoder can extract feature information from moving compact
responses above both the zero-eye model and the stabilized counterfactual.

A hybrid continuous-tau / nonlinear feature decoder now tests the cleanest
candidate-free version of that idea. The endpoint is still direct recovery of a
compact `pyramid_local_field` feature embedding, but the decoder can receive a
continuous trajectory estimate from the promoted continuous-joint model:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/
continuous_tau_mlp_feature_decoder_residual/
continuous_tau_mlp_feature_decoder_manifest.json
```

This runner trains augmented modes on source-disjoint response-bank rows
`(prior response, trajectory) -> phi(source image)` and tests on held-out
observed responses. It now also includes nested residual eye-trace decoders:
first fit a response-only MLP, then fit an eye-trace correction with a
validation-selected shrinkage that includes zero as a fallback. The MLP decoder
is still the upper-bound readout, not a downstream linear decoder. All-scale
feature cosine:

```text
augmented compact-only hidden readout: 0.3695
augmented true-tau residual readout:   0.3683
augmented true-tau raw readout:        0.3521
augmented tau_hat residual readout:    0.3440
augmented tau_hat raw readout:         0.3140
augmented 0x stabilized readout:       0.3243
augmented known-eye readout:           0.3196
```

Paired contrasts:

```text
augmented compact-only - augmented 0x: +0.0452  CI [+0.0292, +0.0618]
true-tau residual - augmented 0x:      +0.0440  CI [+0.0276, +0.0610]
tau_hat residual - augmented 0x:       +0.0196  CI [+0.0029, +0.0374]
true-tau residual - compact-only:      -0.0012  CI [-0.0183, +0.0154]
```

So the red flag was real but local: raw coordinate concatenation was a bad way
to expose eye position. Once known eye trace is nested as a residual correction,
it beats the fair augmented static baseline and essentially matches the
response-only compact decoder. Estimated `tau_hat` also becomes useful in the
residual form, though it remains below true-tau and compact-only.

The final along/across check for this residual MLP method is negative. The
paired axis rows share the same observed response and true eye trace; only the
axis-conditioned trajectory prior/catalog changes. A cache-only posthoc summary
is here:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/
continuous_tau_mlp_feature_decoder_residual/
continuous_tau_mlp_axis_contrast_README.md
```

All-scale feature-cosine contrast `axis_edge_parallel - axis_edge_orthogonal`:

```text
augmented compact-only:     +0.00000  CI [+0.00000, +0.00000]
true-tau residual:         +0.00000  CI [+0.00000, +0.00000]
tau_hat residual:          +0.00013  CI [-0.00514, +0.00620]
raw tau_hat concatenation: -0.01095  CI [-0.02414, +0.00218]
```

So this method supports the candidate-free moving-versus-static feature
recovery claim, but not an along-contour advantage. The only axis-dependent
quantity in this paired setup is the inferred `tau_hat`, and the residual
decoder makes that axis effect essentially vanish.

That fixed-observation guardrail is not the direct along/across response-movie
test. The direct test now exists as a separate candidate-free MLP diagnostic:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/
direct_axis_mlp_feature_decoder/
direct_axis_mlp_feature_decoder_README.md
```

Here, for every axis-conditioned table and trajectory sample, the true
candidate's sampled along- or across-axis response movie is treated as the
observation. The decoder is still source-disjoint and feature-embedding based.
All-scale feature-cosine contrast `axis_edge_parallel - axis_edge_orthogonal`,
with trial/scale clustered confidence intervals:

```text
compact response-only: -0.00960  CI [-0.01556, -0.00380]
true-tau residual:     -0.00557  CI [-0.01093, +0.00019]
raw true-tau concat:   -0.00120  CI [-0.00800, +0.00569]
0x static control:     +0.00000  CI [+0.00000, +0.00000]
```

So the direct candidate-free MLP test also does not support an along-contour
advantage. A follow-up source-fold seed check is important here: with
`fold_seed=20260625`, compact response-only changes sign to `+0.00597`, CI
`[-0.00366, +0.01697]`, while true-tau residual remains near null at
`-0.00276`, CI `[-0.01039, +0.00520]`. The stricter interpretation is therefore
not "across wins", but "the direct MLP axis effect is split-sensitive and not
reliable."

Because the classic axis story came from the matched-static axis cache rather
than the promoted hard-negative continuous-joint cache, the direct MLP test was
also rerun on the matched-static cache:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/
direct_axis_mlp_feature_decoder_matched_static/
direct_axis_mlp_feature_decoder_README.md
```

For that cache, only the response-only and static-control modes are available
because the response tables do not store trajectory coordinates. The all-scale
feature-cosine contrast is:

```text
compact response-only: -0.00541  CI [-0.01492, +0.00426]
0x static control:     +0.00000  CI [+0.00000, +0.00000]
```

Thus the matched-static candidate-free MLP version also does not recover a
reliable along-contour advantage. The alternate source-fold seed also changes
sign (`+0.00191`, CI `[-0.00668, +0.01086]`). The MLP result should therefore
be treated as null/split-sensitive, even though the older known-axis posterior
diagnostic is separately across-favoring.

A separate representation diagnostic now answers a cleaner known-trace control question:
does the V1 twin represent local image features better with measured motion
than with the 0x stabilized counterfactual? At the 1x scale, matched
feature-posterior rows give `0.6678` for `zero_static`, `0.8721` for hidden-eye
`full_exact`, and `0.9358` for `known_eye`. Thus the known-trace 1x control is
well above the stabilized counterfactual (`+0.2680` feature cosine), while the
hidden-eye joint observer preserves most of that gain (`+0.2043`). This
diagnostic supports a representation claim inside the deterministic table; the
promoted no-start observer still supports the harder latent-eye recovery claim.

The clean separation is:

```text
known_eye - zero_static:
  deterministic known-trace gain from measured motion over the stabilized
  counterfactual, not an independent response target.

full_exact - zero_static:
  how much of that gain survives when eye position is hidden and inferred /
  marginalized by the joint observer.

compact_only / compact_removed:
  whether compact/shared response geometry is sufficient for, and important to,
  that recovery.
```

A completed along-versus-across check keeps the axis claim bounded. Re-scoring
the promoted strict no-start joint estimator by `axis_edge_parallel` versus
`axis_edge_orthogonal` gives an all-scale paired feature-cosine contrast of
only `+0.0011` along-minus-across, with confidence intervals crossing zero at
every scale. At 1x, along is `0.9407` and across is `0.9366`, with identical
hard image accuracy (`0.7031`). Therefore the Figure 4D along-contour readout
should remain a matched-static axis-prior result, not a property automatically
inherited by this continuous joint estimator.

Known-eye is included in that diagnostic as a ceiling/control, but it is not an
axis-prior test. Once the true measured eye trace is supplied, the
`axis_edge_parallel` versus `axis_edge_orthogonal` label does not change the
observer, so the paired axis contrast is exactly zero. Zero-eye is also
identical across axis labels for the same reason.

The practical difference is the inference problem being solved. The older
matched-static feature-posterior setup holds the candidate family to
matched-static response alternatives, injects an axis-conditioned trajectory
prior, and asks whether the posterior improves local pyramid feature recovery
relative to zero-eye in `-MSE`. The promoted strict continuous estimator uses
the hard-negative continuous-joint cache, has to infer the latent trace without
a start anchor, and is scored by posterior-weighted feature cosine and hard
image identity. This makes the newer analysis a stricter inheritance check,
not a re-run of the same 4D observer.

## Motivation

The aggregate model establishes that response movies can carry feature
structure. The joint observer asks a different question: can image information
remain usable when the observer does not know which eye trajectory generated
the response? This matters because biological downstream circuits rarely get a
perfect external label for retinal pose. Motion is part of the inference
problem, not merely a nuisance to average away.

The trajectory-table observer is intentionally explicit. It builds a finite
catalog of response movies for candidate images and candidate trajectories,
then compares observers or interventions: one that knows the true trajectory,
one that assumes zero motion, one that marginalizes over a trajectory prior,
and compact-subspace variants that keep, remove, or add back the compact
translation component. This makes the compact claim testable rather than a
visual interpretation of a latent space.

## Notation And Estimator Contract

Shared notation:

```text
I: candidate image/window
tau: candidate eye trajectory
y = f_theta(I, tau): V1-twin response movie
p(tau): trajectory prior
p(I): image prior over the candidate set
phi(I), z: compact feature embedding of image/window I
```

The exact table stores:

```text
T[I, tau] = f_theta(I, tau)
```

For an observed response movie `y_obs = f_theta(I*, tau*)`, the observer scores
each candidate pair with a likelihood:

```text
L(I, tau | y_obs) = p(y_obs | I, tau)
```

The three observer contracts are:

```text
known-eye:
  I_hat = argmax_I L(I, tau* | y_obs)

zero-eye:
  I_hat = argmax_I L(I, tau_0 | y_obs)

joint-eye:
  I_hat = argmax_I sum_tau L(I, tau | y_obs) p(tau)
```

Equivalently, the joint observer forms:

```text
p(I, tau | y_obs) proportional to p(y_obs | I, tau) p(tau) p(I)
p(I | y_obs) = sum_tau p(I, tau | y_obs)
```

Note the scope of the word *joint* here. This is marginalization over a finite,
fixed catalog of trajectories, not inference of a continuous eye position. There
is no trajectory estimate and no recursive state: `sum_tau` runs over a discrete
prior list, so the observer integrates over candidate trajectories rather than
recovering one. The load-bearing assumption is therefore A1, catalog coverage —
the rescue can be no better than the best-covering catalog entry, and a sparse
or mismatched catalog biases joint-eye downward. A continuous trajectory decoder
(see "Tractable Continuous Trajectory Inference" below) removes that ceiling, at
the cost of a motion model and a local-linearity assumption.

The older observer outcome is image-identification accuracy over candidate
sets, with posterior concentration diagnostics such as `N_eff / K` used as
guardrails. The current promoted 4C outcome is feature-posterior recovery,
reported as cosine similarity between recovered and target local feature
vectors.

The continuous feature-embedding branch changes the endpoint contract:

```text
candidate-posterior feature estimate:
  phi_hat = sum_I p(I | y_obs) phi(I)

continuous feature embedding estimate:
  z_hat = E[z | y_obs] under p(y_obs | z, tau) p(z) p(tau)
```

The implemented first pass uses a linear-Gaussian `p(y | z)` readout and treats
trajectory as nuisance variation in the hidden-eye training bank. It is
therefore continuous in `z`, but it is not yet the full joint `(z, tau)`
optimizer described by the ideal contract.

For the compact-subspace intervention, the contract is:

```text
y_full = f_theta(I, tau)
y_zero = f_theta(I, tau_0)
y_compact_only = project_compact(y_full - y_zero) + y_zero
y_compact_removed = y_full - project_compact(y_full - y_zero)
y_compact_addback = y_compact_removed + project_compact(y_full - y_zero)
```

The claim is supported if:

```text
feature_recovery(compact_only) remains close to feature_recovery(full_joint)
feature_recovery(compact_removed) moves toward feature_recovery(zero_eye)
feature_recovery(compact_addback) reconstructs feature_recovery(full_joint)
```

## Plain-English Methods

The 4C analysis asks what an observer can recover when it sees a response movie
but does not know which eye trajectory produced it. The implementation makes
this inference problem explicit by building a table of model responses for many
candidate images and candidate trajectories.

For each candidate image, the V1 twin is run on several rendered retinal
movies. Each movie uses the same image but a different sampled eye trajectory.
Together these responses form a lookup table: if the image were this one and
the trajectory were that one, this is the response movie the model would
produce.

The observed response is then held out and compared with entries in the table.
The known-eye observer is allowed to compare only against the true trajectory.
This is the ceiling for the finite candidate set. The zero-eye observer ignores
motion and compares the observed response with the static response. The joint
observer does not know the true trajectory, so it averages or sums evidence
over the possible trajectories in the prior catalog.

The older observer analyses scored image identification: did the observer pick
the right image from a candidate set? The current promoted panel scores feature
recovery instead. After the joint observer forms a posterior over candidate
images and trajectories, that posterior is used to recover the local image
feature vector. The panel reports how close the recovered feature vector is to
the true feature vector.

The compact intervention is applied to the motion-induced part of the response.
First, subtract the static response from the moving response. Then project that
motion-induced response onto the compact translation subspace, remove that
component, or add it back. The `compact_only` variant keeps only the compact
motion component plus the static response. The `compact_removed` variant removes
that component from the full moving response. The `compact_addback` variant
checks that adding the removed component back reconstructs the full result.

The main comparison is therefore not just "joint is better than zero-eye." It
asks whether the compact translation component carries much of the useful
feature-recovery signal. The expected pattern is: known-eye is highest,
full-joint and compact-addback are close, compact-only retains much of the full
joint result, and compact-removed falls toward the zero-eye result.

Candidate sets and priors are guarded so the test is not too easy. Historical
matched-static observers selected distractor images with similar stabilized
responses, which reduces the chance that static response strength alone solves
the task. The current compact-removal endpoint uses hard-negative feature
posterior recovery. Leave-one-out trajectory priors prevent the exact observed
trace from being reused as an unfair template.

Uncertainty and interpretation are kept at the endpoint level. The posterior is
used as a tool for recovering image features; it is not interpreted as proof
that the animal, or the observer, exactly identified the true eye trace.

## Assumptions

A1. The finite trajectory table samples enough of the relevant response
variation to make marginalization meaningful.

A2. The likelihood scale is calibrated well enough that accuracy ordering is
not a trivial temperature artifact.

A3. Candidate sets are hard enough to prevent static-response shortcuts. The
matched-static image-identity observer is the historical control for this
assumption; the current compact-removal panel uses hard-negative feature
posterior recovery and should be read with that endpoint in mind.

A4. Leave-one-out trajectory priors prevent the observer from using the exact
observed trace as an unfair template.

A5. The posterior is used for image or feature recovery, not as a claim that
the true trajectory is exactly identified.

A6. Compact-subspace projection is a mechanistic intervention on response
components. It can show importance and sufficiency for this endpoint, but not
uniqueness over all useful low-dimensional components. The current matched
static-PC controls do not fail; they are nearly as strong as compact.

## Controls

Known-eye observer:

```text
Upper reference for image information when the true trajectory is specified.
```

Zero-eye observer:

```text
Failure mode for treating motion as if the image were stabilized.
```

Joint-eye observer:

```text
Tests whether marginalizing over plausible trajectories recovers pose-lost
image information.
```

Matched-static distractors:

```text
Candidate images are selected to have similar stabilized V1-twin responses.
This blocks the trivial explanation that joint-eye wins only through static
response strength.
```

Hard-negative candidate sets:

```text
Stress-test image discrimination against structurally similar alternatives.
```

Empirical and OU priors:

```text
Check whether rescue depends on the exact empirical trajectory prior or on a
broader plausible confined-motion prior.
```

Posterior `N_eff / K`:

```text
Checks whether the posterior is diffuse, concentrated, or collapsing to a
trivial single trajectory.
```

Compact-only projection:

```text
Keeps only the compact translation component of the motion-induced response.
Tests whether that component is sufficient to retain feature recovery.
```

Compact-removed projection:

```text
Removes the compact translation component from the response. Tests whether the
feature-posterior endpoint collapses toward zero-eye when that component is
absent.
```

Compact-addback reconstruction:

```text
Adds the compact component back after removal. This is the algebraic/QC control
that the intervention is targeting the intended response component.
```

Static-PC, gain, random, and unit-shuffle controls:

```text
Compact beats random, unit-shuffled, and gain-only alternatives, while
static-response PCs remain a serious matched contender. The current static-PC
tests are strong enough that compact should be framed as important, not unique.
```

## Existing Evidence

Primary current 4C source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_feature_posterior_compact_removed_pyramid_k8_n128_scales_0p5_1_2_v1/
    feature_compact_mechanism_summary.csv
    feature_compact_mechanism_uncertainty.csv
    feature_compact_mechanism_qc.csv
```

Run scope for the promoted endpoint:

```text
candidate mode = hard_negative_structure
feature target = pyramid_local_field
feature PCA k = 8
compact k = 10
motion scales = 0.5x, 1x, 2x
variants = full_exact, zero_static, compact_only, compact_removed, compact_addback
score = feature recovery cosine
```

Current panel values:

```text
zeroed-eye feature recovery:
  0.5x = 0.765
  1.0x = 0.668
  2.0x = 0.576

compact-only feature recovery:
  0.5x = 0.850
  1.0x = 0.838
  2.0x = 0.826

compact-removed feature recovery:
  0.5x = 0.759
  1.0x = 0.635
  2.0x = 0.537

full joint / compact-addback feature recovery:
  0.5x = 0.872
  1.0x = 0.872
  2.0x = 0.871

known-trace control:
  0.5x = 0.927
  1.0x = 0.936
  2.0x = 0.949
```

This is the analysis that directly matches the panel title: compact-only
retains much of full joint feature recovery, while compact removal falls toward
or below the zero-eye curve.

### Diagnostic Figure Pack Added 2026-06-23

A cache-only diagnostic figure pack now checks the promoted feature-posterior
endpoint against the older image-identity joint observer and compact-intervention
QC. This is a checking artifact, not a replacement promotion candidate.

Builder:

```text
declan/figure4_active_sensing_atlas/scripts/build_panel_c_joint_decoder_checks.py
```

Outputs:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/
  panel_C_joint_decoder_check_sheet.png
  panel_C_joint_decoder_check_sheet.pdf
  panel_C_joint_decoder_axis_detail.png
  panel_C_joint_decoder_axis_detail.pdf
  panel_C_joint_decoder_feature_summary.csv
  panel_C_joint_decoder_contrasts.csv
  panel_C_joint_decoder_observer_accuracy.csv
  panel_C_joint_decoder_feature_rows.csv
  panel_C_joint_decoder_checks_README.md
```

The check sheet has six panels:

```text
A. promoted feature endpoint:
   zero-eye, compact-removed, compact-only, full-joint, and known-eye curves.

B. paired feature-cosine contrasts:
   compact-only minus compact-removed, full-joint minus compact-removed, and
   compact-removed minus zero-eye with bootstrap intervals.

C. posterior concentration:
   median N_eff / K for the same response variants.

D. addback and clipping QC:
   compact-addback equals full-joint to numerical precision, while clipping is
   mainly a compact-removed issue at larger scale.

E. older image-identity observer:
   matched-static known-eye / zero-eye / joint-eye accuracy, included to verify
   that the historical observer result has the same qualitative zero-to-joint
   rescue.

F. axis-prior detail:
   split values for the two axis-conditioned compact-source priors.
```

The diagnostic read is consistent with the promoted claim. Averaged over the
two axis priors, feature recovery is:

```text
                 0.5x    1.0x    2.0x
zero-eye         0.765   0.668   0.576
compact removed 0.759   0.635   0.537
compact only    0.850   0.838   0.826
full joint       0.872   0.872   0.871
known eye        0.927   0.936   0.949
```

Thus the visual audit shows the intended ordering: compact-only remains high
and close to full-joint, compact-removed tracks the zero-eye failure mode and
falls below zero-eye at larger scales, and known-eye remains the ceiling.
Compact-addback reconstructs full-joint with maximum raw addback error
`3.5e-18`, so the intervention algebra is behaving as intended.

Historical observer source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
    observer_summary.csv
```

Run scope:

```text
n_images = 64
n_candidates = 8
candidate modes = hard_negative_structure, matched_static_response
observation family = empirical
prior families = empirical, OU
scales = 0.5x, 1.0x
n_prior_trajectories = 8
trajectory prior mode = leave_one_out
likelihood scales = 0.5, 1.0
```

Accuracy ordering:

```text
hard_negative_structure, 0.5x:
  zero 0.578, joint 0.781-0.844

hard_negative_structure, 1.0x:
  zero 0.312, joint 0.734-0.875

matched_static_response, 0.5x:
  zero 0.578, joint 0.750-0.828

matched_static_response, 1.0x:
  zero 0.328, joint 0.672-0.797
```

Matched-static rescue at 1.0x, likelihood scale 1.0:

```text
empirical prior:
  known-eye = 1.000
  zero-eye = 0.328
  joint-eye = 0.766
  median N_eff / K ~= 0.364

OU prior:
  known-eye = 1.000
  zero-eye = 0.328
  joint-eye = 0.797
  median N_eff / K ~= 0.400
```

The matched-static result remains one of the cleanest historical observer
findings because the zero-eye failure cannot be dismissed as an easy static
response mismatch. It supports the joint eye/image decoding half of the title,
but it is no longer the exact endpoint plotted in current 4C.

Static-response-PC specificity audit:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_symmetric_subspace_removal_prod_v1/

outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_feature_symmetric_subspace_removal_prod_v1/

outputs/matched_twin_covariance_closure_static_pc_predictor_prod_v1/
```

Here "static-response PCs" means the main population response patterns seen
when the model views many static image windows. They are not eye-movement
patterns by construction. They are a strong control because small eye movements
move an image locally, and local image changes naturally follow directions that
already live in the static image-response space.

The new tests ask whether compact translation geometry still looks special
after comparing it with those static-image directions.

In the image-identity decoder, removing compact directions and removing
static-response-PC directions hurt by almost the same amount:

```text
full-minus-removed accuracy loss:
  compact removed   = 0.337
  static-PC removed = 0.315
```

In the feature-recovery endpoint used by Panel 4C, compact removal hurts
somewhat more, but static-PC removal is still nearly as damaging:

```text
full-minus-removed feature neg-MSE loss:
  compact removed   = 45.042
  static-PC removed = 38.900
```

When the shared overlap between compact and static PCs is removed first, very
little unique structure remains:

```text
feature neg-MSE loss after removing only residualized pieces:
  compact residual after static PCs = 3.670
  static-PC residual after compact  = 0.963
```

The covariance predictor test is the strongest check because it is not just a
"how much tangent variance can this basis capture?" test. It asks which basis
actually predicts recorded FEM-linked covariance. In the current full-sample
Allen `2022-02-16` run, the static-PC predictor matches or slightly beats the
compact predictor:

```text
k=10 covariance capture:
  no projection:
    compact predictor   = 0.742
    static-PC predictor = 0.748

  after global_rate+target_pc1 projection:
    compact predictor   = 0.417
    static-PC predictor = 0.425
```

These gaps are small and change sign across endpoints: compact removal is a bit
worse in the ablation tests, while static PCs are a bit higher in the covariance
prediction test. That is the pattern expected if there is no reliable
compact-specific advantage over static PCs.

Plain-English conclusion:

```text
The compact component matters. But what matters is mostly the part it shares
with the ordinary image-response manifold. Current results support a shared
manifold story, not a compact-only mechanism.
```

## Mechanistic Posture

Figure 4C can be written as a mechanistic bridge, but not yet as a definitive
mechanism proof.

The current evidence supports the claim that structured motion-induced response
variation in the V1 twin can be used by a joint image-and-eye observer. The
compact-geometry analyses sharpen that statement by showing that an
image-disjoint compact translation subspace is a plausible carrier of much of
the motion-dependent likelihood structure. This is stronger than a purely
descriptive observer result, because it identifies a concrete response
component that can preserve the latent-eye rescue.

The stricter feature-space compact-removal test is now complete for the current
promoted endpoint. The newer static-PC tests show that the compact component is
not cleanly separable from the response directions already used for static
images. The safe language is:

```text
compact geometry supports or carries much of the joint-observer rescue;
compact-only is a sufficiency/mechanism guardrail;
compact-removed loss is evidence that this response component is important;
static-response PCs are a matched control and perform nearly as well;
compact geometry is not the currently supported unique mechanism.
```

Avoid language implying that the decoder explicitly uses a compact trajectory
prior, that the compact basis uniquely explains the rescue, or that the
posterior identifies the animal's true eye trajectory.

The best current mechanistic wording is:

```text
Eye movements push the V1 population along a compact set of response directions.
Those directions are largely shared with the main response patterns evoked by
static images. This shared structure can support latent-eye feature recovery,
but it should not be described as a dedicated compact-only eye-movement code.
```

## Diagnostics And Failure Modes

The main failure modes are:

```text
joint-eye wins by static response strength rather than motion-aware inference;
the scale effect is only zero-eye failure;
the posterior collapses to a meaningless or trivial trajectory;
the prior catalog leaks the observed trace;
the trajectory catalog is too sparse to cover tau*, capping joint-eye and
  mimicking a compact-removed collapse;
candidate hardness or source imbalance drives the effect;
compact-mechanism projections are overread as unique mechanisms;
static PCs explain comparable useful structure and are not cleanly beaten.
```

Current handling:

```text
Keep compact-only, compact-removed, compact-addback, zero-eye, and known-eye in
the main panel.
Route matched-static image identity and posterior concentration to historical
support or supplement.
Use compact projection as importance/sufficiency evidence, not a
unique-mechanism proof over static PCs.
Mention that static-PC controls are nearly matched whenever the compact
mechanism is interpreted.
Check catalog coverage (e.g. distance from tau* to its nearest catalog entry in
response space); cross-check joint-eye against the continuous Kalman observer so
a compact-removed collapse is confirmed real rather than catalog under-coverage.
```

## Current Claim Boundary

Supported:

```text
When retinal pose is latent, joint image/trajectory inference recovers local
feature information above a zero-eye observer, and the compact translation
component is important for that recovery: compact-only retains much of the full
joint feature recovery, while compact removal collapses recovery toward the
zero-eye curve.
```

Also supported:

```text
The useful compact component is mostly shared with static image-response
directions. This means Panel 4C can say that a shared compact image-response
structure supports the latent-eye recovery, not that compact geometry is unique.
```

Not yet supported:

```text
The posterior exactly identifies the animal's true eye trajectory.
Empirical FEMs are proven optimal relative to all plausible priors.
The compact translation basis is the unique mechanism of the rescue relative to
static response PCs or every other low-dimensional response component.
The decoder uses compact geometry as an explicit eye-trajectory prior.
Axis-specific edge-parallel behavior follows directly from this observer.
```

Now disfavored by the completed static-PC tests:

```text
The finite-difference compact basis is privileged over the static image-response
manifold for explaining FEM-linked covariance.
The compact translation subspace should be presented as a dedicated
eye-movement-only code.
```

## Tractable Continuous Trajectory Inference (Beyond the Catalog)

Status update: this section started as the design for a continuous alternative
to catalog marginalization. A cache-only version has now been implemented and
promoted as the no-anchor feature-recovery readout. The current promoted
artifact uses:

```text
observer: noanchor_quadratic_strict_scale_prior_predeclared
basis dims by scale: 0.5:10, 1.0:20, 2.0:20
ridge by scale:      0.5:0.01, 1.0:0.1, 2.0:0.1
posterior temp:      0.5:0.125, 1.0:0.125, 2.0:0.5
trajectory prior:    0.5x AR(1), 1.0x AR(1), 2.0x matched-Brownian scale 8
heldout feature cosine: 0.9371
emitted feature cosine: 0.9378
image accuracy:      0.7083
```

The run is reproducible via:

```text
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.figure4_active_sensing_atlas.scripts.run_panel_c_promoted_continuous_joint_observer
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.figure4_active_sensing_atlas.scripts.verify_panel_c_promoted_continuous_joint_observer --expect-full
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.figure4_active_sensing_atlas.scripts.audit_panel_c_continuous_joint_feature_calibration
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.figure4_active_sensing_atlas.scripts.audit_panel_c_joint_decoder_inherited_contracts
```

The audit command is the reusable gate for future encoder candidates: it
selects posterior temperatures on held-out source rows when source identities
are available, and evaluates posterior-weighted feature cosine on the heldout
split. A replacement encoder should beat this source-row-heldout
feature-recovery gate, not merely improve hard image accuracy.

## Next Target: Linear Candidate-Free Synthetic-Prior Feature Observer

Working decision, 2026-07-01: the desired next 4C variant is not the nonlinear
MLP branch and not the finite candidate-image posterior endpoint. The intended
observer contract is:

```text
observed response movie
  -> infer continuous latent eye trace tau_hat
     under compact response geometry and synthetic_empirical_confined prior
  -> infer continuous feature embedding z_hat
  -> score z_hat against the true image/window feature embedding
```

This should be a plug-in continuation of the current continuous quadratic
observer and the earlier continuous feature-embedding reconstruction branch.
It should use `synthetic_empirical_confined` only as an empirically calibrated
synthetic FEM process prior: held-out training FEM traces calibrate the
confined-step generator, new synthetic traces are sampled, and the continuous
trajectory prior is fit to those generated traces. It should not be described
as purely empirical traces or fully empirical replay.

Important distinction from current artifacts:

```text
Promoted continuous-joint observer:
  response -> continuous tau/image scores -> posterior over candidate images
           -> posterior-weighted candidate feature vector

Candidate-free linear feature embedding diagnostic:
  response -> continuous feature embedding z_hat
```

The target is to combine the second endpoint with the first observer's
continuous trajectory inference, using the synthetic empirical confined prior.
The feature endpoint should be linear/linear-Gaussian, not MLP.

Current catalog-like uses should be separated explicitly:

1. Candidate-image posterior endpoint: should be removed for this variant.
2. Trajectory-candidate selection/replay: should be removed for this variant.
3. Aggregate trajectory statistics for the prior: allowed. This is the role of
   `synthetic_empirical_confined`.
4. Response-table samples used to fit the response-vs-displacement map: still
   present in the cache-only implementation unless replaced by finite
   difference or analytic compact derivatives. Label this as
   response-table-calibrated if it remains.
5. Supervised feature targets/source rows used to train and score the feature
   embedding decoder: allowed, with source-row-disjoint folds.

Implementation plan:

1. Keep the current `synthetic_empirical_confined` process-prior option in
   `analyze_continuous_joint_trajectory.py`, but treat it as a trajectory prior
   provider rather than a candidate-feature endpoint.
2. Add a linear candidate-free feature observer runner, probably as a sibling
   of `build_panel_c_continuous_feature_embedding_reconstruction.py`, that can
   read a continuous-joint run or invoke the continuous scorer to obtain
   `tau_hat` under the synthetic prior.
3. Use compact observed response features plus optional linear tau terms. Start
   with the simplest nested linear-Gaussian design:

   ```text
   z ~ N(0, I)
   response_compact | z, tau_hat ~ Gaussian linear model
   z_hat = E[z | response_compact, tau_hat]
   ```

   Also include response-only and zero/static baselines under the same
   source-row folds.
4. Train only on source-row-disjoint response-bank rows. Do not score by
   posterior averaging over candidate images. In the stricter implementation,
   feature-decoder training rows are the observed response-table rows themselves,
   not expanded cached prior-trajectory response rows.
5. Emit trials, summary, contrasts, and manifest files with explicit provenance
   columns:

   ```text
   feature_endpoint = continuous_embedding
   trajectory_process_model = synthetic_empirical_confined
   trajectory_prior_description = empirically_calibrated_synthetic_fem_prior
   response_map_source = response_table_calibrated | finite_difference
   decoder_family = linear_gaussian
   uses_candidate_posterior_endpoint = false
   uses_mlp = false
   ```

6. Compare against the promoted candidate-posterior observer on the same
   source-row-heldout feature cosine gate, but report it as a different
   endpoint: continuous feature embedding recovery, not candidate posterior
   feature recovery.
7. If the cache-only version is positive, the stricter follow-up is to replace
   the response-table-calibrated response-vs-displacement map with
   finite-difference or analytic compact derivatives. That would leave only
   aggregate FEM statistics for the trajectory prior and supervised source rows
   for feature-target fitting.

Success criteria:

```text
Primary: source-row-heldout feature cosine above zero/static baselines.
Secondary: match or approach the promoted candidate-posterior feature recovery.
Guardrails: no MLP, no candidate-posterior endpoint, no empirical trace replay,
and clear labeling of any response-table calibration that remains.
```

Implementation update, 2026-07-01:

```text
declan/figure4_active_sensing_atlas/scripts/build_panel_c_linear_synthetic_prior_feature_observer.py
```

This runner implements the cache-only version of the target contract. The
post-review version no longer reads `filtered_state_means[true_candidate]`.
It fits a pooled response-table displacement geometry, uses
`synthetic_empirical_confined` as the full-path transition prior, and obtains a
candidate-free `tau_hat` by MAP optimization. It then fits source-row-disjoint
linear-Gaussian feature observers from observed response-table rows. Default
modes are:

```text
response_only
synthetic_tau_linear
synthetic_tau_interactions
zero_static
```

The optional `true_tau_linear`, `true_tau_interactions`, and
`known_eye_response_only` modes are diagnostic controls, not the primary model.
The output manifest and trials carry explicit provenance fields:

```text
uses_candidate_posterior_endpoint = false
uses_nonlinear_mlp = false
uses_trajectory_catalog_replay_endpoint = false
uses_true_image_conditioned_tau_hat = false
uses_response_table_trajectory_training_rows = false
tau_hat_source = candidate_free_pooled_response_geometry_full_path_map
observation_geometry_source = pooled_response_table_local_response_vs_displacement_map
trajectory_prior_label = synthetic_empirical_confined uses an empirically calibrated synthetic FEM prior
```

Smoke checks passed:

```text
uv run python -m compileall declan/figure4_active_sensing_atlas/scripts/build_panel_c_linear_synthetic_prior_feature_observer.py

uv run python -m declan.figure4_active_sensing_atlas.scripts.build_panel_c_linear_synthetic_prior_feature_observer \
  --max-tables 6 --scales 2.0 --n-bootstrap 0 --synthetic-prior-samples 4 \
  --quadratic-optimizer-max-iter 1 --feature-dim 1 --n-folds 2 \
  --observer-modes response_only,synthetic_tau_linear,synthetic_tau_interactions,zero_static \
  --out-dir /tmp/linear_synthetic_prior_feature_observer_smoke_fixed

uv run python -m declan.figure4_active_sensing_atlas.scripts.build_panel_c_linear_synthetic_prior_feature_observer \
  --max-tables 6 --scales 2.0 --n-bootstrap 0 --synthetic-prior-samples 4 \
  --quadratic-optimizer-max-iter 1 --feature-dim 1 --n-folds 2 \
  --observer-modes response_only,synthetic_tau_linear,synthetic_tau_interactions,true_tau_linear,true_tau_interactions,zero_static,known_eye_response_only \
  --out-dir /tmp/linear_synthetic_prior_feature_observer_smoke_fixed_all_modes
```

The metric policy is now explicit: use heldout posterior-weighted feature
cosine as the primary development task, and report exact image identity as the
harder MAP readout. This avoids over-penalizing near-miss posteriors whose
feature estimate is already moving toward the true candidate. The current
decision artifact is:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/continuous_joint/continuous_joint_candidate_decision_table.csv
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/continuous_joint/continuous_joint_candidate_decision_table.png
```

It promotes the strict inferred-start scale-prior observer as the no-start
endpoint (`0.9371` heldout feature cosine, `0.7083` image accuracy), treats
known-start as a less-strict candidate (`0.9361` for AR(1), `0.9367` for the
scale-prior hybrid), marks the
matched-Brownian known-start prior as a smoke-positive but full-cache-blocked
diagnostic (`0.9360`, `0.6992`), records a scale-specific AR(1)/Brownian prior
hybrid as the current predeclared less-strict feature lead (`0.9367`, `0.7070`), and keeps
affine x1000 diagnostic-only despite
its higher cosine (`0.9374`) because its gain is intercept-dependent and its
image accuracy falls (`0.6927`).

The matched-Brownian prior path is now implemented correctly for the quadratic
profile objective: the 2x2 covariance estimated from heldout trajectory samples
is used in the final optimized trajectory energy, not only in the linear start.
The smoke64 covariance-scale sweep peaks at scale 8 (`0.9688` feature cosine,
`0.8281` image accuracy), but the full-cache scale-8 gate lands just below
known-start AR(1) (`0.9360` vs `0.9361`) and lowers image identity. Treat it as
a useful prior diagnostic, not a promotion.

The scale-specific prior hybrid keeps AR(1) at 0.5x/1.0x and uses Brownian8
only at 2.0x, where the full-cache slice audit favors the looser matched
Brownian prior. This raises the full-cache heldout feature cosine to `0.9367`
with unchanged image accuracy `0.7070`. The same rule has now been rerun from
source with predeclared analyzer options and matches the posthoc posterior-row
hybrid exactly. Treat it as the leading less-strict feature-primary candidate;
the strict inferred-start observer remains the no-start endpoint.

The stronger result is that the same predeclared scale-specific prior also
improves the strict inferred-start endpoint itself: heldout feature cosine rises
from `0.9343` to `0.9371` with image accuracy unchanged at `0.7083`. That strict
source rerun is now the promoted no-start result.

A new diagnostic lead now exists for the identified `2.0x` parallel bottleneck:
`quadratic_affine_poisson_profile` improves the full-slice heldout feature
cosine from `0.9066` to `0.9220` against a matched origin-constrained k20
quadratic control. Treat it as an observation-model lead, not a promotion,
because the all-scale/axis and offset guardrails below are the fair promotion
tests.

The first all-scale/axis affine gate blocked promotion: the unguarded affine
scale-conditioned model improved default feature cosine (`0.9171` vs `0.9108`)
but fell below the promoted heldout calibrated feature gate (`0.9331` vs
`0.9343`) and lowered image accuracy (`0.6693` vs `0.7083`). The intercept
fraction was tiny at 0.5x but grew to roughly one third of the coefficient norm
in the 2.0x slices, so the follow-up added an explicit intercept-ridge guard.

The guarded affine result is now a principled feature-primary lead, not yet a
clean replacement. With intercept ridge multiplier `1000`, split-swapped model
selection chooses `affine_x1000` on both heldout halves and yields heldout
feature cosine `0.9374`, above the origin-constrained promoted gate `0.9343`.
The same model lowers hard image accuracy (`0.6927` vs `0.7083`), so the
interpretation is: the affine offset helps posterior feature recovery, but the
MAP image-ID endpoint still favors the origin-constrained model. Treat
`affine_x1000` as the next candidate to stress-test with a causal static-offset
ablation and a write-lock/promotion pass, not as an automatic promotion.

The first offset guardrail is positive but incomplete. The new
`continuous_joint_affine_offset_guardrail` diagnostic shows that the x1000
penalty cuts the largest 2.0x median intercept fractions from roughly
`0.33-0.39` to `0.14-0.15` while retaining the split-swapped feature lead. This
argues against the simplest "affine wins only by a huge static offset" failure
mode, but it is still an intercept-burden check rather than a direct ablation of
static/candidate offsets.

The direct ablation gate is now complete. The analyzer accepts
`--quadratic-affine-intercept-scale 0`, which fits the same affine maps but
zeros the intercept contribution during trajectory profiling and Poisson
scoring. On the full cache, normal x1000 beats the intercept-ablated x1000
control in heldout feature cosine (`0.9374` vs `0.9184`), posterior true mass
(`0.5772` vs `0.5412`), and image accuracy (`0.6927` vs `0.6380`). The ablated
control also falls below the origin-constrained promoted observer
(`0.9343` feature cosine, `0.7083` image accuracy). Interpretation: the affine
feature lead is intercept-dependent. This is useful as a diagnostic, but it
blocks promotion unless the intercept can be justified as legitimate
motion-conditioned structure rather than static/candidate offset leakage.

I also tested a more constrained version of that idea:
`quadratic_prior_mean_poisson_profile`. It fits deviations around the
heldout-safe trajectory-prior mean and uses the implied prior-mean offset rather
than a freely fit intercept. On a matched 64-table smoke, it improves over
intercept removal (`0.9551` vs `0.9402` heldout feature cosine) but remains
below both origin (`0.9637`) and free affine x1000 (`0.9591`). That makes
prior-mean centering a useful negative control, not the next full-cache
candidate.

The strongest clean candidate is instead known-start quadratic inference. This
keeps the origin-constrained scale-conditioned quadratic observation model but
adds a soft prior on the first measured eye-position sample. Full-cache
heldout feature cosine improves from `0.9343` to `0.9361`, posterior true mass
from `0.5990` to `0.6029`, and image accuracy is essentially unchanged
(`0.7083` to `0.7070`). A split-swapped selector restricted to inferred-start
vs known-start chooses known-start on both halves. Because this uses the first
eye sample, it should be framed as a less strict no-anchor endpoint, but it is
a principled trajectory-prior improvement rather than a response-offset
shortcut.

The calibrated known-start production artifact is:

```text
continuous_joint_quadratic_poisson_scale_conditioned_knownstart_calibrated_full
```

Using the promoted fixed scale-temperature schedule, that artifact reaches
emitted feature cosine `0.9374`. The stricter model-selection number remains
`0.9361`, because it is selected and evaluated split-swapped.

Candidate verifier and manifest:

```text
declan.figure4_active_sensing_atlas.scripts.verify_panel_c_knownstart_continuous_joint_observer --expect-full
continuous_joint_knownstart_observer_manifest.json
```

The rest of this section explains why that observer class was the tractable
move, and which assumptions still bound it.

The catalog observer marginalizes over a fixed list of trajectories; it never
infers a continuous eye path, and it is ceilinged by catalog coverage (A1). A
continuous joint observer removes that ceiling, and there is a tractable way to
do it that reuses machinery we already have rather than reaching for a full
particle filter.

The Figure 4 geometry gives a first-order model of the reafferent response: to
first order the motion-induced response is linear in displacement,

```text
y(t) ≈ lambda_zero(I, t) + J(I) tau(t),      J(I) ≈ U_trans A(I)
```

Projecting the observed response into the compact (or static-PC) coordinates
removes the static part and linearizes the problem in displacement:

```text
z(t) = U^T [ y_obs(t) - lambda_zero(I, t) ] ≈ A(I) tau(t) + eps(t)
```

with `eps(t)` the whitened residual/readout noise in those coordinates. Put a
confined-motion state model on the trajectory (OU / AR(1)):

```text
tau(t) = alpha tau(t-1) + eta(t),   eta ~ N(0, Q)
```

Then `(tau(t), z(t))` is a linear-Gaussian state-space model, and for each
candidate image the trajectory is marginalized analytically and continuously by
a Kalman filter/smoother — no catalog. The per-image evidence is the Kalman
innovation likelihood,

```text
log p(y_obs | I) = sum_t log N( z_t ; A(I) tau_hat_{t|t-1}, S_t )
```

and identity/feature recovery uses `argmax_I p(y_obs | I) p(I)` exactly as now.
The known/zero/joint triple maps without a catalog: known-eye clamps
`tau = tau*`, zero-eye clamps `tau = 0` on the moving input, joint-eye runs the
Kalman marginalization.

Why this is the right tractable choice:

```text
- continuous: marginalizes the OU trajectory prior in closed form; no coverage
  ceiling.
- cheap: closed-form per timestep; no particle population, no dense catalog.
- reuses existing machinery: A(I) is the finite-difference twin Jacobian J(I)
  already built and provenance-audited for Figure 4.
- yields tau_hat: the smoother returns a continuous trajectory estimate, so you
  can finally report trajectory-recovery error vs drift (the Wu Fig 3b/d
  attribution: is the rescue from better eye inference or a better signal?),
  which the catalog observer could not give cleanly.
```

Validity domain and the honest caveats:

```text
- linearization regime: z ≈ A(I) tau holds for small (subpixel / sub-RF)
  displacements. Beyond that A(I) is displacement-dependent and the linear-
  Gaussian model breaks. Report the linearization residual vs displacement and
  restrict claims to where it is small (the same subpixel boundary as the
  rendering / LogMAR ceiling).
- larger excursions: relinearize J(I, tau_t) along the trajectory each step
  (iterated EKF / UKF). Still essentially closed-form, still no catalog.
- non-linear fallback: if the regime is genuinely non-linear (foveal sweeps
  across many RFs), use a small particle filter (Wu used N=10) conditioned per
  candidate image. Still cheaper than a catalog dense enough to cover trajectory
  space, because it samples where the posterior mass is.
- basis: run U as compact AND as static-PC (symmetric), since compact is not
  unique over the static manifold; the basis swap is the mechanism test, not a
  fixed choice.
- noise: use the full or low-rank residual covariance for eps, not diagonal, for
  commensurability with the covariance-aware analyses.
```

How it sits next to the catalog. Catalog and Kalman are complementary, and
reporting both is the rigorous move: the catalog is assumption-light on the
motion model (no OU / linearity assumption) and auditable but coverage-ceilinged;
the Kalman observer is assumption-heavier (OU prior plus local linearity) but
continuous and ceiling-free. If they agree, the catalog was not coverage-limited;
if the Kalman observer exceeds the catalog, the catalog was under-covering. That
directly tests A1 rather than asserting it.

This is still an image-candidate-discrete, trajectory-continuous observer — the
feature/identity endpoint is unchanged. Full continuous image recovery is the
separate Wu-style reconstruction below.

## Optional Extension: Wu-style Pixel Reconstruction with a Natural-Image Prior

This is a forward-looking alternative observer, not a current 4C result. It is
recorded here because, if pursued, it belongs in the 4C set rather than 4B: it
is a latent-eye joint image-and-eye observer scored by recovery quality, which
is what 4C already is. It should not replace the current feature-posterior
compact-removal endpoint.

### Why it lands in 4C and not 4B

4B's defining property is a held-out information lower bound: a bits axis. A
learned image prior injects information about natural images that did not come
from the response, so the moment a prior enters the loop the score stops being
"information recoverable from the twin response" and becomes "reconstruction
quality under a prior." That contaminates 4B's quantity. 4C is already a
recovery-quality panel (feature cosine / neg-MSE), and its observer is already
latent-eye and joint, so a prior-regularized reconstruction is a richer member
of the same family, not a different claim. The phase motivation that prompts
this — the pyramid target discards within-block phase, and cross-band phase
alignment under a linear decoder — is a reconstruction question, and
reconstruction is 4C territory.

### What it is

Following Wu et al., the target is the image in pixel space, and the prior
`p(I)` is a denoiser-implicit natural-image prior: the same prior object that
underlies diffusion / score-based models, but run as a MAP denoiser inside a
Plug-and-Play / half-quadratic-splitting loop, not necessarily as a full
stochastic sampling chain. The joint observer (Wu's joint-LNBRC-dCNN) alternates
an image-update step against an eye-trajectory update — EM with a particle
filter over the trajectory — recovering image and eye path together when pose is
latent.

The property that matters for us: the prior, not the data, regularizes the
high-dimensional pixel estimate. This is the direct answer to the dimensionality
blowup that made finer pooling or plain pixel reconstruction unworkable as a 4B
target (a roughly 100,000-dimensional target on 384 images). With the prior
carrying the image-statistics burden, the pixel estimate is tractable on limited
data, and phase is preserved because nothing is pooled away.

### What transfers from the existing 4C contracts

The observer triple maps one-to-one onto Wu's conditions:

```text
Wu known-LNBRC-dCNN  <->  known-eye observer (true trajectory supplied)
Wu zero-LNBRC-dCNN   <->  zero-eye observer (assume no motion)
Wu joint-LNBRC-dCNN  <->  joint-eye observer (marginalize / infer trajectory)
```

One caveat on this mapping: Wu's joint-LNBRC-dCNN *infers* a continuous
trajectory (EM plus a particle filter), whereas the current 4C joint-eye
observer *marginalizes a fixed catalog*. So moving to Wu-style reconstruction is
a change of observer class — catalog-marginalization to continuous trajectory
inference — not merely adding a prior to the same observer. The Tractable
Continuous Trajectory Inference section above is the lighter way to make that
same change of class without the pixel prior; the Wu route adds the pixel target
and the natural-image prior on top of it.

The compact-subspace intervention composes with reconstruction: run it with
`compact_only`, `compact_removed`, and `compact_addback` response components and
ask whether reconstruction quality follows the same pattern as the feature
endpoint (compact-only near full, compact-removed toward zero-eye). So the
mechanism test survives in reconstruction form. Given the static-PC results, it
inherits the same control: any compact reconstruction benefit must be matched
against a static-PC version of the same intervention, because compact is largely
shared with the static image-response manifold and is not unique over it.

### What does not transfer (the boundaries)

```text
metric:        reconstruction quality (MS-SSIM / LPIPS / feature distance),
               NOT bits. This is a legibility/recovery claim, not an
               information-content claim about the twin response.
forward model: Wu inverts an explicit spike likelihood p(s|y); the twin emits
               deterministic rate maps, so the observation model p(y_obs|I,tau)
               must be defined on the twin's rates (Gaussian readout), and
               inversion runs through the twin rather than a fitted GLM.
prior leak:    a prior trained on natural images partly supplies the image
               information the active-sensing question is about. Acceptable for
               a reconstruction-quality framing; not acceptable if the result
               is then read back as an encoding/information claim.
compact:       a compact-aware reconstruction prior inherits the static-PC
               specificity caveat and must be controlled against a
               static-PC-aware prior, not just random/shuffle nulls.
```

The single load-bearing boundary: with a prior in the loop you are measuring how
well the scene can be reconstructed, not how much the response carries. Keep the
bits-axis claim in 4B and report Wu-style reconstruction on its own quality
axis.

### Decision rule

```text
If the question stays "how many bits about the image does motion / compact
structure add to the response":
  -> keep the feature decoder (4B bits axis; 4C feature-recovery endpoint).
  -> do NOT add an image prior; it contaminates the bits quantity.

If the question becomes "can the latent-eye scene be reconstructed, phase and
all, and does motion / compact structure help that reconstruction":
  -> Wu-style pixel + natural-image prior is the right tool.
  -> report reconstruction quality, not bits.
  -> this is the natural bridge to the compact-aware joint-prior arc
     (candidate second paper), and carries the same static-PC control.
```

Treat this as a separate-axis companion to 4C, or as the entry point to the
joint-prior paper, not as a replacement for the current feature-posterior
endpoint.

## Production Rerun Implications

The final Figure 4C panel should preserve four contracts:

```text
known-trace control above latent-eye recovery
compact-only near full joint recovery
compact-removed near zero-eye recovery
static-PC controls described as nearly matched, not just as minor guardrails
```

If the optional prior-depth target is run:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_n128_c4_k16_rel0p25_prior32_power_v1
```

then it should be treated as a power and prior-depth extension for the
trajectory-table observer. It should not replace the current 4C
feature-posterior compact-removal endpoint unless it includes the same compact
projection intervention and the same feature-space score.
