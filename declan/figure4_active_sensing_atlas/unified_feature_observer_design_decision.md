# Unified Feature Observer Design Decision

Date: 2026-07-05

Status: adopted communication/design decision for the draft1 Figure 4 rewrite.

Companion score decision:

```text
declan/figure4_active_sensing_atlas/unified_feature_observer_score_decision.md
```

## Decision

Do not present aggregate feature decoder, local image/trajectory pairing, and
joint feature-eye decoder as three separate main analyses. That makes Figure 4
read like an atlas of attempts.

Present one feature-recovery observer family with:

```text
one feature target
one feature-recovery score
one split/calibration policy
several eye-information conditions
```

The shared target is:

```text
f = Phi(I)
```

where `Phi(I)` is the fixed local image-feature target, such as block-pooled
pyramid or another predeclared V1-like feature vector. The observer produces:

```text
f_hat = g(r, E)
```

where `E` is the eye-information condition. All main conditions should be
scored on the same held-out feature-recovery axis:

```text
S(f_hat, f)
```

Critical separation:

```text
primary 4C observer: response movie -> feature estimate
compact branch: subspace intervention / mechanistic audit
```

The primary active-sensing consequence should not require the Figure 3 compact
basis or compact forward model to work. The current compact-forward
candidate-free branch is useful as a compact-dependent audit, but its tau
estimate is entangled with the compact observation model. If optimized tau fits
compact residuals better than recorded tau, that is evidence of model
misspecification or compact-coordinate compensation, not automatically evidence
for true latent-eye recovery.

Paper-facing 4C should therefore first use the least geometry-committed
observer possible:

```text
r_{1:T} -> f_hat
```

or, as a restrained dimensionality reduction:

```text
U_staticPC^T r_{1:T} -> f_hat
```

Only after this primary observer is defined should compact geometry enter as a
mechanistic intervention.

Preferred score for communication:

```text
S = R2_cv
```

where `R2_cv` is held-out normalized MSE in the locked, train-normalized feature
space. It is computed as pooled out-of-fold SSE/SST across samples and feature
dimensions, using each fold's train-mean baseline. Feature cosine is a
secondary robustness score, not the main figure axis. The current diagonal
information-like 4B score and Gaussian predictive likelihood can remain as
supplemental/theory analyses unless they are recomputed under the same target,
split, and covariance contract.

Source weighting:

```text
primary linear-Gaussian observer fits should be source-balanced
```

Candidate lists and trajectory catalogs reuse image sources unevenly. If the
estimand is source-uniform feature recovery, each training source should
contribute equal total weight within a fold/spec. Row-unweighted fits are
allowed only as reproducibility diagnostics and should be labeled as
candidate/trajectory-frequency weighted.

Axis-family limitation:

```text
pooled-prior response-feature observer != along/across test
```

If a runner fits one model pooled across prior-family labels and evaluates the
same observed response under both `axis_edge_parallel` and
`axis_edge_orthogonal` labels, it must not report parallel-minus-orthogonal
contrasts. Along/across claims require a dedicated per-axis fit/evaluation
contract.

## Trajectory Coordinate Contract

Before naming observer conditions, decompose each movie trajectory as:

```text
e_n,t = ebar_n + etilde_n,t
```

where `ebar_n` is the movie/window mean eye position and `etilde_n,t` is the
within-movie residual motion. For BackImage, the crop is extracted at the
window's mean fixation position, so the renderer's crop-centered coordinate is:

```text
u_n,t = e_n,t - ebar_n
```

Consequently, cached `tau = 0` / `zero_lambda_counts` is not a global
absolute-eye-position zero oracle. It is static at the movie/window's matched
mean position. Manuscript wording should therefore reserve `zero` for
observer assumptions and use explicit stimulus labels:

```text
static_global_zero       e_n,t = 0                  optional oracle/control
static_matched_mean      e_n,t = ebar_n             main BackImage static baseline
motion_mean_centered     e_n,t = etilde_n,t         BackImage motion rendering
motion_full              e_n,t = ebar_n + etilde_n,t if rendering in screen coordinates
zero_eye_on_motion       observer assumes no residual motion for a moving response
```

Legacy code/output names such as `zero_static` or `static_zero` should be
treated as aliases for `static_matched_mean` only when the cache reference mode
is `patch_center_static_tau_zero`.

## Main Conditions

### 1. Static Matched-Mean Baseline

```text
r = R(I at ebar_n, no residual motion)
```

Question:

```text
What image features are recoverable without temporal retinal motion, while
preserving each movie/window's mean retinal placement?
```

### 2. Motion-Rendered / Known-Trajectory Condition

```text
r = R(I, tau_true)
```

Two related labels must be kept separate:

```text
motion-rendered: trajectory used to generate the response
known-trajectory observer: trajectory explicitly available to the observer/readout
```

The observer either explicitly receives `tau_true`, or more weakly the response
has been rendered under the empirical trajectory and compared against the
stabilized baseline on the same score axis. Only the first case should be called
a known-trajectory observer.

This is the main 4B condition.

Safe wording if `tau_true` is not an explicit decoder input:

```text
motion-rendered feature recovery under empirical trajectories
```

Avoid:

```text
pose-aware decoder
```

unless the decoder/observer is actually given the trajectory or explicitly
marginalizes/inverts it.

### 3. Zero-Eye Or Trajectory-Hidden Baseline

These are different baselines and should be labeled separately:

```text
zero-eye: observer assumes no movement
trajectory-hidden / marginal: trajectory unavailable and must be ignored,
  marginalized, or inferred
```

In a zero-eye baseline, the observer receives motion-induced responses but
interprets them with a default no-movement assumption:

```text
g(r_motion, residual tau = 0)
```

Question:

```text
When motion is not accounted for, do the same FEM-driven response changes act
as nuisance variability?
```

### 4. Joint / Latent-Eye Condition

The observer is not handed the trajectory but can infer or marginalize over
latent trajectory structure:

```text
f_hat_joint = g_joint(r)
```

This is the main 4C condition only if it clears the promotion gate below.

## Panel Mapping

### 4B

4B becomes one of:

```text
known-trajectory feature recovery
motion-rendered feature recovery under empirical trajectories
```

Use the known-trajectory label only when the trajectory is explicitly supplied
to the observer/readout or the response cache is explicitly indexed by
trajectory for the readout. Otherwise use the motion-rendered label.

Main question:

```text
If retinal trajectory is accounted for, does drift-like motion improve recovery
of local image features relative to stabilization?
```

### 4C

4C is the same feature target and same score, but with trajectory hidden:

```text
known trajectory > zero-eye
joint latent-eye > zero-eye
```

If the baseline is not zero-eye but a marginal or no-trajectory-hidden observer,
call it hidden/marginal rather than zero-eye.

The clean summary metric is recovered fraction of the known-trajectory gap:

```text
gap_recovered = (S_joint - S_baseline) / (S_known - S_baseline)
```

Communication:

```text
How much of the trajectory-known feature benefit can be recovered when eye
trajectory is latent?
```

This is the useful-versus-nuisance bridge: FEM-driven variability is not
inherently noise or signal. Its role depends on whether the observer can account
for retinal motion.

## Local Pairing

Move local exact image/trajectory pairing out of the main Figure 4 logic.

It asks a stricter and different question:

```text
Are exact real image-trajectory pairings better than matched swaps?
```

Current corrected results support local motion-delta feature sensitivity, but
do not yet support a load-bearing exact image-trace pairing claim. Treat local
pairing as a supplemental specificity test.

## Compact Geometry

Compact geometry should enter as an intervention inside the same 4C observer,
not as a separate decoder branch:

```text
S_full_response
S_compact_only
S_compact_removed
S_static_PC_only
```

Use the same target, score, splits, and calibration:

```text
These are subspace interventions inside the same observer and are scored
against the same feature target with the same R2_cv definition.
```

The safe claim is:

```text
compact-only preserves much of latent-eye feature recovery, compact removal
collapses recovery toward zero-eye, but much of the structure overlaps the
ordinary static image-response manifold.
```

If that cannot be shown cleanly in one inset, keep compact in Figure 3 or in
the supplement.

Avoid making the latent-eye observer depend intrinsically on compact-response
coordinates. A compact-only observer can be shown as an intervention, but a
compact-forward observer should be labeled as compact-dependent because tau is
then inferred through the compact response model.

## Promotion Gates

### Minimum Promotable Figure 4

Promote the core useful-versus-nuisance distinction if:

```text
S_known_trajectory > S_stabilized
S_known_trajectory > S_zero
```

using `S = R2_cv` on the locked feature target.

Interpretation:

```text
Motion can be useful when accounted for, but becomes nuisance when trajectory
is hidden.
```

### Stronger Figure 4

Add the joint latent-eye condition to the main panel only if:

```text
S_joint > S_zero
```

under source-row/source-trial held-out calibration and the same `R2_cv` feature
score. Stronger wording should require:

```text
S_known > S_joint > S_baseline
gap_recovered =
  (S_joint - S_baseline) / (S_known - S_baseline)
  > 0
```

with uncertainty intervals. Report the ratio only when the known-minus-baseline
denominator is positive and meaningfully above uncertainty, and always report
the raw contrasts alongside it. If the ratio exceeds 1, report it directly
rather than capping it.

Interpretation:

```text
A latent-eye observer recovers part of the known-trajectory feature benefit.
```

### If Joint Remains Brittle

Do not introduce a third main decoder to compensate. Show:

```text
stabilized
known trajectory
zero-eye or hidden/marginal trajectory baseline
```

and treat trajectory recovery as a diagnostic/supplemental observer analysis.

If only the compact-forward branch works, do not promote it as the main 4C
latent-eye result until a geometry-uncommitted response-space or static-PC
observer also passes the same score gate. If the compact-forward branch fails,
that does not by itself kill the active-sensing hypothesis, because it may be a
failure of the compact observation model rather than of latent-eye feature
recovery.

## Replacement Main-Text Paragraph

```text
We used a single feature-recovery framework to compare eye-information
conditions. The visual target was a fixed local image-feature vector, and all
conditions were scored by held-out normalized MSE, reported as cross-validated
feature `R2_cv`. In the stabilized condition, the response was generated without
retinal motion. In the motion-rendered condition, responses were generated
under empirical drift-like trajectories. When the trajectory was explicitly
available to the readout, this became a known-trajectory observer and provided
an upper-bound estimate of how useful retinal motion can be when accounted for.
In the zero-eye or hidden-trajectory baseline, the same motion-induced response
changes were interpreted without the corresponding trajectory information, so
retinal motion contributed nuisance variability.
Finally, in the joint condition, the observer attempted to recover image
features while accounting for latent eye trajectory. This design makes the key
comparison within one observer family: the same FEM-driven response changes can
be useful when trajectory is known or inferred, but nuisance when trajectory is
hidden.
```

Supplement sentence:

```text
Exact image-trajectory pairing was tested separately as a specificity control
and is reported in the supplement.
```

## Immediate Rewrite Implications

- Stop calling 4B "the aggregate feature decoder" in main text.
- Stop introducing local pairing as a main proof.
- Stop introducing 4C as a separate decoder family.
- Use the shared `R2_cv` score axis wherever possible.
- If 4B remains on a diagonal-information axis, label it as secondary until a
  shared feature-recovery version exists.
- Do not make compact geometry necessary for 4C to work; compact belongs as
  compact-only, compact-removed, compact-addback, and static-PC interventions
  after the primary observer.
- Report 4C as recovered fraction of the known-trajectory gap when the joint
  observer is robust enough and the denominator is stable.
