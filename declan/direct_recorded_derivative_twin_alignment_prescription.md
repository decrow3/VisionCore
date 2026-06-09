# Direct Recorded Derivative / Twin Tangent Alignment Analysis

## Purpose

Build a bounded supplemental analysis asking whether eye-position derivatives estimated directly from recorded V1 point into the same translation geometry identified in the digital twin.

This is a strengthening analysis, not a new thesis. It should be treated as a direct recorded-data bridge to the compact tangent geometry, with explicit stop rules and conservative interpretation.

## Core Question

The current main result says:

> Finite-difference fitted-twin retinal-translation covariances predict a reliable component of recorded FEM covariance, and the controlled prediction is retained after restricting the fitted-twin source to a compact cross-fit k=10 tangent subspace.

This proposed analysis asks a more direct but noisier question:

> If we estimate eye-position response derivatives directly from recorded V1 repeats, do those recorded derivatives preferentially lie in the compact fitted-twin translation-tangent geometry?

## Important Guardrail

Do **not** make the primary claim:

> Each image has a clean recorded local derivative manifold that matches the twin's signed horizontal and vertical tangent axes.

That claim was historically fragile. The older STG analyses found reliable aggregate eye-linked structure, but did not robustly recover clean image-specific derivative manifolds or single-window trajectory geometry from recorded data.

The primary claim should instead be:

> Directly estimated recorded eye-position sensitivity is enriched in the compact fitted-twin translation-tangent subspace, above appropriate unit, context, and RF/readout-preserving nulls.

If this succeeds, it strengthens the paper by showing that the compact tangent geometry is visible not only through covariance closure, but also through a direct recorded eye-sensitivity object.

If it fails, it should be interpreted as a support/reliability limitation of direct derivative estimation, not as a failure of the main covariance result.

## Relationship To Existing Analyses

This analysis sits between two existing branches:

1. **Old STG recorded derivative branch**
   - Estimated recorded eye-sensitivity maps such as `B_emp`.
   - Found reliable aggregate eye-linked geometry.
   - Found weaker and session-dependent model-to-recording alignment.
   - Did not robustly support clean image-specific local derivative manifolds.

2. **Current finite-difference covariance closure**
   - Uses fitted-twin finite-difference predictions `delta_r_i = J_i e_i`.
   - Forms source covariance and compares it to recorded `Sigma_FEM`.
   - Shows robust 24/24 session covariance capture above unit-shuffle.
   - Compact k=10 restricted source retains essentially the full controlled effect.

The new analysis should reuse the stronger matched-unit and finite-difference infrastructure from the current closure pipeline, while borrowing the recorded derivative idea from STG.

## Analysis Tiers

Run the tiers in order. Tier 1 is the primary analysis. Tiers 2 and 3 are optional/diagnostic and should not be allowed to derail the paper.

### Tier 1: Recorded Derivatives In Compact Twin Basis

Primary question:

> Do recorded eye-position derivatives lie in the compact fitted-twin tangent subspace?

For each recorded context `c`, estimate a recorded eye-position derivative matrix:

```text
B_rec,c : [n_units, 2]
```

Then measure how much of this recorded derivative energy lies in a cross-fit compact twin tangent basis:

```text
capture_rec,c(k) = || U_twin,k.T @ B_rec,c ||_F^2 / || B_rec,c ||_F^2
```

where:

```text
U_twin,k : [n_units, k]
```

is learned from fitted-twin finite-difference tangents using held-out image/trial/context separation.

This tier does not require exact signed `x/y` axis matching. It asks only whether recorded eye sensitivity occupies the same compact translation coordinate system.

### Tier 2: Context-Matched Recorded/Twin Derivative Alignment

Secondary question:

> For matched contexts, does the recorded derivative subspace align with the fitted-twin finite-difference derivative subspace above shuffled controls?

For each context:

```text
B_rec,c   : recorded eye-position derivative, [n_units, 2]
J_twin,c  : fitted-twin finite-difference Jacobian, [n_units, 2]
```

Compute orthonormal bases:

```text
Q_rec,c  = orth(B_rec,c)
Q_twin,c = orth(J_twin,c)
```

Primary metric:

```text
subspace_overlap_c = 0.5 * || Q_rec,c.T @ Q_twin,c ||_F^2
```

Compare matched context alignment to:

- image/context-shuffled twin tangents,
- unit-shuffled twin tangents,
- RF/readout-preserving constrained nulls,
- random 2D subspaces.

This tier is more biologically direct, but more fragile than Tier 1.

### Tier 3: Signed Axis Diagnostics

Diagnostic-only question:

> Do recorded `b_x` and `b_y` align with twin `b_x` and `b_y` in the expected signed or axis-selective way?

Metrics:

```text
cos_xx = cos(B_rec[:, x], J_twin[:, x])
cos_xy = cos(B_rec[:, x], J_twin[:, y])
cos_yx = cos(B_rec[:, y], J_twin[:, x])
cos_yy = cos(B_rec[:, y], J_twin[:, y])
```

or axis selectivity:

```text
axis_selectivity_x = |cos_xx| - |cos_xy|
axis_selectivity_y = |cos_yy| - |cos_yx|
```

This should never be the headline metric. Coordinate sign, eye calibration, axis rotation, torsion, and finite-cloud nonlinearities can make signed-axis recovery brittle even when the subspace result is real.

## Inputs

Use the current matched recorded/twin finite-difference closure infrastructure where possible.

Required:

- recorded responses in matched recorded/twin unit space,
- measured eye positions or eye offsets for each sample/trial,
- context labels: image/time/window or equivalent,
- matched unit mask,
- recorded FEM covariance target, if using closure-style comparisons,
- fitted-twin finite-difference Jacobians in the same matched unit order,
- projection-control code from Panel F closure.

Preferred source caches:

- Fig2 recorded covariance decomposition cache,
- Fig3 fitted-twin matched-unit cache,
- finite-difference closure outputs,
- compact tangent basis outputs, if already cached.

Important:

The unit order must be identical for:

```text
recorded responses
B_rec
J_twin
U_twin,k
Sigma_FEM
```

Fail loudly if this cannot be verified.

## Context Definition

Define a context `c` as narrowly as the data support:

Preferred:

```text
same image identity + same time/window index + same stimulus-history bin
```

Acceptable fallback:

```text
same image/time window used in the recorded covariance decomposition
```

Avoid contexts that mix different lagged stimulus histories unless this is unavoidable and explicitly recorded.

For every context, record:

- session,
- image ID or image/window key,
- time/window index,
- number of trials/samples,
- eye-position covariance,
- eye-position range,
- response reliability,
- whether context was retained or rejected.

## Recorded Derivative Estimation

For each session and context:

```text
R_c : [n_samples, n_units]
X_c : [n_samples, 2]
```

where `R_c` is recorded response and `X_c` is centered eye position or eye offset.

Model:

```text
R_c = 1 mu_c.T + X_c B_rec,c.T + epsilon
```

Fit:

```text
B_rec,c.T = (X_c.T X_c + lambda I)^(-1) X_c.T (R_c - mean(R_c))
```

Equivalent shape:

```text
B_rec,c = (R_c - mean(R_c)).T @ X_c @ inv(X_c.T @ X_c + lambda I)
```

Use ridge regression. Do not fit unregularized OLS unless context support is very high and the eye-position design is well-conditioned.

### Ridge Parameter

Use one of these two approaches:

1. **Fixed ridge grid chosen globally before alignment**
   - Select `lambda` by recorded split-half derivative reliability, not by twin alignment.

2. **Nested within-session CV**
   - Choose `lambda` to predict held-out recorded responses from eye position.
   - Do not choose `lambda` to maximize alignment with the twin.

Recommended initial grid:

```text
lambda_grid = [0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0] * trace(X.T @ X) / 2
```

Record the selected value for every session/context.

## Reliability Gates

Do not analyze every context blindly. Direct derivative estimation is support-limited.

Minimum gates:

```text
min_samples_per_context = 20 or higher if available
min_eye_rank = 2
min_eye_cov_eig_small > small_threshold
max_eye_design_condition_number = e.g. 100
min_response_valid_units = 50
```

Derivative reliability:

Split each context into two halves, fit:

```text
B_rec,c,A
B_rec,c,B
```

Compute:

```text
split_half_overlap_c = 0.5 * || orth(B_A).T @ orth(B_B) ||_F^2
```

or vector correlation of flattened derivatives.

Compare to an eye-label shuffle null within context.

Recommended gates:

- retain all contexts for a transparent full analysis,
- define a reliability-qualified subset for the primary inferential panel,
- report both.

Avoid hidden selection. If using a reliability-qualified subset, define the threshold before testing twin alignment.

## Projection Controls

Repeat the main metrics under the same controls used in Panel F:

- `none`
- `global-rate`
- `target PC1`
- `global-rate + target PC1`

Projection must be applied consistently to recorded derivatives, twin tangents, compact bases, and nulls.

For derivative matrices:

```text
B_projected = P @ B
```

For compact bases:

```text
U_projected = orth(P @ U)
```

For covariance targets, use existing Panel F projection code.

Primary conservative condition:

```text
global-rate + target PC1 removed
```

## Compact Twin Basis Construction

For Tier 1, construct `U_twin,k` from fitted-twin finite-difference tangents in matched recorded/twin unit space.

Tangent stack:

```text
B_twin_train = [J_x(context_1), J_y(context_1), ..., J_x(context_N), J_y(context_N)]
```

Compute:

```text
U_twin,k = top k eigenvectors of B_twin_train @ B_twin_train.T
```

Use cross-fitting:

- If image IDs are available: image-disjoint train/test folds.
- If only trial IDs are available: trial-disjoint folds, and record this limitation.
- If neither is available: use session-level or context-level split, but label it clearly.

Primary `k` values:

```text
k = 2, 5, 10, 20
```

Headline should include `k=10` for continuity with Panel F compact restricted analysis, but show a k-sweep.

## Tier 1 Metrics

For each retained recorded context:

```text
capture_rec,c(k) = || U_twin,k.T @ B_rec,c ||_F^2 / || B_rec,c ||_F^2
```

Session summary:

- mean context capture,
- median context capture,
- reliability-weighted mean capture,
- excess over null,
- bootstrap CI over contexts within session,
- session-level effect for cross-session inference.

Recommended reliability weighting:

```text
weight_c = max(0, split_half_overlap_c - median(shuffle_overlap_c))
```

Also report unweighted summaries to avoid overengineering.

## Tier 1 Nulls

Use multiple nulls, in increasing relevance.

### Null 1: Random k-dimensional subspace

Draw random orthonormal bases in matched unit space.

Purpose:

- sanity check,
- not sufficient alone.

### Null 2: Unconstrained unit shuffle of `U_twin,k`

Permute rows of `U_twin,k`.

Purpose:

- continuity with existing unit-shuffle logic.

### Null 3: Image/context-shuffled compact basis

Learn `U_twin,k` from unrelated contexts or incorrect image folds where possible.

Purpose:

- tests content specificity.

### Null 4: RF/readout-preserving constrained row permutation

Permute units only within RF/readout bins, using the same binning machinery as the RF/readout-preserving covariance-closure null.

Purpose:

- tests whether compact-basis capture exceeds generic retinotopic/readout layout.

This is the most reviewer-facing null if available.

## Tier 2 Metrics

For matched context `c`:

```text
Q_rec,c  = orth(B_rec,c)
Q_twin,c = orth(J_twin,c)
overlap_c = 0.5 * || Q_rec,c.T @ Q_twin,c ||_F^2
```

If derivative rank is effectively one due to eye-position spread, report a rank-1 version:

```text
overlap1_c = |q_rec1.T @ q_twin1|^2
```

Compare observed matched overlap to:

- context-shuffled twin `J_twin`,
- image-shuffled twin `J_twin`,
- unit-shuffled twin `J_twin`,
- RF/readout-preserving constrained unit null.

Primary effect:

```text
effect_c = overlap_matched_c - median(overlap_null_c)
```

Session effect:

```text
effect_session = mean or median over reliability-qualified contexts
```

Cross-session inference:

- bootstrap over sessions,
- sign test across sessions.

## Tier 3 Metrics

Signed-axis diagnostics:

```text
cos_xx = corr_or_cos(B_rec[:, 0], J_twin[:, 0])
cos_xy = corr_or_cos(B_rec[:, 0], J_twin[:, 1])
cos_yx = corr_or_cos(B_rec[:, 1], J_twin[:, 0])
cos_yy = corr_or_cos(B_rec[:, 1], J_twin[:, 1])
```

Axis-selective diagnostic:

```text
axis_x = |cos_xx| - |cos_xy|
axis_y = |cos_yy| - |cos_yx|
```

Report as supplemental only. If it fails but Tier 1 succeeds, the analysis still supports compact derivative enrichment.

## RF/Readout-Preserving Null Integration

Reuse the RF/readout-preserving null prescription:

```text
rf_readout_preserving_null_prescription.md
```

For this analysis, the constrained permutation should preserve:

- session,
- matched unit set,
- RF/readout x/y bins,
- tangent norm or derivative norm,
- mean response scale,
- model quality if available.

Apply the constrained null to:

1. `U_twin,k` row identity for Tier 1,
2. `J_twin,c` row identity for Tier 2,
3. signed axes for Tier 3 diagnostics.

Use fixed within-bin permutations for source-target alignment metrics. Samplewise permutations are only needed for compactness-spectrum tests, not for derivative capture.

## Handling Coordinate Conventions

Be explicit about eye-coordinate conventions.

Record:

- whether eye x/y are in degrees, pixels, or arcmin,
- whether y-axis is screen-up or image-row-down,
- whether offsets are relative to trial mean, context mean, or fixation center,
- finite-difference sign convention for image shifts.

For Tier 1, exact signed convention matters less because the metric uses subspace capture.

For Tier 2, subspace overlap is invariant to 2D rotation/sign flips within the derivative plane.

For Tier 3, signed-axis interpretation requires verified coordinate convention. If not verified, label Tier 3 as unsigned diagnostic only.

## Outputs

Write to:

```text
outputs/direct_recorded_derivative_twin_alignment/
```

Required files:

```text
recorded_derivative_manifest.json
context_inventory.csv
recorded_derivative_reliability.csv
tier1_compact_basis_capture.csv
tier1_compact_basis_bootstrap_summary.csv
tier2_matched_derivative_alignment.csv
tier2_matched_derivative_bootstrap_summary.csv
tier3_signed_axis_diagnostics.csv
null_summary.csv
audit.json
README.md
```

Optional figures:

```text
figures/tier1_compact_capture_by_k.png
figures/tier1_compact_capture_by_k.pdf
figures/tier1_session_effects.png
figures/tier2_matched_vs_context_shuffle.png
figures/derivative_reliability_vs_alignment.png
figures/context_support_diagnostics.png
```

## Manifest Requirements

The manifest must include:

- code commit/version,
- input cache paths,
- model checkpoint/config,
- sessions included,
- matched unit counts,
- response time window,
- eye-position coordinate convention,
- context definition,
- minimum context support gates,
- ridge grid and selection rule,
- projection controls,
- compact basis construction rule,
- cross-fit grouping mode,
- null types and null draw counts,
- random seeds,
- whether RF/readout metadata were available.

## Audit Checks

Fail if:

- unit order cannot be verified across recorded, twin, and target objects,
- fewer than a predeclared number of sessions have enough contexts,
- `B_rec` is estimated without centering eye position,
- ridge selection uses twin alignment as an objective,
- compact basis train/test split leaks the same image/context into both train and test when image-disjoint labels are available,
- projection controls are applied inconsistently.

Warn if:

- many retained contexts are rank-1 in eye position,
- `B_rec` split-half reliability is near shuffle in most contexts,
- results are positive only without projection controls,
- results are positive only for PSD target-dependent contexts,
- Tier 2 signed axes fail while Tier 1 succeeds,
- RF/readout metadata are missing.

## Statistical Inference

Primary inference should be session-level.

For each session:

```text
effect_session = mean_context_capture_observed - median_context_capture_null
```

Then report across sessions:

- mean effect,
- median effect,
- bootstrap 95% CI over sessions,
- sign count,
- exact sign-test p-value.

Within-session context bootstraps are useful diagnostics but should not be the headline if contexts are not independent.

## Recommended Stop Rules

Use these stop rules to avoid analysis sprawl.

### Continue / promote to supplement if:

- Tier 1 is positive over unit-shuffle and random-basis nulls,
- preferably remains positive over RF/readout-preserving null,
- effect appears in most sessions,
- effect survives `global-rate + target PC1` projection,
- k-sweep is monotonic/plausible rather than cherry-picked,
- recorded derivative reliability is above shuffle.

### Keep as diagnostic only if:

- Tier 1 is positive only in low-control settings,
- Tier 2 is weak or session-specific,
- effect depends heavily on reliability threshold,
- RF/readout-preserving null absorbs most of the effect.

### Drop from manuscript if:

- recorded derivative reliability is indistinguishable from eye-shuffle,
- Tier 1 is null across sessions,
- positive result requires post hoc context selection,
- unit-order or coordinate-convention audits fail.

## Possible Figure

### Panel SxA: Recorded derivative reliability

Show split-half recorded derivative reliability vs eye-shuffle null.

Takeaway:

```text
Direct recorded eye-position derivatives are measurable in reliability-qualified contexts.
```

### Panel SxB: Compact twin basis capture

Plot `capture_rec(k)` for observed compact twin basis vs random, unit-shuffle, and RF/readout-preserving null.

Takeaway:

```text
Recorded derivatives preferentially occupy the compact twin translation subspace.
```

### Panel SxC: Session effects

Show session-level excess capture under the conservative projection control.

Takeaway:

```text
The effect is not driven by one context or one session.
```

### Panel SxD: Matched derivative alignment

Optional. Show matched recorded/twin subspace overlap vs context-shuffled twin tangents.

Takeaway:

```text
Context-matched derivative alignment is supportive but treated as secondary.
```

## Caption Language

If Tier 1 succeeds:

> We estimated eye-position response derivatives directly from repeated recorded V1 responses and asked whether these recorded derivatives occupied the compact fitted-twin translation-tangent subspace. A cross-fit compact basis learned from fitted-twin finite-difference tangents captured recorded derivative energy above random, unit-shuffled, and RF/readout-preserving nulls. Thus, the compact translation geometry identified in the twin is visible in a direct recorded eye-sensitivity object, not only in covariance closure.

If Tier 1 succeeds but Tier 2 is weak:

> Exact context-matched signed derivative alignment was weaker and is treated as diagnostic, consistent with limited per-context support and finite-cloud nonlinearities in the recordings.

If Tier 1 fails:

> Direct per-context recorded derivative estimates were not reliable enough to test image-specific tangent alignment. This negative diagnostic does not contradict the aggregate covariance closure result, which is estimated at a more reliable second-moment scale.

## Manuscript Interpretation

Preferred successful wording:

> As a direct recorded-data check, we estimated eye-position derivatives from repeated V1 responses. Although exact image-specific signed tangent recovery is support-limited, the reliable component of recorded eye sensitivity was enriched in the compact fitted-twin translation basis. This supports the interpretation that the recorded FEM covariance bridge reflects a shared reafferent translation geometry rather than only a model-side construction.

Preferred cautious wording:

> Direct derivative alignment was weaker than covariance closure, consistent with the limited support available for per-context eye-position slopes. We therefore treat the direct derivative analysis as supportive/diagnostic and rely on covariance closure as the primary recorded-data bridge.

## Things Not To Claim

- Do not claim every image has a clean recorded tangent plane.
- Do not claim exact signed horizontal/vertical derivative recovery unless Tier 3 is very strong and coordinate conventions are audited.
- Do not claim a null result disproves reafferent covariance.
- Do not select contexts based on twin alignment.
- Do not let this analysis supersede the main finite-difference covariance closure unless it is unexpectedly very strong.

## Implementation Sketch

```python
for session in sessions:
    data = load_matched_recorded_twin_data(session)
    contexts = build_contexts(data)
    unit_metadata = load_unit_metadata(session)

    for context in contexts:
        R = context.recorded_responses      # [samples, units]
        X = context.eye_offsets             # [samples, 2]

        if not passes_support_gates(R, X):
            mark_rejected(context)
            continue

        B_rec, reliability = fit_crossfit_recorded_derivative(
            R, X, ridge_grid=ridge_grid
        )

        J_twin = load_or_compute_context_twin_jacobian(context)

        save_context_derivatives(context, B_rec, J_twin, reliability)

    folds = make_crossfit_folds(contexts, mode="image_or_trial_disjoint")

    for fold in folds:
        U_twin_by_k = learn_compact_twin_basis(
            J_twin_train=fold.train.J_twin,
            projection_control=projection,
            k_list=[2, 5, 10, 20],
        )

        for context in fold.test:
            B = project_derivative(context.B_rec, projection)

            for k, U in U_twin_by_k.items():
                capture_obs = frob_capture(B, U)

                for null in nulls:
                    U_nulls = make_null_bases(U, null, unit_metadata)
                    capture_null = [frob_capture(B, Un) for Un in U_nulls]

                    save_tier1_metric(
                        session=session,
                        context=context.id,
                        k=k,
                        projection=projection,
                        capture_obs=capture_obs,
                        capture_null_median=np.median(capture_null),
                    )

            # Tier 2 optional
            overlap_obs = subspace_overlap(context.B_rec, context.J_twin)
            overlap_null = context_shuffle_or_unit_null_overlap(...)
            save_tier2_metric(...)

aggregate_session_level_effects()
write_manifest_and_audit()
```

