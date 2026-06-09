# RF/Readout-Preserving Null for Finite-Difference Covariance Closure

## Purpose

Build a no-retraining reviewer-facing null analysis for the finite-difference covariance closure result. The analysis asks whether the observed compact tangent spectrum and recorded-covariance capture require the trained model's precise unit-by-unit translation geometry, or whether they can be explained by generic smooth retinotopic/readout structure.

The null should preserve broad unit properties such as retinotopic/readout location, response scale, finite-difference tangent norm, and optionally model quality, while breaking the exact mapping between each recorded unit and its fitted-twin translation response.

## Scientific Question

The current positive result is:

> Fitted-twin finite-difference retinal-translation responses predict a reliable component of recorded FEM covariance in matched recorded/twin unit space.

A skeptical reviewer may ask:

> Would any smooth retinotopic population with similar RF/readout layout produce a compact translation-tangent family and above-null covariance capture?

This null addresses that question without retraining a new digital twin.

## Required Interpretation Guardrail

This is a control analysis, not a new main discovery branch. The preferred final interpretation is one of:

1. **Observed beats constrained null.**
   The recorded covariance bridge depends on the fitted model's specific unit-level translation geometry beyond generic retinotopic/readout organization.

2. **Observed matches constrained null.**
   The compactness/closure is largely explained by generic retinotopic smoothness and matched unit layout. This does not invalidate reafference, but it changes the claim from "learned V1 feature geometry" toward "retinotopic translation geometry expressed in V1-like readouts."

3. **Mixed result.**
   Generic retinotopic structure explains part of the effect, but the fitted twin retains excess capture under the strongest controls.

Do not frame this null as a test of whether FEM covariance is real. That is established by the recorded-data decomposition. This null tests the specificity of the model-derived geometric bridge.

## Existing Observed Analysis

Use the same matched-unit finite-difference closure pipeline as the current Panel F result.

For each valid sample/window `i`:

```text
J_i          = fitted-twin finite-difference translation Jacobian, shape [n_units, 2]
e_i          = measured eye offset or displacement, shape [2]
delta_r_i    = J_i e_i
Sigma_src    = cov_i(delta_r_i)
Sigma_target = recorded FEM covariance in matched recorded/twin unit space
```

The current source variant of greatest interest is:

```text
fd_sample_eye_trace_cov
```

The compact restricted variant is:

```text
fd_sample_eye_trace_xfit_compact_k10_cov
```

The primary metric is covariance capture over unit-shuffle:

```text
U_src,k = top k eigenvectors of Sigma_src
capture = tr(U_src,k.T @ Sigma_target @ U_src,k) / tr(Sigma_target)
effect = capture - median(capture_unit_shuffle)
```

Use the same projection controls already in the finite-difference closure sweep:

- `none`
- `global-rate`
- `target PC1`
- `global-rate + target PC1`

Use both raw and PSD target variants, with PSD as the headline only if raw survives and is shown in supplement.

## Null Principle

A naive unit shuffle destroys too much structure. It breaks unit identity, but it also breaks RF/readout layout and may create an easy null.

The constrained null should instead preserve:

- matched unit count,
- session identity,
- retinotopic/readout location bins,
- response scale or mean rate,
- finite-difference tangent norm,
- optionally model quality such as `ccnorm` or `ccmax`,
- optionally baseline covariance/FEM strength strata.

It should break:

- exact unit-by-unit pairing between recorded FEM covariance and fitted-twin translation tangents,
- exact cross-image/sample identity of each unit's tangent response,
- learned feature-specific alignment between units across the population.

## Recommended Primary Null

### RF/readout-bin constrained row permutation

For each session, assign units to exchangeability bins based on available unit metadata.

Candidate bin features, in priority order:

1. readout or RF center x coordinate,
2. readout or RF center y coordinate,
3. finite-difference tangent norm,
4. mean predicted rate or response variance,
5. model quality metric, e.g. `ccnorm` or `ccmax`.

Within each bin, permute unit rows of the finite-difference predicted increments before forming the source covariance.

For each sample:

```text
delta_r_i = J_i e_i
delta_r_i_null[u] = delta_r_i[perm_bin_i(u)]
Sigma_src_null = cov_i(delta_r_i_null)
```

Important: use a null that breaks exact unit identity. A single fixed global row permutation across all units may preserve too much eigenspectrum and may be equivalent to a trivial relabeling for compactness. For recorded-covariance capture, a fixed row permutation is valid as a constrained source-to-target mismatch null, but for compactness it is not sufficient.

Therefore compute two related variants:

1. **Fixed constrained row permutation**
   - One within-bin unit permutation per null draw, applied consistently across all samples.
   - Best for Panel F recorded-covariance capture because it preserves the source covariance structure while breaking source-target unit identity within RF/readout strata.

2. **Samplewise constrained row permutation**
   - A within-bin unit permutation is drawn independently per sample, or per image/window fold.
   - Best for tangent-family compactness because it disrupts cross-sample unit-specific tangent consistency while preserving local unit metadata.

Do not use samplewise permutation as the only Panel F null, because it may be too destructive for a covariance-source control. Use it as a stronger diagnostic.

## Required Null Variants

Run these in increasing order of stringency.

### Null A: Existing unconstrained unit shuffle

This is the current reference null. Keep it for continuity.

Expected role:

- easiest null,
- useful baseline,
- not sufficient for reviewer concern.

### Null B: RF/readout-bin fixed permutation

Primary reviewer-facing constrained null for recorded covariance capture.

Preserves:

- session,
- RF/readout bin,
- broad response/tangent scale strata.

Breaks:

- exact unit correspondence between fitted-twin translation source and recorded FEM covariance target.

### Null C: RF/readout-bin samplewise permutation

Strong diagnostic for compact tangent spectrum and cross-image consistency.

Preserves:

- per-sample distribution of predicted increments within metadata bins.

Breaks:

- stable unit identity across image/history samples.

### Null D: Norm-matched random rotations within bins

Optional. Within each metadata bin, replace predicted increment vectors by random orthogonal mixtures among units in that bin.

For each bin `b`:

```text
D_b = delta_r[:, units_in_bin_b]
Q_b = random orthogonal matrix, shape [n_b, n_b]
D_b_null = D_b @ Q_b
```

This preserves within-bin total variance exactly but destroys unit identity and feature-specific alignment.

Use only if bin sizes are large enough.

## Binning Strategy

Use adaptive bins to avoid empty or tiny strata.

Recommended default:

1. Start with 2D readout/RF location bins.
   - Example: quantile-bin x into 3 bins and y into 3 bins.
   - This gives up to 9 spatial bins.

2. Within each spatial bin, split by tangent norm into 2 quantile bins if there are enough units.

3. Optionally split by mean rate or model quality if there are still enough units.

Minimum bin size:

```text
min_bin_units = 6
```

If a proposed split creates bins smaller than `min_bin_units`, merge back to the parent bin.

Record every bin's:

- number of units,
- x/y center range,
- tangent norm range,
- mean rate range,
- model quality range if used.

## Metadata Sources

Use whatever is already available in the matched finite-difference closure cache.

Likely sources:

- `neuron_mask`,
- matched Fig3 included units,
- readout grid positions,
- unit metadata from Fig3 digital twin cache,
- model performance fields such as `ccnorm` or `ccmax`,
- finite-difference tangent norm computed directly from `J_i`.

If explicit RF/readout centers are unavailable, use a fallback hierarchy:

1. model readout spatial position if present,
2. fitted RF center estimated from stimulus gradients/effective RF if cached,
3. unit metadata from model training cache,
4. tangent-profile summary bins only.

If no spatial metadata can be recovered, do not call the null RF-preserving. Call it a response/tangent-norm constrained unit null.

## Metrics To Report

### Panel F closure metrics

For each session, target variant, projection control, source variant, and `k`:

- observed capture,
- unconstrained unit-shuffle median capture,
- RF/readout-bin fixed-permutation median capture,
- observed excess over unconstrained unit shuffle,
- observed excess over RF/readout constrained null,
- bootstrap CI across sessions,
- sign-test count and p-value.

Primary row:

```text
source = fd_sample_eye_trace_cov
target = PSD and raw
k = 2
projection = global-rate + target PC1
```

Also report the compact restricted source:

```text
source = fd_sample_eye_trace_xfit_compact_k10_cov
```

### Compact tangent spectrum metrics

For the tangent union matrix:

```text
B = [b_x(I_1), b_y(I_1), ..., b_x(I_N), b_y(I_N)]
```

Report:

- observed cumulative variance curve,
- observed participation ratio,
- unconstrained unit-shuffle null curve,
- RF/readout-bin samplewise-permutation null curve,
- optional within-bin rotation null curve.

Important:

For spectrum/compactness, a fixed row permutation is not a valid null because eigenvalues of `B B.T` are unchanged by row permutation. Use samplewise or imagewise constrained permutations.

### Cross-fit compact basis metrics

If feasible, repeat the compact k=10 restricted closure with constrained nulls:

- learn compact basis on training samples/trials,
- project held-out `J_i e_i` through compact basis,
- form restricted covariance,
- compare recorded capture against constrained fixed-permutation null.

This directly tests whether the current "compact/full ratio = 1.01x" survives a stronger unit-layout-aware null.

## Suggested Output Files

Write to:

```text
outputs/matched_twin_covariance_closure_rf_null/
```

Required files:

```text
rf_null_manifest.json
rf_null_unit_bins.csv
rf_null_capture_metrics.csv
rf_null_bootstrap_summary.csv
rf_null_spectrum_summary.csv
rf_null_compact_k10_summary.csv
rf_null_audit.json
README.md
```

Optional figures:

```text
figures/rf_null_panelF_capture.png
figures/rf_null_panelF_capture.pdf
figures/rf_null_tangent_spectrum.png
figures/rf_null_bin_diagnostics.png
```

## Manifest Requirements

The manifest must record:

- git commit or code version,
- input cache paths,
- model checkpoint/config,
- sessions included,
- matched unit counts per session,
- finite-difference step,
- device,
- random seed,
- number of null draws,
- binning features used,
- minimum bin size,
- target covariance variant,
- projection controls,
- source variants,
- whether compact cross-fit grouping used `trial_inds`, image ID, or another key.

## Audit Checks

The run should fail loudly if:

- fewer than 20 of 24 sessions are valid for the main analysis,
- any session has fewer than 50 matched units after filtering,
- a constrained-null bin has fewer than `min_bin_units` units after merging,
- observed and null source covariances have inconsistent units/order,
- target raw/PSD trace denominators are missing,
- compact cross-fit grouping is not recorded,
- random seeds are absent.

The run should warn if:

- RF/readout metadata are unavailable and fallback bins are used,
- more than 30% of units fall into a single bin,
- constrained null and observed source spectra are nearly identical under the samplewise null,
- raw target results disagree in sign with PSD target results.

## Statistical Summary

Use session-level inference for the main claim.

For each metric:

```text
effect_s = observed_s - median(null_s)
```

Across sessions:

- mean effect,
- bootstrap 95% CI across sessions,
- sign count,
- exact sign-test p-value,
- median effect as robustness summary.

Do not bootstrap samples within a session as the headline. Samples are not independent in the same way sessions are.

## Success Criteria

The strongest positive outcome is:

```text
global-rate + target PC1, PSD target, k=2:
observed capture > RF/readout-bin fixed-permutation null in most or all sessions
bootstrap CI for excess excludes zero
compact k=10 restricted source remains comparable to full FD source
raw target shows same sign
```

A good enough reviewer-facing result is:

```text
observed capture remains positive over constrained null in a majority of sessions,
the session-level bootstrap CI is positive or near-positive,
and the constrained null explains only part of the original unit-shuffle effect.
```

A negative result is still interpretable:

```text
RF/readout constrained null matches observed capture.
```

In that case, the paper should say:

> The recorded covariance bridge is consistent with retinotopically organized retinal-translation geometry. We cannot distinguish from these data whether the compactness reflects feature-specific learned V1 geometry beyond generic smooth RF/readout organization.

## Pseudocode

```python
for session in sessions:
    data = load_fd_closure_cache(session)
    J = data["J"]                         # [n_samples, n_units, 2]
    eye = data["eye_offset"]              # [n_samples, 2]
    target = data["Sigma_FEM_recorded"]   # [n_units, n_units]
    metadata = load_unit_metadata(session)

    delta = np.einsum("sud,sd->su", J, eye)
    Sigma_src_obs = cov_rows(delta)

    bins = make_adaptive_unit_bins(
        metadata,
        features=["rf_x", "rf_y", "tangent_norm", "mean_rate", "ccnorm"],
        min_bin_units=6,
    )

    for null_draw in range(n_nulls):
        perm_fixed = constrained_permutation(bins, rng)
        delta_fixed = delta[:, perm_fixed]
        Sigma_src_fixed = cov_rows(delta_fixed)

        delta_samplewise = np.empty_like(delta)
        for s in range(delta.shape[0]):
            perm_s = constrained_permutation(bins, rng)
            delta_samplewise[s] = delta[s, perm_s]
        Sigma_src_samplewise = cov_rows(delta_samplewise)

        for projection in projections:
            target_p, src_obs_p = apply_projection(target, Sigma_src_obs, projection)
            _, src_fixed_p = apply_projection(target, Sigma_src_fixed, projection)

            capture_obs = covariance_capture(src_obs_p, target_p, k=2)
            capture_fixed = covariance_capture(src_fixed_p, target_p, k=2)

            save_metric(session, projection, null_draw, capture_obs, capture_fixed)
```

## Recommended Figure/Caption Language

Figure title:

```text
Translation-covariance capture exceeds RF/readout-preserving null
```

Caption language:

> To test whether the fitted-twin covariance bridge could be explained by generic retinotopic/readout organization, we generated constrained null sources by permuting finite-difference translation responses only among units with similar readout/RF location and response scale. This preserves broad retinotopic layout while breaking exact unit-level correspondence between fitted-twin translation geometry and recorded FEM covariance. Observed capture was compared with both the original unconstrained unit-shuffle null and this RF/readout-preserving null under the same projection controls.

If positive:

> The fitted-twin translation source remained above the RF/readout-preserving null, indicating that the recorded covariance bridge depends on more than generic smooth retinotopic organization.

If mixed:

> The constrained null reduced the excess capture, indicating that retinotopic/readout organization explains part of the bridge, while residual positive capture suggests feature-specific unit geometry may also contribute.

If negative:

> The constrained null matched observed capture, indicating that the current analysis supports a retinotopic reafferent-geometry interpretation but does not isolate feature-specific learned geometry beyond RF/readout layout.

## Things Not To Do

- Do not retrain a new model for this analysis.
- Do not use only an unconstrained unit shuffle.
- Do not use a fixed row permutation as a compactness-spectrum null.
- Do not hide the raw target result if PSD clipping is used.
- Do not describe a positive constrained-null result as proving the animal uses this geometry.
- Do not let this control become a second main story unless it cleanly resolves the reviewer concern.

