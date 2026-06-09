# Direct Recorded Derivative / Twin Tangent Alignment

This folder is for the bounded supplemental analysis described in
`declan/direct_recorded_derivative_twin_alignment_prescription.md`.

The analysis asks whether eye-position derivatives estimated directly from
recorded V1 repeats are enriched in the compact fitted-twin translation
geometry. It should only be promoted if it survives the controls listed below.

## Primary Claim Gate

Promote only if Tier 1 is positive across sessions after the conservative
projection control:

```text
global-rate + target PC1 removed
```

and remains positive over:

- random subspace null,
- unconstrained unit-shuffle null,
- RF/readout-preserving fixed within-bin permutation null.

Do not promote this as clean image-specific signed x/y tangent recovery. The
older STG analyses showed that signed/context-specific derivative recovery is
fragile. The safer question is whether the reliable component of recorded
eye-position sensitivity occupies the compact twin tangent subspace.

## Tiers

### Tier 1: Recorded Derivatives In Compact Twin Basis

For each session and recorded context:

```text
R_c = 1 mu_c.T + X_c B_rec,c.T + epsilon
```

with `R_c` in matched recorded/twin unit order and centered eye position
`X_c`. Estimate `B_rec,c` by ridge regression, then measure:

```text
capture_rec,c(k) = || U_twin,k.T @ B_rec,c ||_F^2 / || B_rec,c ||_F^2
```

`U_twin,k` must be learned from fitted-twin finite-difference tangents in the
same matched unit order and cross-fit away from the tested context when possible.

### Tier 2: Context-Matched Derivative Alignment

Diagnostic/supportive:

```text
overlap_c = 0.5 * || orth(B_rec,c).T @ orth(J_twin,c) ||_F^2
```

Compare to context-shuffled, unit-shuffled, random, and RF/readout-preserving
nulls.

### Tier 3: Signed Axis Diagnostics

Diagnostic only. Signed x/y axis comparisons should not become the headline
unless coordinate conventions and reliability are unusually clean.

## Required Outputs

Write analysis outputs to:

```text
outputs/direct_recorded_derivative_twin_alignment/
```

Expected files:

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

## Implementation Notes

- Reuse matched-unit and finite-difference infrastructure from
  `declan/matched_twin_covariance_closure/run_finite_difference_closure.py`.
- Reuse RF/readout binning ideas from the RF-backed covariance closure run:
  `outputs/matched_twin_covariance_closure_rf_null_step025_rfbacked_v2`.
- Use the cids-to-STA-cache mapping from the closure runner rather than the
  older STG assumption that selected unit count equals STA cache unit count.
- Use old STG code as historical reference, especially RF loading and the
  signed-axis failure mode:
  `declan/shared_transformation_geometry/run_stg_retinotopy_tangent_identity.py`.
- Select ridge parameters by recorded reliability or response prediction, never
  by twin alignment.
- Make session-level inference the headline: mean effect, bootstrap CI over
  sessions, sign count, and sign-test p-value.

## Stop Rules

Continue toward a supplemental figure if:

- recorded derivative reliability exceeds eye-label shuffle in most retained
  contexts,
- Tier 1 survives global-rate + target-PC1 projection,
- Tier 1 survives RF/readout-preserving null,
- effects are positive in most sessions,
- k-sweep is smooth/plausible.

Keep as diagnostic only if:

- effects survive only random/unit-shuffle nulls,
- RF/readout-preserving null absorbs the effect,
- results depend heavily on reliability threshold.

Drop from manuscript if:

- recorded derivatives are not reliable above eye-shuffle,
- Tier 1 is null across sessions,
- unit-order or context-definition audits fail.

