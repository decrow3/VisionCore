# fixRSVP Cross-Session Allen Check

## Scope

Matched Step 0/1 fixRSVP runs were compared across four Allen sessions using the same backend settings:

- unit mode: `image_phase_radius`
- `min_samples_per_unit=32`
- `max_units=16`
- `max_samples_per_unit=64`
- unfiltered backend pooling (`local_state_keep_fraction=1.0`)
- shared multidataset digital twin `resnet_none_convgru`, epoch 147

The goal of this check was to test whether the current fixRSVP local tangent-geometry signal is specific to `Allen_2022-02-24` or recurs across nearby Allen sessions under matched analysis settings.

## Session Summary

| Session | Dataset idx | Retained manifest units | Small-good Step 0 fraction | All-unit $A_J$ delta | All-unit 95% CI | Small-good $A_J$ delta | Small-good 95% CI | Readout |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| Allen_2022-02-16 | 0 | 67 | 0.5000 | 0.0799 | [0.0220, 0.1739] | 0.0799 | [0.0407, 0.1748] | clear positive |
| Allen_2022-03-04 | 1 | 79 | 0.6875 | 0.0511 | [-0.0311, 0.1481] | 0.1032 | [-0.0181, 0.1530] | ambiguous positive trend |
| Allen_2022-02-24 | 10 | 55 | 0.5625 | 0.0235 | [-0.0196, 0.1168] | 0.0513 | [-0.0331, 0.2778] | weak / ambiguous |
| Allen_2022-04-08 | 2 | 127 | 0.3750 | 0.0415 | [-0.0540, 0.0893] | 0.1029 | [-0.0073, 0.2021] | weak / ambiguous |

## Main Interpretation

This cross-session check is still informative after the code fixes, but the repaired picture is more cautious than the earlier pre-fix read.

The strongest session remains real. `Allen_2022-02-16` still shows a clean matched-over-shuffled effect, and it now does so under a stricter null that excludes same-image matches and under unit-local Step 0 central-mass gating.

The broader panel is mixed. `Allen_2022-03-04`, `Allen_2022-02-24`, and `Allen_2022-04-08` all retain positive point estimates in at least one summary, but their bootstrap intervals now cross zero under the repaired code path. So the result is no longer best described as clearly positive in multiple sessions. It is better described as one clear session plus several suggestive but ambiguous sessions.

The paper-safe claim at this stage is:

- image-matched Jacobian planes can predict model FEM covariance geometry above matched image-shuffled controls in Allen fixRSVP,
- at least one Allen session shows a clear local positive effect under the repaired code path,
- other sessions show suggestive but not yet decisive evidence,
- and the effect remains strongest, when present, in the local small-displacement regime.

## Local Claim Boundary

The main boundary remains Step 0 over realistic empirical displacements. The current evidence supports a local tangent-geometry account, not a full finite-displacement linear explanation.

The cleanest example is still `Allen_2022-02-16`. That session remains stronger than the repaired `Allen_2022-02-24` run because the matched-minus-shuffled effect is larger and more consistent even after the null and Step 0 fixes:

- all-unit $A_J$ delta: `0.0799`, bootstrap 95% CI `[0.0220, 0.1739]`
- small-displacement-good $A_J$ delta: `0.0799`, bootstrap 95% CI `[0.0407, 0.1748]`
- all-unit $V_J$ delta remains positive at the session level, with median matched capture above the image-shuffled control
- the small-displacement-good subset remains the primary positive readout

That is meaningful matched-over-shuffled structure, not just separation from a random-subspace null.

But even in this stronger session, the first-order response prediction is still local. In the repaired `Allen_2022-02-16` run, `50.0%` of units exceed median $R^2_{lin} > 0.5$ over `0.062`, `0.125`, and `0.250 px`, while only `6.25%` do so over the unit-local empirical central-mass displacements, whose median values are `0.578` and `0.831 px`.

So the right conclusion remains:

> The image-translation Jacobian predicts model FEM covariance geometry in a local small-displacement regime, but a strict first-order response prediction does not yet hold over the central empirical displacement range.

That is not a failure of the framework. It is more naturally read as evidence that fixRSVP responses sample a nonlinear transformation manifold whose local tangent remains informative even when a single linearization no longer explains the full finite trajectory.

## Step-Size Stability

The finite-difference Jacobian still looks reasonably stable in the stronger sessions, especially `Allen_2022-02-16`. The repaired run continues to show good replay sanity checks and nontrivial Step 1 separation without any sign that the tangent estimate is purely numerical noise.

That supports an important distinction: the local tangent is not numerical noise, even when finite-displacement linear prediction degrades.

## Why Small-Displacement Grouping Is Primary

The small-displacement grouping remains the better interpretive split than the full Step 0 good / conditional / poor regime.

For example, in the repaired panel the full-Step-0 grouping remains less clean than the small-displacement grouping because the former is dominated by larger-displacement failures. A unit can still carry a useful local tangent while failing the broader finite-displacement regime. The small-displacement-good split therefore remains the more direct test of the actual scientific claim.

For that reason, the small-displacement-good subset should remain the primary analysis focus, with the all-unit summary kept as context.

## Possible Drivers Of The Session Split

- Session eye statistics differ. `Allen_2022-03-04` and `Allen_2022-04-08` have noticeably larger centered-radius tails than `Allen_2022-02-24`, but only `Allen_2022-03-04` shows a clear alignment gain, so broader displacement support alone is not sufficient.
- Step 0 quality differs by session. `Allen_2022-04-08` still has zero units in the all-displacement-good regime and only a modest small-displacement-good fraction, which is consistent with weaker local linearization support where the alignment claim should be strongest.
- Session composition likely matters beyond simple sample count. `Allen_2022-04-08` has the largest retained manifest (127 units) yet the weakest small-good alignment signal, so larger unit availability does not automatically improve the matched-minus-shuffled geometry readout.

- Model or backend validity may differ by session. For `Allen_2022-04-08`, the null result should not immediately be treated as biological; it may reflect weaker model fit, poorer eye-trace validity, different image composition, or a mismatch between the current unit mode and the replay-defined local state.

## Current Cross-Session Claim

The right cross-session claim is not:

> The Jacobian generalizes to natural images.

It is closer to:

> In matched Allen fixRSVP runs, at least one session shows a clear local matched-over-shuffled Jacobian geometry effect, while several additional sessions show suggestive but not yet decisive positive trends. The current evidence therefore supports a local natural-image tangent signal, but not a stable cross-session generalization claim.

That is still a meaningful framework result. It moves the story beyond optotypes while staying honest about session dependence, the stricter repaired null, and the unresolved central-mass Step 0 limitation.

## Output References

- `Allen_2022-02-16`: `outputs/jacobian_predictive_framework/allen_2022_02_16_image_phase_radius_u16/step01_run_overview.md`
- `Allen_2022-03-04`: `outputs/jacobian_predictive_framework/allen_2022_03_04_image_phase_radius_u16/step01_run_overview.md`
- `Allen_2022-02-24`: `outputs/jacobian_predictive_framework/allen_2022_02_24_image_phase_radius_u16/step01_run_overview.md`
- `Allen_2022-04-08`: `outputs/jacobian_predictive_framework/allen_2022_04_08_image_phase_radius_u16/step01_run_overview.md`

## Recommended Next Step

The next useful step is not more one-session local-state trimming. It is a compact session-level summary layer built on this four-session panel.

The cross-session table should keep, for each session:

- number of backend units
- fraction small-displacement-good
- all-unit matched-minus-shuffled $A_J$ delta with CI
- small-displacement-good $A_J$ delta with CI
- all-unit matched-minus-shuffled $V_J$ delta with CI
- small-displacement-good $V_J$ delta with CI
- Step 0 central-mass pass fraction
- finite-difference step alignment

If this pattern keeps holding, the most useful figure would be a compact bridge figure:

- Panel A: Step 0 $R^2_{lin}$ versus displacement magnitude, showing small-displacement validity and finite-displacement breakdown.
- Panel B: matched-minus-shuffled $A_J$ deltas across sessions, with all-unit and small-displacement-good markers plus bootstrap CIs.
- Panel C: matched-minus-shuffled $V_J$ deltas across sessions.
- Panel D: a schematic stating the honest interpretation: local tangent predicts covariance direction, while the full empirical FEM trajectory samples a nonlinear manifold.

That figure would tell the correct story rather than hiding the central-mass failure.