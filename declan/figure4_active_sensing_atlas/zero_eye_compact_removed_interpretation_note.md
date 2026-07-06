# Zero-Eye And Compact-Removed Interpretation Note

Date: 2026-06-21
Status: separate interpretation note; Panel C edits paused

## The Terminology Collision

`zeroed eye` can mean two different things unless we keep the axes separate:

1. A static-response counterfactual:
   the model response table is evaluated with the eye displacement set to zero
   or held at the static reference. In the current code this is closest to
   `zero_static` / `zero_lambda_counts`.

2. A decoder without measured eye-position access:
   the decoder is not given the true eye trace and must infer or marginalize
   over possible eye trajectories. In the current code this is the `joint`
   observer when contrasted with `known_eye`.

Those are not the same condition. `known_eye` vs `joint` is about information
available to the decoder. `zero_static` vs `full_exact` vs `compact_only` vs
`compact_removed` is about which predicted response table the decoder is asked
to use.

## Current Panel C Conditions

The feature-space compact-removal audit is best read as:

```text
observed response:
  full recorded-eye response

known eye:
  decoder scores candidates using the true eye trajectory

full joint:
  decoder is not given the true eye trajectory; it marginalizes over candidate
  trajectories in the full response table

zero_static / zeroed eye:
  decoder scores the full observed response against a static-reference response
  table

compact_only:
  decoder scores against zero + compact_delta

compact_removed:
  decoder scores against zero + residual_delta
```

Companion diagnostic plot:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/
  panel_C_distinct_condition_feature_recovery.png
  panel_C_distinct_condition_feature_recovery_values.csv
```

So `zero_static` is not an information-theoretic no-eye lower bound. It is a
particular misspecified static model. It can still recover feature information
from candidate-specific static responses.

## Why Compact-Removed Can Fall Below Zero-Eye

`compact_removed` is not `zero_static` with less information. It is a different
misspecified response table:

```text
compact_removed = zero + (full_delta - compact_delta)
```

The residual can contain structured but misleading candidate/trajectory
variation after the compact component is removed. When the posterior is formed
from that residual-only table, the decoder can become confidently wrong about
the candidate feature. A static-reference model can be less informative but
also less misleading.

That is why compact-removed feature recovery can sit below the zero-eye
curve. This should not be described as negative eye information. The safer read
is:

```text
Removing the compact component leaves a residual response model that is more
misleading than the zero-eye model for this posterior feature-recovery score.
```

## Suggested Wording

Use `zero-eye` when the intended meaning is the static-eye assumption, but
define it at first use: the observed response comes from the moved movie, while
the observer scores it with the zero-eye-motion/static-reference response
table.

Use `latent-eye joint observer` or `hidden-eye joint observer` when the intended
meaning is a decoder without the measured eye trace.

Use `known-trace control` when the candidate observer is given the true eye
trajectory in the deterministic table. Reserve `known-pose ceiling` for a
validated image-conditioned forward/rendering observer, not for the compact
linear residual diagnostics.

Avoid saying that `compact_removed` does worse than "zero-eye information"
unless the text immediately defines zero-eye as the static-reference response
model. Otherwise it reads as if the residual contains less information than no
eye access, which is not the right interpretation.

## Claim Boundary

The compact-removal result supports a mechanism statement about this response
projection:

```text
The compact component carries much of the response structure that lets the
latent-eye joint observer recover features.
```

It does not by itself prove that the residual has negative information, that
the animal computes this posterior, or that the compact subspace is the unique
useful structure.
