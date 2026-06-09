# Notes From Historic STG Code

Reference file:

```text
declan/shared_transformation_geometry/run_stg_retinotopy_tangent_identity.py
```

## Useful Patterns To Reuse

- The STG code treats signed x/y alignment as a diagnostic, not a robust primary
  endpoint. That matches the guardrail for this new analysis.
- RF metadata are loaded in a layered way:
  1. explicit RF keys in fixRSVP payloads if present,
  2. STA/STE cache fallback,
  3. twin readout mask fallback for model-side RFs.
- It writes per-image rows and session summaries separately, which is a good
  pattern for Tier 2/Tier 3 diagnostics.

## Important Change From STG

The older recorded RF fallback expected the STA/STE cache unit count to match the
selected fixRSVP unit count. That fails for all-cell STA caches. The newer
covariance-closure code maps:

```text
matched common_units -> selected session YAML cids -> all-cell STA cache rows
```

This new direct-derivative analysis should use that newer mapping.

## What Not To Resurrect

- Do not make signed `bx`/`by` retinotopy the primary endpoint.
- Do not require image-specific recorded tangent planes to be clean.
- Do not select contexts based on twin alignment.

