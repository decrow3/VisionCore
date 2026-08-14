# Final corrected-cohort replacement protocol

Use this protocol only after the interim 49×973 bridge/input-only cohort.

## Images

Retain the 49 legacy identities passing the Phase-2 corrected-crop validity gate. Select 51 replacements from the regenerated 240-Hz BackImage window table by: corrected crop validity, exact saved-frame availability, no duplicate `(session, trial_idx, scored interval)`, and stratified sampling across session, RMS contrast, orientation coherence, and corrected local SF centroid. Store the candidate pool, fixed seed, stratum definitions, and every rejection reason.

## Traces

Retain the 973 explicit-history-valid legacy identities. Select 27 replacements from the corrected eligible pool using the same 32-history/40-score contract, with no duplicate source interval and stratification across corrected path length, RMS radius, and corrected >32-Hz position-power fraction. Corrected event coverage, if used, must be sampled only after the regenerated 240-Hz event labels are frozen.

## Gate

The resulting 100-image × 1,000-trace cohort is a new immutable manifest. It must not be assembled by silently retaining invalid legacy identities.

## Implemented cohort (2026-08-13)

This gate has now passed. The outcome-independent builder is
`build_rr100_corrected_production_cohort.py`, and the frozen cohort is in
`outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1/`.

- Images: 49 valid legacy identities plus 51 validated replacements selected
  across session and tertiles of corrected RMS contrast, orientation
  coherence, and local SF centroid.
- Traces: 973 legacy identities with valid explicit history plus 27 validated
  replacements. The eligible pool occupied 25 of the 27 possible path-length
  × RMS-radius × >32-Hz-power tertile cells. Selection covers all 25 occupied
  cells once, with the remaining two slots assigned to the least-represented
  occupied cells.
- Selection seed: `20260813`. No neural response or model-output quantity was
  used to select either replacement set.
- Event labels were not used as a sampling axis. Any later event-conditioned
  analysis therefore remains secondary and must use regenerated corrected
  labels.

The complete candidate ledgers, rejection reasons, selected replacement
tables, input QA figure, and immutable hashes are stored beside the cohort
manifest. The production runner has additionally frozen the 100 exact image
patches and the 1,000 corrected `32 history + 40 scored` trajectory segments in
`outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/`.
A bounded GPU preflight has scored one atomic ten-movie block. It is a systems
validation only: no complete balanced round is available for analysis, and the
production bank has not been launched.
