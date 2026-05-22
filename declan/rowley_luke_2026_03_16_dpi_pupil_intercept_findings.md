# Luke 2026-03-16: DPI vs pupil intercept diagnostics

Date written: 2026-04-30

## Scope

This note documents the focused comparison between the DPI-derived eye traces and the affine-transformed pupil traces for the session `Luke_2026-03-16`, using the intercept-diagnostic outputs from `scripts/test_rowley10.py`.

Primary summary source:

- `outputs/figures/mcfarland/Luke_2026-03-16_v10/frac_fem_summary_Luke_2026-03-16.txt`

Supporting artifacts:

- `outputs/figures/mcfarland/Luke_2026-03-16_v10/right-dpi/intercept_diagnostics_Luke_2026-03-16.pdf`
- `outputs/figures/mcfarland/Luke_2026-03-16_v10/right-pupil/intercept_diagnostics_Luke_2026-03-16.pdf`
- `outputs/figures/mcfarland/Luke_2026-03-16_v10/left-dpi/intercept_diagnostics_Luke_2026-03-16.pdf`
- `outputs/figures/mcfarland/Luke_2026-03-16_v10/left-pupil/intercept_diagnostics_Luke_2026-03-16.pdf`

## Why this comparison matters

The production McFarland summary uses percentile bins and reports a relatively high `1 - alpha` FEM fraction. The intercept-diagnostic branch asks a narrower question: what happens to that estimate when we examine the near-zero distance behavior more directly using adaptive degree bins with a minimum pair-count floor.

The key question here was whether transformed pupil traces behave like DPI traces once everything is brought into degree space.

## Session context

- Pool A: 87 units
- Pool B: 76 units
- Left-eye covariance trials: 275
- Right-eye covariance trials: 282

This is a reasonably strong session, so the comparison is worth taking seriously.

## Main result

DPI and transformed pupil are fairly close under the current percentile-bin headline summary, but they diverge substantially in the adaptive near-zero intercept diagnostics.

The cleanest evidence comes from the right eye, because right DPI and right pupil use the same adaptive bin edges:

- adaptive edges: `[0.0, 0.08, 0.16, 0.32, 0.64, 1.2]`
- right DPI adaptive counts: `[2392, 11066, 37481, 96334, 95539]`
- right pupil adaptive counts: `[4089, 13812, 48355, 110288, 77592]`

Both branches therefore satisfy the minimum-pair floor in the first adaptive bin, and the DPI-versus-pupil mismatch on the right eye is not explained by sparse-bin failure.

## Right-eye comparison

### Production summary

- Right DPI current percentile first-bin FEM median: `0.7949`
- Right pupil current percentile first-bin FEM median: `0.7779`

At this level, the two traces look broadly similar.

### Adaptive intercept diagnostics

- Right DPI adaptive lowest-bin FEM median: `0.7913`
- Right pupil adaptive lowest-bin FEM median: `0.6652`

- Right DPI adaptive linear first-bin FEM median: `0.7684`
- Right pupil adaptive linear first-bin FEM median: `0.5970`

- Right DPI adaptive linear zero FEM median: `0.8132`
- Right pupil adaptive linear zero FEM median: `0.6227`

- Right DPI adaptive isotonic FEM median: `0.7328`
- Right pupil adaptive isotonic FEM median: `0.5951`

Interpretation:

- The production percentile-bin summary makes DPI and pupil look close.
- The adaptive near-zero diagnostics separate them clearly.
- On matched degree bins, pupil is lower than DPI by about `0.126` at the adaptive lowest-bin estimate.
- On the adaptive linear-first estimate, pupil is lower than DPI by about `0.171`.
- On the adaptive linear-zero estimate, pupil is lower than DPI by about `0.191`.

This means the right-eye pupil mismatch is a real content difference in the near-zero diagnostic, not just a binning artifact.

### Right-eye slope comparison

- Right DPI current percentile diag-slope summary: fraction positive `0.382`, median slope `-0.0018`
- Right DPI adaptive diag-slope summary: fraction positive `0.447`, median slope `-0.0006`

- Right pupil current percentile diag-slope summary: fraction positive `0.316`, median slope `-0.0009`
- Right pupil adaptive diag-slope summary: fraction positive `0.513`, median slope `0.0000`

Interpretation:

- Right DPI retains a slight negative near-zero trend in the adaptive diagnostic.
- Right pupil becomes effectively flat to slightly positive.
- This is consistent with the transformed pupil trace carrying a weaker near-zero distance dependence than DPI for this session.

## Left-eye comparison

The left-eye comparison points in the same direction, but it is somewhat less clean because the pupil branch merges to a wider first adaptive bin.

### Adaptive binning

- Left DPI adaptive edges: `[0.0, 0.08, 0.16, 0.32, 0.64, 1.2]`
- Left DPI first-bin pairs: `2124`

- Left pupil adaptive edges: `[0.0, 0.16, 0.32, 0.64, 1.2]`
- Left pupil first-bin pairs: `10104`

So left pupil has a broader first bin than left DPI.

### Adaptive intercept diagnostics

- Left DPI current percentile first-bin FEM median: `0.8105`
- Left pupil current percentile first-bin FEM median: `0.7761`

- Left DPI adaptive lowest-bin FEM median: `0.7737`
- Left pupil adaptive lowest-bin FEM median: `0.6307`

- Left DPI adaptive linear first-bin FEM median: `0.7734`
- Left pupil adaptive linear first-bin FEM median: `0.6307`

- Left DPI adaptive linear zero FEM median: `0.7824`
- Left pupil adaptive linear zero FEM median: `0.6307`

- Left DPI adaptive isotonic FEM median: `0.6807`
- Left pupil adaptive isotonic FEM median: `0.6058`

Interpretation:

- Left pupil is again systematically below left DPI in the adaptive intercept summaries.
- However, this comparison is partly confounded by the wider left-pupil first adaptive bin (`0.16 deg` instead of `0.08 deg`).

## Conclusion

For `Luke_2026-03-16`, the transformed pupil traces do not fully reproduce the near-zero intercept behavior seen in DPI.

The strongest statement comes from the right eye:

- DPI and pupil are similar under the coarse production summary.
- DPI and pupil diverge strongly in the adaptive near-zero intercept diagnostics.
- Because right DPI and right pupil share the same adaptive bin edges and both clear the count floor, this mismatch is not explained by sparse bins or edge placement.

Working interpretation:

- The affine pupil-to-DPI mapping is sufficient to bring traces into the same rough degree-space regime.
- It is not sufficient to guarantee that the pupil-derived trace preserves the same near-zero motion structure as DPI.
- For this session, the pupil branch appears to produce a flatter and weaker near-zero eye-covariance relationship than DPI.

## Recommended follow-up

The next diagnostic that would most directly test this interpretation is a shared-edge overlay of the adaptive `Ceye(d)` curves for right DPI and right pupil in this session. That would reveal whether pupil is flatter specifically in the first one or two distance bins, or whether the full distance relationship is shifted.