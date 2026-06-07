# Figure 4 Coding Agent Handoff

Goal:
Revise `generate_covTFTS_figure_3.py` so Figure 4 communicates the compact reafferent-geometry story more clearly, while keeping unfinished analyses as loaded placeholders rather than hardcoded claims.

Current intended main figure:
- A: Recorded V1 anchor
- B: Image-specific local translation charts
- C: Compact tangent spectrum
- D: Cross-image generalization
- E: Tangent-subspace information placeholder / loaded result
- F: Optional operating-regime placeholder / loaded result, not hardcoded into the figure for now

## Panel A: Recorded V1 Anchor

Keep the current structure:
- Left: mean noise correlation before vs after eye-position/FEM correction.
- Right: cumulative eigenspectrum of the removed FEM covariance component.

Small polish:
- Avoid claiming stimulus alignment unless that is explicitly plotted.
- Prefer wording such as "low-dimensional FEM-linked covariance."
- If space allows, improve labels:
  - `Cumulative variance (FEM covariance)`
  - `FEM covariance eigenvalue rank`
  - consider `Eye-pos. corrected` instead of `FEM-corr.` if it fits.

## Panel B: Replace Current Dot Cloud

Problem:
The current PCA scatter of blue/purple/gray dots does not pass the glance test. It is hard to tell what each dot means or what the reader should take away.

Replace with a real-data "local translation charts" panel.

Preferred implementation:
- Use base twin responses `r0(I)` for each image/history object.
- Project `r0(I)` into a 2D response or tangent-relevant PCA space.
- For a sparse subset of objects, draw local tangent glyphs centered at `r0(I)`:
  - blue: `b_x(I)`
  - purple: `b_y(I)`
- Use 20-40 objects rather than all objects.
- Optional: draw projected endpoints `r0(I) +/- alpha*b_x(I)` and `r0(I) +/- alpha*b_y(I)` as short arrows, crosses, or tiny parallelograms.
- Remove the gray null from Panel B. Null comparisons belong in C/D.

Suggested title:
`Image-specific local translation charts`

Take-away:
Small retinal translations define local response directions attached to each image/history state. The translation axes vary across images, so compactness across the full tangent family is not guaranteed.

## Panel C: Improve Null Spectrum Display

Current panel is conceptually strong, but the gray dashed diagonal looks like a placeholder.

Changes:
- If actual unit-shuffle null cumulative spectra are available, plot a gray null band or multiple faint gray null spectra.
- If only null participation-ratio statistics are available, label honestly as `Unit-shuffle PR reference`, not `Null reference`.
- Consider limiting x-axis to about 32 components instead of 40.
- Tune aspect ratio so the observed curve's rapid rise is obvious at first glance.
- Keep the `PR = 9.0 vs null ≈ 31` annotation, if supported by loaded data.

Take-away:
The observed tangent-family spectrum rises much faster than the unit-shuffle reference, showing compact shared population geometry.

## Panel D: Rename And Clarify

Rename:
- From `Image-disjoint generalization`
- To `Cross-image generalization`

Y-axis:
Use:
```text
Held-out translation
tangent variance captured
```

Do not use `Held-out translation variance captured`; that is too loose. The panel measures variance captured in local translation-induced response tangents, not finite translated-response variance.

Caption/take-away:
A basis learned from translation tangents in one set of images captures held-out translation-tangent variance from other images far above unit-shuffle null.

## Panel E: Load Tangent-Subspace Information

Panel E should be wired like a real analysis output, not computed ad hoc in the figure script.

For now:
- Keep a clean placeholder if the Panel E CSV is missing.
- Placeholder should avoid overclaiming.
- Suggested placeholder text:
  - `Tangent-subspace Fisher gain`
  - `production run pending`

Later expected plot:
- Compare full FEM gain, tangent-projected gain, orthogonal-projected gain, and null/shuffle.
- The primary metric should match the Panel E analysis output from `run_tangent_subspace_information.py`.

Important scientific caution:
Make sure the plotted metric matches the analysis summary. If using Fisher trace, do not silently plot only a pattern-only variant unless labeled.

## Panel F: Do Not Hardcode Strong Claim In Figure Script

For now, table Panel F as a main result unless a standalone analysis produces the stronger case.

Immediate code changes:
- Remove hardcoded interpretive text such as:
  - `Drift should lie in the local tangent regime; microsaccades may shift between local charts.`
- If a placeholder Panel F is shown, make it neutral.
- Prefer not to include the current covariance-overlap operating-regime panel in the main figure unless explicitly requested.

Planned standalone analysis:
- Create a separate script later, e.g. `run_fem_operating_regimes.py`.
- Figure script should only load finished Panel F outputs, analogous to Panel E.

Expected later outputs:
- `panelF_operating_curve.csv`
- `panelF_fem_event_ranges.csv`
- `panelF_manifest.json`

Stronger-case y-axis candidates:
- Best: tangent-subspace Fisher gain fraction vs displacement scale.
- Second-best: local linear prediction quality vs displacement scale.
- Weaker diagnostic: covariance/tangent overlap vs displacement scale.

## Metadata / README Updates

Update all generated metadata to match the revised panels:
- Top file docstring panel list.
- Generated `README.md`.
- `panel_data_manifest.json` keys.
- Figure titles and panel names.

Avoid stale labels such as:
- `Partial covariance bridge` for main Panel F if using operating regimes.
- `Image-disjoint generalization` if the visible title is now `Cross-image generalization`.

## Do Not Block On Later Validation

Do not block this coding pass on natural-image-regime validation.

Later todo:
Redo or adapt tangent-family analyses for natural-image presentations outside the rapidly updating FixRSVP sequence, to confirm the compact geometry is not an artifact of the RSVP temporal regime.
