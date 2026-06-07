# covTFTS_figure Panel-by-Panel Prescription

## Purpose

Create the final **covTFTS_figure** for the manuscript. This figure is the synthesis / landing figure that connects the recorded V1 covariance result to the canonical twin mechanism.

The figure should be a **data-anchored synthesis**, not a purely conceptual cartoon. It should combine small schematic elements with real quantitative plots, and it must keep the epistemic boundaries explicit:

1. **Recorded V1 result**: established biological finding.
2. **Canonical twin tangent-family result**: model-side mechanism.
3. **Tangent-to-covariance bridge**: meaningful but partial link.

The figure should not include optotype performance or the natural-image scale / ecology analysis.

## Final figure title

Use this exact title unless a manuscript-specific style guide requires something shorter:

> **Figure 4. Image-specific retinal translations form a compact, image-generalizing reafferent geometry.**

## Figure-wide claim

The single bounded claim that the figure should make unavoidable is:

> The recorded V1 result is a low-dimensional, stimulus-aligned reafferent covariance. The canonical twin supplies a model-side mechanism for why such covariance can remain compact across image content: retinal translations generate content-dependent tangents, but those tangents form a compact, image-generalizing family.

## What this figure must not imply

Do **not** imply any of the following:

- a universal signed displacement axis
- a two-dimensional shared tangent plane across all images
- a complete tangent-to-covariance identity
- direct proof of the same tangent subspace in recorded V1
- an ecological / optimality claim
- a discrimination / performance claim

## Final panel list

The figure should have **six panels**, labeled:

- **A. Recorded anchor**
- **B. Why this is nontrivial across images**
- **C. Tangent construction in the canonical twin**
- **D. Compact tangent family**
- **E. Image-disjoint generalization**
- **F. Partial covariance bridge**

## Overall layout recommendation

A clean 2 × 3 layout is preferred:

```text
Top row:    A | B | C
Bottom row: D | E | F
```

Alternative if panel widths differ:

```text
Top strip:  A (small anchor strip spanning part of the top)
Main body:  B, C on top row
            D, E, F on bottom row
```

But the safest implementation is a standard 2 × 3 layout with A visually smaller / simpler than the others.

## Visual grammar and epistemic boundaries

Use **three visual registers**.

### 1. Recorded V1 panels
Applies primarily to Panel A.

Recommended styling:
- grayscale, black, or dark neutral tones
- solid border
- label in panel title or subtitle: **Recorded V1**

### 2. Canonical twin / model panels
Applies to Panels C, D, and E.

Recommended styling:
- blue, purple, or another clearly distinct color family
- solid border, but clearly marked as **Canonical twin**
- legends / annotations should explicitly say “twin” where appropriate

### 3. Partial bridge panel
Applies to Panel F.

Recommended styling:
- same model color family but lighter tint or dashed connector elements
- visually indicate incompleteness or partiality
- any arrows between panels should be dashed for this bridge

## Global typography and style rules

- Use readable panel labels: **A–F**
- Use concise panel subtitles
- Do not overload the figure with long prose
- Use short annotations within panels for headline values
- Keep quantitative panels readable at manuscript scale
- Panels D and E must be large enough to function as real results panels, not tiny insets

## Input data sources

Use the already generated results, not ad hoc reanalysis, unless a file is missing.

### Recorded covariance source
Use the already finalized recorded V1 covariance analysis / Fig. 3 outputs for:
- effect of FEM correction on shared variability
- low-dimensionality of FEM covariance
- signal alignment / stimulus-alignment summary

The exact file choice can be made by the coding agent from the existing output directory, but it should match the values already being used in the draft and manuscript.

### Canonical twin / TFTS source
Use the finalized TFTS outputs, especially from:

```bash
outputs/twin_feature_tangent_structure_prod_limited_synth
```

and any finalized summary files associated with:
- union compactness
- image-disjoint generalization
- covariance bridge / Analysis 5
- object counts / QC

If alternative output roots exist with cleaner final summaries, prefer the finalized production root already used in the manuscript draft.

## Required figure output files

Create at minimum:

```bash
outputs/covTFTS_figure/covTFTS_figure.png
outputs/covTFTS_figure/covTFTS_figure.pdf
outputs/covTFTS_figure/covTFTS_figure.svg
outputs/covTFTS_figure/README.md
outputs/covTFTS_figure/panel_data_manifest.json
```

Also create panel-wise intermediate files if helpful:

```bash
outputs/covTFTS_figure/panel_A_source_summary.csv
outputs/covTFTS_figure/panel_D_compactness_summary.csv
outputs/covTFTS_figure/panel_E_generalization_summary.csv
outputs/covTFTS_figure/panel_F_bridge_summary.csv
```

## Required implementation philosophy

Do **not** hand-draw the final figure in Illustrator or PowerPoint as the primary implementation.
Do **not** manually embed screenshots unless necessary for a specific schematic element.

The figure should be generated reproducibly from code, ideally in Python / matplotlib, with any schematic elements drawn programmatically or composited from programmatically generated components.

The README should document:
- which source files were used
- what values were plotted
- any manual styling adjustments
- any unresolved data ambiguities

---

# Panel-by-panel specification

## Panel A. Recorded anchor

### Goal

Remind the reader of the empirical biological anchor without re-running or re-arguing Fig. 3 in full.

### Scientific message

> In recorded foveal V1, conditioning on eye position reduces classical shared variability, and the removed covariance is low-dimensional and aligned with visually driven population structure.

### Function in the figure

This panel anchors the entire figure in recorded biology, so the figure reads as:
- **recorded result first**
- **twin mechanism second**

rather than as a “look at the twin” figure.

### Recommended visual format

A compact summary strip or mini-panel with **two small visual elements** and one short annotation block.

Preferred format:
1. small schematic or mini-bar/arrow showing shared variability decreases after FEM conditioning
2. small mini-plot or icon indicating low-dimensional / signal-aligned covariance
3. one short text callout

If space is tight, this can be a compact composite instead of two separate axes.

### Acceptable data content

Use already finalized recorded results only.

Minimal content to show:
- shared variability / noise correlation reduced after FEM correction
- FEM covariance is low-rank / low-PR / concentrated in a few modes
- FEM covariance overlaps visually driven / stimulus-driven structure

Do not introduce new statistics here. Use already established ones from the manuscript.

### Panel subtitle

Use:

> **Recorded V1: FEM-linked covariance is reafferent, low-dimensional, and stimulus-aligned**

### Required annotation language

Include a text callout such as:

> FEM correction reduces shared variability  
> Removed covariance is low-dimensional and signal-aligned

If exact numerical values are already reconciled and final, they may be included in tiny text. If any numerical ambiguity remains, keep the annotation qualitative.

### What not to do

- do not repeat the full Fig. 3
- do not show too many recorded subplots
- do not let Panel A dominate the figure

---

## Panel B. Why this is nontrivial across images

### Goal

Make the key logical problem visually obvious:
for a single image, 2D eye motion gives a local 2D translation neighborhood, but across many images compactness is not guaranteed.

### Scientific message

> Two-dimensional eye motion does not by itself guarantee that FEM-linked covariance stays compact across image content.

### Function in the figure

This is the most important conceptual panel. It prevents reviewers from dismissing the result as a trivial consequence of 2D eye motion.

### Recommended visual format

A schematic panel split into two halves.

#### Left half: single-image / local case
Show:
- one example image patch
- a tiny eye-position cloud or displacement arrows
- a local 2D response neighborhood / plane in population space

Label:
> single image

#### Right half: across-image null expectation
Show:
- several different image patches
- each with its own local tangent plane / arrows
- the planes pointing in unrelated population directions
- resulting aggregate cloud shown as high-dimensional / smeared

Label:
> across images, compactness is not guaranteed

Optional label on the right:
> null expectation

### Panel subtitle

Use:

> **Why this is nontrivial across images**

### Required annotation language

Add a short statement:

> For one image, retinal translation gives a local low-dimensional neighborhood.  
> Across many images, those neighborhoods could point in unrelated directions.

### Schematic design constraint

Do **not** make the right side too tidy. The right side should visually communicate:
- unrelated tangent directions
- high-dimensional mixture
- no shared compact family under the null expectation

### What not to do

- do not attach quantitative statistics to this panel
- do not imply that the null expectation was directly measured; this is a logical possibility / schematic

---

## Panel C. Tangent construction in the canonical twin

### Goal

Show how the model-side tangent objects are constructed.

### Scientific message

> In the canonical twin, each full stimulus-history object has image- and history-specific retinal-translation tangents.

### Function in the figure

This panel introduces the model object whose structure is quantified in Panels D–F.

### Recommended visual format

Schematic with three steps:

1. **Stimulus-history object**
   - show a stack or strip of frames, or a “history object” icon
   - indicate it is a full history object, not a static image average

2. **Small retinal translations**
   - show ±x and ±y shifts or tiny displacement arrows
   - indicate these are small perturbations

3. **Population response tangents**
   - show two tangent vectors, \(b_x(I)\) and \(b_y(I)\)
   - depict these as image-specific
   - show multiple example objects, each with its own different tangent directions

### Panel subtitle

Use:

> **Canonical twin: small retinal shifts define image-specific tangents**

### Required annotation language

Include:
- “full stimulus-history object”
- “small horizontal and vertical retinal shifts”
- “response tangents \(b_x(I)\), \(b_y(I)\)”

### Important design constraint

Do **not** draw these tangents as collapsing onto one shared 2D plane.
Instead, show multiple object-specific tangents lying within a broader shared bundle / slab.

This is important because the actual result is:
- compact, approximately 9-dimensional
- not universal 2D
- not a single signed x/y axis

### Optional micro-annotation

Add:
> image- and history-specific

### What not to do

- do not make this panel a large quantitative plot
- do not imply recorded data here; label clearly as **canonical twin**

---

## Panel D. Compact tangent family

### Goal

Provide the first load-bearing quantitative TFTS result:
the tangent family is compact relative to a null.

### Scientific message

> The union of image-specific tangents occupies a compact, not two-dimensional, subspace.

### Function in the figure

This panel shows that the across-image problem in Panel B resolves in the twin: the tangent family is structured and compact.

### Recommended plot type

Primary recommendation:
- point plot, bar plot, or line plot showing participation ratio at multiple displacement scales
- observed values versus unit-shuffled null expectation with confidence intervals if available

Preferred x-axis:
- tangent displacement scale (0.125, 0.25, 0.5 arcmin)

Preferred y-axis:
- participation ratio (or equivalent compactness metric already used in TFTS)

Plot observed:
- one marker / bar per displacement scale

Plot null:
- comparison markers / bars or shaded null band

### Required numerical annotation

The primary headline annotation must include:

> 0.25 arcmin: PR ≈ 9 vs null ≈ 31

Also include neighboring scales if available, but 0.25 is the key value.

If exact production values differ slightly across files, use the finalized manuscript-consistent values.

### Panel subtitle

Use:

> **Image-specific tangents occupy a compact, not two-dimensional, subspace**

### Required annotation language

Short callout:
> Compact tangent family across objects and images

Optional secondary note:
> Similar compactness at 0.125 and 0.5 arcmin

### Design constraint

Make the observed-vs-null difference visually clear.
This panel must be readable and interpretable on its own.

### What not to do

- do not shrink this panel to an inset
- do not use a schematic here; it must be a real quantitative plot

---

## Panel E. Image-disjoint generalization

### Goal

Provide the second load-bearing quantitative TFTS result:
the compact tangent structure generalizes across held-out image identities.

### Scientific message

> A tangent basis learned from one set of image identities captures held-out image tangents above null.

### Function in the figure

This panel proves the compact family is not just a same-image leakage artifact.

### Recommended plot type

A line plot of:
- **x-axis**: basis dimension \(k\)
- **y-axis**: held-out tangent variance captured (or the exact finalized capture metric)

Show:
- empirical image-disjoint result
- null / unit-shuffle comparison
- possibly confidence intervals or variability bands if available

Use the finalized image-disjoint split only, not the contaminated object-random split.

### Required numerical annotation

Primary annotation must include:

> 10D basis captures ≈ 0.50 held-out tangent variance  
> null ≈ 0.11  
> 0% image-ID leakage

If the capture value differs slightly depending on the exact summary file, use the finalized production/manuscript value.

### Panel subtitle

Use:

> **The compact tangent structure generalizes to held-out image identities**

### Required annotation language

Include:
- “image-disjoint train/test”
- “held-out image identities”
- “0% image-ID overlap” or “0% image-ID leakage”

### Optional supporting annotation

If helpful, add:
> Effect remains after removing same-image leakage

### Design constraint

This is a load-bearing panel and must be large, real, and readable.

### What not to do

- do not show the object-random split as the main result
- do not mix several split modes into this panel
- do not clutter with too many secondary curves

---

## Panel F. Partial covariance bridge

### Goal

Show that local tangents provide a meaningful but incomplete bridge to the full FEM covariance.

### Scientific message

> Tangent-predicted covariance aligns with full sampled FEM covariance, especially for local eye-position clouds, but the match is partial and weakens with broader finite-displacement clouds.

### Function in the figure

This panel links the model-side tangent family back to the recorded-style covariance object, while preserving the important caveat that the link is only partial.

### Recommended plot type

Plot:
- **x-axis**: eye-position cloud scale (or whatever the finalized Analysis 5 scale axis is)
- **y-axis**: overlap / subspace alignment / bridge metric between tangent-predicted covariance and full sampled covariance

Show:
- empirical curve for the twin / canonical population
- possibly a null or chance reference if already available and useful
- if multiple scales or conditions exist, keep only the essential one(s)

### Required qualitative emphasis

The plot and its annotations must make three things clear:

1. the bridge is **meaningful**
2. the bridge is **incomplete**
3. the overlap is strongest for **local** clouds and weaker for broader clouds

### Panel subtitle

Use:

> **Local tangents provide a partial bridge to FEM covariance**

### Required annotation language

Include a short annotation such as:

> Strongest for local eye-position clouds  
> Weakens for broader finite-displacement clouds

Optional additional phrase:
> meaningful but incomplete

### Visual grammar

Any cross-panel arrow or connector from Panels D/E to F should be dashed or partial.
Do not draw this as a solid causal completion.

### What not to do

- do not title this “Tangents explain FEM covariance”
- do not oversell the overlap as complete
- do not turn this into a large multipanel analysis dump

---

# Figure caption

Use the following draft caption as the starting point. The coding agent may adapt formatting, but the content should remain faithful.

## Draft caption

**Figure 4. Image-specific retinal translations form a compact, image-generalizing reafferent geometry.**  
(A) Recorded foveal V1 contains a reafferent covariance component: conditioning on measured eye position reduces classical shared variability, and the removed component is low-dimensional and aligned with visually driven population structure. This recorded result establishes the biological anchor.  
(B) Low-dimensional covariance for a single image follows from local retinal translation, but compact covariance across many images is not guaranteed. If each image generated translation tangents in unrelated population directions, aggregating across content would produce a high-dimensional mixture.  
(C) In the canonical digital twin, small retinal translations of each full stimulus-history object define image- and history-specific response tangents, \(b_x(I)\) and \(b_y(I)\). These tangents need not share a universal signed x/y direction.  
(D) The union of these image-specific tangents nevertheless occupies a compact population subspace. At 0.25 arcmin, the tangent-family participation ratio was approximately 9, far below the unit-shuffled null of approximately 31, with similar compactness at neighboring displacement scales.  
(E) The compact tangent structure generalized across image identities. In image-disjoint train/test splits, a tangent basis learned from one set of image identities captured held-out image tangents above unit-shuffled nulls. At 0.25 arcmin, a 10-dimensional basis captured approximately half of held-out tangent variance, compared with approximately 0.11 under the null, with no image-ID leakage between train and test folds.  
(F) Local tangents provided a meaningful but incomplete bridge to the full FEM covariance. Tangent-predicted covariance aligned best with the full sampled covariance for local eye-position clouds and less strongly for broader finite-displacement clouds, consistent with real FEMs sampling curved and history-dependent response neighborhoods rather than an exactly linear local plane.

Together, these panels make the central logic explicit. The recorded V1 result is a low-dimensional, stimulus-aligned reafferent covariance. The canonical twin supplies a model-side mechanism for why such covariance can remain compact across content: retinal translations generate content-dependent tangents, but those tangents form a compact, image-generalizing family. The result does not imply a universal signed displacement axis, a complete tangent-to-covariance identity, or direct proof of the same tangent subspace in recorded V1.

---

# Implementation checklist

## Required outputs

The coding agent should produce:

1. Final figure files:
   - `covTFTS_figure.png`
   - `covTFTS_figure.pdf`
   - `covTFTS_figure.svg`

2. A README:
   - panel descriptions
   - source files used
   - final values used for annotations
   - any deviations from this prescription

3. A panel data manifest:
   - exact files / tables / columns used for each panel

4. Any intermediate plotted summaries if panel-specific data extraction is nontrivial

## Required status report

After implementation, report:

1. code compile status
2. final source files used for each panel
3. any unresolved ambiguity in the recorded anchor values
4. exact plotted values used in Panels D, E, and F
5. whether any panel had to be simplified due to missing source artifacts
6. whether the figure fully satisfies this prescription

## Hard stop rules

Do not:
- add the ecology panel
- add optotype or discrimination panels
- introduce new exploratory analyses to “improve” the figure
- silently swap in object-random results for image-disjoint results
- overclaim the tangent-to-covariance bridge

If a source artifact is missing, stop and report the missing dependency rather than improvising a new analysis.

## Final recommendation

Treat this as the committed main-text synthesis figure. The figure is successful if a reader can understand, in one pass, the following logic:

> Recorded V1 shared variability during fixation is largely reafferent.  
> Across-image compactness is not guaranteed by 2D eye motion alone.  
> In the canonical twin, image-specific translation tangents occupy a compact, image-generalizing subspace.  
> This provides a plausible model-side mechanism for why FEM-linked covariance remains structured across content.
