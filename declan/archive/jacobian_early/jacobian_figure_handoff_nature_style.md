# Handoff document: Main multipanel figure for Jacobian / identity-transformation geometry result

## Goal

Create a single main-text figure for the FEM V1 paper that communicates, clearly and attractively, the central result chain:

1. **Jacobian = translation tangent plane**  
   The image-translation Jacobian defines the local population-response directions corresponding to moving the *same stimulus* on the retina.

2. **Translation mimicry = identity/transformation confusability**  
   Some identity differences between E orientations can be substantially explained as local translations of one orientation. This confusability is structured and pair-specific.

3. **Phase landscape = confusability is spatially heterogeneous**  
   The overlap between identity and translation varies strongly over subpixel retinal phase, so no single fixed position summarizes the regime.

The figure should be **beautiful, concise, high-impact, and broadly legible**, with the aesthetic and clarity expected of a strong *Nature*-style main figure.

This is a main-text figure, not a supplement figure. It should prioritize intuition, conceptual clarity, and a small number of compelling quantitative results. Details and extra controls can be deferred to supplement.

---

## Overall figure story

The reader should understand the following after one careful look:

- FEM-induced variability is **not generic noise**.
- The Jacobian identifies the **local translation directions** in V1 population space.
- Those translation directions can **mimic some identity changes better than others**.
- This confusability depends strongly on **subpixel phase**.
- Therefore, the representation should be understood as a **structured identity/transformation landscape**, not as a simple fixed code or nuisance term.

The figure should communicate this story with minimal jargon on the page itself. The technical language can appear in the caption, but the panel visuals should be intuitive.

---

## Proposed figure structure

Use **five panels**, labeled **A–E**, arranged in a clean 2-row layout.

Recommended layout:

- **Top row**: A, B, C
- **Bottom row**: D, E

Where:
- **A** is a conceptual schematic
- **B–E** are data panels

The bottom row panels should be slightly wider than top-row single panels if needed, especially **E**.

A good arrangement would be:

- Row 1: **A** (schematic), **B** (subspace alignment summary), **C** (pairwise mimicry matrices)
- Row 2: **D** (mimicry crossover across LogMAR), **E** (phase landscape heatmaps)

If space allows, **C** can span a bit more width than B, because the mimicry matrices are important and visually rich.

---

## Visual style requirements

### Overall style

Aim for a polished, publication-ready, *Nature*-style look:

- clean white background
- minimal chartjunk
- thin axes
- consistent typography
- restrained but effective color usage
- clear hierarchy of emphasis
- intuitive annotations
- high data-to-ink ratio

### Typography

- Use a clean sans-serif font, for example Arial, Helvetica, or a close equivalent.
- Panel labels **A–E** should be bold and prominent.
- Axis labels and legends should be large enough to read when the figure is reduced to journal-column size.
- Avoid dense text inside panels. Use short annotations only.
- Keep notation simple. If equations appear, use at most one very small formula callout in panel A.

### Color palette

Use a restrained, coherent palette with the following specified hex codes:

| Role | Hex | Notes |
|---|---|---|
| **Real FEM** | `#1F4E79` (deep blue) or `#0E4D64` (teal) | choose one and use consistently |
| **Trial-mean stabilized** | `#C45A11` (rust) or `#D97706` (orange) | choose one and use consistently |
| **Null / controls** | `#9CA3AF` (medium gray) | for null bands, control traces |
| **Heatmaps** | viridis or magma | both are colorblind-safe and grayscale-distinguishable |

Requirements:
- colorblind-safe (all specified colors satisfy this)
- sufficient contrast in grayscale
- consistent use of real vs stabilized colors across **all** panels — the same hex must appear in panels B, C, D, and E

### Line and marker style

- Use medium-weight lines for main traces
- Use thinner lines for secondary overlays
- Use filled markers for main measured conditions
- Use hollow markers only for reference/anchor points if needed
- Keep error bars subtle but visible

### Figure dimensions

*Nature* uses **89mm single-column** and **183mm double-column**. This 5-panel 2-row figure should target:

- **Width: 183mm** (full double-column)
- **Height: 115–130mm**

Set these explicitly in the script (`figsize=(7.2, 4.7)` in inches, or `figsize=(183/25.4, 120/25.4)`). Do not produce a square figure or leave dimensions at matplotlib defaults. Getting the dimensions right at the start prevents re-scaling artifacts in font sizes and line weights.

### Aesthetic requirements

The figure should look intentional and elegant, not like a diagnostic script output.

In particular:
- align panel edges carefully
- standardize padding and spacing
- standardize axis styles
- standardize color scales when comparisons are intended
- add subtle visual emphasis to the key results
- do not overcrowd panels with too many numbers

---

## Data source expectations

Use the finalized outputs from the Jacobian / mimicry / phase-landscape analyses.

Likely relevant result files and summary docs include:
- `jacobian_identity_geometry_results(2).md`
- any cached numeric outputs used to generate:
  - subspace alignment vs null
  - pairwise mimicry matrices
  - mean mimicry across LogMAR for real and stabilized conditions
  - phase-landscape heatmaps (preferably fine 33 × 33 for lm = −0.20 and lm = −0.35)

The coding agent should locate and load the underlying numeric arrays rather than scraping text from markdown wherever possible.

If both coarse and fine phase landscapes exist, use the **fine 33 × 33** landscapes for the main figure. Coarse landscapes can go to supplement.

---

## Panel-by-panel specification

# Panel A. Conceptual schematic: Jacobian = local translation tangent plane

## Purpose

Introduce the key concept visually and intuitively for a broad audience.

This panel should explain:
- same stimulus at nearby retinal positions
- corresponding movement in population-response space
- Jacobian defines the local tangent plane
- if identity differences point into this plane, translation can mimic identity change

## Suggested design

Use a clean, stylized schematic, not raw model output.

Recommended structure:

### Left subpanel
Show a small E-optotype at one reference position, plus two slightly translated copies:
- center
- shifted slightly in x
- shifted slightly in y

Use arrows to indicate translation.

### Middle subpanel
Show a stylized mapping arrow from image space to population space.
Label lightly:
- “retinal translation”
- “population response”

### Right subpanel
Use a **2D projection with explicit cartoon style. Do not use perspective rendering or pseudo-3D.**

Show population-response space as a flat 2D diagram:
- a point representing the source orientation response
- the Jacobian tangent plane drawn as a **parallelogram**, with two labeled basis arrows: “Δx translation” and “Δy translation”
- a second point for a target orientation, connected to the first by the identity-difference vector **d_{a→b}**

**Crucially, show the decomposition explicitly:**

```
d_{a→b} = J_a Δp*  (in-plane component, “mimicked by translation”)
         + d⊥       (orthogonal component, “irreducibly identity”)
```

Draw this as:
- `J_a Δp*` as a labeled arrow lying in the tangent parallelogram
- `d⊥` as a labeled arrow perpendicular to the plane
- dashed projection lines from the tip of `d_{a→b}` down to the plane and along the normal, to make the decomposition visually legible

The schematic should make the reader understand: the identity-change vector has a part that *looks like a translation* (in-plane) and a part that *cannot be explained by any translation* (orthogonal). This decomposition is the conceptual core of the entire paper.

## Important constraints

- This should remain simple and conceptual.
- Do not use 3D perspective rendering. A clean 2D vector diagram with a parallelogram for the tangent plane is strongly preferred.
- Do not overload with equations.
- If you include notation, keep it minimal: short vector labels and the decomposition formula as a small callout.
- The in-plane vs orthogonal decomposition **must** be visually explicit — this is the highest-priority conceptual element of the entire figure.

## Suggested annotation

A small annotation inside or below panel A:

> Small eye movements move the response along a local translation plane.

or

> The Jacobian defines the local response directions for translating the same stimulus.

---

# Panel B. Jacobian plane aligns with FEM covariance

## Purpose

Show that the Jacobian is not an abstract derivative. It captures the dominant low-dimensional directions of FEM-induced variability.

## Metric

Plot **subspace alignment**, defined as the overlap between:
- the 2D Jacobian tangent plane
- the dominant 2D FEM covariance subspace

The manuscript text uses approximately 0.40–0.60 for alignment and says this is well above matched null expectations.

## Null definition

**The null must be defined precisely and consistently with the text.** The null distribution was computed as noise-corrected alignment via residual covariance — specifically, the expected subspace overlap when orientation labels are shuffled over matched-dimension random subspaces drawn from the residual covariance structure. Do not substitute a different null (e.g., uniform random subspaces, or uncorrected shuffles) without updating the caption accordingly. The actual null p95 is approximately 0.007, and measured alignment is roughly 2–4× above that threshold.

## Plot options

### Preferred option
A grouped dot/point-range plot or compact bar-plus-points summary showing subspace alignment across LogMAR and/or orientation, with null comparisons.

Possible x-axis structure:
- LogMAR values: −0.20, −0.25, −0.30, −0.35, −0.40

Color/overlay:
- real alignment values as colored points
- null distribution or null mean shown in gray

Alternative:
- show pooled distribution as violin/box + points
- or show per-condition stripplot with summary mean

## Best compromise for main figure

Use a **compact summary plot** that avoids clutter.

Recommendation:
- x-axis: LogMAR
- y-axis: subspace alignment
- colored points or line for measured values
- gray band or gray points for matched null (as defined above)
- small text annotation “2–4× above null” **only if computed directly from the loaded data**; do not add this annotation as plausible-sounding text without verifying it numerically against the actual outputs

If orientations are numerous and clutter the plot, use thin light points for individual orientation/condition values and a heavy mean line.

## Visual emphasis

The panel should clearly show:
- measured values cluster high
- null values are much lower

## Annotation

One short note in the panel if helpful:

> FEM variability lies in the Jacobian-predicted translation plane.

---

# Panel C. Pairwise translation mimicry matrices

## Purpose

Show that identity/transformation confusability is **anisotropic across orientation pairs**, not a single scalar nuisance magnitude.

## Main content

Display pairwise mimicry matrices for **two representative regimes**:
- **lm = −0.20**
- **lm = −0.35**

and for **two conditions**:
- real FEM
- trial-mean stabilized

This gives a 2 × 2 set of matrices.

## Matrix definition

Rows = source orientation  
Columns = target orientation  
Entry = fraction of identity difference explained by translation of the source orientation

Mask or blank diagonal entries.

## Layout

**Journal-size decision:** A 2×2 array of 4×4 matrices places 64 mimicry entries on a single panel, which risks unreadability at Nature column width. Apply the following rule before rendering:

> If each matrix would be smaller than 20mm square at the target figure dimensions (183mm wide), **reduce to a single matrix** (real FEM, lm = −0.20, since that is where pairwise structure is most striking) and move the full 2×2 array to supplement.

If each matrix is ≥ 20mm square, use four small heatmaps arranged as:

- top-left: real, −0.20
- top-right: stabilized, −0.20
- bottom-left: real, −0.35
- bottom-right: stabilized, −0.35

Use a **single shared colorbar** for all four matrices.

## Design details

- same color scale across all four matrices
- orientation labels should be text: **0°, 90°, 180°, 270°** — use small rotated E icons only if reliably implementable and not crowded at panel size; default to text
- diagonal masked in light gray or white
- use a perceptually uniform sequential scale

## Optional annotation

If space allows, highlight one or two notable pairs with unobtrusive boxes or arrows, for example:
- high mimicry pair
- low mimicry pair

But do not clutter.

## Take-home message

The reader should immediately see:
- some pairs are hot, some are cool
- the pattern changes with condition and scale

---

# Panel D. Mean mimicry across LogMAR, showing crossover

## Purpose

Show the scale-dependent crossover between real FEM and trial-mean stabilized conditions. This is one of the clearest demonstrations that confusability depends on sampling regime and stimulus scale.

## Main plot

Line plot of **mean pairwise mimicry** vs LogMAR.

### Traces
- real FEM: blue/teal
- trial-mean stabilized: orange/rust

### X-axis
- LogMAR values in order
- emphasize primary interpretable regime from −0.20 to −0.35
- include −0.40 if desired, but clearly mark it as a model-native saturation control

### Y-axis
- mean pairwise mimicry

### Error display
Use either:
- SEM / bootstrap CI around the mean
- or thin error bars

## Optional overlay
If space and readability permit, show **signal-to-tangent alignment α** as thinner dashed traces or in a small inset.

However, mimicry should remain the main focus. If α clutters the panel, move α to supplement.

## Essential visual message

The panel must make the crossover obvious:
- real FEM higher at −0.20 / −0.25
- stabilized higher by −0.35

Let the colored lines and shared legend carry the message. **Do not add separate left/right annotations (“real > stabilized”, “stabilized > real”) — these are redundant if the visual is clear and add clutter.** Instead, use at most a single annotation near the crossover point: a small arrow with the label “scale-dependent crossover”. If even this annotation crowds the panel, omit it entirely and rely on the trace plot and panel subtitle.

## Important caveat

If −0.40 is shown, visually indicate it as a **saturation control**, not as an independent interpretable size point. This can be done by:
- a shaded gray background region
- lighter line style
- small annotation “model-native saturation”

---

# Panel E. Fine phase landscape heatmaps

## Purpose

Show that identity/transformation confusability varies strongly over subpixel retinal phase.

This is the panel that explains why no single fixed phase, including fixed_center, can summarize the regime.

## Main content

Use **fine 33 × 33 phase-grid heatmaps** for **mean mimicry**.

Recommended to show **two heatmaps**:
- lm = −0.20
- lm = −0.35

These are the two key regimes.

If space is limited, show only mean mimicry, not multiple metrics.
Signal-to-tangent alignment can go to supplement if needed.

## Layout

Two heatmaps side by side (or one heatmap if overlays require it — see above).

Shared color scale.

Axes:
- x-axis: retinal phase x offset, **in arcminutes**
- y-axis: retinal phase y offset, **in arcminutes**
- At the model's 37.5 ppd, ±3 pixels = ±4.8 arcmin. Use this to label the axis range as **−4.8 to +4.8 arcmin** in both dimensions.
- Do not use “model pixels” as the axis unit — this requires the reader to know the model PPD. Use arcmin.

## Overlay elements

**Mandatory main-figure overlays:**
- mark the **fixed-center** point with a distinct marker, e.g. black circle with white edge
- overlay **trial-mean stabilized positions** as a point cloud or small scatter (this is what contextualizes the landscape biologicaly — it answers "where does the eye actually go on this landscape?")

These two overlays are required. The heatmap alone is descriptive; the overlays make the panel interpretive.

**Optional (include only if not cluttering the heatmap):**
- real FEM sample density contour

If the density contour would crowd the panel at journal size, omit it from the main figure and move it to supplement. Do not omit the trial-mean cloud or the fixed-center marker under any circumstances.

If both mandatory overlays would crowd the heatmaps, reduce from two heatmaps to one (lm = −0.35, the higher-dynamic-range landscape) and keep both overlays on the single heatmap.

## Important annotation

The panel should make clear that:
- high-mimicry pockets exist near the center
- fixed-center is not a special or representative point

Possible annotation:

> Nearby phases span most of the mimicry range.

or

> Subpixel position reshapes identity/transformation confusability.

---

## Figure-level caption guidance

The coding agent does not need to write the manuscript caption, but the visual structure should support a caption with this flow:

- A: Jacobian defines local translation tangent plane
- B: FEM covariance aligns with that plane
- C: translation mimicry quantifies pairwise identity/transformation confusability
- D: mimicry changes with LogMAR and shows a real-vs-stabilized crossover
- E: confusability varies strongly across subpixel retinal phase

The final figure should make this logic visually obvious.

---

## Required output files

Please generate:

1. **Main figure PDF**
   - publication quality vector PDF
   - final dimensions suitable for journal use

2. **Main figure PNG**
   - high-resolution raster export for review

3. **Editable source**
   - Python script or notebook that reproduces the figure from raw/intermediate data
   - if using Illustrator or Inkscape for final polish, also provide the editable file

4. **Panel-level exports**
   - optional but preferred, especially if later revision is needed

5. **Manifest / README**
   - **exact paths** to source data files used per panel (not just filenames — full paths from repo root)
   - software versions: matplotlib, numpy, scipy, and any other packages used
   - **random seeds** for any non-deterministic elements (null distributions, sample contours, bootstrap CIs)
   - the script **entry point** for regenerating the figure end-to-end from cached intermediate data — a single command should reproduce all output files
   - list any assumptions or data transformations
   - document fonts and colormaps

   Reproducibility is a hard requirement. The figure must be regenerable from scratch by running one command against the cached intermediate data. Figures that require manual steps, in-memory state, or undocumented preprocessing are not acceptable.

---

## Data and analysis priorities

### Highest priority main-figure content
These should definitely appear in the main figure:
- panel A schematic
- panel B subspace alignment vs null
- panel C pairwise mimicry matrices
- panel D mimicry crossover across LogMAR
- panel E fine phase landscape heatmaps

### Secondary content that may be omitted from the main figure if cluttered
- signal-to-tangent alignment α in panel D
- trial-mean / real phase clouds in panel E
- exact numerical annotations inside matrices

### Content for supplement instead
- full mimicry matrices for all LogMARs
- raw vs z-scored comparisons
- projection vs ridge-LS agreement
- coarse vs fine phase landscape comparison
- saturation audit details
- additional phase-landscape metrics beyond mean mimicry

---

## Accessibility and broad-audience clarity requirements

This figure must be understandable to readers outside the immediate subfield.

Therefore:
- panel A must do a lot of conceptual work
- panel titles or short subtitles are encouraged if tasteful
- avoid unexplained shorthand on the figure itself
- use orientation labels and simple annotations rather than dense mathematical notation
- distinguish clearly between “same stimulus translated” and “different stimulus identity”

A good test:
- a visually literate neuroscientist who has not read the full paper should be able to look at the figure and understand the main point in under 2 minutes

---

## Recommended panel subtitles

If the design supports short panel subtitles, use something close to:

- **A**  Local translation directions in population space
- **B**  FEM covariance aligns with the Jacobian plane
- **C**  Translation mimicry is pair-specific
- **D**  Confusability changes with scale and sampling regime
- **E**  Confusability varies across retinal phase

These subtitles should be subtle and not visually compete with the main content.

---

## Quantitative details to preserve

Make sure the figure and scripts preserve the following key claims from the finalized analyses:

- Subspace alignment is high, approximately **0.40–0.60**, and clearly above null.
- Mean pairwise mimicry in raw-rate space is substantial, approximately **0.25–0.49** depending on condition and LogMAR.
- The **real vs stabilized crossover** is visible:
  - real higher at **−0.20 / −0.25**
  - stabilized higher by **−0.35**
- Pairwise mimicry is strongly anisotropic.
- Fine phase landscapes show large dynamic ranges in mimicry:
  - at **−0.20**, approximately **0.0908–0.7394**
  - at **−0.35**, approximately **0.1233–0.8724**

Do not overload the figure with these numbers, but make sure the plotted data reflect them faithfully.

---

## Suggested workflow for the coding agent

1. Locate and load the finalized numerical outputs for:
   - subspace alignment and null
   - pairwise mimicry matrices
   - mean pairwise mimicry across LogMAR
   - fine phase landscapes

2. Build each panel first as a clean standalone draft.

3. Harmonize style across panels:
   - fonts
   - axis line weights
   - panel spacing
   - colors
   - labels
   - color scales

4. Build the full multipanel figure.

5. Iterate on:
   - readability at reduced size
   - visual balance across panels
   - clarity of the main story

6. Produce the final polished PDF and PNG.

---

## Quality-control checklist

Before finalizing, verify:

- [ ] **Panel A (highest risk):** uses 2D projection, no pseudo-3D perspective rendering
- [ ] **Panel A:** explicitly shows the in-plane / orthogonal decomposition of d_{a→b} with labeled arrows and dashed projection lines
- [ ] **Panel A:** the conceptual core (mimicry = in-plane component) is visually obvious without reading the caption
- [ ] **Panel B:** null is plotted using the noise-corrected residual-covariance definition, not a generic shuffle null
- [ ] **Panel B:** "2–4× above null" annotation (if present) was computed from the actual loaded data, not written as assumed text
- [ ] **Panel C:** each matrix is ≥ 20mm square at target figure size, or reduced to single matrix per the decision rule
- [ ] Panel C matrices share a common color scale and are legible
- [ ] **Panel D:** crossover is visually obvious; no redundant "real > stabilized / stabilized > real" side annotations
- [ ] **Panel E:** trial-mean stabilized position cloud overlay is present (mandatory)
- [ ] **Panel E:** fixed-center point is clearly marked
- [ ] **Panel E:** axes are labeled in arcminutes (±4.8 arcmin range), not model pixels
- [ ] Figure is 183mm × 115–130mm (verify actual output dimensions)
- [ ] Colors match specified hex codes and are consistent across panels B, C, D, E
- [ ] All axis labels and legends are readable at publication size
- [ ] The figure remains understandable in grayscale printing
- [ ] No panel contains unnecessary clutter or over-annotation
- [ ] Manifest/README records exact file paths, software versions, and random seeds
- [ ] The figure can be regenerated end-to-end with a single command
- [ ] The figure tells a coherent single story from left to right, top to bottom

**Panel A risk note:** Even with the more prescriptive guidance above, Panel A is the most likely panel to require human revision. A coding agent may produce something technically correct but visually flat. Budget for one round of human touch-up on Panel A specifically, or provide a hand-drawn reference sketch for the agent to match stylistically.

---

## Final design philosophy

This figure should feel like the **visual centerpiece** of the Jacobian result section.

It should not read like a collection of diagnostics. It should read like a clear argument:

1. FEMs move the response along a predictable translation plane.
2. That translation plane can mimic some identity changes.
3. Which identity changes are confusable depends strongly on retinal phase.

If done well, the figure should make it obvious why the Jacobian matters for understanding neural encoding: it provides a local geometric substrate for separating, or failing to separate, **stimulus identity** from **retinal transformations caused by the animal’s own eye movements**.
