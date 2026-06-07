# Updated covTFTS Figure 4 Prescription: Data-forward mechanism figure

## Working figure title

**Figure 4. Image-specific retinal translations form a compact, image-generalizing reafferent geometry.**

## Revised figure strategy

The previous figure worked as a conceptual synthesis, but it leaned too heavily on schematics. The revised Figure 4 should function as a **mechanism/results figure with schematic insets**, not a visual abstract. The central claim is subtle and must be shown with data:

> Translation tangents are image-specific, but they are not arbitrary. They occupy a compact, image-generalizing population geometry in the canonical twin.

The figure should therefore make three distinctions visible:

1. **Residual noise versus reafferent covariance**  
   The biological anchor is recorded V1: FEM correction reduces classical shared variability, and the removed covariance is low-dimensional and aligned with stimulus-driven structure.

2. **Universal displacement axis versus compact tangent family**  
   The twin result is not that all images share one signed x/y translation axis. The result is that image-specific tangents differ across content but occupy a compact, shared population subspace.

3. **Local tangent geometry versus full sampled FEM covariance**  
   Local tangents provide a meaningful but incomplete bridge to full sampled FEM covariance. This is a partial mechanism, not an identity.

## Key design principle

Use cartoons only as **insets or scaffolding around real data**. The figure should feel empirically compelled, not merely logically explained.

The most important new visual object should be an actual representation of the tangent family, such as:

- a tangent-vector similarity/correlation matrix,
- a low-dimensional projection of tangent vectors,
- an eigenspectrum or cumulative-variance plot of the tangent covariance,
- or a combination of these.

The reader should directly see that the tangents are neither identical nor random.

## Revised panel structure

Use six panels, but with more data weight than the previous version.

```text
A. Recorded reafferent covariance anchor
B. Tangents are image-specific but structured
C. Compact tangent spectrum
D. Compactness across displacement scale
E. Image-disjoint generalization
F. Partial covariance bridge
```

If space is limited, Panel D can become an inset within Panel C. If a real tangent similarity/projection panel is hard to generate quickly, Panel B can use a simplified PCA projection of the tangent vectors rather than a heatmap.

---

# Panel A. Recorded reafferent covariance anchor

## Goal

Anchor the figure in recorded foveal V1, using real summarized data rather than a cartoon.

## Scientific message

> FEM-linked covariance is a measured biological object: conditioning on eye state reduces classical shared variability, and the removed covariance is low-dimensional and aligned with stimulus-driven population structure.

## Recommended content

Use a compact set of real callbacks from the recorded analysis. Preferred options:

1. **Noise-correlation correction callback**
   - uncorrected versus FEM-corrected mean noise correlation,
   - or paired/session summary of correction effect,
   - or a compressed version of the uncorrected-vs-corrected scatter.

2. **Low-dimensional FEM covariance callback**
   - eigenspectrum of `Σ_FEM`,
   - or participation-ratio summary showing FEM covariance concentrated in roughly two modes.

3. **Stimulus-alignment callback**
   - subspace overlap between `Σ_FEM` and `Σ_PSTH`,
   - or variance-capture summary showing FEM variance lies in the PSTH/stimulus-driven subspace.

## Minimal acceptable version

If space is tight, use two mini-plots:

- left: uncorrected vs corrected shared variability,
- right: FEM/PSTH subspace alignment or FEM eigenspectrum.

## Panel title

**Recorded V1: FEM correction exposes low-dimensional reafferent covariance**

## Notes

Do not use a purely schematic scatter as the recorded anchor. This panel should look empirical.

---

# Panel B. Tangents are image-specific but structured

## Goal

Show the missing visual object: the tangent family itself.

## Scientific message

> Tangents differ across image histories, but their population directions are not arbitrary.

## Preferred visualization options

### Option 1: Tangent similarity matrix

Construct a matrix of pairwise cosine similarities or signed/absolute similarities among tangent vectors.

Possible organization:

```text
rows/columns = tangent vectors from image-history objects
entries = cosine similarity, absolute cosine similarity, or projection similarity
```

Useful ordering options:

- group by image identity,
- group by displacement direction (`b_x`, `b_y`),
- sort by projection onto leading tangent PC,
- or cluster by similarity.

Include a matched null matrix:

- unit-shuffled tangent vectors,
- or random-unit permutation null,
- or show null distribution as a side histogram.

The ideal panel would show:

```text
observed tangent similarity has visible block/low-dimensional structure
unit-shuffled null is diffuse / structureless
```

### Option 2: Tangent PCA projection

Project tangent vectors into the first two or three PCs of the tangent family.

Plot:

- each point = one tangent vector,
- color = displacement direction (`b_x` versus `b_y`) or image identity,
- shape/alpha = image/history object.

This is visually intuitive but less rigorous than the similarity matrix.

### Option 3: Example images plus tangent projections

Show a small set of example image-history objects and the corresponding `b_x(I), b_y(I)` vectors projected into the first two tangent PCs.

The visual should show that signed directions differ across images while remaining within a common region of tangent space.

## Recommended implementation

Use Option 1 if feasible. Use Option 2 if a similarity matrix is too cluttered or the raw tangent vectors are easier to project than to arrange.

## Panel title

**Image-specific tangents differ, but are not arbitrary**

## Required annotation

Include a short statement:

> Not a universal signed x/y axis

and, if using a null comparison:

> Unit-shuffled null destroys the shared structure

## Notes

This panel replaces the previous full schematic “why compactness is nontrivial” panel. The conceptual point can be included as a small inset:

```text
single image: local 2D plane
across images: planes need not align
```

but the main visual should be data.

---

# Panel C. Compact tangent spectrum

## Goal

Make compactness visually obvious as a spectral property, not only as a three-point participation-ratio summary.

## Scientific message

> The tangent family occupies a compact population subspace relative to unit-shuffled controls.

## Recommended plot

Primary plot:

```text
x-axis: tangent PC / component number
y-axis: cumulative variance explained
```

or:

```text
x-axis: component number
y-axis: normalized eigenspectrum / variance fraction
```

Show:

- observed tangent family,
- unit-shuffled null,
- preferably at the primary displacement scale, 0.25 arcmin.

For cumulative variance, the observed curve should rise faster than the null. For eigenspectrum, the observed spectrum should be more concentrated.

## Annotation

Include the primary PR value:

> 0.25 arcmin: PR ≈ 9.0 vs null ≈ 31.0

## Panel title

**Tangent variance is concentrated in a compact subspace**

## Notes

This panel should be one of the main quantitative anchors. If cumulative variance data are not directly saved, compute from the saved tangent matrix or union spectrum. If only PR values are available, keep the PR plot but recognize that the panel will be visually weaker.

---

# Panel D. Compactness across displacement scale

## Goal

Show that compactness is not a one-off result at a single finite-difference scale.

## Scientific message

> Tangent-family compactness is present across the tested local displacement scales.

## Recommended plot

Use the existing participation-ratio summary across displacement scales:

```text
x-axis: tangent displacement scale (arcmin)
y-axis: participation ratio
curves: observed versus unit-shuffled null
```

Existing values:

```text
0.125 arcmin: PR ≈ 7.8 vs null ≈ 27.5
0.250 arcmin: PR ≈ 9.0 vs null ≈ 31.0
0.500 arcmin: PR ≈ 6.6 vs null ≈ 24.3
```

## Panel title

**Compactness persists across local displacement scale**

## Notes

This panel has only three x-axis points. It should therefore be smaller than Panels C and E, or appear as an inset within Panel C, unless additional displacement scales are available from a finalized run.

Do not imply this is a broad scale analysis. It is a local finite-difference sensitivity check.

---

# Panel E. Image-disjoint generalization

## Goal

Provide the load-bearing control that compactness is not same-image leakage.

## Scientific message

> A tangent basis learned from one set of image identities captures tangents from held-out image identities above null.

## Plot

Keep the basis-capture curve, but make it more data-rich if possible.

Recommended content:

```text
x-axis: basis dimension k
y-axis: held-out tangent variance captured
curves: observed image-disjoint result versus unit-shuffled null
```

Add, if available:

- fold-level thin traces,
- confidence or bootstrap bands,
- all displacement scales as faint background or small multiples, with 0.25 arcmin highlighted.

Primary annotation:

> k = 10: ≈0.55 held-out capture vs ≈0.12 null  
> image-disjoint split, 0% image-ID leakage

## Panel title

**A compact basis generalizes to held-out image identities**

## Notes

This is probably the strongest panel. It should be large and clean. Do not show object-random results as the main curve. Do not include fallback warnings or preview labels.

---

# Panel F. Partial covariance bridge

## Goal

Show that the tangent geometry relates to full sampled FEM covariance, but only partially.

## Scientific message

> Local tangents align with full sampled FEM covariance most strongly for local eye-position clouds, and less strongly for broader finite-displacement clouds.

## Recommended plot

Existing plot:

```text
x-axis: eye-position cloud scale
y-axis: subspace overlap with full sampled FEM covariance
curve: tangent-predicted covariance versus sampled covariance
```

Existing primary-scale values for 0.25 arcmin:

```text
cloud scale 0.25: overlap ≈ 0.395
cloud scale 0.50: overlap ≈ 0.344
cloud scale 1.00: overlap ≈ 0.295
cloud scale 2.00: overlap ≈ 0.245
```

## Required improvement

Add a reference if available. Preferred reference options:

1. unit-shuffled tangent basis null,
2. random tangent basis null,
3. self-split reliability ceiling for the sampled covariance,
4. same-object tangent upper bound,
5. or a clearly labeled chance/reference line.

If no null or ceiling is available, keep this panel visually modest and explicitly label it:

> Diagnostic bridge, not a full explanation

## Panel title

**Local tangents provide a partial bridge to FEM covariance**

## Axis label

Use a precise metric label, such as:

> Subspace overlap with sampled `Σ_FEM`

rather than vague “shared variability.”

## Notes

This panel should not be titled “Tangents explain FEM covariance.” It should preserve the incomplete nature of the bridge.

---

# Optional schematic inset

The previous conceptual logic can be retained as a small schematic inset, but not as a full panel.

Suggested inset text:

```text
For one image: retinal translation defines a local 2D response neighborhood.
Across images: compactness is not guaranteed.
Observed in twin: image-specific tangents form a compact family.
```

This inset can live in Panel B or at the top of Panel C.

---

# Revised draft figure legend

**Figure 4. Image-specific retinal translations form a compact, image-generalizing reafferent geometry.**  
(A) Recorded foveal V1 contains a reafferent covariance component. Conditioning responses on measured eye state reduces classical shared variability, and the removed `Σ_FEM` component is low-dimensional and aligned with stimulus-driven population structure. This recorded result is the biological object the model-side analysis seeks to explain.  
(B) In the canonical twin, small retinal translations of each full stimulus-history object define image- and history-specific response tangents. These tangents are not expected to share a universal signed x/y direction. A direct visualization of the tangent family shows that tangents differ across image histories but retain structured similarity relative to unit-shuffled controls.  
(C) The tangent family occupies a compact population subspace. At the primary displacement scale of 0.25 arcmin, tangent variance was concentrated in a low-dimensional spectrum, with participation ratio approximately 9.0 compared with approximately 31.0 under the unit-shuffled null.  
(D) Compactness was stable across the tested local finite-difference scales. Across 0.125, 0.25, and 0.5 arcmin displacements, the observed tangent-family participation ratio remained far below the corresponding unit-shuffled nulls.  
(E) The compact tangent structure generalized across image identities. In image-disjoint train/test splits, a tangent basis learned from one set of image identities captured held-out image tangents above unit-shuffled controls. At 0.25 arcmin, a 10-dimensional basis captured approximately 0.55 of held-out tangent variance, compared with approximately 0.12 under the null, with no image-ID overlap between train and test folds.  
(F) Local tangents provided a meaningful but incomplete bridge to full sampled FEM covariance. Tangent-predicted covariance aligned most strongly with sampled covariance for local eye-position clouds and less strongly for broader finite-displacement clouds, consistent with the idea that real FEMs sample curved and history-dependent response neighborhoods rather than an exactly linear local plane.

Together, these panels distinguish three objects that are easily conflated. The recorded V1 result is a low-dimensional, stimulus-aligned reafferent covariance. The canonical twin supplies a model-side mechanism for why such covariance can remain compact across content: retinal translations generate content-dependent tangents, but those tangents form a compact, image-generalizing family. The tangent-to-covariance bridge is partial, so the result does not imply a universal signed displacement axis, a complete linear account of FEM covariance, or direct proof of the same tangent subspace in recorded V1.

---

# Figure-generation priorities

1. Replace schematic-heavy Panels B/C with a real tangent-family visualization.
2. Make Panel A data-anchored using compressed recorded results.
3. Make compactness spectral if possible, not only a three-point PR plot.
4. Keep image-disjoint generalization as the strongest quantitative panel.
5. Add a null or ceiling to Panel F if available; otherwise label it explicitly as a diagnostic partial bridge.
6. Avoid adding ecology, optotype, or decoding/performance content.
