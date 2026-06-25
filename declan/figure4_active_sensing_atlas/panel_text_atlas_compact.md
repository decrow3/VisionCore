# Figure 4 Active Sensing Compact Panel Atlas

Status: LLM-friendly composite companion to `panel_text_atlas.md`.

2026-06-21 correction note: Panel B temporal-PCA absolute-gain prose/images are
superseded. Use the corrected static-mean aggregate posthoc and the all-readout
audit: `mean`/`delta_mean` are absolute aggregate candidates, `delta_mean` is
the local mechanistic bridge, and temporal PCA/DCT variants are order-sensitive
empirical-vs-control diagnostics. Regenerate the contact sheets if every
embedded old Panel B note must be removed.

Use this file when the goal is to read the figure logic quickly. The detailed
contact sheet remains the source for per-subpanel provenance, caveats, and
supplement routing.

## Main Spine Composite

![Main spine composite](figures/composites/main_spine_composite.png)

Current compressed story:

```text
A: FEMs convert a fixed screen image into a retinal movie.
B: Empirical drift-like movies produce feature-relevant response changes.
C: A joint image-and-eye observer preserves image-feature information when eye
   position is latent.
D: Local image geometry defines useful motion axes, with guardrails.
E: Measured free-viewing FEM axes align with local image geometry.
```

Primary candidate panels:

```text
A1/A2/A4, B3/B4, C2/C3, D1/D4, E2/E3
E6/E7/E8 travel with E3 as behavior-metric provenance/supplement.
```

## Module A: Retinal Movie Premise

![Module A composite](figures/composites/module_A_composite.png)

Load-bearing read:

```text
During fixation, a fixed screen image becomes a retinal movie. The QC panels
show that FEM movies add temporal contrast and motion power relative to a
stabilized counterfactual while preserving matched movie power. A4 records the
BackImage/V1-twin provenance bridge for downstream analyses.
```

Boundary:

```text
Module A licenses the physical premise, not functional optimality. A5 is a
covariance bridge/supplement candidate because its denominators are mixed.
```

## Module B: FEM Movies Add Feature-Decodable Structure

![Module B composite](figures/composites/module_B_composite.png)

Load-bearing read:

```text
The corrected aggregate BackImage run shows a readout split. Mean/delta-mean
readouts test whether motion-derived response summaries add feature-decodable
signal beyond the static mean response. Temporal PCA/DCT variants preserve
trajectory order and test empirical-vs-control specificity. OU-like confined
motion is audit-pending rather than a settled headline null.
```

Boundary:

```text
This is a distributional feature-decoding claim, not literal mutual information
and not exact trajectory optimality. Do not present temporal PCA as the
absolute gain-over-static headline unless it survives fair static-baseline and
nested-regularization gates.
```

## Module C: Joint Image-And-Eye Observer

![Module C composite](figures/composites/module_C_composite.png)

Load-bearing read:

```text
The promoted continuous no-anchor observer is best judged by posterior feature
recovery rather than exact image identity. In the verified full
scale-calibrated artifact, posterior expected features reach mean cosine 0.9358
to the true image feature, while exact image accuracy is 0.7083. The older
finite trajectory-table observer remains useful context: it shows known-eye
highest, zero-eye impaired, and joint-eye inference recovering much of the
known-minus-zero image-identity gap by marginalizing over plausible
trajectories.
```

Boundary:

```text
Feature recovery is the primary C diagnostic; exact image identity is a hard
secondary endpoint. The rendered C composite still mostly reflects the
finite-cache lineage. Compact geometry is a sufficiency/mechanism bridge, not
unique mechanism proof.
```

## Module D: Image-Dependent Useful Motion Directions

![Module D composite](figures/composites/module_D_composite.png)

Load-bearing read:

```text
Useful motion axes are defined relative to local image geometry. Edge-parallel
motion strongly preserves local pixels and V1-twin responses relative to
orthogonal motion, while axis-conditioned observer preferences depend on
candidate set and scale.
```

Boundary:

```text
Do not compress Module D into a universal "parallel is always best" or
"orthogonal is always best" claim. Current V1-twin response objectives do not
yet beat raw edge geometry as behavior predictors.
```

## Module E: Free-Viewing FEMs Follow Image Geometry

![Module E composite](figures/composites/module_E_composite.png)

Behavior metric provenance sheet:

![Module E contour-following diagnostics](figures/composites/module_E_contour_following_diagnostics.png)

Load-bearing read:

```text
Measured free-viewing FEM axes are modestly but reliably aligned with local edge
geometry. Endpoint zones are enriched near edge-parallel directions, especially
for reliable/high-confidence windows. E6-E8 restore the original distribution,
confidence, and endpoint-null diagnostics behind the compact E3 summary.
```

Boundary:

```text
Module E supports behavior-geometry alignment, not proof that the tested
response objective is optimized. Metric-convention differences should remain
visible in caption or supplement, and E8 should travel with E3 whenever the
endpoint-zone metric is explained.
```

## Remaining Global Flags

```text
F002: final composite selection and typography remain open.
F003: feature-decoding proxy, not literal mutual information.
F004: empirical specificity is scale/control dependent.
F005: local exact image-trace pairing remains sensitivity/supplemental.
F006: compact mechanism is sufficient/supportive, not unique.
F007: axis preference depends on candidate set and scale.
F008: edge-parallel preservation is local stability, not full objective.
F009: raw edge geometry remains the behavior baseline to beat.
F011: promoted C is the continuous feature-recovery observer, but the rendered
      panels still need final integration.
F012: behavior metric convention must stay explicit.
```

## Regeneration

```bash
.venv/bin/python -m declan.figure4_active_sensing_atlas.scripts.build_panel_composites
```
