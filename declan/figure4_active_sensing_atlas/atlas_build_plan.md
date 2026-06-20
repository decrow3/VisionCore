# Atlas Build Plan

Status: draft plan for the local Figure 4 atlas document.

## Objective

Build an expanded Figure 4 atlas document in this folder before compressing the
main figure. The atlas should be assembled mostly by subtraction from existing
results: first gather all viable modules, then decide which panels survive into
the main figure and which become supplements.

## Working Product

The working document is:

```text
declan/figure4_active_sensing_atlas/figure4_active_sensing_atlas.md
```

It should eventually contain:

- a short Results-section lead;
- the five expanded modules A-E;
- each panel's visual role, supported claim, source result, and claim boundary;
- a candidate main-figure compression;
- supplement routing.

## Cache-First Workflow

1. Lock the narrative spine.
   - Start from active sensing and latent retinal pose, not compact geometry.
   - Keep compact geometry as a mechanism panel after the joint observer.

2. Build the source ledger before adding numbers to prose.
   - Use `provenance_ledger.md` as the authority for paths, run status, and
     claim boundaries.
   - Every numeric claim in the atlas should point to one source family in the
     ledger.

3. Draft modules A-E in full.
   - For each module, choose the minimum set of main panels and a larger
     supplement set.
   - Prefer existing figures/QC panels when they already communicate the point.

4. Promote evidence by strength.
   - Main-ready: cleaned aggregate FEM information, matched-static joint
     observer, behavior alignment.
   - Main or supplement depending on story: edge-parallel preservation and
     axis-conditioned observer.
   - Mechanism supplement unless it becomes load-bearing: compact projection.
   - Supplement or methods: Vernier failure, posterior diagnostics, motion QC.

5. Add new code only for figure assembly or cache summaries.
   - Acceptable: small plotting scripts or table extractors that consume
     existing CSV/NPZ outputs.
   - Avoid: new V1-twin inference, new trajectory generation, or new observer
     runs until a documented panel gap requires them.

## Module Decisions

### A: Retinal Movie Premise

Use existing cartoon/QC material where possible. This module should be visually
simple and explanatory. It can lean on previous covariance figures only as a
bridge.

Promotion gate:

```text
Reader understands why a fixed image becomes a moving retinal input and why
image content makes the response changes structured.
```

### B: Encoding Benefit

Use the cleaned BackImage aggregate run as the primary evidence. The main panel
should show static-plus-motion gain and empirical-minus-control contrasts. The
scale guardrail is important because it protects against the "more motion is
always better" critique.

Promotion gate:

```text
Show positive empirical temporal-PCA gains over static, robust OU advantage,
and small-scale Brownian/rotated specificity, with explicit proxy language.
```

### C: Joint Observer

Use exact-cache trajectory-table observer results. The main claim is readout
feasibility under latent pose, not compact geometry. Matched-static distractors
are the strongest panel because they rule out the easiest static-response
explanation.

Promotion gate:

```text
Show known-eye high, zero-eye impaired, joint-eye above zero-eye, and posterior
concentration that is partial but meaningful.
```

### D: Image-Dependent Motion Directions

Treat D as prediction-space and mechanism-space. Axis-conditioned observer
results show axis priors help, while edge-parallel preservation gives a clean
local geometric explanation. The preferred axis is not yet universal.

Promotion gate:

```text
Phrase as image-dependent useful axes and objective dependence, not as a solved
parallel-versus-orthogonal biological law.
```

### E: Behavioral Test

Use drift-edge alignment and raw-edge baseline as the clean behavioral result.
Model objectives are useful comparisons but not yet the strongest explanation.

Promotion gate:

```text
Show that free-viewing drift/fixation-cloud orientation is modestly but
reliably aligned with local image geometry, and state what the model does not
yet explain.
```

## Near-Term To-Do

- Add a citation check note before manuscript use. The pasted brief included
  external PubMed/Nature references; this folder has not verified them.
- Decide whether the atlas should import existing PNG/PDF panels by path or
  rebuild them into a unified figure style.
- Add a small `figures/` subfolder only if we start generating atlas-specific
  panel composites.
- Add `decision_log.md` after the first compression pass, so rejected panels
  have a traceable reason.

