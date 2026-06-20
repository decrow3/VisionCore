# Figure 4 Active Sensing Atlas

This folder is the document-first workspace for the expanded Figure 4 atlas.
It turns the 2026-06-19 atlas brief into a local, cache-first plan for deciding
what survives into the compressed main figure.

The organizing claim is:

```text
Fixational eye movements transform static images into informative retinal
movies. V1 responses contain structure that can use those movies, but only if
retinal motion is treated as part of the inference problem rather than as noise.
```

## Ground Rules

- Prefer existing caches, result tables, figures, and summary scripts.
- Do not add new workhorse analysis code unless a documented figure panel has no
  credible existing source.
- Keep claim boundaries explicit: twin-scoped, deterministic-decoding proxy,
  exact-cache observer, compact-mechanism, local axis prediction, and behavior
  are separate evidence layers.
- Record every promoted result in `provenance_ledger.md` before using it in the
  atlas prose.

## Files

- `figure4_active_sensing_atlas.md`
  - Working document skeleton for the five expanded modules.
- `working_results_draft.md`
  - First readable Results draft with panel-level source values and inline
    flags.
- `atlas_build_plan.md`
  - Stepwise plan for turning the atlas into figure-ready prose and panels.
- `panel_source_map.md`
  - Panel-by-panel map from the atlas brief to existing code/results.
- `panel_manifest.csv`
  - Build-facing manifest of candidate atlas panels, status, sources, and
    flags.
- `main_figure_compression_v0.md`
  - First candidate compressed main Figure 4.
- `panel_text_atlas.md`
  - Human-facing contact sheet with every generated subpanel, read, role,
    boundary, and active flags.
- `panel_text_atlas_compact.md`
  - LLM-friendly composite-first atlas with one module image per section and a
    single main-spine composite.
- `panel_text_atlas.pdf`
  - PDF export of the panel text atlas with embedded images.
- `supplement_routing_v0.md`
  - First supplement routing map for QC, controls, and mechanism panels.
- `provenance_ledger.md`
  - Local notes on which analyses, caches, and result folders support each
    claim.
- `incomplete_results_flags.md`
  - Running list of incomplete, unresolved, or claim-limited result branches.
- `claim_critical_diagnostics_queue.md`
  - Consolidated queue of anticipated failure modes, diagnostics, and
    promotion/demotion gates for core or claim-critical analyses.
- `scripts/`
  - Cache-only helper scripts for atlas-specific figure panels.
  - `scripts/build_panel_composites.py` regenerates
    `figures/composites/*_composite.png` from the current subpanel PNGs.
  - `scripts/export_panel_text_atlas_pdf.py` regenerates
    `panel_text_atlas.pdf` from the Markdown and local PNGs without external
    PDF dependencies.
- `figures/`
  - Atlas-specific generated panels and extracted source values.
  - `figures/panel_A/` contains the cache-only Panel A premise/QC subpanels
    and source value tables.
  - `figures/panel_B/` contains the cache-only Panel B subpanels and source
    value tables.
  - `figures/panel_C/` contains the cache-only Panel C subpanels and source
    value tables.
  - `figures/panel_D/` contains the cache-only Panel D subpanels and source
    value tables.
  - `figures/panel_E/` contains the cache-only Panel E subpanels and source
    value tables.
  - `figures/composites/` contains LLM-friendly module and main-spine composite
    PNGs.

## Current Evidence Spine

1. Module A: retinal-motion premise and rendering/QC now have cache-only
   atlas subpanels, with A5 kept as a covariance bridge/guardrail.
2. Module B: aggregate BackImage feature-decoding results support
   distributional FEM-like motion benefit over static and OU-like controls, with
   strongest Brownian/rotated specificity at small scales.
3. Module C: BackImage exact trajectory-table observer shows known-eye >
   joint-eye > zero-eye, including a matched-static-response control.
4. Module D: axis-conditioned observer and edge-parallel stability results
   support image-dependent axes, but the preferred axis remains objective and
   candidate-set dependent.
5. Module E: behavior supports modest but reliable alignment of drift/fixation
   cloud orientation with local image geometry; current model objectives do not
   yet beat raw edge geometry cleanly.
