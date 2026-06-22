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
- `provisional_panel_contract_v0.csv`
  - Cache-first provisional main/supplement panel contract for review while
    figure assets are being promoted from the completed power reruns.
- `provisional_figure4_v0.md`
  - Provisional Figure 4 package with selected panel roles, values, claim
    boundaries, and a draft legend.
- `4a_companion_retinal_movie_premise.md`
  - Reasoning document for the retinal-movie premise and rendering QC panel.
- `4b_companion_aggregate_fem_model.md`
  - Reasoning document for the aggregate FEM feature-decodability model.
- `4b_companion_local_Iz_pairing_model.md`
  - Reasoning document for the local `I_z` image-trace pairing sensitivity
    model.
- `4c_companion_joint_posterior_observer_model.md`
  - Reasoning document for the joint image/trajectory observer model.
- `4d_companion_along_edge_model_feature_encoding.md`
  - Reasoning document for the along-edge model feature-encoding panel.
- `panel_C_feature_space_compact_removed_handoff.md`
  - Coding-agent handoff for the missing Panel C feature-space compact-only /
    compact-removed / addback control.
- `4e_companion_behavior_geometry_bridge.md`
  - Reasoning document for the behavior contour-following bridge and raw-edge
    boundary.
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
  - `scripts/build_selected_figure4.py` regenerates
    `figures/composites/figure4_selected_v3.*` from the selected A-E
    promotion candidates.
  - `scripts/build_selected_figure4_v4_design.py` regenerates
    `figures/composites/figure4_selected_v4.*` as a larger design-first
    composite, redrawing the quantitative panels in a shared visual system.
  - `scripts/build_selected_figure4_v5_compact_layout.py` regenerates
    `figures/composites/figure4_selected_v5.*` as the current compact-layout
    design draft with A/B on the top row.
  - `scripts/build_panel_c_feature_recovery_options.py` regenerates the focused
    Panel C feature-recovery option sheet in
    `figures/panel_C/promotion_candidates/feature_recovery_options/`.
  - `scripts/export_panel_text_atlas_pdf.py` regenerates
    `panel_text_atlas.pdf` from the Markdown and local PNGs without external
    PDF dependencies.
- `figures/`
  - Atlas-specific generated panels and extracted source values.
  - `figures/panel_A/` contains the cache-only Panel A premise/QC subpanels
    and source value tables. `figures/panel_A/promotion_candidates/` contains
    the current single-panel 4A promotion candidates, including real BackImage
    image-set and recorded-fixation variants of A1.
  - `figures/panel_B/` contains the cache-only Panel B subpanels and source
    value tables. `figures/panel_B/promotion_candidates/` contains the current
    single-panel 4B promotion candidates, but the previously selected
    temporal-PCA absolute-gain panel is superseded by the corrected static-mean
    posthoc and needs redraw.
  - `figures/panel_C/` contains the cache-only Panel C subpanels and source
    value tables. `figures/panel_C/promotion_candidates/` contains the current
    single-panel 4C promotion candidates; the current composite uses feature
    recovery option 5, the zero-eye / compact-subspace / known-eye ceiling
    panel, with image-identity and compact-removal audits kept as context.
  - `figures/panel_D/` contains the cache-only Panel D subpanels and source
    value tables. `figures/panel_D/promotion_candidates/` contains the current
    single-panel 4D promotion candidates, with candidate 1 selected as the
    edge-parallel preservation mechanism support panel.
  - `figures/panel_E/` contains the cache-only Panel E subpanels and source
    value tables. `figures/panel_E/promotion_candidates/` contains the current
    single-panel 4E promotion candidates, with candidate 3A selected as the
    behavior image-coherence bridge.
  - `figures/composites/` contains LLM-friendly module and main-spine composite
    PNGs plus the selected provisional `figure4_selected_v5.png` compact-design
    composite.

## Current Evidence Spine

1. Module A: retinal-motion premise and rendering/QC now have cache-only
   atlas subpanels, with A5 kept as a covariance bridge/guardrail.
2. Module B: the completed aggregate BackImage power rerun now uses a corrected
   static-mean baseline. The current target is a role split:
   `mean`/`delta_mean` are absolute aggregate candidates, `delta_mean` is the
   local mechanistic bridge, and temporal PCA/DCT variants are order-sensitive
   empirical-vs-control diagnostics. The Panel B candidate and selected v5
   composite have been redrawn once from the corrected posthoc, but the
   all-readout Panel-B-style atlas and OU trace-control audit should be reviewed
   before any write-lock.
3. Module C: BackImage feature-posterior observer shows that zero-eye feature
   recovery degrades with motion scale, while latent-eye joint inference
   remains stable without known eye position; the matching
   feature-space compact-removal/addback decomposition is still pending.
4. Module D: edge-parallel stability supports a local image-geometry
   preservation mechanism, while axis preference and response-objective
   optimality remain guarded.
5. Module E: behavior supports modest but reliable alignment of drift/fixation
   cloud orientation with local image geometry, with the selected 3A panel
   foregrounding the image-coherence dependence; current model objectives do
   not yet beat raw edge geometry cleanly.
