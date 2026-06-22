# Agent Context

This file is the short reading path for coding agents. It exists because the
repo contains many historical plans and result narratives that are useful for
provenance but expensive to load by default.

## Start Here

For ordinary code work, read in this order:

1. `README.md`
2. `CLAUDE.md`
3. `pyproject.toml`
4. The package README closest to the files you will edit
5. For `declan/` work, `declan/README.md`
6. Only then, targeted sections of `declan/MANIFEST.md` or
   `declan/ANALYSIS_NARRATIVE.md`

Do not bulk-read every Markdown file in `declan/`. Many are older plans,
handoffs, prescriptions, or posthoc narratives that have been superseded.

## Repo Layers

- `models/`, `training/`, `eval/`, `VisionCore/`: core Python package for V1
  digital-twin models, training, evaluation, and reusable statistics/covariance
  utilities.
- `experiments/`: model and dataset YAML configs plus training launch scripts.
- `declan/`: research workspace for FEM/V1 covariance, active sensing, Figure 4,
  BackImage, Vernier, and compact retinal-translation geometry.
- `scripts/`: older and mixed analysis scripts. Treat as less canonical unless
  a current README or manifest points there.
- `jake/`, `ryan/`, `tejas/`: collaborator workspaces. Avoid reorganizing them
  without checking ownership/provenance.
- `outputs/`, `results/`, `figures/`, `logs/`: generated artifacts. Most are
  ignored; do not assume source-of-truth status from size or recency alone.
- `outputs/artifacts/temporal_decoding/data/rates/`: local bulky cached
  temporal-decoding rate arrays. The legacy path
  `scripts/temporal_decoding/data/rates` may be a symlink for compatibility;
  treat it as generated data, not script source.
- `outputs/artifacts/mcfarland/`: local bulky McFarland model/readout pickle
  exports. Legacy `scripts/mcfarland_*.pkl` paths may be symlinks for older
  analyses; treat them as generated data.

## Current Declan Reading Path

Use this path for the current FEM / active-sensing manuscript thread:

1. `declan/README.md`: compact workspace router. Use it to select the relevant
   package README and avoid old root-level plans.
2. `declan/ANALYSIS_NARRATIVE.md`: interpretation ledger. Search for the thread
   or status rather than reading the whole file.
3. `declan/MANIFEST.md`: navigation map and chronology. Search targeted
   sections rather than reading end to end.
4. `declan/canonical_active_sensing/README.md`: guarded production surface for
   BackImage aggregate, local-pairing, joint-posterior, adjudication, and figure
   pack runs.
5. `declan/canonical_geometry/README.md`: guarded production surface for raw-edge
   residual adjudication and geometry figure-pack runs.
6. `declan/figure4_active_sensing_atlas/claim_critical_diagnostics_queue.md`:
   claim-critical failure-mode checklist for long canonical runs and promoted
   panels.
7. `declan/fem_v1_current_status_and_way_forward.md`: high-level manuscript
   route framing.
8. `declan/active_sensing_roadmap_after_vernier_fixation_image_structure.md`:
   current active-sensing synthesis.
9. Package READMEs for implementation details:
   - `declan/figure4_active_sensing_atlas/README.md`
   - `declan/fig4_active_sensing/README.md`
   - `declan/fixation_statistics_by_stimulus/` source plus local summaries
   - `declan/backimage_trajectory_observer/README.md`
   - `declan/axis_conditioned_backimage_trajectory_observer/README.md`
   - `declan/compact_retinal_translation_geometry/README.md`
   - `declan/vernier_active_sensing/README.md`
   - `declan/matched_twin_covariance_closure/README.md`
   - `declan/direct_recorded_derivative_twin_alignment/README.md`

## Current Figure 4 / Active-Sensing Path

For Figure 4 active-sensing work, prefer:

1. `declan/canonical_active_sensing/README.md` for long active-sensing runs.
2. `declan/canonical_geometry/README.md` for raw-edge residual/geometry runs.
3. `declan/figure4_active_sensing_atlas/claim_critical_diagnostics_queue.md`
   before treating any core panel as claim-ready.
4. `declan/figure4_active_sensing_atlas/README.md` for the atlas build surface.
5. `declan/figure4_active_sensing_atlas/panel_manifest.csv` and
   `declan/figure4_active_sensing_atlas/panel_source_map.md` for panel sources.
6. `declan/fig4_active_sensing/README.md` for the consolidated Figure 4 package.
7. `declan/active_sensing_roadmap_after_vernier_fixation_image_structure.md`
   and `declan/active_sensing_unit_space_provenance.md` for scientific framing.

The atlas and canonical wrapper directories may be untracked in this workspace.
Treat them as active work, not cleanup fodder, until they are committed or
explicitly declared disposable.

## Markdown Context Hygiene

Prefer current package READMEs over older root-level plan files. Load the older
plans only when the task is specifically about that branch or provenance.

Avoid by default:

- very large result logs such as `declan/backimage_trajectory_observer/results_log.md`
- old broad planning docs unless the manifest names them as current
- duplicated collaborator reports under both `declan/fig2/` and `ryan/fig2/`
- historical e-optotype / temporal-decoding Figure 5 scaffolds unless the task
  is explicitly about that old branch

Status words matter:

- `Promoted`: can support a current claim, still check caveats.
- `Supportive`: useful evidence, usually bounded.
- `Diagnostic`: machinery or clue, not a headline claim.
- `Historical`: provenance or guardrail, not current default context.
- `Closed` / `not promoted` / `pre-fix`: do not treat as current evidence.

## Cleanup References

- `CLEANUP_CANDIDATES.md`: generated artifacts, archive/cold-storage targets,
  and safe purge candidates.
- `MARKDOWN_CLEANUP_CANDIDATES.md`: stale-plan and context-window cleanup audit.
- `docs/documentation_routing_plan.md`: current documentation-routing review
  and staged cleanup plan.
- `docs/cache_inventory.md`: large cache/output path-compatibility registry.
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/README.cleanup.md`:
  BackImage/Figure 4 output-root cleanup audit.
