# Cleanup Candidates

Generated: 2026-06-20

This is a non-destructive cleanup audit. No files have been moved or deleted as
part of creating this document. The goal is to make cleanup reviewable before
turning it into commands.

## Operating Rules

- Archive before delete when a file records scientific provenance, a failed
  branch, or a historical guardrail.
- Delete only generated clutter that is ignored, reproducible, or clearly a
  local log/bundle artifact.
- Keep active manuscript and figure sources discoverable even when they are
  messy.
- For `declan/`, follow the rule in `declan/MANIFEST.md`: if a script becomes
  obsolete, mark it historical rather than silently deleting it.
- Do not modify the untracked `declan/figure4_active_sensing_atlas/`,
  `declan/canonical_active_sensing/`, or `declan/canonical_geometry/`
  workspaces until they are either committed or explicitly declared disposable.
- Before cold-storing any generated artifact from `scripts/` or `declan/`, check
  whether code still defaults to that exact path. If so, preserve compatibility
  with a pointer, symlink, manifest entry, or code/config default update.

## Status Labels

- `Purge`: safe local deletion candidate after one command review.
- `Archive`: move under an `archive/` or `historical/` folder with a README.
- `Cold-storage`: large output/cache candidate for off-repo storage, not source
  deletion.
- `Review`: needs provenance/import/reference checks before action.
- `Keep`: intentionally retained despite age or size.

## Quick Wins

| Status | Path / Pattern | Why | Suggested action |
| --- | --- | --- | --- |
| Purge | `__pycache__/` outside `.venv` | Ignored bytecode, about a few MB total. | Delete with `find . -path ./.venv -prune -o -type d -name __pycache__ -print` after review. |
| Purge | `.pytest_cache/` | Ignored local test cache, small. | Delete when cleaning local state. |
| Purge | `Vision_core_repo_bundle_*.txt`, `repo_bundle.txt` | Ignored repo snapshots; multiple stale copies at root. | Delete or move one latest bundle to cold storage if still useful. |
| Purge | root `*.log`: `allhires_*.log`, `trow10*.log` | Ignored run logs at repo root. | Delete after confirming no current note cites exact contents. |
| Review | `trow10_no` | Tracked root log-like text file with no extension. It is the main tracked cleanup oddity found. | Rename into a documented archive or `git rm` after confirming no citation/import. |

## High-Impact Local Bulk

These are mostly ignored by `.gitignore`, so cleanup is about local disk and
provenance rather than git hygiene.

| Status | Path | Approx size | Why | Suggested action |
| --- | ---: | ---: | --- | --- |
| Review / Cold-storage | `scripts/temporal_decoding/data/rates/` | 47G | Ignored rate cache. Current docs mark e-optotype cached-rate scaffolding as historical/debugging for Figure 5, but several old scripts still default to this path. | Move to external/cold storage only with a pointer/regeneration note or after updating defaults. |
| Cold-storage | `outputs/stats/eoptotype_jacobian_field_smoothness_armB_core7_20260601/` | 24G | Huge historical e-optotype smoothness output; likely not current headline. | Preserve only summary tables/figures in repo outputs, cold-store raw pair tables. |
| Review | `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/` | about 27G | Contains active canonical BackImage/Figure 4 sources and obsolete pathfinders together. | Do not bulk delete. Create a sub-audit by result folder. |
| Review | `outputs/twin_covariance_structure/` | 20G | Older covariance-structure outputs; some are historical guardrails. | Archive/cold-store only after matching against `declan/MANIFEST.md` and narrative. |
| Keep / Review | `outputs/cache/` | 8.7G | Core copied caches for Fig2/Fig3, matched twin closure, RF contours. | Keep unless a cache is superseded by a documented newer copy. |
| Review / Cold-storage | `declan/spatial_info_fixrsvp_eye_scales_frames_per_im.pkl` | 3.8G | Ignored pickle at source-tree level, likely generated cache, but at least one script reads/writes this exact path. | Move to `outputs/cache/` or cold storage only after updating the script path or leaving a compatibility pointer. |
| Review / Cold-storage | `scripts/mcfarland_outputs_standard.pkl`, `scripts/mcfarland_outputs_mono.pkl` | 2.4G / 2.5G | Ignored generated artifacts under source folder, but multiple active/legacy scripts and `jake/twininfo` default to these exact paths. | Treat as path-dependent caches: move only with symlinks, explicit config/env overrides, or code default updates. |
| Cold-storage | `declan/continuous_pass_results/*.npz` | 1.5G each for several files | Ignored historical result arrays. | Cold-store raw arrays; keep plots/summary notes only if current. |

## Path-Compatibility Blockers

These are the main places where "large ignored file" does not yet mean
"safe to move":

- `scripts/mcfarland_outputs_mono.pkl` is the default source for
  `jake/twininfo/population.py`, `declan/vernier_active_sensing/forward.py`,
  several compact/tangent geometry scripts, and many old temporal/eoptotype
  scripts.
- `scripts/mcfarland_outputs.pkl` is still the default for some older
  active-sensing efficiency and Jacobian scripts.
- `scripts/temporal_decoding/data/rates/` is historical for current claims, but
  old temporal/eoptotype decoders still assume it exists.
- `declan/spatial_info_fixrsvp_eye_scales_frames_per_im.pkl` is source-tree
  generated state; moving it is desirable, but the reader/writer path must move
  with it.

Before relocating any of these, create a small cache registry such as
`docs/cache_inventory.md` or `outputs/cache/README.md` with `artifact`,
`current_location`, `cold_storage_location`, `regenerate_command`,
`path_compatibility`, and `last_verified` fields.

## Archive Candidates: Code And Markdown Threads

These should be archived with provenance notes, not deleted outright.

| Status | Path / Thread | Evidence | Suggested action |
| --- | --- | --- | --- |
| Archive | `declan/active_sensing_movie_information/run_active_sensing_movie_information.py` | README calls it a temporary exploratory runner; Jake `jake/twininfo/` is canonical for production movie-information pipeline. | Move to `declan/active_sensing_movie_information/archive/` or mark historical in-place. |
| Archive | Historical e-optotype Figure 5 scaffolds under `declan/active_sensing_movie_information/` and `scripts/temporal_decoding/` | README and manifest say old e-optotype cached-rate checks are historical/debugging, not Figure 5 evidence. | Archive docs/scripts together after identifying current natural-image replacements. |
| Archive | Pre-fix axis-conditioned BackImage runs and notes | `declan/axis_conditioned_backimage_trajectory_observer/README.md` names early pilots as pre-fix diagnostics only. | Keep clean shared-source code active; archive pre-fix result folders and any launch notes that imply old claims. |
| Archive | Local BackImage pairing pathfinders | `backimage_local_pairing_Iz_revisit_plan.md` marks early pathfinders diagnostic only. | Archive pre-patch/pathfinder outputs; retain cleaned fixed-manifest seed runs as reviewed evidence. |
| Archive | Structured decoder / forward denoising branch | `declan/MANIFEST.md` status is mixed/not promoted; forward denoising did not pass shuffled-eye specificity gate. | Add an archive README summarizing why the branch is diagnostic. |
| Archive | `declan/fig4_cov_TFTS/` older figure builders | `declan/compact_retinal_translation_geometry/README.md` says compact retinal-translation geometry is upgrade path/eventual replacement. | Keep until Figure 4/TFTS dependencies are fully mapped, then archive as predecessor. |
| Archive | Old STG / signed-axis derivative work | Direct derivative README calls older STG analyses historical reference and warns signed/context-specific recovery is fragile. | Mark as historical reference, not active claim source. |
| Archive | `declan/fixation_statistics_by_stimulus/archive/` existing archived file(s) | Already archived in place. | Add/maintain archive README if more files move here. |

## Output Folder Sub-Audit Targets

These folders deserve their own folder-level audit before moving/deleting
anything. The suggested review artifact is a small `README.cleanup.md` or CSV
with columns: `folder`, `status`, `source_script`, `superseded_by`,
`keep_files`, `cold_store_files`, `delete_files`.

| Priority | Folder | Reason |
| --- | --- | --- |
| High | `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/` | Mixes current Figure 4 evidence with obsolete pathfinders and pre-fix diagnostics. |
| High | `outputs/active_sensing_movie_information/` | Contains current covariance/whitening summaries plus superseded input-whitening and old Figure 5 scaffolds. |
| High | `outputs/stats/eoptotype_jacobian_field_smoothness_armB_core7_20260601/` | Very large CSV outputs; likely raw pair tables can be cold-stored. |
| Medium | `outputs/twin_covariance_structure/` | Large, older, partly historical covariance structure branch. |
| Medium | `outputs/compact_retinal_translation_geometry/` | Current-ish, but contains multiple diagnostics/null CSVs and chart-swap variants. |
| Medium | `declan/transformation_dynamics_cache/` | Ignored cache arrays in source tree; likely move/cold-store. |
| Medium | `declan/displacement_decoding_cache/` | Ignored cache arrays in source tree; likely move/cold-store. |

## Keep / Do Not Touch Yet

| Status | Path / Thread | Reason |
| --- | --- | --- |
| Keep | `models/`, `training/`, `eval/`, `VisionCore/`, `experiments/` | Core package and current runtime skeleton. |
| Keep | `declan/MANIFEST.md`, `declan/ANALYSIS_NARRATIVE.md` | Primary provenance map and interpretation ledger. |
| Keep | `declan/fig4_active_sensing/` | Clean current Figure 4 active-sensing workspace. |
| Keep | `declan/canonical_active_sensing/` | Current guarded production surface for active-sensing/adjudication runs. |
| Keep | `declan/canonical_geometry/` | Current guarded production surface for raw-edge residual and geometry runs. |
| Keep | `declan/fixation_statistics_by_stimulus/` | Current BackImage/Figure 4 analysis code, despite containing some archive candidates. |
| Keep | `declan/backimage_trajectory_observer/` | Current exact trajectory-table observer and compact-mechanism diagnostics. |
| Keep | `declan/axis_conditioned_backimage_trajectory_observer/` | Current shared-source axis-conditioned utilities/tests; only pre-fix outputs are suspect. |
| Keep | `declan/compact_retinal_translation_geometry/` | Current compact-geometry replacement path. Archive only older predecessor code after dependency check. |
| Keep | `declan/vernier_active_sensing/` | Useful coordinate-frame diagnostic and trajectory-table observer comparison. |
| Keep | `declan/figure4_active_sensing_atlas/` | Untracked active atlas workspace. Do not cleanup until ownership/status is explicit. |
| Keep | `outputs/cache/fig2_*`, `outputs/cache/fig3_*` | Referenced by matched twin covariance closure and recorded/twin bridges. |

## Broad Footprint-Reduction Plan

This is the highest-leverage way to reduce repo complexity without breaking
scientific provenance:

1. Checkpoint active untracked work first: the cleanup docs, canonical wrapper
   folders, and Figure 4 atlas should be committed or explicitly marked
   disposable before any broader cleanup.
2. Create two source-of-truth entry layers: `AGENT_CONTEXT.md` for coding agents
   and package-level READMEs for humans/scripts. Older plans should point into
   these, not compete with them.
3. Add an output audit index for the active large roots, especially
   `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/`. A CSV
   or `README.cleanup.md` per large output root is enough.
4. Move generated caches out of source folders only after preserving path
   compatibility. The source tree should eventually hold code and small
   manifests, not multi-GB pickle/NPZ artifacts.
5. Deduplicate exact Markdown mirrors and bundle/log clutter before attempting
   harder scientific archive moves. That gives immediate context reduction with
   low claim risk.
6. For historical scientific branches, archive by topic with a README containing
   `status`, `superseded_by`, `last useful result`, and `do_not_claim`.

## Candidate Process

1. Commit or otherwise checkpoint this audit.
2. Do a safe local purge branch for ignored clutter only:
   - bytecode caches,
   - `.pytest_cache`,
   - root repo bundles,
   - root ignored logs.
3. Create archive READMEs for code/document branches before moving files:
   - `status`,
   - `superseded_by`,
   - `last useful result`,
   - `do_not_claim`.
4. Create a cache inventory before relocating path-dependent generated files.
5. Audit `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/`
   folder-by-folder before deleting any output there.
6. Cold-store large ignored raw arrays before deleting local copies.
7. Only then consider tracked cleanup such as `trow10_no` and old tracked
   `declan/results` arrays.

## Useful Dry-Run Commands

These are intentionally dry-run/listing commands. Review their output before
turning any into deletion commands.

```bash
find . -path ./.venv -prune -o -type d -name __pycache__ -print
find . -maxdepth 1 -type f \( -name 'Vision_core_repo_bundle_*.txt' -o -name 'repo_bundle.txt' -o -name '*.log' \) -print
git ls-files | rg '(^trow10_no$|__pycache__|\.pyc$|\.log$|\.pkl$|\.npz$|\.npy$|\.mp4$|\.pdf$|\.png$)'
du -sh outputs/* declan/* scripts/* 2>/dev/null | sort -h
```
