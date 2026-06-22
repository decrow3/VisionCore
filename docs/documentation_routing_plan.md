# Documentation Routing Review

Status: current cleanup plan.
Last updated: 2026-06-22.
Read by default: no.

This review captures the current documentation routing problem and the planned
fixes. The goal is to make agents load a compact route first, then only the
domain-specific evidence needed for the task.

## Current State

- `AGENT_CONTEXT.md` is the repo-wide entry point and already gives good
  context hygiene rules.
- `declan/ANALYSIS_NARRATIVE.md` has been compressed into a current synthesis.
  The detailed historical version now lives at
  `declan/ANALYSIS_NARRATIVE_DETAILED_2026-06-22.md`.
- `declan/MANIFEST.md` is still useful but long. It should be searched or read
  in targeted sections, not treated as a default full-context document.
- `declan/` still contains many root-level plans, handoffs, prescriptions, and
  writeups that look equally canonical from filenames alone.
- Several package READMEs are too large for default reading:
  `declan/backimage_trajectory_observer/README.md` and
  `declan/active_sensing_movie_information/README.md` are the largest examples.

## Worst Routing Offenders

| Offender | Problem | Preferred Fix |
|---|---|---|
| Many root-level `declan/*_plan.md`, `*_handoff.md`, `*_prescription.md` files | Agents cannot tell current from historical without opening them | Add routing headers; archive or supersede old ones |
| `declan/MANIFEST.md` | Valuable but too long for default full reads | Keep as searchable chronology; route through `declan/README.md` first |
| `declan/backimage_trajectory_observer/README.md` | README has grown into a long report/log | Split into compact README plus `results_log.md` / `claims.md` |
| `declan/active_sensing_movie_information/README.md` | Mixes entry points, interpretation, and historical planning | Split into compact README plus `claims.md` and current-output notes |
| `declan/figure4_active_sensing_atlas/figure_build_log.md` | Build/provenance log is large and tempting | Keep as log; never default-read except for build provenance |
| Archived but confident plans | Old docs can sound current | Ensure archive READMEs and per-file headers state supersession |

## Target Routing Shape

```text
AGENT_CONTEXT.md
  -> declan/README.md
    -> package README closest to the task
      -> compact claims/status doc
        -> targeted manifest or historical handoff only when needed
```

For Figure 4 active-sensing work:

```text
AGENT_CONTEXT.md
  -> declan/README.md
  -> declan/figure4_active_sensing_atlas/README.md
  -> declan/figure4_active_sensing_atlas/provisional_panel_contract_v0.csv
  -> panel companion doc for the specific panel
```

For natural-image / Figure 5 work:

```text
AGENT_CONTEXT.md
  -> declan/README.md
  -> declan/active_sensing_movie_information/README.md
  -> declan/ANALYSIS_NARRATIVE.md
  -> targeted current plan or output summary
```

## Update Plan

1. Add `declan/README.md` as the workspace landing page.
2. Update `AGENT_CONTEXT.md` so `declan/README.md` sits before `MANIFEST.md`.
3. Add routing headers to large active root-level docs that are still useful.
4. Move clearly historical root-level docs into `declan/archive/` subfolders, or
   mark them `Historical` / `Superseded` in-place when moving would break
   existing references.
5. Split oversized package READMEs:
   - Keep `README.md` as commands, entry points, current outputs, caveats.
   - Move interpretation to `claims.md`.
   - Move chronological details to `results_log.md`.
6. Keep `MANIFEST.md` as chronology and file provenance. Do not make it carry
   the interpretation burden now handled by `ANALYSIS_NARRATIVE.md`.

## Header Template

Use this at the top of any long analysis document:

```text
Status: Current | Supportive | Diagnostic | Historical | Superseded | Open
Read by default: yes | no
Canonical code: path or none
Canonical outputs: path or none
Superseded by: path or none
Last updated: YYYY-MM-DD
```

## First Batch Candidates

Add headers or archive these first:

- `declan/fem_v1_maximal_story_priority_checklist.md`
- `declan/compact_retinal_translation_geometry_implementation_spec.md`
- `declan/Covariance_aware_FEM_optimality_analysis_prescription.md`
- `declan/active_sensing_roadmap_after_vernier_fixation_image_structure.md`
- `declan/vernier_active_sensing_analysis_plan.md`
- `declan/figure4_multipanel_plus_sup.md`
- `declan/figure4_geometry_bridge_audit_plan_v2.md`
- `declan/FEM_population_coding_writeup.md`
- `declan/free_viewing_latent_information_test_plan.md`
- `declan/fem_next_steps_plan.md`

## Acceptance Checks

- `rg --files` should expose source and compact routing docs, not generated
  artifacts.
- `git ls-files -i -c --exclude-standard` should remain `0`.
- A new agent should be able to answer "where do I start for Figure 4?" from
  `AGENT_CONTEXT.md` and `declan/README.md` without opening `MANIFEST.md`.
- Long docs should say whether they are current before line 20.
