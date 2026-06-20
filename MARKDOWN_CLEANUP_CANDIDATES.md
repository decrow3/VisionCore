# Markdown Cleanup Candidates

Generated: 2026-06-20

This is a non-destructive audit focused on stale plans, handoffs, prescriptions,
and duplicate reports. Its goal is to make the repository easier for coding
agents to read without losing scientific provenance.

## Inventory Snapshot

- Markdown files outside `.venv` and `outputs`: 211 as of the follow-up audit
  on 2026-06-20. This count will drift as handoffs and generated notes are
  added; regenerate it before treating the number as a baseline.
- Largest context sources:
  - `declan/ANALYSIS_NARRATIVE.md` at about 115 KB.
  - `ryan/methods_eyepos_matching/writeup.md` at about 87 KB.
  - `declan/MANIFEST.md` at about 76 KB.
  - `ryan/behavior-vs-vision/STRATEGY.md` at about 54 KB.
  - `declan/backimage_trajectory_observer/results_log.md` at about 54 KB.
  - Several root-level `declan/*plan*.md` and `*handoff*.md` files at 30-52 KB.
- Exact duplicate report mirrors found:
  - `declan/fig2/bias_diagnosis/*.md` and `ryan/fig2/bias_diagnosis/*.md`.
  - `declan/fig2/time_varying_noise_corr/*.md` and
    `ryan/fig2/time_varying_noise_corr/*.md`.

## Proposed Policy

- Keep `README.md`, `CLAUDE.md`, and `AGENT_CONTEXT.md` small and current.
- Keep package-level READMEs as the primary entrypoints.
- Keep `declan/MANIFEST.md` and `declan/ANALYSIS_NARRATIVE.md` as provenance
  ledgers, but do not make agents read them wholesale.
- Keep `declan/canonical_active_sensing/README.md`,
  `declan/canonical_geometry/README.md`, and
  `declan/figure4_active_sensing_atlas/claim_critical_diagnostics_queue.md`
  visible as current production/gating entrypoints.
- Move stale root-level `declan/*plan*.md`, `*handoff*.md`, and
  `*prescription*.md` files into topic archive folders once their current
  status is represented in the manifest/narrative.
- Prefer pointer stubs or archive READMEs over deleting old scientific notes.
- Deduplicate exact mirrors by keeping one owner copy and replacing the other
  with a pointer, but only after checking collaborator ownership.
- Before any move, run a targeted reference check against `declan/MANIFEST.md`,
  `declan/ANALYSIS_NARRATIVE.md`, and package READMEs. If a live doc still
  names the old path, update the link or leave a small pointer stub.

## Keep As Agent Entry Points

| Status | File | Why |
| --- | --- | --- |
| Keep | `AGENT_CONTEXT.md` | Short coding-agent reading path. |
| Keep | `README.md` | Project installation and basic package description. |
| Keep | `CLAUDE.md` | Existing agent guidance for model/training/eval. |
| Keep | `declan/MANIFEST.md` | Human-maintained map of `declan/` chronology and active branches. |
| Keep | `declan/ANALYSIS_NARRATIVE.md` | Interpretation ledger; search targeted sections only. |
| Keep | `declan/canonical_active_sensing/README.md` | Guarded entrypoint for active-sensing production and figure-pack runs. |
| Keep | `declan/canonical_geometry/README.md` | Guarded entrypoint for raw-edge residual adjudication and geometry figure-pack runs. |
| Keep | `declan/figure4_active_sensing_atlas/claim_critical_diagnostics_queue.md` | Claim-critical diagnostics and failure-mode gate before long canonical runs. |
| Keep | `declan/figure4_active_sensing_atlas/README.md`, `panel_manifest.csv`, `panel_source_map.md` | Current Figure 4 atlas, panel source map, and provenance surface. |
| Keep | `declan/fem_v1_current_status_and_way_forward.md` | Current high-level manuscript route framing. |
| Keep | `declan/active_sensing_roadmap_after_vernier_fixation_image_structure.md` | Current active-sensing synthesis. |
| Keep | package READMEs under active `declan/*/README.md` | Best local source for implementation status and run commands. |

## High-Confidence Archive Candidates

These are strong candidates for archive folders because later docs explicitly
supersede, narrow, or demote them.

Do not move these blindly. Several are still named by the manifest or narrative
as historical guardrails. The safe version of this pass is: create the archive
folder README, move the file, then update every live reference or leave a
one-paragraph pointer at the old path.

| Status | File / group | Rationale | Suggested destination |
| --- | --- | --- | --- |
| Archive | `declan/temporal_decoding_analysis_plan_consolidated_v2.md`, `declan/temporal_decoding_analysis_implementation_plan.md`, `declan/temporal_decoding_diagnostic_plan.md`, `declan/temporal_analysis_issues_and_alternatives.md` | Old temporal-decoding planning cluster; current Figure 5/active-sensing framing moved elsewhere. | `declan/archive/temporal_decoding/` |
| Archive | `declan/bigpicture_phase1_fem_v1_coding_agent_plan_v2.md`, `declan/bigpicture_fem_v1_high_impact_analysis_plan_v2.md`, `declan/revised_analysis_plan.md`, `declan/results_summary.md` | Broad early FEM planning/results docs. Later manifest/narrative encode current status and closed branches. | `declan/archive/early_bigpicture/` |
| Archive | `declan/jacobian_identity_transformation_analysis_plan.md`, `declan/analysis_plan_jacobian_v3.md`, `declan/jacobian_predictive_framework_handoff_revised.md`, `declan/jacobian_figure_handoff_nature_style.md`, `declan/jacobian_predictive_framework_progress_summary.md`, `declan/jacobian_identity_geometry_results.md` | Older Jacobian planning and handoff cluster; useful historical guardrails but not default agent context. | `declan/archive/jacobian_early/` |
| Archive | `declan/Keystone_Geometry_Crossover_handoff_v2.md` | Superseded by `Keystone_Geometry_Crossover_handoff_v3.md`. | `declan/archive/superseded_handoffs/` |
| Archive | `declan/shared_transformation_geometry_handoff.md` | Superseded by `shared_transformation_geometry_handoff_v2.md` and later compact geometry docs. | `declan/archive/superseded_handoffs/` |
| Archive | `declan/Global Iz FEMs.md` | Backup/older note; aggregate FEM information plan is the current route. | `declan/archive/backimage_latent/` |
| Archive | `declan/A General Info Framework for FEM Functio.md` | Contains "Do not promote" signal; not current agent context. | `declan/archive/general_info_framework/` |
| Archive | `declan/FEMs_Eoptotype_checks.md`, `declan/fem_eoptotype_hyperacuity_results.md` | E-optotype branch is repeatedly marked historical/debugging for current Figure 5/active-sensing use. `ANALYSIS_NARRATIVE.md` still tells readers to consult `fem_eoptotype_hyperacuity_results.md` when resuming that old branch, so this needs a pointer/update rather than a silent move. | `declan/archive/eoptotype/` |
| Archive | `scripts/curvature_onset_fem_scale_match_plan.md`, `scripts/pipeline_hypotheses.md`, `scripts/Inclusion.md` | Older script-adjacent plans; should not be default context. | `scripts/archive/notes/` |

## Review Before Archiving

These may still be useful, but they should not be default reading unless the
task touches that thread.

| Status | File / group | Keep reason | Cleanup idea |
| --- | --- | --- | --- |
| Review | `declan/Covariance_aware_FEM_optimality_analysis_prescription.md` | Large but still explains covariance-aware branch and historical tangent convention issues. | Move into `declan/active_sensing_movie_information/docs/` or archive once README/narrative fully cover it. |
| Review | `declan/compact_retinal_translation_geometry_implementation_spec.md` | Large spec for current replacement path. | Keep near package or move under `declan/compact_retinal_translation_geometry/docs/`. |
| Review | `declan/vernier_active_sensing_analysis_plan.md` | Still useful background; package README is a better default. | Move under `declan/vernier_active_sensing/docs/` or archive after README captures current status. |
| Review | `declan/free_viewing_latent_information_test_plan.md` | Recent and tied to BackImage aggregate path, but not default reading. | Keep until BackImage active-sensing figure stabilizes, then archive. |
| Review | `declan/backimage_aggregate_fem_information_plan.md` | Current-ish plan backing Figure 4 module B. | Keep for now; later fold into package README/figure atlas. |
| Review | `declan/backimage_local_pairing_Iz_revisit_plan.md` | Contains cleaned local-pairing caveats and pathfinder warnings. | Keep until local-pairing branch is supplement-routed. |
| Review | `declan/fem_v1_maximal_story_priority_checklist.md` | Very recent checklist, but long and high-context. | Keep for human planning; agents should not read by default. |
| Review | `declan/backimage_trajectory_observer/results_log.md` | Current result log, but very large. | Split into summarized README plus archived detailed log chunks. |
| Review | `declan/figure4_multipanel_plus_sup.md`, `declan/Fig4_reruns_plan.md` | Older Figure 4 planning may be partly superseded by `declan/fig4_active_sensing/` and atlas. | Archive once Figure 4 atlas/compression path is committed. |
| Review | `docs/polar_v1_*`, `docs/VIVIT_*` | Older model-design docs, potentially useful if those modules return. | Move to `docs/archive/model_experiments/` with an index if no active code uses them. |
| Review | `ryan/behavior-vs-vision/STRATEGY.md`, `ryan/methods_eyepos_matching/writeup.md` | Collaborator-owned, polished scientific context. | Do not move without owner/provenance review; add agent-context warning only. |

## Exact Duplicate Markdown Mirrors

These are ideal context-reduction targets because byte-for-byte duplicates
exist. Ownership should decide which copy remains canonical.

| Status | Duplicate group | Suggested action |
| --- | --- | --- |
| Review | `declan/fig2/bias_diagnosis/*.md` vs `ryan/fig2/bias_diagnosis/*.md` | Keep one folder as canonical; replace the other with a short pointer README or archive note. |
| Review | `declan/fig2/time_varying_noise_corr/*.md` vs `ryan/fig2/time_varying_noise_corr/*.md` | Same as above. |

Verified identical files in `bias_diagnosis`:

```text
FINAL_REPORT.md
SYNTHESIS.md
h1_report.md
h2_report.md
h3_h5_report.md
h4_report.md
h6_report.md
shuffle_null_shift_report.md
weighting_fix_report.md
weighting_mismatch_report.md
```

Verified identical files in `time_varying_noise_corr`:

```text
h1_report.md
h1b_report.md
h2_report.md
```

## Suggested Archive Layout

```text
declan/archive/
  README.md
  early_bigpicture/
  temporal_decoding/
  jacobian_early/
  eoptotype/
  superseded_handoffs/
  backimage_latent/
scripts/archive/
  notes/
docs/archive/
  model_experiments/
```

Each archive folder should have a `README.md` with:

```text
Status:
Archived on:
Superseded by:
Still useful for:
Do not use for:
```

## Broad Context-Reduction Moves

These are the best high-level moves for making the repo easier to navigate
without flattening the scientific provenance:

1. Keep one top-level agent path: `AGENT_CONTEXT.md`, current package READMEs,
   the manifest, and targeted narrative sections.
2. Move historical plans into topic archives only after their current status is
   represented in a package README, the manifest, or the narrative.
3. Replace exact duplicate report mirrors with one canonical owner copy plus a
   pointer README in the other location.
4. Split very large active logs, especially
   `declan/backimage_trajectory_observer/results_log.md`, into a short current
   summary plus archived detailed chunks.
5. Keep long-run handoffs near the package they operate on. Root-level handoffs
   should be temporary unless they are listed in the manifest as current.

## Next Mechanical Pass

1. Regenerate the Markdown inventory count.
2. Add archive folder READMEs first.
3. Run targeted reference checks for each candidate.
4. Move only the high-confidence archive candidates whose references can be
   updated or stubbed.
5. Replace exact duplicate report mirrors with pointer READMEs only after
   deciding canonical ownership.
6. Update `declan/MANIFEST.md` once moves happen.
7. Keep `AGENT_CONTEXT.md` as the short top-level reading path.
