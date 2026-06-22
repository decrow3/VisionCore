# Declan Workspace Routing

Status: current routing entry point.
Last updated: 2026-06-22.
Read by default: yes.

This folder is a research workspace, not a single package. It contains current
analysis code, figure-building code, compact summaries, historical plans, and
long provenance documents. Use this file to choose what to read next.

## Default Reading Path

1. `../AGENT_CONTEXT.md` for repo-wide rules and artifact boundaries.
2. This file for the `declan/` workspace route.
3. The package `README.md` closest to the code or figure you are editing.
4. `ANALYSIS_NARRATIVE.md` for current interpretation and claim boundaries.
5. Targeted sections of `MANIFEST.md` only when you need chronology or file
   provenance.

Do not bulk-read every Markdown file in this directory. Many root-level plans,
handoffs, prescriptions, and result logs are historical or branch-specific.

## Current Navigation Map

| Task | Read First | Then Read | Avoid By Default |
|---|---|---|---|
| Figure 4 active-sensing figure package | `figure4_active_sensing_atlas/README.md` | `figure4_active_sensing_atlas/provisional_panel_contract_v0.csv`, `figure4_active_sensing_atlas/claim_critical_diagnostics_queue.md` | old broad Figure 4 plans unless cited by the atlas README |
| Canonical BackImage active-sensing runs | `canonical_active_sensing/README.md` | `canonical_active_sensing/provenance/current_outputs.md`, relevant config JSON | older one-off BackImage plans unless reproducing history |
| Raw-edge / behavior geometry adjudication | `canonical_geometry/README.md` | `figure4_active_sensing_atlas/4e_companion_behavior_geometry_bridge.md` | early Jacobian archive plans |
| Compact retinal-translation geometry | `compact_retinal_translation_geometry/README.md` | `compact_retinal_translation_geometry/static_pc_control_adjudication_note.md` | old global covariance-identity plans |
| Natural-image / Figure 5 active sensing | `active_sensing_movie_information/README.md` | `ANALYSIS_NARRATIVE.md`, targeted Figure 5 plans | old e-optotype temporal-decoding plans |
| Vernier active sensing | `vernier_active_sensing/README.md` | targeted run scripts and current outputs | broad active-sensing synthesis unless needed |
| Matched twin covariance closure | `matched_twin_covariance_closure/README.md` | current run script and output summaries | early proof-of-concept handoffs |

## Current Compact Summaries

- `ANALYSIS_NARRATIVE.md`: current scientific synthesis and caveats.
- `MANIFEST.md`: chronological file map. Search it; do not read end to end.
- `ANALYSIS_NARRATIVE_DETAILED_2026-06-22.md`: archived detailed narrative for
  exact historical values and paths.

## Status Language

- `Promoted`: can support a current claim with caveats.
- `Supportive`: useful but not a standalone claim.
- `Diagnostic`: useful for interpretation or debugging, not a headline.
- `Historical`: provenance only.
- `Superseded`: do not use as current guidance unless tracing history.
- `Open`: planned or incomplete.

## Documentation Rules

New or updated analysis documents should begin with a short routing header:

```text
Status: Current | Supportive | Diagnostic | Historical | Superseded | Open
Read by default: yes | no
Canonical code: path or none
Canonical outputs: path or none
Superseded by: path or none
Last updated: YYYY-MM-DD
```

Long chronological logs should live near the package they describe or under
`archive/`. Current package READMEs should stay short and route to deeper
evidence rather than accumulating all historical interpretation.
