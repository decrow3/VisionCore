# Production Consolidation Repo Audit Plan

Last updated: 2026-06-20.

## Purpose

After the feature-decomposition v4 closure pass, consolidate the active-sensing
codebase so production geometry and active analyses are easy to rerun, audit,
and hand off.

The target structure is two self-contained subfolders:

```text
declan/canonical_geometry/
declan/canonical_active_sensing/
```

Do not move code blindly. First inventory the variants, identify canonical
entry points, add wrappers/configs/tests, and only then retire or archive older
variants.

## Why This Is Needed

The repo now contains many exploratory scripts, posthocs, figure packs, and
handoffs. That was useful during discovery, but production runs need:

- one obvious entry point per analysis;
- explicit configs rather than hidden constants;
- provenance manifests for output folders;
- smoke tests that catch stale path or schema changes;
- clear separation between geometry evidence and model active-sensing evidence.

## Proposed Folders

### `declan/canonical_geometry/`

Scope:

- raw edge / image geometry;
- drift-edge alignment summaries;
- pixel and V1-twin edge-parallel preservation;
- compact retinal translation / transformation geometry where it supports the
  BackImage story;
- figure panels that ground the behavioral geometry claim.

Expected contents:

```text
README.md
configs/
scripts/
tests/
provenance/
```

Candidate source scripts to audit:

```text
declan/fixation_statistics_by_stimulus/analyze_backimage_raw_edge_roadblock.py
declan/fixation_statistics_by_stimulus/run_backimage_edge_parallel_stability_screen.py
declan/fixation_statistics_by_stimulus/posthoc_backimage_twin_stability_metric_audit.py
declan/fixation_statistics_by_stimulus/summarize_backimage_twin_stability_metric_audit.py
declan/figure4_active_sensing_atlas/scripts/plot_panel_d_subpanels.py
declan/figure4_active_sensing_atlas/scripts/plot_panel_e_subpanels.py
```

### `declan/canonical_active_sensing/`

Scope:

- aggregate FEM information;
- local `I_z` pairing;
- joint posterior observer;
- feature-decomposition adjudication;
- canonical active-sensing figure pack.

Expected contents:

```text
README.md
configs/
scripts/
tests/
provenance/
```

Candidate source scripts to audit:

```text
declan/fixation_statistics_by_stimulus/run_backimage_aggregate_fem_information.py
declan/fixation_statistics_by_stimulus/summarize_backimage_aggregate_incremental_motion.py
declan/fixation_statistics_by_stimulus/run_backimage_local_pairing_Iz_revisit.py
declan/fixation_statistics_by_stimulus/run_backimage_trajectory_table_observer.py
declan/backimage_trajectory_observer/analyze_feature_posterior.py
declan/fixation_statistics_by_stimulus/analyze_backimage_feature_decomposition_adjudication.py
declan/fixation_statistics_by_stimulus/make_backimage_aggregate_fem_figure_pack.py
```

## Repo-State Audit

Run after the current detached joint `rel_0p25x` watcher finishes:

```bash
git status --short
rg --files declan | sort > /tmp/declan_files.txt
.venv/bin/python -m py_compile \
  declan/fixation_statistics_by_stimulus/analyze_backimage_feature_decomposition_adjudication.py \
  declan/fixation_statistics_by_stimulus/analyze_backimage_raw_edge_roadblock.py \
  declan/fixation_statistics_by_stimulus/run_backimage_aggregate_fem_information.py \
  declan/fixation_statistics_by_stimulus/summarize_backimage_aggregate_incremental_motion.py \
  declan/fixation_statistics_by_stimulus/run_backimage_local_pairing_Iz_revisit.py \
  declan/fixation_statistics_by_stimulus/run_backimage_trajectory_table_observer.py \
  declan/backimage_trajectory_observer/analyze_feature_posterior.py \
  declan/fixation_statistics_by_stimulus/make_backimage_aggregate_fem_figure_pack.py
```

If `pytest` is available:

```bash
.venv/bin/python -m pytest \
  declan/fixation_statistics_by_stimulus/tests \
  declan/backimage_trajectory_observer/tests
```

Current caveat: earlier checks found `pytest` unavailable in the environment,
so the first audit may need to report that rather than forcing dependency work.

## Canonicalization Rules

For each promoted analysis:

```text
[ ] one production config file
[ ] one production run script or wrapper
[ ] one smoke-test command
[ ] one output manifest schema
[ ] one figure/table output contract
[ ] no hardcoded exploratory output directory unless documented as default
[ ] no dependence on stale folders when a repaired folder exists
[ ] clear claim boundary in README
```

Do not delete exploratory scripts in the same pass. Mark them as archived or
legacy only after the canonical wrappers reproduce the expected outputs.

## Suggested Production API

Geometry:

```bash
.venv/bin/python -m declan.canonical_geometry.run_raw_edge_audit --config declan/canonical_geometry/configs/raw_edge_v1.json
.venv/bin/python -m declan.canonical_geometry.make_geometry_figure_pack --config declan/canonical_geometry/configs/figure_geometry_v1.json
```

Active sensing:

```bash
.venv/bin/python -m declan.canonical_active_sensing.run_aggregate_fem --config declan/canonical_active_sensing/configs/aggregate_fem_k16_v1.json
.venv/bin/python -m declan.canonical_active_sensing.run_local_pairing --config declan/canonical_active_sensing/configs/local_pairing_k16_v1.json
.venv/bin/python -m declan.canonical_active_sensing.run_joint_posterior --config declan/canonical_active_sensing/configs/joint_posterior_k16_v1.json
.venv/bin/python -m declan.canonical_active_sensing.adjudicate_feature_spec --config declan/canonical_active_sensing/configs/feature_adjudication_v1.json
.venv/bin/python -m declan.canonical_active_sensing.make_active_sensing_figure_pack --config declan/canonical_active_sensing/configs/figure_active_sensing_v1.json
```

These wrappers can initially call the existing scripts. The goal is stable
production entry points, not a risky rewrite.

## First Implementation Pass

1. Add the two folders with `README.md`, `configs/`, `scripts/`, `tests/`, and
   `provenance/`.
2. Add config JSON files that encode the exact v4/two-readout active-sensing
   spec and the current raw-edge geometry spec.
3. Add thin wrapper modules that call existing scripts using structured config.
4. Add smoke tests that validate config parsing and dry-run/cache-only behavior.
5. Update figure scripts to accept canonical config instead of hardcoded k4/k8
   and stale output roots.
6. Generate `provenance/current_outputs.md` listing the output dirs that support
   each figure claim.

## Stop Conditions

Pause consolidation if:

- v4 feature adjudication changes the target away from
  `pyramid_local_field k16`;
- the joint `rel_0p25x` watcher fails feature identity validation;
- a promoted wrapper cannot reproduce the existing cache-only summaries;
- tests reveal schema drift in current production outputs.

## Success Definition

The repo is production-ready when a new coding agent can start from:

```text
declan/canonical_geometry/README.md
declan/canonical_active_sensing/README.md
```

and rerun the figure-supporting analyses, or at least verify their cached
outputs, without reading the full historical handoff stack.
