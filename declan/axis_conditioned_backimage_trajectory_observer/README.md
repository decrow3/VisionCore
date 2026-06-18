# Axis-conditioned BackImage trajectory observer

This package is the implementation workspace for the axis-conditioned BackImage
trajectory-observer branch described in:

```text
declan/axis_conditioned_backimage_trajectory_observer_plan.md
```

The analysis asks whether local edge-parallel trajectory catalogs preserve
natural-image identity under trajectory marginalization better than matched
edge-orthogonal catalogs, using the existing BackImage trajectory-table observer
infrastructure.

## Current contents

- `axis_conditioned_traces.py`: pure NumPy utilities for constructing matched
  edge-parallel and edge-orthogonal traces from an observed BackImage drift
  trace.
- `tests/test_axis_conditioned_traces.py`: focused tests for axis geometry,
  matching, deterministic metadata, and RMS clipping.

## Current integration path

The trace utilities are wired into the existing runner as explicit motion
families:

```text
axis_edge_parallel
axis_edge_orthogonal
```

Runner:

```text
declan/fixation_statistics_by_stimulus/run_backimage_trajectory_table_observer.py
```

The observer scoring should continue to use:

```text
declan/backimage_trajectory_observer/observer.py
```

The runner supports two axis catalog modes:

- `--axis-catalog-mode shared`: each sampled source window uses its own
  `image_edge_axis_deg`, matching the existing shared trajectory-catalog design.
- `--axis-catalog-mode per_candidate`: each candidate patch receives the same
  sampled source trace identities, but each trace is re-rendered relative to
  that candidate patch's own `image_edge_axis_deg`. This is the stricter
  image-conditioned prior needed for edge-parallel versus edge-orthogonal
  response tables.

In `per_candidate` mode, the retained prior source pool excludes every
candidate-set `source_row`, not just the true image source row. It also rejects
sampled sources whose rendered axis-conditioned trace is an exact hash match or
near-RMSE duplicate of the observed trace under any candidate axis. The response
manifest reports `excluded_candidate_source_rows`,
`excluded_candidate_source_row_count`, `excluded_exact_trace_hash`, and
`excluded_near_duplicate_rmse`.

When more than one axis-conditioned prior family is requested, `per_candidate`
mode now samples one shared retained source list per trial/candidate set/scale
and reuses it for `axis_edge_parallel` and `axis_edge_orthogonal`. Claim-level
axis comparisons should verify this in the manifest/audit:

```text
axis_shared_source_catalog = True
axis_shared_source_catalog_fraction = 1.0
median_source_jaccard = 1.0
```

Both modes write axis fields into `motion_catalog.csv`; axis-conditioned rows
also appear in `axis_trajectory_catalog.csv`.

Useful dry-run smoke:

```bash
PYTHONPATH=/home/declan/VisionCore uv run python -m declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer \
  --out-dir outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_trajectory_observer_dryrun_smoke \
  --max-images 4 \
  --n-candidates 2 \
  --candidate-set-modes hard_negative_structure \
  --observation-family empirical \
  --prior-families axis_edge_parallel,axis_edge_orthogonal \
  --axis-catalog-mode per_candidate \
  --observed-rms-scales 0.5 \
  --trajectory-prior-mode leave_one_out \
  --n-prior-trajectories 1 \
  --likelihood-scales 1.0 \
  --dry-run
```

This package should continue to own only the axis-conditioned trace construction,
axis-family metadata, and posthoc summaries specific to the edge-parallel versus
edge-orthogonal comparison. The finite-table observer likelihood implementation
should remain in `declan/backimage_trajectory_observer`.

## Current caveat

Early hard-negative pilots generated before the shared-source fix should be
treated as pre-fix diagnostics only:

```text
backimage_axis_conditioned_trajectory_observer_percandidate_gpu1_pilot32_c4_k8
backimage_axis_conditioned_trajectory_observer_percandidate_gpu1_pilot64_c4_k16
backimage_axis_conditioned_trajectory_observer_percandidate_gpu1_target128_c4_k32
```

Those outputs compared distribution-matched but not strictly shared
parallel/orthogonal source catalogs. Their orthogonal-over-parallel readout is
therefore not a clean biological result. Any promoted run should be regenerated
with the current runner and pass the shared-source audit checks above.

## Current clean pilots

The first completed clean shared-source matched-static pilot is:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1/
```

It uses `matched_static_response`, `n=64`, `c=4`, `k=16`, empirical
observations, and `0.5x` axis-conditioned priors. The audit passes the
shared-source gates:

```text
axis_shared_source_catalog_fraction = 1.0
source Jaccard = 1.0
paired prior rows = 4096
```

Readout:

```text
known-eye = 1.000
zero-eye = 0.641
axis_edge_parallel joint = 55/64 = 0.859
axis_edge_orthogonal joint = 53/64 = 0.828
```

This is a positive pilot, not a claim-level result. The edge-parallel advantage
is only two trials, so it needs larger shared-source replication before becoming
a biological conclusion.

The first completed clean shared-source hard-negative replacement is:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_shared_source_gpu1_n64_c4_k16_v1/
```

It uses `hard_negative_structure`, `n=64`, `c=4`, `k=16`, empirical
observations, and `0.5x` axis-conditioned priors. The audit passes the
shared-source gates:

```text
axis_shared_source_catalog_fraction = 1.0
source Jaccard = 1.0
paired prior rows = 4096
parallel/orthogonal motion-stat deltas = 0
```

Readout:

```text
known-eye = 1.000
zero-eye = 0.641
axis_edge_parallel joint = 54/64 = 0.844
axis_edge_orthogonal joint = 57/64 = 0.891
```

This run confirms that both axis-conditioned trajectory priors rescue image
identity over the zero-eye observer when the catalog comparison is clean.
However, the axis direction is mixed: hard-negative accuracy favors
edge-orthogonal, while paired true-score and margin diagnostics retain some
edge-parallel signal. Treat the axis-specific claim as unresolved until larger
shared-source runs replicate across candidate modes and seeds.

## Current limitation

The `per_candidate` mode currently requires all prior families to be
axis-conditioned and requires `--trajectory-prior-mode leave_one_out`. That keeps
the response table rectangular while avoiding ambiguous mixes of shared and
candidate-conditioned trajectory catalogs.
