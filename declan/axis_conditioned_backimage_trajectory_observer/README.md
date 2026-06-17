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

## Current limitation

The `per_candidate` mode currently requires all prior families to be
axis-conditioned and requires `--trajectory-prior-mode leave_one_out`. That keeps
the response table rectangular while avoiding ambiguous mixes of shared and
candidate-conditioned trajectory catalogs.
