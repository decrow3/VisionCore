# Canonical Geometry

This package contains stable, config-driven entry points for BackImage geometry
and raw-edge analyses. The current production surface promotes both the
raw-edge residual adjudication and the cache-first geometry figure pack.
Wrappers refuse existing non-empty `out_dir` paths by default; pass
`--allow-existing-output` only for intentional refreshes.

Example:

```bash
.venv/bin/python -m declan.canonical_geometry.run_raw_edge_audit --print-command
.venv/bin/python -m declan.canonical_geometry.make_geometry_figure_pack --print-command
.venv/bin/python -m declan.canonical_geometry.make_geometry_figure_pack --validate-only
.venv/bin/python -m declan.canonical_geometry.validate_configs
.venv/bin/python -m declan.canonical_geometry.validate_configs --check-output-freshness
```

Current output provenance lives in `provenance/current_outputs.md`.
