# Compact Retinal-Translation Geometry

This folder is the upgrade path and eventual replacement for
`declan/fig4_cov_TFTS`.

The first implementation is a data-forward harness around the production
artifacts that already exist in the workspace:

- `outputs/twin_feature_tangent_structure_prod_v2`
- `outputs/matched_twin_covariance_closure_rf_null_step025_rfbacked_v2`

It writes the panel files named in
`declan/compact_retinal_translation_geometry_implementation_spec.md` under
`outputs/compact_retinal_translation_geometry` by default.

## Run

```bash
uv run python -m declan.compact_retinal_translation_geometry.run_compact_retinal_translation_geometry
```

Useful overrides:

```bash
uv run python -m declan.compact_retinal_translation_geometry.run_compact_retinal_translation_geometry \
  --tfts-root outputs/twin_feature_tangent_structure_prod_v2 \
  --closure-root outputs/matched_twin_covariance_closure_rf_null_step025_rfbacked_v2 \
  --out-root outputs/compact_retinal_translation_geometry
```

## Audit

After generating the panel tables, run the spec-facing audit suite:

```bash
uv run python -m declan.compact_retinal_translation_geometry.run_compact_geometry_audits
```

This writes:

- `outputs/compact_retinal_translation_geometry/audit.json`
- `outputs/compact_retinal_translation_geometry/tables/acceptance_matrix.csv`
- `outputs/compact_retinal_translation_geometry/tables/session_summary.csv`
- focused audit tables for RF/readout bins, Panel B compactness, Panel C
  leakage/generalization, Panel E covariance closure, Panel D denominators,
  metric-structure readiness, decoding readiness, and direct recorded-derivative
  support.

To use the spec-valid main-session set while retaining small matched-unit
sessions as diagnostics:

```bash
uv run python -m declan.compact_retinal_translation_geometry.run_compact_geometry_audits \
  --demote-small-sessions
```

This additionally writes `panelE_session_effects_min50.csv` and
`panelE_covariance_closure_min50_summary.csv`, comparing all sessions against
the main set with `n_common_units >= 50`.

## Metric Validation

The current spec promotes hidden-coordinate metric validation to the main
coordinate-like test. Run it after the panel builder and before the audit:

```bash
uv run python -m declan.compact_retinal_translation_geometry.run_metric_structure_validation
```

This writes the promoted metric-validation tables:

- `metric_structure_local_metric.csv`
- `metric_structure_quadratic_prediction.csv`
- `metric_structure_opposition.csv`
- `metric_structure_scaling.csv`
- `metric_structure_composition.csv`
- `metric_structure_coordinate_recovery.csv`
- `metric_structure_cross_image_regularities.csv`
- `metric_structure_summary.csv`

The current tangent-map cache supports cardinal `+/-x` and `+/-y` translated
responses across the available step sweep. The validator therefore reports
local metrics, cardinal/step-sweep quadratic prediction, scaling, opposition,
and coordinate recovery now, while marking diagonal composition and true
direction-held-out prediction as unavailable until diagonal/arbitrary
translations are added to the cache.

## What Is Reused

- Panel A uses real `r0`, `bx`, and `by` entries from
  `tangent_maps/twin_tangent_maps.pkl`.
- Panel B adapts the tangent union spectrum and unit-shuffle null spectrum from
  `union_spectrum/`.
- Panel C adapts the image-disjoint train/test compact-basis results from
  `split_modes/image_disjoint/`.
- Panel E adapts full and compact finite-difference covariance-closure metrics
  from the RF-backed closure run.
- Panel D is a conservative budget adapter from the available closure traces and
  capture values. It explicitly marks unavailable denominators rather than
  inventing them.

## Still To Promote

The optional metric-structure and recorded displacement-decoding analyses need
raw translated response grids and recorded repeat-pair response/eye-position
objects. This folder should own those implementations when they are promoted
from supplement or exploratory status.

The current audit runner consumes the promoted metric-validation tables when
present. Recorded displacement decoding remains not-run until the required
repeat-pair response/eye-position objects are available.
