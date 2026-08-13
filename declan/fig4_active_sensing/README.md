# Figure 4 Active Sensing

This folder is the clean Figure 4 active-sensing workspace.

Current state: `generate_fig4_active_sensing.py` builds a cache-first headline
figure from the cleaned BackImage aggregate FEM-information analysis and local
image-geometry support tables. The figure is intended as the functional
counterpart to the paper's FEM-linked reafferent variability story: self-generated
retinal motion is not only shared variability to subtract, but can supply
feature-relevant temporal samples while being constrained by local
image-preserving geometry. Panels D-E are arranged as prediction followed by
behavior: edge-parallel motion is predicted to preserve local image/V1-twin
structure, and measured drift axes overrepresent edge-parallel orientation
zones relative to a uniform axial baseline.
The older active-sensing movie-information figure is now preserved as
historical/supporting context rather than the default Figure 4 active-sensing
output.

Default inputs:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/
    incremental_static_plus_motion_relids
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_image_structure_reviewed_v2_screenfiltered_yfix
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_edge_parallel_stability_screen_yfix_n256_pop256
```

The generator uses `aggregate_motion_metadata.csv` for compact drift-bank QC and
`backimage_image_fem_windows.csv` for session-bootstrap CIs on the edge-axis
alignment panel.

Default outputs:

```text
outputs/fig4_active_sensing/active_sensing_headline_figure
```

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.fig4_active_sensing.generate_fig4_active_sensing
```

The map-first checkpoint for the updated RR100 parametric SF/TF preferences is
generated separately so it does not silently overwrite the historical Figure
4 SF grouping:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.fig4_active_sensing.plot_rr100_sf_tf_parametric_preferences
```

It writes joint-preference diagnostics, auditable example SF-by-TF surfaces,
the old/new unit-level comparison, and the algorithmic example-selection table
to `outputs/fig4_active_sensing/rr100_sf_tf_parametric_preference_plots_v1`.
The historical `0.5 cpd` SF split is outside the support of the new fitted
preferences and must not be reused without an explicit replacement grouping
rule.

The subsequent pre-figure iteration checks use quartiles of the new preferred
SF estimates and stop before rebuilding any Figure 4 SSI panel:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.fig4_active_sensing.plot_rr100_sf_quartile_iteration_checks
```

They are written to
`outputs/fig4_active_sensing/rr100_sf_quartile_iteration_checks_v1` in the same
diagnostic order used for the prior grouping pass: marginal definition, group
SF-by-TF maps, individual contours, and fitted ellipses.

The first downstream checkpoint reproduces the old pilot SF-by-trace-path test
with those quartiles, and deliberately stops before phase conditioning:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.fig4_active_sensing.rerun_backimage_real_trace_pilot_sf_quartiles
```

It writes the exact page-2 analog, a drift-only trend audit, the old-to-new
assignment crosswalk, and machine-readable source identities to
`outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/checkpoint_01_pilot_sf_by_trace_path`.

Checkpoint 2 reproduces the old page-7 spike-weighted SF comparison and adds
trace-first, equal-unit, and stabilized-baseline audits:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.fig4_active_sensing.rerun_backimage_real_trace_phase2_sf_quartiles
```

It writes to
`outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/checkpoint_02_phase2_sf_by_trace_path`.

Checkpoint 3 reproduces the page-12 contour-matched unit-first curves using Q1
versus Q4, while retaining all quartiles and selection support in audits:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.fig4_active_sensing.rerun_backimage_real_trace_contour_matched_sf_quartiles
```

It writes to
`outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/checkpoint_03_contour_matched_sf_quartiles`.

The targeted checkpoint-3 map drill-down renders u054, the median Q3 unit u005,
and the median Q4 control u046 on algorithmically selected real-trace movies:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.fig4_active_sensing.render_contour_matched_unit_drilldown --device cuda:0
```

It writes targeted activation maps, map-derived SSI/rate timecourses, selection
provenance, and cached-versus-rerendered SSI validation beneath checkpoint 3 in
`targeted_unit_drilldown_v1`.

Internal matrix keys `q01` and `q06` denote the shortest and longest sixths of
the trace-path-length distribution, respectively; rendered figures label these
as “short-path trace” and “long-path trace” to avoid confusion with SF quartiles.

Checkpoint 4 reproduces the page-13 contour-orthogonal unit-first curves using
Q1 versus Q4, with all SF quartiles and unit-level support retained in audits:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.fig4_active_sensing.rerun_backimage_real_trace_contour_orthogonal_sf_quartiles
```

It writes to
`outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/checkpoint_04_contour_orthogonal_sf_quartiles`
and deliberately stops before the page-14 aligned-versus-orthogonal overlay.

Checkpoint 5 overlays the aligned and orthogonal curves for the page-14 analog
and adds a paired within-unit difference-of-differences audit:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.fig4_active_sensing.rerun_backimage_real_trace_aligned_vs_orthogonal_sf_quartiles
```

It writes to
`outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/checkpoint_05_aligned_vs_orthogonal_sf_quartiles`.

Checkpoint 6 rebuilds the page-15 response-strength control for the full set of
new SF-quartile units, comparing preferred/aligned and orthogonal image windows
for mean rate, expected spikes, and SSI:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.fig4_active_sensing.rerun_backimage_alignment_response_sf_quartiles
```

It writes to
`outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/checkpoint_06_alignment_response_sf_quartiles`.

Checkpoint 7 regenerates the page-16 all-images population summary using all
85 valid parametric fits in tie-aware SF quartiles, with both spike-weighted and
equal-unit estimands retained:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.fig4_active_sensing.rerun_backimage_all_images_population_sf_quartiles
```

It writes to
`outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/checkpoint_07_all_images_population_sf_quartiles`.

Checkpoint 8 regenerates page 17 by projecting trace steps across and along
each image's contour axis and forming equal-count image-by-trace component bins:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.fig4_active_sensing.rerun_backimage_all_images_component_sf_quartiles
```

It writes to
`outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/checkpoint_08_all_images_component_sf_quartiles`.

Checkpoint 9 regenerates page 18 using only the 53 strong-contour image
windows, all 85 valid parametric fits, and the new tie-aware SF quartiles. It
retains spike-weighted and equal-unit estimands as a population-weighting audit:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.fig4_active_sensing.rerun_backimage_strong_contours_population_sf_quartiles
```

It writes to
`outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/checkpoint_09_strong_contours_population_sf_quartiles`
and stops before the page-19 across/along component decomposition.

Checkpoint 10 regenerates page 19 on the same strong-contour image windows,
decomposing trace motion into across- and along-contour component-path bins:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.fig4_active_sensing.rerun_backimage_strong_contours_component_sf_quartiles
```

It writes to
`outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/checkpoint_10_strong_contours_component_sf_quartiles`
and stops before later orientation-conditioned analyses.

Checkpoint 11 regenerates historical figure label 020 for strong-contour image
windows paired with orientation-aligned units, retaining the historical
trial-mean stabilized baseline while replacing the SF grouping with the new
tie-aware quartiles:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.fig4_active_sensing.rerun_backimage_contour_matched_population_sf_quartiles
```

It writes to
`outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/checkpoint_11_contour_matched_population_sf_quartiles`.
Later cell-matched Figure 4 revisions are intentionally kept separate rather
than silently mixed into this bridge checkpoint.

Checkpoint 12 regenerates historical figure label 021 by decomposing the same
orientation-aligned selection into across- and along-contour component paths:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.fig4_active_sensing.rerun_backimage_contour_matched_component_sf_quartiles
```

It writes to
`outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/checkpoint_12_contour_matched_component_sf_quartiles`
and stops before the intermediate-orientation population analysis.

Checkpoint 13 regenerates historical figure label 022 for the disjoint
intermediate-orientation unit–contour relation using the new SF quartiles:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.fig4_active_sensing.rerun_backimage_contour_intermediate_population_sf_quartiles
```

It writes to
`outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/checkpoint_13_contour_intermediate_population_sf_quartiles`
and stops before the intermediate-orientation component decomposition.

Checkpoint 14 regenerates historical figure label 023 by decomposing the
intermediate-orientation selection into across- and along-contour component
paths:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.fig4_active_sensing.rerun_backimage_contour_intermediate_component_sf_quartiles
```

It writes to
`outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/checkpoint_14_contour_intermediate_component_sf_quartiles`
and stops before the orientation-orthogonal population analysis.

Checkpoints 15-17 finish historical figure labels 024-026: the
orientation-orthogonal population view, its across/along decomposition, and the
four-quartile mixed-context presentation panel:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.fig4_active_sensing.rerun_backimage_final_pages_sf_quartiles
```

They write to `checkpoint_15_contour_orthogonal_population_sf_quartiles`,
`checkpoint_16_contour_orthogonal_component_sf_quartiles`, and
`checkpoint_17_mixed_context_sf_quartiles` beneath the common output root.

The complete 26-figure sequence can then be collected into a fresh 27-page PDF
(including its contents page) without modifying the original reference PDF:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.fig4_active_sensing.collect_backimage_sf_quartile_multipage
```

It writes
`outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/backimage_real_trace_key_figures_sf_quartiles_multipage_v1.pdf`
plus a source-identity manifest beside it.

## Low/high SF half summary

To collapse the new parametric SF assignments into two equally sized groups
and regenerate the key population, component, weighting-audit, and mixed-context
plots, run:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.fig4_active_sensing.rerun_backimage_key_results_sf_halves
```

The split is the observed median preferred SF among the 85 valid fits: low SF
is `preferred_sf_cpd <= 2.550888783357088` (43 units) and high SF is above that
threshold (42 units). The script writes figures, per-relation summaries,
bootstrap intervals, unit/image selections, the auditable half assignment table,
and a manifest to
`outputs/fig4_active_sensing/backimage_real_trace_sf_half_checks_v1`. Existing
quartile outputs are not modified.

Main claim boundary:

- The figure uses the canonical 756-unit V1 twin, not the older 16-channel
  natural-image movie-information endpoint.
- The plotted endpoint is deterministic static-plus-motion feature-decoding gain
  over a static-only decoder in `-MSE` units, not literal mutual information.
- The supported claim is distributional and scale/readout scoped: empirical
  drift-like motion supplies feature-relevant temporal samples beyond static
  responses and robustly beats OU-like controls, with the clearest
  Brownian/rotated advantage at small scales retained in the control panel.
- Panels D-E are the local-geometry payoff: edge-parallel motion is predicted to
  preserve local image/V1-twin structure, and measured drift axes overrepresent
  edge-parallel orientation zones.
- Do not read this as exact trajectory prediction; the supported claim is a
  functional constraint on drift geometry.
- Motion sanity checks are documented in the generated stats manifest rather
  than plotted as a main panel.
