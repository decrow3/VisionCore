# Figure 4 Active Sensing

This folder is the clean Figure 4 active-sensing workspace.

Current state: `generate_fig4_active_sensing.py` builds a cache-first headline
figure from the cleaned BackImage aggregate FEM-information analysis and local
image-geometry support tables. The figure is intended as the functional
counterpart to the paper's FEM-linked reafferent variability story: self-generated
retinal motion is not only shared variability to subtract, but can supply
feature-relevant temporal samples while being constrained by local
image-preserving geometry. Panel E is arranged as prediction followed by
behavior: edge-parallel motion is predicted to preserve local image/V1-twin
structure, and measured drift axes are biased toward those stable directions.
The older active-sensing movie-information figure is now preserved as
historical/supporting context rather than the default Figure 4 active-sensing
output.

Default inputs:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/
    incremental_static_plus_motion_relids
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_image_structure_reviewed_v2_screenfiltered
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

Main claim boundary:

- The figure uses the canonical 756-unit V1 twin, not the older 16-channel
  natural-image movie-information endpoint.
- The plotted endpoint is deterministic static-plus-motion feature-decoding gain
  over a static-only decoder in `-MSE` units, not literal mutual information.
- The supported claim is distributional and scale/readout scoped: empirical
  drift-like motion supplies feature-relevant temporal samples beyond static
  responses and robustly beats OU-like controls, with the clearest
  Brownian/rotated advantage at small scales.
- The local-geometry panel is the payoff: edge-parallel motion is predicted to
  preserve local image/V1-twin structure, and measured drift axes are biased
  toward those stable directions.
- Do not read this as exact trajectory prediction; the supported claim is a
  functional constraint on drift geometry.
- Motion sanity checks are documented in the generated stats manifest rather
  than plotted as a main panel.
