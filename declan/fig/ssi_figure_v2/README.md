# SSI figure v2

Draft figure-generation folder for a multipanel SSI figure based on
`SSI_figure_v2_rkr (1).pdf`.

The generator is currently a composition scaffold with selected existing
analysis panels pulled in. It combines:

- schematic FEM/stabilized movie blocks, response maps, and stimulus/crop assets
  from `declan/fig_ssi/make_ssi_contour_schematic.py`,
- cell-baselined BackImage trajectory-path panels for B/C/E/F from
  `declan/active_sensing_movie_information/make_backimage_panel_b_orientation_match_15deg_sf05.py`,
- the matched-bin component-path errorbar/bracket panel for G,
- a reduced-bin unwrapped contour-relative position-spread panel for H, and
- a Figure-4E-style edge-coherence alignment panel for I.

## Typography

Use these sizes as the default style guide for manual refinement on an 8.5 x
11 inch page:

- Main figure title: 13.5 pt, bold.
- Figure subtitle/provenance line: 8.5 pt, gray.
- Large section headers for A and D: 14 pt panel letter, 12 pt bold section
  title.
- Quantitative panel titles for B/C/E/F/G/H/I: 8.5-9 pt, bold, colored only
  when the panel encodes low-SF blue or high-SF orange.
- Axis labels: 7-8 pt. Keep compact path-bin panels near 7 pt; use 8 pt for
  roomier standalone axes.
- Tick labels: 6.8-7 pt.
- Legends and unit/pair support notes: 5.7-6.1 pt.
- Schematic annotations: 5.2-7.2 pt, with the smallest text reserved for
  diagram labels rather than claims/results.
- Avoid adding threshold definitions to the figure itself; keep those in the
  methods/provenance JSON unless a panel absolutely needs them.

The live panels read precomputed values from:

```text
outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/phase1_phase2_conditioning_v1/plot_collections/
```

Panel-specific scripts live in `panels/` and can be run independently:

```bash
uv run python declan/fig/ssi_figure_v2/panels/panel_bcef_path_bins.py
uv run python declan/fig/ssi_figure_v2/panels/panel_g_matched_bins_bracket.py
uv run python declan/fig/ssi_figure_v2/panels/panel_h_unwrapped_edge_coherence.py
uv run python declan/fig/ssi_figure_v2/panels/panel_i_edge_alignment.py
```

Their data sources are:

```text
outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/phase1_phase2_conditioning_v1/plot_collections/backimage_real_trace_panel_b_cell_baseline_sf05_coh020_match15_{values,selection_summary,summary}.*
outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/phase1_phase2_conditioning_v1/plot_collections/backimage_real_trace_panel_c_aligned_sf_ge_0p5_match15_matched_bins_bracket_{values,last_bin_contrast,summary}.*
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/b_position_spread_unwrapped_profiles_by_edge_coherence.csv
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv
```

B/C/E/F and G intentionally share the same methods-facing selection definitions:
`sf_split_metric < 0.5` for low-SF units, `sf_split_metric >= 0.5` for
high-SF/aligned high-SF units, image contour coherence `>= 0.20`, orientation
selectivity `>= 0.05` for aligned unit-image matching, and unit-contour
orientation difference `<= 15 deg` for aligned panels. These details are not
printed on the figure; the generator writes them to:

```text
outputs/fig/ssi_figure_v2/ssi_figure_v2_methods_provenance.json
```

Run from the repository root:

```bash
uv run python declan/fig/ssi_figure_v2/generate_ssi_figure_v2.py
```

Outputs are written to:

```text
outputs/fig/ssi_figure_v2/ssi_figure_v2.{png,pdf,svg}
outputs/fig/ssi_figure_v2/ssi_figure_v2_panel_boxes.svg
```

`ssi_figure_v2_panel_boxes.svg` is an editable empty-box composition template
with one SVG group per panel (`panel-A` through `panel-I`). Move or resize those
boxes during manual layout refinement, then pass the edited SVG back as a guide.
