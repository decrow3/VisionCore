# ssi_figure_v4 cache artifact manifest

Purpose: make the Fig. 4 handoff auditable. This file is a cache provenance
manifest for the `ssi_figure_v4` bundle: one row per shared artifact, with its
panel/use, intended producer, important inputs, model dependency, expected
runtime, regeneration status, and known blockers.

All paths are repo-relative unless explicitly marked as absolute. The final v4
panel lettering is A-H. Some source modules and provenance files retain older
local names such as `panel_g`, `panel_h`, `panel_j`, and `panel_k`.

## Current package status

The current handoff bundle lives at:

```text
outputs/fig/ssi_figure_v2/handoff/ssi_figure_v4_derived_refresh_20260801/
```

Verified on 2026-08-04. The cache overlay and dependency patch checksums are:

```text
531c003a01429610dcc0be4fb5cf056995f88f7bb1fd729c1a9e3ddb4b37575f  ssi_figure_v4_cache_overlay.tar.gz
ae268c232288f62f950858c2b5130a4707c53c46b22b50b33a5f08a13d72476e  ssi_figure_v4_pyproject_dependency_patch.diff
```

The code overlay contains this manifest, so its checksum should be computed
externally after packaging rather than embedded here.

The handoff lists contain 44 code files, 27 cache-first artifacts, and 10
lower-level cache roots. Every listed cache-first artifact and lower-level cache
root existed locally at the time this manifest was written.

## Key model and data identity

Primary model population:

```text
V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_medoidPosthocminRepcomplete0p45_movieMedoid
```

This is captured as `rr100_version` in the lower-level SSI matrix summaries and
the Panel A schematic RR100 summary. The exact checkpoint filesystem path is
not recorded in the figure cache artifacts; local code resolves it through
`load_population_view(...)` and `CanonicalTwinScorer`.

Frame convention:

- Model input history is 32 lags, or 267 ms at 120 Hz.
- The main trace-bank SSI matrix and behavior bridge use 40 scored trace
  samples, whose sample centers span 0.325 s.
- Panel A's illustrated model input cube uses the last 32 samples from a
  center-40 trace.

Reviewed BackImage sessions in the shared behavior/image-structure cache:

```text
Allen_2022-02-16, Allen_2022-02-18, Allen_2022-02-24, Allen_2022-03-02,
Allen_2022-03-04, Allen_2022-03-30, Allen_2022-04-01, Allen_2022-04-06,
Allen_2022-04-08, Allen_2022-04-13, Allen_2022-04-15, Allen_2022-06-01,
Allen_2022-06-10, Allen_2022-08-05, Logan_2019-12-20, Logan_2019-12-23,
Logan_2019-12-24, Logan_2019-12-26, Logan_2019-12-30, Logan_2019-12-31,
Logan_2020-01-06, Logan_2020-01-07, Logan_2020-01-09, Logan_2020-01-10,
Logan_2020-01-15, Logan_2020-02-28, Logan_2020-02-29, Logan_2020-03-02,
Logan_2020-03-04, Logan_2020-03-06
```

## Cache-first artifacts

These are the small artifacts that Ryan needs for a minimal recomposition or
lightweight derived refresh. They are the files listed in
`ssi_figure_v4_cache_filelist.txt`.

| Artifact path | Figure panel/use | Producer script | Command/config | Inputs | Model checkpoint | Expected runtime | Can regenerate now? | Known blockers |
|---|---|---|---|---|---|---|---|---|
| `outputs/fig/ssi_figure_v2/panels/panel_bcef_path_bins_values.csv` | v4 B and D, path-bin SSI summaries | `declan/active_sensing_movie_information/make_backimage_panel_b_orientation_match_15deg_sf05.py`, then `declan/fig/ssi_figure_v2/panels/panel_bcef_path_bins.py` | `uv run python declan/active_sensing_movie_information/make_backimage_panel_b_orientation_match_15deg_sf05.py`; `uv run python declan/fig/ssi_figure_v2/panels/panel_bcef_path_bins.py` | Deep SSI matrix `.../backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/phase1_phase2_conditioning_v1/plot_collections/...`, stabilized baseline, unit tuning table | RR100 model via lower SSI matrix | Minutes if lower caches exist; about 20.3 GPU-hours if deep matrix absent | Yes, from listed lower caches | Exact original phase1/phase2 launch command is not saved in this handoff; wrapper provenance identifies source values and summary |
| `outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic_values.csv` | v4 E model RMS/path/range dose curves; v4 G bridge model dose input | `declan/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic.py` | `uv run python declan/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic.py` | Lower B/D source tables, trace-bank reference values, model population definitions | RR100 model via lower SSI matrix | Seconds to minutes from lower caches | Yes | Historical filename says `panel_g`; final v4 uses it mainly for E and G |
| `outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic_last_bin_contrasts.csv` | v4 E rightmost-bin across-minus-along bracket and p value | Same as previous row | Same as previous row | Same as previous row | Same as previous row | Seconds to minutes | Yes | Same historical naming caveat |
| `outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic_trace_bank_reference.csv` | v4 E gray reference band and behavior bridge dose scale | Same as previous row | Same as previous row | Deep SSI trace-bank metrics; drift-only component RMS/path/range values | Same as previous row | Seconds to minutes | Yes | Far-tail values are retained in CSV even when omitted from displayed Panel E |
| `outputs/fig/ssi_figure_v2/panels/panel_g_alternative_x_axes_diagnostic_populations.csv` | v4 E/G population definitions and selected-unit counts | Same as previous row | Same as previous row | Unit tuning groups and contour relation filters | Same as previous row | Seconds to minutes | Yes | Population labels include all, aligned, oblique, orthogonal variants; final v4 emphasizes high-SF aligned |
| `outputs/fig/ssi_figure_v2/panels/panel_h_unwrapped_edge_coherence_values.csv` | v4 F, real FEM RMS as function of contour-relative angle/coherence | `declan/fixation_statistics_by_stimulus/plot_backimage_contour_motion_components.py`, then `declan/fig/ssi_figure_v2/panels/panel_h_unwrapped_edge_coherence.py` | `uv run python declan/fixation_statistics_by_stimulus/plot_backimage_contour_motion_components.py --no-recompute-traces`; `uv run python declan/fig/ssi_figure_v2/panels/panel_h_unwrapped_edge_coherence.py` | Reviewed BackImage image-structure windows and contour motion component windows | None | Minutes from saved trace columns; longer if recomputing traces from raw session data | Yes, if reviewed windows exist | Source run metadata says the original plot run used `recompute_traces=true`; the handoff recipe uses `--no-recompute-traces` for cache-first refresh |
| `outputs/fig/ssi_figure_v2/panels/panel_h_unwrapped_edge_coherence_random_orientation_reference.csv` | v4 F random-orientation reference | Same as previous row | Same as previous row | Same as previous row | None | Minutes from saved trace columns | Yes | Same trace recompute caveat |
| `outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1/panel_f_descriptive_hierarchical_profiles_v1/panel_f_hierarchical_profiles.csv` | updated descriptive v4 F candidate, fixed-animal equal-weight contour-relative profiles and hierarchical CIs | `declan/fig/ssi_figure_v2/behavior_confounds/build_panel_f_descriptive_hierarchical_profiles.py` | `uv run python declan/fig/ssi_figure_v2/behavior_confounds/build_panel_f_descriptive_hierarchical_profiles.py` | Reviewed `contour_motion_component_windows.csv`; 11,749 windows, 1,962 trials, 30 sessions | None | Seconds | Yes | Candidate is preserved separately and is not yet wired into `compose_ssi_figure_v4.py` |
| `outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1/panel_f_descriptive_hierarchical_profiles_v1/visual_variations_v1/panel_f_visual_compendium.pdf` | updated v4 F polar, normalized-shape, animal, and session visual audit | `declan/fig/ssi_figure_v2/behavior_confounds/plot_panel_f_profile_variations.py` | `uv run python declan/fig/ssi_figure_v2/behavior_confounds/plot_panel_f_profile_variations.py` | Updated Panel F hierarchical profile, contrast, and trial-profile tables | None | Seconds | Yes | Diagnostic compendium; zoomed polar view has a nonzero radial origin and is not the proposed main panel |
| `outputs/fig/ssi_figure_v2/panels_v3/panel_k_patch_radius_alignment_slope_values.csv` | v4 H, patch-radius slope curve | `declan/fixation_statistics_by_stimulus/summarize_backimage_patch_radius_sensitivity.py`, then `declan/fig/ssi_figure_v2/panels/panel_k_patch_radius_alignment_slope.py` | `uv run python declan/fixation_statistics_by_stimulus/summarize_backimage_patch_radius_sensitivity.py`; `uv run python declan/fig/ssi_figure_v2/panels/panel_k_patch_radius_alignment_slope.py` | Patch-radius image-structure outputs at 0.25, 0.5, default 1.0/slope, plus expanded alignment-sweep cache | None | Seconds to minutes from lower patch-radius caches | Yes | Final v4 calls this Panel H; file and source module retain `panel_k` historical naming |
| `outputs/fig/ssi_figure_v2/panels/behavior_component_path_by_coherence_windows.csv` | v4 F/G, behavior-window table used for observed and rotated contour matching | `declan/fig/ssi_figure_v2/panels/behavior_component_path_by_coherence.py` | `uv run python declan/fig/ssi_figure_v2/panels/behavior_component_path_by_coherence.py` | Reviewed BackImage windows and contour motion component windows | None | Seconds to minutes | Yes | Absolute input paths appear in provenance JSON but repo-relative equivalents exist on solo |
| `outputs/fig/ssi_figure_v2/panels/behavior_component_path_by_coherence_summary.csv` | v4 F behavior summary | Same as previous row | Same as previous row | Same as previous row | None | Seconds to minutes | Yes | Summary is session-bootstrap over session medians, not per-window iid uncertainty |
| `outputs/fig/ssi_figure_v2/panels/behavior_component_path_by_coherence_alignment_summary.csv` | v4 F contour-alignment summary | Same as previous row | Same as previous row | Same as previous row | None | Seconds to minutes | Yes | Uses local image edge axis from Sobel-gradient structure tensor |
| `outputs/fig/ssi_figure_v2/panels/behavior_component_path_by_coherence_directional_path_random_orientation_reference.csv` | v4 F/G behavior random-orientation reference | Same as previous row | Same as previous row | Same as previous row | None | Seconds to minutes | Yes | Random orientation reference is flat over relative angle by construction |
| `outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_coherence_contrasts.csv` | v4 G bridge diagnostic/supporting contrast table | `declan/fig/ssi_figure_v2/behavior_model_bridge/run_behavior_model_bridge.py` | `uv run python declan/fig/ssi_figure_v2/behavior_model_bridge/run_behavior_model_bridge.py` | Behavior windows plus Panel E model dose tables | RR100 model only through dose tables | Seconds to minutes | Yes | This interpolates through one-dimensional model dose curves; it is not a direct 2D model evaluation |
| `outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_prediction_summary.csv` | v4 G observed behavior-weighted predicted SSI by coherence | Same as previous row | Same as previous row | Same as previous row | Same as previous row | Seconds to minutes | Yes | Predictions outside model dose range are set to NaN and reported in summaries |
| `outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_random_rotation_match_null_rotation_values.csv` | v4 G random-rotation null replicate values | `declan/fig/ssi_figure_v2/behavior_model_bridge/run_random_rotation_match_null.py` | `uv run python declan/fig/ssi_figure_v2/behavior_model_bridge/run_random_rotation_match_null.py` | Behavior windows plus Panel E model dose tables | RR100 model only through dose tables | Minutes; null uses 256 rotations and 10000 bootstrap resamples | Yes | Rotations are a bridge null on projected behavior metrics, not direct model movies |
| `outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_random_rotation_match_null_session_values.csv` | v4 G paired session values for null summary | Same as previous row | Same as previous row | Same as previous row | Same as previous row | Minutes | Yes | Session bootstrap, not image bootstrap |
| `outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_random_rotation_match_null_summary.csv` | v4 G main observed-minus-rotated support table | Same as previous row | Same as previous row | Same as previous row | Same as previous row | Minutes | Yes | Component-mean marginal is an average of contour-normal and contour-parallel one-dimensional predictions |
| `outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_random_rotation_prediction_by_coherence_session_values.csv` | v4 G paired session values split by coherence | `declan/fig/ssi_figure_v2/behavior_model_bridge/run_random_rotation_prediction_by_coherence.py` | `uv run python declan/fig/ssi_figure_v2/behavior_model_bridge/run_random_rotation_prediction_by_coherence.py` | Behavior windows plus Panel E model dose tables | RR100 model only through dose tables | Minutes; null uses 256 rotations and 10000 bootstrap resamples | Yes | Same one-dimensional marginal bridge caveat |
| `outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_random_rotation_prediction_by_coherence_summary.csv` | v4 G plotted observed-minus-rotated advantage by coherence | Same as previous row | Same as previous row | Same as previous row | Same as previous row | Minutes | Yes | Same one-dimensional marginal bridge caveat |
| `outputs/fig_ssi/trace_provenance/schematic_crop_real_backimage_trace_center40.csv` | v4 A, source real FEM trace before 32-frame lag crop | `declan/fig_ssi/make_ssi_contour_schematic.py` | Generated by schematic provenance path; normal rebuild reads this cache | Reviewed BackImage trace provenance | None directly | Cache read is instant; recompute from raw BackImage trace should be seconds | Partly | Exact trace-selection command is not separately captured |
| `outputs/fig_ssi/trace_provenance/schematic_crop_real_backimage_trace_full128.csv` | v4 A/C audit provenance for selected real trace | `declan/fig_ssi/make_ssi_contour_schematic.py` | Generated by schematic provenance path | Reviewed BackImage trace provenance | None directly | Cache read is instant | Partly | Exact trace-selection command is not separately captured |
| `outputs/fig_ssi/rr100_schematic_endpoint_final_maps/cache/schematic_rr100_final_maps.npz` | v4 A final real-vs-stabilized activation maps | `declan/fig_ssi/compute_schematic_rr100_final_maps.py` | `uv run python declan/fig_ssi/compute_schematic_rr100_final_maps.py` | Selected BackImage patch, center-40 trace cropped to 32 lags, RR100 model | RR100 model via `load_population_view` and `CanonicalTwinScorer` | About minutes on GPU; cache read is instant | Yes, if model loaders and raw BackImage patch resolve | Exact checkpoint path not recorded, only RR100 version |
| `outputs/fig_ssi/rr100_schematic_endpoint_final_maps/schematic_rr100_final_map_unit_metrics.csv` | v4 A unit selection and map metrics | Same as previous row | Same as previous row | Same as previous row | Same as previous row | About minutes on GPU | Yes | Same checkpoint-path caveat |
| `outputs/fig/ssi_figure_v2/panels/cache/coherence_gallery.npz` | v4 C, example local contour patches by coherence | `declan/fig/ssi_figure_v2/panels/build_coherence_gallery_cache.py` | `uv run python declan/fig/ssi_figure_v2/panels/build_coherence_gallery_cache.py` | Reviewed BackImage image-structure windows and raw BackImage canvases | None | Seconds to minutes | Yes | Requires raw BackImage canvases via local fixation-statistics helpers |
| `outputs/fig/ssi_figure_v2/panels/cache/panel_a_layout_overrides.json` | v4 A layout support | Hand-tuned local layout cache | Normal recomposition reads this cache | Figure layout tuning | None | Instant | Yes, as cache | Manual tuning artifact; no scientific recompute needed |
| `outputs/fig/ssi_figure_v2/panels/cache/panel_d_layout_overrides.json` | v4 C layout support | Hand-tuned local layout cache | Normal recomposition reads this cache | Figure layout tuning | None | Instant | Yes, as cache | Manual tuning artifact; no scientific recompute needed |
| `outputs/fig/ssi_figure_v2/panels/cache/panel_a_network_icon.pdf` | v4 A compositing asset | `declan/fig/ssi_figure_v2/panels/extract_panel_a_network_icon.py` | Optional: `uv run python declan/fig/ssi_figure_v2/panels/extract_panel_a_network_icon.py` | `declan/fig/ssi_figure_v2/ssi_figure_v2_3.pdf` | None | Seconds | Yes, if reference PDF is available | If this PDF is shipped, Ryan does not need PyMuPDF/PDF extraction for ordinary recomposition |

## Lower-level cache roots

These are larger or more raw derived outputs that are not part of the small
cache overlay unless Ryan wants to regenerate the cache-first tables. They are
the files and directories listed in `ssi_figure_v4_large_filelist.txt`.

| Artifact path | Figure panel/use | Producer script | Command/config | Inputs | Model checkpoint | Expected runtime | Can regenerate now? | Known blockers |
|---|---|---|---|---|---|---|---|---|
| `outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/` | Lower-level model output for v4 B/D/E/G | `declan/active_sensing_movie_information/run_backimage_real_trace_ssi_matrix_pilot.py`, shard merge, plus `run_backimage_real_trace_stabilized_baseline.py` | Inferred two shards with `--n-images 100 --n-traces 1000 --n-timepoints 40 --patch-size-px 540 --trace-sampling quantile --max-trace-path-length-arcmin 350 --min-microsaccade-traces 200 --pilot-frame-batch-size 16 --pilot-trace-batch-size 8 --image-shard-start 0/50 --image-shard-stop 50/100`; baseline command `uv run python declan/active_sensing_movie_information/run_backimage_real_trace_stabilized_baseline.py --matrix-dir .../merged` | Reviewed BackImage image-structure windows; raw BackImage canvases; unit tuning table | RR100 model version listed above | Scoring shards total about 73029 s, or 20.3 GPU-hours; stabilized baseline about 462 s | Yes in principle on solo | Exact original shell command and merge command were not saved verbatim |
| `outputs/active_sensing_movie_information/backimage_rr100_instantaneous_unit_maps_latest_v1/` | Lower input for unit tuning and frequency probes | `declan/active_sensing_movie_information/plot_backimage_rr100_instantaneous_unit_maps.py` or related RR100 unit-map generator | Not captured in this handoff | Raw BackImage patches and RR100 model | RR100 model version listed above | Unknown, GPU likely | Unknown from handoff alone | Producer command not captured; keep as true input unless Ryan needs to rebuild unit tuning |
| `outputs/active_sensing_movie_information/backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/` | Unit SF grouping for v4 B/D/E/G | `declan/active_sensing_movie_information/run_backimage_rr100_frequency_tuning_probe.py` | Defaults include `--duration-s 1.5 --frame-rate-hz 120 --n-lags 32 --discard-frames 32 --image-size 101 --device cuda:1 --batch-size 16`; exact original launch not captured | RR100 instantaneous unit maps and model population | RR100 model version listed above | Unknown, GPU likely | Exact original launch and source unit-map provenance are not captured |
| `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure_reviewed_v2_screenfiltered_yfix/` | Reviewed BackImage window table for v4 C/F/G and trace/image selection | `declan/fixation_statistics_by_stimulus/run_backimage_image_structure_analysis.py` | Config from `run_metadata.json`: `--input-window-features outputs/fixation_statistics_by_stimulus_all_sessions_after_review/window_features.csv --patch-radius-deg 1.0 --n-shuffles 200 --n-splits 5 --min-patch-fraction-inside-image 0.98 --max-patch-fraction-background 0.05 --phases mid_fixation,late_fixation` | `window_features.csv`, raw BackImage canvases | None | Minutes to hours depending raw image loading | Yes if raw data helpers resolve | Raw session data path assumptions live outside this package |
| `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure_reviewed_v2_screenfiltered_yfix_slope_v1/` | 1.0 deg patch-radius slope variant for v4 H | `declan/fixation_statistics_by_stimulus/run_backimage_image_structure_analysis.py` | Same as previous row, except `--n-shuffles 0 --out-dir ..._slope_v1` | `window_features.csv`, raw BackImage canvases | None | Minutes to hours | Yes if raw data helpers resolve | This is the 1.0 deg patch-radius input; there is no `backimage_image_structure_patch_radius_1p0_v1/` path in the current handoff |
| `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/window_features.csv` | True upstream input for BackImage image-structure analyses | Fixation-statistics extraction pipeline | Not captured in this figure handoff | Raw session metadata, fixation windows, eye traces, BackImage trial metadata | None | Unknown | Treat as true input for this package | Producer command and upstream raw-data contract are outside the small Fig. 4 overlay |
| `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure_patch_radius_0p25_v1/` | Patch-radius sensitivity lower input for v4 H | `declan/fixation_statistics_by_stimulus/run_backimage_image_structure_analysis.py` | Config from `run_metadata.json`: `--patch-radius-deg 0.25 --n-shuffles 0 --n-splits 5 --min-patch-fraction-inside-image 0.98 --max-patch-fraction-background 0.05` | `window_features.csv`, raw BackImage canvases | None | Minutes to hours | Yes if raw data helpers resolve | Raw session data path assumptions live outside this package |
| `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure_patch_radius_0p5_v1/` | Patch-radius sensitivity lower input for v4 H | `declan/fixation_statistics_by_stimulus/run_backimage_image_structure_analysis.py` | Same as previous row, with `--patch-radius-deg 0.5` | `window_features.csv`, raw BackImage canvases | None | Minutes to hours | Yes if raw data helpers resolve | Raw session data path assumptions live outside this package |
| `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_contour_motion_component_plots_v1/` | Lower behavior/motion summaries for v4 F/G | `declan/fixation_statistics_by_stimulus/plot_backimage_contour_motion_components.py` | Config from `run_metadata.json`: input windows `.../backimage_image_structure_reviewed_v2_screenfiltered_yfix_slope_v1/backimage_image_fem_windows.csv`, `n_bootstrap=1000`, `dt=1/120`, `recompute_traces=true`; cache-first recipe may use `--no-recompute-traces` | Reviewed image-structure windows, `window_features.csv`, raw traces if recomputing | None | Minutes from cached trace columns; longer if recomputing traces | Yes | Original metadata has `recompute_traces=true`; handoff run order chooses a cache-first mode |
| `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_patch_radius_sensitivity_v1/` | Lower patch-radius summary for v4 H | `declan/fixation_statistics_by_stimulus/summarize_backimage_patch_radius_sensitivity.py` | `uv run python declan/fixation_statistics_by_stimulus/summarize_backimage_patch_radius_sensitivity.py` | 0.25, 0.5, and 1.0/slope image-structure runs plus expanded cached alignment sweep | None | Seconds to minutes if sweep cache is present; longer if rebuilding expanded sweep | Yes | Expanded 0.75-3.0 deg alignment sweep is cached inside this root rather than separate image-structure directories |

## Minimal conceptual pipeline

```text
raw session data and model weights
  -> window_features.csv and BackImage image-structure windows
  -> contour motion component summaries
  -> behavior component path/coherence summaries

raw BackImage patches, sampled real traces, and RR100 model
  -> 100 image x 1000 trace SSI matrix plus stabilized baseline
  -> phase1/phase2 conditioning and path-bin source tables
  -> panel_bcef_path_bins_values.csv and panel_g_alternative_x_axes_diagnostic_*.csv

behavior summaries + model dose curves
  -> behavior_model_bridge_*.csv
  -> random-rotation bridge summaries

image-structure patch-radius runs
  -> patch_radius_sensitivity_v1
  -> panel_k_patch_radius_alignment_slope_values.csv

schematic caches and layout assets
  -> per-panel PDFs
  -> outputs/fig/ssi_figure_v2/ssi_figure_v4.pdf
```

## Raw/source data required before any cache exists

- RR100 model weights/config resolvable by `load_population_view(...)` and
  `CanonicalTwinScorer`.
- BackImage raw session canvases/trials resolvable by
  `declan.fixation_statistics_by_stimulus.image_features._backimage_canvas`.
- Eye-position traces and reviewed fixation-window metadata for the 30 sessions
  listed above.
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/window_features.csv`
  or the upstream extraction pipeline that can recreate it.
- Unit tuning and grouping source tables, especially the dynamic log-Gaussian
  marginal SF grouping under
  `outputs/active_sensing_movie_information/backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/`.

## Validation targets

Cache-first reproduction:

```bash
uv run python declan/fig/ssi_figure_v2/compose_ssi_figure_v4.py
```

Expected outputs:

```text
outputs/fig/ssi_figure_v2/ssi_figure_v4.pdf
outputs/fig/ssi_figure_v2/ssi_figure_v4_provenance.json
outputs/fig/ssi_figure_v2/panels_v4/
```

Recommended validation levels:

1. Cache inventory: every path in `ssi_figure_v4_cache_filelist.txt` exists.
2. Numeric spot checks: verify Panel E first nonzero RMS-bin and last visible
   across-minus-along values, Panel G highest-coherence observed-minus-rotated
   value, and Panel H max slope at 1.25 deg against the manuscript notes.
3. Figure rebuild: compare the regenerated `ssi_figure_v4.pdf` against the
   shared PDF. Pixel-identical output is ideal but not required if Matplotlib/PDF
   versions differ; panel-level numerical CSVs should match.
4. Deep refresh: if rebuilding model outputs, compare summary dimensions:
   100 images, 1000 traces, 100000 movies, 100 units, `n_timepoints=40`, and the
   RR100 model version above.

## Known package risks

- The exact checkpoint file path is not recorded, only the RR100 population
  version name used by local loaders.
- The deepest SSI matrix launch commands were reconstructed from summaries and
  parser defaults, not from a saved shell transcript.
- Several direct panel artifacts have historical source names that do not match
  final v4 lettering.
- `window_features.csv` is a true input for the compact Fig. 4 package; its
  raw extraction pipeline is not fully bundled here.
- Some provenance JSONs contain absolute `/home/declan/VisionCore/...` paths.
  Ryan on the same solo machine can use them as-is; for portability they should
  be interpreted as repo-relative under `/home/declan/VisionCore`.
