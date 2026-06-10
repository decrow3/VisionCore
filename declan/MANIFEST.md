# declan Manifest

Last curated: 2026-06-09.

This is a human-maintained map of the `declan/` workspace. It is intentionally
ordered newest-to-oldest in reading order where chronology is recoverable. The
repo has several live threads moving in parallel, so dates should be read as
"when this thread became active or was last materially touched", not as a strict
scientific dependency graph.

For the companion narrative of motivations, outcomes, and revised
interpretations, see `ANALYSIS_NARRATIVE.md`.

## How To Update

Add new work near the top, under the current date or a new date block. Prefer a
short thread-level description plus a file list. If a folder grows a README,
make this manifest point to that README and keep only the navigation summary
here. If a script becomes obsolete, mark it `historical` rather than deleting
the entry.

Chronology sources used here:

- Git history for committed `declan/` changes.
- File mtimes for uncommitted and generated artifacts.
- Existing README, handoff, plan, prescription, and spec documents.

Known caveat: generated caches/results often have older mtimes than the script
or handoff that interprets them.

## Current Active / Uncommitted Work

As of 2026-06-09, active uncommitted `declan/` files include:

- `active_sensing_movie_information/README.md`
- `active_sensing_movie_information/figure5_additional_checks_prep.md`
- `active_sensing_movie_information/run_figure5_cached_rate_checks_5_to_9.py`
- `active_sensing_movie_information/run_figure5_natural_image_population_checks_5_to_9.py`
- `active_sensing_movie_information/summarize_figure5_additional_checks.py`
- `compact_retinal_translation_geometry/run_compact_geometry_audits.py`
- `compact_retinal_translation_geometry/run_relative_displacement_decoding.py`
- `Figure5_active_sensing_triage_plan.md`

These are probably the best first files to check before trusting older Figure 5
or compact-geometry notes.

## Chronology: Newest To Oldest

### 2026-06-09: Active-Sensing Movie Information / Figure 5

Primary folder: `active_sensing_movie_information/`.

Purpose: active-sensing figure work built around Jake's `jake.twininfo`
pipeline, natural-image movie information, spatial SSI, Fisher/information
metrics, and matched-motion controls.

Important files:

- `active_sensing_movie_information/README.md`: current interpretation,
  claim guardrails, canonical pipeline, and notes about which outputs are safe
  to use.
- `active_sensing_movie_information/active_sensing_movie_information_plan.md`:
  full scientific and figure plan.
- `active_sensing_movie_information/data_and_code_inventory.md`: inventory of
  upstream assets and code to inspect.
- `active_sensing_movie_information/generate_active_sensing_movie_information_figure.py`:
  figure assembler from `outputs/twininfo/...` products, with audit/caption
  helpers.
- `active_sensing_movie_information/generate_retinal_movie_transform_qc.py`:
  retinal movie transform QC generator.
- `active_sensing_movie_information/run_active_sensing_movie_information.py`:
  temporary exploratory runner; not the canonical final pipeline.
- `active_sensing_movie_information/figure5_additional_checks_prep.md`:
  current checklist for additional Figure 5 audits and population checks.
- `active_sensing_movie_information/run_figure5_natural_image_population_checks_5_to_9.py`:
  current natural-image-only population check runner for Checks 5-9.
- `active_sensing_movie_information/run_figure5_cached_rate_checks_5_to_9.py`:
  historical e-optotype cached-rate scaffold for checks machinery.
- `active_sensing_movie_information/summarize_figure5_additional_checks.py`:
  summary/audit helper for additional Figure 5 checks.
- `Figure5_active_sensing_triage_plan.md`: top-level triage plan.
- `Figure5_random_amp_cloud_matched_control_spec.md`: spec for matched random
  amplitude/cloud controls.

Current interpretation: the safe claim is that retinal image motion exposes
structured information-bearing response variation in the deterministic twin,
especially in natural-image movies. The current notes explicitly warn against
claiming that real FEM trajectories are uniquely optimal based only on the
spatial-SSI endpoint.

### 2026-06-09 / 2026-06-08: Compact Retinal-Translation Geometry

Primary folder: `compact_retinal_translation_geometry/`.

Purpose: upgrade path and eventual replacement for `fig4_cov_TFTS`, using
existing production outputs to build spec-facing compact geometry tables,
audits, metric validation, and recorded displacement decoding.

Important files:

- `compact_retinal_translation_geometry/README.md`: runbook and relationship
  to upstream production artifacts.
- `compact_retinal_translation_geometry/run_compact_retinal_translation_geometry.py`:
  panel-table builder from existing TFTS and covariance-closure outputs.
- `compact_retinal_translation_geometry/run_compact_geometry_audits.py`:
  acceptance/audit suite for compact geometry outputs.
- `compact_retinal_translation_geometry/run_metric_structure_validation.py`:
  hidden-coordinate/local-metric validation from translated response grids.
- `compact_retinal_translation_geometry/run_relative_displacement_decoding.py`:
  recorded relative displacement decoding, currently active/uncommitted.
- `compact_retinal_translation_geometry_implementation_spec.md`: governing
  implementation spec and acceptance criteria.

The folder currently reuses:

- `outputs/twin_feature_tangent_structure_prod_v2`
- `outputs/matched_twin_covariance_closure_rf_null_step025_rfbacked_v2`

### 2026-06-08: Direct Recorded Derivative / Twin Tangent Alignment

Primary folder: `direct_recorded_derivative_twin_alignment/`.

Purpose: bounded supplemental analysis testing whether recorded eye-position
derivatives are enriched in compact fitted-twin translation geometry.

Important files:

- `direct_recorded_derivative_twin_alignment/README.md`: current claim gate,
  tiers, stop rules, and expected outputs.
- `direct_recorded_derivative_twin_alignment/run_direct_recorded_derivative_alignment.py`:
  main runner for recorded derivative reliability, compact-basis capture,
  matched derivative alignment, signed-axis diagnostics, and null summaries.
- `direct_recorded_derivative_twin_alignment/STG_REFERENCE_NOTES.md`:
  notes imported from older shared-transformation-geometry failure modes.
- `direct_recorded_derivative_twin_alignment_prescription.md`: original
  analysis prescription.

Promote only if Tier 1 survives global-rate plus target-PC1 projection and
RF/readout-preserving nulls across sessions.

### 2026-06-07 / 2026-06-08: Matched Twin Covariance Closure

Primary folder: `matched_twin_covariance_closure/`.

Purpose: compare recorded FEM covariance in Ryan's matched recorded/twin unit
space to fitted-twin eye-position and finite-difference tangent covariance.

Important files:

- `matched_twin_covariance_closure/README.md`: current provenance, basis
  definitions, controls, finite-difference results, and audit summaries.
- `matched_twin_covariance_closure/run_cache_closure.py`: first-pass cache-only
  closure using matched Ryan Fig2/Fig3 products.
- `matched_twin_covariance_closure/run_finite_difference_closure.py`: stricter
  model-based finite-difference closure.
- `matched_twin_covariance_closure/summarize_finite_difference_results.py`:
  audit, bootstrap, and sign-test summaries.
- `matched_twin_covariance_closure/rf_readout_preserving_null_prescription.md`:
  prescription for stronger RF/readout-aware nulls.

### 2026-06-07: Figure 4 Covariance / TFTS Figure Work

Primary folder: `fig4_cov_TFTS/`.

Purpose: figure-generation and panel-level analysis workspace for covariance
and twin-feature-tangent-structure results. This is now partly superseded by
`compact_retinal_translation_geometry/`, but still contains figure scripts and
handoffs.

Important files:

- `fig4_cov_TFTS/generate_covTFTS_figure.py`: current figure generator.
- `fig4_cov_TFTS/generate_covTFTS_figure_4.py`: later figure variant.
- `fig4_cov_TFTS/generate_covTFTS_figure_draft.py`: draft figure generator.
- `fig4_cov_TFTS/generate_spatial_content_modulation_figure.py`: figure for
  spatial-content modulation / natural structure controls.
- `fig4_cov_TFTS/run_panelF_covariance_overlap.py`: Panel F covariance overlap
  runner.
- `fig4_cov_TFTS/run_panelF_natural_structure.py`: Panel F natural-structure
  runner.
- `fig4_cov_TFTS/tests/test_tangent_subspace_information.py`: focused tests for
  tangent-subspace information machinery.
- `fig4_cov_TFTS/covTFTS_figure_panel_prescription.md`: panel prescription.
- `fig4_cov_TFTS/covTFTS_figure_data_forward_prescription.md`: data-forward
  prescription.
- `fig4_cov_TFTS/figure4_panelF_natural_structure_coda_plan.md`: Panel F coda
  plan.
- `fig4_cov_TFTS/panelE_tangent_subspace_information_handoff.md`: detailed
  handoff for tangent-subspace information.
- `fig4_cov_TFTS/update.md`: local update notes.
- `figure4_multipanel_plus_sup.md`: top-level Figure 4 multipanel and
  supplement handoff.
- `Fig4_reruns_plan.md`: top-level rerun plan.

### 2026-06-04: Natural Image Tangent Scale

Primary folder: `natural_image_tangent_scale/`.

Purpose: quantify where local translation/tangent approximations break down in
natural-image movies, including displacement scale, image structure, and FEM
amplitude comparisons.

Important files:

- `natural_image_tangent_scale/run_natural_image_tangent_scale.py`: main
  analysis runner.
- `Natural_Image_Tangent_Scale_Analysis_Handoff.md`: handoff.
- `natural-image-tangent-scale-analysis.md`: analysis note.

### 2026-06-04 / 2026-06-03: Twin Feature Tangent Structure

Primary folder: `twin_feature_tangent_structure/`.

Purpose: compact basis and split/generalization analyses for fitted-twin
translation tangents across natural images and conditions.

Important files:

- `twin_feature_tangent_structure/run_twin_feature_tangent_structure.py`: main
  runner for cached tangent union spectra, split modes, basis summaries, and
  audits.
- `twin_feature_tangent_structure_handoff.md`: handoff.
- `Twin_Feature_Tangent_Structure_Prescription.md`: prescription.

### 2026-06-03: Shared Transformation Geometry

Primary folder: `shared_transformation_geometry/`.

Purpose: early STG pipeline for recorded/twin tangent maps, support census,
residual RDM geometry, image-similarity controls, template matching, shared-mode
projection sweeps, and cross-session aggregation.

Important files:

- `shared_transformation_geometry/README.md`: main runbook.
- `shared_transformation_geometry/run_stg_support_census.py`: session support
  census.
- `shared_transformation_geometry/run_stg_tangent_stage1.py`: signed tangent-map
  analysis for recorded or twin sources.
- `shared_transformation_geometry/run_stg_shared_mode_projection_sweep.py`:
  recorded shared-mode projection sweep.
- `shared_transformation_geometry/run_stg_tangent_template_confirmation.py`:
  recorded-vs-twin template confirmation.
- `shared_transformation_geometry/run_stg_residual_rdm_stage2.py`: diagnostic
  residual RDM runner.
- `shared_transformation_geometry/run_stg_residual_rdm_stage23.py`: canonical
  residual geometry plus image-similarity controls.
- `shared_transformation_geometry/run_stg_direct_recorded_twin_tangent_match.py`:
  direct recorded/twin tangent matching.
- `shared_transformation_geometry/run_stg_retinotopy_tangent_identity.py`:
  retinotopy/tangent identity analysis and historical reference for later
  alignment work.
- `shared_transformation_geometry/run_stg_aggregate_stage5.py`: cross-session
  aggregation.
- `shared_transformation_geometry/utils.py`: shared helpers.
- `shared_transformation_geometry_handoff.md` and
  `shared_transformation_geometry_handoff_v2.md`: top-level handoffs.

### 2026-06-03: Twin Covariance Structure

Primary folder: `twin_covariance_structure/`.

Purpose: digital-twin reafferent covariance geometry: covariance estimation,
eigenspectra, participation ratio, subspace overlap, translation-tangent
alignment, image specificity, occupancy controls, and single-unit-to-population
bridges.

Important files:

- `twin_covariance_structure/README.md`: main runbook and output inventory.
- `twin_covariance_structure/run_twin_covariance_structure.py`: A1-A6 runner.
- `twin_covariance_structure/run_a2_audit.py`: A2 control-construction and
  trace-count audit.
- `twin_covariance_structure/run_a3_audit.py`: A3 image-specificity audit.
- `twin_covariance_structure/run_a3_fixrsvp_audit.py`: fixRSVP A3 audit using
  empirical image windows.
- `twin_covariance_structure/build_a3_high_support_summary.py`: summary helper
  for high-support A3 results.
- `twin_covariance_structure/generate_control_response_caches.py`: response
  cache generation for transformed eye controls.
- `twin_covariance_structure/covariance_core.py`: covariance primitives.
- `twin_covariance_structure/subspace_metrics.py`: overlap/capture/angle
  utilities.
- `twin_covariance_structure/eye_controls.py`: occupancy and eye-control
  transforms.
- `twin_covariance_structure/plotting.py`: figure builders.
- `Twin_Covariance_Structure_Prescription.md`: prescription.
- `twin_covariance_analysis_plan.md`: top-level plan.

### 2026-06-01: Keystone / Big-Picture FEM Geometry

Purpose: bridge and planning docs around the "keystone" geometry idea,
active-sensing efficiency, and fixRSVP neural trajectory readiness.

Important files:

- `Keystone_Geometry_Crossover_handoff_v2.md` and
  `Keystone_Geometry_Crossover_handoff_v3.md`: crossover handoffs.
- `bigpicture_fem_v1_high_impact_analysis_plan_v2.md`: high-impact analysis
  plan.
- `bigpicture_phase1_fem_v1_coding_agent_plan_v2.md`: coding-agent plan.
- `e1_active_sensing_efficiency_revised_handoff.md`: active-sensing efficiency
  handoff.
- `fixrsvp_neural_trajectory_analysis_plan_revised.md`: revised neural
  trajectory plan.
- `fixrsvp_trajectory_implementation_readiness.md`: readiness note.
- `mono_covariance_decomposition.py`: covariance-decomposition exploration.
- `mono_empirical_keystone_fullgrid.py`: empirical keystone full-grid
  exploration.

### 2026-05-29: Jacobian Audit / Predictive Framework

Purpose: audit and handoff layer around local Jacobian identity/translation
geometry and path-integrated separability.

Important files:

- `Empirical Transformation Geometry Analysis.md`: broad empirical geometry
  analysis note.
- `eoptotype_jacobian_field_smoothness_handoff.md`: e-optotype Jacobian field
  smoothness handoff.
- `fem_path_integrated_separability_handoff.md`: path-integrated separability
  handoff.
- `figure4_geometry_bridge_audit_plan_v2.md`: Figure 4 geometry bridge audit.
- `fixrsvp_cross_session_check.md`: cross-session check.
- `jacobian_predictive_framework_progress_summary.md`: progress summary.

### 2026-05-26: Jacobian Identity Geometry / Figures

Purpose: main local Jacobian analysis, translation mimicry, phase landscape,
matched-position ablation, and figure generation.

Important files:

- `jacobian_test3.py`: main local Jacobian/translation covariance analysis.
- `jacobian_test4.py`: follow-on Jacobian test script.
- `geometry_utils.py`: geometry helpers used by Jacobian figure work.
- `main_figure_jacobian.py`: main figure generator.
- `figure_jacobian_identity_geometry.py`: identity geometry figure generator.
- `translation_mimicry.py`: tests whether retinal translation can mimic
  identity changes.
- `phase_landscape.py`: static-offset phase landscape analysis.
- `matched_position_ablation.py`: matched-position ablation runner.
- `gru_passthrough_test.py`: GRU passthrough and temporal-integration
  diagnostic.
- `fem_global_intervention.py`: global FEM intervention analysis.
- `make_priority2_causal_alignment_figure.py`: render priority-2 causal
  alignment figure from intervention outputs.
- `FEM_population_coding_writeup.md`: population-coding writeup.
- `fem_eoptotype_hyperacuity_results.md`: e-optotype hyperacuity results.
- `fem_next_steps_plan.md`: next-steps plan.
- `jacobian_figure_handoff_nature_style.md`: figure style handoff.
- `jacobian_identity_geometry_results.md`: results note.
- `jacobian_identity_transformation_analysis_plan.md`: analysis plan.
- `jacobian_predictive_framework_handoff_revised.md`: revised handoff.

Result folders:

- `jacobian_results/`: `.npz`, text, and PDF outputs for Jacobian tests.
- `matched_position_results/`: matched-position ablation figure/data.
- `results/phase_landscape_*`: phase landscape smoke/coarse/fine outputs.
- `results/translation_mimicry_*`: translation mimicry smoke/primary outputs.
- `fem_global_intervention_results/`: global intervention figures/data.
- `gru_passthrough_figures/`: temporal-integration and feature-RSA figures.

### 2026-05-21: Early FEM / Temporal Decoding / COM Dynamics

Purpose: first large batch of FEM/e-optotype analysis scripts and planning
docs, including temporal decoding, center-of-mass dynamics, displacement
decoding, and covariance/intervention experiments.

Important files:

- `analysis_plan_jacobian_v3.md`: early Jacobian analysis plan.
- `revised_analysis_plan.md`: revised plan.
- `results_summary.md`: summary of early results.
- `FEMs_Eoptotype_checks.md`: e-optotype checks.
- `temporal_decoding_analysis_plan_consolidated_v2.md`: consolidated temporal
  decoding plan.
- `temporal_decoding_analysis_implementation_plan.md`: implementation plan.
- `temporal_decoding_diagnostic_plan.md`: diagnostic plan.
- `temporal_analysis_issues_and_alternatives.md`: issues and alternatives.
- `rowley_luke_2026_03_16_dpi_pupil_intercept_findings.md`: Rowley/Luke pupil
  intercept findings.
- `rowley_session_config_generation.md`: session config generation notes.
- `com_dynamics.py`: center-of-mass/spatial-moment dynamics and decoding.
- `com_dynamics_spec.md`: COM dynamics spec.
- `transformation_dynamics.py`: transformation-dynamics analysis.
- `transformation_dynamics_plan.md`: transformation-dynamics plan.
- `displacement_decoding.py`: displacement decoding analysis.
- `displacement_decoding_spec.md`: displacement decoding spec.
- `eoptotype_continuous_pass.py`: continuous-pass e-optotype analysis.
- `fem_covariance_geometry.py`: FEM covariance geometry analysis.
- `fem_differential_intervention.py`: differential intervention analysis.
- `fem_global_intervention.py`: global intervention analysis.
- `translation_covariance.py`: translation-covariance analysis.
- `translation_covariance_figures.py`: translation-covariance figure builder.
- `translation_covariance_trajectories_figures.py`: trajectory figures.
- `translation_covariance_trajectories_figures0.py`: earlier trajectory figure
  variant.
- `translation_covariance_trajectories_figures_interactive.py.py`: interactive
  trajectory figure variant; note the duplicate `.py.py` suffix.
- `overnight_backimage_long_sweeps_20s_re/note`: note for rerun sweep folder.

Result/cache folders:

- `continuous_pass_results/`: continuous-pass `.npz` and `.png` outputs.
- `displacement_decoding_cache/`: cached displacement grids/rates.
- `displacement_decoding_figures/`: displacement decoding PDFs and summary.
- `fem_covariance_geometry_results/`: real/stabilized FEM covariance geometry
  outputs.
- `fem_differential_intervention_results/`: differential intervention output.
- `transformation_dynamics_cache/`: cached moment and delta-rate tensors.
- `transformation_dynamics_figures/`: transformation dynamics figures.

### 2026-01 to 2026-04: Early Backimage / Translation-Covariance Artifacts

Purpose: early eye-trace, backimage, translation-covariance, and natural-image
sweep artifacts. Many of these are bulky generated assets rather than active
source files.

Important files/folders:

- `translation_covariance/`: January translation-covariance `.npz`, `.pkl`,
  capture matrices, shuffle products, and older figures/trajectory figures.
- `overnight_backimage_sweeps/`: first overnight backimage sweep `.pkl` files.
- `overnight_backimage_long_sweeps_20s/`: 20s backimage sweep artifacts and
  `RUN_SUMMARY.json`.
- `overnight_backimage_long_sweeps_20s_re/`: rerun subset and note.
- `test_sweeps/`: generated sweep videos and saccade histograms.
- `E_diagnostics_human_240ppd/`: human-resolution E diagnostic outputs by size.
- `E_diagnostics_model_37ppd/`: model diagnostic run marker/output.
- `E_diagnostics_model_37ppd_resnet_none_convgru/`: model diagnostic outputs by
  size; large generated artifact folder.
- `backimage_fixation_results.pkl` and `.meta.json`: early backimage fixation
  cache.
- `backimage_image_cache.pkl`: image cache.
- `fixrsvp_fixation_pool.pkl` and `.meta.json`: fixRSVP fixation pool.
- `hybrid_eye_trace_*.pkl`: early hybrid eye trace sweep caches.
- `spatial_info_fixrsvp_eye_scales_frames_per_im.pkl`: large spatial-info
  cache.
- `SpikeSortingTools_repo_collation.txt`: text collation of SpikeSortingTools
  repo material.

## Top-Level Folder Index

Use this section when you know the folder but not the date.

| Path | Contents |
| --- | --- |
| `active_sensing_movie_information/` | Figure 5 / active-sensing movie-information plans, figure generator, QC, and population checks. |
| `compact_retinal_translation_geometry/` | Spec-facing compact translation-geometry panel builder, audits, metric validation, and displacement decoding. |
| `direct_recorded_derivative_twin_alignment/` | Supplemental recorded derivative vs twin tangent alignment runner and notes. |
| `matched_twin_covariance_closure/` | Cache and finite-difference closure of recorded FEM covariance by fitted-twin tangents. |
| `fig4_cov_TFTS/` | Figure 4 covariance/TFTS figure scripts, panel analyses, handoffs, and tests. |
| `natural_image_tangent_scale/` | Natural-image tangent scale/breakdown analysis. |
| `twin_feature_tangent_structure/` | Fitted-twin tangent basis compactness and split/generalization analyses. |
| `shared_transformation_geometry/` | Early recorded/twin shared transformation geometry pipeline. |
| `twin_covariance_structure/` | Reafferent covariance geometry analysis and audit scripts. |
| `results/` | Generated phase-landscape, translation-mimicry, and Jacobian identity outputs. |
| `jacobian_results/` | Generated Jacobian analysis outputs and interpretation note. |
| `translation_covariance/` | Early translation-covariance generated products and figures. |
| `continuous_pass_results/` | Continuous-pass e-optotype generated outputs. |
| `*_results/`, `*_figures/`, `*_cache/` | Generated artifacts for the corresponding top-level script. |
| `overnight_backimage_*`, `test_sweeps/`, `E_diagnostics_*` | Early generated sweep/diagnostic artifacts. |
| `__pycache__/` and nested `__pycache__/` | Python bytecode caches; not source. |

## Thread Map

These threads overlap scientifically but are useful for navigation.

| Thread | Current home | Historical/related files |
| --- | --- | --- |
| Active-sensing natural-image information | `active_sensing_movie_information/` | `Figure5_active_sensing_triage_plan.md`, `Figure5_random_amp_cloud_matched_control_spec.md`, `e1_active_sensing_efficiency_revised_handoff.md` |
| Compact retinal translation geometry | `compact_retinal_translation_geometry/` | `fig4_cov_TFTS/`, `twin_feature_tangent_structure/`, `matched_twin_covariance_closure/` |
| Recorded derivative/twin tangent alignment | `direct_recorded_derivative_twin_alignment/` | `shared_transformation_geometry/`, `matched_twin_covariance_closure/` |
| Covariance closure | `matched_twin_covariance_closure/` | Ryan cache copies under `outputs/cache/` and RF-backed closure outputs under `outputs/` |
| Shared transformation geometry | `shared_transformation_geometry/` | `shared_transformation_geometry_handoff*.md` |
| Twin covariance structure | `twin_covariance_structure/` | `Twin_Covariance_Structure_Prescription.md`, `twin_covariance_analysis_plan.md` |
| Twin feature tangent structure | `twin_feature_tangent_structure/` | `Twin_Feature_Tangent_Structure_Prescription.md`, `twin_feature_tangent_structure_handoff.md` |
| Jacobian identity / translation mimicry | top-level Jacobian scripts and `results/` | `jacobian_*`, `translation_mimicry.py`, `phase_landscape.py`, `geometry_utils.py` |
| Early temporal/COM/displacement work | top-level scripts and cache folders | `temporal_*`, `com_dynamics*`, `displacement_decoding*`, `transformation_dynamics*` |
| Early backimage sweeps | `overnight_backimage_*`, `test_sweeps/` | `backimage_*`, `hybrid_eye_trace_*`, `spatial_info_*` caches |

## Cleanup Notes

Obvious cleanup candidates, if/when this workspace is later reorganized:

- Generated artifacts under `declan/` could move to `outputs/declan/` or a
  scratch/artifact tree.
- `__pycache__/` folders can be ignored or removed.
- `translation_covariance_trajectories_figures_interactive.py.py` has a likely
  accidental duplicate suffix, but preserve it until references are checked.
- Top-level scripts from 2026-05-21 and 2026-05-26 are intertwined with result
  folders; move only after adding import/CLI regression checks.
- Do not delete historical e-optotype Figure 5 scaffolds until the natural-image
  pipeline is fully settled; mark them historical in docs instead.
