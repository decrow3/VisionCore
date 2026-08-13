# SSI map-first debugging precedent

## Purpose

Use this reference to reconstruct the successful human-AI analysis sequence from the BackImage spatial-SSI work. Treat it as a methodological precedent, not as evidence that a new mechanism is true.

## The sequence that worked

1. Start with the thing itself: render activation maps for a concrete image, path, unit, condition, and frame.
2. Compare multiple maps slowly: use a compact panel with the image patch, motion trace, selected units, and condition sweeps side by side.
3. Make example selection auditable: calculate unit-level criteria and save role-based selections rather than relying on memorable examples.
4. Drill into the interesting unit: render all-frame sheets with instantaneous SSI attached to each tile, followed by map-derived timecourses.
5. Summarize only after the maps are understood: then inspect condition curves, group summaries, normalization diagnostics, and population plots.

The transferable rule is concrete maps first, auditable selection second, detailed unit inspection third, and aggregate summaries last.

## Canonical implementation and artifacts

Primary plotting implementation:

```text
declan/active_sensing_movie_information/plot_backimage_rr100_instantaneous_unit_maps.py
```

Narrative and metric-contract notes:

```text
declan/active_sensing_movie_information/backimage_spatial_ssi_revival_note.md
```

Representative output directory:

```text
outputs/active_sensing_movie_information/backimage_rr100_instantaneous_unit_maps_latest_v1/
```

Important artifacts include:

```text
backimage_rr100_instantaneous_unit_maps_compact.png
backimage_rr100_instantaneous_unit_maps_all_timepoints.pdf
selected_units.csv
frame_ssi_timecourses.csv
backimage_rr100_u017_instantaneous_unit_maps_frame_ssi_annotated.pdf
backimage_rr100_u017_frame_ssi_timecourse_rows_colored.png
backimage_rr100_orientation_group_zscored_ssi_curves.png
population_ssi_summary/
```

The compact figure deliberately uses one instantaneous frame rather than a trajectory-averaged map. It combines a concrete patch and trace with per-unit condition sweeps so the human can see the response object before reading a summary statistic.

## Auditable example selection

The plotting script constructs unit-level measurements, merges orientation-probe summaries, and writes `selected_units.csv`. In the representative run, the roles were:

```text
orientation_tuning_aligned_with_contour
orientation_tuning_orthogonal_to_contour
off_axis_orientation_control
```

The saved table also records the unit, reference condition and frame, activation-map axis and anisotropy, mean rate, instantaneous map SSI, preferred orientation, orientation selectivity, probe responses, and the criterion-derived role.

This mattered because it separated three distinct acts:

- surveying many units;
- applying explicit criteria;
- choosing which small set deserved expensive or visually dense follow-up.

For a new mechanism, redefine the roles to match the question. For example, a power-redistribution analysis could use largest map change, largest SSI change, high predicted shift with low observed gain, low predicted shift with high observed gain, low-SF control, and high-SF control.

## Human checkpoint behavior

At each stage, present the figure rather than just announcing its existence. Ask the human to react to visible structure before expanding the analysis.

Useful checkpoint questions include:

- Are these the right concrete contrasts?
- Which map changes appear meaningful rather than color-scale artifacts?
- Which units deserve all-frame inspection?
- Does the instantaneous metric describe what the map visibly does?
- Which dissociation would be most informative next?
- Is the proposed population summary still answering the question exposed by the maps?

Do not batch all stages into a single polished analysis unless the user asks for an autonomous run. The successful workflow depended on observations at one stage changing the next stage.

## Scientific and provenance guardrails

### Keep instantaneous and mean-map SSI distinct

SSI is nonlinear. Averaging SSI calculated from instantaneous maps is not equivalent to calculating SSI from a trajectory-averaged map. The promoted BackImage contour-axis contract uses instantaneous spatial maps over time, weighted by expected spikes. Use mean-map SSI only as a separately labeled diagnostic.

### Keep rate and spike quantities beside bits per spike

Activation suppression can be hidden by a normalized population endpoint. Preserve mean rates, raw spatial-information quantities, expected spikes, bits per second where available, and paired bits-per-spike differences.

### Prefer differences over unstable ratios

Near-zero static or reference SSI can create extreme ratios. Prefer condition-minus-baseline differences and absolute quantities unless the ratio denominator is demonstrably stable.

### Treat the baseline as part of the claim

In the prior BackImage work, `stabilized` is trial-mean stabilization, not a deterministic static-center oracle. Do not silently change that construction or equate the two baselines.

### Check trace provenance before reusing old results

The revival note documents a trace-construction bug in older selected-window contour-axis runs: full 128-sample windows were compressed into 40 model timepoints. Do not treat affected outputs as calibrated scale-1 real-FEM results. Inspect metadata for the affected `reconstructed_trace_bank_from_selected_windows` provenance and prefer the corrected `center_cropped_native_selected_window_trace_n_timepoints` contract.

### Label targeted renders honestly

The instantaneous-map helper materializes a small targeted cache because the production summary caches do not store every condition-by-time-by-unit spatial map. Describe such output as a targeted visualization render, not as a broad production rerun.

## Adaptation to a power-shift question

Use a staged first pass:

1. Restrict the contrast to normal motion versus counterfactual stabilization.
2. Plot the image patch, path, speed, coarse spatial-frequency power, and the derived temporal-frequency landing separately.
3. Plot matched activation maps for several units across frames and conditions.
4. Plot normal-minus-stabilized difference maps.
5. Save role-based unit selections, including dissociations between predicted shift and observed map or SSI change.
6. Inspect full time-resolved panels for the selected units.
7. Only then test whether the proxy explains map change and SSI change across units or movies.

If the proxy fails, preserve that result at the map level. The purpose of the workflow is to expose failure before it is obscured by regression output.
