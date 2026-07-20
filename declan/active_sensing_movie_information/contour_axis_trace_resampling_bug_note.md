# Contour-Axis Trace Resampling Bug Note

Date documented: 2026-07-16

## Status

Previous BackImage contour-axis RR100 spatial-SSI analyses that used selected
BackImage eye-trace windows should be treated as interpretation-limited until
they are explicitly audited or rerun with the corrected trace construction.

Do not treat old contour-axis SSI outputs as evidence for a precise natural
motion scale, aligned-vs-orthogonal effect size, or low-SF/high-SF scale
comparison unless the run metadata and trace inventory show that the source
trace was not temporally compressed.

## Bug

The affected runner is:

```text
declan/active_sensing_movie_information/run_backimage_contour_axis_rr100_spatial_ssi.py
```

The affected path reconstructed selected-window traces by calling the aggregate
trace-bank builder, which resampled the full selected eye-position window to
`--n-timepoints`.

For the prior aligned-contour runs, the selected windows were 128 samples, while
the SSI movie length was `n_timepoints=40`. This compressed a full 128-sample
recorded trace into 40 model frames. That preserves a rough path shape, but it
does not preserve the native temporal scale or the intended scale-1 trace
semantics.

The bug is especially important for diffusion-style motion metrics, because
diffusion constant is an MSD slope per unit time. Compressing a longer trace into
fewer model frames can inflate the apparent diffusion constant even when the
displayed trace is nominally "scale=1".

## Confirmed Affected Outputs

The following output families were confirmed to use 128-sample selected windows
with `n_timepoints=40` and the old
`reconstructed_trace_bank_from_selected_windows` source-trace contract:

```text
outputs/active_sensing_movie_information/
  backimage_contour_axis_rr100_sf_contour_alignment_long_axis30_n576_low0p05_high0p5_v1/
    contour_rr100_spatial_ssi_pairs27/

outputs/active_sensing_movie_information/
  backimage_contour_axis_rr100_sf_contour_alignment_long_axis30_n576_low0p05_high0p5_rot90_v1/
    contour_rr100_spatial_ssi_pairs27/

outputs/active_sensing_movie_information/
  backimage_contour_axis_rr100_spatial_ssi_n128_across_sweep_v1/
```

Downstream summaries, figures, and diagnostics that consume those caches inherit
the issue. This includes unit-first contour-matched plots, aligned-vs-orthogonal
population summaries, orientation-stratified summaries, rotation crossover
diagnostics, and any manuscript-style interpretation built from those outputs.

Other old outputs may also be affected if their `run_metadata.json` shows:

```text
source_trace_contract = reconstructed_trace_bank_from_selected_windows
n_timepoints < global_stop - global_start for selected_windows.csv
```

or if their inventory source-trace metrics match the old compressed
reconstruction rather than a native snippet.

## Quantitative Check

For the long aligned-contour source windows:

```text
selected window length: 128 samples
SSI n_timepoints: 40
median inventory/source path-length ratio: about 0.17
median compressed/full diffusion ratio: about 6.23
median compressed/native-40-crop diffusion ratio: about 6.36
```

A direct reconstruction check showed that the old
`movie_condition_inventory.csv` source-trace path length and RMS matched
`_build_trace_bank(..., 40)` exactly. This confirms that the old contour-axis
SSI cache used compressed traces, not native 40-frame snippets.

## Corrected Semantics

The corrected runner now uses native center-cropped snippets for selected-window
traces and trace-bank traces. The intended scale-1 behavior is:

```text
source trace = native n_timepoints snippet
rendered trace = same native n_timepoints snippet before condition scaling
source/rendered diffusion delta = 0 for unscaled snippets
```

The corrected source-trace contract is:

```text
center_cropped_native_selected_window_trace_n_timepoints
```

The cache schema was bumped to `3` so corrected runs do not silently reuse old
cache artifacts.

Corrected trace-bank outputs should also include the predeclared metric summary:

```text
trace_bank_metadata.csv
trace_bank_metric_summary.csv
trace_bank_metric_summary_panel.{png,pdf}
```

Do not rely on diffusion constant alone for trace-scale interpretation. The
standard bundle should include path length/path speed, RMS radius or BCEA68,
MSD diffusion constant, covariance anisotropy/axis ratio/orientation, lag-1
autocorrelation, and microsaccade contamination metrics. Microsaccade event
count, event-sample fraction, threshold, and peak speed are mandatory audit
columns because microsaccade-containing snippets occupy a distinct high-scale,
high-anisotropy regime.

## Similar Code Paths

A follow-up code audit found that the shared aggregate BackImage trace-bank
helper also had the risky pattern: it built the in-memory rendered trace by
resampling the full source window to `n_timepoints`:

```text
declan/fixation_statistics_by_stimulus/run_backimage_aggregate_fem_information.py
  _build_trace_bank(...)
```

That helper has now been changed so its default rendered-trace policy is:

```text
center_crop_native
```

Under the corrected default, empirical source windows with at least
`n_timepoints` samples produce a native center-cropped snippet. The legacy
full-window compression behavior is still available only when explicitly
requested:

```text
--trace-window-policy resample_full_window
```

New outputs generated through this helper should expose the trace contract in
their tables using fields such as `trace_render_contract`,
`trace_window_policy_requested`, `trace_window_policy`,
`source_window_n_samples`, `rendered_trace_n_samples`, and
`source_to_render_time_compression`.

Known callers of this shared helper include:

```text
declan/fixation_statistics_by_stimulus/run_backimage_aggregate_fem_information.py
declan/fixation_statistics_by_stimulus/run_backimage_trajectory_table_observer.py
declan/fixation_statistics_by_stimulus/run_backimage_local_pairing_Iz_revisit.py
declan/fixation_statistics_by_stimulus/run_backimage_aggregate_trace_catalog.py
declan/fixation_statistics_by_stimulus/audit_backimage_ou_trace_controls.py
declan/backimage_trajectory_observer/plot_global_fixation_fourier_component_flow.py
```

This code change does not repair old outputs. Any existing run from the callers
above that lacks the new contract columns, or that reports
`trace_render_contract=resampled_full_source_window_to_n_timepoints`, should be
treated as a rendered/compressed trace analysis unless rerun. Scale, diffusion,
speed, temporal-frequency, and microsaccade-contamination claims from those
outputs must be checked against the rendered-trace contract before being treated
as native FEM claims.

The corrected contour-axis trace-bank path and the random patch x trace SSI
matrix runner use:

```text
center_cropped_native_n_timepoints
```

Those paths crop a native snippet before scoring rather than compressing the
entire source window. The native snippet builder has also been hardened so it
does not call the temporal resampling helper after the crop.

## Required Handling

Before trusting any previous contour-axis SSI analysis:

1. Inspect `run_metadata.json` for `n_timepoints`, `trace_source_contracts`, and
   `source_meta.source_trace_contract`.
2. Inspect the corresponding `selected_windows.csv` window lengths.
3. Check whether `movie_condition_inventory.csv` source-trace path length and
   RMS match the native snippet or the old compressed reconstruction.
4. Rerun affected caches with the corrected runner when the result depends on
   motion scale, diffusion scale, aligned-vs-orthogonal magnitude, or SF-group
   scale curves.

Safe language until rerun:

> Prior contour-axis RR100 spatial-SSI results are useful as exploratory
> analyses of a temporally compressed eye-trace manipulation, but they should not
> be interpreted as calibrated scale-1 real-FEM results.
