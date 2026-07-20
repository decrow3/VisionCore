# BackImage Spatial-SSI Revival Note

Date: 2026-07-07

## 2026-07-16 Interpretation-Limiting Bug Notice

Do not treat the old BackImage contour-axis RR100 spatial-SSI outputs in this
note as calibrated scale-1 real-FEM results until they are audited or rerun.

A trace-construction bug was confirmed in the selected-window path of:

```text
declan/active_sensing_movie_information/run_backimage_contour_axis_rr100_spatial_ssi.py
```

Previous selected-window contour-axis runs compressed full 128-sample BackImage
eye-trace windows into `n_timepoints=40` model traces. This affects motion-scale
and diffusion-scale interpretation, and it propagates to downstream
aligned-vs-orthogonal, orientation-stratified, rotation-crossover, and
unit-first contour-matched summaries built from those caches.

Before trusting any old contour-axis SSI result, check whether its metadata uses
`reconstructed_trace_bank_from_selected_windows` and whether the selected source
windows are longer than the SSI `n_timepoints`. Affected caches should be rerun
with the corrected native-snippet source-trace contract:

```text
center_cropped_native_selected_window_trace_n_timepoints
```

Detailed incident note:

```text
declan/active_sensing_movie_information/contour_axis_trace_resampling_bug_note.md
```

## Why This Note Exists

The earlier BackImage movie-information work already has a usable spatial-SSI
cache. The safest revival path is to treat that cache as the starting point,
then add Vernier-style contract checks before making new claims or rerunning
large jobs.

This is separate from the BackImage trajectory-table observer. The trajectory
observer asks whether exact nuisance marginalization over a finite trajectory
catalog rescues image identity. The movie-information branch asks whether
retinal motion increases a spatial-information efficiency proxy in V1-model
rate movies.

## Canonical Prior Implementation

Use `jake/twininfo`, not the temporary exploratory runner in this folder, as
the canonical implementation.

Primary source run:

```text
outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu/
```

Important files:

```text
metadata/05_lagcube_information_summary.csv
cache/cumulative_information_series.npz
metadata/03_trajectory_control_qc.csv
metadata/01_trace_examples_used.csv
```

Figure/summary layer:

```text
outputs/active_sensing_movie_information/active_sensing_movie_information_figure/
```

Useful existing scripts:

```text
declan/active_sensing_movie_information/generate_active_sensing_movie_information_figure.py
declan/active_sensing_movie_information/generate_retinal_movie_transform_qc.py
declan/active_sensing_movie_information/plot_backimage_scaled_real_unit_ssi.py
declan/active_sensing_movie_information/summarize_figure5_additional_checks.py
declan/active_sensing_movie_information/summarize_sf_localization.py
jake/twininfo/pipeline.py
jake/twininfo/lagcube_information.py
jake/twininfo/information.py
```

## Cache Inventory From The Existing Run

Current source-cache inventory:

```text
summary rows: 1728
paired movie keys: 108
images: 27
trace windows: 4 total = 2 fixation + 2 microsaccade
conditions: 16
time samples: 128
spatial readout: 16 units per output-grid pixel
spatial bins: 51 x 51 = 2601
```

The summary table declares `spatial_ssi_uses_shift_grid=False`, which is the
right contract for this branch: spatial positions in the convolutional rate map
define the SSI ensemble, not artificial shift-grid states.

One provenance caveat: `metadata/run_config.json` reflects the latest
`--augment-existing` invocation, so its `conditions` field is not the full
completed condition set. For cache audits, use the summary/cache records as the
source of truth for completed conditions, or write a merged manifest before a
new promoted rerun.

## Metric Contract

The primary metric is:

```text
final_cumulative_spatial_ssi_bits_per_spike
```

It is computed from `cumulative_spatial_ssi(center_rate_map)`, where
`center_rate_map` is the unshifted full convolutional response map with shape
approximately:

```text
time x selected_units_per_pixel x output_row x output_col
```

The underlying unit-level SSI normalization is intentional:

```text
E_x[(r_u(x) / mean_x r_u(x)) log2(r_u(x) / mean_x r_u(x))]
```

The primary endpoint is the cumulative bits divided by cumulative expected
spikes. This is a prefix-normalized ratio, so it should not be treated as a
monotone additive curve. Always keep these companion diagnostics with it:

```text
final_cumulative_spatial_ssi_bits
final_cumulative_spatial_ssi_bits_per_second
final_cumulative_expected_spikes
```

The model-input path uses the raw-pixel normalization contract:

```text
stimulus_normalization = pixelnorm_raw_u8_minus_127_div_255
```

The code path is `lag_cubes_to_stim(cubes) = (raw_u8 - 127.0) / 255.0`.

## Baseline Contract

BackImage `stabilized` is a trial-mean baseline, not the Vernier-style
deterministic static-center oracle:

```text
stabilized = repeat(mean(trace[:t_max]), t_max)
```

Random trajectory controls are also centered on the measured fixation mean.
This preserves the local image neighborhood while perturbing motion statistics.

Therefore the recent Vernier caveat transfers as a guardrail, not as a known
bug: do not introduce or compare against a single `static_center` BackImage
condition unless it is explicitly labeled as a deterministic oracle.

Historical alias:

```text
phase_order_shuffle == trajectory_order_shuffle
```

It is a trajectory-order control, not the visual phase-scramble condition. The
visual phase-scramble condition is `pyramid_phase_scrambled`.

## Prior Result Summary

Primary real-vs-stabilized endpoint, paired across 108 image/trace/crop movies:

```text
real - stabilized = +0.0351548 bits/spike
fraction positive = 0.981
```

Time-resolved real-minus-stabilized SSI stayed positive on average throughout
the 128-frame window. The full-window epoch summary reported:

```text
mean delta = +0.0383340 bits/spike
fraction movies positive = 0.954
```

Matched-motion controls bounded trajectory-specific claims:

```text
real - random_amp                 = -0.0184445 bits/spike
real - random_amp_cloud_matched   = -0.0166514 bits/spike
real - random_cov                 = -0.0163895 bits/spike
```

So the prior safe claim is:

```text
retinal image motion increases this deterministic V1-model spatial-information
efficiency proxy relative to trial-mean stabilization
```

not:

```text
measured FEM trajectories are uniquely optimal
```

Spatial-frequency controls were directionally useful:

```text
sf_low      - stabilized_sf_low      = +0.0099677 bits/spike
sf_mid_low  - stabilized_sf_mid_low  = +0.0364357 bits/spike
sf_mid_high - stabilized_sf_mid_high = +0.0533056 bits/spike
sf_high     - stabilized_sf_high     = +0.0506534 bits/spike
```

The phase-scrambled visual control also showed a positive FEM-minus-stabilized
gain:

```text
pyramid_phase_scrambled - stabilized_pyramid_phase_scrambled
  = +0.0327953 bits/spike
```

That supports a spectral/fine-structure interpretation more than a
natural-phase-specific one.

Spike-count audit from the figure summary:

```text
raw spatial information: +88.8 percent
expected spikes:         +45.6 percent
```

The per-spike endpoint survives spike-count normalization, but raw bits and
expected spikes must remain in the result table.

## Vernier Lessons To Carry Over

1. Separate metric variants in names and metadata.
   Spatial-map SSI, shift-grid SSI, Fisher trace, and image-identity decoding
   should not share one ambiguous "information" label.

2. Prefer absolute or paired-delta quantities over fold-change.
   The Vernier denominator problem applies here too if we start plotting
   ratios. Use `condition - baseline` bits/spike, raw bits, and expected spikes.

3. Treat baseline construction as part of the scientific claim.
   `stabilized` is trial-mean-stabilized and is suitable as the main baseline.
   A single canonical static phase would be an oracle/sanity check only.

4. Record windowing and sample alignment.
   This branch uses all overlapping current-frame samples via
   `block_endpoint_lag_cubes`, with `analysis_sample_index` saved in the NPZ.
   If a future endpoint-only or terminal-window variant is added, give it a new
   name and cache schema.

5. Add identity checks before combining caches.
   The current output should be audited for summary-vs-series final equality,
   completed condition inventory, duplicate keys, normalization contract, and
   condition aliasing before promotion.

6. Do hierarchical statistics.
   The existing figure layer uses image-then-trace bootstrap. Keep that; the
   apparent `n=108` movie count is not the same as 108 independent images.

## Contour-Axis SSI Run Contract

The next BackImage SSI revival should reuse the existing contour-axis machinery
rather than rebuild that layer from scratch. Relevant local sources include:

```text
declan/fixation_statistics_by_stimulus/plot_backimage_contour_motion_components.py
declan/axis_conditioned_backimage_trajectory_observer/axis_conditioned_traces.py
declan/fixation_statistics_by_stimulus/run_backimage_trajectory_table_observer.py
declan/backimage_trajectory_observer/
```

The intended stimulus family is contour-relative: find the salient local
contour/edge axis inside fixation windows, decompose the source trajectory into
along-contour and across-contour components, then synthesize movies by scaling
those components. Existing stashed response tables/caches for these traces
should be reused when their identity and luminance contracts match the SSI
analysis; otherwise regenerate rather than silently mixing contracts.

Carry over these Vernier-derived guardrails:

1. Keep the stimulus luminance normalization identical to the model-training
   and production BackImage SSI contract. Do not use a display/QC movie scale as
   the model input scale.
2. Track activation-map suppression at the unit level. Absolute population SSI
   can hide units whose activity is suppressed by a motion condition.
3. Avoid unit ratios against static SSI as primary evidence. Near-zero or bad
   static SSI denominators can inflate fold-change views; use absolute
   bits/spike, paired deltas from baseline, raw bits, and expected spikes.
4. Build the static/trial-mean baseline with position nuisance. Static SSI
   should include multiple mean trajectory locations when possible, because
   slight mean-position changes can materially affect responses.
5. Use at least `n=128` trajectories per condition for promoted contour-axis
   claims. Smaller runs are plotting/contract smokes only.
6. Keep the first SSI readout simple: average/cumulative information over the
   trajectory samples. Do not switch to the endpoint/terminal-trajectory readout
   for this revival pass.
7. Run long jobs outside the sandbox on GPU with detached `nohup`/`setsid`
   commands, progress printed to log files, and only compact status summaries in
   chat.

Primary hypothesis test for this branch:

```text
Hold along-contour motion at 1x, sweep across-contour motion scale, and ask
whether BackImage spatial SSI depends on across scale. The working hypothesis is
that SSI peaks below 1x across-contour motion rather than at the measured 1x
scale.
```

## Contour-Axis SSI Metric Design Choice

The contour-axis RR100 BackImage analysis has two legitimate spatial-SSI
contracts. They answer different questions because SSI is nonlinear, so the
order of temporal averaging and spatial-information calculation matters.

Legacy time-resolved spatial SSI:

```text
For each frame t and unit u:

  I_u(t) = mean_x[(r_u(t,x) / mean_x r_u(t,x))
                  log2(r_u(t,x) / mean_x r_u(t,x))]

Then average over the trajectory with rate/expected-spike weights:

  I_u,time = sum_t I_u(t) mean_x r_u(t,x) dt
             / sum_t mean_x r_u(t,x) dt
```

This is the contract used by the completed
`backimage_contour_axis_rr100_spatial_ssi_n128_across_sweep_v1` run. It is a
valid "how sharp was the map at each instant, weighted by response" metric.
As of 2026-07-08, this is again the default BackImage contour-axis SSI
contract unless a run or figure explicitly says otherwise. The activation-map
diagnostic for this contract should therefore show instantaneous maps across
time, not trajectory-averaged maps.

Mean-map spatial SSI:

```text
First average the activation map over the trajectory:

  R_u(x) = mean_t r_u(t,x)

Then compute spatial SSI on that displayed/trajectory-averaged map:

  I_u,mean-map = mean_x[(R_u(x) / mean_x R_u(x))
                        log2(R_u(x) / mean_x R_u(x))]
```

This remains a useful diagnostic for trajectory-averaged activation sheets, but
it is not the default promoted BackImage SSI metric. The mean-map detour showed
that trajectory averaging can change the interpretation, but those averaged
maps are not the right visual object for the current question unless a figure
is explicitly asking about temporally pooled spatial footprints.

The default rerun and plotting contract is therefore:

```text
primary_ssi_metric = time_resolved
primary table columns = averaged instantaneous-map spatial SSI
diagnostic cached columns = time-resolved spatial SSI and mean-map spatial SSI when available
population time-resolved SSI = expected-spike-weighted average over units and frames
activation-map figures = instantaneous maps over time for selected image/unit/condition cases
```

The selected-image ramper audit made the distinction concrete: for unit `u067`,
selected images increased under the time-resolved metric from about `0.162`
bits/spike at across=1x to about `0.231` at across=3x, while the SSI computed
from the displayed mean maps decreased from about `0.087` to about `0.069`.
That is not a cache mismatch; it is the metric contract difference. The current
default is to interpret the former, while using instantaneous activation-map
sheets to understand why individual units rise or fall.

Current instantaneous-map figure helper:

```text
declan/active_sensing_movie_information/plot_backimage_rr100_instantaneous_unit_maps.py
```

It intentionally materializes a small targeted cache because the n=128 SSI
summary caches do not store every `condition x time x unit x H x W` map. The
output should be labeled as a targeted visualization render, not a broad
production SSI rerun.

## Contour-Axis Execution Plan And Cache Reuse

Use RR100 as the primary population for the new contour-axis SSI run wherever
the data contract allows it. The current movie-medoid population view is:

```text
V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_medoidPosthocminRepcomplete0p45_movieMedoid
```

Load it with:

```text
declan.redundancy_resolved_v1_population.load_population_view
declan.redundancy_resolved_v1_population.apply_population_view
```

The saved view has `input_channels=756`, `n_units=100`, membership shape
`(100, 756)`, and `pooling_mode=medoid`. Spatial maps are per unit, so a
canonical full-756 spatial-map cache is completely reusable for RR100 SSI. The
model-forward boundary should therefore be:

```text
full_rate_map: T x 756 x H x W
rr100_rate_map = apply_population_view(full_rate_map, rr100_view)
spatial_ssi(rr100_rate_map)
```

If the relevant full-756 spatial maps already exist for the exact
image/window/trace/condition/luminance contract, do not rerun the model just to
make RR100 SSI. Recompute SSI and activation-map summaries from that cache. If
only lower-dimensional responses exist, first locate or generate the missing
full spatial-map cache.

The existing 756-unit BackImage trajectory-table response caches are still
useful. They store response tensors such as `prior_lambda_counts` with shape
like:

```text
candidate x trajectory x time x 756
```

When `response_cache_manifest.csv` reports `n_units=756` and the unit-order
identity check passes, those cached responses can be converted to RR100 without
rerunning the model:

```text
rr100_counts = einsum("...tc,rc->...tr", full756_counts, rr100_view.membership)
```

The existing helper `_apply_response_population` in
`declan/figure4_active_sensing_atlas/scripts/build_panel_c_continuous_feature_embedding_reconstruction.py`
implements this exact count-space reduction. Use these converted RR100 response
tables for cached trajectory-table/center-response sanity checks, unit-space
comparisons, and fast readout diagnostics.

Do not treat the cached 756 response tables as spatial-SSI caches. They have no
`H x W` spatial map axis, so they cannot support spatial SSI, activation-map
suppression diagnostics, or unit activation-map panels by themselves. Those
outputs require a full-756 spatial-map cache, either pre-existing or newly
generated.

Concrete run sequence:

1. Audit and freeze inputs.
   Use the clean n=128 axis-conditioned selected-window/catalog outputs as the
   first source of window identities and trace/source matching. Verify
   `axis_shared_source_catalog=True`, shared sampled sources across axis
   families, `n_units=756` for cached response tables, full-spatial-map cache
   availability by condition where possible, RR100 input-channel compatibility,
   and the exact population-version string in every manifest.

2. Cached-response RR100 smoke.
   Convert a subset of existing 756 response tables to RR100 and verify that
   cached RR100 center-response summaries reproduce the expected axis-family
   ordering and scales. This is a cheap check of unit order and population-view
   plumbing; it is not the promoted SSI result.

3. Spatial-SSI smoke.
   First try to materialize the tiny RR100 SSI smoke from any existing full-756
   spatial maps. If the exact maps do not exist, run a tiny full-756 spatial-map
   sweep on a few BackImage windows, with `along_scale=1` and a small
   across-scale list such as `0,0.5,1`. Confirm the luminance contract, trace
   decomposition, static-position baseline, RR100 reduction, absolute SSI, raw
   bits, expected spikes, and unit suppression outputs.

4. Production contour-axis SSI.
   For every target condition, reuse existing full-756 spatial maps when their
   identity and luminance contracts match. Otherwise run detached on GPU with
   `n>=128` windows/trajectories, `along_scale=1`, and across scales like
   `0,0.125,0.25,0.5,0.75,1,1.5,2,3`. The primary population is RR100 derived
   from full-756 maps. Keep legacy 16-unit runs as explicitly labeled
   robustness/continuity subsets, not as the main claim.

5. Plot and promote only after audits.
   Main panels should show absolute RR100 spatial SSI bits/spike, paired deltas
   from the position-nuisance static baseline, raw bits, expected spikes, and
   unit-level activation-map suppression. Ratio plots can remain optional
   diagnostics.

Audit status as of 2026-07-07:

```text
script:
declan/active_sensing_movie_information/audit_backimage_contour_axis_rr100_inputs.py

output:
outputs/active_sensing_movie_information/backimage_contour_axis_rr100_input_audit/
```

The audit passed all hard identity checks for the n=128 matched-static
contour-axis cache:

```text
selected windows: 128/128, unique source_row, image_feature_ok=True
response manifest: 768 rows, n_units=756, dry_run=False
axis source: axis_shared_source_catalog=True for all rows
parallel/orthogonal sampled source rows: shared for 384/384 trial groups
axis catalog: 49152/49152 matched rows, max clipping_fraction=0
RR100 view: input_channels=756, n_units=100, membership=(100,756), medoid
sampled response tables: 12/12 reduce cleanly from 756 units to RR100
```

The audit produced two warnings, both expected:

```text
cached response tables have n_timebins=40, so they are auxiliary readout
sanity checks rather than spatial-SSI timebase caches

no candidate full-756 spatial-map arrays were found in the searched roots:
axis run directory, outputs/active_sensing_movie_information, outputs/twininfo
```

The full spatial-map inventory scanned 8577 NPZ files without truncation and
found no `T x 756 x H x W`-style tensors. This does not prove that no such cache
exists anywhere in the project, but it means the current contour-axis SSI path
should assume we need to generate the missing full-756 spatial maps before RR100
SSI can be computed. The existing 756-unit response tables remain reusable for
RR100 response sanity/readout diagnostics only.

The contour-axis RR100 spatial-SSI runner now lives in:

```text
declan/active_sensing_movie_information/run_backimage_contour_axis_rr100_spatial_ssi.py
```

Its promoted trajectory contract is combined, not the old pure-prior
along-versus-across observer split:

```text
source trace: scale-1 measured BackImage trace from the audited selected window
axis: selected window image_edge_axis_deg
trace = along_scale * along_component + across_scale * across_component
default promoted question: along_scale=1, sweep across_scale
```

For this older response cache, the response-table NPZ files do not actually
contain `observed_trajectory_xy`, despite the newer writer supporting that key.
The runner therefore reconstructs the scale-1 measured traces from
`selected_windows.csv` using the same BackImage trace-bank builder used by the
axis-conditioned observer. The written run metadata records this as:

```text
trace_source_contracts:
  reconstructed_trace_bank_from_selected_windows
```

First GPU smoke as of 2026-07-07:

```text
output:
outputs/active_sensing_movie_information/backimage_contour_axis_rr100_spatial_ssi_smoke_one/

command shape:
python -m declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi \
  --max-trials 1 \
  --across-scales 0,0.5,1 \
  --device cuda:1 \
  --batch-size 8 \
  --force
```

The smoke wrote an RR100 cache, absolute SSI tables, highlighted-unit table, and
Vernier-style activation-map figure:

```text
cache/backimage_contour_axis_rr100_spatial_ssi_cache.npz
condition_summary.csv
unit_ssi_table.csv
highlighted_units.csv
backimage_contour_axis_rr100_spatial_ssi_absolute_with_activation_maps.png
backimage_contour_axis_rr100_spatial_ssi_absolute_with_activation_maps.pdf
```

The one-window smoke is only a plumbing/visual check. Its population SSI was:

```text
static: 0.060425 bits/spike
along=1, across=0:   0.060494 bits/spike
along=1, across=0.5: 0.072901 bits/spike
along=1, across=1:   0.082099 bits/spike
```

So the single-window peak was at across=1, not below 1. This should not be
promoted as evidence against the hypothesis; it only shows that the
full756-to-RR100 spatial-map path, absolute SSI computation, unit activation-map
summary, and plot generation are working.

Completed n=128 production sweep from 2026-07-07:

```text
output:
outputs/active_sensing_movie_information/backimage_contour_axis_rr100_spatial_ssi_n128_across_sweep_v1/

pid:
outputs/active_sensing_movie_information/backimage_contour_axis_rr100_spatial_ssi_n128_across_sweep_v1/background_logs/run.pid

log:
outputs/active_sensing_movie_information/backimage_contour_axis_rr100_spatial_ssi_n128_across_sweep_v1/background_logs/run.log

command contract:
--max-trials 0
--across-scales 0,0.125,0.25,0.5,0.75,1,1.5,2,3
--device cuda:1
--batch-size 8
--top-units 12
--force
```

This run used the legacy time-resolved spatial SSI as its primary metric. Keep
it as a diagnostic/continuity run now that the contour-axis metric design choice
promotes SSI of the trajectory-averaged activation map.

The run completed all `1280/1280` condition evaluations and wrote:

```text
cache/backimage_contour_axis_rr100_spatial_ssi_cache.npz
movie_condition_inventory.csv
condition_summary.csv
unit_ssi_table.csv
highlighted_units.csv
summary.json
backimage_contour_axis_rr100_spatial_ssi_absolute_with_activation_maps.png
backimage_contour_axis_rr100_spatial_ssi_absolute_with_activation_maps.pdf
```

Population SSI summary:

```text
static:                   0.050927 +/- 0.001781 bits/spike
along=1, across=0:        0.060821 +/- 0.002081 bits/spike
along=1, across=0.125:    0.061035 +/- 0.002083 bits/spike
along=1, across=0.25:     0.061609 +/- 0.002084 bits/spike
along=1, across=0.5:      0.063022 +/- 0.002111 bits/spike
along=1, across=0.75:     0.064459 +/- 0.002148 bits/spike
along=1, across=1:        0.065916 +/- 0.002192 bits/spike
along=1, across=1.5:      0.068588 +/- 0.002275 bits/spike
along=1, across=2:        0.070863 +/- 0.002352 bits/spike
along=1, across=3:        0.073333 +/- 0.002430 bits/spike
```

The population curve increased monotonically over the tested across-scale range.
The peak was at `across=3x`, not below `1x`, so this n=128 RR100 spatial-SSI run
does not support the working below-1x peak hypothesis under this combined-trace
contract. It does show a positive effect of adding/amplifying across-contour
motion when along-contour motion is held at `1x`.

Mean-map-primary rerun status as of 2026-07-08:

```text
script:
declan/active_sensing_movie_information/run_backimage_contour_axis_rr100_spatial_ssi.py

schema_version:
2

promoted metric:
--primary-ssi-metric mean_map

diagnostics cached in same NPZ:
unit_mean_map_bits_per_movie
unit_time_resolved_bits_per_movie
population_mean_map_bits_per_movie
population_time_resolved_bits_per_movie
```

The one-window mean-map-primary GPU smoke completed here:

```text
outputs/active_sensing_movie_information/backimage_contour_axis_rr100_spatial_ssi_mean_map_smoke_one/
```

Smoke population summary:

```text
static:                 0.060425 bits/spike
along=1, across=0:      0.059489 mean-map, 0.060494 time-resolved
along=1, across=0.5:    0.067074 mean-map, 0.072901 time-resolved
along=1, across=1:      0.066952 mean-map, 0.082099 time-resolved
```

In this one-window smoke, the promoted mean-map metric peaked at `across=0.5x`,
while the legacy time-resolved metric still increased through `across=1x`.

The full mean-map-primary n=128 rerun completed cleanly:

```text
output:
outputs/active_sensing_movie_information/backimage_contour_axis_rr100_spatial_ssi_mean_map_primary_n128_across_sweep_v1/

pid:
outputs/active_sensing_movie_information/backimage_contour_axis_rr100_spatial_ssi_mean_map_primary_n128_across_sweep_v1/background_logs/run.pid

log:
outputs/active_sensing_movie_information/backimage_contour_axis_rr100_spatial_ssi_mean_map_primary_n128_across_sweep_v1/background_logs/run.log

command contract:
--max-trials 0
--across-scales 0,0.125,0.25,0.5,0.75,1,1.5,2,3
--device cuda:1
--batch-size 8
--top-units 12
--primary-ssi-metric mean_map
--force
```

It completed all `1280/1280` condition evaluations and wrote:

```text
cache/backimage_contour_axis_rr100_spatial_ssi_cache.npz
movie_condition_inventory.csv
condition_summary.csv
unit_ssi_table.csv
highlighted_units.csv
summary.json
backimage_contour_axis_rr100_spatial_ssi_absolute_with_activation_maps.png
backimage_contour_axis_rr100_spatial_ssi_absolute_with_activation_maps.pdf
```

Population summary, promoted mean-map SSI first and legacy time-resolved SSI in
parentheses:

```text
static:                 0.050927 +/- 0.001781  (0.050927 +/- 0.001781)
along=1, across=0:      0.052812 +/- 0.001977  (0.060821 +/- 0.002081)
along=1, across=0.125:  0.052714 +/- 0.001990  (0.061035 +/- 0.002083)
along=1, across=0.25:   0.052684 +/- 0.001998  (0.061609 +/- 0.002084)
along=1, across=0.5:    0.052628 +/- 0.002034  (0.063022 +/- 0.002111)
along=1, across=0.75:   0.052472 +/- 0.002073  (0.064459 +/- 0.002148)
along=1, across=1:      0.052241 +/- 0.002114  (0.065916 +/- 0.002192)
along=1, across=1.5:    0.051352 +/- 0.002212  (0.068588 +/- 0.002275)
along=1, across=2:      0.050273 +/- 0.002308  (0.070863 +/- 0.002352)
along=1, across=3:      0.047279 +/- 0.002371  (0.073333 +/- 0.002430)
```

The promoted mean-map curve peaks at `across=0x`, below the measured `1x`
scale. The legacy time-resolved diagnostic reproduces the old monotonic
increase through `across=3x`. This is the expected consequence of making
trajectory-averaged activation-map sharpness the promoted SSI contract.

Follow-up isotropic scale check:

```text
output:
outputs/active_sensing_movie_information/backimage_contour_axis_rr100_spatial_ssi_mean_map_isotropic_n128_scales_0_0p5_1_2_v1/

command contract:
--max-trials 0
--sweep-mode isotropic
--across-scales 0,0.5,1,2
--device cuda:1
--batch-size 8
--top-units 12
--primary-ssi-metric mean_map
--force
```

Here `--across-scales` is reused as the isotropic scale list, so each condition
uses:

```text
along_scale = across_scale = motion_scale
```

Population summary, promoted mean-map SSI first and legacy time-resolved SSI in
parentheses:

```text
0x:    0.050927 +/- 0.001781  (0.050927 +/- 0.001781)
0.5x:  0.052374 +/- 0.001942  (0.059729 +/- 0.002021)
1x:    0.052241 +/- 0.002114  (0.065916 +/- 0.002192)
2x:    0.048507 +/- 0.002343  (0.072784 +/- 0.002422)
```

Under the promoted mean-map SSI contract, isotropic `0.5x` is the peak among
these four scales, `1x` is close behind, and `2x` falls below the static `0x`
condition. The time-resolved diagnostic again keeps increasing with scale.

Follow-up crossed grid check:

```text
scale set:
0,0.25,0.5,1,2

merged output:
outputs/active_sensing_movie_information/backimage_contour_axis_rr100_spatial_ssi_mean_map_grid5_merged_v1/

new missing-pairs cache:
outputs/active_sensing_movie_information/backimage_contour_axis_rr100_spatial_ssi_mean_map_grid5_missing_pairs_n128_v1/
```

The merged 5x5 grid reuses the exact `along=1` row from the completed across
sweep and the exact isotropic diagonal where available, then fills the remaining
17 cells with the missing-pairs run. The missing-pairs run completed all
`2176/2176` condition evaluations.

Population mean-map SSI matrix:

```text
rows = along-contour scale, columns = across-contour scale

          0       0.25    0.5     1       2
0      0.050927 0.051766 0.052118 0.052607 0.052032
0.25   0.051758 0.051902 0.052096 0.052377 0.051637
0.5    0.052173 0.052234 0.052374 0.052354 0.051182
1      0.052812 0.052684 0.052628 0.052241 0.050273
2      0.053097 0.052836 0.052441 0.051375 0.048507
```

Within this coarse grid, the peak is `along=2, across=0` at `0.053097`
bits/spike, `+0.002169` above the `0x,0x` static/trial-mean map. The strongest
drop is at `along=2, across=2`, which falls below static at `0.048507`.

## Current Unit-First Scale-Setting Candidate

Status as of 2026-07-13: the live contour-axis figure candidate is no longer
the pooled mean-map population curve. The cleanest current result uses the
time-resolved instantaneous-map SSI contract, but changes the aggregation so
that each unit is compared with its own static baseline before population
averaging.

The current main diagnostic is:

```text
outputs/active_sensing_movie_information/
  backimage_contour_axis_rr100_sf_contour_alignment_long_axis30_n576_low0p05_high0p5_v1/
    unit_first_original_only_alignment_split_v1/
      backimage_rr100_original_only_contour_matched_across_scale_setting_main.png
      backimage_rr100_original_only_contour_matched_across_scale_setting_main.pdf
      unit_first_original_only_alignment_split_summary.csv
      paired_crossover_key_1x_caption_numbers.csv
```

The broader control figure is:

```text
backimage_rr100_original_only_unit_first_alignment_2x4_high_low_overlay_refined_v2.png
```

and should be treated as a supplement/control view because it includes
orthogonal pairings and along-axis sweeps.

### Measurement Contract

Use dynamic SF groups from the corrected RR100 grating-tuning readout:

```text
low SF:  preferred SF <= 0.05 cpd, n=31 RR100 units
high SF: preferred SF >= 0.5 cpd,  n=29 RR100 units
```

For the original-only contour-matched plot, usable orientation-tuned units with
aligned samples were:

```text
low SF:  23 units with contour-matched samples
high SF: 22 units with contour-matched samples
```

For each fixation/window, a unit is called contour-matched when its preferred
orientation is within the contour-alignment threshold of the local contour
axis. The figure keeps only those contour-matched unit-window pairs. This makes
the plot a "best case" for contour-matched responses, not a pooled
aligned-minus-orthogonal contrast.

For each unit `u`, window `f`, and motion condition `(a,l)`:

```text
I_u,f(a,l) = time-resolved spatial SSI in bits/spike
```

where `a` is the across-contour motion scale and `l` is the along-contour
motion scale. The plotted value is computed unit-first:

```text
Delta I_u(a,l) =
  mean_f I_u,f(a,l) over contour-matched windows
  - mean_f I_u,f(0,0) over contour-matched fully static windows

plotted curve =
  mean_u Delta I_u(a,l)
```

Thus the motion effect is unit-matched: each unit is compared with its own
fully static condition before averaging units. The low-SF and high-SF curves
are still different unit groups. The shaded band in the display figure is SEM
across units and should not be used as the paired statistical test.

### Main Across-Motion Values

Contour-matched, original movies only, static-subtracted SSI:

```text
along=0, sweep across:

across scale   low SF    high SF
0.00           0.0000    0.0000
0.25           0.0086    0.0086
0.50           0.0143    0.0170
1.00           0.0235    0.0288
2.00           0.0357    0.0278

along=1, sweep across:

across scale   low SF    high SF
0.000          0.0127    0.0125
0.125          0.0138    0.0135
0.250          0.0155    0.0151
0.500          0.0189    0.0192
0.750          0.0221    0.0233
1.000          0.0250    0.0259
1.500          0.0302    0.0265
2.000          0.0346    0.0228
3.000          0.0391    0.0099
```

The paired unit-first crossover/bootstrap numbers for caption or text are in:

```text
paired_crossover_key_1x_caption_numbers.csv
```

Key `Delta C = C(s) - C(0)` rows:

```text
high SF pure across 1x:      +0.0113 [0.0072, 0.0158]
high SF pure along 1x:       -0.0109 [-0.0147, -0.0071]
high SF across+along 1x:     +0.0020 [-0.0023, 0.0062]

low SF pure across 1x:       +0.0079 [0.0067, 0.0092]
low SF pure along 1x:        -0.0067 [-0.0078, -0.0056]
low SF across+along 1x:      +0.0014 [-0.0001, 0.0028]
```

### Interpretation

The candidate story is:

```text
The contour-matched high-SF channel may set the upper useful scale of FEMs.
```

More explicitly: low-SF contour-matched information continues to increase
beyond the natural `1x` scale, while high-SF contour-matched information is
the first to plateau or decline near `1x`. Pure across-contour motion gives a
high-SF plateau after `1x`; the strong high-SF turnover is clearest when an
along-contour component is already present (`along=1`).

Use "scale-setting channel" rather than "rate-limiting step." This avoids
claiming that high-SF units limit all information. Larger motion can still help
coarse low-SF localization while beginning to compromise fine contour-matched
information.

### Guardrails

- Do not promote pooled aligned-minus-orthogonal curves as the mechanistic
  alignment result. They are contaminated by fixed absolute-orientation
  anisotropy in the RR100 high-SF population.
- Keep the rotation control as robustness/diagnostic evidence: it exposes the
  fixed-axis high-SF anisotropy, but the main figure should be understandable
  without the original/rot90 counterfactual construction.
- Keep orthogonal pairings and along-axis sweeps as controls. They show that
  along-contour motion can erode the contour-matched advantage even when
  absolute SSI increases.
- Do not use SEM-band overlap as the significance test. Use the paired
  unit-first aligned-minus-orthogonal crossover/bootstrap intervals.
- The mean-map-primary section above is retained as history and a diagnostic
  branch. It is not the live promoted metric for the scale-setting figure.

### Microsaccade Snippet Scale Extension

The contour-axis RR100 runner now has a microsaccade source mode for the
analogous "scale the real motion and run the movie through the twin" question:

```text
python -m declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi \
  --axis-run-dir outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_matched_static_percandidate_gpu1_n128_c4_k16_scales_0p5_1_2_bconsistent_v1 \
  --trial-source-mode microsaccade_snippets \
  --sweep-mode isotropic \
  --across-scales 0,0.25,0.5,1,1.5,2,3 \
  --n-timepoints 64
```

If `--selected-windows-csv` is not provided, this mode uses the source
BackImage window table recorded in the axis-run metadata rather than the
drift-filtered selected windows. It detects real microsaccade-like high-speed
events, deduplicates overlapping source-window detections, cuts a
pre/event/post snippet, and keeps the full post-event tail inside the source
window by default. The default structural QC also rejects snippets with an
additional detected event in the tail.

The first smoke inventory was:

```text
outputs/active_sensing_movie_information/
  backimage_microsaccade_snippet_rr100_spatial_ssi_dryrun_smoke/
```

That dry run scanned 2000 source windows, found 28 usable microsaccade
snippets after deduplication and QC, and wrote a 2-snippet x 4-condition
isotropic-scale inventory. It did not run the twin.

Future scale unification: drift windows and microsaccade snippets should
eventually be reported on the same raw arcmin axis, and also on a unit-relative
phase axis:

```text
movement_arcmin * preferred_SF_cycles_per_arcmin
```

This would test whether high-SF units turn over at smaller raw displacements
because the same microsaccade or drift step sweeps through more of their
preferred spatial period.

## Practical Revival Path

First cache-only pass:

1. Write or run a small audit that verifies:
   - no duplicate `(example_id, kind, image_index, crop_rank, condition)` rows;
   - every paired key has the expected condition inventory;
   - summary final columns match the last sample in
     `cache/cumulative_information_series.npz`;
   - `spatial_ssi_uses_shift_grid=False` for all rows;
   - `stabilized` and random controls preserve mean position in trajectory QC;
   - `random_amp_cloud_matched` has no fallback rows in the promoted cache.
2. Regenerate the figure-layer summaries from the existing cache.
3. Add a compact manifest next to the figure outputs that records the merged
   condition inventory and the SSI contract.

Then, if the cache passes, use these primary panels/claims:

```text
real vs trial-mean stabilized
real vs matched random-motion controls
SF-band FEM-minus-stabilized interaction
fixation-only vs microsaccade-window stratification
raw bits / expected spikes companion audit
```

If literal MP4 stimulus videos are needed, note that the production SSI run was
configured with `make_stimulus_movies=false`. The existing QC scripts can
reconstruct the selected retinal movies from the saved image/trace metadata,
but a video-facing rerun should use `--make-stimulus-movies` or a focused
movie-only export rather than changing the SSI cache in place.

## Scaled-Real Unit SSI Plot

The Vernier-style BackImage unit-SSI scale-line plot now lives in:

```text
declan/active_sensing_movie_information/plot_backimage_scaled_real_unit_ssi.py
```

It renders each selected natural-image crop under:

```text
scaled_trace = mean(trace) + scale * (trace - mean(trace))
```

so `0x` is the trial-mean stabilized baseline, `1x` is the measured trace, and
larger values amplify the same trace around its own mean. The metric is final
cumulative spatial SSI per expected spike from full convolutional rate maps,
with unit activation maps averaged over movie time and selected source movies.
The default line panels are now absolute SSI in bits/spike, not a per-unit
division by the `0x` static/trial-mean baseline. The old fold-change diagnostic
is still available with `--line-y-mode log2_ratio`, but should be treated as a
fragile view because near-zero static SSI can dominate the apparent effect size.

The first CPU-bound one-movie pilot is:

```text
outputs/active_sensing_movie_information/backimage_spatial_ssi_scale_line_pilot_onepair/
```

Main files:

```text
backimage_scaled_real_unit_ssi_absolute_with_activation_maps.png
backimage_scaled_real_unit_ssi_absolute_with_activation_maps.pdf
backimage_scaled_real_scale_summary.csv
backimage_scaled_real_unit_ssi_table.csv
backimage_scaled_real_highlighted_units.csv
cache/backimage_scaled_real_unit_ssi_cache.npz
```

When the needed full-756 spatial maps are absent, a multi-image run should be
treated as a GPU/background job. The one-movie pilot is a visual and contract
smoke, not a promoted result. It used the small production spatial-map
population for continuity with the original BackImage SSI cache; the planned
contour-axis run should use RR100 derived from full-756 spatial maps as its
primary population.

## RR100 Single-Cell Frequency Tuning Readout

The RR100 twin still emits tiled post-activation rate maps:

```text
rate_map[t, unit, y, x]
```

For a single-cell scalar tuning probe, do not average over the spatial map by
default. The previous RR100 construction/QC used center-pixel traces to inspect
cell-like time courses, and that remains the intended scalar readout for
spatial- and temporal-frequency tuning:

```text
scalar_response[t, unit] = rate_map[t, unit, center_y, center_x]
```

The map values are already after the model's Softplus output nonlinearity
because `scripts/spatial_info.py::compute_rate_map` applies
`model.model.activation(y_batch)` before the RR100 post-activation population
view is applied. A spatial mean can still be useful as a diagnostic, but it
does not represent a single modeled cell at the central retinotopic location.

The corrected tuning probe lives in:

```text
declan/active_sensing_movie_information/run_backimage_rr100_frequency_tuning_probe.py
```

Its default contract is:

```text
stimulus grid = shared grating orientation x spatial frequency x temporal frequency
computed units = all RR100 units for every shared movie
scalar_readout = center_pixel of the post-activation RR100 rate map
SF grid = 0.0125, 0.05, 0.2, 0.8, 3.2, 12.8
TF grid = 0, 0.2, 0.8, 3.2, 12.8, 47.2
TF=0 baseline = seeded shuffled starting phases, averaged as static mean rate
phase repeats = 4 static phases and 2 dynamic phases by default
selected-unit heatmaps = phase-averaged mean center-pixel rate, including TF=0
static tuning = best orientation/SF by mean rate at TF=0
dynamic tuning = best orientation/SF/TF by phase-RMS response amplitude
```

This matters because unit-specific preferred-orientation movies cannot be
computed for the entire RR100 population "at once"; using a shared orientation
grid preserves the repeated-stimulus contract and lets each unit's preferred
orientation, SF, and TF be summarized after readout.

For very low TF probes, dynamic F1/amplitude estimates are secondary because a
1.5 s movie does not span a full cycle at the lowest temporal frequencies. The
default TF grid now starts dynamic conditions at `0.2 Hz`; the `0 Hz` row is the
phase-shuffled static baseline. The main selected-unit SF/TF heatmap uses mean
center-pixel rate rather than sinusoidal amplitude; the cyan marker can still
indicate the dynamic amplitude peak as a reference.

The default spatial grid stops at `12.8 cpd`. With the current `ppd =
37.50476617`, the pixel-grid Nyquist limit is about `18.75 cpd`; the previously
tested `47.2 cpd` spatial grating is therefore above Nyquist and should be
treated as an aliased diagnostic rather than a valid SF tuning point. The
manifest records `spatial_nyquist_cpd`, `temporal_nyquist_hz`, and any requested
frequencies above those limits.

## Current Claim Boundary

Promotable with audit:

```text
For natural-image BackImage movies, retinal motion increases a deterministic
V1-model spatial-SSI efficiency proxy relative to trial-mean stabilization,
especially for mid/high spatial-frequency content.
```

Not promotable from this metric alone:

```text
measured FEM statistics are uniquely optimal
retinal motion necessarily improves a pose-blind downstream decoder
the result directly measures biological spike-train information
```
