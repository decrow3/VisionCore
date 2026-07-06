# V1-RR100 Twin Unit Consolidation Notes

Last updated: 2026-07-01

## Short Version

We started from the 756-channel canonical V1 digital twin and built a reduced
population view that removes likely duplicate or near-duplicate modeled units
across recording sessions while preserving the convolutional/spatial use of each
unit. The current compact working candidate is:

```text
V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_medoidPosthocminRepcomplete0p45_movieMedoid
```

In prose, this is the **V1-RR100 movie-medoid** population:

- 756 canonical input channels
- 113 channels excluded as weak/bad under the construction QC
- 100 representatives
- 58 multi-unit groups
- 42 singleton representatives
- medoid pooling: each group is represented by one actual modeled unit, not by
  the average activation map

The conservative validated multi-stimulus population before the final aggressive
compression had 265 representatives. RR100 is a post-hoc compression of that
vetted base, chosen to get closer to the biological scale we wanted while still
passing the main visual and reconstruction sanity checks.

## What Redundancy We Removed

The canonical twin is convolutional: each modeled unit is evaluated across
spatial positions. We did **not** collapse that spatial grid. That redundancy is
the useful part of the twin, because it lets a single modeled cell type sample
different retinal positions.

The redundancy targeted here was channel redundancy: different channels in the
756-unit canonical population that behaved like the same or nearly the same unit
after centering and combining modeled units across sessions. Those can arise
from repeated recordings of the same biological unit, near-duplicate units across
days, or units at different retinotopic positions that become redundant after the
canonical centering step.

## Core Representation

For a stimulus movie, the full twin produces an activation movie:

```text
(time, channel, height, width) = (T, 756, H, W)
```

For redundancy discovery, each channel was represented by a fingerprint formed
from its activation over time and space. Operationally, we flattened the
activation movie across time and spatial positions for each channel, with
z-scoring as the default normalization. Similarity was then measured with
correlation in this high-dimensional activation-fingerprint space.

t-SNE/PCA plots were used as visual diagnostics only. They helped us see whether
clusters looked plausible, but the merge decisions were made from correlation
and movie-based QC, not from 2D embedding distance.

## Initial Clustering Choices

The first exploratory comparison swept:

- raw fingerprint correlation vs PCA/cosine similarity
- complete vs average linkage
- thresholds such as 0.60, 0.75, and 0.90

Complete linkage was favored because it is conservative: every member of a group
has to remain close to the rest of the group under the selected threshold.
Average linkage produced very large groups whose weakest pairwise/member
relationships could be poor, so it was useful as a stress test but not as the
main grouping rule.

The first-stage construction threshold used in the final multi-stimulus path was
complete linkage at correlation `0.65`.

## Why We Added Movie-Based Splitting

Simple fingerprint clustering was not enough. Some groups that looked reasonable
under the construction fingerprint had weak or even negative relationships in
specific held-out movies or stimulus regimes. The important fix was to validate
groups in the actual activation movie space:

- member-to-cluster-centroid correlation after flattening `(time, H, W)`
- weakest pairwise member correlation as a diagnostic
- pool-expand reconstruction of the full 756-channel activation movie
- visual inspection of activation maps and center-pixel traces

The main split rule was based on generic movie quality, not on a downstream
science result. Groups were recursively split when their movie member-centroid
quality failed the threshold, and pairwise failures were tracked as additional
warnings. A final cleanup converted groups with unresolved centroid failures into
singletons.

## Stimulus Coverage

BackImage movies were the original construction stimulus family because they
have natural images and real eye traces. We then added FixRSVP because it drives
more diverse responses and exposed group failures that BackImage alone could
miss.

The multi-stimulus construction used a conservative similarity matrix:

```text
multi_corr = min(BackImage_corr, FixRSVP_corr)
```

Thus a pair had to look similar under both stimulus families to be mergeable.
The final multi-stimulus construction battery used one FixRSVP dynamic case plus
seven held-out BackImage cases.

## Conservative Multi-Stimulus Base

The conservative base population was:

```text
V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75
```

Summary:

- 265 representatives
- 110 groups
- 155 singletons
- 113 excluded channels
- 5 groups force-split in the final cleanup
- final movie centroid gate: `0.75`
- multi-stimulus worst group member-centroid corr: `0.757`
- multi-stimulus median group member-centroid corr: `0.916`
- mean global reconstruction corr: `0.839`

This was the conservative "safe" redundancy-resolved population, but it was
larger than the desired biological-scale target.

## Post-Hoc Compression Toward 100 Units

To get closer to roughly 100 units, we compressed the already-vetted 265-rep
base rather than restarting from raw channels. The post-hoc compression:

1. Built representative fingerprints from cached construction movies.
2. Computed representative-to-representative correlations.
3. Combined case-specific correlations with a conservative `min` rule.
4. Merged representatives with complete linkage.
5. Swept thresholds and selected the threshold nearest 100 representatives.

The threshold sweep selected `0.45`:

```text
V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_medoidPosthocminRepcomplete0p45
```

This produced exactly 100 representatives: 58 multi-unit groups and 42
singletons.

## Mean Pooling vs Movie Medoids

We considered two ways to turn each group into one representative:

- **Mean pooling** averages all group members. This reconstructs the full
  population better but can blur activation maps, especially when members are
  similar up to gain or have slight spatial/phase differences.
- **Movie medoid pooling** chooses one actual channel per group. This keeps the
  representative cell-like and avoids cancellation/blurring in the activation
  map.

For each group, the movie medoid was the channel with the best worst-case
member-centroid relationship across the construction movies, with median
member-centroid correlation and `ccnorm` as tie-breaker information.

We promoted the movie-medoid version as the working RR100 twin because it is
more interpretable as a reduced set of actual modeled units. The mean-pooled
version remains useful as a reconstruction upper bound.

## RR100 QC Summary

The dedicated RR100 QC script is:

```text
declan/inspect_rr100_movie_medoid_population.py
```

It generates correlation heatmaps, group-quality summaries, activation maps,
center-pixel traces, sharp singleton/representative plots, and reconstruction
tables.

Across the RR100 QC battery, medoid group quality was:

- selected medoid to group centroid: median `0.960`, mean `0.951`
- member-to-centroid minimum within group/case: median `0.805`, mean `0.798`
- selected medoid to weakest member: median `0.679`, mean `0.674`

This is not a claim that every pair inside every compressed group is identical.
Some aggressive RR100 groups still have weak pairwise/member relationships in
particular cases. The point is that the selected medoid is usually close to the
group's common movie response and that the reduced population remains useful as
a compact representative set.

## Reconstruction Checks

The raw pool-expand reconstruction asks: if we reduce the 756-channel movie to
the reduced population and then expand each representative back to all members
of its group, how well do we reconstruct the original full movie?

For RR100 movie-medoid, raw copy expansion across 14 QC cases gave:

- mean global reconstruction corr: `0.570`
- minimum global reconstruction corr: `0.479`
- mean 5th-percentile channel corr: `0.652`
- mean median channel corr: `0.877`

For comparison:

- RR100 mean pooling: mean global corr `0.682`
- earlier RR192 mean population: mean global corr `0.807`

This raw metric penalizes the medoid view for amplitude/gain differences inside
a group. When we allowed a nonnegative channel-wise affine expansion fit, RR100
movie-medoid improved to:

- mean global reconstruction corr: `0.955`
- minimum global reconstruction corr: `0.942`

That gain-sensitive check was important: it showed that much of the missing raw
variance was scale/offset mismatch rather than a complete loss of response
shape. We still treat the raw reconstruction loss as a real caveat when exact
full-population amplitudes matter.

## Vernier/SSI Sanity Checks

We then used the Vernier active-sensing walkthrough as a minimal downstream
functional check. The relevant script is:

```text
notebooks/vernier_active_sensing_walkthrough.py
```

For spatial spiking information (SSI), RR100 retained a substantial fraction of
the full 756-channel per-spike SSI but, as expected, much less total population
rate:

| condition | full SSI | RR100 SSI | RR100/full SSI | RR100/full total rate |
| --- | ---: | ---: | ---: | ---: |
| static center | 0.00723 | 0.00506 | 0.700 | 0.267 |
| real FEM | 0.00908 | 0.00603 | 0.664 | 0.267 |
| horizontal/across only | 0.00953 | 0.00610 | 0.641 | 0.266 |
| vertical/along only | 0.00856 | 0.00552 | 0.645 | 0.266 |

For the RR100 Vernier pose-aware Fisher check:

| condition | pose-aware Fisher | vs static | mean SSI |
| --- | ---: | ---: | ---: |
| static center | 0.03384 | 1.000 | 0.00506 |
| horizontal/across only | 0.03642 | 1.076 | 0.00610 |
| vertical/along only | 0.04124 | 1.219 | 0.00552 |
| real FEM | 0.03457 | 1.022 | 0.00603 |

This is qualitatively similar to the full-twin axis-only result but much weaker:
the reduced twin keeps a small pose-aware Vernier modulation, but it does not
support a strong "along is intrinsically better" conclusion.

The matched-total anisotropic Brownian check was also deliberately conservative:
it compared across-elongated and along-elongated Brownian traces while holding
total motion variance fixed. In the longer RR100 run:

- isotropic Fisher: `0.0560`
- across-elongated Fisher: `0.0570` (`1.018x` isotropic)
- along-elongated Fisher: `0.0585` (`1.044x` isotropic)
- SSI stayed essentially unchanged (`~1.00x` isotropic)

Trace-level paired differences were small relative to SEM, so this should be
read as basically null for matched-total anisotropic Brownian motion in RR100.

## How To Load RR100

Use the portable population loader and pass the explicit RR100 version:

```python
from declan.redundancy_resolved_v1_population import (
    apply_population_pooling,
    load_population_view,
)

RR100_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75"
    "_medoidPosthocminRepcomplete0p45_movieMedoid"
)

view = load_population_view(version_name=RR100_VERSION)
reduced_movie = apply_population_pooling(full_movie_tchw, view.membership)
```

For medoid specs, `view.membership` is one-hot: it selects the chosen medoid
channel for each group. `view.cluster_membership` records all members in each
group and is the right matrix for group-size summaries or expand-to-full QC.

## Main Artifacts

Construction and population specs:

```text
declan/inspect_canonical_twin_channel_redundancy.py
declan/run_cached_rr_medoid_compression_frontier.py
declan/redundancy_resolved_v1_population.py
outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints/
```

RR100 QC:

```text
declan/inspect_rr100_movie_medoid_population.py
outputs/redundancy_resolved_v1_twin/rr100_movie_medoid_qc_V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_medoidPosthocminRepcomplete0p45_movieMedoid/
```

Vernier/SSI checks:

```text
notebooks/vernier_active_sensing_walkthrough.py
declan/vernier_active_sensing/run_rr100_anisotropic_brownian.py
outputs/notebook_vernier_walkthrough/ssi_population_comparison/
outputs/notebook_vernier_walkthrough/rr100_anisotropic_brownian_long/
```

## Key Saved Plots

Paths below are relative to the repository root.

Initial fingerprint and redundancy structure:

```text
outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints/fingerprint_corr_heatmap_Rochester_Meliora1_JPG_trace000.png
outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints/redundancy_similarity_linkage_metric_grid.png
outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints/redundancy_embedding_complete_tsne_Rochester_Meliora1_JPG_trace000.png
outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints/redundancy_embedding_cluster_activation_audit_center_pixel_complete_corr0p65_tsne_Rochester_Meliora1_JPG_trace000ppng.png
```

Conservative multi-stimulus construction and final cleanup:

```text
outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints/candidate_audit_summary_V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_MultiStim_construction.png
outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints/candidate_audit_summary_V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_MultiStim_construction_final.png
outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints/candidate_reduced_corr_heatmap_V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_MultiStim_construction_final.png
outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints/candidate_group_activation_maps_V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_MultiStim_construction_final.png
outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints/candidate_group_trace_overlays_center_pixel_V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_MultiStim_construction_final.png
```

Post-hoc compression to the 100-representative target:

```text
outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints/cached_medoid_compression_frontier_summary_V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_cachedMedoid_posthoc_min_complete_target100_cases8.png
outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints/movie_medoid_corr_heatmap_V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_medoidPosthocminRepcomplete0p45_movieMedoid.png
```

RR100 movie-medoid QC:

```text
outputs/redundancy_resolved_v1_twin/rr100_movie_medoid_qc_V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_medoidPosthocminRepcomplete0p45_movieMedoid/figures/rr100_movie_medoid_construction_corr_heatmap.png
outputs/redundancy_resolved_v1_twin/rr100_movie_medoid_qc_V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_medoidPosthocminRepcomplete0p45_movieMedoid/figures/rr100_movie_medoid_heldout_corr_heatmap.png
outputs/redundancy_resolved_v1_twin/rr100_movie_medoid_qc_V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_medoidPosthocminRepcomplete0p45_movieMedoid/figures/rr100_movie_medoid_group_quality_summary.png
outputs/redundancy_resolved_v1_twin/rr100_movie_medoid_qc_V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_medoidPosthocminRepcomplete0p45_movieMedoid/figures/rr100_reconstruction_metrics.png
outputs/redundancy_resolved_v1_twin/rr100_movie_medoid_qc_V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_medoidPosthocminRepcomplete0p45_movieMedoid/figures/rr100_reconstruction_metrics_with_nonnegative_affine.png
outputs/redundancy_resolved_v1_twin/rr100_movie_medoid_qc_V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_medoidPosthocminRepcomplete0p45_movieMedoid/figures/rr100_group_activation_maps_BeachWave_jpg.png
outputs/redundancy_resolved_v1_twin/rr100_movie_medoid_qc_V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_medoidPosthocminRepcomplete0p45_movieMedoid/figures/rr100_group_center_pixel_traces_BeachWave_jpg.png
outputs/redundancy_resolved_v1_twin/rr100_movie_medoid_qc_V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_medoidPosthocminRepcomplete0p45_movieMedoid/figures/rr100_sharp_representatives_BeachWave_jpg.png
outputs/redundancy_resolved_v1_twin/rr100_movie_medoid_qc_V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_medoidPosthocminRepcomplete0p45_movieMedoid/figures/rr100_sharp_singletons_BeachWave_jpg.png
```

Downstream Vernier/SSI checks:

```text
outputs/notebook_vernier_walkthrough/ssi_population_comparison/vernier_ssi_population_bars.png
outputs/notebook_vernier_walkthrough/ssi_population_comparison/vernier_ssi_population_timecourses.png
outputs/notebook_vernier_walkthrough/ssi_population_comparison/rr100_vernier_along_across_poseaware_ssi_only.png
outputs/notebook_vernier_walkthrough/ssi_population_comparison/rr100_vernier_axis_suppression_bars_poseaware_ssi.png
outputs/notebook_vernier_walkthrough/rr100_anisotropic_brownian_long/rr100_anisotropic_brownian_poseaware_ssi_bars.png
outputs/notebook_vernier_walkthrough/rr100_anisotropic_brownian_long/rr100_anisotropic_brownian_trace_level_spread.png
outputs/notebook_vernier_walkthrough/vernier_active_sensing_walkthrough_executed.pdf
```

## Current Interpretation

RR100 is a useful compact working twin, not a proof that the full 756-channel
population has no remaining structure worth preserving. It is best used when we
want a smaller, cell-like representative population that avoids obvious duplicate
channels and is easier to inspect visually.

For exact reconstruction of the full canonical twin, RR192 or the full 756
population is safer. For interpretable reduced-population science checks, the
RR100 movie-medoid version is the current preferred compact candidate, with the
caveat that sensitive conclusions should be checked against the full twin or a
less aggressive reduced population.
