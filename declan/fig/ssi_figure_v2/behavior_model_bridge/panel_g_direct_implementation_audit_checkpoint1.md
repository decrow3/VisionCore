# Panel G direct implementation audit — Checkpoint 1

## Question

Before interpreting the direct real-minus-rotation result as a biological or
model negative, determine whether it differs from the historical interpolated
Panel G because of a rendering, coordinate-frame, population, SSI, aggregation,
or cohort-contract mismatch.

## Findings that pass

1. **Trace identity passes.** All 1,000 production traces equal the saved
   Figure 4 central-40-sample trace bank exactly (maximum absolute error zero).
2. **Real-movie identity passes.** Six native image-trace pairs also occur in
   the original 100-image x 1,000-trace matrix. New versus old unit SSI,
   expected spikes, and mean rates agree exactly for five pairs and within
   `5.96e-8` for the sixth. This validates patch extraction, stimulus
   normalization, model version, response alignment, RR100 indexing, and the
   time-resolved SSI implementation for the real condition.
3. **The old image-support gate does not recover the direct effect.** Restricting
   native pairs to the old bank's coherence, contrast, and inside-image support
   leaves the direct coherence association near zero.
4. **Equal-pair versus spike-pooled aggregation is not the main explanation.**
   Reassembling direct population SSI by pooling information numerators and
   expected-spike denominators, as in the historical curves, retains the
   non-monotonic result. The historical-frame aligned population has pooled
   percent differences versus rotation of `+0.17`, `-0.45`, `+0.34`, and
   `-0.07%` across the four coherence bins.

## Implementation mismatches found

### 1. Unit-orientation coordinate frame

`prior_preferred_orientation_deg` is an image-array-frame angle. Both
`run_panel_g_exact_pair_production.py::_population_masks` and
`run_direct_exact_pair_ssi.py::_unit_selections` compare it directly with the
gaze-frame `image_edge_axis_deg`. The established alignment implementation
converts the contour first:

```text
contour_axis_image_deg = (-contour_axis_gaze_deg) mod 180
```

Only 8.7% of pairs retain exactly the same aligned-unit membership after the
conversion; mean membership Jaccard is 0.235. Reassembly from the all-unit
caches shows that correcting the membership does **not** restore the effect
(corrected direct coherence Spearman `-0.030`).

The historical `panel_g_alternative_x_axes_diagnostic.py` makes the same direct
gaze-versus-image comparison. Thus this is a real scientific labeling bug
shared by the old and new aligned populations, not the source of their large
difference. Both paths need correction before a final aligned-population claim.

### 2. Cohort is not the original movie-pair set

The production cohort reuses the 1,000 Figure 4 traces but restores each to its
own native image. It is not the original 100 x 1,000 matrix of image-trace
pairs; only six production pairs occur in that matrix. This was a principled
native-pair choice, but it cannot isolate interpolation from image/pair support.

### 3. Drift-only curve support versus behavior application

The historical RMS curves were fit on the 800 drift-only traces, while the
behavior bridge was applied to all windows. The 1,000-pair cohort deliberately
contains 200 microsaccade traces, so its old-surrogate high-coherence means are
greatly amplified by those traces. In the full historical behavior cohort,
microsaccades are only about 2.7% of windows above coherence 0.5, and the
drift-only surrogate mean remains positive, so this is an amplification and
support problem rather than a complete explanation.

### 4. Historical mean effect is tail-sensitive

For the full historical bridge, the high-coherence surrogate distributions
have small medians relative to means:

| coherence | mean (pp) | median (pp) |
|---|---:|---:|
| 0.5-0.8 | +0.078 | -0.0004 |
| 0.8-1 | +0.165 | +0.0126 |

In the 0.5-0.8 bin, the largest 1% of finite window effects contribute 77% of
the net summed advantage. In the 0.8-1 bin, the largest 5% contribute
approximately all of the net summed advantage. The old positive result is
therefore reproducible but not a broad per-window shift.

## Calibration check on the original matrix

The historical aligned-high-SF RMS surrogate was compared with actual
population SSI across the 800 drift-only traces separately within each of the
100 original images. Median within-image Spearman correlation was `0.031`, and
exactly half the images had a positive correlation. Pooled within-image rank
correlation was `0.063`.

This does not prove the counterfactual rotation effect is absent. It shows that
the pooled one-dimensional curve is weakly calibrated as an individual-image
predictor even on the matrix from which it was constructed. Strong image-level
heterogeneity is therefore a live explanation for the interpolation/direct
difference.

## Saved diagnostics

```text
outputs/fig/ssi_figure_v2/behavior_model_bridge/
  panel_g_exact_pair_fig4_trace_bank_n1000_v1/implementation_audit_checkpoint1/
    old_matrix_exact_pair_identity.csv
    corrected_alignment_reassembly.csv
    spike_pooled_direct_reassembly.csv
    historical_high_coherence_surrogate_by_microsaccade.csv
    old_matrix_within_image_surrogate_calibration.csv
```

## Smallest decisive next test

Use a frozen, role-stratified sample of actual pairs from the original 100 x
1,000 matrix, not native re-pairings. Reuse each cached real condition and
freshly evaluate only its rotated twins. Include:

- large positive historical surrogate predictions;
- near-zero predictions;
- negative predictions;
- multiple images with positive and negative within-image calibration;
- drift-only examples first;
- correctly converted and historical-frame unit populations side by side.

Render the exact input movies and selected-unit maps for this small set before
launching another population run. If direct rotations reproduce the surrogate
inside the original matrix support, the native-image cohort caused the earlier
discrepancy. If they do not, the pooled marginal curve is not a valid
counterfactual response function.
