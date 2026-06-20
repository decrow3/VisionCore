# Figure 4 Active Sensing Atlas

Status: working skeleton, cache-first.

## Working Results Lead

Having shown that fixational eye movements account for a structured component
of foveal V1 variability, the expanded Figure 4 should ask whether that
reafferent variability can be interpreted as an active-sensing signal. During
fixation, the screen stimulus is fixed but the retinal stimulus is not: drift
and microsaccades translate the image across the retina, converting a static
image into a small retinal movie. The atlas should then ask whether this movie
adds recoverable structure to V1-twin responses, whether that structure remains
usable when eye position is latent, and whether animals' measured FEM
directions are related to the local image geometry that makes such movies
useful.

The figure should not start from "compact tangent geometry" as the headline.
Compact geometry is a mechanism candidate for the observer result, not the
premise for every downstream claim.

## Module A: FEMs Turn A Static Image Into A Retinal Movie

Purpose:

```text
The screen image is fixed, but the retinal image is not fixed.
```

Candidate panels:

- A1: fixed screen image, measured eye trace, and three shifted retinal crops.
- A2: stabilized versus FEM movie rows.
- A3: small-translation cartoon explaining why local image gradients matter.
- A4: image patch + eye trace -> retinal movie -> V1 twin -> response movie.
- A5: bridge from retinal motion to FEM-linked covariance.

Likely sources:

- `declan/fig4_active_sensing/`
- `declan/active_sensing_movie_information/`
- `declan/figure4_active_sensing_atlas/figures/panel_A/`
- `outputs/active_sensing_movie_information/active_sensing_movie_information_figure_frozen_20260615_pre_backimage_collab_pack/`
- `outputs/covTFTS_figure_frozen_20260615_pre_backimage_active_sensing_collab_pack/`

Generated read:

```text
Panel A subpanels now cover the fixed-screen-to-retinal-movie premise,
stabilized-versus-FEM transform QC, local-gradient sampling cartoon, canonical
BackImage/V1-twin pipeline, and a mixed-denominator covariance bridge.
```

Claim boundary:

```text
Retinal motion is part of the sensory input during fixation. The panel should
teach the physical premise and provide a bridge to FEM-linked covariance, not
claim functional optimality by itself.
```

## Module B: FEM Movies Can Improve Visual Encoding

Purpose:

```text
Does retinal motion add feature-decodable structure beyond the static response?
```

Candidate panels:

- B1: target image features and V1 response movie.
- B2: response summaries: mean, temporal PCs/DCT, motion delta.
- B3: motion families: static, empirical, OU, Brownian, rotated.
- B4: static + motion gain over static.
- B5: empirical minus matched controls.
- B6: scale guardrail showing that larger motion is not simply better.
- B7: optional local pairing supplement.

Primary existing result:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/
    incremental_static_plus_motion_relids/
```

Result notes:

- Canonical 756-unit V1 twin.
- 256 images, K=4 trace samples, grouped-by-image CV.
- Families: empirical, OU, Brownian, rotated.
- Scales: 0.25x, 0.5x, 1x, 1.5x, 2x.
- Drift-only, common-unclipped source pool.
- Accepted drift-only sources: 151 / 256.
- Median effective/requested RMS: 1.0 for every family/scale.
- Clipped fraction: 0.0 for every family/scale.

Main safe wording:

```text
In a cleaned BackImage aggregate run, empirical drift-like motion adds
feature-decoding signal beyond static V1-twin responses and outperforms
OU-like confined controls across scale. The advantage over Brownian/generic
motion is strongest at small biologically plausible scales and narrows at
larger scales, so the claim is scale- and readout-dependent.
```

## Module C: Joint Image-And-Eye Observer

Purpose:

```text
If retinal motion is useful, can an observer use it without being handed the
true eye trace?
```

Candidate panels:

- C1: observer schematic with latent trajectory marginalization.
- C2: known-eye, zero-eye, and joint-eye accuracy ordering.
- C3: matched-static-response distractor control at 1.0x.
- C4: posterior concentration, `N_eff / K`.
- C5: scale-gap guardrail showing larger rescue when zero-eye fails.
- C6: compact-mechanism guardrail, not unique-mechanism proof.
- C7: optional Vernier failure versus natural-image success intuition.

Primary existing result:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
```

Key matched-static read:

```text
matched_static_response, 1.0x:
  zero-eye = 0.328
  joint-eye = 0.672 to 0.797

At likelihood scale 1.0:
  empirical prior joint = 0.766, median N_eff / K = 0.364
  OU prior joint = 0.797, median N_eff / K = 0.400
```

Compact mechanism source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
    compact_mechanism_image_disjoint_fold0_n512_k2_5_10_20_rand8_log_v1/
```

Mechanism boundary:

```text
The exact-cache observer establishes the trajectory-marginalized rescue. The
image-disjoint compact projection supports compact-translation sufficiency
above random/unit-shuffle/gain controls, but static-PC controls remain a close
alternative low-dimensional explanation.
```

## Module D: Image-Dependent Useful Motion Directions

Purpose:

```text
Which motion directions are useful for which images?
```

Candidate panels:

- D1: local edge-parallel and edge-orthogonal axis schematic.
- D2: axis-conditioned observer accuracy.
- D3: axis-preference guardrail across candidate set and scale.
- D4: edge-parallel preservation audit for pixels and V1-twin responses.
- D5: objective-alignment guardrail versus raw edge geometry.

Primary source families:

- `declan/axis_conditioned_backimage_trajectory_observer/`
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1/`
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/`
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_edge_parallel_stability_screen_yfix_n256_pop256/`

Current read:

```text
Axis-conditioned priors rescue image identity over zero-eye, but preferred axis
depends on candidate set and scale. Edge-parallel motion strongly preserves
local pixels and V1-twin responses relative to edge-orthogonal motion.
```

Claim boundary:

```text
Use D to generate and compare image-dependent motion predictions. Do not claim
one universal biological axis unless a specific objective and candidate set
support it.
```

## Module E: Free-Viewing FEMs Follow Image Geometry

Purpose:

```text
Do animals deploy FEMs in directions related to local image geometry?
```

Candidate panels:

- E1: representative free-viewing image patch with local edge and drift axes.
- E2: behavioral drift-edge alignment strength.
- E3: endpoint-zone enrichment relative to a uniform angular expectation.
- E4: weighted versus unweighted metric-convention guardrail.
- E5: supported versus not-yet-supported summary.
- E6: full drift-edge distribution and session diagnostic.
- E7: confidence and signed-delta diagnostic.
- E8: endpoint/null diagnostic for the E3 summary.

Primary behavior source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_edge_alignment_distribution_inspection/
```

Current behavior read:

```text
All windows:
  mean session drift-edge cos2 = 0.105
  95% CI = [0.067, 0.145]

Reliable axes:
  mean session drift-edge cos2 = 0.140
  95% CI = [0.089, 0.188]

High confidence:
  mean session drift-edge cos2 = 0.269
  95% CI = [0.138, 0.396]

Parallel endpoint-zone enrichment:
  all windows = 1.304x uniform expectation
  reliable axes = 1.427x uniform expectation
  high confidence = 2.124x uniform expectation
```

Raw edge baseline source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_conditional_fixation_objectives_twin_axis_only_n256/
    alignment_by_objective_summary.csv
```

Raw edge read:

```text
raw_edge_axis:
  mean session cos2 = 0.182
  weighted session cos2 = 0.218
  positive sessions = 23 / 29
```

Claim boundary:

```text
The behavior supports image-contingent FEM geometry. It does not yet show that
animals optimize a specific V1-twin objective better than raw local edge
geometry.
```

## Candidate Main-Figure Compression

- Main 4A: FEMs create a retinal movie from a fixed image.
- Main 4B: empirical motion adds feature-decodable structure beyond static and
  beats controls most cleanly at small scales.
- Main 4C: known-eye > joint-eye > zero-eye, with matched-static gap recovery.
- Main 4D: local image geometry defines useful motion axes, with axis-readout
  guardrails.
- Main 4E: free-viewing drift aligns with local image geometry.

Likely supplements:

- retinal rendering QC;
- motion-family matching;
- response summary comparisons;
- Vernier failure versus natural-image success;
- posterior `N_eff`;
- compact-only/compact-removed mechanism tests;
- axis-conditioned observer details;
- behavioral null controls.
