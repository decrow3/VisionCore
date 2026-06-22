# Provisional Figure 4 v0

Status: provisional, 2026-06-21
Source handoff: `provisional_figure4_companion_docs_handoff.md`
Panel contract: `provisional_panel_contract_v0.csv`
Selected composite: `figures/composites/figure4_selected_v5.png`

This document is a cache-first Figure 4 package that can support scientific
review after the first power-rerun integration. It deliberately keeps the
strongest current result separate from the attractive but unproven
interpretation that measured eye movements optimize the tested model objective.

## Working Figure Claim

Measured fixational eye movements turn fixed natural images into structured
retinal movies. In the V1 twin, those movies produce feature-relevant response
changes and can support feature recovery when eye trajectory is latent.
Local image geometry supplies a plausible axis for useful movement, and
measured drift geometry is modestly but reliably contour-following. The current
behavioral bridge is convergence of useful model geometry and measured
image-geometry alignment, not a direct behavioral optimality test.

## Why This Is Provisional

The current corrected feature candidate is:

```text
primary feature readout: pyramid_local_field k16 delta_mean
secondary temporal diagnostic: pyramid_local_field k16 temporal_pca
```

The joint `rel_0p25x` completion and first higher-power Figure 4 reruns are now
complete. A corrected v6 aggregate posthoc now uses the static mean response as
the static baseline for all motion summaries. This supersedes the earlier v5
temporal-PCA aggregate interpretation. A follow-up all-readout/nested-alpha
audit makes the feature target a role split rather than a single winner:
`mean` and `delta_mean` are the absolute aggregate candidates, `delta_mean`
remains the local mechanistic bridge, and temporal PCA/DCT variants are
order-sensitive empirical-vs-control diagnostics. The package remains
provisional until the OU trace-control audit is closed and an explicit
promotion/lock pass is requested.

## 4A. One Image Becomes A Movie

Purpose:

```text
Show that drift creates real temporal retinal input structure, not an abstract
trajectory variable.
```

Selected assets:

```text
declan/figure4_active_sensing_atlas/figures/panel_A/promotion_candidates/4A_candidate_3_real_high_contrast_positive.png
declan/figure4_active_sensing_atlas/figures/panel_A/A2_movie_transform_qc.png
declan/figure4_active_sensing_atlas/figures/panel_A/A4_backimage_pipeline_bridge.png
```

Current read:

```text
temporal contrast RMS: real 11.245, stabilized 0.000
motion power versus stabilized: real 1462.431, stabilized 0.000
movie power mean: real 15178.177, stabilized 15185.182
BackImage provenance: 256 images, 29 sessions, 151 drift-only trace sources,
canonical 756-unit V1 twin
```

Claim boundary:

```text
Panel A establishes the retinal-movie input and rendering QC. It does not by
itself establish information gain, behavioral utility, or optimality.
```

Selection note:

```text
The promoted single-panel A candidate is the real BackImage high-contrast
positive-alignment A1 variant. It uses a real BackImage canvas crop and recorded
eyepos trace; candidate 1 was closer to the original A1 proportions but was
centered on a dark patch.
```

## 4B. Motion Supports Feature Encoding

Purpose:

```text
Show how biological-like image motion changes feature-decodable V1-twin
responses when exact eye trajectory is known to the model.
```

Primary production source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1/
    incremental_staticmean_plus_motion_tworeadout_v2/
```

All-readout audit source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1/
    incremental_staticmean_plus_motion_allreadouts_v1/readout_atlas_figures/
```

Selected asset:

```text
declan/figure4_active_sensing_atlas/figures/panel_B/promotion_candidates/4B_candidate_3_power_rerun_absolute_gain.png
```

Corrected power-rerun main values:

```text
pyramid_local_field k16 delta_mean, n=384:
  empirical - static_mean:
    0.25x -0.65 CI [-2.17, +0.93]
    0.5x  +0.16 CI [-1.35, +1.77]
    1x    +1.76 CI [+0.58, +3.05]

  empirical - OU:
    0.25x -0.20 CI [-1.04, +0.77]
    0.5x  -1.29 CI [-2.14, -0.47]
    1x    -0.11 CI [-0.86, +0.67]

  empirical - Brownian:
    0.25x +0.09 CI [-0.74, +0.92]
    0.5x  -0.49 CI [-1.30, +0.27]
    1x    +0.66 CI [-0.06, +1.34]

  empirical - rotated:
    0.25x +0.55 CI [-0.05, +1.14]
    0.5x  +0.49 CI [-0.23, +1.30]
    1x    +0.59 CI [-0.07, +1.34]

temporal_pca secondary diagnostic:
  empirical - static_mean is negative at 0.25x, 0.5x, and 1x.
  empirical - OU remains strongly positive at 0.25x, 0.5x, and 1x.
```

All-readout/nested-alpha read:

```text
mean is the strongest absolute aggregate candidate under nested alpha.
delta_mean is the clearest static-subtracted motion-induced/local-pairing readout.
temporal PCA/DCT variants preserve trajectory order and separate empirical from OU.
OU is audit-pending, not yet a headline negative control.
```

Claim boundary:

```text
This is a deterministic, grouped-by-image feature-decoding proxy. It supports a
modest corrected aggregate delta-mean signal at natural scale and a stronger
local-pairing mechanistic signal elsewhere in the package. It should not be
framed as temporal-PCA motion adding feature information beyond a real static
baseline. The temporal-PCA/DCT result is now an order-sensitive relative
diagnostic showing that empirical temporal structure differs from OU-like
confined motion, pending the OU trace-control audit.
```

Production note:

```text
Panel B candidate 3 and the selected v5 composite have been redrawn from the
corrected n384 k16 static-mean posthoc. The aggregate claim should remain
modest and should not reuse the superseded temporal-PCA absolute-gain wording.
Before manuscript promotion, replace or supplement the single-readout B panel
with a Panel-B-style all-readout view and record an OU verdict.
```

Feature adjudication:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_feature_decomposition_adjudication_v6_staticmean_corrected_power_rerun_primary_scales/

delta_mean: score_with_joint_axis_term = 1.912, aggregate_score = 0.332,
  local_Iz_score = 2.139
temporal_pca: score_with_joint_axis_term = 0.608, aggregate_score = 0.593,
  local_Iz_score = 0.575
```

## 4C. Compact-Subspace Recovery

Purpose:

```text
Show that when eye position is latent, zeroed-eye feature recovery degrades as
motion scale grows, while compact-subspace feature recovery remains stable and
compact removal collapses recovery toward the zeroed-eye curve.
```

Primary source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_feature_posterior_compact_removed_pyramid_k8_n128_scales_0p5_1_2_v1/
    feature_compact_mechanism_summary.csv
```

Selected asset:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/promotion_candidates/4C_candidate_5_joint_feature_posterior_recovery.png
declan/figure4_active_sensing_atlas/figures/panel_C/promotion_candidates/feature_recovery_options/4C_option_6_compact_necessity_audit.png
```

Headline feature-posterior recovery:

```text
hard negatives n128, pyramid_local_field PCA k=8, compact k=10:
  zeroed-eye feature recovery cosine:
    0.5x = 0.765
    1.0x = 0.668
    2.0x = 0.576

  compact-only feature recovery cosine:
    0.5x = 0.850
    1.0x = 0.838
    2.0x = 0.826

  compact-removed feature recovery cosine:
    0.5x = 0.759
    1.0x = 0.635
    2.0x = 0.537

  full joint / compact-addback recovery cosine:
    0.5x = 0.872
    1.0x = 0.872
    2.0x = 0.871

  known-eye ceiling:
    0.5x = 0.927
    1.0x = 0.936
    2.0x = 0.949
```

Claim boundary:

```text
This is now an absolute feature-posterior endpoint rather than an image-identity
accuracy endpoint or a gain normalized to the moving zero-eye baseline. The
main read is that compact-only recovery retains much of the full joint feature
recovery, while compact-removed falls toward the zeroed-eye curve. This supports
compact-subspace necessity for this feature-posterior intervention, not a claim
that the animal computes this posterior, that compact structure is unique, or
that behavior optimizes this model objective.
```

Selection note:

```text
The promoted C panel now uses the feature-space compact-removal audit:
zeroed-eye, compact subspace, compact removed, and known-eye ceiling. Both
compact-source priors are retained in the result table/report; the main panel
plots their mean for the compact-only and compact-removed curves.
```

## 4D. Along-Edge Priors Improve Feature Recovery

Purpose:

```text
Show that the hidden-eye feature decoder recovers more feature signal from
along-edge than across-edge trajectory priors in the matched-static axis test.
```

Primary sources:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_matched_static_feature_posterior_gabor_pyramid_k4_8_uncertainty_v2/
  backimage_axis_conditioned_hard_negative_feature_posterior_gabor_pyramid_k4_8_uncertainty_v1/
  backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1/
  backimage_axis_conditioned_hard_negative_shared_source_gpu1_n64_c4_k16_v1/
```

Selected asset:

```text
declan/figure4_active_sensing_atlas/figures/panel_D/D2_axis_feature_recovery.png
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v5.png
```

Main values:

```text
matched-static 0.5x, pyramid_local_field k8 feature posterior:
  along-edge joint-zero feature gain = +6.052 [-MSE]
  across-edge joint-zero feature gain = +3.684 [-MSE]
  paired along-minus-across = +2.368, CI [+0.392, +4.589], p = 0.0257

matched-static image-identity observer:
  edge-parallel joint accuracy = 0.859
  edge-orthogonal joint accuracy = 0.828
  parallel-minus-orthogonal = +0.031

hard-negative guardrail:
  n64 feature posterior parallel-minus-orthogonal = -0.745
  CI = [-3.147, +1.631]
  n64 image-identity observer weakly favors orthogonal, 0.891 versus 0.844
```

Claim boundary:

```text
The promoted D story is a readout result: feature recovery is better along than
across local contours in the matched-static hidden-eye branch. The hard-negative
branch limits the axis claim, so D should not imply a universal edge-parallel
policy. Edge-parallel preservation of pixels and V1-twin responses remains
supporting mechanism evidence, not the main panel readout and not proof that
the animal optimizes the tested model objective.
```

Selection note:

```text
The promoted D panel should use the axis-conditioned feature-posterior readout:
absolute along/across gains over zero-eye plus the paired along-minus-across
contrast. The older preservation audit should move to caption or supplement.
```

## 4E. Real Drift Follows Coherent Edges

Purpose:

```text
Show that measured FEM-edge alignment strengthens when the local image
orientation is coherent.
```

Selected asset:

```text
declan/figure4_active_sensing_atlas/figures/panel_E/promotion_candidates/4E_candidate_3a_image_coherence_focus.png
```

Headline values:

```text
3A image-coherence focus:
  0.0-0.1 coherence bin mean cos2 = 0.059, n = 1415 windows
  0.6-0.7 coherence bin mean cos2 = 0.218, n = 631 windows
  0.7-0.8 coherence bin mean cos2 = 0.304, n = 579 windows
  0.9-1.0 coherence bin mean cos2 = 0.274, n = 101 windows

unweighted session mean cos2:
  all windows = 0.105, CI [0.067, 0.145]
  reliable axes = 0.140, CI [0.089, 0.188]
  high confidence = 0.269, CI [0.138, 0.396]
```

Claim boundary:

```text
Measured drift/fixation-cloud axes are modestly but reliably contour-following.
This is the behavioral bridge to the model geometry. It is not a causal
intervention and does not yet show that a V1-twin objective explains behavior
beyond raw edge geometry.
```

Selection note:

```text
The promoted single-panel E candidate is 3A, the focused
image-orientation-coherence trend. Candidate 3B remains a FEM-anisotropy
reliability check, candidate 3C remains a polar/rose directionality option, and
endpoint/null diagnostics stay available for caption or supplement guardrails.
```

## Optional 4F. Claim Summary

Use only if the main layout needs a sixth panel or if the supplement needs a
compact scope map.

```text
aggregate FEM: feature-decodable motion structure
local pairing: mechanistic sensitivity, not headline-stable
joint posterior: latent trajectory inference
behavior: contour-following geometry
caveats: raw edge baseline remains hard; OU/readout audit and write-lock pending
```

## Draft Figure Legend

Figure 4. Fixational eye movements create structured retinal movies whose
motion-dependent response changes are usable in V1-twin models and align with
measured image-contingent movement geometry. (A) A fixed screen image becomes a
moving retinal crop during fixation; stabilized controls remove temporal
contrast and motion power without changing mean movie power. (B) Corrected
static-mean aggregate readouts split the claim: mean/delta-mean support the
absolute feature-decoding question, while temporal PCA/DCT retain trajectory
ordering for empirical-vs-control diagnostics; OU remains audit-pending. (C) In the newer
feature-posterior joint model, marginalizing over latent eye position recovers
local Gabor/pyramid feature encoding above a zero-eye observer, without
requiring known eye position. (D) Local image geometry defines plausible motion
axes: in the matched-static hidden-eye feature decoder, along-edge trajectory
priors recover more feature signal than matched across-edge priors, while
hard-negative controls limit the claim to a scoped axis result. (E) Measured
free-viewing FEM/fixation-cloud alignment
with local edges rises when the local image orientation is coherent, with
anisotropy, polar, endpoint, and null diagnostics retained as guardrails. The
figure supports a convergence between useful motion geometry in the model and
measured behavioral geometry; it does not prove that animals optimize the
tested model objective.

## Companion Documents

The methods/logic companion set lives beside this file:

```text
companion_aggregate_fem_model.md
companion_local_Iz_pairing_model.md
companion_joint_posterior_observer_model.md
companion_behavior_geometry_bridge.md
```

These are written as explanatory argument documents in the style of the
inhomogeneous-stimuli methods note: hidden assumption, notation, estimator
contract, controls, evidence, diagnostics, and claim boundary.
