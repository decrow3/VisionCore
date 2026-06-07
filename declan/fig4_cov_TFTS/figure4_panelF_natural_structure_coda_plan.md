# Figure 4 Panel F: Narrow Natural-Image-Structure Coda

## Status

Proposed optional main-figure panel.

This plan replaces the current `Tangent R2 peaks near drift scale` Panel F idea with a narrower and stronger test:

> Do structured natural-image patches preferentially recruit the compact reafferent tangent geometry at the displacement scales occupied by fixational drift?

The default manuscript should remain Figure 4A-E. Panel F earns a main-figure slot only if the bounded analysis below lands cleanly. If it does not, cut F or move diagnostics to supplement.

## Why This Version

The original Jake/Rucci framing was that fixational eye movements may be matched to natural images in a way that goes beyond spectral power. Rucci-style analyses show that drift redistributes natural-image power into temporal modulations. Jake's extension was that the digital twin could ask whether FEMs also interact with higher-order natural-image structure: local phase alignment, edges, contours, and multiscale organization.

The current Figure 4A-E already makes the secure paper:

- recorded V1 shared variability is strongly reafferent,
- local retinal translations define image-specific response tangents,
- those tangents form a compact, image-generalizing geometry,
- and that geometry carries FEM-related information.

Panel F should not try to "land the paper." It should be a modest ecological coda that reconnects the compact reafferent-geometry result to the active-sensing literature. The safest version stays on the natural-image manifold and asks whether naturally occurring image structure modulates drift-scale recruitment of the tangent geometry.

## Why Not Current Panel F

The current Panel F concept is:

> Fisher-weighted local tangent prediction is strongest at small drift-scale displacements and declines toward microsaccade-scale displacements.

This is useful as a diagnostic, but weak as a main result:

- local tangents should generally work best locally;
- Fisher weighting can create an apparent scale peak that is not pure geometry;
- if the twin was trained on FEM-jittered input, best performance near drift scale can look partly circular;
- the panel does not directly test natural-image statistics.

Keep unweighted tangent-prediction quality as a supplement or inset, but do not make it the ecological claim.

## Primary Question

Among intact natural-image patches, do high-structure patches route a larger fraction of their translation-induced response change through the compact, image-disjoint tangent subspace than low-structure but energy-matched patches, especially within the small-displacement regime occupied by fixational drift?

This avoids making phase-scrambled images the primary comparison, because phase scrambles are partly off the model's natural training manifold. Phase scrambling remains useful as a secondary diagnostic for the "beyond power" claim.

## Claim Hierarchy

### Safe Claim

Structured natural-image patches route a larger fraction of translation-induced response change through the compact reafferent geometry at small, drift-scale displacements than less structured matched natural patches.

### Stronger Claim If Phase Control Lands

This drift-scale fractional routing depends on natural image organization beyond matched spectral power, because phase scrambling attenuates the effect.

### Claims To Avoid

- Real FEMs are optimal.
- Drift is tuned to natural-image phase structure.
- The deterministic twin proves perceptual benefit.
- Microsaccades are optimized chart resets.
- The phase-scramble result alone demonstrates ecology.

## Analysis Overview

Use the deterministic digital twin as a V1 response operator, not as a performance oracle.

For each natural image crop or stimulus-history object:

1. Compute local x/y retinal-translation tangents.
2. Project finite-displacement response differences onto the image-disjoint tangent basis from Panels C-E.
3. Sweep retinal displacement scale from near-zero to drift-scale to microsaccade-scale.
4. Stratify intact natural patches by local structure.
5. Ask whether high-structure patches route a larger fraction of their finite response change through the compact tangent subspace in the drift band than low-structure matched patches.
6. Use phase scrambling as a secondary diagnostic to test whether the effect is reduced when power is preserved but phase-defined structure is disrupted.

First-pass target:

> Test whether high-structure natural-image patches route a larger fraction of finite translation-induced response change through the image-disjoint compact tangent basis than matched low-structure natural patches, especially within the small-displacement regime occupied by drift, and whether this effect exceeds random/unit-shuffled bases.

## Recommended Primary Metric

Use an unweighted fractional structural metric first:

```text
tangent_subspace_fraction(s)
  = E_delta || P_B [ r(I, e0 + delta_s) - r(I, e0) ] ||^2
            / || r(I, e0 + delta_s) - r(I, e0) ||^2
```

where:

- `I` is an intact natural image crop or stimulus-history object;
- `delta_s` is a displacement at scale `s`;
- `r(I, e)` is the deterministic twin response;
- `B` is the image-disjoint compact tangent basis learned from training images;
- `P_B` is projection onto the tangent basis;
- `s` is displacement magnitude in arcmin.

This keeps the main result structural. It does not require a noise model, a decoder, or a perceptual-performance interpretation. It also avoids a major confound in a raw magnitude metric: high-structure patches have concentrated gradients, so they may simply produce larger response changes under translation. The fraction asks a more specific geometric question: of the response change that a patch produces, how much is routed through the compact shared tangent geometry established in Panels C-E?

Report the orthogonal complement explicitly:

```text
orthogonal_fraction(s) = 1 - tangent_subspace_fraction(s)
```

This turns Panel F into a partition of finite-displacement response change into compact-tangent versus outside-tangent components.

Secondary magnitude metric:

```text
tangent_subspace_sensitivity(s)
  = E_delta || P_B [ r(I, e0 + delta_s) - r(I, e0) ] ||^2 / s^2
```

Use this only as a supporting sensitivity analysis. A high-structure > low-structure difference in magnitude alone is not enough, because it may reduce to "edges drive larger translation responses than texture."

### Low-signal guard

The fractional metric can become unstable when the finite response change is tiny. The analysis must write and audit:

```text
delta_r_norm = || r(I, e0 + delta_s) - r(I, e0) ||^2
```

Required safeguards:

- define a fixed low-signal threshold before looking at group effects;
- flag low-signal rows at every image, condition, direction, and scale;
- exclude flagged rows from the primary fraction summary;
- report the excluded fraction by structure group and displacement scale;
- verify that the high-vs-low result is not driven by unequal low-signal exclusion rates;
- rerun a sensitivity check with a different reasonable threshold.

If low-structure patches are heavily excluded at small displacements, the main fractional result must be interpreted cautiously or moved to supplement.

Optional secondary metrics:

- Poisson-weighted Fisher-like tangent sensitivity, clearly labeled as a declared-noise analysis.
- Raw tangent-subspace magnitude, to quantify total response drive after the fractional result is established.
- Unweighted finite-displacement tangent R2, as a local-linearity diagnostic.

## Image-Structure Stratification

Rank natural patches by one or more structure measures. Keep this simple in the first pass.

Do not turn the first pass into a feature-engineering project. Pick one primary structure score and one backup.

Preferred first-pass measure:

1. Multiscale phase/edge structure, if already available or easy from Jake's image-selection pipeline.

Backup:

2. Gradient/edge energy, matched for RMS contrast and spatial-frequency power.

Other possible measures for later refinement:

- local phase congruency or cross-scale phase alignment;
- multiscale orientation agreement;
- middle-band energy or hotspot scores if already available from Jake's image-selection pipeline.

Construct two groups:

- `high_structure`: edge-rich, phase-aligned, multiscale-organized natural patches.
- `low_structure_matched`: lower-structure natural patches matched as much as possible for RMS contrast, mean luminance, local energy, and spatial-frequency power.

The matching matters. Without it, the result can collapse into "high contrast patches drive the model more."

## Secondary Phase-Scramble Diagnostic

Run only after the primary within-natural-image stratification is working.

For each intact crop, generate a spectrum-matched phase-scrambled control:

- preserve global Fourier amplitude spectrum;
- randomize phase;
- keep luminance normalization identical;
- use multiple seeds per crop if cheap.

If feasible and needed, add a stronger local/pyramid phase control:

- preserve local band energy more tightly;
- disrupt cross-scale phase alignment;
- avoid making this a large separate project unless the global phase result is promising.

Interpretation:

- If intact high-structure drift-scale tangent fraction exceeds phase controls, the result supports the "beyond spectral power" coda.
- If intact and phase-scrambled curves overlap, do not claim higher-order natural-image structure.
- If intact beats phase across all scales without a drift-band effect, claim natural-structure dependence but not drift-scale tuning.

## Movement / Displacement Sweep

Keep the movement manipulation simple.

Primary sweep:

- isotropic finite displacements at log-spaced scales;
- directions sampled uniformly;
- scales spanning:
  - near-zero/local derivative scale,
  - empirical drift-scale displacements over the analysis integration window,
  - larger microsaccade-like displacements.

Overlay empirical bands:

- drift displacement range from measured traces;
- microsaccade amplitude range if event definitions are reliable.

Do not start with five movement ensembles. Optional later controls:

- amplitude-scaled real drift fragments;
- Brownian/dither controls matched for displacement covariance;
- temporally shuffled real drift.

These should not be required for deciding whether F survives.

## Relation To Drift And Microsaccades

The intended interpretation is modest:

- drift samples within local response charts;
- microsaccade-scale displacements increasingly leave a single local chart;
- microsaccades may reposition the fovea between local neighborhoods, but this panel should not claim that without event-locked evidence.

Evidence needed for the drift claim:

- high-structure natural patches show elevated tangent-subspace fraction in the empirical drift band;
- the true tangent basis exceeds shuffled/random bases;
- the effect is robust over image identities.

Evidence not provided by this panel:

- optimization of the animal's eye movement policy;
- causal necessity of drift;
- microsaccade function.

## Nulls And Required Controls

Minimum controls:

1. Image-disjoint tangent basis.
   - Learn `B` on training images.
   - Evaluate all Panel F curves on held-out image identities.

2. Random-subspace and unit-shuffled bases.
   - The true tangent basis must capture a larger fraction of high-structure response change than null bases.

3. Energy/power matching across natural patch groups.
   - High- and low-structure patches should be matched or statistically adjusted for RMS contrast, local energy, and power spectrum summaries.

4. Image-bootstrap confidence intervals.
   - Bootstrap image identities, not just displacement samples.

5. Unweighted metric first.
   - If the effect appears only under Poisson/Fisher weighting, label it as noise-model-dependent and do not use as the main ecological panel.

Secondary control:

6. Phase scramble.
   - Use as a diagnostic for the "beyond power" claim, not as the primary evidence for ecology.

## Pass / Fail Criteria

Panel F is main-figure worthy only if all of the following hold:

1. High-structure natural patches show a greater tangent-subspace fraction than low-structure matched natural patches.
2. The high-minus-low fractional difference is strong in the empirical drift-displacement band and declines, flattens, or becomes less basis-specific at larger microsaccade-like displacements.
3. The effect is present for the unweighted fractional metric, not only for raw response magnitude or Poisson/Fisher weighting.
4. The true image-disjoint tangent basis exceeds unit-shuffled/random-subspace nulls.
5. Confidence intervals over image identities support the effect.
6. The caption-safe claim is modest and does not imply optimality.

A literal peak at drift scale is not required. Because a local tangent geometry may be strongest at the smallest tested displacements, an effect that is high across the small-displacement regime and remains strong through the drift band can still support the safer coda. A uniform high > low difference across all scales is not enough for a drift-scale claim. It may support only the weaker statement that structured natural images recruit the compact tangent geometry more than matched low-structure patches.

Outcome categories:

- Main-figure-worthy: high-structure excess is strong in the drift/small-displacement regime, exceeds nulls, and is reduced or less basis-specific at microsaccade-like scales.
- Supplement-worthy: high-structure excess is present across all small scales, including drift, but is not uniquely drift-peaked or scale-specific.
- Cut: high- and low-structure matched patches do not differ, the true basis does not beat null bases, or the result appears only in raw magnitude/Fisher-weighted metrics.

If any of these fail:

- cut Panel F from the main figure, or
- move the result to supplement/future work.

## Suggested Figure Design

Primary panel:

- x-axis: displacement scale in arcmin, log-scaled.
- y-axis: fraction of finite-displacement response change captured by the image-disjoint tangent basis.
- curves:
  - high-structure natural patches;
  - low-structure matched natural patches;
  - random-basis or unit-shuffle null.
- overlays:
  - empirical drift band;
  - empirical microsaccade band, if reliable.

Optional inset or supplement:

- high-minus-low fractional difference curve, with zero line and image-bootstrap CI;
- intact natural vs Fourier phase-scrambled control;
- raw tangent-subspace magnitude, to show whether fraction and magnitude tell the same story;
- unweighted tangent R2 vs displacement scale.

Candidate title:

```text
Natural-image structure shapes drift-scale reafference
```

More conservative title:

```text
Structured natural patches recruit the compact tangent geometry
```

Avoid:

```text
Tangent R2 peaks near drift scale
Natural image structure selects the optimal FEM scale
```

## Suggested Results Wording If It Works

> Finally, we asked whether this compact reafferent geometry was recruited differently by the structure of natural images. We stratified held-out natural-image patches by local phase/edge structure while matching contrast and spectral energy, then swept counterfactual retinal displacement scale in the digital twin. For high-structure patches, a larger fraction of the finite translation-induced response change fell within the image-disjoint tangent subspace than for matched low-structure patches. This high-minus-low difference was strong across the small-displacement regime occupied by drift, exceeded shuffled-basis controls, and became reduced or less basis-specific at larger microsaccade-like displacements. Spectrum-matched phase scrambling attenuated this small-displacement recruitment, suggesting that the FEM-linked geometry depends on natural image organization beyond power alone.

Conservative last sentence:

> Thus, while the deterministic twin does not establish optimal eye movement behavior, it shows that natural image structure can selectively route drift-scale retinal translations through the compact V1 reafferent geometry.

## Suggested Wording If It Does Not Work

> We also tested whether local natural-image structure modulated recruitment of the compact tangent geometry across displacement scale. This analysis did not yield a robust image-level effect after matching contrast/energy and comparing against shuffled-basis controls. We therefore do not use the model to claim that drift is tuned to higher-order natural-image structure; the main conclusion remains the compact, image-generalizing reafferent geometry established in Panels B-E.

## Remote Repo Anchors

The text bundle suggests these existing pieces are relevant:

- `scripts/spatial_info.py`
  - likely source for `make_counterfactual_stim`, `get_spatial_readout`, and `compute_rate_map_batched`.

- `scripts/natimg_digitaltwin_spatialinfo_declan.py`
  - possible starting point for natural-image response/information computations.

- `scripts/check_fixrsvp_model_fisherinfo.py`
  - useful reference for Fisher-like computations, but avoid making Fisher the primary metric.

- `scripts/fourier_shift.py`
  - likely useful for controlled image translations.

- `scripts/example_pyramid_simulator.py` and `scripts/figure_fixrsvp_pyramid_simulation.py`
  - possible references for pyramid/local image controls.

- Older transformation-dynamics code in the bundle includes:
  - `run_model_trial`,
  - `compute_delta_r_for_stimulus`,
  - `phase_randomize_velocity`.

Do not revive old decoder-accuracy or E-optotype performance branches for this panel.

## Expected Output Files

If implemented, the analysis should write:

```text
panelF_natural_structure_scale_sweep.csv
panelF_image_structure_metrics.csv
panelF_empirical_event_ranges.csv
panelF_phase_scramble_diagnostic.csv
panelF_manifest.json
```

Minimum CSV columns:

```text
image_id
split
structure_group
image_condition
displacement_arcmin
metric_name
metric_value
basis_type
bootstrap_id_or_fold
```

Primary `metric_name` values should include:

```text
tangent_subspace_fraction
orthogonal_fraction
high_minus_low_fraction
raw_tangent_subspace_sensitivity
unweighted_tangent_R2
```

Manifest should record:

- model checkpoint / readout;
- image source and crop IDs;
- structure metric definitions;
- matching variables;
- displacement scales and directions;
- tangent-basis source and train/test split;
- null-basis construction;
- phase-scramble seeds and normalization;
- empirical drift/microsaccade range definitions.

## Bottom Line

Run this only as a bounded coda. The paper is already strong with Figure 4A-E. The value of this Panel F is to preserve Jake's original active-sensing/natural-image-statistics intuition in a scientifically safer form:

> not "real FEMs are optimal," but "natural image structure recruits the compact reafferent geometry at drift scale."

If that sentence is not directly supported by the data, cut F.
