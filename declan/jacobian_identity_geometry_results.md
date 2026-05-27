# Jacobian Identity/Transformation Geometry: Initial Results Note

Compiled from the current implementation outputs in:

- `declan/results/translation_mimicry_primary/`
- `declan/results/phase_landscape_coarse/`
- `declan/results/jacobian_identity_geometry_final/`

This note stays within the deterministic encoding-geometry framing. It does **not** make noisy-observer, behavioral, or V2 claims.

## 1. What is now implemented

The following implementation targets from the handoff are now in place:

- `declan/geometry_utils.py`
- `declan/translation_mimicry.py`
- `declan/phase_landscape.py`
- `declan/figure_jacobian_identity_geometry.py`

Supporting Jacobian bundles were extended to the primary LogMAR sweep:

- `declan/jacobian_results/test3_lm-0.25.npz`
- `declan/jacobian_results/test3_lm-0.30.npz`
- `declan/jacobian_results/test3_lm-0.35.npz`

The primary analyses were run for:

- real vs trial-mean stabilized conditions
- LogMARs `-0.20, -0.25, -0.30, -0.35`
- `-0.40` retained only as a plateau / saturation control

## 2. Translation mimicry: headline pattern

The main mimicry output is `translation_mimicry_overview.csv` in raw-rate space.

Mean pairwise projection-based mimicry (`mimicry_unconstrained`, raw space):

| LogMAR | real | stabilized |
|---|---:|---:|
| -0.20 | 0.403 | 0.309 |
| -0.25 | 0.361 | 0.246 |
| -0.30 | 0.318 | 0.343 |
| -0.35 | 0.376 | 0.488 |
| -0.40 | 0.398 | 0.484 |

### Interpretation

1. At `-0.20` and `-0.25`, identity differences are more translation-confusable in the real condition than in the trial-mean stabilized condition.
2. Around `-0.30`, the two conditions are similar.
3. By `-0.35` and `-0.40`, the stabilized condition shows larger mimicry than the real condition.

This means mimicry is **not** a simple monotone proxy for the old `alpha` result or for the real-vs-stabilized decoding crossover. That is scientifically reasonable rather than pathological:

- `alpha` is a covariance / signal-subspace overlap quantity.
- mimicry is a pairwise class-mean / orientation-specific Jacobian quantity.

The two track related but not identical geometric objects.

### Numerical stability

The projection-vs-least-squares gap is negligible throughout the primary range:

- typically `1e-6` to `1e-5`
- only rising modestly at `-0.40`

So, in the current sweep, the orthonormal projection metric and ridge-LS interpretation are effectively consistent.

### Pairwise structure is important

The mean mimicry values do hide meaningful orientation-pair structure.

Examples from `translation_mimicry_pairwise_summary.csv` in raw space:

- real, `-0.20`: `0↔180 = 0.659`, `0↔270 = 0.697`, `0↔90 = 0.416`, `90↔180 = 0.266`
- real, `-0.30`: `0↔180 = 0.686`, `0↔90 = 0.473`, `0↔270 = 0.540`, `90↔180 = 0.053`
- stabilized, `-0.35`: `0↔180 = 0.637`, `0↔90 = 0.643`, `0↔270 = 0.652`, `90↔180 = 0.452`

Two robust observations follow.

1. The pairwise geometry is anisotropic: some orientation differences lie much more strongly in the translation tangent space than others.
2. The especially small `90↔270` and, in some regimes, `90↔180` values show that mimicry is not just tracking a single global nuisance magnitude. It is resolving pair-specific identity/transformation structure.

So the pairwise matrices are not just a visualization detail; they contain interpretable structure beyond the means.

### Raw vs normalized sensitivity

The script now saves both raw-rate and neuron-wise z-scored results, and produces an explicit comparison figure:

- `translation_mimicry_raw_vs_zscore.png`

The qualitative pairwise structure is largely preserved after z-scoring:

- high-mimicry pairs remain high
- near-zero pairs remain near-zero
- condition crossover structure remains visible

But magnitudes do shift. This means Euclidean geometry is not being driven purely by a tiny set of high-rate units, while still confirming that normalization choice changes the exact scale of the effect. That is the right sensitivity outcome.

## 3. Translation mimicry vs signal alignment

From `translation_mimicry_overview.csv` (raw space):

| LogMAR | alpha pooled real | alpha pooled stabilized |
|---|---:|---:|
| -0.20 | 0.628 | 0.514 |
| -0.25 | 0.652 | 0.464 |
| -0.30 | 0.475 | 0.535 |
| -0.35 | 0.546 | 0.702 |
| -0.40 | 0.537 | 0.681 |

This mirrors the mimicry condition crossover:

- real > stabilized at `-0.20/-0.25`
- stabilized > real by `-0.35/-0.40`

So the new pairwise mimicry analysis is coherent with the older alignment picture, but the mimicry metric sharpens it by asking a more specific question:

> How much of a concrete identity-difference vector lies in the translation tangent space of a particular source identity?

That is the right encoding-side question for identity/transformation confusability.

## 4. Phase landscape: headline pattern

The coarse landscape run used:

- LogMARs `-0.20, -0.35, -0.40`
- grid `17 x 17`
- range `+-3` model pixels

At the fixed-center grid point (center of the sampled landscape):

| Metric | -0.20 | -0.35 | -0.40 |
|---|---:|---:|---:|
| mean pairwise separation | 0.0720 | 0.0615 | 0.0637 |
| alpha pooled | 0.5157 | 0.6949 | 0.6435 |
| mean mimicry | 0.2864 | 0.2455 | 0.2602 |

Across the full phase grid:

- mean pairwise separation ranges from `0.046` to `0.517`
- alpha pooled ranges from `0.298` to `0.990`
- mean mimicry ranges from `0.079` to `0.818`

### Interpretation

1. The deterministic geometry varies strongly over phase.
2. Fixed center is just one point in a broad landscape, not a privileged summary of the regime.
3. The phase dependence is large enough that trial-mean stabilized clouds and real FEM position clouds should be interpreted relative to a heterogeneous map, not a single oracle point.

This is exactly what the phase-landscape analysis was meant to establish.

### Fine 33 x 33 pass at the two key regimes

The requested fine pass is now complete in `declan/results/phase_landscape_fine/` for `-0.20` and `-0.35`.

At the center point:

| Metric | -0.20 | -0.35 |
|---|---:|---:|
| mean pairwise separation | 0.0720 | 0.0615 |
| alpha pooled | 0.5282 | 0.6297 |
| mean mimicry | 0.2313 | 0.2503 |
| max mimicry | 0.6069 | 0.5965 |

Across the full fine grid:

- `-0.20`: mean pairwise separation `0.0498` to `0.5171`, alpha pooled `0.3469` to `0.9760`, mean mimicry `0.0908` to `0.7394`
- `-0.35`: mean pairwise separation `0.0522` to `0.3954`, alpha pooled `0.4908` to `0.9868`, mean mimicry `0.1233` to `0.8724`

The important consequence is that the fine pass does not overturn the coarse interpretation. Instead it sharpens it:

1. The fixed-center point remains relatively modest in mimicry even in the two most interpretable regimes.
2. Very high-mimicry pockets still exist nearby in phase space.
3. The geometry therefore depends strongly on local retinal phase, not just on LogMAR alone.

That is the figure-relevant conclusion: the deterministic geometry is spatially heterogeneous, and the center point should be read as an anchor rather than as a sufficient summary.

## 5. What the current results support

The current implementation supports the following deterministic-geometry claims:

1. The image-translation Jacobian provides a concrete local tangent-space model for same-identity-under-translation response variation.
2. Identity/transformation confusability can be quantified pairwise via translation mimicry, and this confusability depends on LogMAR and condition.
3. The phase landscape is highly non-uniform: class separation, signal/tangent alignment, and mimicry all vary across subpixel phase.
4. The fixed-center point should be treated as a deterministic anchor inside that landscape, not as a biological baseline.

## 6. What remains incomplete or provisional

1. The central figure script has now been restyled into a cleaner figure-driven version, but the empirical bridge panel is still a fallback explanatory panel rather than a matched projection result.
2. The fine phase pass (`33 x 33`) for `-0.20` and `-0.35` is currently running; the coarse landscape is complete and interpretable.
3. FEM component decomposition has not been implemented yet and remains secondary.
4. The empirical bridge panel remains a fallback explanatory panel because there is still no matched real-data/model-neuron projection for this E-optotype analysis.

## 7. Practical next step for the write-up

The cleanest immediate narrative is:

> In the deterministic digital twin, retinal translation moves activity along image-specific tangent planes identified by the image-translation Jacobian. Pairwise translation mimicry shows when identity differences can be locally explained as translations of a source identity, while the phase landscape shows that this confusability depends strongly on retinal phase. Together these results formalize identity/transformation confusability as a geometric property of the V1 population code rather than as generic noise.

That is now supported by implemented analyses rather than only by the handoff plan.