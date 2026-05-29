# Jacobian-field and identity-vector smoothness diagnostics

## Purpose

This handoff specifies the next analysis needed before locking the separability/path-integration manuscript framing.

The current E-optotype results support a scale-dependent, magnitude-driven active-sensing story:

- At fine scale (`LogMAR = -0.20`), FEM-sampled phases expose substantially larger pairwise E-orientation identity differences than the stabilized center.
- These identity signals vary in direction across phase and cancel under a signed linear path-sum readout.
- A nonlinear / energy-like readout recovers large gains from FEM-sampled phases.
- At coarse scale (`LogMAR = +0.20`), the stabilized center already carries identity signal and the FEM advantage disappears.

However, we have **not yet directly measured whether the local Jacobian field itself is smooth or rough across retinal phase**.

The step-prediction result showed local differentiability:

```text
r(p + Δp) - r(p) ≈ J(p_mid) Δp
```

for drift-sized steps. But this does **not** imply that `J(p)` itself remains similar as `p` varies across the drift cloud.

This diagnostic should measure:

1. the spatial decorrelation length of the Jacobian field `J(p)`;
2. the spatial decorrelation length of identity vectors `d_ab(p)`;
3. whether decorrelation is anisotropic across x and y phase axes;
4. whether the fine-scale cancellation result reflects rough translation geometry, rotating identity vectors, or an inappropriate center-defined reference axis.

---

# Scientific questions

## Q1. How smooth is the Jacobian field?

For each retinal phase `p`, define the local translation Jacobian:

```text
J_a(p) = [dr/dx, dr/dy]
```

Question:

```text
How far can the image move before span(J(p)) substantially changes?
```

The answer is the field correlation/decorrelation length:

```text
ell_J
```

This should be reported in arcmin and compared to:

```text
drift step RMS
drift step p90
drift cloud RMS / radius
stimulus scale / LogMAR
```

## Q2. Is field smoothness anisotropic?

The decorrelation length may differ along horizontal and vertical phase displacement.

For E optotypes, this matters because moving along a stroke and moving across a stroke are not equivalent.

Measure:

```text
ell_J_radial
ell_J_x
ell_J_y
```

where:

- `ell_J_radial` uses Euclidean phase distance;
- `ell_J_x` uses approximately pure x-separated pairs;
- `ell_J_y` uses approximately pure y-separated pairs.

## Q3. Do identity vectors rotate across phase?

For each identity pair `(a,b)`:

```text
d_ab(p) = r_b(p) - r_a(p)
```

Question:

```text
Is d_ab(p) stable across phase, or does the identity direction itself rotate/reverse?
```

This distinguishes:

- stable identity vector but poor center reference axis;
- genuinely phase-varying identity geometry;
- identity geometry that varies only far from center;
- identity geometry that varies differently at fine and coarse scales.

## Q4. Is cancellation caused by rough J, rotating d_ab, or both?

The fine-scale signed readout cancellation could arise from:

1. rough/rapidly varying `J(p)`;
2. stable `J(p)` but rotating `d_ab(p)`;
3. smooth local patches but integration across too large a phase cloud;
4. bad center-defined identity reference axis.

This analysis should resolve which case applies.

---

# Pre-registered interpretation regimes

The main result is a quantitative decorrelation length, not just a categorical case label.

Use these regimes to interpret `ell_J`.

## Regime 1: trivially smooth field

```text
ell_J >> drift cloud radius, e.g. > 10 arcmin
```

Interpretation:

```text
The Jacobian field is effectively one tangent geometry over the sampled cloud. The field framework adds less explanatory value; center-vs-drift identity magnitude remains the main result.
```

## Regime 2: cloud-scale smoothness

```text
ell_J ~ 2-10 arcmin
```

Interpretation:

```text
The field varies over the cloud but remains coherent over substantial local neighborhoods. Linear integration may work over short patches, but not across the whole fixation cloud.
```

This is compatible with complex-cell-like pooling and still supports a population-geometric description.

## Regime 3: step-to-local-patch smoothness

```text
ell_J ~ 0.5-2 arcmin
```

Interpretation:

```text
The field is smooth over individual drift steps but changes meaningfully over short drift trajectories. This strongly supports the local-field framework: FEMs traverse multiple local geometric regimes.
```

## Regime 4: sub-step roughness

```text
ell_J < 0.5 arcmin
```

Interpretation:

```text
The field changes at or below drift-step scale. This would be surprising under simple complex-cell pooling and would imply very fine population-geometric structure.
```

---

# Pre-registered model predictions

These predictions should be included in the README before interpreting the results.

## Pure complex-cell pooling expectation

If the result mostly reflects classical complex-cell-like spatial pooling:

```text
ell_J should be on the order of the model's effective pooling/subunit scale, plausibly ~1-3 arcmin.
ell_J may be moderately similar across LogMARs because pooling scale is fixed by the model.
The dominant principal angle may remain stable while the secondary direction decorrelates faster.
Direction-specific decorrelation (x vs y) may depend on stroke orientation.
```

## Curved-field / fine-feature expectation

If the population field is shaped by fine stimulus structure beyond a simple fixed pooling scale:

```text
ell_J should shrink at finer LogMAR.
LogMAR -0.20 should have a shorter ell_J than LogMAR +0.20.
Identity-vector decorrelation should also be stronger at LogMAR -0.20.
```

## Trivially smooth expectation

If the local-field framing is overemphasized:

```text
ell_J should exceed the drift cloud radius at both LogMARs.
J alignment should remain high across all tested phase distances.
Cancellation must then come from identity-vector behavior or readout/reference-axis choices, not field roughness.
```

---

# Analysis 1. Jacobian-field decorrelation

## Phase grid

Use a **regular spatial phase grid**, not only actual eye-trace samples.

The primary analysis should be in Euclidean retinal coordinates, not trajectory time.

### Grid recommendation

Pilot first on one condition:

```text
LogMAR = -0.20
orientation = 0
```

Recommended grid:

```text
dense core: spacing 0.25 arcmin over ±3 arcmin
outer grid: spacing 0.5-1.0 arcmin over ±10 arcmin
```

or another compute-feasible design that gives dense coverage at small distances.

Important:

```text
The most important region is 0-1 arcmin, because this brackets drift-step scale.
```

If compute is limited, prioritize dense small-distance sampling over wide cloud coverage.

Record grid details in `run_config.json`:

```text
grid_core_radius_arcmin
grid_core_spacing_arcmin
grid_outer_radius_arcmin
grid_outer_spacing_arcmin
n_phase_points
n_forward_positions
finite_difference_step_px
finite_difference_step_arcmin
```

## Jacobian computation

For each phase `p` and source orientation/image `a`:

```text
J_a(p) = [dr/dx, dr/dy]
```

Use the same finite-difference convention as the step-Jacobian analysis.

## Subspace alignment metric

For two Jacobians `J_i`, `J_j`, define projectors onto their column spaces:

```text
P_i = Q_i Q_i.T
P_j = Q_j Q_j.T
```

where `Q_i`, `Q_j` are orthonormal bases after rank trimming.

Primary metric:

```text
J_alignment_mean = trace(P_i P_j) / k
```

where `k` is the effective shared subspace dimension, normally `k=2`.

For the 2D full-rank case:

```text
J_alignment_mean = (cos²(theta_1) + cos²(theta_2)) / 2
```

This is bounded in `[0, 1]`.

Also report the individual principal-angle cosines:

```text
J_principal_cos_1
J_principal_cos_2
J_principal_angle_1_deg
J_principal_angle_2_deg
```

Rationale:

```text
The mean alignment can hide asymmetric behavior where one translation direction is stable and the other rotates. Individual principal angles are required for interpretation.
```

## Pairwise distances

For each phase pair:

```text
delta_x_arcmin
delta_y_arcmin
phase_distance_arcmin = sqrt(delta_x_arcmin² + delta_y_arcmin²)
```

Also classify the pair orientation:

```text
separation_axis = radial | x_axis | y_axis | diagonal
```

Suggested classification:

```text
x_axis: abs(delta_y) <= grid_spacing / 2
y_axis: abs(delta_x) <= grid_spacing / 2
radial: all pairs
diagonal: neither x_axis nor y_axis
```

## Direction-specific decorrelation

Compute separate alignment curves for:

```text
radial distance
x-axis separation
y-axis separation
```

Report:

```text
ell_J_radial_0p9
ell_J_radial_0p75
ell_J_radial_0p5

ell_J_x_0p9
ell_J_x_0p75
ell_J_x_0p5

ell_J_y_0p9
ell_J_y_0p75
ell_J_y_0p5
```

This tests anisotropy of field smoothness.

---

# Analysis 2. Identity-vector decorrelation

For each identity pair `(a,b)`:

```text
d_ab(p) = r_b(p) - r_a(p)
```

Compute phase-pair cosines:

```text
identity_cosine = cos(d_ab(p_i), d_ab(p_j))
identity_abs_cosine = abs(identity_cosine)
identity_inner_product = d_ab(p_i).T @ d_ab(p_j)
identity_norm_i = ||d_ab(p_i)||
identity_norm_j = ||d_ab(p_j)||
```

Also compute the translation-orthogonal version:

```text
f_perp_ab(p) = P_perp_a(p) d_ab(p)
perp_identity_cosine = cos(f_perp_ab(p_i), f_perp_ab(p_j))
perp_identity_abs_cosine = abs(perp_identity_cosine)
perp_identity_inner_product = f_perp_ab(p_i).T @ f_perp_ab(p_j)
```

## Region stratification

Separate phase pairs by region:

```text
near_center
off_center
cross_center_offcenter
```

Suggested definitions:

```text
near_center: both phases within 1 arcmin of center
off_center: both phases outside 1 arcmin of center
cross_center_offcenter: one near, one off-center
```

Record threshold in `run_config.json`.

Rationale:

```text
Cosines between near-zero identity vectors can be numerically unstable. Center-phase identity differences are weak at fine LogMAR, so near-center and off-center phase pairs should be interpreted separately.
```

Downweight or flag pairs where:

```text
identity_norm_i * identity_norm_j < norm_product_min_threshold
```

Output both normalized cosines and raw inner products.

---

# Outputs

## Directory

Use:

```text
outputs/stats/eoptotype_jacobian_field_smoothness/
```

## Required files

```text
run_config.json
README.md

jacobian_field_alignment_by_phase_pair.csv
jacobian_field_decorrelation_summary.csv

identity_vector_alignment_by_phase_pair.csv
identity_vector_decorrelation_summary.csv

field_smoothness_interpretation_summary.csv
```

## Required figures

```text
figures/jacobian_field_alignment_vs_phase_distance_radial.png
figures/jacobian_field_alignment_vs_delta_x.png
figures/jacobian_field_alignment_vs_delta_y.png
figures/jacobian_field_decorrelation_length_by_logmar.png

figures/jacobian_principal_cosines_vs_phase_distance.png

figures/identity_vector_cosine_vs_phase_distance.png
figures/perp_identity_vector_cosine_vs_phase_distance.png
figures/identity_vector_cosine_by_region.png

figures/field_vs_identity_decorrelation_summary.png
```

---

# CSV schema: jacobian_field_alignment_by_phase_pair.csv

Columns:

```text
logmar
orientation
phase_i_x_arcmin
phase_i_y_arcmin
phase_j_x_arcmin
phase_j_y_arcmin
delta_x_arcmin
delta_y_arcmin
phase_distance_arcmin
separation_axis

rank_i
rank_j
J_alignment_mean
J_principal_cos_1
J_principal_cos_2
J_principal_angle_1_deg
J_principal_angle_2_deg
J_col_x_cosine
J_col_y_cosine
```

Notes:

```text
J_col_x_cosine and J_col_y_cosine are column-wise cosines and should be interpreted cautiously because the 2D subspace can rotate internally.
```

---

# CSV schema: jacobian_field_decorrelation_summary.csv

Columns:

```text
logmar
orientation
separation_axis
n_phase_pairs

median_alignment_0_0p25_arcmin
median_alignment_0p25_0p5_arcmin
median_alignment_0p5_1_arcmin
median_alignment_1_2_arcmin
median_alignment_2_4_arcmin
median_alignment_4_8_arcmin

ell_J_0p9_arcmin
ell_J_0p75_arcmin
ell_J_0p5_arcmin

median_principal_cos_1_0p5_1_arcmin
median_principal_cos_2_0p5_1_arcmin

drift_step_rms_arcmin
drift_step_p90_arcmin
drift_cloud_rms_arcmin
```

If the alignment never crosses a threshold, record:

```text
above_range
```

or use `NaN` plus a status column:

```text
ell_J_0p75_status = below_min | resolved | above_max
```

---

# CSV schema: identity_vector_alignment_by_phase_pair.csv

Columns:

```text
logmar
source_orientation
target_orientation
path_mode_or_phase_set
region_pair_type

phase_i_x_arcmin
phase_i_y_arcmin
phase_j_x_arcmin
phase_j_y_arcmin
delta_x_arcmin
delta_y_arcmin
phase_distance_arcmin
separation_axis

identity_norm_i
identity_norm_j
identity_norm_product
identity_cosine
identity_abs_cosine
identity_inner_product

perp_identity_norm_i
perp_identity_norm_j
perp_identity_norm_product
perp_identity_cosine
perp_identity_abs_cosine
perp_identity_inner_product

norm_product_flag
```

---

# CSV schema: identity_vector_decorrelation_summary.csv

Columns:

```text
logmar
source_orientation
target_orientation
path_mode_or_phase_set
region_pair_type
separation_axis
n_phase_pairs

median_identity_cosine
median_identity_abs_cosine
fraction_negative_identity_cosine
median_identity_inner_product

median_perp_identity_cosine
median_perp_identity_abs_cosine
fraction_negative_perp_identity_cosine
median_perp_identity_inner_product

ell_identity_cosine_0p75_arcmin
ell_identity_cosine_0p5_arcmin
ell_perp_identity_cosine_0p75_arcmin
ell_perp_identity_cosine_0p5_arcmin
```

---

# Interpretation matrix

The README should contain an explicit interpretation section.

## Case A: smooth J, stable identity vectors

```text
J alignment high across drift cloud
identity vector cosine high across drift cloud
```

Interpretation:

```text
Cancellation likely reflects a poor reference axis or a readout issue, not genuinely phase-varying geometry.
```

## Case B: smooth J, rotating identity vectors

```text
J alignment high
identity vector cosine decays or changes sign
```

Interpretation:

```text
Translation geometry is smooth, but identity geometry is phase-dependent.
```

## Case C: rough J, rotating identity vectors

```text
J alignment decays over drift-scale or cloud-scale distances
identity vector cosine also decays
```

Interpretation:

```text
Curved local-field framework is strongly supported. FEMs traverse multiple local geometric regimes.
```

## Case D: J smooth over individual steps but not over full cloud

```text
ell_J > drift_step_RMS
ell_J < drift_cloud_RMS
```

Interpretation:

```text
Local step prediction and long-window cancellation are reconciled. Linear accumulation may work within local smoothness patches, while full-trajectory signed sums cancel.
```

Case D is likely and should be treated as an important positive outcome, not a compromise.

---

# Readout implications

The current signed-sum readout is order-invariant and cancels for fine-scale E-optotype phases.

Do not conclude that trajectory order is biologically irrelevant.

Preferred statement:

```text
Under the integrated-sum readout used here, trajectory order is invisible by construction. Whether order matters under biologically plausible readouts depends on the field decorrelation length and the downstream integration time constant.
```

If `ell_J` is between drift-step RMS and drift-cloud RMS, then a leaky/windowed integration analysis becomes a principled follow-up.

---

# Pilot run

Before full run, do one pilot:

```text
LogMAR = -0.20
orientation = 0
grid = dense enough to resolve 0-1 arcmin distances
```

Check:

```text
compute time
n_phase_points
n_phase_pairs
alignment curve sanity
whether alignment starts near 1 at small distances
```

Then scale to:

```text
LogMARs: -0.20, +0.20
Orientations: 0, 90, 180, 270
```

---

# Manuscript consequence

Do not lock final language until this diagnostic is reviewed.

Current provisional claim:

```text
At fine scale, FEM-sampled phases expose larger identity differences than the stabilized center, and an energy-like readout can recover this signal.
```

After this diagnostic, refine whether the mechanism is:

```text
1. rough Jacobian-field geometry;
2. smooth translation geometry but rotating identity vectors;
3. multiple smoothness patches within a curved field;
4. a reference-axis/readout issue.
```
