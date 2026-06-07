# Handoff: Tangent-subspace information analysis for Figure 4E

## Working goal

Implement a new analysis that tests whether the compact, image-generalizing reafferent tangent basis carries the spatial information gained from real fixational eye movements (FEMs) in the canonical digital twin.

This is intended as the working Figure 4E payoff panel:

> The compact reafferent tangent subspace captures the spatial information gained from real fixational eye movements.

This analysis should connect two existing code/results streams:

1. `jake/twininfo`: natural-image retinal movies, real versus stabilized eye traces, model lag-cube responses, Poisson Fisher information, spatial single-spike information, phase/spectral controls, and gain summaries.
2. `declan/twin_feature_tangent_structure` plus `outputs/twin_feature_tangent_structure_prod_limited_synth`: compact translation-tangent basis from the canonical 756-cell twin population, including union spectra and image-disjoint train/test basis results.

The result should not be framed as behavioral performance, generic acuity improvement, or E-optotype discrimination. It should be framed as a model-observer analysis of spatial/pose information in the twin.

---

## Scientific question

The empirical paper shows that measured FEMs produce a structured, low-dimensional, stimulus-aligned covariance component in foveal V1. The tangent analysis shows that, in the canonical twin, image-specific retinal translations do not share a universal signed x/y displacement axis, but their tangent vectors occupy a compact, image-generalizing population subspace.

Panel E asks the next question:

> Are the population directions that appear as FEM-linked shared variability also the directions through which active retinal sampling exposes spatial displacement information?

The desired home-run result is:

1. Real FEM retinal movies have greater spatial information than stabilized retinal movies under a specified twin/Poisson observer.
2. Most of that real-FEM information gain is captured by the compact tangent basis.
3. The orthogonal complement and dimensionality-matched null bases capture much less.

This would convert the interpretation from “FEMs create structured noise correlations” to “FEM-linked covariance is signal in the wrong coordinate frame for a pose-blind analysis.”

---

## High-level implementation plan

Create a new standalone runner rather than overloading the existing `jake.twininfo.pipeline`.

Recommended location:

```text
jake/twininfo/run_tangent_subspace_information.py
```

Alternative if figure-production scripts are centralized:

```text
scripts/figure4/run_tangent_subspace_information_panelE.py
```

Recommended output root:

```text
outputs/tangent_subspace_information/<run_name>/
```

The analysis should reuse existing `jake/twininfo` functions wherever possible, especially for loading model/data, rendering retinal lag cubes, computing shifted rate maps, selecting populations, and computing spatial information.

---

## Existing code to reuse

### From `jake/twininfo`

Use these modules as the primary code base:

```text
jake/twininfo/common.py
jake/twininfo/trace_selection.py
jake/twininfo/image_selection.py
jake/twininfo/retinal_examples.py
jake/twininfo/lagcube_information.py
jake/twininfo/information.py
jake/twininfo/pipeline.py
```

Relevant existing concepts/functions from the README and code map:

- `trace_selection.py`: select real fixation and microsaccade windows.
- `image_selection.py`: choose natural images/crops and generate natural-image controls.
- `retinal_examples.py`: render exact retinal crops and control movies.
- `lagcube_information.py`: compute model lag-cube rate maps and cumulative information traces.
- `information.py`: Poisson Fisher, finite-difference derivatives, and spatial single-spike information.
- `pipeline.py`, especially `run_information_step`: current production information path.

The existing `twininfo` pipeline already writes:

```text
metadata/00_population_units.csv
metadata/01_trace_examples*.csv
metadata/02_image_crop_hotspots.csv
metadata/05_lagcube_information_summary.csv
cache/cumulative_information_series.npz
figures/05_*
figures/06_*
figures/07_*
```

Use these artifacts when available, but do not assume all needed derivative/rate-map arrays are cached. Add a new cache if needed.

### From `declan/twin_feature_tangent_structure`

Use the production TFTS output as the basis source. The current preferred production run appears to be:

```text
outputs/twin_feature_tangent_structure_prod_limited_synth/
```

Expected useful files:

```text
canonical_unit_manifest.csv
tangent_maps/twin_tangent_maps.pkl
union_spectrum/twin_tangent_union_spectrum.csv
union_spectrum/twin_tangent_union_summary.csv
train_test_basis/twin_tangent_train_test_basis.csv
covariance_approx/twin_linear_covariance_approx.csv
MANUSCRIPT_REPORT.md
README.md
```

The basis should be derived from the image-disjoint tangent-family analysis, not from object-random leakage-prone splits. The default should use:

```text
delta_arcmin = 0.25
basis_k = 10
basis_source = image_disjoint or union_basis_with image-disjoint validation
n_units = 756 canonical cells
```

If the existing output does not save an explicit basis matrix `U`, add a small utility in the TFTS module to reconstruct and save it from `twin_tangent_maps.pkl`.

---

## Core metric choice

Use derivative-projection Fisher as the main metric.

Do **not** initially compute Poisson information on projected rates, because linear projections of rates can be negative and are not literal Poisson rates.

Instead, use the standard local Poisson Fisher form:

```text
F = J.T diag(1 / mu) J
```

where:

- `mu` is the original nonnegative model rate vector.
- `J` is the derivative of the model rate vector with respect to spatial displacement, usually two columns: `dmu/dx` and `dmu/dy`.

Then decompose the derivative, not the rates:

```text
J_full = J
J_tangent = U_k @ U_k.T @ J
J_orth = (I - U_k @ U_k.T) @ J
J_shuffle = U_shuffle @ U_shuffle.T @ J
J_random = U_random @ U_random.T @ J
```

Compute Fisher using the same original `mu` denominator for all derivative components:

```text
F_full = J_full.T diag(1 / mu) J_full
F_tangent = J_tangent.T diag(1 / mu) J_tangent
F_orth = J_orth.T diag(1 / mu) J_orth
F_shuffle = J_shuffle.T diag(1 / mu) J_shuffle
F_random = J_random.T diag(1 / mu) J_random
```

For scalar summaries, use the trace:

```text
spatial_fisher_trace = trace(F)
```

This asks whether displacement sensitivity is concentrated in the compact reafferent tangent subspace.

---

## Important nuance: Euclidean versus Poisson metric projection

The simplest projection is Euclidean:

```text
P = U_k @ U_k.T
J_U = P @ J
```

This is acceptable for the first implementation, but record it explicitly as a Euclidean response-space projection evaluated with a Poisson Fisher metric.

If time allows, add an optional Poisson-whitened projection:

```text
W = diag(1 / sqrt(mu + eps))
J_white = W @ J
U_white = orth(W @ U_k)
J_U_white = U_white @ U_white.T @ J_white
F_U = J_U_white.T @ J_U_white
```

This version asks whether the tangent basis captures Fisher information after local Poisson whitening. It may be the cleaner final method, but it is more complex because `mu` changes with time/image/condition. Implement Euclidean first, then add the whitened version as a sensitivity check.

---

## Required basis handling

### Basis source

Load or construct `U_k` as an orthonormal basis over the canonical twin unit axis.

Expected shape:

```text
U_full: (n_units, n_basis_vectors)
U_k:    (n_units, k)
```

Use `k = 10` by default.

The unit axis must match the population response axis used by `twininfo`. This is the most important implementation audit.

### Unit manifest alignment

Before computing anything, verify that the unit order in the TFTS basis matches the unit order in the `twininfo` response vectors.

Required checks:

1. Load `outputs/twin_feature_tangent_structure_prod_limited_synth/canonical_unit_manifest.csv`.
2. Load `outputs/twininfo/<run>/metadata/00_population_units.csv`, or the equivalent population selection used in the current run.
3. Determine whether `twininfo` is using:
   - one response per canonical unit only, or
   - selected canonical unit at every retinotopic grid position.

This matters because the TFTS basis is probably over 756 canonical cells, while `twininfo` may expand the population by evaluating selected units over multiple retinotopic grid positions.

### If `twininfo` uses only canonical units

Then `U_k` can be applied directly to the unit axis.

### If `twininfo` uses unit × grid-position population

Do **not** blindly apply the 756-cell basis to the flattened expanded population.

Possible solutions, in preferred order:

1. Run a new `twininfo` configuration restricted to the same canonical center-grid unit responses used by TFTS.
2. Build a block-diagonal or repeated-position projection only if the same canonical unit set is repeated across grid positions and this is scientifically intended.
3. Treat expanded-grid information as a separate analysis and do not use it for the first Panel E result.

For the first implementation, prefer option 1: match the population to the TFTS canonical unit axis, even if this requires a smaller or center-grid-only `twininfo` run.

### Null bases

Implement at least two nulls:

1. `unit_shuffle`: permute rows of `U_k`, then re-orthonormalize.
2. `random_orthogonal`: draw a random Gaussian `(n_units, k)` matrix and QR-orthonormalize.

Also compute the true orthogonal complement:

```text
J_orth = J - U_k @ (U_k.T @ J)
```

The orthogonal complement is not a null basis; it is the residual derivative outside the tangent subspace.

---

## Input conditions

Minimum main conditions:

```text
real
stabilized
```

Optional controls, after the main result works:

```text
pyramid_phase_scrambled
sf_low
sf_mid_low
sf_mid_high
sf_high
```

The main Panel E should focus on real versus stabilized and basis decomposition. Phase/spectral controls can go into supplement or a secondary output figure.

---

## Recommended command-line interface

Implement a CLI like:

```bash
conda run --no-capture-output -n yatesfv python -m jake.twininfo.run_tangent_subspace_information \
  --run-name panelE_tangent_subspace_v1 \
  --twininfo-run outputs/twininfo/production_all_images \
  --tfts-run outputs/twin_feature_tangent_structure_prod_limited_synth \
  --basis-k 10 \
  --basis-delta-arcmin 0.25 \
  --basis-source image_disjoint \
  --conditions real stabilized \
  --projection-mode derivative_euclidean \
  --metric fisher_trace \
  --n-null-repeats 100 \
  --recompute
```

Also support a small validation command:

```bash
conda run --no-capture-output -n yatesfv python -m jake.twininfo.run_tangent_subspace_information \
  --run-name panelE_validation_three_images \
  --image-indices 24 29 30 \
  --n-crops-per-image 1 \
  --n-examples-per-kind 2 \
  --population-size 16 \
  --tfts-run outputs/twin_feature_tangent_structure_prod_limited_synth \
  --basis-k 10 \
  --basis-delta-arcmin 0.25 \
  --conditions real stabilized \
  --projection-mode derivative_euclidean \
  --n-null-repeats 10 \
  --recompute
```

The validation command may need to reduce `basis-k` if the selected validation population is too small. If so, report it clearly and do not use validation numbers for the manuscript.

---

## Detailed analysis steps

### Step 0: Resolve population and basis compatibility

Create a function:

```python
def load_tangent_basis(tfts_run: Path, delta_arcmin: float, k: int, basis_source: str) -> BasisBundle:
    ...
```

Return:

```python
@dataclass
class BasisBundle:
    U: np.ndarray                  # (n_units, k), orthonormal
    U_full: np.ndarray             # (n_units, n_available_basis)
    unit_manifest: pd.DataFrame
    delta_arcmin: float
    k: int
    source: str
    metadata: dict
```

If no basis matrix is saved, reconstruct it:

1. Load `tangent_maps/twin_tangent_maps.pkl`.
2. Extract `bx`, `by` for the requested `delta_arcmin` and valid objects/images.
3. Stack columns: `B = [bx_1, by_1, bx_2, by_2, ...]`.
4. Centering: default should be **no centering**, because tangents are derivative vectors, not samples around a mean. If centering is considered, make it an explicit optional flag and default false.
5. Compute SVD: `B = U S Vt`.
6. Save reconstructed basis to:

```text
outputs/twin_feature_tangent_structure_prod_limited_synth/train_test_basis/tangent_basis_delta0.25.npy
```

or in the new output directory with clear provenance.

Required basis audit:

```python
assert np.allclose(U.T @ U, np.eye(k), atol=1e-5)
assert U.shape[0] == n_units_expected
```

### Step 1: Load or generate natural-image examples

Use the same logic as `jake.twininfo.pipeline`:

- image indices
- crop centers
- trace examples
- real versus stabilized retinal movies
- optional phase/spectral controls

Prefer to reuse an existing `outputs/twininfo/<run>` if it has the required metadata and caches. Otherwise generate within the new script using the same functions.

The script should write a manifest of every analyzed movie:

```text
metadata/panelE_movie_manifest.csv
```

Columns:

```text
movie_id
image_index
crop_rank
crop_center_x
crop_center_y
example_id
trace_id
trace_kind
condition
n_timepoints
source_twininfo_run
```

### Step 2: Compute shifted model rate maps

For each movie, compute shifted rate maps around the current retinal movie. Use existing helpers in `lagcube_information.py`.

The shift set should support finite differences in x and y. Use the same shift convention as existing `finite_difference_shift_set` and `finite_difference_derivatives`.

Required shift states:

```text
center
x_plus
x_minus
y_plus
y_minus
```

Optional for robustness:

```text
cross-grid or square-grid shifts for SSI/control metrics
```

For the main derivative-projection Fisher result, only the five finite-difference states are required.

Save cache:

```text
cache/panelE_rate_maps_or_derivatives.npz
```

Minimum arrays:

```text
mu_center[movie, time, unit]
dmu_dx[movie, time, unit]
dmu_dy[movie, time, unit]
condition[movie]
movie_id[movie]
```

If the existing code naturally stores arrays as `(time, unit, row, col)`, convert to a canonical flat unit axis before projection. Document this conversion.

### Step 3: Compute full and projected Fisher traces

For each movie and time bin:

```python
mu = mu_center[t]                         # (n_units,)
J = np.stack([dmu_dx[t], dmu_dy[t]], axis=1)  # (n_units, 2)
```

Stabilize rates:

```python
mu_safe = np.maximum(mu, eps_rate)
```

Recommended default:

```text
eps_rate = 1e-6 or existing twininfo epsilon
```

Projection functions:

```python
def project_derivative(J, U):
    return U @ (U.T @ J)

def fisher_trace(mu, J):
    w = 1.0 / np.maximum(mu, eps_rate)
    # Equivalent to trace(J.T diag(w) J)
    return float(np.sum(w[:, None] * J * J))
```

Compute:

```text
basis_type = full:              J
basis_type = tangent_k:         P_U J
basis_type = orthogonal:        J - P_U J
basis_type = unit_shuffle_k:    P_Ushuffle J
basis_type = random_k:          P_Urandom J
```

Save per-time output:

```text
results/panelE_cumulative_fisher_by_condition.csv
```

Columns:

```text
run_name
movie_id
image_index
crop_rank
example_id
trace_id
trace_kind
condition
basis_type
basis_k
basis_delta_arcmin
projection_mode
metric
time_index
time_sec
instantaneous_information
cumulative_information
final_information
```

Cumulative information should match the convention in existing `twininfo` information outputs. If existing code integrates over time by multiplying by `DT` or summing per-bin contributions, follow that convention and record it.

### Step 4: Compute gain over stabilized input

For each matched movie set, compare real to stabilized using the same image/crop/trace identity.

Define:

```text
I_real_full
I_real_tangent
I_real_orthogonal
I_real_shuffle
I_stabilized_full
```

Primary quantities:

```text
full_fem_gain = I_real_full - I_stabilized_full
basis_gain = I_real_basis - I_stabilized_full
fraction_full_fem_gain_captured = basis_gain / full_fem_gain
```

However, if stabilized projection-specific information is also computed, include both baselines:

```text
basis_specific_gain = I_real_basis - I_stabilized_basis
full_baseline_gain = I_real_basis - I_stabilized_full
```

For the main result, the cleanest summary is likely:

```text
fraction_full_fem_gain_captured = (I_real_tangent - I_stabilized_full) / (I_real_full - I_stabilized_full)
```

But this can be negative or exceed one. Do not clip. Report the raw value and summarize robustly.

Save:

```text
results/panelE_final_information_gain.csv
```

Columns:

```text
movie_group_id
image_index
crop_rank
example_id
trace_id
trace_kind
basis_type
basis_k
I_stabilized_full
I_real_full
I_real_basis
full_fem_gain
basis_gain_vs_stabilized_full
fraction_full_fem_gain_captured
```

Add robust summary:

```text
results/panelE_subspace_capture_summary.csv
```

Columns:

```text
basis_type
basis_k
n_movie_groups
median_fraction_gain_captured
mean_fraction_gain_captured
bootstrap_ci_low
bootstrap_ci_high
median_gain
bootstrap_gain_ci_low
bootstrap_gain_ci_high
```

Use bootstrap over movie groups. Prefer grouping by image identity or image/crop depending on sample size; include both if feasible.

### Step 5: Nulls and uncertainty

Implement `n_null_repeats` for unit-shuffled and random bases.

For each repeat:

```text
basis_type = unit_shuffle_k_repeatNN
basis_type = random_k_repeatNN
```

Then summarize nulls as distributions:

```text
results/panelE_basis_null_summary.csv
```

Columns:

```text
null_type
basis_k
n_repeats
median_fraction_gain_captured
ci_low
ci_high
real_value
empirical_p
```

Empirical p-value can be one-sided:

```text
p = (1 + number of null repeats >= real_value) / (1 + n_null_repeats)
```

### Step 6: Figures

Create a compact Figure 4E-ready figure and supporting diagnostic figures.

Main output:

```text
figures/panelE_tangent_subspace_information.pdf
figures/panelE_tangent_subspace_information.svg
figures/panelE_tangent_subspace_information.png
```

Recommended main panel layout:

#### E1: Representative cumulative information trace

Curves:

```text
stabilized full
real full
real tangent basis k=10
real orthogonal complement
real unit-shuffle/null basis
```

Y-axis:

```text
Cumulative spatial Fisher information
```

or:

```text
Cumulative spatial Fisher trace
```

X-axis:

```text
Time from movie onset (ms)
```

Use one representative image/crop/trace where the result is typical, not the strongest example. Select the movie closest to the median full-FEM gain.

#### E2: Summary across movies

Bars or paired dots:

```text
full real FEM gain
tangent basis gain
orthogonal complement gain
unit-shuffled/null basis gain
random basis gain
```

Preferred y-axis:

```text
Fraction of full FEM gain captured
```

Set full population to 1 by definition, show tangent, orthogonal, and nulls relative to it.

Do not overplot too many null repeats in the main panel. Show null distribution as a light band or violin.

Supporting figures:

```text
figures/panelE_all_cumulative_traces_by_condition.pdf
figures/panelE_gain_by_image.pdf
figures/panelE_gain_by_trace_kind.pdf
figures/panelE_null_distributions.pdf
figures/panelE_basis_k_sensitivity.pdf
figures/panelE_projection_mode_sensitivity.pdf
```

---

## Minimal pseudocode

```python
from pathlib import Path
import numpy as np
import pandas as pd


def main(args):
    out_dir = Path("outputs/tangent_subspace_information") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load basis
    basis = load_tangent_basis(
        Path(args.tfts_run),
        delta_arcmin=args.basis_delta_arcmin,
        k=args.basis_k,
        basis_source=args.basis_source,
    )

    # 2. Load or build movie manifest from twininfo
    movie_manifest = load_or_build_movie_manifest(args)

    # 3. Resolve population order
    population = load_twininfo_population(args)
    audit_population_basis_alignment(population, basis)

    # 4. Load model
    model_bundle = load_model_bundle(args)

    rows_time = []
    rows_final = []

    null_bases = make_null_bases(basis.U, n_repeats=args.n_null_repeats, seed=args.seed)

    for movie_group in matched_real_stabilized_groups(movie_manifest):
        derivatives_by_condition = {}

        for condition in args.conditions:
            movie = render_or_load_movie(movie_group, condition, args)
            mu, dmu_dx, dmu_dy = compute_or_load_derivatives(model_bundle, movie, population, args)
            derivatives_by_condition[condition] = (mu, dmu_dx, dmu_dy)

            for basis_name, U in basis_dict(basis.U, null_bases).items():
                time_series = compute_projected_fisher_time_series(mu, dmu_dx, dmu_dy, U, basis_name)
                rows_time.extend(time_series_to_rows(movie, condition, basis_name, time_series))

        # 5. Gain summaries matched real vs stabilized
        rows_final.extend(compute_gain_rows(movie_group, derivatives_by_condition, basis, null_bases, args))

    time_df = pd.DataFrame(rows_time)
    final_df = pd.DataFrame(rows_final)
    time_df.to_csv(out_dir / "results/panelE_cumulative_fisher_by_condition.csv", index=False)
    final_df.to_csv(out_dir / "results/panelE_final_information_gain.csv", index=False)

    summary_df = summarize_capture(final_df)
    summary_df.to_csv(out_dir / "results/panelE_subspace_capture_summary.csv", index=False)

    plot_panelE(time_df, final_df, summary_df, out_dir / "figures")
    write_readme(args, summary_df, out_dir)
```

---

## Tests to add

Create tests under:

```text
jake/twininfo/tests/test_tangent_subspace_information.py
```

Required unit tests:

### 1. Projection preserves full derivative decomposition

```python
J_tangent = P @ J
J_orth = J - J_tangent
assert np.allclose(J, J_tangent + J_orth)
```

### 2. Orthonormal basis audit

```python
assert np.allclose(U.T @ U, np.eye(k), atol=1e-5)
```

### 3. Fisher trace nonnegative

```python
assert fisher_trace(mu, J) >= 0
assert fisher_trace(mu, P @ J) >= 0
```

### 4. Full basis recovers full Fisher

If `U` spans the whole unit space:

```python
assert np.allclose(fisher_trace(mu, J), fisher_trace(mu, U @ (U.T @ J)))
```

For Euclidean projection plus nonuniform Poisson weights, this only holds if the projection is full rank. Use full identity basis for this test.

### 5. Orthogonal derivative is Euclidean-orthogonal to tangent basis

```python
assert np.allclose(U.T @ J_orth, 0, atol=1e-5)
```

### 6. Manifest matching

Check that real and stabilized conditions match on:

```text
image_index
crop_rank
example_id
trace_id
```

### 7. No negative-rate issue

The derivative-projection path should never project rates, so no projected rate should be passed as `mu` to a Poisson function. Test that the function API accepts only `mu_center` from the original model.

---

## Validation / smoke run

Before production, run a small validation:

```bash
conda run --no-capture-output -n yatesfv python -m jake.twininfo.run_tangent_subspace_information \
  --run-name smoke_panelE_three_images \
  --image-indices 24 29 30 \
  --n-crops-per-image 1 \
  --n-examples-per-kind 2 \
  --tfts-run outputs/twin_feature_tangent_structure_prod_limited_synth \
  --basis-k 10 \
  --basis-delta-arcmin 0.25 \
  --conditions real stabilized \
  --n-null-repeats 5 \
  --recompute
```

Smoke success criteria:

1. Script runs without errors.
2. Unit/basis alignment audit passes.
3. `panelE_cumulative_fisher_by_condition.csv` exists and has nonzero rows.
4. `panelE_final_information_gain.csv` exists and contains full, tangent, orthogonal, and null basis types.
5. Fisher values are finite and nonnegative.
6. Figures are written.

---

## Production run

After smoke passes:

```bash
conda run --no-capture-output -n yatesfv python -m jake.twininfo.run_tangent_subspace_information \
  --run-name panelE_production_all_images_k10_delta025 \
  --twininfo-run outputs/twininfo/production_all_images \
  --tfts-run outputs/twin_feature_tangent_structure_prod_limited_synth \
  --basis-k 10 \
  --basis-delta-arcmin 0.25 \
  --basis-source image_disjoint \
  --conditions real stabilized \
  --projection-mode derivative_euclidean \
  --metric fisher_trace \
  --n-null-repeats 100 \
  --recompute
```

If there is no compatible existing `production_all_images` run, use the same image/crop/trace settings as Jake’s production pipeline:

```text
n_crops_per_image = 3
n_examples_per_kind = 10
population_size = canonical/TFTS-compatible population
```

---

## Decision criteria for main Figure 4E

The result is main-figure-worthy if all of the following hold:

1. `I_real_full > I_stabilized_full` across a robust fraction of movie groups.
2. `tangent_k10` captures most of the full real-FEM gain.
3. The orthogonal complement captures substantially less than the tangent basis.
4. Unit-shuffled and random orthogonal nulls are below the true tangent basis.
5. Effects are not driven by one image, one crop, or one trace kind.
6. The result survives a reasonable `k` sensitivity check, e.g. `k = 2, 5, 10, 20`.

Suggested summary language if it passes:

> The spatial information introduced by real fixational trajectories was concentrated in the same compact tangent subspace that generalized across image identities. Thus, to a pose-aware observer, the same population geometry carries local information about retinal pose changes in the current stimulus history.

If it only partially passes, report the exact failure mode and do not force it into Figure 4E.

---

## Failure modes and stop rules

### Stop immediately if unit alignment is unresolved

Do not compute projection statistics if the `twininfo` response axis cannot be matched to the TFTS basis axis.

### Stop if the analysis silently expands the population

Do not flatten unit × grid responses and apply a 756-cell basis unless a deliberate block-structured basis is implemented and documented.

### Stop if projected rates are used as Poisson means

The primary analysis projects derivatives, not rates. Projected rates can be negative and should not be used as Poisson means without a separate justified method.

### Stop if only E-optotype/mimicry results are available

Mimicry can support the discussion or supplement, but it is not the Figure 4E analysis.

### Do not claim behavioral improvement

Even if real FEMs increase Fisher information, the claim is about a specified twin/Poisson spatial observer, not measured behavior.

---

## Optional extensions after the main result

### 1. Poisson-whitened projection

Add `--projection-mode derivative_poisson_whitened`.

This may become the preferred final method if it is stable and agrees with the Euclidean projection result.

### 2. Spatial SSI projection variant

This is more delicate because SSI operates on response distributions/rates rather than just derivatives. Only implement after Fisher projection is complete.

Possible approach:

```text
r_projected = r_mean + P @ (r - r_mean)
r_projected = max(r_projected, eps_rate)
```

Clearly label as exploratory/sensitivity because positivity handling affects the result.

### 3. Natural-image phase/spectral controls

After main real/stabilized result, repeat for:

```text
pyramid_phase_scrambled
sf_low
sf_mid_low
sf_mid_high
sf_high
```

Question:

> Does tangent-subspace information capture depend on natural-image phase or spatial-frequency structure?

Keep this out of the main panel unless it is exceptionally clean.

### 4. Trace-kind stratification

Compare:

```text
fixation-only traces
microsaccade-containing traces
```

This can show whether the tangent-subspace information gain is drift-dominated, microsaccade-dominated, or shared.

---

## Required final report from coding agent

After implementation, report:

1. Code path added and command used.
2. Whether the script compiles and tests pass.
3. Exact TFTS basis source file and delta used.
4. Exact population alignment decision.
5. Number of images, crops, traces, and movie groups analyzed.
6. Whether real full FEM exceeds stabilized full.
7. Fraction of full FEM gain captured by tangent `k=10`.
8. Orthogonal-complement result.
9. Unit-shuffled/random null result and empirical p-values.
10. Any deviations from this handoff.
11. Whether the result is strong enough for main Figure 4E.
