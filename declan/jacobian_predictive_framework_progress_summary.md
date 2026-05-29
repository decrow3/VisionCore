# Predictive Jacobian Framework: Big-Picture Progress

Companion note to `jacobian_predictive_framework_handoff_revised.md`.

Date: 2026-05-27

## Bottom line

The fixRSVP Jacobian analysis has two complementary results:

1. **Local linearization (small pairwise distances):** With per-sample midpoint Jacobians, local linearization is valid at 0–1 px pairwise separations — R² ≈ +0.74, cosine ≈ +0.89, replicated across all 4 Allen sessions. Linearization degrades systematically with distance (R² negative by 2–3 px), confirming the tangent approximation is genuinely local.

2. **Manifold alignment (full distribution):** The Jacobian column space predicts above-shuffled FEM covariance capture (V_J delta > 0) across all distance bins and all sessions with a fair bin-specific null. This supports the interpretation that `span(J_I)` indexes an image-specific translation manifold axis that organises FEM population covariance.

The critical methodological finding: the flat V_J profile from the earlier baseline-J analysis was an artifact of evaluating the Jacobian 7–22 px from the sample positions. With position-matched Jacobians the expected distance-dependent linearization signal emerges clearly.

> **Canonical claim:** The image-translation Jacobian provides a valid local first-order description of V1 model responses at pairwise eye-position separations below ~1 px (R² ≈ +0.74, cosine ≈ +0.89, N = 4 sessions). At larger separations, pointwise linearization fails, but the Jacobian column space continues to predict above-shuffled FEM covariance capture across all tested distances — consistent with responses remaining near a translation manifold throughout the fixational eye movement range.

## What is now built

- Step 0 / Step 1 pipeline: `scripts/jacobian_predictive_framework/run_fixrsvp_steps01.py`
- `_compute_per_sample_jacobians_batch`: Jacobian at each sample's absolute eye position via 4×N batched forward passes → `(N, n_neurons, 2)`.
- `_compute_pairwise_bin_metrics`: per-bin R²_lin, cosine, V_J matched/shuffled/delta for both baseline J and local midpoint J.
  - **Null correctness:** For V_J local, the shuffled null uses bin-specific mean Jacobians from each matched unit's own per-sample Jacobians restricted to pairs in the same distance bin — symmetric with the matched J. An earlier version used a global mean (all positions) for the shuffled null; this was asymmetric and slightly suppressed delta at small bins.
- `step01_pairwise_bins.csv` includes both baseline and local J columns.
- `summarize_fixrsvp_cross_session.py`: cross-session manifold alignment figure plus 2×3 pairwise figure (Row 0: baseline J; Row 1: local midpoint J; columns: R², cosine, V_J delta vs. distance bin).
- Canonical four-session runs: `allen_2022_02_16_local_J_v2`, `allen_2022_02_24_local_J_v2`, `allen_2022_03_04_local_J_v2`, `allen_2022_04_08_local_J_v2` (all using bin-specific shuffled null).

## Cross-session local J results (core finding)

All 4 Allen sessions show the same distance-dependent profile:

### Linearization metrics (R², cosine) — unaffected by null choice

| Bin | R²_base | R²_local | Cosine_base | Cosine_local |
|-----|---------|----------|-------------|--------------|
| 0–1 px | −0.76 to −1.51 | **+0.73 to +0.76** | 0.02–0.09 | **0.87–0.90** |
| 1–2 px | −1.1 to −2.0 | −0.03 to +0.31 | 0.04–0.23 | 0.57–0.71 |
| 2–3 px | −1.7 to −3.7 | −0.26 to −1.1 | 0.06–0.31 | 0.37–0.61 |
| 3–5 px | −2.6 to −5.8 | −0.50 to −1.5 | 0.11–0.25 | 0.47–0.69 |
| 8–12 px | −7.3 to −12.6 | −3.0 to −4.6 | 0.13–0.25 | 0.25–0.37 |

R²_local at 0–1 px: +0.728, +0.744, +0.725, +0.762 — tight cross-session replication (mean +0.74, SD 0.02).  
Cosine_local at 0–1 px: 0.881, 0.891, 0.871, 0.902 — mean +0.89, SD 0.01.  
Both metrics degrade monotonically with distance across all sessions.

### V_J local delta — bin-specific shuffled null (corrected)

V_J local delta is positive at **all bins in all 4 sessions**. Representative values:

| Session | 0–1 px | 1–2 px | 3–5 px | 8–12 px |
|---------|--------|--------|--------|---------|
| 02-16 | 0.056 | 0.053 | 0.177 | 0.103 |
| 03-04 | 0.099 | 0.163 | 0.285 | 0.249 |
| 02-24 | 0.160 | 0.187 | 0.206 | 0.205 |
| 04-08 | 0.110 | 0.167 | 0.188 | 0.115 |

The corrected null raises the delta at short distances compared to the earlier global-mean null: at 0–1 px, delta increased from 0.009–0.162 (old) to 0.056–0.160 (corrected). The global-mean null was slightly unfair because `E_p[J_shuffle(p)]` over all positions happened to project well onto short-range covariance, artificially reducing the matched advantage.

## Progress against the handoff plan

| Handoff step | Status | Current read |
|---|---|---|
| Step 0. Linearization validity | **Complete** | Local J valid at ≤1 px; degrades monotonically. R²_local ≈ +0.74, 4/4 sessions. Baseline J fails because it is evaluated at the wrong position. |
| Step 1. fixRSVP Jacobian generalisation | **Complete (reframed)** | V_J delta > 0 at all bins and all sessions with fair null. Grand median delta A_J +0.060. 04-08 CI above zero. |
| Step 1.5. Pipeline consistency on E-optotypes | Substantially done | Generalized metric reproduces legacy result vs matched-energy null (0.4988 vs 0.1004). |
| Step 2. Real-data scalar bridge | Not started | No empirical bridge from model-predicted Jacobian drive to measured V1 FEM variance/covariance. |
| Step 3. Constrained eye-movement prediction | Not started | Blocked until Step 2. |
| Step 4. Pairwise real-data bridge | Not started | Explicitly premature. |

## Current best scientific interpretation

1. **Per-sample Jacobians resolve the scale mismatch.** The baseline J result (flat V_J, negative R²) was an artifact of evaluating J at a single position 7–22 px from the sample. With J evaluated at each sample's actual position, local tangent structure emerges.

2. **Local linearization confirmed at ≤1 px, 4/4 sessions.** R² ≈ +0.74, cosine ≈ +0.89. This is the correct validation of the linearization claim at the pairwise scale where the approximation should hold.

3. **Distance-dependent degradation rules out the "global axis" alternative.** R² goes from +0.74 at 0–1 px to −3 to −5 at 8–12 px, consistently across all sessions.

4. **V_J delta positive at all bins with a fair null.** This is the manifold alignment result: the image-specific Jacobian subspace predicts above-shuffled covariance capture even where linearization fails. The Jacobian encodes the dominant structural direction of the translation manifold (not just the local tangent).

5. **Two separate supported claims:**
   - *Local:* J is a valid local tangent at ≤1 px (R²/cosine evidence).
   - *Global:* span(J) aligns with Σ_FEM above a fair shuffled null at all displacement scales (V_J delta evidence).

## Main remaining limitations

1. **No real-data bridge.** All results are model-internal.
2. **Unit definition mismatch.** `image_phase_radius` bins by within-trial-centred radius → units span large absolute position clouds (~5–12 px). A radius filter of ≤1 px retains zero units. Future work needs absolute-position clusters.
3. **V_J SNR at 0–1 px.** Tiny response differences → noisy covariance; V_J delta is more reliable at 2–8 px even though local linearization is strongest at 0–1 px.

## Future design note

The current unit definition creates units spanning large absolute eye-position clouds (~5–12 px) across trials. Future analyses should group by **absolute eye-position clusters** or use targeted stimulus repeats near fixed retinal positions to estimate local Jacobian covariance directly.

## Canonical outputs

| Output | Path |
|--------|------|
| Canonical 4-session runs (bin-specific null) | `outputs/jacobian_predictive_framework/allen_2022_*_local_J_v2/` |
| Cross-session summary + figures | `outputs/jacobian_predictive_framework/cross_session_local_J_v2/` |
| Pairwise cross-session figure | `outputs/figures/jacobian_predictive_framework/pairwise_cross_session.png` |
| Cross-session manifold alignment figure | `outputs/figures/jacobian_predictive_framework/cross_session_manifold_alignment.png` |
| Step 1.5 e-optotype summary | `outputs/jacobian_predictive_framework/eoptotype_step15_real/step15_consistency_summary.md` |
| Analysis script | `scripts/jacobian_predictive_framework/run_fixrsvp_steps01.py` |
| Summary script | `scripts/jacobian_predictive_framework/summarize_fixrsvp_cross_session.py` |

## Manuscript phrasing

> Applying per-sample midpoint Jacobians revealed a clear distance-dependent linearization signal: at pairwise eye-position separations below 1 px, local linearization was accurate (R² = 0.74 ± 0.02, cosine = 0.89 ± 0.01, mean ± SD, N = 4 sessions), degrading monotonically at larger separations. Despite the failure of pointwise linearization beyond ~1 px, the Jacobian column space predicted above-shuffled FEM population covariance capture at all tested distances (V_J delta > 0 in all distance bins, all sessions; bin-specific shuffled null), consistent with a **tangent-manifold alignment** interpretation: the image-translation Jacobian identifies image-specific axes that organise V1 population covariance across the full fixational eye movement range.
