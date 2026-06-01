# Keystone Test Handoff v3: Does the Transformation Geometry Predict the FEM Acuity Crossover?

## Coding-agent spec — d′ reformulation after the v2 run

**Why v3.** The v2 run revealed that the Tier-1 observable was mis-specified. `G_sep = Sep_avg(real) − Sep_avg(stab)` measures separation of position-averaged class means; averaging pulls every class mean toward the grand mean, so `G_sep` is structurally **negative and tiny** (observed −1e-4 to −5e-4), never crosses zero, and omits the positional-variance **cost** that drives the crossover. Two v2 runs on different accuracy columns gave opposite continuous verdicts (Spearman ρ = −0.10 vs +1.00) on a 4–5-point core grid — i.e., unidentifiable. **No v2 label is earned**; the correct status is "observable mis-specified and grid underpowered," not "geometry descriptive."

**What v3 changes.**
1. New observable: a **decoder-free `d′_geom`** with separation in the numerator and intrinsic + positional variance in the denominator, computed two ways — **finite-cloud first** (no Jacobian), **Jacobian approximation second** — and compared.
2. Function reference is the **4-way D1 crossover** (manuscript anchor), with 6 pairwise deltas secondary. Geometry mirrors this: multiclass primary, pairwise secondary.
3. `Σ_intrinsic` defined explicitly (Poisson diagonal primary; fixed isotropic sensitivity).
4. Dense LogMAR grid; minimum core-point count enforced before any continuous/specificity verdict.
5. Quarantine `smoke2`; window-specific D1 time-mean delta is the only function source.
6. Expanded terminal labels, including `jacobian_linearization_failure`.

This is a **model-only** analysis.

---

## The firewall (updated)

- **Geometry observables** are computed only from deterministic mean responses `μ_θ(p, L)`, their finite-cloud covariance `Cov_p[μ]`, and (for the second arm) Jacobians `J_θ(p, L)`. The denominator's `Σ_intrinsic` is a fixed Poisson/isotropic diagonal derived from mean rates. **Never** the trained D1 decoder, its weights, or its empirical noise.
- The geometry→accuracy crosswalk uses a **fixed, parameter-free nearest-class-mean ideal observer** under the assumed noise model. This is not a trained decoder and does not see the D1 task; it is the Bayes rule for `{ḡ, Σ_total}`. State this in the readme to preempt circularity concerns.
- **Function observable** is the existing D1 accuracy (window-specific, time-mean delta).

If the geometry arm ever reads D1 weights or D1 residual noise, the test is circular and invalid.

---

## Core scientific question

Does a decoder-free `d′_geom`, built from deterministic response variation over the eye-position cloud, predict the **sign, transition LogMAR, and magnitude** of the 4-way D1 FEM advantage, **beyond** single-position difficulty — and does the local Jacobian approximation reproduce that finite-cloud object?

---

## Observable hierarchy (the heart of v3)

Notation: orientations θ ∈ {0,90,180,270}; pair (a,b); position/phase `p`; size `L`; mean response `μ_θ(p,L) ∈ ℝ^N`; real eye-position cloud `cloud_real(L)`; stabilized point `p0(L)` = trial-mean gaze.

**Shared pieces**
- Position-averaged class mean (what the time-mean decoder discriminates): `ḡ_θ(cond,L) = E_{p∼cloud(cond,L)}[ μ_θ(p,L) ]`. For stabilized, `cloud_stab = {p0}` so `ḡ_θ(stab) = μ_θ(p0)`.
- Intrinsic noise (primary): `Σ_int = diag(r̄)` Poisson, `r̄` the global mean rate per neuron (fixed across all `L` and conditions). Sensitivity: `Σ_int = σ²·I` fixed isotropic.
- Window scaling: both conditions average `N_frames`; intrinsic term enters as `Σ_int / N_frames`. Treat `N_eff` for the positional term as a documented modeling choice with a sensitivity sweep (see guardrails), not a hidden constant.

### Arm A — finite-cloud `d′_geom` (PRIMARY; needs only μ over the cloud, no Jacobian)
- Positional covariance, empirical: `Σ_pos^A(cond,L) = Cov_{p∼cloud(cond,L)}[ μ_θ(p,L) ]` (orientation-conditioned, then pooled; `Σ_pos^A(stab) ≈ 0`).
- Total: `Σ_tot^A(cond,L) = Σ_int/N_frames + Σ_pos^A(cond,L)`.
- Pairwise: `d′^A_pair(a,b,cond,L) = || ḡ_a − ḡ_b || / sqrt( û_abᵀ Σ_tot^A û_ab )`, `û_ab` unit vector along `ḡ_a − ḡ_b`.
- Multiclass (PRIMARY geometry observable): **predicted 4-way accuracy** `Acc_geom^A(cond,L)` from a fixed nearest-class-mean ideal observer drawing samples `~ N(ḡ_θ, Σ_tot^A(cond))` (Monte Carlo, fixed seed). Parameter-free given `{ḡ, Σ_tot}`.
- Contrasts (the d′ analogs of Δacc): `Δd′^A_pair(a,b,L) = d′^A(real) − d′^A(stab)`; `ΔAcc_geom^A(L) = Acc_geom^A(real) − Acc_geom^A(stab)`.

The crossover lives here because `û_abᵀ Σ_pos û_ab` is large exactly when the identity-difference axis aligns with translation-variance directions — i.e., high translation mimicry. So Arm A already encodes the Tier-2 mechanism implicitly; the explicit `ΔM` (below) is its decomposition.

### Arm B — Jacobian `d′_J` (SECONDARY; needs raw J; can be deferred)
- Positional covariance, linearized: `Σ_pos^B(cond,L) = J_θ(p0,L) Σ_eye(cond,L) J_θ(p0,L)ᵀ` (orientation-conditioned), `Σ_eye` the eye-position covariance of the cloud.
- Everything else as in Arm A → `d′^B`, `Acc_geom^B`, `Δd′^B`, `ΔAcc_geom^B`.

### Arm A vs B comparison
- `jacobian_approximation_error.csv`: per `L`, the discrepancy between `Σ_pos^A` and `Σ_pos^B` (Frobenius, leading-eigenvector alignment) and between `ΔAcc_geom^A` and `ΔAcc_geom^B`. This separates "geometry predicts D1" (Arm A) from "the local tangent explains it" (Arm B).

### Difficulty controls (single-position, no cloud)
- `S_center`, `S_stab`, `S_cloud_mean` as in v2 (pairwise + multiclass predicted accuracy at a single position, `Σ_pos = 0`). `S_best = max`.

### Tier-2 mechanism (explicit tangent)
- Translation mimicry `ΔM(L)` from raw J (the projection of the identity-difference axis onto the translation tangent). Confirms whether the Arm-A denominator inflation is specifically the translation tangent.

---

## LogMAR grid (densified)

```text
-0.40 -0.375 -0.35 -0.325 -0.30 -0.275 -0.25 -0.225 -0.20 -0.175 -0.15 -0.125 -0.10
```
- `−0.40` = `render_limit_control` (boundary only).
- Core range `−0.35 … −0.20`. **Minimum 8 populated core points** required before any continuous or specificity verdict; otherwise `inconclusive_underpowered_grid`. (v2 had 4 — this is the binding fix for the ρ instability.)

---

## Cache audit and sequencing (important)

**Arm A is not blocked on the raw-Jacobian recompute.** It needs only `μ_θ(p,L)` over the cloud, which the phase-landscape machinery already produces (or cheaply can). Run Arm A first and report its verdict. Arm B (and `ΔM`) require raw `J` arrays (v2 had "norms only"); re-run `eoptotype_jacobian_field_smoothness.py` storing full `J`, then run Arm B. Do not gate the primary finite-cloud answer on the `J` recompute.

Step 0 emits `cache_availability_report.csv` (`quantity, source, on_grid, has_per_phase_mu, has_raw_J, arm_A_ready, arm_B_ready, needs_recompute, cost_note`) and `grid_reconciliation.csv`.

Function source: **only** the explicit D1 time-mean sweep (`real_minus_stabilized_d1_time_mean_accuracy`, window-specific). Quarantine all `smoke2` / aggregate / `rate_normalized_decoder_accuracy` artifacts. Also fix the provenance print so the logged D1 column matches the column actually loaded (v2 logged `rate_normalized...` while using the time-mean flag).

---

## Required analysis structure

1. **Step 0** — cache audit, grid reconciliation, function-source lock, arm-readiness.
2. **Step 1 (Arm A)** — `ḡ`, `Σ_pos^A`, `Σ_tot^A`, `d′^A_pair`, `Acc_geom^A`, contrasts; `S_center/S_stab/S_cloud_mean`.
3. **Step 2** — function curves: **4-way D1** (primary) and 6 pairwise (secondary), window-specific, trial-bootstrapped.
4. **Step 3 — coincidence (Arm A).** Zero-crossing of `ΔAcc_geom^A` vs the 4-way `Δacc`; `dL`; bootstrap CI. (Now testable: `ΔAcc_geom` can be negative.)
5. **Step 4 — continuous (Arm A).** Regress 4-way `Δacc` on `ΔAcc_geom^A` (and on multiclass `d′^A`); slope, R², Spearman. Gated on ≥8 core points.
6. **Step 5 — specificity (Arm A).** Partial correlation and nested model of `Δacc` on geometry beyond `S_best`.
7. **Step 6 (Arm B)** — repeat 1,3,4 with `Σ_pos^B`; compute `jacobian_approximation_error`.
8. **Step 7 — Tier-2** — `ΔM`; does the explicit tangent track the Arm-A denominator inflation / the transition.
9. **Step 8 — intrinsic-noise sensitivity** — repeat Arm A with Poisson vs fixed isotropic; does the sign/crossover depend on it.
10. **Step 9 — pairwise** — Arm A coincidence/specificity per pair; which pairs drive it.
11. **Step 10 — nulls** — phase-shuffle, pair-label-shuffle on the continuous relationship. (Note: a size-regression null is weak with few points; report but don't over-weight.)
12. **Step 11 — window robustness** — Tests across windows (now enabled by window-specific accuracy).
13. **Step 12** — tables, figures, decision, readme.

---

## Statistical detail

- **Coincidence:** zero-crossings of `ΔAcc_geom^A` and 4-way `Δacc` within the core range; `|dL_median| <` one grid step (0.025) and `dL` 95% CI contains 0. If `ΔAcc_geom` still has no crossing, label `observable_cannot_test_coincidence` (should not happen for a correctly built d′; if it does, inspect `Σ_pos`).
- **Continuous (requires ≥8 core points):** OLS + Spearman of `Δacc` on `ΔAcc_geom^A`; CI from joint bootstrap (trials for `Δacc`, phases for geometry).
- **Specificity:** partial `ρ(Δacc, ΔAcc_geom^A | S_best)` CI excludes 0 **and** nested ΔR² ≥ 0.15 with AIC favoring `+geom`.
- **Arm A vs B:** `jacobian_geometry_predicts` requires Arm B to pass coincidence+continuous **and** `jacobian_approximation_error` below threshold (`ΔAcc_geom^A − ΔAcc_geom^B` RMS < 0.02 across core, leading-eigenvector alignment > 0.7).
- **Intrinsic-noise robustness:** the coincidence sign must be stable across Poisson vs isotropic; if not, flag `sign_depends_on_noise_model` in the readme (not a terminal label, a caveat on whichever label is assigned).

---

## Required output tables

```text
cache_availability_report.csv
grid_reconciliation.csv
finite_cloud_dprime_geometry.csv      # L, condition, pair/all, d'_A, Acc_geom_A, Sigma_pos_proj, S_center/stab/cloud_mean
jacobian_dprime_geometry.csv          # L, condition, pair/all, d'_B, Acc_geom_B
dprime_geometry_vs_D1_crosswalk.csv   # L, dAcc_geom_A, dAcc_geom_B, delta_acc_4way, delta_acc_pair, S_best
jacobian_approximation_error.csv      # L, frob_diff, eig_alignment, dAcc_A_minus_B
function_curve.csv                    # 4-way + pairwise, window-specific, CIs
keystone_tests.csv                    # all tests, mean + pairwise, Arm A + Arm B, with verdict components
geometry_dprime_decision_table.csv    # one row, full component set + decision_label
keystone_readme.md
```

`keystone_readme.md` answers: (1) Arm-A coincidence within core? (2) Arm-A continuous, beyond `S_best`? (3) does Arm B reproduce Arm A (linearization OK)? (4) does `ΔM` confirm the tangent mechanism? (5) which pairs drive it? (6) sign stable across intrinsic-noise model? (7) ≥8 core points? (8) decision label + whether Figure 4 leads with geometry.

---

## Required figures

- **Fig A — keystone overlay (Arm A).** 4-way `Δacc(L)` and `ΔAcc_geom^A(L)` on twin axes, both zero-crossings + CIs, `−0.40` shaded. → paper panel 4F.
- **Fig B — continuous.** `Δacc` vs `ΔAcc_geom^A` across the dense grid, fit, colored by `L`.
- **Fig C — difficulty control.** `Δacc` vs geometry after partialling `S_best`.
- **Fig D — Arm A vs B.** `ΔAcc_geom^A` and `ΔAcc_geom^B` overlaid; approximation error inset.
- **Fig E — Tier-2.** `ΔM(L)` against the Arm-A denominator-projection transition.
- **Fig F — pairwise small multiples.**
- **Fig G — intrinsic-noise sensitivity.** Coincidence under Poisson vs isotropic.

---

## Decision logic

```text
finite_cloud_geometry_predicts_D1_crossover
    ≥8 core points
    AND Arm-A coincidence TRUE
    AND Arm-A continuous significant
    AND specificity beyond S_best TRUE
  → Response-cloud geometry predicts the crossover. (Necessary basis for the geometry-led figure.)

jacobian_geometry_predicts_D1_crossover
    finite_cloud_geometry_predicts_D1_crossover TRUE
    AND Arm B also passes AND jacobian_approximation_error below threshold
    AND ΔM tracks the transition
  → The LOCAL TANGENT geometry predicts and mechanistically explains the crossover.
    Strongest result; Figure 4 leads with the Jacobian/equivariant-manifold story.

finite_cloud_predictive_jacobian_not   (alias: jacobian_linearization_failure)
    Arm A passes but Arm B fails / approximation error above threshold
  → Response-cloud geometry predicts the crossover, but the local Jacobian does NOT capture it
    (finite cloud too large or surface too nonlinear). Geometry-led figure stands on the
    finite-cloud d′; report the linearization breakdown honestly rather than as a geometry failure.

geometry_tracks_difficulty_only
    Arm-A continuous significant but specificity beyond S_best FALSE
  → Geometry tracks task difficulty, not mechanism. Lead with acuity; geometry to supplement.

observable_cannot_test_coincidence
    ΔAcc_geom has no zero-crossing in core (unexpected for a correct d′)
  → Inspect Σ_pos construction before any scientific read.

inconclusive_underpowered_grid
    < 8 core points, or coincidence sign flips across runs/metrics
  → Densify/complete the grid; not a null.

render_limit_confounded
    effect requires −0.40 or curves saturate within core
  → Refine rendering / add finer sizes; do not call a verdict.
```

Caveat tags (appended to whichever label applies, not terminal): `sign_depends_on_noise_model`, `pairwise_driven` (one/two pairs carry it).

---

## Implementation guardrails

1. Firewall: geometry from `μ`, `Cov_p[μ]`, `J`, and a fixed `Σ_int` only; the crosswalk uses a fixed nearest-mean ideal observer, never the D1 decoder.
2. Run **Arm A first** and report its verdict before, and independently of, the raw-`J` recompute for Arm B.
3. Function side: 4-way D1 primary; window-specific time-mean delta only; `smoke2`/aggregate quarantined.
4. `≥ 8` populated core points before any continuous/specificity verdict.
5. Report `d′` for both intrinsic-noise models; flag if the sign depends on the choice.
6. Document `N_eff`/window scaling as an explicit modeling choice with a sensitivity check; do not bury a constant.
7. `−0.40` is a boundary control only.
8. Report mean (multiclass) and pairwise; do not average pair structure away.
9. Fix the provenance print to match the loaded column.
10. Model-only; no recorded data.
11. No new observable variants if a result is null — report the predefined label and stop.

---

## Minimal acceptance criteria

All tables above present; readme ends with one decision label, any caveat tags, and an explicit statement of whether (and on which arm) Figure 4 leads with geometry. Arm A must complete even if Arm B is deferred for `J` caching.

---

## Final stop rule

Run Arm A on the dense grid with the locked 4-way time-mean function source and both intrinsic-noise models; report its verdict. Then run Arm B once raw `J` is cached. Report the terminal label and caveats. Sanctioned reruns are implementation failures only: grid/source mismatch, firewall violation, degenerate `μ`/`J`/`Σ_pos`, < 8 core points, or `Σ_int` mis-derivation. Do not widen the slice or invent observable variants to chase a positive result. If Arm A is null with ≥ 8 core points and stable sign, that is an informative negative — the geometry does not predict the crossover beyond difficulty, and Figure 4 leads with acuity.
