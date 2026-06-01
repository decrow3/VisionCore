# Keystone Test Handoff v2: Does the Transformation Geometry Predict the FEM Acuity Crossover?

## Coding-agent spec — revised after review

**Changes from v1:** (1) LogMAR grid trimmed to avoid the rendering-limit regime; `−0.40` demoted to a boundary control. (2) Primary observable renamed `G_sep` (cloud-separability gain) and explicitly marked as a decoder-free mean-response quantity, **not** itself a Jacobian object. (3) Pairwise outputs now required, not optional. (4) Difficulty control expanded to three single-position baselines. (5) Decision-label set revised. (6) Cache-first: no new model inference unless cached `μ`, `J`, and accuracy are unavailable. **Added in review:** a `mechanism_tangent_tracks` requirement so the top label can't pass via sampling alone (see Decision logic).

---

## Purpose

This is the single adjudicating analysis for whether the transformation-geometry framing earns main-text status in Figure 4.

The functional result is in hand: under the digital twin, real FEMs improve E-orientation discrimination below the resolution limit and degrade it above, with a sign-flip near LogMAR ≈ −0.32 to −0.35 (decoder D1, mean-rate readout). The geometry is in hand: response-to-translation is a smooth, low-rank, image-specific manifold, with scale- and phase-dependent translation mimicry.

What does **not** exist is the link. This spec tests one claim, in two tiers:

> **Tier 1 (functional-geometry link).** A decoder-free, mean-response separability quantity — the gain in identity separation from averaging over the real-FEM position cloud versus a single stabilized position — predicts the sign, the transition LogMAR, and the magnitude of the measured FEM accuracy advantage, **beyond** single-position task difficulty.
>
> **Tier 2 (Jacobian mechanism).** That separability transition is attributable to the translation-tangent structure — i.e., the mimicry / tangent-escape contrast tracks the same transition.

Tier 1 establishes that *cloud sampling geometry* predicts the crossover. Tier 2 establishes that it does so *via the equivariant translation manifold*, which is the actual novelty. The strongest label requires both. This is a **model-only** analysis.

---

## The firewall (read before implementing)

- **Geometry observables** are computed *only* from deterministic mean responses `μ_θ(p, L)` and image-translation Jacobians `J_θ(p, L)`. **Never** from a trained decoder, a noise covariance, or trial-resampled accuracy.
- **Function observable** is the existing D1 decoder accuracy on noisy population samples.

If any geometry quantity uses the decoder or its noise model, the test is circular. Enforce in code structure (separate modules; no shared decoder object).

---

## Core scientific question

Across the LogMAR ladder, does `G_sep(L)` cross zero at the same LogMAR as `Δacc(L)`, predict `Δacc(L)` continuously and beyond the best single-position difficulty baseline, and is that transition explained by the translation-tangent contrast `ΔM(L)`?

---

## Definitions (lock these before coding)

Notation: orientations θ ∈ {0, 90, 180, 270}; pairs (a,b); retinal position/phase `p`; size `L` (LogMAR); twin population mean `μ_θ(p, L) ∈ ℝ^N`; image-translation Jacobian `J_θ(p, L) ∈ ℝ^{N×2}` (columns ∂μ/∂x, ∂μ/∂y, forward-mode AD).

Position clouds, matched to the traces that produce the functional result:
- `cloud_real(L)` = retinal positions sampled by real FEM traces over the primary window.
- `cloud_stab(L)` = single point at trial-mean gaze (exact stabilization).
- `center(L)` = single point at stimulus center (no gaze offset).

**Primary keystone observable — decoder-free, NOT a Jacobian quantity:**

- Position-averaged class mean: `ḡ_θ(cond, L) = E_{p ∼ cloud(cond,L)}[ μ_θ(p, L) ]`.
- Averaged separation: `Sep_avg(cond, L) = mean_{a<b} || ḡ_a(cond,L) − ḡ_b(cond,L) ||₂`.
- **`G_sep_mean(L) = Sep_avg(real, L) − Sep_avg(stab, L)`** — cloud-separability gain.
- **`G_sep_pair(a,b,L) = || ḡ_a(real,L) − ḡ_b(real,L) || − || ḡ_a(stab,L) − ḡ_b(stab,L) ||`** — required, per pair.

State in the readme: *`G_sep` is a decoder-free mean-response separability predictor (the functional mirror of D1). The Jacobian-specific mechanism is tested separately via `ΔM`, signal-to-tangent alignment, and JΣeyeJᵀ-style alignment.*

**Difficulty controls — single-position separability (no cloud averaging):**

- `S_center(L) = mean_{a<b} || μ_a(center,L) − μ_b(center,L) ||`.
- `S_stab(L)   = mean_{a<b} || μ_a(stab,L)   − μ_b(stab,L)   ||`.
- `S_cloud_mean(L) = E_{p∼cloud_real}[ mean_{a<b} || μ_a(p,L) − μ_b(p,L) || ]`.
- Pairwise `S_pair(a,b,L)` for each of the three baselines.
- **Best single-position baseline** `S_best(L) = max(S_center, S_stab, S_cloud_mean)` — used in the specificity test so the boring "everything gets harder with size" explanation is given its strongest form.

**Jacobian-mechanism observables (Tier 2):**

- Translation mimicry `M_{a→b}(p,L) = || P_{J_a}(μ_b − μ_a) ||² / || μ_b − μ_a ||²`, `P_{J_a}` the projector onto the 2-column tangent of `a`.
- `M_mean(cond,L)`, `M_pair(a,b,L)` over the cloud; contrast `ΔM_mean(L) = M(real,L) − M(stab,L)`, `ΔM_pair(a,b,L)`.
- Optional: signal-to-tangent alignment and the JΣeyeJᵀ-vs-empirical-PC alignment (the bridge-equation check, gap #6 in the draft ledger) if cached Jacobian/covariance products are available.

Metric: Euclidean primary. If a fixed metric is used, it must be a single global diagonal identical across all `L` and conditions — never per-condition or decoder-derived (firewall).

**Function observable (existing):**
- `Δacc_mean(L; W) = acc_real(L;W) − acc_stab(L;W)`, four-way orientation, D1 mean-rate decoder, primary `W` (default 60).
- `Δacc_pair(a,b,L; W)` — required, per pair.

---

## LogMAR grid

```text
-0.40  -0.35  -0.30  -0.25  -0.20  -0.15  -0.10
```

- `−0.40` is flagged `render_limit_control`: used only to verify boundary behavior, **not** as an independent stimulus size the verdict can rest on. Do **not** include `−0.45` or `−0.50`.
- The decision leans on the **core range `−0.35` to `−0.20`**. A crossing or effect that exists only when `−0.40` is included triggers `render_limit_confounded`.

---

## Relevant existing files / cache-first rule

**Do not run new model inference unless cached `μ_θ(p,L)`, `J_θ(p,L)`, and D1 accuracy curves are unavailable or incompatible with the grid.** First produce `grid_reconciliation.csv` and `cache_availability_report.csv`.

```bash
grep -R "logmar\|accuracy\|stabilized\|real_FEM" -n scripts outputs | grep -i "sweep\|hyperacu\|eopto"   # D1 accuracy
grep -R "mimicry\|tangent\|jacobian\|phase_grid\|class_separation\|mu_\|mean_response" -n scripts outputs # geometry cache
grep -R "forward.*AD\|jvp\|DifferentiableStimulus\|compute_rate_map" -n scripts                          # μ/J producers
```

`cache_availability_report.csv` columns: `quantity, source_path, on_target_grid, has_per_phase, has_per_pair, needs_recompute, recompute_cost_note`. If `μ`/`J` were only ever stored inside mimicry summaries (not as raw per-phase means/Jacobians), recompute is required to form `ḡ` and the `S_*` baselines — flag this prominently, it changes the effort estimate.

---

## Primary script to create

```text
scripts/jacobian_predictive_framework/run_keystone_geometry_crossover.py
```

## Required output directory

```text
outputs/jacobian_predictive_framework/keystone_crossover_<run_label>/
    geometry_curves/   function_curves/   tests/   figures/   logs/
```

## Required CLI

```bash
python scripts/jacobian_predictive_framework/run_keystone_geometry_crossover.py \
  --checkpoint-dir <...>/multidataset_120_long/checkpoints \
  --model-type resnet_none_convgru --model-index 0 \
  --mcfarland-outputs mcfarland_outputs.pkl --dataset-idx 10 \
  --logmar-grid -0.40 -0.35 -0.30 -0.25 -0.20 -0.15 -0.10 \
  --render-limit-control -0.40 --core-range -0.35 -0.20 \
  --orientations 0 90 180 270 \
  --primary-window 60 --windows 12 30 60 \
  --difficulty-baselines center stab cloud_mean \
  --accuracy-sweep AUTO --cache-dir AUTO \
  --phase-grid 33 --n-bootstrap 2000 \
  --out-dir outputs/jacobian_predictive_framework/keystone_crossover_20260531 \
  --run-label 20260531 --device cuda --random-seed 0
```
Optional: `--metric euclidean|fixed_diag`, `--nulls phase_shuffle pair_shuffle cloud_isotropic`, `--recompute-accuracy`, `--pilot-only` (core range only).

Pairwise computation is **always on** (not a flag).

---

## Required analysis structure

1. **Step 0 — cache audit + grid reconciliation.** Produce `cache_availability_report.csv`, `grid_reconciliation.csv`. Recompute `μ`/`J`/accuracy only if missing/incompatible.
2. **Step 1 — geometry curves.** `ḡ`, `Sep_avg`, `G_sep_mean`, `G_sep_pair`; `S_center`, `S_stab`, `S_cloud_mean` (+ pairwise), `S_best`; `M`, `ΔM_mean`, `ΔM_pair`. Firewall-clean.
3. **Step 2 — function curves.** `Δacc_mean`, `Δacc_pair` with trial-level bootstrap (paired by trace).
4. **Step 3 — Test 1 (coincidence, mean).** Zero-crossings of `G_sep_mean` and `Δacc_mean` within the core range; `ΔL`; bootstrap CI.
5. **Step 4 — Test 2 (continuous, mean).** Regress `Δacc_mean` on `G_sep_mean`; slope, sign, R², Spearman.
6. **Step 5 — Test 3 (specificity).** Partial `ρ(Δacc_mean, G_sep_mean | S_best)`; nested `Δacc ~ S_best` vs `Δacc ~ S_best + G_sep` (ΔR², AIC).
7. **Step 6 — Test 4 (Tier-2 mechanism).** Does `ΔM_mean(L)` track the `G_sep`/`Δacc` transition? Correlate `ΔM_mean` with `G_sep_mean` and `Δacc_mean`; set `mechanism_tangent_tracks`.
8. **Step 7 — pairwise.** Repeat Tests 1–4 per pair. Tabulate which pairs pass. (`0↔180`, `0↔270`, `90↔180`, etc. are expected to differ.)
9. **Step 8 — nulls.** phase-shuffle, pair-shuffle, cloud-isotropic; recompute Tests 1–2 (mean and pairwise).
10. **Step 9 — window robustness.** Tests 1–3 across `--windows`; report W-stability.
11. **Step 10 — tables, figures, decision, readme.**

---

## Statistical detail

- **Crossings:** linear-interpolate to zero between bracketing core-range points. Report `no_crossing` if none. If the only crossing requires `−0.40`, set `render_limit_confounded`.
- **Bootstrap:** `Δacc` resamples trials (paired by trace); geometry resamples phases (and sessions/readout splits if present); join per iteration for `ΔL` CI. n=2000.
- **Coincidence verdict:** `ΔL` 95% CI contains 0 **and** `|ΔL_median| <` one grid step (0.05), within the core range.
- **Specificity verdict:** partial `ρ(Δacc, G_sep | S_best)` CI excludes 0 **and** nested ΔR² ≥ 0.15 with AIC favoring the `+G_sep` model.
- **mechanism_tangent_tracks:** `ΔM_mean` correlates with `G_sep_mean` (and ideally `Δacc_mean`) with CI excluding 0, **and** the `ΔM` transition LogMAR is within one grid step of the `G_sep` transition.
- **Null pass:** real coincidence/slope exceed phase- and pair-shuffle nulls (empirical p).

---

## Required output tables

### `geometry_curve.csv`
`L, render_limit_control, condition, Sep_avg, S_center, S_stab, S_cloud_mean, S_best, M_mean, mean_tangent_alignment, n_phase, n_pairs, metric`

### `geometry_pairwise.csv`
`L, pair, G_sep_pair, S_center_pair, S_stab_pair, S_cloud_mean_pair, M_pair, dM_pair`

### `function_curve.csv`
`L, condition, window, accuracy, acc_ci_low, acc_ci_high, delta_acc_mean, delta_ci_low, delta_ci_high, n_trials`

### `function_pairwise.csv`
`L, pair, window, delta_acc_pair, ci_low, ci_high, n_trials`

### `keystone_contrasts.csv`
`L, window, G_sep_mean, dM_mean, delta_acc_mean, S_best`

### `keystone_tests.csv`
`test, level(mean|pair), pair, observable, statistic_name, value, ci_low, ci_high, null_p, window, verdict_component`
Components at minimum: `coincidence_dL`, `continuous_slope`, `continuous_R2`, `continuous_spearman`, `partial_rho_given_Sbest`, `nested_delta_R2`, `nested_AIC_delta`, `mechanism_tangent_tracks`, plus `*_phase_shuffle`, `*_pair_shuffle`, `*_cloud_isotropic`, and per-pair rows.

### `keystone_decision_table.csv`
One row.
`run_label, primary_window, L_func_core, L_geom_core, dL, dL_ci, continuous_significant, specificity_passed, mechanism_tangent_tracks, nulls_passed, window_stable, pairs_passing, render_limit_confounded, decision_label, manuscript_implication, next_action`

### `keystone_readme.md`
Answers: (1) coincidence within core range? (2) continuous prediction? (3) beyond `S_best`? (4) does `ΔM` track the transition (Tier 2)? (5) which pairs pass? (6) exceed nulls? (7) W-stable? (8) `−0.40` confound? (9) decision label and whether Figure 4 leads with geometry.

---

## Required figures

- **Fig A — keystone overlay (mean).** `Δacc_mean(L)` and `G_sep_mean(L)` on twin axes, zero line, both crossings + CIs, `−0.40` shaded as control. → paper panel 4F.
- **Fig B — continuous.** Scatter `Δacc_mean` vs `G_sep_mean`, fit, colored by `L`.
- **Fig C — difficulty control.** `Δacc_mean` vs `G_sep_mean` after partialling out `S_best`; side panel vs `S_best` alone.
- **Fig D — Tier-2 mechanism.** `ΔM_mean(L)` overlaid on `G_sep_mean(L)`.
- **Fig E — pairwise small multiples.** Per-pair overlay of `Δacc_pair` and `G_sep_pair`; highlight passing pairs.
- **Fig F — nulls.** Real `ΔL`/slope vs shuffle null distributions.

---

## Decision logic

```text
geometry_predicts_global_crossover
    coincidence TRUE (core range, not dependent on −0.40)
    AND continuous_significant TRUE (mean)
    AND specificity_passed TRUE (G_sep beyond S_best)
    AND mechanism_tangent_tracks TRUE
    AND nulls_passed TRUE
  → Figure 4 leads with geometry: the translation-tangent geometry predicts AND explains the crossover.

geometry_predicts_crossover_via_sampling_not_tangent      [added in review]
    as above but mechanism_tangent_tracks FALSE
  → Cloud-separability predicts the crossover, but not via the translation-tangent mechanism.
    The equivariant-manifold claim is unsupported; this is "sampling helps, dressed as geometry."
    Lead with the acuity/sampling result; keep the tangent geometry as descriptive supplement.

geometry_partially_predicts_pairwise_benefit
    global mean fails, but a subset of pairs passes coincidence + specificity (+ tangent if available)
  → Report which pairs and why; biologically meaningful but not a global claim.

geometry_tracks_difficulty_not_mechanism
    continuous_significant TRUE but specificity_passed FALSE (S_best explains Δacc; G_sep adds nothing)
  → Geometry descriptive of difficulty; lead with acuity; geometry to supplement.

geometry_descriptive_not_predictive
    geometry curves well-behaved but neither coincide nor predict Δacc
  → Drop the predicts-function claim; geometry is a separate descriptive result.

render_limit_confounded
    only crossing/effect requires −0.40, or curves saturate within the core range
  → Inconclusive; refine rendering or add finer sizes inside the core; do not call a verdict.

underpowered_or_missing_inputs
    CIs wider than the core range, or cached μ/J/accuracy unavailable/incompatible and not regenerable
  → Fix inputs / add resolution; not a null.
```

---

## Implementation guardrails

1. **Firewall:** geometry from `μ, J` only; never the decoder or its noise model.
2. Geometry and function on the **same** grid and the **same** real-FEM clouds.
3. Decoder-free metric only (Euclidean or one fixed global diagonal).
4. `G_sep` is the functional mirror, not the Jacobian claim; the Jacobian/equivariance thesis lives or dies on Tier 2 (`mechanism_tangent_tracks`). State this in the readme.
5. Coincidence is necessary, not sufficient; specificity (beyond `S_best`) + Tier 2 are what license the causal-geometry claim.
6. `−0.40` is a boundary control, never a load-bearing point.
7. Report mean and pairwise; do not let pair structure be averaged away.
8. Verdict at primary `W`, with W-stability reported; no window cherry-picking.
9. Model-only; do not touch recorded data.
10. No new geometry variants if the result is null — report the predefined label and stop.

---

## Minimal acceptance criteria

```text
cache_availability_report.csv
grid_reconciliation.csv
geometry_curve.csv
geometry_pairwise.csv
function_curve.csv
function_pairwise.csv
keystone_contrasts.csv
keystone_tests.csv
keystone_decision_table.csv
keystone_readme.md
Figures A–F
```
Readme ends with one decision label and an explicit statement of whether Figure 4 leads with geometry.

---

## Final stop rule

This is the adjudicating keystone test. Run once, on the defined grid (core range `−0.35`..`−0.20`, `−0.40` as control), primary window, predefined nulls, three difficulty baselines, mean and pairwise, with the Tier-2 tangent check. Report the verdict. Sanctioned reruns are implementation failures only: grid mismatch, firewall violation, degenerate `μ`/`J`, missing cache that must be regenerated, or accuracy failing an easy positive control (large-LogMAR near ceiling). Do not widen the slice or add geometry variants to chase a positive result.
