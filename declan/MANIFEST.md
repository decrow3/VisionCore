# declan Manifest

Last curated: 2026-06-16.

This is a human-maintained map of the `declan/` workspace. It is intentionally
ordered newest-to-oldest in reading order where chronology is recoverable. The
repo has several live threads moving in parallel, so dates should be read as
"when this thread became active or was last materially touched", not as a strict
scientific dependency graph.

For the companion narrative of motivations, outcomes, and revised
interpretations, see `ANALYSIS_NARRATIVE.md`.

## How To Update

Add new work near the top, under the current date or a new date block. Prefer a
short thread-level description plus a file list. If a folder grows a README,
make this manifest point to that README and keep only the navigation summary
here. If a script becomes obsolete, mark it `historical` rather than deleting
the entry.

Chronology sources used here:

- Git history for committed `declan/` changes.
- File mtimes for uncommitted and generated artifacts.
- Existing README, handoff, plan, prescription, and spec documents.

Known caveat: generated caches/results often have older mtimes than the script
or handoff that interprets them.

## Current Active / Uncommitted Work

As of 2026-06-16, active uncommitted work has moved beyond the 2026-06-09
Figure 4/Figure 5 split. The most important files to check first are:

- `active_sensing_roadmap_after_vernier_fixation_image_structure.md`: current
  active-sensing synthesis after Vernier, fixation-regime, BackImage
  image-structure, scaled twin drift-geometry adjudication, and the
  input-whitening negative result. It now also records the n=256 local
  Gabor/pyramid latent-information scale sweep, effective-scale audit, active
  optimized seed-dependence replication, and the cleaned n=256 aggregate
  natural-image FEM information result.
- `backimage_aggregate_fem_information_plan.md`: newest Figure-level candidate
  plan for testing whether the empirical FEM distribution improves ensemble
  natural-image representation over static, OU-matched, Brownian-matched, and
  shuffled controls, with effective-scale, clipping, motion-energy, temporal-PC
  response summaries, signal-motion overlap, and shared/fixed decoder
  regularization guardrails. The current substantial result is
  `backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched`;
  use its corrected `incremental_static_plus_motion_relids` posthoc for
  incremental static-plus-motion claims.
- `active_sensing_unit_space_provenance.md`: response-space ledger separating
  16-channel matched/session results, sampled-population results, and full
  756-channel canonical results. Use it before comparing pose-aware,
  geometry-aware, and pose-blind observers. General rule: matched/session
  readouts are for empirical-session or individual-neuron comparability; the
  canonical large digital twin is the default for normative/mechanistic
  population-level questions.
- `fixation_statistics_by_stimulus/`: reviewed fixation-regime and BackImage
  image-structure analysis package, including the twin drift-geometry runner.
  `run_backimage_twin_drift_geometry.py` now also includes the conditional
  fixation objective family: pixel isophote/stability cost, response-stability
  cost, saturating path-refresh benefit, and refresh-plus-stability tradeoffs.
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/`: reviewed
  BackImage/fixation-statistics outputs. The scaled `29`-session twin
  drift-geometry adjudication found raw local edge orientation strongest
  (`+0.182` session mean cos2, `23/29` positive, random-axis `p_ge = 0.0004`).
  Current optimized PA/PB/Pareto twin axes did not beat raw edge geometry;
  `optimized_PB - raw_edge_axis` was `-0.201` with CI `[-0.348, -0.064]`.
  Provenance audit: the folder's `n256` label refers to `max_windows=256`;
  saved metadata reports `twin_population_n=64`, so this is a sampled
  population diagnostic rather than a full 756-channel canonical result.
  Summary artifacts live under
  `backimage_twin_drift_geometry_scaled_n256_twin_axis_only/summary/`.
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_conditional_fixation_objectives_proxy_n256/`:
  first conditional-fixation proxy run. Pixel isophote/stability matched the
  raw-edge benchmark but did not clearly beat it: `optimized_pixel_isophote`
  session mean cos2 `+0.200`, raw edge `+0.182`, paired delta `+0.018` with CI
  `[-0.044, +0.084]`.
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_conditional_fixation_objectives_twin_axis_only_n256/`:
  comparable true V1 response-stability conditional-fixation run. Main result:
  V1 response stability underperformed raw edge (`optimized_response_stability`
  session mean cos2 `-0.019`; paired delta vs raw edge `-0.201`, CI
  `[-0.349, -0.069]`). The cache-first residual summary in
  `conditional_residual_summary/` did not rescue this via confidence strata:
  high-confidence response-stability windows still had negative deltas versus
  raw edge.
- `Non_circular_FEM_information_tests_prescription.md`: non-circular Figure 5
  extension plan centered on input whitening, recorded pose-aware information,
  spatial-frequency localization, and sustained accumulation.
- `active_sensing_movie_information/run_input_whitening_optimum.py` and
  `active_sensing_movie_information/summarize_input_whitening_optimum.py`:
  superseded pooled temporal-PSD input-statistics test. Output:
  `outputs/active_sensing_movie_information/input_whitening/`. Main result:
  larger retinal motion spreads pooled temporal power and increases the old
  entropy/flatness metric, but this is not a faithful Rucci-style spatial
  power-law whitening test.
- `active_sensing_movie_information/run_rucci_powerlaw_whitening_audit.py`:
  newer non-circular input-statistics audit. Outputs include
  `outputs/active_sensing_movie_information/rucci_powerlaw_whitening_smoke_v2/`
  and the longer pending
  `outputs/active_sensing_movie_information/rucci_powerlaw_whitening_rawfix_n64_allimages_v2/`.
  It measures spatial-frequency flattening of frame-to-frame modulation power,
  keeps `amplitude_scale` separate from `diffusion_scale = amplitude_scale^2`,
  and writes trace/image-level summaries plus power-gated decisions. Smoke
  runs show total modulation power increasing with motion, while Rucci-style
  spatial flattening peaks at small nonzero motion.
- `active_sensing_movie_information/summarize_fem_scale_tradeoff.py`:
  cache-first tradeoff extension combining input whitening, covariance-aware
  pose costs, Vernier acuity, and analytic motion-cost proxies. Output:
  `outputs/active_sensing_movie_information/fem_scale_tradeoff/`. First pass:
  covariance and diffusion costs counter the upper-bound whitening optimum but
  mostly overshoot to small scales; one-sided above-biological window costs can
  place the optimum at `D_scale=1` under strong weights, partly by construction.
  Treat this as a directional diagnostic, not an explanatory model.
- `Covariance_aware_FEM_optimality_analysis_prescription.md`: covariance-aware
  movement-scale prescription separating pose-aware information from
  pose-blind covariance penalties.
- `jake/twininfo/covariance_optimality.py` and
  `jake/twininfo/run_covariance_optimality.py`: implemented covariance-aware
  scaled-trajectory machinery for production `jake.twininfo` runs; production
  run `covopt_full_gpu1` completed 3888/3888 rows. A 2026-06-13 audit fixed
  the pose-aware path so it now uses the same covariance-Fisher/ridge code path
  as pose-blind with zero extra movement covariance.
- `active_sensing_movie_information/summarize_covariance_optimality.py`:
  summary, decision-table, and plotting helper for covariance-optimality
  outputs.
- `outputs/active_sensing_movie_information/covariance_optimality/covopt_full_gpu1/`:
  completed covariance-aware operating-regime summary. Main result: empirical
  `D=1` usually lies on a high-efficiency plateau rather than a sharp optimum;
  corrected pose-aware minus pose-blind covariance Fisher gaps are consistently
  positive, especially for microsaccade traces, but random amplitude controls
  match or exceed real.
- `outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/`
  and `.../posthoc_geometry_hierarchy/`: canonical sampled-population
  covariance hierarchy (`N=256`) with gap-closure posthoc tables and figures.
  Current interpretation: strong low-rank covariance rescue, translation-
  geometry specificity untested. At empirical `D=1`, the current
  `cov_geometry_aware_k` closes most of the pose-aware versus pose-blind gap,
  but it is implemented as the top eigenspace of the movement covariance being
  corrected, so it is an oracle top-PC correction rather than an independent
  compact translation-tangent basis.
- `outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/tangent_geometry_pathfinding_d1/`:
  exact D=1 partial-overlap tangent pathfinding on the 116 units shared between
  the `N=256` covariance cache and the canonical 756-channel Figure 4/TFTS
  tangent manifest. Tangent subspaces beat random and usually unit-shuffled
  tangent controls, but oracle top movement-PC remains much better
  (`topPC_much_better_than_tangent` for all 32 decision rows). Current
  interpretation: compact translation tangents capture a non-random slice of
  the covariance penalty, but the strong hierarchy rescue is still mostly
  low-rank movement-covariance/top-PC accounting. Next target: evaluate
  `cov_tangent_geometry_aware_k` in a fully matched response space, preferably
  the full canonical 756-channel model for population-level questions.
- `outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/tangent_geometry_pathfinding_d1_cache_tangent/`:
  same-cache tangent sanity check using Jacobian columns from the exact
  116-unit evaluation cache. This does not rescue the functional readout
  mechanism under the original residual semantics, but that interpretation is
  now paused.
- `declan/active_sensing_movie_information/run_noise_side_closure_audit.py` and
  `outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/noise_side_closure_audit_sampled/`:
  corrected noise-side-only closure audit using `Sigma_k = R Sigma_FEM R.T` for
  every basis and leaving task derivatives/responses untouched. Sanity checks
  pass (`D=0` null and synthetic `Sigma_FEM = J Sigma_e J.T`). Sampled D=1
  result: manifest tangent is stronger under corrected semantics, while cache
  tangent still leaves high residual pooled-residual covariance trace. Current
  status: pause the compact functional useful-negative label; treat the sampled
  result as evidence for possible `J`/`Sigma_FEM` provenance mismatch pending a
  full corrected audit. The compact structural branch remains promotable as a
  FEM-linked covariance structure mechanism, not as a demonstrated functional
  readout mechanism.
- `declan/active_sensing_movie_information/run_covariance_target_provenance_audit.py`,
  `outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/covariance_target_provenance_d1/`,
  and
  `outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/covariance_target_provenance_d1_highk/`:
  trace-capture provenance audit across exact reconstructed `J Sigma_e J.T`,
  within-pair movement covariance, and pooled-residual covariance. This did not
  run additional closure variants. At `k=20`, cache tangent captures about
  `0.63` of exact `J Sigma_e J.T` trace and `0.59` of pooled/within-pair trace,
  while oracle top-PC captures about `0.89` and `0.97`. At `k=116`, the current
  cache-tangent construction still captures only about `0.74` of exact
  `J Sigma_e J.T` and `0.64` of pooled/within-pair trace, while manifest/full
  response-space basis captures all trace. Updated status: compact covariance
  functional branch is paused pending covariance-provenance and cache-basis
  decomposition; do not interpret the previous useful-negative label yet.
- `declan/active_sensing_movie_information/run_covariance_projection_debug.py`
  and
  `outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/projection_debug_d1/`:
  minimal projection/provenance assertions. Identity and `orth(J_exact)` pass
  on the same 116-unit covariance targets; `orth(J_exact)` gives trace capture
  `1.0`, residual trace near zero, direct J residual near zero, closure `1.0`,
  and rank `116`. The existing cache tangent basis is rank `115` at `k=116`
  because it is built from unit-centered J columns, so `U U.T` is not identity
  and the uncentered J residual is about `0.257`. Unit hashes match
  (`baac97be3d382ca4`), identifying a centering/response-space definition issue
  rather than unit-order provenance mismatch.
- `declan/active_sensing_movie_information/run_uncentered_j_tangent_closure_audit.py`
  and
  `outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/uncentered_j_tangent_closure_d1_sample6_k2_20_116/`:
  focused corrected noise-side-only closure pass comparing oracle top-PC,
  uncentered exact-J tangent, centered exact-J tangent, manifest tangent,
  random, and unit-shuffled manifest bases. Sampled D=1 result: uncentered
  exact-J nearly matches oracle for exact `J Sigma_e J.T` and pooled/within
  covariance (`k=20` pooled closure `0.998` versus oracle `0.999`), while
  centered exact-J remains poor because it omits the common mode. Updated
  status: positive matched-cache mechanism. Uncentered local translation
  sensitivity explains the movement-covariance rescue in the matched cache. The
  independent canonical/manifest tangent basis remains partial and is not
  sufficient for the full rescue.
- `declan/active_sensing_movie_information/summarize_uncentered_j_tangent_closure.py`
  and
  `outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/uncentered_j_tangent_closure_d1_sample6_k20_summary/`:
  final focused k=20 summary table/figure for closure, residual trace, and
  signal fraction. At `k=20`, uncentered exact-J is essentially oracle-like
  across exact-J, pooled-residual, and within-pair targets; centered exact-J
  closure remains low for pooled/within (`0.241`), and manifest tangent closure
  is intermediate (`0.502`) with low signal fraction (`0.210`). Main message:
  the common-mode uncentered tangent is essential.
- `declan/active_sensing_movie_information/rebuild_manifest_tangent_basis_conventions.py`
  and
  `outputs/active_sensing_movie_information/compact_basis_exports/manifest_tangent_basis_conventions/`:
  diagnostic rebuild of canonical TFTS manifest bases from raw cached `bx/by`
  tangent maps. The manifest basis can be rebuilt uncentered without rerunning
  the twin, but the historical centered convention was centering across source
  tangent samples per unit, not removing the response-space common mode within
  each tangent vector. At default `0.25` arcmin, uncentered-vs-historical
  overlap is `0.999` at `k=2` and `k=20`; this convention change alone is
  unlikely to rescue independent manifest closure.
- `structured_translation_decoder_analysis.md` and
  `compact_retinal_translation_geometry/run_windowed_siamese_relative_decoding.py`:
  structured decoder framing plus gain-orthogonal/rank-1 gain-null additions to
  the windowed Siamese decoder.
- `compact_retinal_translation_geometry/run_tejas_style_eyepos_decoder.py`:
  permissive absolute eye-position decoder sanity check.
- `forward_twin_reafferent_denoising_analysis.md` and
  `forward_twin_reafferent_denoising/run_forward_twin_reafferent_denoising.py`:
  single-trial forward-twin reafferent denoising hypothesis and runner; latest
  matched diagnostics are useful but do not pass the shuffled-eye specificity
  gate.
- `content_routed_retinal_registration_analysis_plan.md` and
  `compact_retinal_translation_geometry/run_correct_chart_swap_alignment.py`:
  correct-chart versus wrong-chart content-routed retinal-registration analysis.
  Current result is a narrow gain-bottom compact positive, with all-unit primary
  still diagnostic/null.
- `vernier_active_sensing_analysis_plan.md` and `vernier_active_sensing/`:
  controlled Vernier hyperacuity branch. First pass supports phase-cloud
  sampling over static center, not unique biological trajectory order.
- `fig1/generate_fig1*.py`, `fig2/generate_figure2_3_combined.py`, and
  `fig3/generate_figure3_combined.py`: refreshed manuscript figure assembly.
- `fig4_cov_TFTS/plot_covariance_binning_sweep_panel.py`: standalone
  covariance-closure response-window stability panel.

Treat the new prescriptions as claim discipline, not finished results. The
covariance-optimality and input-whitening routes now have interpreted
production outputs; other new runners should still be treated as implemented
routes whose outputs need their own review before promotion.

Recent numerical-audit note: the compact covariance-closure ratio can be
slightly above `1.0` because the full and compact sources are compared as
separately constructed sources, not as a strict nested full-versus-subspace
estimator. Read it as retained closure, not compact outperforming full. The
large Jacobian magnitude mismatch, stimulus-specific 100% intervention,
E-optotype SSI/decoding split, pooled subspace-ablation ambiguity, and
near-perfect within-image displacement decoding are historical guardrails rather
than active headline claims.

Current content-routing note: correct-chart alignment is no longer a pure null,
but it is also not yet a promoted bridge. The broad all-unit primary failed its
gate; the gain-bottom compact subset passed wrong-chart and control comparisons.
Treat this as a specific diagnostic that needs pre-registered unit selection or
replication before becoming a claim.

## Chronology: Newest To Oldest

### 2026-06-15: BackImage n=256 Latent Scale Sweep, Seed Replication, and Aggregate FEM Information Plan

Primary locations:

- `free_viewing_latent_information_test_plan.md`
- `active_sensing_roadmap_after_vernier_fixation_image_structure.md`
- `backimage_aggregate_fem_information_plan.md`
- `fixation_statistics_by_stimulus/run_backimage_latent_information_screen.py`
- `fixation_statistics_by_stimulus/summarize_backimage_latent_information_screen.py`
- `fixation_statistics_by_stimulus/audit_backimage_latent_real_random.py`
- `fixation_statistics_by_stimulus/summarize_backimage_aggregate_cache_proxy.py`
- `fixation_statistics_by_stimulus/run_backimage_aggregate_fem_information.py`
- `fixation_statistics_by_stimulus/summarize_backimage_aggregate_incremental_motion.py`
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_latent_information_scalesweep_n256_rel0125-2_rand8_delta/`
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_latent_information_scalesweep_n256_rel0125-2_rand8_delta/aggregate_cache_proxy_full_postfix_nested/`
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_latent_information_scalesweep_n256_rel0125-2_rand8_delta_seed1_manifest_optimized_tb2/`
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/`
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/incremental_static_plus_motion_relids/`

Purpose: complete the locked canonical n=256 BackImage Gabor/pyramid
latent-information scale sweep, audit effective scale/clipping/subsampling
stability, patch the runner for replayable seed-dependence checks and faster
canonical trace batching, launch an optimized same-window seed-1 replication,
and formalize a broader aggregate FEM information plan.

Completed n=256 scale sweep:

- Canonical 756-unit twin, fixed Gabor/pyramid local fields,
  `pose_blind_delta_mean`, rand8, scales `0.125x`, `0.25x`, `0.5x`, `1x`, and
  `2x` observed RMS.
- Gabor `k=4`, `0.25x`: real-minus-random `+3.48`,
  CI `[+0.75, +6.87]`; unclipped subset `+2.85`, CI `[+0.21, +6.25]`.
- Pyramid `k=8`, `0.25x`: real-minus-random `+2.19`,
  CI `[+0.62, +4.18]`; unclipped subset `+1.86`, CI `[+0.45, +3.67]`.
- Gabor `k=4`, `1x`: positive but guarded, `+2.59` with CI crossing zero.
- Pyramid `k=8`, `1x`: global weak, but unclipped subset `+1.40`,
  CI `[+0.05, +2.71]`.
- Clipping rises with nominal scale: `1.2%`, `3.9%`, `9.4%`, `18.8%`, and
  `40.2%`.
- Subsampling explains n=64/n=128 variability: n=64 Gabor `k=4`, `1x`
  subsamples can be negative; n=128 Gabor `k=4`, `0.25x` is usually positive.
- Leave-session-out is reassuring at `0.25x`: Gabor `k=4`
  `+2.17/+3.52/+4.09`; pyramid `k=8` `+1.41/+2.25/+2.38`.

Runner/posthoc updates:

- `run_backimage_latent_information_screen.py` now supports `--window-manifest`
  replay, stable `source_row` output for new runs, fixed-alpha sensitivity mode,
  canonical trace batching via `--twin-trace-batch-size`, and
  `--check-trace-batch-equivalence`.
- `summarize_backimage_latent_information_screen.py` reports scale clipping and
  merges clipping fractions into candidate/scale summaries.
- `audit_backimage_latent_real_random.py` adds effective-scale splits,
  clipped/unclipped real-minus-random summaries, subsampling, leave-session-out,
  and regime stratification.

Active optimized replication:

- Output:
  `backimage_latent_information_scalesweep_n256_rel0125-2_rand8_delta_seed1_manifest_optimized_tb2`.
- Same physical windows as the completed n=256 run via `--window-manifest`,
  random-axis seed changed to `1`, canonical trace batching enabled, and
  trace-batch equivalence preflight enabled.
- The first trace-batch-8 launch exceeded GPU memory during preflight; the live
  run uses `--twin-trace-batch-size 2` and `--twin-batch-size 48`.

New aggregate plan:

- `backimage_aggregate_fem_information_plan.md` shifts the figure-level
  question from exact per-fixation axis optimality to distributional adaptation:
  whether empirical FEM motion distributions improve ensemble natural-image
  representation compared with static, OU-matched, Brownian-matched, and
  shuffled controls.
- The backup note is preserved as `Global Iz FEMs.md`.
- `summarize_backimage_aggregate_cache_proxy.py` is a cache-first bridge: it
  reuses the completed n=256 latent and response arrays to compute aggregate
  decoding/covariance proxy summaries without new twin inference. The script
  now keeps raw random axes and defines `random_axis_mean` as the mean of
  decoded random-axis scores. The post-fix full proxy folder is
  `aggregate_cache_proxy_full_postfix_nested`: motion-versus-static is strongly
  positive and grows with scale, while real-vs-random specificity is narrow
  (`0.25x` Gabor `k=4` and pyramid `k=8` survive with positive CIs).
- Guardrails: report effective RMS and clipping for every nominal scale, match
  motion energy, include paired/unpaired image-trace modes, use shared/fixed
  decoder regularization for figure-level claims, treat OU as the primary
  synthetic null and Brownian as secondary, use temporal PCs as the primary
  pilot response summary, promote signal-motion subspace overlap, call
  deterministic ridge scores linear decodability/information proxies, and do
  not interpret largest-motion wins as biological optimality.

Substantial aggregate result:

- Output:
  `backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched`.
- Corrected incremental posthoc:
  `incremental_static_plus_motion_relids`. The first automatic
  `incremental_static_plus_motion` folder used old-style scale IDs in the launch
  command and has empty gain tables.
- Canonical `756`-unit twin, `256` images, `K=4`, grouped-by-image CV,
  drift-only/common-unclipped trace bank, source traces reused across scales,
  empirical/OU/Brownian/rotated controls, scales `0.25x`, `0.5x`, `1x`,
  `1.5x`, and `2x`, and Gabor/pyramid local-field latents at `k=4,8`.
- Motion bookkeeping was clean: `151/256` drift-only trace sources accepted,
  median effective/requested RMS `1.0` for every family/scale, and clipped
  fraction `0.0` throughout.
- Primary temporal-PCA incremental result: empirical motion added decoding
  signal beyond static responses. Gabor `k=4` gains were `+14.31`, `+13.04`,
  `+9.10`, `+9.98`, and `+9.07` from `0.25x` through `2x`; all CIs were
  positive. Pyramid `k=8` gains were smaller but consistently positive:
  `+5.20`, `+4.89`, `+3.93`, `+4.44`, and `+4.21`.
- Empirical beat OU robustly across scale. Gabor `k=4` empirical-minus-OU
  incremental-gain advantages were `+21.24`, `+19.59`, `+17.16`, `+18.69`,
  and `+18.03` from `0.25x` through `2x`.
- Empirical beat Brownian and rotated most clearly at small scales. For Gabor
  `k=4`, `0.25x` empirical-minus-Brownian was `+10.52` and
  empirical-minus-rotated was `+15.27`; at `0.5x`, they were `+7.89` and
  `+11.21`. Brownian became competitive at `1x-2x`.
- Interpretation: empirical drift-like statistics add feature-decoding signal
  and beat OU-like confined motion, with the most specific advantage at
  `0.25x-0.5x`. This is not a simple "more motion is better" result, but it is
  also not proof of exact trajectory optimality.

Status: `Open / local I_z has stable small-scale support; optimized
same-window seed replication running; aggregate n=256 result supports
readout- and scale-dependent empirical-drift benefit over OU`.

### 2026-06-14: BackImage Latent-Information Fixes and Edge-Parallel Stability Synthesis

Primary locations:

- `free_viewing_latent_information_test_plan.md`
- `active_sensing_roadmap_after_vernier_fixation_image_structure.md`
- `fixation_statistics_by_stimulus/run_backimage_latent_information_screen.py`
- `fixation_statistics_by_stimulus/posthoc_backimage_twin_stability_metric_audit.py`
- `fixation_statistics_by_stimulus/summarize_backimage_twin_stability_metric_audit.py`
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_latent_information_pathfinder_fixall_n64_rel0125-05_rand8_delta/`
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_latent_information_pathfinder_gabor_realrand_n128_rel0125-05_rand8_nogrd/`
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_twin_stability_metric_audit/`

Purpose: review, patch, and rerun the BackImage latent-information branch after
the Gabor/pyramid implementations looked suspicious; then audit whether the
twin edge-parallel stability metric was dominated by response scale.

Implementation updates:

- Gabor local fields now include even, odd, and amplitude maps on the local
  grid.
- Pyramid local fields use the expanded local grid.
- Model outputs are cropped back to the requested trace length before observer
  construction.
- Delta observers subtract the aligned static response for the same window.

Post-fix latent results:

- First pathfinder:
  `backimage_latent_information_pathfinder_fixall_n64_rel0125-05_rand8_delta`.
  At `0.125x` observed RMS, canonical 756-unit twin, `k=4`, rand8, Gabor
  pose-blind delta gave real-minus-random `+9.02`
  CI `[+1.56, +18.15]` and real-minus-edge `+11.58`
  CI `[+1.59, +24.55]`.
- Pyramid pose-blind delta was directionally similar but noisier:
  real-minus-random `+10.28`, CI `[-0.53, +26.07]`; real-minus-edge
  `+25.04`, CI `[-3.78, +73.38]`.
- The apparent larger Gabor-only run,
  `backimage_latent_information_pathfinder_gabor_realrand_n128_rel0125-05_rand8_nogrd`,
  is not a clean post-fix replication. Its saved Gabor local field has shape
  `(128, 384)`, while the fixed 8x8 even/odd/amplitude Gabor local field has
  shape `(N, 4608)`, and it used absolute rather than delta observers.
- The stage2 run included pyramid and scales up to `2x`, but it was still n=64,
  used the older 4x4 feature dimensionality, and used absolute observers. A
  larger clean fixed-Gabor/fixed-pyramid delta replication had not yet been run
  at that point.
- Clean n=128 canonical replication:
  `backimage_latent_information_cleanrep_n128_rel0125-05_rand8_delta`.
  This used canonical 756 units, fixed Gabor local fields `(128, 4608)`, fixed
  pyramid local fields `(128, 3072)`, `pose_blind_delta_mean`, rand8, and
  `0.125x/0.25x/0.5x` observed RMS. Primary Gabor `k=4`, `0.125x`:
  real-minus-random `+3.31` CI `[-0.14, +8.57]`, p(delta<=0) `0.0396`;
  real-minus-edge `-0.36` CI `[-2.25, +1.09]`; edge-minus-random `+3.67`
  CI `[+0.08, +9.28]`. Larger relative scales produced several real-vs-edge
  positives, including Gabor `k=4`, `0.5x` `+6.60` CI `[+1.53, +11.83]`,
  pyramid `k=4`, `0.5x` `+7.26` CI `[+2.35, +12.08]`, and pyramid `k=8`,
  `0.25x` `+8.57` CI `[+1.62, +20.85]`.

Twin stability metric audit:

- Endpoint cache:
  `backimage_twin_stability_metric_audit/twin_endpoint_tail_vectors.npz`.
- Signed edge-parallel stability was positive across pixel and twin metrics:
  pixel session mean `+300.5` CI `[+172.8, +408.8]`; twin raw MSE
  `+0.0004545` CI `[+0.0003716, +0.0005432]`; response-norm `+0.02456`
  CI `[+0.01993, +0.02931]`; per-rate `+0.003688`
  CI `[+0.002902, +0.004501]`; full-cov whitened `+0.1706`
  CI `[+0.1511, +0.1890]`.
- Pixel and twin signed advantages agree across windows: full-cov whitened
  `r = +0.277` CI `[+0.168, +0.417]`; diagonal-whitened `r = +0.287`
  CI `[+0.139, +0.419]`.
- The old relative twin disruption score's wrong-direction trend is now treated
  as metric-fragile; signed normalized disruption does not support a robust
  wrong-direction claim.

Status: `Open / local-preservation result strengthened; latent-information
optimizer branch remains scale-dependent`.

Current claim boundary:

```text
BackImage drift is not yet explained by global feature-information
maximization. The defensible result is that edge-parallel small retinal motion
preserves local pixel and V1-twin structure. The corrected Gabor/pyramid
`I_z` branch is alive but not yet figure-settling: the clean n=128 run gives a
primary small-scale Gabor real-vs-random signal, no primary real-vs-edge win at
`0.125x`, and several real-vs-edge positives by `0.5x`.
```

Next gate:

- Run a locked Gabor/pyramid scale sweep through `1x` observed RMS before
  continuous trajectory optimization. Classify the `1x` outcome as
  real-specific, edge-structured, generic-motion, edge-avoidance, or
  no-feature-benefit.
- Treat edge-parallel signed stability as the robust BackImage support result,
  and raw edge geometry as the baseline any model objective must beat.

### 2026-06-13: Active-Sensing Roadmap, Fixation-Regime, BackImage Image-Structure, and Input-Whitening Branch

Primary locations:

- `active_sensing_roadmap_after_vernier_fixation_image_structure.md`
- `fixation_statistics_by_stimulus/`
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/`
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure_reviewed_v2_screenfiltered/`
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_twin_drift_geometry_pilot_twin_axis_only/`
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_twin_drift_geometry_scaled_n256_twin_axis_only/`
- `active_sensing_movie_information/run_input_whitening_optimum.py`
- `active_sensing_movie_information/summarize_input_whitening_optimum.py`
- `active_sensing_movie_information/run_rucci_powerlaw_whitening_audit.py`
- `active_sensing_movie_information/summarize_fem_scale_tradeoff.py`
- `outputs/active_sensing_movie_information/input_whitening/`
- `outputs/active_sensing_movie_information/rucci_powerlaw_whitening_smoke_v2/`
- `outputs/active_sensing_movie_information/fem_scale_tradeoff/`

Purpose: update the active-sensing story after the Vernier hyperacuity branch,
fixation-statistics-by-stimulus analysis, BackImage local image-structure
analysis, scaled BackImage twin drift-geometry adjudication, and completed
input-whitening analysis.

Current synthesis:

- FEMs are best framed as context-dependent retinal sampling whose utility
  depends on local image geometry, movement scale, and observer coordinate frame.
- Vernier results promote phase sampling under pose-aware readout, but demote
  exact trace order and compact-aware pose-blind rescue.
- The old pooled temporal-PSD whitening branch is superseded as a
  Rucci-whitening test. It showed temporal power spreading with larger motion,
  not spatial power-law compensation. The newer Rucci-style audit shows a
  dissociation in smoke runs: total modulation power peaks at large motion, but
  spatial power-law flattening peaks at small nonzero motion. Whitening still
  does not set biological amplitude by itself; it has become a multi-objective
  constraint rather than one scalar optimum.
- The first cache-only tradeoff summary shows that adding covariance, acuity,
  or generic diffusion costs can move the optimum away from the whitening upper
  boundary, but not automatically back to biology. Pose-blind covariance and
  diffusion costs tend to overshoot toward very small scales; a one-sided
  above-biological window penalty can recover `D_scale=1` under strong weights,
  but is not explanatory as written. The next concrete test is
  Rucci-style spatial power-law whitening and V1 temporal-sensitivity-weighted
  whitening.
- Fixation statistics differ substantially across stimulus/behavioral regimes;
  fixRSVP is a tight fixation-maintenance regime, not a universal FEM policy.
- BackImage scalar local-image features do not robustly predict scalar FEM
  metrics, but drift/fixation-cloud axes show modest local edge/spectral
  alignment.
- The scaled twin drift-geometry run did not show model-specific explanatory
  power beyond raw edge geometry: `raw_edge_axis` session mean cos2 was
  `+0.182`, while `optimized_PB` was `-0.019`, `optimized_PA` was `-0.008`,
  and `optimized_Pareto_lambda_0.5` was `-0.010`. The paired
  `optimized_PB - raw_edge_axis` delta was `-0.201`, CI
  `[-0.348, -0.064]`.

Status: `Open / scaled BackImage drift-geometry adjudication completed; input-whitening branch closed as useful negative`.
Do not claim that the current V1-twin PA/PB/Pareto objectives predict drift
geometry. The fair current claim is that observed BackImage drift is robustly
edge-aligned, pure pose-aware response-modulation is the wrong objective, and
the tested pose-blind/Pareto twin objectives do not add explanatory power
beyond raw local edge geometry.
Do not claim that input whitening predicts biological FEM amplitude. The fair
claim is that retinal motion reformats natural-image input, but pooled
temporal-power spreading and Rucci-style spatial power-law flattening have
different scale optima and must be balanced by acuity, pose, covariance,
stability, and motor constraints.

### 2026-06-12: Content-Routed Correct-Chart Alignment

Primary locations:

- `content_routed_retinal_registration_analysis_plan.md`
- `compact_retinal_translation_geometry/run_correct_chart_swap_alignment.py`
- `compact_retinal_translation_geometry/summarize_correct_chart_swap_alignment.py`
- `outputs/compact_retinal_translation_geometry/gainbottom_match_rate_norm_structure_unitdot_v1/`
- `outputs/compact_retinal_translation_geometry/gainbottom_match_rate_norm_unitdot_v1/`
- `outputs/compact_retinal_translation_geometry/gainbottom_pool_same_image_wrong_time_unitdot_v1/`
- `outputs/compact_retinal_translation_geometry/gainbottom_pool_same_time_unitdot_v1/`

Purpose: test the stronger content-routed version of compact translation
geometry: the correct image/time-specific fitted-twin chart should explain
recorded response differences better than wrong charts and gain/subspace
controls.

Current result:

- All-unit primary in the best matched variant remained diagnostic: compact
  `k=10`, `global_rate`, true-minus-wrong mean `0.0118`, CI
  `[-0.1148, 0.1182]`, `3/5` positive sessions; controls failed.
- Gain-bottom compact subset was specifically positive: `57` units,
  true-minus-wrong mean `0.0927`, CI `[0.0253, 0.1717]`, `5/5` positive
  sessions; all required control CI lows were above zero.
- Same-image/wrong-time variants could make all-unit true-minus-wrong positive,
  but controls failed, so they are diagnostics rather than claim rows.
- Leakage audits passed in the reported runs.

Status: `Diagnostic / narrow positive`. This is the first recorded-data hint
for content-routed compact charts, but it is not yet a promoted all-unit Figure
4 bridge. Require pre-registered unit selection, replication, and pseudo-spike
positive-control checks before promoting.

### 2026-06-12: Vernier Active-Sensing Hyperacuity Branch

Primary locations:

- `vernier_active_sensing_analysis_plan.md`
- `vernier_active_sensing/README.md`
- `vernier_active_sensing/run_vernier_active_sensing.py`
- `vernier_active_sensing/summarize_vernier_active_sensing.py`
- `outputs/vernier_active_sensing_first_pass/`
- `outputs/vernier_active_sensing_component_smoke/`
- `outputs/vernier_active_sensing_component_scale/`

Purpose: replace compound E-optotype functional stress tests with a cleaner
continuous Vernier offset endpoint under explicit pose-aware and pose-blind
observer models.

First-pass result:

- Rendering audit passed the basic symmetry checks: `+/-` offsets had matched
  total luminance, model pixel pitch was about `1.60` arcmin, and pixel-level
  Fisher was about `28` per arcmin squared for both `0.25` and `0.5` arcmin
  finite-difference steps.
- Pose-aware first pass used cached model rates for `16` traces, `60` frames,
  and `756` units.
- At `0.25` arcmin, phase-cloud matched positions beat static center:
  mean Fisher `0.2733` versus `0.1753`, threshold ratio `0.822`.
- Real FEM also beat static center: mean Fisher `0.2676`, threshold ratio
  `0.820`, positive in `16/16` traces.
- Real FEM was essentially tied with phase-cloud matched positions, and
  order-shuffled positions also tied phase-cloud, so exact biological temporal
  order is not supported as special in the first pass.
- `scaled_real_0.5` was strongest in the first pass: mean Fisher `0.3293`,
  threshold proxy `1.765`; `scaled_real_1.5` was worse than phase-cloud.
- Pose-blind readout was much weaker for motion conditions and should be treated
  as an observer-assumption caveat, not an absolute threshold estimate.

Status: `Open / supportive for phase-cloud sampling`. The safe claim is that
nearby phase-cloud sampling improves pose-aware Vernier information relative to
a static center. Do not claim unique optimality of real FEM order. The
first-pass and component-scale summaries use `756` canonical units. The
scale/pose sweep's main pose-aware rows also use `756` units, while its
compact-aware controls use a `256`-unit `top_abs_fd` subset from the original
756-unit space.

### 2026-06-12: Non-Circular FEM Information / Covariance-Aware Optimality

Primary locations:

- `Non_circular_FEM_information_tests_prescription.md`
- `Covariance_aware_FEM_optimality_analysis_prescription.md`
- `active_sensing_unit_space_provenance.md`
- `jake/twininfo/covariance_optimality.py`
- `jake/twininfo/run_covariance_optimality.py`
- `active_sensing_movie_information/summarize_covariance_optimality.py`
- `active_sensing_movie_information/run_input_whitening_optimum.py`
- `active_sensing_movie_information/summarize_input_whitening_optimum.py`
- `outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu/covariance_optimality/covopt_full_gpu1/`
- `outputs/active_sensing_movie_information/covariance_optimality/covopt_full_gpu1/`
- `outputs/active_sensing_movie_information/input_whitening/`

Purpose: extend Figure 5 without making a circular "the twin proves FEM
optimality" argument. The new direction separates input-level efficient-coding
tests, recorded-cortex pose-aware information, and model-based covariance-aware
movement-scale diagnostics.

Important ideas:

- Input whitening should be tested from natural-image statistics plus drift
  kinematics, without using the fitted twin as the main optimality oracle.
- Recorded-cortex pose-aware information should ask whether the same spikes are
  more informative when retinal pose or recent eye history is known.
- Covariance-aware model tests should compare independent/pose-aware,
  covariance-aware/pose-aware, and covariance-aware/pose-blind Fisher
  efficiency across movement scale.
- The plausible geometry-aware ladder is
  `cov_pose_aware >= cov_geometry_aware >= cov_pose_blind`, but this is still a
  matched-response-space hypothesis. Current `covopt_full_gpu1` curves use the
  16 center/session-matched biological twin channels from the natural-image
  active-sensing run. The historical compact add-back/remove-out scaffold used
  the canonical 756-channel Figure 4/TFTS basis, so it is not the middle rung
  for these 16-channel covariance curves. See
  `active_sensing_unit_space_provenance.md` before interpreting any
  middle-observer result.
- Empirical FEM scale can be overlaid as an operating-regime diagnostic, but
  not claimed as globally optimal unless non-tautological and gain/noise
  sensitivity checks survive.

Implemented machinery:

- Scaled trajectory family parsing and deterministic scaling:
  `scaled_real_D...`, `random_amp_scaled_D...`,
  `random_amp_cloud_matched_scaled_D...`, and
  `trajectory_order_shuffle_scaled_D...`.
- Center-response and finite-difference rate collection from existing
  `jake.twininfo` run metadata.
- Movement-induced covariance estimates using pooled-residual and within-pair
  estimators.
- Independent and covariance-aware Fisher summaries, gain/noise sensitivity
  rows, covariance spectra, coding/signal alignment diagnostics, and summary
  decision tables.

Production result:

- The full GPU run completed `3888 / 3888` rate rows across `108` image/trace
  pairs, four scaled trajectory families, and nine movement scales.
- Primary metric: final Fisher trace per expected spike. Empirical `D=1`
  landed on an `empirical_on_plateau` or `peak_near_empirical` regime in 7/8
  family-by-trace-kind cases.
- The clearest near-empirical peak was `random_amp_scaled / fixation`:
  empirical value `77.22`, peak at `D=1.5` with value `81.38`, empirical
  fraction of peak `0.949`.
- The main negative/discriminator case was
  `trajectory_order_shuffle_scaled / microsaccade`: peak at `D=0.125`, value
  `88.73`, empirical `D=1` value `66.40`, empirical fraction of peak `0.748`.
- Corrected pose-aware minus pose-blind covariance Fisher gaps at empirical
  `D=1` were positive in all families: fixation gaps ranged `0.027-0.084`,
  microsaccade gaps ranged `0.092-0.258` (`n=54` each). For measured real
  motion specifically, fixation was `0.038 +/- 0.003` and microsaccade was
  `0.195 +/- 0.016`.
- Gain/noise sensitivity labels were stable in all tested grids: each
  family/kind label was unchanged across `9/9` gain/noise settings.
- The original input-whitening run completed `1458` retinal movies and
  `157464` metric rows, but is now interpreted narrowly: measured drift spreads
  pooled temporal power relative to stabilization, and larger motion increases
  the old temporal entropy/flatness metric. A newer Rucci-style audit asks the
  spatial power-law question directly from frame-to-frame modulation spectra;
  smoke runs show total modulation power increasing with motion but
  spatial-flattening metrics peaking at small nonzero motion.

Status: `Closed / supportive with guardrails`. The covariance-aware result
supports an efficient operating-regime claim, not a unique optimum claim:
empirical FEM scale generally lies on a high-efficiency plateau, and
conditioning on retinal pose recovers information that pose-blind covariance
accounting treats as nuisance. The code-audited post-fix result removes the old
`D=0` ridge artifact but does not change the main guardrail: random amplitude
controls match or exceed real on the pose-gap metric. The whitening branch is a
useful cleanup rather than a scale-setting answer: pooled temporal-power
spreading, Rucci-style spatial flattening, and task/feature information have
different scale optima. Do not phrase either branch as proof that biological
FEM amplitude is exactly optimized.

### 2026-06-12: Structured Translation Decoders and Forward Denoising

Primary locations:

- `structured_translation_decoder_analysis.md`
- `compact_retinal_translation_geometry/run_windowed_siamese_relative_decoding.py`
- `compact_retinal_translation_geometry/run_tejas_style_eyepos_decoder.py`
- `forward_twin_reafferent_denoising_analysis.md`
- `forward_twin_reafferent_denoising/run_forward_twin_reafferent_denoising.py`

Purpose: turn compact translation geometry into falsifiable recorded-data
predictions beyond "eye position matters."

Important ideas:

- The structured decoder target is a content-routed local translation code that
  cannot be reduced to rank-1 global gain.
- The headline decoder discriminator is the displacement component orthogonal
  to the local gain axis.
- The windowed Siamese decoder now has gain-orthogonal metrics, Poisson
  chart-weighting, and a local rank-1 gain inverse null.
- The Tejas-style eye-position decoder is intentionally permissive and should
  be used as a sanity/convention check, not as a compact-geometry claim.
- Forward-twin denoising asks whether
  `twin(real eye trace) - twin(stabilized trace)` predicts held-out recorded
  residual fluctuations beyond shuffled-eye, gain-only, random-subspace, and
  unit-shuffled controls.

Latest forward-denoising outputs:

- `outputs/forward_twin_reafferent_denoising_preview_patched_matched/`
- `outputs/forward_twin_reafferent_denoising_diag_zero_beh/`
- `outputs/forward_twin_reafferent_denoising_diag_fixed_alpha/`
- `outputs/forward_twin_reafferent_denoising_diag_image_time/`

Latest forward-denoising result:

- The corrected matched preview used 24/24 sessions, trial folds,
  `max_samples=128`, cross-fit response gains/alpha, `eye_reference=zero`,
  `stabilized_behavior=same`, and 20 matched shuffled-eye nulls including
  compact-projected shuffled-eye controls.
- Full-forward compact k=10 denoising was positive in raw held-out variance
  reduction and beat random/unit-shuffled compact controls:
  compact minus random-k `+0.000904 [0.000465, 0.001437]`; compact minus
  unit-shuffled compact `+0.000887 [0.000443, 0.001426]`.
- The same compact correction did **not** clearly beat shuffled-eye controls:
  compact minus compact-projected shuffled-eye `+0.000188
  [-0.000206, 0.000645]`, `12/24` sessions positive; compact minus full
  shuffled-eye `-0.000051 [-0.000453, 0.000412]`, `11/24` positive.
- Diagnostics did not rescue eye-trace specificity. Zero-behavior and
  image-time-fold sweeps preserved random/unit-shuffle excess but remained null
  versus compact-projected shuffled-eye. Fixed-alpha was a calibration failure
  for compact denoising.

Status: `Mixed / not promoted`. Structured decoder and forward denoising are
implemented and audited enough to interpret. The compact geometry/covariance
closure story remains intact, but these single-trial decoder/denoising bridges
should not be promoted as main positive evidence without a new endpoint.

### 2026-06-12: Manuscript Figure Assembly Refresh

Primary locations:

- `fig1/generate_fig1.py`
- `fig1/generate_fig1b.py`
- `fig1/generate_fig1c.py`
- `fig1/generate_fig1d.py`
- `fig1/generate_fig1f.py`
- `fig1/fig1a.svg`
- `fig2/generate_figure2_3_combined.py`
- `fig2/generate_fig3b.py`
- `fig2/generate_fig3f.py`
- `fig3/generate_figure3_combined.py`
- `fig4_cov_TFTS/plot_covariance_binning_sweep_panel.py`
- `scripts/diagnose_luke_fig2_inclusion.py`
- `vernier_active_sensing/`

Purpose: recompose the early manuscript figures around the tightened story.

Important changes:

- Figure 1 now uses an A-I layout: experimental schematic, gaze distribution,
  RF map, gaze/raster/PSTH population example, and single-unit gaze-sort
  example. The RF panel highlights the example unit, and the panel-A inset uses
  the `dpieg.png` image.
- Figure 2/3 combined now pools included subjects by default, can split
  subjects with `--split-subjects`, clarifies the covariance decomposition, and
  replaces the old eigenspectrum panel slot with pairwise noise correlations at
  8 ms.
- New Figure 3 compositor selects the main-text digital-twin mechanism chain:
  twin schematic, empirical-vs-model FEM modulation, eye-state zeroing,
  non-universal translation axes, compact tangent subspace, image-disjoint
  generalization, and translation-predicted recorded covariance.
- The covariance-binning sweep panel checks whether the covariance-closure
  effect is stable across spike-count windows.

Status: `Active figure polish`. These files are figure assembly and
presentation work; use the underlying analysis outputs for scientific claim
status.

### 2026-06-09: Active-Sensing Movie Information / Figure 5

Primary folder: `active_sensing_movie_information/`.

Purpose: active-sensing figure work built around Jake's `jake.twininfo`
pipeline, natural-image movie information, spatial SSI, Fisher/information
metrics, and matched-motion controls.

Important files:

- `active_sensing_movie_information/README.md`: current interpretation,
  claim guardrails, canonical pipeline, and notes about which outputs are safe
  to use.
- `active_sensing_movie_information/active_sensing_movie_information_plan.md`:
  full scientific and figure plan.
- `active_sensing_movie_information/data_and_code_inventory.md`: inventory of
  upstream assets and code to inspect.
- `active_sensing_movie_information/generate_active_sensing_movie_information_figure.py`:
  figure assembler from `outputs/twininfo/...` products, with audit/caption
  helpers.
- `active_sensing_movie_information/generate_retinal_movie_transform_qc.py`:
  retinal movie transform QC generator.
- `active_sensing_movie_information/run_active_sensing_movie_information.py`:
  temporary exploratory runner; not the canonical final pipeline.
- `active_sensing_movie_information/figure5_additional_checks_prep.md`:
  current checklist for additional Figure 5 audits and population checks.
- `active_sensing_movie_information/run_figure5_natural_image_population_checks_5_to_9.py`:
  current natural-image-only population check runner for Checks 5-9.
- `active_sensing_movie_information/run_figure5_cached_rate_checks_5_to_9.py`:
  historical e-optotype cached-rate scaffold for checks machinery.
- `active_sensing_movie_information/summarize_figure5_additional_checks.py`:
  summary/audit helper for additional Figure 5 checks.
- `Figure5_active_sensing_triage_plan.md`: top-level triage plan.
- `Figure5_random_amp_cloud_matched_control_spec.md`: spec for matched random
  amplitude/cloud controls.

Current interpretation: the safe claim is that retinal image motion exposes
structured information-bearing response variation in the deterministic twin,
especially in natural-image movies. The current notes explicitly warn against
claiming that real FEM trajectories are uniquely optimal based only on the
spatial-SSI endpoint.

### 2026-06-09 / 2026-06-08: Compact Retinal-Translation Geometry

Primary folder: `compact_retinal_translation_geometry/`.

Purpose: upgrade path and eventual replacement for `fig4_cov_TFTS`, using
existing production outputs to build spec-facing compact geometry tables,
audits, metric validation, and recorded displacement decoding.

Important files:

- `compact_retinal_translation_geometry/README.md`: runbook and relationship
  to upstream production artifacts.
- `compact_retinal_translation_geometry/run_compact_retinal_translation_geometry.py`:
  panel-table builder from existing TFTS and covariance-closure outputs.
- `compact_retinal_translation_geometry/run_compact_geometry_audits.py`:
  acceptance/audit suite for compact geometry outputs.
- `compact_retinal_translation_geometry/run_metric_structure_validation.py`:
  hidden-coordinate/local-metric validation from translated response grids.
- `compact_retinal_translation_geometry/run_relative_displacement_decoding.py`:
  recorded relative displacement decoding, currently active/uncommitted.
- `compact_retinal_translation_geometry_implementation_spec.md`: governing
  implementation spec and acceptance criteria.

The folder currently reuses:

- `outputs/twin_feature_tangent_structure_prod_v2`
- `outputs/matched_twin_covariance_closure_rf_null_step025_rfbacked_v2`

Current claim boundary: the promoted result is structural compact
retinal-translation geometry and recorded covariance closure. The compact source
retains full-source closure within tolerance; do not describe the small
compact-over-full ordering in summary tables as a compact superiority result.
The strict recorded relative-displacement decoder and gain-orthogonal structured
decoder are controlled nulls under current audits, not positive Panel F bridges.

### 2026-06-08: Direct Recorded Derivative / Twin Tangent Alignment

Primary folder: `direct_recorded_derivative_twin_alignment/`.

Purpose: bounded supplemental analysis testing whether recorded eye-position
derivatives are enriched in compact fitted-twin translation geometry.

Important files:

- `direct_recorded_derivative_twin_alignment/README.md`: current claim gate,
  tiers, stop rules, and expected outputs.
- `direct_recorded_derivative_twin_alignment/run_direct_recorded_derivative_alignment.py`:
  main runner for recorded derivative reliability, compact-basis capture,
  matched derivative alignment, signed-axis diagnostics, and null summaries.
- `direct_recorded_derivative_twin_alignment/STG_REFERENCE_NOTES.md`:
  notes imported from older shared-transformation-geometry failure modes.
- `direct_recorded_derivative_twin_alignment_prescription.md`: original
  analysis prescription.

Promote only if Tier 1 survives global-rate plus target-PC1 projection and
RF/readout-preserving nulls across sessions.

### 2026-06-07 / 2026-06-08: Matched Twin Covariance Closure

Primary folder: `matched_twin_covariance_closure/`.

Purpose: compare recorded FEM covariance in Ryan's matched recorded/twin unit
space to fitted-twin eye-position and finite-difference tangent covariance.

Important files:

- `matched_twin_covariance_closure/README.md`: current provenance, basis
  definitions, controls, finite-difference results, and audit summaries.
- `matched_twin_covariance_closure/run_cache_closure.py`: first-pass cache-only
  closure using matched Ryan Fig2/Fig3 products.
- `matched_twin_covariance_closure/run_finite_difference_closure.py`: stricter
  model-based finite-difference closure.
- `matched_twin_covariance_closure/summarize_finite_difference_results.py`:
  audit, bootstrap, and sign-test summaries.
- `matched_twin_covariance_closure/rf_readout_preserving_null_prescription.md`:
  prescription for stronger RF/readout-aware nulls.

### 2026-06-07: Figure 4 Covariance / TFTS Figure Work

Primary folder: `fig4_cov_TFTS/`.

Purpose: figure-generation and panel-level analysis workspace for covariance
and twin-feature-tangent-structure results. This is now partly superseded by
`compact_retinal_translation_geometry/`, but still contains figure scripts and
handoffs.

Important files:

- `fig4_cov_TFTS/generate_covTFTS_figure.py`: current figure generator.
- `fig4_cov_TFTS/generate_covTFTS_figure_4.py`: later figure variant.
- `fig4_cov_TFTS/generate_covTFTS_figure_draft.py`: draft figure generator.
- `fig4_cov_TFTS/generate_spatial_content_modulation_figure.py`: figure for
  spatial-content modulation / natural structure controls.
- `fig4_cov_TFTS/run_panelF_covariance_overlap.py`: Panel F covariance overlap
  runner.
- `fig4_cov_TFTS/run_panelF_natural_structure.py`: Panel F natural-structure
  runner.
- `fig4_cov_TFTS/tests/test_tangent_subspace_information.py`: focused tests for
  tangent-subspace information machinery.
- `fig4_cov_TFTS/covTFTS_figure_panel_prescription.md`: panel prescription.
- `fig4_cov_TFTS/covTFTS_figure_data_forward_prescription.md`: data-forward
  prescription.
- `fig4_cov_TFTS/figure4_panelF_natural_structure_coda_plan.md`: Panel F coda
  plan.
- `fig4_cov_TFTS/panelE_tangent_subspace_information_handoff.md`: detailed
  handoff for tangent-subspace information.
- `fig4_cov_TFTS/update.md`: local update notes.
- `figure4_multipanel_plus_sup.md`: top-level Figure 4 multipanel and
  supplement handoff.
- `Fig4_reruns_plan.md`: top-level rerun plan.

### 2026-06-04: Natural Image Tangent Scale

Primary folder: `natural_image_tangent_scale/`.

Purpose: quantify where local translation/tangent approximations break down in
natural-image movies, including displacement scale, image structure, and FEM
amplitude comparisons.

Important files:

- `natural_image_tangent_scale/run_natural_image_tangent_scale.py`: main
  analysis runner.
- `Natural_Image_Tangent_Scale_Analysis_Handoff.md`: handoff.
- `natural-image-tangent-scale-analysis.md`: analysis note.

### 2026-06-04 / 2026-06-03: Twin Feature Tangent Structure

Primary folder: `twin_feature_tangent_structure/`.

Purpose: compact basis and split/generalization analyses for fitted-twin
translation tangents across natural images and conditions.

Important files:

- `twin_feature_tangent_structure/run_twin_feature_tangent_structure.py`: main
  runner for cached tangent union spectra, split modes, basis summaries, and
  audits.
- `twin_feature_tangent_structure_handoff.md`: handoff.
- `Twin_Feature_Tangent_Structure_Prescription.md`: prescription.

### 2026-06-03: Shared Transformation Geometry

Primary folder: `shared_transformation_geometry/`.

Purpose: early STG pipeline for recorded/twin tangent maps, support census,
residual RDM geometry, image-similarity controls, template matching, shared-mode
projection sweeps, and cross-session aggregation.

Important files:

- `shared_transformation_geometry/README.md`: main runbook.
- `shared_transformation_geometry/run_stg_support_census.py`: session support
  census.
- `shared_transformation_geometry/run_stg_tangent_stage1.py`: signed tangent-map
  analysis for recorded or twin sources.
- `shared_transformation_geometry/run_stg_shared_mode_projection_sweep.py`:
  recorded shared-mode projection sweep.
- `shared_transformation_geometry/run_stg_tangent_template_confirmation.py`:
  recorded-vs-twin template confirmation.
- `shared_transformation_geometry/run_stg_residual_rdm_stage2.py`: diagnostic
  residual RDM runner.
- `shared_transformation_geometry/run_stg_residual_rdm_stage23.py`: canonical
  residual geometry plus image-similarity controls.
- `shared_transformation_geometry/run_stg_direct_recorded_twin_tangent_match.py`:
  direct recorded/twin tangent matching.
- `shared_transformation_geometry/run_stg_retinotopy_tangent_identity.py`:
  retinotopy/tangent identity analysis and historical reference for later
  alignment work.
- `shared_transformation_geometry/run_stg_aggregate_stage5.py`: cross-session
  aggregation.
- `shared_transformation_geometry/utils.py`: shared helpers.
- `shared_transformation_geometry_handoff.md` and
  `shared_transformation_geometry_handoff_v2.md`: top-level handoffs.

### 2026-06-03: Twin Covariance Structure

Primary folder: `twin_covariance_structure/`.

Purpose: digital-twin reafferent covariance geometry: covariance estimation,
eigenspectra, participation ratio, subspace overlap, translation-tangent
alignment, image specificity, occupancy controls, and single-unit-to-population
bridges.

Important files:

- `twin_covariance_structure/README.md`: main runbook and output inventory.
- `twin_covariance_structure/run_twin_covariance_structure.py`: A1-A6 runner.
- `twin_covariance_structure/run_a2_audit.py`: A2 control-construction and
  trace-count audit.
- `twin_covariance_structure/run_a3_audit.py`: A3 image-specificity audit.
- `twin_covariance_structure/run_a3_fixrsvp_audit.py`: fixRSVP A3 audit using
  empirical image windows.
- `twin_covariance_structure/build_a3_high_support_summary.py`: summary helper
  for high-support A3 results.
- `twin_covariance_structure/generate_control_response_caches.py`: response
  cache generation for transformed eye controls.
- `twin_covariance_structure/covariance_core.py`: covariance primitives.
- `twin_covariance_structure/subspace_metrics.py`: overlap/capture/angle
  utilities.
- `twin_covariance_structure/eye_controls.py`: occupancy and eye-control
  transforms.
- `twin_covariance_structure/plotting.py`: figure builders.
- `Twin_Covariance_Structure_Prescription.md`: prescription.
- `twin_covariance_analysis_plan.md`: top-level plan.

### 2026-06-01: Keystone / Big-Picture FEM Geometry

Purpose: bridge and planning docs around the "keystone" geometry idea,
active-sensing efficiency, and fixRSVP neural trajectory readiness.

Important files:

- `Keystone_Geometry_Crossover_handoff_v2.md` and
  `Keystone_Geometry_Crossover_handoff_v3.md`: crossover handoffs.
- `bigpicture_fem_v1_high_impact_analysis_plan_v2.md`: high-impact analysis
  plan.
- `bigpicture_phase1_fem_v1_coding_agent_plan_v2.md`: coding-agent plan.
- `e1_active_sensing_efficiency_revised_handoff.md`: active-sensing efficiency
  handoff.
- `fixrsvp_neural_trajectory_analysis_plan_revised.md`: revised neural
  trajectory plan.
- `fixrsvp_trajectory_implementation_readiness.md`: readiness note.
- `mono_covariance_decomposition.py`: covariance-decomposition exploration.
- `mono_empirical_keystone_fullgrid.py`: empirical keystone full-grid
  exploration.

### 2026-05-29: Jacobian Audit / Predictive Framework

Purpose: audit and handoff layer around local Jacobian identity/translation
geometry and path-integrated separability.

Important files:

- `Empirical Transformation Geometry Analysis.md`: broad empirical geometry
  analysis note.
- `eoptotype_jacobian_field_smoothness_handoff.md`: e-optotype Jacobian field
  smoothness handoff.
- `fem_path_integrated_separability_handoff.md`: path-integrated separability
  handoff.
- `figure4_geometry_bridge_audit_plan_v2.md`: Figure 4 geometry bridge audit.
- `fixrsvp_cross_session_check.md`: cross-session check.
- `jacobian_predictive_framework_progress_summary.md`: progress summary.

### 2026-05-26: Jacobian Identity Geometry / Figures

Purpose: main local Jacobian analysis, translation mimicry, phase landscape,
matched-position ablation, and figure generation.

Important files:

- `jacobian_test3.py`: main local Jacobian/translation covariance analysis.
- `jacobian_test4.py`: follow-on Jacobian test script.
- `geometry_utils.py`: geometry helpers used by Jacobian figure work.
- `main_figure_jacobian.py`: main figure generator.
- `figure_jacobian_identity_geometry.py`: identity geometry figure generator.
- `translation_mimicry.py`: tests whether retinal translation can mimic
  identity changes.
- `phase_landscape.py`: static-offset phase landscape analysis.
- `matched_position_ablation.py`: matched-position ablation runner.
- `gru_passthrough_test.py`: GRU passthrough and temporal-integration
  diagnostic.
- `fem_global_intervention.py`: global FEM intervention analysis.
- `make_priority2_causal_alignment_figure.py`: render priority-2 causal
  alignment figure from intervention outputs.
- `FEM_population_coding_writeup.md`: population-coding writeup.
- `fem_eoptotype_hyperacuity_results.md`: e-optotype hyperacuity results.
- `fem_next_steps_plan.md`: next-steps plan.
- `jacobian_figure_handoff_nature_style.md`: figure style handoff.
- `jacobian_identity_geometry_results.md`: results note.
- `jacobian_identity_transformation_analysis_plan.md`: analysis plan.
- `jacobian_predictive_framework_handoff_revised.md`: revised handoff.

Result folders:

- `jacobian_results/`: `.npz`, text, and PDF outputs for Jacobian tests.
- `matched_position_results/`: matched-position ablation figure/data.
- `results/phase_landscape_*`: phase landscape smoke/coarse/fine outputs.
- `results/translation_mimicry_*`: translation mimicry smoke/primary outputs.
- `fem_global_intervention_results/`: global intervention figures/data.
- `gru_passthrough_figures/`: temporal-integration and feature-RSA figures.

### 2026-05-21: Early FEM / Temporal Decoding / COM Dynamics

Purpose: first large batch of FEM/e-optotype analysis scripts and planning
docs, including temporal decoding, center-of-mass dynamics, displacement
decoding, and covariance/intervention experiments.

Important files:

- `analysis_plan_jacobian_v3.md`: early Jacobian analysis plan.
- `revised_analysis_plan.md`: revised plan.
- `results_summary.md`: summary of early results.
- `FEMs_Eoptotype_checks.md`: e-optotype checks.
- `temporal_decoding_analysis_plan_consolidated_v2.md`: consolidated temporal
  decoding plan.
- `temporal_decoding_analysis_implementation_plan.md`: implementation plan.
- `temporal_decoding_diagnostic_plan.md`: diagnostic plan.
- `temporal_analysis_issues_and_alternatives.md`: issues and alternatives.
- `rowley_luke_2026_03_16_dpi_pupil_intercept_findings.md`: Rowley/Luke pupil
  intercept findings.
- `rowley_session_config_generation.md`: session config generation notes.
- `com_dynamics.py`: center-of-mass/spatial-moment dynamics and decoding.
- `com_dynamics_spec.md`: COM dynamics spec.
- `transformation_dynamics.py`: transformation-dynamics analysis.
- `transformation_dynamics_plan.md`: transformation-dynamics plan.
- `displacement_decoding.py`: displacement decoding analysis.
- `displacement_decoding_spec.md`: displacement decoding spec.
- `eoptotype_continuous_pass.py`: continuous-pass e-optotype analysis.
- `fem_covariance_geometry.py`: FEM covariance geometry analysis.
- `fem_differential_intervention.py`: differential intervention analysis.
- `fem_global_intervention.py`: global intervention analysis.
- `translation_covariance.py`: translation-covariance analysis.
- `translation_covariance_figures.py`: translation-covariance figure builder.
- `translation_covariance_trajectories_figures.py`: trajectory figures.
- `translation_covariance_trajectories_figures0.py`: earlier trajectory figure
  variant.
- `translation_covariance_trajectories_figures_interactive.py.py`: interactive
  trajectory figure variant; note the duplicate `.py.py` suffix.
- `overnight_backimage_long_sweeps_20s_re/note`: note for rerun sweep folder.

Result/cache folders:

- `continuous_pass_results/`: continuous-pass `.npz` and `.png` outputs.
- `displacement_decoding_cache/`: cached displacement grids/rates.
- `displacement_decoding_figures/`: displacement decoding PDFs and summary.
- `fem_covariance_geometry_results/`: real/stabilized FEM covariance geometry
  outputs.
- `fem_differential_intervention_results/`: differential intervention output.
- `transformation_dynamics_cache/`: cached moment and delta-rate tensors.
- `transformation_dynamics_figures/`: transformation dynamics figures.

### 2026-01 to 2026-04: Early Backimage / Translation-Covariance Artifacts

Purpose: early eye-trace, backimage, translation-covariance, and natural-image
sweep artifacts. Many of these are bulky generated assets rather than active
source files.

Important files/folders:

- `translation_covariance/`: January translation-covariance `.npz`, `.pkl`,
  capture matrices, shuffle products, and older figures/trajectory figures.
- `overnight_backimage_sweeps/`: first overnight backimage sweep `.pkl` files.
- `overnight_backimage_long_sweeps_20s/`: 20s backimage sweep artifacts and
  `RUN_SUMMARY.json`.
- `overnight_backimage_long_sweeps_20s_re/`: rerun subset and note.
- `test_sweeps/`: generated sweep videos and saccade histograms.
- `E_diagnostics_human_240ppd/`: human-resolution E diagnostic outputs by size.
- `E_diagnostics_model_37ppd/`: model diagnostic run marker/output.
- `E_diagnostics_model_37ppd_resnet_none_convgru/`: model diagnostic outputs by
  size; large generated artifact folder.
- `backimage_fixation_results.pkl` and `.meta.json`: early backimage fixation
  cache.
- `backimage_image_cache.pkl`: image cache.
- `fixrsvp_fixation_pool.pkl` and `.meta.json`: fixRSVP fixation pool.
- `hybrid_eye_trace_*.pkl`: early hybrid eye trace sweep caches.
- `spatial_info_fixrsvp_eye_scales_frames_per_im.pkl`: large spatial-info
  cache.
- `SpikeSortingTools_repo_collation.txt`: text collation of SpikeSortingTools
  repo material.

## Top-Level Folder Index

Use this section when you know the folder but not the date.

| Path | Contents |
| --- | --- |
| `active_sensing_movie_information/` | Figure 5 / active-sensing movie-information plans, figure generator, QC, and population checks. |
| `compact_retinal_translation_geometry/` | Spec-facing compact translation-geometry panel builder, audits, metric validation, displacement decoding, structured/gain-orthogonal decoder work, and permissive Tejas-style decoder checks. |
| `vernier_active_sensing/` | Controlled Vernier hyperacuity analysis package, renderer, model runner, and summary/figure helper. |
| `fixation_statistics_by_stimulus/` | Fixation-regime statistics, BackImage local image-structure analyses, registration QC, random-location controls, and twin drift-geometry runner. |
| `forward_twin_reafferent_denoising/` | Forward twin reafferent denoising runner for held-out recorded residual correction tests. |
| `direct_recorded_derivative_twin_alignment/` | Supplemental recorded derivative vs twin tangent alignment runner and notes. |
| `matched_twin_covariance_closure/` | Cache and finite-difference closure of recorded FEM covariance by fitted-twin tangents. |
| `fig4_cov_TFTS/` | Figure 4 covariance/TFTS figure scripts, panel analyses, handoffs, and tests. |
| `fig1/`, `fig2/`, `fig3/` | Main-text figure assembly scripts and panel generators. |
| `natural_image_tangent_scale/` | Natural-image tangent scale/breakdown analysis. |
| `twin_feature_tangent_structure/` | Fitted-twin tangent basis compactness and split/generalization analyses. |
| `shared_transformation_geometry/` | Early recorded/twin shared transformation geometry pipeline. |
| `twin_covariance_structure/` | Reafferent covariance geometry analysis and audit scripts. |
| `../jake/twininfo/covariance_optimality.py` | Cross-package covariance-aware FEM optimality helpers used by the active-sensing Figure 5 extension. |
| `../jake/twininfo/run_covariance_optimality.py` | Cross-package covariance-aware FEM optimality runner for existing production `jake.twininfo` outputs. |
| `results/` | Generated phase-landscape, translation-mimicry, and Jacobian identity outputs. |
| `jacobian_results/` | Generated Jacobian analysis outputs and interpretation note. |
| `translation_covariance/` | Early translation-covariance generated products and figures. |
| `continuous_pass_results/` | Continuous-pass e-optotype generated outputs. |
| `*_results/`, `*_figures/`, `*_cache/` | Generated artifacts for the corresponding top-level script. |
| `overnight_backimage_*`, `test_sweeps/`, `E_diagnostics_*` | Early generated sweep/diagnostic artifacts. |
| `__pycache__/` and nested `__pycache__/` | Python bytecode caches; not source. |

## Thread Map

These threads overlap scientifically but are useful for navigation.

| Thread | Current home | Historical/related files |
| --- | --- | --- |
| Active-sensing natural-image information | `active_sensing_movie_information/` | `Figure5_active_sensing_triage_plan.md`, `Figure5_random_amp_cloud_matched_control_spec.md`, `e1_active_sensing_efficiency_revised_handoff.md`, `Non_circular_FEM_information_tests_prescription.md` |
| Active-sensing roadmap / BackImage image structure | `active_sensing_roadmap_after_vernier_fixation_image_structure.md`, `fixation_statistics_by_stimulus/` | `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/`, `vernier_active_sensing_analysis_plan.md` |
| Vernier active sensing | `vernier_active_sensing/` | `vernier_active_sensing_analysis_plan.md`, `outputs/vernier_active_sensing_first_pass/` |
| Covariance-aware FEM operating regime | `jake/twininfo/covariance_optimality.py`, `jake/twininfo/run_covariance_optimality.py` | `Covariance_aware_FEM_optimality_analysis_prescription.md`, `active_sensing_movie_information/summarize_covariance_optimality.py` |
| Structured translation decoding | `compact_retinal_translation_geometry/run_windowed_siamese_relative_decoding.py` | `structured_translation_decoder_analysis.md`, `compact_retinal_translation_geometry/run_tejas_style_eyepos_decoder.py` |
| Content-routed chart alignment | `compact_retinal_translation_geometry/run_correct_chart_swap_alignment.py` | `content_routed_retinal_registration_analysis_plan.md`, `compact_retinal_translation_geometry/summarize_correct_chart_swap_alignment.py` |
| Forward twin reafferent denoising | `forward_twin_reafferent_denoising/` | `forward_twin_reafferent_denoising_analysis.md`, `matched_twin_covariance_closure/`, `compact_retinal_translation_geometry/` |
| Compact retinal translation geometry | `compact_retinal_translation_geometry/` | `fig4_cov_TFTS/`, `twin_feature_tangent_structure/`, `matched_twin_covariance_closure/` |
| Manuscript figure assembly | `fig1/`, `fig2/`, `fig3/` | `fig4_cov_TFTS/plot_covariance_binning_sweep_panel.py`, `scripts/diagnose_luke_fig2_inclusion.py` |
| Recorded derivative/twin tangent alignment | `direct_recorded_derivative_twin_alignment/` | `shared_transformation_geometry/`, `matched_twin_covariance_closure/` |
| Covariance closure | `matched_twin_covariance_closure/` | Ryan cache copies under `outputs/cache/` and RF-backed closure outputs under `outputs/` |
| Shared transformation geometry | `shared_transformation_geometry/` | `shared_transformation_geometry_handoff*.md` |
| Twin covariance structure | `twin_covariance_structure/` | `Twin_Covariance_Structure_Prescription.md`, `twin_covariance_analysis_plan.md` |
| Twin feature tangent structure | `twin_feature_tangent_structure/` | `Twin_Feature_Tangent_Structure_Prescription.md`, `twin_feature_tangent_structure_handoff.md` |
| Jacobian identity / translation mimicry | top-level Jacobian scripts and `results/` | `jacobian_*`, `translation_mimicry.py`, `phase_landscape.py`, `geometry_utils.py` |
| Early temporal/COM/displacement work | top-level scripts and cache folders | `temporal_*`, `com_dynamics*`, `displacement_decoding*`, `transformation_dynamics*` |
| Early backimage sweeps | `overnight_backimage_*`, `test_sweeps/` | `backimage_*`, `hybrid_eye_trace_*`, `spatial_info_*` caches |

## Cleanup Notes

Obvious cleanup candidates, if/when this workspace is later reorganized:

- Generated artifacts under `declan/` could move to `outputs/declan/` or a
  scratch/artifact tree.
- `__pycache__/` folders can be ignored or removed.
- `translation_covariance_trajectories_figures_interactive.py.py` has a likely
  accidental duplicate suffix, but preserve it until references are checked.
- Top-level scripts from 2026-05-21 and 2026-05-26 are intertwined with result
  folders; move only after adding import/CLI regression checks.
- Do not delete historical e-optotype Figure 5 scaffolds until the natural-image
  pipeline is fully settled; mark them historical in docs instead.
