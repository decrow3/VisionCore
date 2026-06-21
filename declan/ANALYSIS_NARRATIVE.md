# declan Analysis Narrative

Last curated: 2026-06-20.

Companion to `MANIFEST.md`. The manifest answers "where is it?" This file
answers "why did we do it, what happened, and how did later work change the
interpretation?"

This is a living synthesis from the markdown plans, handoffs, READMEs, and
result notes in `declan/`, plus the run summaries under `outputs/` when they
closed a thread. It deliberately keeps both the at-the-time interpretation and
the later revision when a control or follow-up narrowed the claim.

## Reading Rules

- `Closed` means there is enough result evidence to treat the thread as
  resolved for current purposes.
- `Promoted` means the thread has a result that can plausibly carry a figure or
  manuscript claim, with the stated guardrails.
- `Supportive` means useful evidence, but not a standalone headline.
- `Historical` means useful context or machinery, but superseded by a later
  framing or control.
- `Open` means a plan/spec exists, and sometimes code exists, but the analysis
  is not yet fully interpreted.

## Current Synthesis

The story has moved through four broad phases:

1. Early FEM population-coding work found a real E-optotype crossover and a
   tempting temporal/covariance story, but later controls narrowed the mechanism
   to first-order spatial sampling in the mean-rate code.
2. The Jacobian/translation-covariance work rescued the structural part of the
   idea: image translation directions robustly organize FEM-linked covariance,
   even when magnitude identities and temporal-code interpretations fail.
3. The Figure 4/Figure 5 split is cleaner: Figure 4 is about compact
   reafferent retinal-translation geometry and recorded covariance closure;
   Figure 5 is about active-sensing movie information efficiency, with explicit
   limits on claims of real-trajectory optimality.
4. The newest direction is more disciplined and more falsifiable. Figure 5
   extensions now avoid twin-circular optimality by separating input-level
   whitening, covariance-aware pose-aware/pose-blind penalties, and recorded
   cortex anchors. The first production covariance-aware operating-regime run
   landed as supportive but not a unique-optimum result: empirical FEM scale is
   usually on a high-efficiency plateau, while corrected pose-aware covariance
   accounting consistently recovers information discarded by pose-blind
   accounting. A 2026-06-13 code audit fixed a small ridge-path mismatch in the
   pose-aware/pose-blind comparison; the corrected gaps remain positive but are
   slightly smaller than the first summary. The first cache-based recorded
   pose-aware GLM ladder landed as a controlled null
   rather than a positive bridge. The first corrected Figure 4 structured
   decoder also landed as a controlled null: the compact chart did not recover
   gain-orthogonal displacement beyond rank-1 gain or chart-shuffle controls.
   A newer content-routed correct-chart analysis validated the chart-swap
   machinery with strong pseudo controls and found a targeted gain-bottom hint,
   but the recorded effect is split- and session-sensitive and is now best
   treated as a diagnostic branch rather than a bridge-rescue path. Forward-twin
   residual correction remains a useful but unpromoted Figure 4 extension. A new
   Vernier branch now provides a cleaner hyperacuity-style Figure 5 test:
   phase-cloud motion beats a static center, real and order-shuffled motion
   roughly match the phase-cloud control, and reduced real motion was strongest
   in the first pass. The newest active-sensing synthesis is now centered on
   coordinate-frame dependence, task-specific motion scale, regime-dependent FEM
   statistics, local image geometry, and a Pareto-style tradeoff rather than
   exact trace optimality. The original pooled temporal-PSD input-whitening run
   is now superseded as a Rucci-style whitening test: it showed that larger
   motion spreads temporal power, not that spatial power-law whitening favors
   larger-than-biological motion. The newer Rucci-style audit asks the spatial
   modulation question directly and shows, in smoke runs, that total modulation
   power grows with motion while spatial power-law flattening peaks at small
   nonzero motion. Whitening therefore remains an important input-statistics
   constraint, but not a standalone scale-setting answer. The latest BackImage
   local Gabor/pyramid screen now supports a regime-dependent small-scale
   real-vs-random `I_z` signal, strongest near `0.25x` observed RMS, but not a
   clean global or `1x` infomax claim. The follow-up aggregate natural-image FEM
   information run is now the stronger BackImage result: in a cleaned `n=256`,
   `K=4`, grouped-by-image CV run, empirical drift-like motion adds
   feature-decoding signal beyond static V1-twin responses, robustly beats
   OU-like confined controls, and does not simply improve with more motion. The
   advantage over Brownian/generic motion is strongest at `0.25x-0.5x` and
   narrows at `1x-2x`, so the claim remains scale-, readout-, and twin-scoped
   rather than a proof of exact trajectory optimality. The reopened local
   BackImage pairing branch now adds a narrower positive: after fixing the
   trace-bank, feature-geometry, and sampled matched-control logic, actual
   image-trace pairings beat matched unpaired empirical trace swaps for
   `delta_mean` feature-response gains in both Gabor and pyramid local fields.
   This supports local image-contingent motion-delta structure beyond aggregate
   empirical FEM statistics, but it is not yet a broad temporal-code or
   unique-axis-optimality result because temporal PCA/DCT summaries are weak and
   rotated-trace controls remain competitive. A separate axis-conditioned
   BackImage observer branch now directly tests edge-parallel versus
   edge-orthogonal priors with shared source catalogs. Both clean `n=64` runs
   show joint-eye rescue above zero-eye, but the preferred axis depends on the
   candidate set: matched-static weakly favors edge-parallel, while
   hard-negative weakly favors edge-orthogonal. A feature-posterior
   joint-decoding posthoc now sharpens the functional endpoint: joint feature
   recovery beats zero-eye robustly, but the axis result is bounded by the new
   bootstrap/permutation uncertainty. Matched-static keeps a parallel-positive
   feature-recovery signal, strongest for `pyramid k8`, whereas hard-negative
   feature recovery trends orthogonal with no significant axis contrasts. The
   safest current interpretation is therefore trajectory-aware feature recovery,
   not yet a clean along-contour mechanism split. The old stronger `target128`
   orthogonal advantage is now diagnostic-only because it predates the
   unmatched-catalog fix, and the active `n=128`, `0.5x/1x/2x` shared-source run
   is the next replication gate.

Consolidation update, 2026-06-20:

- The active-sensing/geometry work now has two guarded production surfaces:
  `declan/canonical_active_sensing/` for aggregate, local-pairing,
  joint-posterior, adjudication, and active-sensing figure-pack runs; and
  `declan/canonical_geometry/` for raw-edge residual adjudication plus the
  geometry figure pack. These wrappers should be the entry points for long
  production jobs because they validate configs, print commands, and refuse
  accidental writes into existing non-empty output folders.
- The current feature target is a two-readout candidate, not a final lock:
  `pyramid_local_field k16 temporal_pca` is the aggregate/ensemble readout
  candidate, while `pyramid_local_field k16 delta_mean` is the local
  mechanistic-sensitivity readout. The target remains provisional until the
  joint `rel_0p25x` feature-posterior completion and final adjudication review
  are closed.
- The Figure 4 active-sensing atlas now has a consolidated
  `claim_critical_diagnostics_queue.md`. This is the gatekeeping document for
  anticipated failure modes before main claims or long canonical runs are
  trusted. It centralizes the diagnostics that had been scattered across the
  raw-edge handoff, priority checklist, atlas flags, and canonical provenance:
  same-window model-objective versus raw-edge tables, within-session residual
  tests, global-axis nuisance audits, shared-source/candidate-hardness audits,
  population/readout sensitivity, and preservation-versus-modulation
  decompositions.
- Panel E provenance is now explicit. E3 remains the compact endpoint-zone
  enrichment redraw from `endpoint_zone_enrichment_summary.csv`; E6/E7/E8 copy
  the original behavior inspection panels for the full distribution/session
  diagnostic, confidence/signed-delta diagnostic, and endpoint/null diagnostic.
  E8 should travel with E3 whenever the contour-following behavior metric is
  explained, because it shows why the `cos(2 delta)` endpoint bins must be read
  against the transformed uniform-angle null.
- The model-objective branch is best treated as the most likely methods-style
  deep-dive trigger. The current negative is scientifically useful: raw edge
  geometry remains the behavior baseline to beat, and apparent objective wins
  must first survive residual explanation beyond raw edge confidence, global
  screen-axis nuisance checks, candidate/source-overlap checks, and canonical
  population/readout sensitivity. If those gates fail, the honest main-paper
  split is behavior follows local image geometry while the V1 twin explains
  possible utility, preservation, and trajectory-aware inference consequences;
  it is not proof that the animal optimizes the tested objective.

Numerical audit update, 2026-06-12:

- The compact covariance-closure source can slightly exceed the full
  finite-difference source in reported mean capture because the current
  comparison uses separately constructed full and cross-fit compact sources, not
  a strict nested projection of the same source under a shared estimator. Treat
  ratios such as `1.005` as "no detectable closure cost", not as evidence that
  compact beats full.
- The large Jacobian magnitude mismatch, the stimulus-specific 100% intervention
  result, the E-optotype `lm=-0.20` SSI-versus-orientation split, the pooled
  FEM-subspace ablation, and the near-perfect within-image displacement decoder
  are historical guardrails. They are not current active headline claims.
- The current recorded pose-aware GLM ladder and gain-orthogonal structured
  decoder are controlled nulls, not rescue routes for those older functional
  interpretations.
- The patched matched-context relative-displacement decoder is also a
  constrained diagnostic, not a promoted bridge: same-image/time pairing now
  works as intended, eye-label-shuffle support is explicitly required, and the
  refreshed six-session run shows signal that weakens sharply under skeptic
  projection controls rather than a compact-specific readout.
- The current correct-chart swap alignment is a diagnostic branch, not a
  promoted bridge: pseudo controls pass clearly, the all-unit recorded effect is
  not robust, and the gain-bottom positive does not yet survive the fold/session
  sensitivity work as a preregistered targeted result.

## 2026-06-19: Shape of the Translation Geometry / Flow-Manifold Interpretation

Status: `Open / supportive geometry intuition, not yet a formal manifold theorem`.

Primary code and outputs:

- `backimage_trajectory_observer/analyze_global_fixation_geometry.py`
- `backimage_trajectory_observer/plot_global_fixation_trajectory_lines_3d.py`
- `backimage_trajectory_observer/analyze_global_fixation_trajectory_flow.py`
- `backimage_trajectory_observer/plot_local_fixation_fan_geometry.py`
- `backimage_trajectory_observer/run_phase_aware_response_geometry_pilot.py`
- `twin_feature_tangent_structure/run_shape_atlas_probe.py`
- `twin_feature_tangent_structure/run_phase_rotation_probe.py`
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/global_fixation_geometry_hardneg_empirical_scale0p5/`
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_trajectory_observer_percandidate_gpu1_target128_c4_k32/local_fixation_fan_dense16_patchpx_axisfamilies_v1/`
- `outputs/twin_feature_tangent_structure_prod_v2/shape_atlas_probe/`
- `outputs/twin_feature_tangent_structure_prod_v2/shape_atlas_probe_finite_smoke/`
- `outputs/twin_feature_tangent_structure_prod_v2/phase_rotation_probe_centered_check/`

Motivation:

The phrase "compact tangent geometry" is correct but opaque. This follow-up
asked what shape the geometry actually has when rendered as response activity:
a torus, loops, ribbons, sheets, a fan, or a vector field attached to static
image content. It also asked whether Fourier/spatial-frequency intuition helps:
translation should rotate local phase, with phase advance scaling with spatial
frequency, so the compact tangent subspace might reveal phase-like coordinates.

What changed conceptually:

The most useful frame is no longer "a 2D manifold because translation is 2D."
Each fixation-local image patch should be treated as its own object. Static
responses identify local content, and translations attach a local vector field
or trajectory family to those content states. The recognizable object in the
BackImage cache is therefore closer to:

```text
a static content manifold with a shared, structured translation-flow field
```

than to a clean global x/y coordinate sheet or torus.

Global BackImage trajectory geometry:

- In the original hard-negative empirical cache, the `motion_delta` response
  cloud was globally low-dimensional and fan-like: PC1 explained about `71.5%`,
  PC1-2 about `83.3%`, and the top 10 PCs about `92.4%` of sampled
  state-timepoint variance.
- Linking complete trajectories was more informative than plotting the
  downsampled point cloud. The full selected set contained `4096` complete
  candidate/trajectory paths and `163840` response states. Sparse linked plots
  made the geometry look like directed curves moving through a curved sheet or
  fan, not independent dust.
- A flow metric on all directed segments supported the visual impression. With
  `159744` directed segments, local direction coherence was `0.309` versus a
  shuffled-direction null of `0.138` (`2.23x`), and angular-momentum coherence
  was `0.138` versus `0.0099` (`13.9x`). PC3 stretching made the figure easier
  to see but did not create the effect; the unscaled PC3 metric stayed strong.
- The best verbal description is "curved flowing sheet" or "fan with shared
  local flow and rotational/curl component", not "single rigid vortex" and not
  "closed torus".

Local smoothness check:

- Zooming into dense BackImage fixation neighborhoods preserved local flow
  coherence. In the larger axis-conditioned cache, the densest 16 source
  windows covered a radius of about `28 px`, with `5504` trajectory groups and
  `220160` response points.
- In that local patch, direction coherence in the global fan PCA was `0.328`
  versus `0.159` under shuffled segment directions (`2.06x`); local-only PCA
  gave a similar `2.01x` lift.
- But physical closeness of fixation centers only weakly predicted neural
  trajectory-centroid closeness: source-position versus neural-centroid
  distance Spearman was about `0.18`. The older `n=64` empirical cache showed
  the same pattern more weakly, with Spearman about `0.14`.
- Therefore the smooth object is the local flow field through response space,
  not a simple surface smoothly parameterized by image x/y position alone.

Phase-aware / spatial-frequency readout:

- A corrected phase-aware pilot used relative Fourier phase advance targets.
  It found that `motion_delta` responses cross-validatedly predict relative
  phase in spatial-frequency bands: for example `2-4 cpd` neural-to-phase mean
  R2 was `0.340`, and `4-8 cpd` was `0.386`, with high phase-vector cosine on
  held-out data.
- Absolute averaged phase was a bad simple target because component phases
  cancel; target resultant lengths were low and neural prediction was poor.
- The phase-aware projection did not turn the neural activity into a clean
  circular phase plane. This supports the mechanism-level intuition that
  response flow carries spatial-frequency-scaled phase-advance information,
  while rejecting the stronger visual claim that the manifold is obviously a
  simple phase circle or torus.

Compact tangent subspace shape:

- The compact tangent atlas gave the clearest shape language for the Figure 4
  tangent result. Cached cardinal endpoints in the compact tangent basis were
  far above random-basis capture and were mostly planar-but-irregular or
  line/ribbon-like. At `delta=0.25`, `k=10`, compact endpoint energy was about
  `0.612` versus random about `0.013`, with plane fraction about `0.923`.
- Regenerated finite ring/grid orbits on a 12-object smoke run showed stronger
  curvature: at `k=10`, plane fraction was about `0.932`, linear sheet R2 about
  `0.540`, quadratic sheet R2 about `0.967`, with `3/12` objects labeled
  `elliptical_loop`.
- The phase-rotation probe found paired-plane hints and moderately elliptical
  local loops. At `k=10`, ellipse circularity median was about `0.714`, but the
  global generator fit remained weak (`R2` about `0.22`). This argues for
  partial phase-like organization inside compact tangent geometry, not a clean
  shared translation generator.

Interpretation:

The current intuition should be:

```text
Small retinal translations move V1-twin responses through a compact,
curved, fan-like flow geometry. Locally, each fixation/image patch has its own
x/y tangent chart. Across many patches, those charts do not align as universal
signed x/y axes, but their finite trajectories share a low-dimensional,
locally coherent flow field. Spatial-frequency phase advance is present in the
responses, yet the neural geometry expresses it as ribbons, curved sheets, and
flow/curl structure rather than as a clean torus.
```

Claim boundary:

This is a visualization/geometry intuition layer, not a promoted functional
claim. It should help explain the compact tangent result and guide figures, but
it should not be used to claim a formal manifold topology, behavioral
optimality, or a globally smooth map from retinal position to neural state. The
strongest current statement is that the shape is recognizable and
low-dimensional as a response-flow field, while local image content and
frequency structure modulate the chart attached to each fixation.

## 2026-06-13: Active-Sensing Roadmap After Vernier, Fixation Regime, Image Structure, and Input Whitening

Status: `Open / organized synthesis with scaled BackImage twin drift-geometry adjudication and completed input-whitening negative result`.

Primary docs, code, and outputs:

- `active_sensing_roadmap_after_vernier_fixation_image_structure.md`
- `active_sensing_unit_space_provenance.md`
- `vernier_active_sensing_analysis_plan.md`
- `fixation_statistics_by_stimulus/`
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/`
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_twin_drift_geometry_pilot_twin_axis_only/`
- `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_twin_drift_geometry_scaled_n256_twin_axis_only/`
- `outputs/active_sensing_movie_information/input_whitening/`
- `outputs/active_sensing_movie_information/fem_scale_tradeoff/`

Motivation:

The active-sensing branch now has four important constraints: Vernier motion is
useful mainly under pose-aware readout and prefers reduced scale; fixation
statistics differ strongly across stimulus/behavioral regimes; BackImage local
image features weakly predict scalar FEM metrics but show robust local
edge-axis alignment; and the scaled BackImage twin drift-geometry run does not
show model-specific explanatory power beyond raw edge geometry. A fifth
constraint comes from the input-whitening cleanup: the old pooled temporal-PSD
metric showed temporal power spreading with larger motion, while the newer
Rucci-style spatial modulation audit points to small nonzero scales for
power-law flattening. The updated roadmap reframes the twin as a tool
for predicting local useful/harmful motion geometry and tradeoff surfaces, not
as an oracle proving that measured eye traces are globally optimal.

Current synthesis:

```text
Fixational eye movements create structured, phase-dependent input to foveal V1.
This structure can be useful when retinal pose is known, costly when pose is
hidden, and shaped by behavioral regime and local image geometry. FEMs also
reformat natural-image input, but pooled temporal-power spreading and
Rucci-style spatial power-law flattening have different scale optima.
```

Input-whitening result, superseded as a Rucci-style whitening test:

- The completed run evaluated `1458` retinal movies and `157464` metric rows
  under measured-drift, Brownian, and OU motion families.
- Estimated biological fixation drift was `D = 0.00110667 deg^2/s`
  (`3.984 arcmin^2/s`), with fit `R2 = 0.916`.
- In the primary `4-40 cpd`, `1-30 Hz` passband, measured biological drift
  moved the PSD slope from `-4.207` under stabilization to `-1.047`, spectral
  entropy from `0.194` to `0.579`, and spectral flatness from `0.006` to
  `0.345`.
- But the old no-cost temporal entropy/flatness objective usually kept
  improving to the top of the tested grid: `956 / 972` passband-metric optima
  chose `D_scale = 3`; all entropy and flatness optima chose `D_scale = 3`.
- A newer Rucci-style spatial audit shows this was not the correct whitening
  question. In smoke runs, total frame-to-frame modulation power still peaks at
  large motion, but spatial power-law flattening and derivative-transfer checks
  peak at small nonzero motion.
- The paired image/crop bootstrap was not computed by this runner, so the
  whitening result is currently a deterministic/SEM summary with passband
  sensitivity, not a bootstrap-resampled uncertainty claim.

Interpretation:

```text
The old pooled temporal-PSD metric measures temporal power spreading, not
Rucci-style power-law whitening. Spatial power-law flattening, total modulation
amount, and task/feature information must be treated as separate objectives.
```

This closes off the simplest ecological account:

```text
Biological FEM amplitude is not explained by either the old pooled temporal-PSD
metric or the first Rucci-style spatial flattening smoke by itself.
```

That cleanup is useful because it forces the active-sensing story into a
tradeoff framework. Pooled temporal modulation amount pushes motion scale
upward; Rucci-style spatial flattening, Vernier acuity, and pose-blind
covariance costs push toward smaller or more constrained motion; behavioral
regime and local image geometry determine where that tradeoff is expressed.

First cache-only tradeoff extension:

- `summarize_fem_scale_tradeoff.py` combines the completed input-whitening,
  covariance-optimality, and Vernier component-scale summaries without rerunning
  movies or the twin.
- The main conclusion is not "we found the missing cost." It is that the
  ingredients point in sensible directions, but a generic scalar tradeoff does
  not yet explain biological scale.
- `whitening_minus_pose_blind_covariance_cost` moves the optimum away from the
  upper whitening boundary, but in this first normalization it overshoots to
  small scale once the covariance penalty is strong enough. Median `D_opt` over
  the tested weights was `0.125`.
- `whitening_plus_vernier_acuity` is limited by the Vernier scale support
  (`0.25-1.5`) and tends to prefer `D=0.25`, consistent with the fine-acuity
  branch favoring reduced motion.
- Generic diffusion cost also overshoots to small scale. A one-sided
  above-biological window penalty is the only simple proxy in the first pass
  that recovers `D_scale = 1` under strong weights, but this is partly by
  construction because the penalty explicitly makes above-biological motion
  costly.

Interpretation:

```text
Adding costs can counter the whitening boundary, but the first simple cost
proxies do not automatically recover biological scale. Biological-scale recovery
requires a more specific and measurable constraint, such as fixation-window
loss, stability, blur, pose precision, usable temporal band, or motor cost.
```

Claim boundary:

This tradeoff pass is diagnostic only. The weights are not fitted, the motion
costs are analytic proxies, the Vernier term has limited scale support, and
independent normalization can move optima toward extremes depending on scale,
sign, and grid density. Its value is in ranking follow-up objective families,
not in claiming that a biological utility function has been identified.

Next non-circular tests:

The Rucci-style spatial power-law audit should be completed at larger trace and
image/crop scale with trace-level and image-level uncertainty, tiny-power
exclusion, and derivative-limit sanity checks. V1 temporal-sensitivity-weighted
whitening remains complementary: raw retinal temporal power asks whether the
temporal spectrum is flat, but foveal V1 does not use all temporal frequencies
equally. The better question is whether drift places natural-image temporal
modulations into the frequencies V1 can encode. Candidate weightings include
model-derived temporal sensitivity from drifting gratings or filtered natural
movies, output modulation spectra, derivative-weighted sensitivity
`|d mu / d s_f|^2`, or noise-normalized response gain.

BackImage image-structure result:

- Scalar local-image features do not robustly predict RMS radius, diffusion,
  speed, path length, anisotropy, return-to-center strength, or high-frequency
  FEM fraction over controls.
- The surviving result is directional: drift/fixation-cloud orientation tends to
  align modestly with local edge and spectral axes, especially in reliable-axis
  subsets.
- Same-image random-location controls suggest actual fixation locations are
  somewhat contrast-biased, but not uniformly higher across all information
  metrics.

Scaled BackImage twin drift-geometry adjudication:

- The scaled run used corrected eye-coordinate order, a `270 px` full-image
  support margin for `540 px` BackImage patches, an axis-only grid, `256`
  windows, `29` sessions, `5000` candidate-grid axis nulls, `5000`
  predicted-axis shuffles, and `5000` session bootstraps. Provenance audit:
  the folder's `n256` label is `max_windows=256`, while saved run metadata
  reports `twin_population_n=64`; this is a sampled-population diagnostic, not
  a full 756-channel canonical-population run.
- `raw_edge_axis` was the strongest biological baseline: session mean cos2
  `+0.182`, weighted `+0.218`, `23/29` positive sessions, random-axis
  `p_ge = 0.0004`.
- `optimized_PB` failed to beat raw edge: session mean cos2 `-0.019`,
  weighted `+0.008`; paired delta versus raw edge `-0.201`, CI
  `[-0.348, -0.064]`, with `5/29` positive sessions.
- `optimized_PA` was near zero/negative: session mean cos2 `-0.008`, weighted
  `-0.002`; paired delta versus raw edge `-0.190`, CI `[-0.389, +0.007]`.
- `optimized_Pareto_lambda_0.5` was also below raw edge: session mean cos2
  `-0.010`, weighted `+0.004`; paired delta versus raw edge `-0.193`, CI
  `[-0.357, -0.020]`.
- `adversarial_Pareto_lambda_0.5` was positive (`+0.167`) but not cleanly above
  predicted-axis shuffle nulls, so it is best treated as objective-landscape or
  image-geometry structure rather than biological optimized-axis evidence.

Interpretation:

```text
Observed BackImage drift is modestly and robustly aligned with local edge
geometry. The current V1-twin PA/PB/Pareto axis objectives do not outperform raw
edge orientation.
```

Post-fix Gabor/pyramid latent-information branch:

- The latent-feature implementation was patched after review. Gabor local
  fields now include even, odd, and amplitude maps on the local grid; pyramid
  local fields use the expanded local grid; model responses are trace-aligned
  before observer construction; and delta observers subtract the matched static
  response.
- First post-fix pathfinder:
  `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_latent_information_pathfinder_fixall_n64_rel0125-05_rand8_delta`.
  At `0.125x` observed RMS, canonical 756-unit twin, `k=4`, and rand8, Gabor
  pose-blind delta gave real-minus-random `+9.02`
  CI `[+1.56, +18.15]` and real-minus-edge `+11.58`
  CI `[+1.59, +24.55]`.
- Pyramid pose-blind delta pointed the same way but was noisier:
  real-minus-random `+10.28`, CI `[-0.53, +26.07]`;
  real-minus-edge `+25.04`, CI `[-3.78, +73.38]`.
- The apparent larger Gabor-only run,
  `backimage_latent_information_pathfinder_gabor_realrand_n128_rel0125-05_rand8_nogrd`,
  is not a clean post-fix replication. Its saved Gabor local field has shape
  `(128, 384)`, while the fixed 8x8 even/odd/amplitude Gabor local field has
  shape `(N, 4608)`, and it used absolute rather than delta observers.
- The stage2 run included pyramid and scales up to `2x`, but it was still n=64,
  used the older 4x4 feature dimensionality, and used absolute observers. We
  therefore do not treat it as a clean larger replication.
- Clean n=128 canonical replication:
  `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_latent_information_cleanrep_n128_rel0125-05_rand8_delta`.
  This run used canonical 756 units, fixed Gabor local fields `(128, 4608)`,
  fixed pyramid local fields `(128, 3072)`, `pose_blind_delta_mean`, rand8, and
  scales `0.125x`, `0.25x`, and `0.5x` observed RMS.
- Primary Gabor `k=4`, `0.125x`: real-minus-random `+3.31`,
  CI `[-0.14, +8.57]`, p(delta<=0) `0.0396`; real-minus-edge `-0.36`,
  CI `[-2.25, +1.09]`; edge-minus-random `+3.67`,
  CI `[+0.08, +9.28]`.
- Secondary scale rows argue against dismissing larger relative scales:
  Gabor `k=4`, `0.5x` real-minus-edge `+6.60`,
  CI `[+1.53, +11.83]`; pyramid `k=4`, `0.5x` real-minus-edge `+7.26`,
  CI `[+2.35, +12.08]`; pyramid `k=8`, `0.25x` real-minus-edge `+8.57`,
  CI `[+1.62, +20.85]`, with real-minus-random `+2.60`,
  CI `[-0.05, +6.27]`.
- Completed n=256 locked scale sweep:
  `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_latent_information_scalesweep_n256_rel0125-2_rand8_delta`.
  This used canonical 756 units, fixed local fields, `pose_blind_delta_mean`,
  rand8, and scales `0.125x`, `0.25x`, `0.5x`, `1x`, and `2x`.
  The cheap effective-scale audit lives at
  `posthoc_real_random_audit_summary.md`.
- The most stable real-vs-random positives are small scale:
  Gabor `k=4`, `0.25x` `+3.48`, CI `[+0.75, +6.87]`, with the unclipped
  subset still positive at `+2.85`, CI `[+0.21, +6.25]`;
  pyramid `k=8`, `0.25x` `+2.19`, CI `[+0.62, +4.18]`, unclipped
  `+1.86`, CI `[+0.45, +3.67]`.
- The `1x` results are alive but guarded: Gabor `k=4`, `1x` is `+2.59` with
  CI crossing zero; pyramid `k=8`, `1x` is weak globally, though the unclipped
  subset is `+1.40`, CI `[+0.05, +2.71]`.
- Clipping is substantial by large nominal scales: `18.8%` at `1x` and
  `40.2%` at `2x`, so large-scale positives must be interpreted by effective
  RMS rather than nominal labels.
- Subsampling from the n=256 output explains the mixed n=64/n=128 pathfinders:
  n=64 Gabor `k=4`, `1x` subsamples can be negative, while n=128 Gabor `k=4`,
  `0.25x` and pyramid `k=8`, `0.25x` are much more often positive.
- An optimized seed-dependence replication is running at
  `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_latent_information_scalesweep_n256_rel0125-2_rand8_delta_seed1_manifest_optimized_tb2`.
  It replays the same physical windows via `--window-manifest`, changes only
  the random-axis seed, and uses the patched canonical trace-batching path with
  `--check-trace-batch-equivalence`. The initial trace-batch-8 attempt OOMed
  during preflight; the active run uses trace batches of 2 and twin batches of
  48.
- New aggregate plan:
  `backimage_aggregate_fem_information_plan.md`. This shifts the figure-level
  question from exact local axis optimality to distributional adaptation:
  whether empirical FEM motion distributions improve ensemble natural-image
  representation over static, OU-matched, Brownian-matched, and shuffled
  controls.
- Cache-first aggregate proxy:
  `fixation_statistics_by_stimulus/summarize_backimage_aggregate_cache_proxy.py`
  reuses the completed n=256 latent/response arrays before new inference. The
  post-fix full nested-alpha output at
  `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_latent_information_scalesweep_n256_rel0125-2_rand8_delta/aggregate_cache_proxy_full_postfix_nested`
  confirms that cached motion-versus-static is strongly positive and grows with
  scale, while real-vs-random specificity is narrow: Gabor `k=4`, `0.25x`
  `+3.480` CI `[+0.642, +7.030]`; pyramid `k=8`, `0.25x` `+2.185`
  CI `[+0.617, +4.251]`. Treat the script as a scoring/regularization bridge,
  not as the full OU/Brownian/unpaired aggregate test.
- Clean aggregate pathfinder:
  `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_pathfinder_n64_k2_drift_only_common_unclipped_rel025-2_not_final`.
  This run used the canonical `756`-unit twin, grouped-by-image CV, a
  drift-only trace bank, common source traces across `0.25x`, `0.5x`, `1x`,
  `1.5x`, and `2x`, and empirical, OU, Brownian, and rotated controls. Motion
  bookkeeping was clean: `40` accepted trace sources, zero clipping, median
  effective/requested RMS `1.0`, and identical trace sources reused across
  scales. The result is not a straight null. For temporal PCA/DCT summaries,
  empirical incremental gain over static became more negative with scale
  rather than improving monotonically, arguing against the simplest
  "more motion is better" failure mode. Empirical traces still beat OU in
  several motion-derived contrasts, especially Gabor temporal PCA/DCT and
  pyramid at smaller scales, so drift-like trajectory statistics remain
  informative. However, rotated empirical traces were competitive with
  empirical, which means the evidence is not specific to original trajectory
  orientation relative to each image. The response summary decides the result:
  temporal PCA/DCT were negative relative to static, while `delta_mean` was
  strongly positive. Interpret this as a readout-dependent aggregate signal:
  temporal-code enhancement is not supported in the pathfinder, but
  low-dimensional motion-induced feature remapping remains alive.
- Patched substantial aggregate run:
  `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched`.
  This run used the canonical `756`-unit twin, `256` images, `K=4` trace
  samples, grouped-by-image CV, drift-only common-unclipped traces, source trace
  reuse across scales, empirical/OU/Brownian/rotated families, scales `0.25x`,
  `0.5x`, `1x`, `1.5x`, and `2x`, and Gabor/pyramid local-field latents at
  `k=4,8`. Motion bookkeeping was clean: `151/256` trace sources passed the
  strict drift-only filter, effective/requested RMS was `1.0` for every
  family/scale, and clipping was `0.0` everywhere. Use the corrected
  incremental posthoc folder
  `incremental_static_plus_motion_relids`; the first automatic
  `incremental_static_plus_motion` folder used old-style scale IDs and has
  empty gain tables.
- Primary aggregate result:
  temporal-PCA empirical motion added feature-decoding signal beyond the full
  static response. For Gabor `k=4`, static-plus-empirical gains were `+14.31`
  `[+7.45, +21.79]` at `0.25x`, `+13.04` `[+6.81, +20.89]` at `0.5x`,
  `+9.10` `[+3.73, +14.86]` at `1x`, `+9.98` `[+5.36, +15.87]` at `1.5x`,
  and `+9.07` `[+3.87, +15.73]` at `2x`. Pyramid `k=8` showed smaller but
  consistent gains: `+5.20`, `+4.89`, `+3.93`, `+4.44`, and `+4.21` across the
  same scale sequence, all with positive CIs.
- Control result:
  empirical temporal-PCA incremental gain beat OU robustly across scale. For
  Gabor `k=4`, empirical-minus-OU was `+21.24`, `+19.59`, `+17.16`, `+18.69`,
  and `+18.03` from `0.25x` to `2x`. Empirical also beat Brownian and rotated
  most cleanly at small scales: at `0.25x`, Gabor `k=4` empirical-minus-Brownian
  was `+10.52` and empirical-minus-rotated was `+15.27`; at `0.5x`, they were
  `+7.89` and `+11.21`. Brownian became competitive at `1x-2x`, so high-scale
  real-specific claims should be guarded.
- Scale interpretation:
  the cleaned aggregate result argues against the worst "more motion is better"
  artifact. Empirical temporal-PCA gain is strongest at `0.25x-0.5x` and then
  plateaus or decreases through `2x`, while effective RMS and clipping
  bookkeeping are clean. This makes the aggregate branch the best current
  BackImage active-sensing candidate, but with a distributional claim:
  empirical drift statistics are useful for the V1-twin representation of
  natural-image feature structure; exact biological trace order/orientation is
  not established as uniquely optimal.

Edge-parallel stability and twin metric audit:

- Endpoint-cache audit:
  `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_twin_stability_metric_audit`.
- The old relative twin disruption score had a weak wrong-direction controlled
  trend, but signed normalized disruption did not. Signed edge-parallel
  stability was first-order positive across pixel and twin metrics.
- Cheap synthesis:
  `backimage_twin_stability_metric_audit/cheap_synthesis/cheap_synthesis_report.md`.
  Session means and CIs: pixel `+300.5` `[+172.8, +408.8]`;
  twin raw MSE `+0.0004545` `[+0.0003716, +0.0005432]`;
  response-norm `+0.02456` `[+0.01993, +0.02931]`;
  per-rate `+0.003688` `[+0.002902, +0.004501]`;
  full-cov whitened `+0.1706` `[+0.1511, +0.1890]`.
- Pixel and twin signed advantages agree across windows: full-cov whitened
  `r = +0.277`, CI `[+0.168, +0.417]`; diagonal-whitened
  `r = +0.287`, CI `[+0.139, +0.419]`. Session-mean correlations remain noisy.

Axis-conditioned trajectory observer update:

- The direct edge-axis observer branch is now implemented and has clean
  shared-source `n=64`, `K=16`, scale `0.5x` runs for both primary candidate-set
  pressures.
- Clean matched-static run:
  `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1`.
  The manifest has `axis_shared_source_catalog=True` for `128/128` rows, source
  Jaccard `1.0`, and zero paired motion-stat deltas. Accuracy: zero-eye
  `0.641`, edge-parallel joint `0.859`, edge-orthogonal joint `0.828`;
  parallel-minus-orthogonal `+0.031` (`+2/64` trials).
- Clean hard-negative run:
  `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_shared_source_gpu1_n64_c4_k16_v1`.
  The manifest again has `axis_shared_source_catalog=True` for `128/128` rows,
  source Jaccard `1.0`, and zero paired motion-stat deltas. Accuracy: zero-eye
  `0.641`, edge-orthogonal joint `0.891`, edge-parallel joint `0.844`;
  orthogonal-minus-parallel `+0.047`. Trial-paired discordance is orthogonal-only
  `6` versus parallel-only `3` (exact McNemar `p ~= 0.51`), so this is a real
  diagnostic pattern but not a claim-level axis preference.
- The pre-fix `target128_c4_k32` hard-negative run remains useful as a warning,
  not as biological evidence. It showed a larger orthogonal advantage
  (`0.875` versus `0.766`, delta `+0.109`), but source Jaccard was only `0.143`
  because the parallel and orthogonal catalogs were not yet strictly
  same-source.
- Interpretation: the unmatched-catalog bug no longer explains the entire
  orthogonal tendency, but the clean effect is modest and candidate-set
  dependent. The branch currently supports "axis priors rescue above zero-eye"
  more strongly than "V1 prefers one biological edge axis."

Updated BackImage claim boundary:

```text
Real drift is not yet explained by a global feature-information maximizing
objective. The credible result is local preservation: edge-parallel motion
disrupts pixels and V1-twin responses less than edge-orthogonal motion. The
corrected Gabor/pyramid branch now has stable small-scale support at `0.25x`,
especially for Gabor `k=4` and pyramid `k=8`, but it is not a clean `1x` or
global infomax result. The aggregate branch is now stronger than the local
screen: empirical drift-like trajectories add feature-decoding signal beyond
static responses and robustly beat OU in the cleaned `n=256` run. The guardrail
is that Brownian/generic motion becomes competitive at larger scales and exact
trajectory orientation is not established as uniquely optimal. The new
axis-conditioned observer results sharpen this boundary: clean axis priors help
image inference, but the parallel/orthogonal sign is small and candidate-set
dependent. Treat the aggregate result as readout- and scale-dependent support
for empirical drift statistics, not as global infomax.
```

Claim boundary:

Do not claim that the current V1-twin objectives predict drift geometry. The
fair current claim is narrower: observed drift is robustly edge-aligned,
edge-parallel motion preserves local pixel/twin structure, pure pose-aware
response-modulation is the wrong objective, and the tested 64-sampled-unit
pose-blind/Pareto twin objectives do not add explanatory power beyond raw image
edge geometry.

Do not claim that input whitening predicts biological FEM amplitude. The fair
claim is that drift has a strong whitening benefit over stabilization, but
unconstrained whitening is incomplete as a scale-setting objective.

Practical next gates:

- Use `declan/figure4_active_sensing_atlas/claim_critical_diagnostics_queue.md`
  as the current gatekeeping index for core or claim-critical active-sensing
  analyses. In particular, model-objective claims require same-window residual
  explanation beyond raw edge geometry, global-axis nuisance checks,
  source-overlap/candidate-hardness audits, and population/readout sensitivity
  before they can move from diagnostic to mechanism.
- Keep corrected coordinate order and the `270 px` full-image patch margin fixed.
- Promote raw edge geometry to the baseline that any local active-sensing model
  must beat.
- Make feature-posterior joint decoding the next big observer push: reuse the
  exact-cache image posterior, attach Gabor/pyramid feature vectors to candidate
  images, and score feature recovery, joint-minus-zero feature gain, and
  known-minus-joint pose cost before treating image-identity accuracy as the
  endpoint for along-contour utility.
- Replicate the axis-conditioned observer with the shared-source catalog fixed:
  larger `n`, both `matched_static_response` and `hard_negative_structure`, and
  the half/natural/double scale sweep `0.5x`, `1.0x`, `2.0x`. Treat any pre-fix
  unmatched-catalog orthogonal advantage as diagnostic-only.
- Test whether raw image geometry, signed preservation, and candidate-set
  pressure explain the now-observed axis-dependent observer pattern.
- Use the twin next for revised objectives: sliding along edges, minimizing
  retinal change, V1 temporal-band whitening, pose precision, or constrained
  stability.
- Finish the optimized same-window seed replication. If the `0.25x`
  real-vs-random positives survive, treat the local `I_z` branch as
  regime-dependent support; if not, demote it behind edge-parallel preservation.
- For the aggregate BackImage FEM branch, the runner and substantial `n=256`,
  `K=4` result are now in hand. The next useful work is figure construction and
  targeted robustness, not another broad pathfinder: fixed/shared-alpha
  sensitivity, seed/source resampling, concise scale-curve plots for
  empirical-minus-OU/Brownian/rotated, and signal-motion covariance panels.
  Temporal PCs and temporal DCT should remain primary temporal-code summaries;
  mean and `delta_mean` should be reported as distinct
  integrating/motion-induced-remapping readouts. Deterministic ridge scores are
  linear decodability/information proxies unless a fixed noise/logdet model is
  added.
- Reopen the local BackImage `I_z` branch only as a local-pairing test, not as
  another broad fixed-axis optimizer screen. The plan is
  `backimage_local_pairing_Iz_revisit_plan.md`: test whether actual
  image-trace pairings beat matched unpaired empirical traces, rotated actual
  traces, OU/Brownian controls, and edge-axis baselines under grouped-by-image
  feature decoding.
- The pre-patch local-pairing pathfinder outputs remain diagnostic only. A code
  review after the first fixed-manifest `K_unpaired=32` pyramid result found
  that `--window-manifest` also restricted the matched-unpaired trace bank to
  the same 128 analysis windows, and the local runner used reduced feature
  geometry (`patch_size_px=160`, `latent_crop_px=96`, `local_field_grid=4`)
  rather than the corrected aggregate convention (`540`, `151`, `8`).
- Clean local-pairing/adjudication outputs now include
  `backimage_local_pairing_Iz_revisit_clean_fixedmanifest_sampledK32_gabor_pyramid_rel025_0p5_1_seed7_v1`,
  the `rel2` sentinel cache, and
  `backimage_feature_decomposition_adjudication_v3_local_rel05_rel2_filled`.
  They use a fixed 128-image manifest, full 3013-row trace pool, sampled
  matched-unpaired controls, corrected Gabor `(128, 4608)` and pyramid
  `(128, 3072)` feature geometry, zero same-trial matches, and zero clipping.
  The claim-relevant file is `incremental_gain_contrasts.csv`, not
  `decode_contrasts.csv`. Earlier clean-run examples showed actual paired
  traces beating matched empirical trace swaps for `delta_mean` in both feature
  families, e.g. Gabor `k=4` `0.25x` `+9.95` CI `[+0.73, +20.62]`, Gabor
  `k=4` `1x` `+8.27` CI `[+2.70, +14.79]`, pyramid `k=8` `0.25x` `+6.09` CI
  `[+1.51, +10.53]`, and pyramid `k=8` `1x` `+3.79` CI `[+1.46, +6.28]`.
  The current cache-filled adjudication separates readouts rather than forcing
  one winner: `pyramid_local_field k16 temporal_pca` is the top
  aggregate/ensemble candidate, while `pyramid_local_field k16 delta_mean`
  remains the local/mechanistic sensitivity readout. Rotated and matched
  controls remain caveats, so the local result is a motion-delta/local-pairing
  result rather than a general temporal-code or optimal-axis result.
- When rerunning revised free-viewing objectives, prefer a full canonical
  population or at least a larger sampled population. Keep 16-channel or
  smaller sampled variants as transfer checks, not discovery space.
- Run rotated Vernier controls before interpreting the current Vernier motion-axis
  result as stimulus-geometry-specific.

## 2026-06-12: Content-Routed Correct-Chart Alignment

Status: `Diagnostic / machinery validated, recorded effect fragile`.

Primary docs, code, and outputs:

- `content_routed_retinal_registration_analysis_plan.md`
- `compact_retinal_translation_geometry/run_correct_chart_swap_alignment.py`
- `compact_retinal_translation_geometry/summarize_correct_chart_swap_alignment.py`
- `compact_retinal_translation_geometry/audit_chart_swap_fold_availability.py`
- `compact_retinal_translation_geometry/diagnose_chart_swap_alignment.py`
- `outputs/compact_retinal_translation_geometry/all_sessions_nfold5_gainbottom_unitdot_v1/`
- `outputs/compact_retinal_translation_geometry/all_sessions_nfold50_gainbottom_unitdot_v1/`
- `outputs/compact_retinal_translation_geometry/all_sessions_trialdisjoint_drifttest_nfold3_gainbottom_unitdot_v1/`
- `outputs/compact_retinal_translation_geometry/all_sessions_trialdisjoint_drifttest_nfold5_gainbottom_unitdot_v1/`
- `outputs/compact_retinal_translation_geometry/chart_swap_fold_availability_audit_v1/`
- `outputs/compact_retinal_translation_geometry/chart_swap_diagnostics_v1/`

Motivation:

The content-routed retinal-registration plan reframes compact geometry correctly:

```text
U_trans is a shared transformation channel, not a shared pose coordinate system.
```

The right recorded-data question is therefore not "can a generic decoder recover
absolute eye position?" It is whether the correct image/time-specific fitted-twin
translation chart explains recorded response differences better than wrong
charts, gain-only structure, random geometry, unit-shuffled geometry, and
RF/readout-preserving controls.

What was implemented:

The A2 correct-chart runner pairs recorded fixRSVP repeats at the same
image/time condition, predicts response differences from the fitted-twin
translation chart, and compares true-chart alignment to matched wrong-chart and
subspace controls. It includes leakage audits, drift masks, compact/full chart
spaces, unit subsets, pseudo-spike positive controls, latency/history sweeps,
wrong-chart matching variants, and stratifications by prediction norm and image
structure. Follow-up work added fold-availability auditing across split rules,
trial-first drift-test variants, pseudo positive-control modes based on
split-aware linear chart injection, and a dedicated diagnostic atlas for
per-session effects and pair composition.

Current result:

The machinery works; the recorded effect is the fragile part.

Positive controls are clear. Split-aware linear chart injection is positive
across the tested variants, including the more difficult fold regimes, so the
pipeline can detect chart-aligned retinal-displacement structure when it is
present.

The cleanest recorded positive is narrow:

- In `all_sessions_nfold5_gainbottom_unitdot_v1`, the compact
  `global_rate | gain_bottom50 | k=10` row had `true_minus_wrong` mean
  `0.0746`, CI `[0.0449, 0.1034]`, `5/5` positive scored sessions, and the
  weakest control CI low was `0.0092`.
- But this is effectively an Allen-only result: Logan had `ok_no_valid_folds`
  under `drift_trial_disjoint n=5`.

Once the fold rule is changed to include Logan or to rebalance held-out trials,
the aggregate recorded effect is no longer robust:

- In `all_sessions_nfold50_gainbottom_unitdot_v1`, the same gain-bottom compact
  row was `0.0272`, CI `[-0.0418, 0.1182]`, `3/6` positive sessions, and
  required controls failed.
- In `all_sessions_trialdisjoint_drifttest_n5_gainbottom_unitdot_v1`, the
  gain-bottom compact row was `-0.0487`, CI `[-0.2421, 0.1447]`, `2/5`
  positive sessions.
- In `all_sessions_trialdisjoint_drifttest_n3_gainbottom_unitdot_v1`, all six
  sessions scored, but the gain-bottom compact row was `0.1268` with a very
  wide CI `[-0.1387, 0.5795]` and only `2/6` positive sessions.

The per-session atlas and pair-composition audit explain why this is not yet a
stable targeted claim:

- Allen_2022-03-02 can swing the aggregate under sparse regimes. In the
  trial-first `n=3` run it contributed only `10` gain-bottom pairs with a
  session CI width of about `2.24`, yet a large positive mean.
- Logan re-enters cleanly at larger `n_folds` or under trial-first testing, but
  its sign is not stably positive across those rules.
- The baseline `drift_trial_disjoint n=5` result is not obviously driven by one
  fold, but pair composition is concentrated in particular time/image bins,
  especially for sparse sessions.

Interpretation:

This branch changed roles during implementation. It is no longer best thought of
as "find the recorded displacement decoder." It is now a constraint-shaping
diagnostic that asks whether any apparent chart advantage is biological or an
artifact of fold/session composition.

The safe statement is:

```text
Compact chart geometry is recoverable under positive controls, but recorded
single-trial chart alignment is weak, subset-dependent, and highly sensitive to
fold/session composition.
```

At present, that means:

- The all-unit recorded bridge is not promoted.
- The gain-bottom positive is a targeted hint, not a stable claim.
- Covariance closure remains the promoted Figure 4 recorded bridge.

Next gates:

- Treat further chart-swap work as forensics, not broad optimization.
- Use `chart_swap_diagnostics_v1` to inspect per-session leverage and
  pair-composition concentration before any additional rerun.
- If this branch continues, make only one preregistered targeted rerun:
  define the subset rule from training/session metadata only, choose the fold
  rule up front, include Allen and Logan, and stop based on that result.
- If the preregistered targeted rerun does not survive, close this branch as a
  useful negative/fragile boundary result rather than a rescue path.

## 2026-06-12: Vernier Active-Sensing Hyperacuity Branch

Status: `Open / first-pass supportive for phase-cloud sampling`.

Primary docs, code, and outputs:

- `vernier_active_sensing_analysis_plan.md`
- `vernier_active_sensing/README.md`
- `vernier_active_sensing/run_vernier_active_sensing.py`
- `vernier_active_sensing/summarize_vernier_active_sensing.py`
- `outputs/vernier_active_sensing_first_pass/`
- `outputs/vernier_active_sensing_component_smoke/`
- `outputs/vernier_active_sensing_component_scale/`

Motivation:

Vernier offset is a cleaner active-sensing endpoint than E-optotype orientation:
it isolates a continuous fine-position variable rather than mixing object
orientation, stroke width, global identity, phase, and scale. The analysis asks
whether FEM-like motion improves recoverable sensitivity to tiny retinal
misalignment under explicit pose-aware and pose-blind observer assumptions.

Rendering/provenance audit:

The first-pass renderer uses a high-resolution Vernier stimulus rendered at
`120` world pixels/degree and sampled onto the model's `101 x 101` retinal grid
at about `37.50` pixels/degree. The audit confirmed balanced total luminance
for `+/-` offsets and finite pixel-level Vernier signal:

- model pixel pitch about `1.60` arcmin;
- finite-difference steps `0.25` and `0.5` arcmin;
- total luminance absolute delta `0.0` for both steps;
- pixel-level Fisher about `28.31` and `27.64` per arcmin squared.

First-pass outcome:

The summary was computed from cached first-pass model rates for `16` traces,
`60` frames, `756` units, and `0.25/0.5` arcmin finite-difference steps. At
`0.25` arcmin under the pose-aware diagonal-Poisson readout:

- `static_center`: mean Fisher `0.1753`, threshold proxy `2.389`.
- `static_phase_cloud_matched_positions`: mean Fisher `0.2733`, threshold proxy
  `1.962`.
- `real_fem`: mean Fisher `0.2676`, threshold proxy `1.958`.
- `order_shuffled_positions`: mean Fisher `0.2713`, threshold proxy `1.959`.
- `scaled_real_0.5`: mean Fisher `0.3293`, threshold proxy `1.765`.
- `scaled_real_1.5`: mean Fisher `0.2195`, threshold proxy `2.169`.

Contrasts:

- Real FEM beat static center in `16/16` traces at both finite-difference steps;
  at `0.25` arcmin, mean Fisher delta was `+0.0923` and threshold ratio was
  `0.820`.
- Static phase-cloud matched positions also beat static center
  (`+0.0981`, threshold ratio `0.822`), so the main benefit is phase-cloud
  sampling rather than a unique real-order trajectory effect.
- Real FEM was essentially tied with the phase-cloud control
  (`-0.0058` mean Fisher delta at `0.25`, positive in `9/16` traces).
- Order-shuffled positions also tied the phase-cloud control, again arguing
  against exact temporal order as the first-pass mechanism.
- Half-scale real motion was strongest in this first pass: versus phase-cloud,
  `scaled_real_0.5` had mean Fisher delta `+0.0559` at `0.25` and `+0.0413` at
  `0.5`; `scaled_real_1.5` was worse than phase-cloud at both steps.

Pose-blind caveat:

The pose-blind diagonal count-plus-marginal readout was much weaker for motion
conditions than pose-aware readout and even made `static_center` look strong.
Use this as a warning that observer assumptions dominate the absolute Vernier
information numbers.

Noise/readout clarification:

The Vernier result did not rely on simulated noisy spike draws or an empirically
fitted trial-noise model. The twin produced deterministic rates, and the
analysis derived Fisher/readout quantities from those rates under explicit
observer assumptions. The pose-aware result used diagonal-Poisson Fisher; the
pose-blind variant used diagonal count-plus-marginal covariance and behaved
very differently. This matters for aggregate natural-image analyses: a
deterministic ridge score is a linear decodability/information proxy, not
literal mutual information, unless a fixed noise/logdet formulation is added.

Incomplete component-scale run:

`outputs/vernier_active_sensing_component_scale/` currently has partial rate
caches and render audits, but no completed manifest or summary. The logged PID
is no longer live. Treat this as an incomplete run, not a result, until rerun or
summarized cleanly.

Interpretation:

The clean current claim is:

```text
For a controlled Vernier stimulus in the V1 twin, nearby phase-cloud sampling
improves pose-aware fine-offset information relative to a static center. Real
FEM and order-shuffled motion mostly match that phase-cloud benefit; exact
biological trajectory order is not yet supported as special. Motion scale
matters, with half-scale real motion strongest in the first pass.
```

This branch is useful because it gives Figure 5 a cleaner hyperacuity endpoint
than E-optotypes, but it remains observer-model-dependent and model-only until
paired with non-circular input statistics or recorded-data anchors.

## 2026-06-12: Non-Circular FEM Information and Covariance-Aware Optimality

Status: `Closed / supportive with guardrails; input-whitening branch closed as useful negative`.

Primary docs and code:

- `Non_circular_FEM_information_tests_prescription.md`
- `Covariance_aware_FEM_optimality_analysis_prescription.md`
- `jake/twininfo/covariance_optimality.py`
- `jake/twininfo/run_covariance_optimality.py`
- `active_sensing_movie_information/summarize_covariance_optimality.py`
- `active_sensing_movie_information/run_input_whitening_optimum.py`
- `active_sensing_movie_information/summarize_input_whitening_optimum.py`
- `outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu/covariance_optimality/covopt_full_gpu1/`
- `outputs/active_sensing_movie_information/covariance_optimality/covopt_full_gpu1/`
- `outputs/active_sensing_movie_information/input_whitening/`

Motivation:

The current Figure 5 result shows that natural-image retinal motion can improve
model spatial information efficiency over stabilization, but random trajectory
controls made "real FEMs are optimal" unsafe. The new direction keeps the
functional question alive while removing the easy circular path.

New interpretation:

There are now three distinct claims, with different evidentiary burdens:

- `Input whitening`: biological drift should be tested from image statistics
  and drift kinematics, not from the fitted twin. The completed run shows that
  drift whitens input relative to stabilization, but no-cost whitening does not
  select biological scale.
- `Recorded pose-aware information`: recorded V1 spikes may be more
  informative about stimulus labels when retinal pose or recent eye history is
  known. This is the direct cortex anchor.
- `Covariance-aware operating regime`: in the twin, movement may increase
  pose-aware information while creating nuisance covariance for pose-blind
  readouts. The safe quantity is the pose-aware minus pose-blind gap across
  movement scale.

What was implemented and run:

The covariance-aware path now has real code and a completed production run. It
reuses production `jake.twininfo` metadata, builds scaled real and
random-control trajectories, computes expected counts and finite-difference
derivatives, estimates movement-induced covariance, and summarizes independent,
covariance-aware, and pose-blind Fisher efficiency curves with gain/noise
sensitivity.

The input-whitening path also completed. It renders retinal natural-image
movies from production crops and selected fixation traces, sweeps measured,
Brownian, and OU motion families over scale, and computes temporal PSD slope,
spectral entropy, spectral flatness, autocorrelation time, and passband power
without using twin responses as the optimality endpoint.

The full GPU run completed `3888 / 3888` rate rows across `108` image/trace
pairs, four scaled trajectory families, and nine movement scales. The summary
tables and figures live under
`outputs/active_sensing_movie_information/covariance_optimality/covopt_full_gpu1/`.

The input-whitening run completed `1458` retinal movies and `157464` metric rows.
The summary tables and figures live under
`outputs/active_sensing_movie_information/input_whitening/`.

Outcome:

The primary metric was final Fisher trace per expected spike. The cleanest
one-sentence result is:

```text
Empirical D=1 is usually in the efficient operating range, but the run does
not show that D=1 is the unique optimum.
```

Peak/plateau calls for the covariance-aware pose-aware metric:

- `random_amp_cloud_matched_scaled / fixation`: empirical `D=1` value
  `74.51`; peak `D=2`, value `82.37`; empirical fraction of peak `0.905`;
  label `empirical_on_plateau`.
- `random_amp_cloud_matched_scaled / microsaccade`: empirical `76.67`; peak
  `D=0.5`, value `81.97`; empirical fraction `0.935`; label
  `empirical_on_plateau`.
- `random_amp_scaled / fixation`: empirical `77.22`; peak `D=1.5`, value
  `81.38`; empirical fraction `0.949`; label `peak_near_empirical`.
- `random_amp_scaled / microsaccade`: empirical `75.43`; peak `D=0`, value
  `79.75`; empirical fraction `0.946`; label `empirical_on_plateau`.
- `scaled_real / fixation`: empirical `64.42`; peak `D=3`, value `73.18`;
  empirical fraction `0.880`; label `empirical_on_plateau`.
- `scaled_real / microsaccade`: empirical `71.21`; peak `D=0`, value `79.75`;
  empirical fraction `0.893`; label `empirical_on_plateau`.
- `trajectory_order_shuffle_scaled / fixation`: empirical `65.17`; peak
  `D=2`, value `67.96`; empirical fraction `0.959`; label
  `empirical_on_plateau`.
- `trajectory_order_shuffle_scaled / microsaccade`: empirical `66.40`; peak
  `D=0.125`, value `88.73`; empirical fraction `0.748`; label
  `resolved_nonempirical_peak`.

Code audit and corrected covariance gaps:

On 2026-06-13 we inspected the implementation rather than trusting the
generated summaries. The first implementation had `cov_pose_aware = f_ind`,
while `cov_pose_blind` used the covariance-Fisher path with ridge
regularization. This made the independent and pose-aware rows identical and
introduced a small nonzero pose gap at `D=0`. The runner now computes
`cov_pose_aware` through the same covariance-Fisher path with no extra movement
covariance. After refreshing core result tables, the `D=0` pose-aware and
pose-blind rows match exactly.

Corrected pose-aware minus pose-blind covariance Fisher gaps at empirical
`D=1` remain positive in every family and larger for microsaccade traces:

- `random_amp_cloud_matched_scaled`: fixation `0.0541 +/- 0.0047`,
  microsaccade `0.2582 +/- 0.0185`.
- `random_amp_scaled`: fixation `0.0842 +/- 0.0063`, microsaccade
  `0.2472 +/- 0.0215`.
- `scaled_real`: fixation `0.0382 +/- 0.0032`, microsaccade
  `0.1952 +/- 0.0163`.
- `trajectory_order_shuffle_scaled`: fixation `0.0275 +/- 0.0024`,
  microsaccade `0.0925 +/- 0.0085`.

This preserves the qualitative read but sharpens it: measured real motion has a
clear pose-blind covariance cost, especially in microsaccade windows, yet
random amplitude controls match or exceed real. The branch supports
pose-relevant reafferent covariance, not unique optimality of measured FEM
trajectories.

The gain/noise sensitivity grid was stable: all eight family-by-kind labels
were unchanged across `9/9` tested gain/noise settings. This makes the
plateau/near-empirical language robust to the tested covariance-gain and noise
floor assumptions.

Input-whitening outcome:

The clean one-sentence result is:

```text
The old pooled temporal-PSD metric shows that larger retinal motion spreads
temporal power; the newer Rucci-style spatial audit shows that power-law
flattening peaks at small nonzero motion in smoke runs.
```

In the primary `4-40 cpd`, `1-30 Hz` passband, measured biological drift moved
PSD slope from `-4.207` to `-1.047`, entropy from `0.194` to `0.579`, and
flatness from `0.006` to `0.345`. The largest tested scale, `D_scale = 3`, then
improved these to slope `-0.810`, entropy `0.834`, and flatness `0.662` for the
measured-drift family. Across the full passband grid, `956 / 972` optima chose
`D_scale = 3`; the only exceptions were `16` measured-drift abs-slope rows at
`D_scale = 0.125` for higher temporal lower-bound passbands.

The whitening summary manifest reports `bootstrap_status = not_computed`; do
not cite bootstrap uncertainty for this branch until a dedicated image/crop
resampling implementation is added.

Claim boundary:

This is supportive Figure 5 evidence, but its wording should stay disciplined.
The displacement derivative and movement-induced covariance are structurally
coupled, so a pose-blind penalty along the displacement axis is expected. It is
valid evidence that retinal pose matters and that empirical FEM amplitude sits
near an efficient operating range in the twin. It is not proof that biological
FEM amplitude is exactly optimized, and it is weaker than a non-circular input
whitening optimum or a positive recorded-cortex pose-aware information result.

The input-whitening wording should also stay disciplined. It is a useful
negative result, not a failure. It shows that stabilization is bad for input
statistics and that biological drift helps, but it rules out raw input
whitening as a single-objective explanation of biological FEM scale. The right
next framing is a tradeoff frontier that combines whitening benefit with
pose-aware information, pose-blind covariance cost, Vernier acuity, V1 temporal
sensitivity, motor cost, and fixation-window constraints.

## 2026-06-12: Recorded Pose-Aware Prediction GLM

Status: `Closed / controlled null`.

Primary docs, code, and outputs:

- `recorded_pose_aware_active_sensing_analysis.md`
- `active_sensing_movie_information/run_recorded_pose_aware_prediction.py`
- `outputs/active_sensing_movie_information/recorded_pose_aware_prediction_pilot_allen_2022-02-16/`
- `outputs/active_sensing_movie_information/recorded_pose_aware_prediction_multisession_6pilot/`

Motivation:

This was the direct recorded-cortex bridge proposed for the active-sensing
story: ask whether measured eye state improves held-out prediction of recorded
V1 spike counts when added to a stimulus-time/PSTH model. The ladder was
intentionally cache-first and content-blind:

```text
M0: PSTH-only Poisson GLM
M_eye_only: eye features without PSTH
M1: PSTH + additive eye position/velocity/radius/speed
M2: PSTH + scalar eye-state/gain factor
M3: PSTH + additive eye features + coarse time-by-eye interactions
```

Controls included trial-disjoint folds, valid-aware shuffled-eye traces,
per-row penalty metadata, leakage audits, and session-bootstrap summaries.

Outcome:

The pilot on `Allen_2022-02-16` was technically clean: 102 units, 5/5 folds,
zero fit failures, zero max-iteration failures, zero shuffle self-donors, and
all leakage checks passed. The six-session batch was also technically clean:
6/6 sessions ok, 30/30 fold leakage audits passed, and no fit or max-iteration
failures.

The result was a null for the planned bridge:

- `M1_additive_eye` did not beat valid-aware shuffled-eye controls across the
  six-session batch: mean real-minus-shuffle `-0.074` bits/spike, CI
  `[-0.216, 0.052]`, 2/6 sessions positive.
- `M1_additive_eye` and its shuffled-eye control were both negative relative to
  the fitted `M0_psth_glm` in every session-level mean. Treat this as a
  secondary diagnostic rather than the central null because the archived run
  used a lighter penalty for the PSTH-only baseline than for augmented models,
  so some M0-relative loss can reflect stronger shrinkage of the shared PSTH
  columns in M1/M2/M3.
- `M3_time_by_eye_interaction` was worse than shuffled-eye, additive-eye, and
  scalar gain controls. `M3 - M1` was `-97.366` bits/spike, CI
  `[-270.324, -0.484]`, 0/6 sessions positive; `M3 - M2` was similarly
  negative.
- Some M3 folds had catastrophically bad held-out likelihoods despite no
  optimizer convergence flags, indicating overfitting/extrapolation pathology
  of this coarse interaction estimator rather than a meaningful biological
  negative.

Interpretation:

This does **not** refute the covariance or compact-geometry result. It refutes
a narrower downstream claim:

```text
Recorded V1 spikes are better predicted by this simple cache-first
pose-aware GLM ladder.
```

The covariance closure can still be true if the FEM-linked effect is sparse,
nonlinear, image-chart dependent, latency-sensitive, primarily a population
covariance phenomenon, or accessible only through geometry-constrained
translation structure. The GLM ladder used here is deliberately content-blind
and is not the true fitted-twin local translation chart.

Claim boundary:

Do not use this as a main Figure 4/Figure 5 knockout panel. If mentioned, frame
it as a useful controlled null:

> A simple cross-validated additive or coarse interaction GLM did not recover a
> pose-aware prediction benefit beyond shuffled-eye controls, indicating that
> the covariance effect is not trivially captured by content-blind eye-state
> regressors.

Strategic consequence:

Keep the recorded-data centerpiece as covariance decomposition plus compact
geometry. For Figure 5, lean on model natural-image information, SF-localized
benefits, sustained accumulation, input-whitening/ecological anchors, and
covariance-aware pose-aware versus pose-blind metrics. Avoid wording that
requires a positive recorded spike-prediction endpoint, such as a broad
"recorded V1 information improves when eye state is known" claim, unless a new
geometry-constrained or latency-aware recorded analysis lands.

Minimal further audits, if this thread is ever reopened:

- One lag sweep to check whether `eyepos_used` should be latency shifted. The
  current Fig3 cache stores eye position in the same trial/time bins as
  `robs/rhat`; no explicit neural-latency-shifted eye regressor was found.
- A tiny M3-only regularization/bin-size diagnostic to confirm the catastrophic
  negative likelihoods shrink to approximately zero rather than become
  positive.
- A cleaner nested-estimator rerun, if M0-relative deltas ever matter, using an
  offset PSTH or matched/unpenalized PSTH terms so eye covariates are isolated
  from baseline shrinkage.
- Stop if those audits do not change the conclusion; do not turn this into a
  rescue campaign.

## 2026-06-12: Structured Decoders and Forward Twin Denoising

Status: `Mixed: structured decoder closed / controlled null; forward denoising not promoted`.

Primary docs and code:

- `structured_translation_decoder_analysis.md`
- `compact_retinal_translation_geometry/run_windowed_siamese_relative_decoding.py`
- `compact_retinal_translation_geometry/run_tejas_style_eyepos_decoder.py`
- `forward_twin_reafferent_denoising_analysis.md`
- `forward_twin_reafferent_denoising/run_forward_twin_reafferent_denoising.py`
- `outputs/compact_retinal_translation_geometry/gain_orth_structured_cuda_test/`
- `outputs/compact_retinal_translation_geometry/gain_orth_structured_prod_v2_gpu0/`
- `outputs/compact_retinal_translation_geometry/gain_orth_structured_prod_v2_gpu1/`
- `outputs/forward_twin_reafferent_denoising_preview_patched_matched/`
- `outputs/forward_twin_reafferent_denoising_diag_zero_beh/`
- `outputs/forward_twin_reafferent_denoising_diag_fixed_alpha/`
- `outputs/forward_twin_reafferent_denoising_diag_image_time/`

Motivation:

The compact geometry result says that local translations generate
image-specific response charts inside a shared compact subspace. The new
decoder and denoising work asks whether that geometry predicts recorded
single-trial or pairwise signals in a way a global gain explanation cannot.

Structured decoder interpretation:

The important decoder test is no longer "can eye position be decoded?" A
flexible decoder might recover eye position from time, image context, global
state, or leakage. The sharper test is:

```text
gain-only decoder ~= chance on displacement orthogonal to local gain
compact structured decoder > gain-only on that component
```

The windowed Siamese decoder now supports the right ingredients for that test:
gain-orthogonal metrics, Poisson-weighted local chart inversion, and a rank-1
global-gain chart null. The Tejas-style absolute eye-position decoder is useful
as a permissive sanity check and convention check, but it should not be
presented as evidence for compact geometry.

Structured decoder outcome:

The corrected v2 structured decoder ran on six sessions split across two GPUs.
The test used trial-disjoint folds, same-condition pairs, fold-train condition
means for the local gain baseline, Poisson-weighted local chart inversion, a
rank-1 gain-projected chart null, and a chart-time-shuffled routing control.
All 30/30 fold leakage audits passed with zero shared trials and zero shared
trial pairs. The same-condition overlap warning is expected under the
trial-disjoint design.

The pooled six-session result was a controlled null on the headline endpoint:

- Compact chart inverse gain-orthogonal balanced sign accuracy:
  `0.4985 +/- 0.0053`.
- Rank-1 gain-only chart null: `0.4901 +/- 0.0056`.
- Chart-time-shuffled compact chart: `0.5083 +/- 0.0053`.
- Compact chart inverse gain-orthogonal correlation:
  `-0.0182 +/- 0.0066`.
- Compact chart inverse gain-orthogonal R2: `-2.66 +/- 0.53`.

The split GPU1 audit alone marked `candidate_positive`, but that was not stable
after pooling with GPU0. The combined result is effectively chance on the
gain-orthogonal direction metric and does not beat the chart-shuffle control.

Interpretation:

This does **not** undermine the promoted compact geometry/covariance closure
claim. It rejects the stronger decoder-level claim that a local twin-built
compact chart can recover recorded relative displacement, specifically the
component orthogonal to global gain, under the current strict controls. The
safe statement is:

```text
Recorded V1 contains compact, image-conditioned translation geometry at the
population/covariance level, but this run did not show a robust compact-specific
gain-orthogonal displacement readout from single-trial response differences.
```

Consequences:

- Do not promote Panel F as a positive decoder bridge.
- Treat the Tejas-style absolute decoder only as a permissive sanity check: it
  can show that neural activity carries some eye-related signal, but not that
  compact reafferent geometry provides a calibrated displacement coordinate.
- Keep the Figure 4 main claim on structural equivariance, compact tangent
  generalization, and covariance closure.
- Any future decoder rescue should change the scientific question explicitly
  rather than tuning this endpoint: e.g. data-built cross-fit local charts,
  latency-aware response windows, or a forward generative observer with a
  better trial-noise model.

Matched-context decoder refresh, 2026-06-16:

The relative-displacement decoder was then audited and patched in three ways:

- pair construction now uses image-aware matched contexts by default
  (`image_time_bin`) rather than allowing same-time / different-image pairing,
- promotion now requires a finite positive eye-label-shuffle effect rather than
  treating missing support as acceptable,
- `target_pc1` projection is estimated from fold-train tangent covariance with
  session-target fallback, rather than from a full-session target covariance
  inside each fold.

The patched production rerun
`outputs/compact_retinal_translation_geometry/relative_displacement_decoding_postpatch_prod_gpu1/`
completed on six sessions with `n_sessions_ok = 6` and remained `diagnostic`.
The refreshed result is informative because it separates "there is some
matched-context displacement signal" from "compact geometry provides a robust
skeptic-proof decoder."

Top-line results at `k=10`:

- `projection_control=none`: compact `R2_mean = 0.0746`, eye-shuffle excess
  `+0.1029`, CI `[0.0502, 0.1569]`.
- `projection_control=global_rate`: compact `R2_mean = 0.0506`, eye-shuffle
  excess `+0.0789`, CI `[0.0350, 0.1202]`.
- `projection_control=target_pc1`: compact `R2_mean = 0.0045`, eye-shuffle
  excess `+0.0345`, CI `[0.0112, 0.0637]`.
- `projection_control=global_rate+target_pc1`: compact `R2_mean = -0.0019`,
  eye-shuffle excess `+0.0284`, CI `[0.0083, 0.0539]`.

Interpretation:

```text
The patched run confirms that recorded responses contain matched-context
displacement information, but it does not show that compact geometry is the
privileged carrier of that information once global-rate and top-target-PC
structure are projected out.
```

Why this matters:

- The metadata/pathology fix landed cleanly: `pair_inventory.csv` now carries
  real `image_id` and `time_context` provenance instead of fallback labels, so
  the run is on the intended same-image footing.
- The raw positive under `projection_control=none` is not enough for a main
  claim because global-rate and especially `target_pc1` absorb most of the
  effect.
- The conservative result is therefore a limit statement: compact geometry is a
  strong covariance-level structural result, but current recorded single-trial
  displacement decoding does not isolate a compact-specific bridge beyond these
  broader low-dimensional response components.

Forward denoising interpretation:

The denoising branch adds a harder single-trial prediction:

```text
Delta r_twin = twin(real visual input, real behavior)
             - twin(stabilized retinal image, stabilized behavior covariates)
```

The primary mode is a stabilized retinal-image control with behavior covariates
held fixed (`stabilized_behavior=same`). Sensitivity runs also tested zeroed
behavior covariates, fixed model amplitude (`fit_alpha=fixed_1`), and
image-time folds. The runner now uses fold-trained response gains, fold-trained
scalar alpha by default, fold-trained compact/random/unit-shuffle bases, and
replicated shuffled-eye nulls. The important matched shuffled-eye null is
`shuffled_eye_trace_compact`, which projects shuffled-eye corrections into the
same train-fold compact tangent basis.

Promotion gate:

Promote only controlled excess effects, not raw variance reduction or raw
decoding accuracy. The critical comparisons are compact structured versus
gain-only on gain-orthogonal displacement, and compact/full forward correction
versus shuffled-eye and gain-only denoising controls.

Current gate result:

The structured decoder failed its promotion gate in `gain_orth_structured_prod_v2`.
Forward denoising is now interpretable and did **not** pass the eye-trace
specificity gate.

The corrected matched preview
`outputs/forward_twin_reafferent_denoising_preview_patched_matched/` used 24/24
sessions, `max_samples=128`, trial folds, `eye_reference=zero`,
`stabilized_behavior=same`, `n_nulls=20`, and matched
`n_eye_shuffle_nulls=20`. The raw compact full-forward correction was positive:

- `full_forward compact_k10` variance reduction:
  `+0.000922 [0.000427, 0.001514]`, positive `18/24`.
- `full_forward compact_k10` FEM-subspace reduction:
  `+0.00723 [0.00273, 0.01203]`, positive `17/24`.

It also beat random and unit-shuffled compact geometry controls:

- Compact minus random-k variance excess:
  `+0.000904 [0.000465, 0.001437]`, positive `19/24`;
  FEM-subspace excess `+0.00668 [0.00261, 0.01140]`.
- Compact minus unit-shuffled compact variance excess:
  `+0.000887 [0.000443, 0.001426]`, positive `18/24`;
  FEM-subspace excess `+0.00688 [0.00282, 0.01160]`.

But it did **not** beat shuffled-eye controls:

- Compact minus compact-projected shuffled-eye variance excess:
  `+0.000188 [-0.000206, 0.000645]`, positive `12/24`;
  FEM-subspace excess `+0.000835 [-0.00200, 0.00387]`.
- Compact minus full shuffled-eye variance excess:
  `-0.000051 [-0.000453, 0.000412]`, positive `11/24`;
  FEM-subspace excess `+0.000550 [-0.00241, 0.00372]`.

Diagnostics:

- `outputs/forward_twin_reafferent_denoising_diag_zero_beh/` kept positive
  compact excess over random/unit-shuffled controls but stayed null versus
  compact-projected shuffled-eye: variance excess `+0.000148
  [-0.000112, 0.000447]`, positive `12/24`.
- `outputs/forward_twin_reafferent_denoising_diag_image_time/` strengthened
  random/unit-shuffle excess but again stayed null versus compact-projected
  shuffled-eye: variance excess `+0.000122 [-0.000377, 0.000640]`,
  positive `11/24`. Treat this as a stress test because image-time folds alter
  PSTH support.
- `outputs/forward_twin_reafferent_denoising_diag_fixed_alpha/` was a
  calibration failure for compact denoising: compact minus gain-only was
  strongly negative, so the fold-fit scalar amplitude was doing real work.

Interpretation:

Forward-twin compact corrections carry a reproducible denoising signal relative
to random and unit-shuffled compact geometry, but the effect is not specific to
the actual trial eye trace under the matched shuffled-eye controls. This is a
useful diagnostic, not a main-figure bridge. It does not weaken the covariance
closure result; it says the current twin/metric captures second-moment geometry
more robustly than precise single-trial eye-trace phase.

## 2026-06-12: Manuscript Figure Assembly Refresh

Status: `Active figure polish`.

Primary code:

- `fig1/generate_fig1.py`
- `fig1/generate_fig1b.py`
- `fig1/generate_fig1c.py`
- `fig1/generate_fig1d.py`
- `fig1/generate_fig1f.py`
- `fig2/generate_figure2_3_combined.py`
- `fig3/generate_figure3_combined.py`
- `fig4_cov_TFTS/plot_covariance_binning_sweep_panel.py`

Interpretation:

The figure work is converging toward a clearer manuscript sequence.

- Figure 1 now foregrounds the experimental setup, gaze control, RF coverage,
  population examples, and a gaze-sorted single-unit example in one A-I layout.
- The combined covariance figure now emphasizes the decomposition of classical
  residual covariance into an FEM component and corrected residual, pools
  included subjects by default, and keeps a subject-split option for audit.
- The new Figure 3 compositor makes the digital-twin mechanism and compact
  reafferent geometry a single main-text chain rather than scattering those
  ideas across older figure workspaces.
- The covariance-binning sweep panel is a stability check: the recorded
  covariance-closure effect should not depend on one arbitrary spike-count
  window.

Claim boundary:

These edits improve exposition. They do not by themselves change claim status.
For scientific claims, defer to the relevant analysis sections and output
summaries.

## 2026-06-09: Active-Sensing Movie Information / Figure 5

Status: `Promoted`, with strong claim discipline.

Primary docs and outputs:

- `active_sensing_movie_information/README.md`
- `active_sensing_movie_information/active_sensing_movie_information_plan.md`
- `active_sensing_movie_information/figure5_additional_checks_prep.md`
- `Figure5_active_sensing_triage_plan.md`
- `outputs/active_sensing_movie_information/active_sensing_movie_information_figure/active_sensing_movie_information_figure.{png,pdf,svg}`
- `outputs/active_sensing_movie_information/active_sensing_movie_information_figure/active_sensing_movie_information_figure_caption.md`
- `outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu/`

Motivation:

After the E-optotype/covariance path became too easy to overstate, this thread
reframed active sensing around natural-image retinal movies. The central
question became: do measured FEM-like retinal movies improve a deterministic V1
twin's spatial information efficiency relative to stabilization, and what image
statistics explain that gain?

At-the-time plan:

- Use Jake's `jake.twininfo` production pipeline as the source of truth.
- Treat cumulative spatial SSI bits per expected spike as the primary endpoint.
- Use raw bits, bits/sec, expected spikes, Fisher, and retinal transform QC as
  companions, not as the primary claim.
- Separate three claims that had been blurring together:
  measured FEMs explain recorded V1 shared variability; retinal motion improves
  a model information proxy; and the animal's exact trajectories are uniquely
  useful.

Main outcome:

- Real FEM retinal motion improved final spatial information efficiency over
  stabilization by about `+0.035` bits/expected spike.
- The saved caption reports real increasing the endpoint from `0.110` to
  `0.145` bits/expected spike, with 95% CI `[0.026, 0.045]`.
- The real-minus-stabilized cumulative curve stayed positive at all sampled
  time points in the figure summary.
- Raw information and expected spike count also increased, but the bits/spike
  endpoint survived spike-count normalization.
- Spatial-frequency controls showed a graded mechanism: lowpass produced a
  small gain, mid/high SF bands produced larger gains.

Important later interpretation:

- Random trajectory controls matter. `random_amp`, `random_cov`, and especially
  `random_amp_cloud_matched` equaled or exceeded real FEMs on the current
  bits/expected-spike endpoint.
- The current safe claim is: real retinal motion improves model spatial
  information efficiency over stabilization through a spectral-temporal
  mechanism.
- The current unsafe claim is: real FEM trajectories are optimal.
- Natural-image-only Checks 5-9 supersede the old cached e-optotype scaffolds
  for Figure 5 evidence.

Historical scaffold:

- `outputs/active_sensing_movie_information/figure5_cached_rate_checks_5_to_9_fixed_lm-020/`
  and the Check 8 add-back run are useful debugging context.
- The cached e-optotype scaffold found real residual structure more aligned
  with stimulus axes than stabilized, higher covariance-efficiency ratio `eta`,
  and positive remove-out effects. But matched/null controls also improved in
  places, and the stimulus was synthetic E-optotype rather than natural image.
- Do not promote those e-optotype checks as Figure 5 evidence.

Open follow-ups:

- Natural-image population Checks 5-9 are the current route for constrained
  population coding, pose-aware recoverability, and amplitude/diffusion sweeps.
- Compact add-back/remove-out should wait until the compact basis is
  dimension-compatible with the natural-image center-channel response space.
- Response-space accounting matters here. The natural-image Checks 5-9 and the
  current covariance-optimality run use 16 center/session-matched biological
  twin channels; the historical compact-geometry add-back scaffold used the
  canonical 756-channel Figure 4/TFTS basis. The tempting hierarchy
  `cov_pose_aware >= cov_geometry_aware >= cov_pose_blind` is plausible, but it
  is not yet an implemented matched comparison in this natural-image branch.

## 2026-06-09 / 2026-06-08: Compact Retinal-Translation Geometry

Status: `Promoted`, with some panels still carrying explicit caveats.

Primary docs and outputs:

- `compact_retinal_translation_geometry/README.md`
- `compact_retinal_translation_geometry_implementation_spec.md`
- `outputs/compact_retinal_translation_geometry/`
- `outputs/compact_retinal_translation_geometry/tables/acceptance_matrix.csv`
- `outputs/compact_retinal_translation_geometry/figures/panelA_local_translation_charts.{png,pdf}`
- `outputs/compact_retinal_translation_geometry/figures/panelB_compact_tangent_spectrum.{png,pdf}`
- `outputs/compact_retinal_translation_geometry/figures/panelC_cross_image_generalization.{png,pdf}`
- `outputs/compact_retinal_translation_geometry/figures/panelE_covariance_closure_full_vs_compact.{png,pdf}`
- `outputs/compact_retinal_translation_geometry/figures/metric_structure_summary.{png,pdf}`

Motivation:

This was created to turn the Figure 4 tangent/covariance material into a
coherent hidden-coordinate-style result: small retinal translations produce
image-dependent response changes, but those changes live in a compact,
image-generalizing population geometry that predicts recorded FEM covariance.

Panel logic:

- A: image-dependent local translation charts.
- B: compact tangent spectrum.
- C: cross-image tangent generalization.
- D: variability budget / denominator context.
- E: recorded covariance closure, full finite-difference source versus compact
  k=10 source.
- Metric validation: the coordinate-like hidden-geometry test.
- F / decoding bridge: optional, promote only if recorded displacement decoding
  survives leakage and null checks.

Main outcomes:

- Panel B compactness passed: observed participation ratio was about `9.04` at
  `0.25` arcmin, far below the unit-shuffle samplewise null around `31.0`.
- Panel C generalization passed: an image-disjoint compact basis at `k=10`
  captured about `0.525` held-out tangent variance versus null around `0.122`.
- Panel E covariance closure passed: full finite-difference translation sources
  predicted recorded FEM covariance above unit-shuffle and RF/readout nulls.
- Compact k=10 retained the closure. The compact-to-full capture ratio at k=2
  was about `1.005`; because the compact source is a separately constructed
  cross-fit source rather than a guaranteed nested restriction of the full
  finite-difference source, this should be read as no detectable closure cost,
  not as compact outperforming full.
- Under the conservative `global_rate+target_pc1` projection, PSD full
  finite-difference source at `k=10` captured about `0.535`; the compact
  cross-fit source captured about `0.536`, with positive effects over RF/readout
  fixed-permutation nulls in 24/24 sessions. This tiny ordering difference is a
  comparison/estimator caveat, not a promoted biological effect.
- Metric structure has partial support: rank-2 local compact metrics pass, and
  displacement scaling is strong (`R2 ~0.995` for norm/metric scaling), but
  coordinate recovery and diagonal composition are not fully landed because the
  current cache only has cardinal `+/-x` and `+/-y` translations.

Important caveats:

- Do not claim a universal literal 2D eye-position map in V1.
- Do not claim behavior or perceptual optimality from these panels.
- Do not claim the compact spectrum survives RF/readout-preserving samplewise
  null or projection-control spectrum until those are explicitly run.
- Do not promote recorded displacement decoding yet; the acceptance matrix
  marks Panel F decoding as not run for promotion, despite smoke/prod machinery
  existing.

Later interpretation:

This is the current structural spine for Figure 4. It absorbed the safer parts
of the older TFTS, covariance closure, and recorded-derivative branches, while
keeping performance/active-sensing claims separated into Figure 5.

## 2026-06-08: Direct Recorded Derivative / Twin Tangent Alignment

Status: `Supportive`.

Primary docs and outputs:

- `direct_recorded_derivative_twin_alignment/README.md`
- `direct_recorded_derivative_twin_alignment_prescription.md`
- `outputs/direct_recorded_derivative_twin_alignment_prod/README.md`
- `outputs/direct_recorded_derivative_twin_alignment_prod/tier1_compact_basis_bootstrap_summary.csv`

Motivation:

The covariance-closure result asks whether fitted-twin finite-difference
translation covariances predict recorded `Sigma_FEM`. This branch asked a more
direct but noisier question: if we estimate eye-position derivatives directly
from recorded V1 repeats, do those derivatives lie in the compact fitted-twin
translation geometry?

At-the-time guardrail:

Do not try to resurrect a clean image-specific signed `x/y` derivative match
between recording and twin. Older STG work showed signed/context-specific
derivative recovery was fragile. The primary claim should be compact-subspace
enrichment, not signed-axis recovery.

Outcome:

- Tier 1 survived the conservative control in the eligible-session set.
- Primary condition: `target_variant=psd`,
  `projection_control=global_rate+target_pc1`,
  `context_subset=reliability_qualified`, `k=10`.
- Capture mean was `0.386`.
- Effect over RF/readout null was `+0.210`, CI `[0.178, 0.246]`.
- Effect over unit-shuffle null was `+0.288`.
- Effect over random-subspace null was `+0.284`.
- Sign consistency was 13/13 eligible sessions, sign-test p `0.000244`.

Interpretation:

This is a supportive direct recorded-data bridge: recorded eye-position
sensitivity is enriched in the compact twin tangent subspace. It strengthens
the compact-geometry story but does not supersede covariance closure and should
not be phrased as signed horizontal/vertical axis recovery.

## 2026-06-07 / 2026-06-08: Matched Twin Covariance Closure

Status: `Promoted`.

Primary docs and outputs:

- `matched_twin_covariance_closure/README.md`
- `matched_twin_covariance_closure/rf_readout_preserving_null_prescription.md`
- `outputs/matched_twin_covariance_closure_finite_difference/`
- `outputs/matched_twin_covariance_closure_rf_null_step025_rfbacked_v2/`

Motivation:

This thread asked whether recorded FEM covariance in Ryan's matched
recorded/twin unit space is captured by fitted-twin eye-position structure and,
more strictly, by fitted-twin finite-difference retinal translation tangents.

At-the-time path:

- Start with cache-only eye-position regression because it was the strongest
  analysis possible from Ryan's Fig2/Fig3 caches alone.
- Replace that proxy with true finite-difference fitted-twin retinal
  translation tangents once model reconstruction was stable.
- Add projection controls and nulls: random subspace, unit shuffle, and later
  RF/readout-preserving nulls.

Outcome:

- The 24-session finite-difference sweep ran successfully.
- PSD `fd_sample_eye_trace_cov`, `k=2`, no projection: mean capture `0.531`,
  mean effect over unit shuffle `0.368`, positive in 24/24 sessions.
- With `global_rate+target_pc1` projection: mean capture `0.220`, mean effect
  `0.177`, positive in 24/24 sessions.
- Bootstrap CIs stayed positive; for PSD samplewise k=2,
  `global_rate+target_pc1` effect was `0.177`, CI `[0.144, 0.212]`.
- Raw target variants were also positive, though PSD is cleaner for
  variance-capture summaries.
- Step-size sensitivity on Allen was stable from 0.25 to 1.0 px.
- The RF/readout-preserving null extension became the stronger reviewer-facing
  version and feeds the compact geometry Panel E.

Interpretation:

This supports a substantial first-order retinal-translation component of
recorded `Sigma_FEM` geometry in matched recorded/twin unit space. It is not a
complete explanation of all FEM covariance.

Later refinement:

The strictest useful wording is now: finite-difference fitted-twin retinal
translation sources, including compact-restricted k=10 sources, predict a
reliable component of recorded FEM-linked covariance above unit and RF/readout
preserving nulls. Avoid "the twin fully reproduces recorded covariance."

## 2026-06-07: Figure 4 Covariance / TFTS Figure Work

Status: `Historical -> Integrated`.

Primary docs and outputs:

- `fig4_cov_TFTS/update.md`
- `fig4_cov_TFTS/covTFTS_figure_panel_prescription.md`
- `fig4_cov_TFTS/covTFTS_figure_data_forward_prescription.md`
- `fig4_cov_TFTS/figure4_panelF_natural_structure_coda_plan.md`
- `outputs/twin_feature_tangent_structure_prod_v2/MANUSCRIPT_REPORT.md`
- `outputs/compact_retinal_translation_geometry/`

Motivation:

This was the first attempt to make a clean Figure 4 out of recorded
reafferent covariance, local translation charts, tangent compactness,
cross-image generalization, and partial covariance bridging.

At-the-time figure claim:

The figure should communicate that recorded V1 shared variability is
reafferent, local retinal translations define image-specific response tangents,
and those tangents form a compact, image-generalizing structure. It should not
claim behavioral benefit or hard-code unfinished Panel E/F interpretations.

Outcomes:

- The tangent-family structural result landed in
  `outputs/twin_feature_tangent_structure_prod_v2/`.
- Production report status: `core_structural_result_passed`.
- At `0.25` arcmin, union compactness PR was `9.04` versus null mean `31.03`.
- Train/test basis at `0.25` arcmin and `k=10` captured median held-out
  variance `0.552` versus null median `0.118`.
- Local first-order covariance approximation showed locality dependence: the
  tangent approximation was more sensible at smaller cloud scales and
  over/under-scaled as the finite cloud grew.

Later interpretation:

The material was too broad as a single ad hoc figure workspace. Its stable
parts were promoted into `compact_retinal_translation_geometry/`. The proposed
Panel F natural-image coda remains conceptually useful but optional: it should
only enter the main figure if high-structure natural patches preferentially
route drift-scale response changes through the compact tangent basis above
matched controls. Otherwise, keep it as supplement or cut it.

## 2026-06-04: Natural Image Tangent Scale

Status: `Open`.

Primary docs:

- `Natural_Image_Tangent_Scale_Analysis_Handoff.md`
- `natural_image_tangent_scale/run_natural_image_tangent_scale.py`

Motivation:

TFTS showed that small retinal translations produce compact,
image-generalizing tangent structure. This follow-up asks how far the local
tangent description remains valid before finite displacement leaves the local
linear regime, and whether that breakdown scale depends on natural-image
structure.

Key guardrail:

Because the twin was trained with FEM-jittered retinal inputs, an absolute
match between tangent breakdown scale and FEM amplitude could be circular. The
non-circular gate is image-structure dependence: breakdown scale must vary
systematically with natural-image structure before making an ecological claim.

Outcome so far:

The module and runner exist, but this thread is not yet summarized as a closed
result in the `declan/` docs. Treat as an open ecological-anchor analysis.

Interpretation if it lands:

- If breakdown scale depends on gradients/SF/structure, the local compact
  geometry is tied to image curvature rather than only to the model's training
  eye-jitter distribution.
- If breakdown scale is flat and merely near FEM amplitude, report the scale
  gate as failed and do not compare to empirical FEM amplitudes.

## 2026-06-04 / 2026-06-03: Twin Feature Tangent Structure

Status: `Promoted`, now largely folded into compact geometry.

Primary docs and outputs:

- `Twin_Feature_Tangent_Structure_Prescription.md`
- `twin_feature_tangent_structure/run_twin_feature_tangent_structure.py`
- `outputs/twin_feature_tangent_structure_prod_v2/MANUSCRIPT_REPORT.md`

Motivation:

Earlier signed cross-image tangent alignment was too strict and could be near
zero, because translation tangents are image-specific. This pivot asked whether
the conserved object is not a signed universal `x/y` axis, but a compact
feature-defined tangent subspace, metric law, or operator family.

At-the-time claim:

Different images generate different translation tangents, but those tangents
are produced by a shared feature operator and may live in a compact,
image-generalizing subspace.

Outcome:

- Core structural stop rule passed.
- Union compactness was well above null at all tested deltas.
- Train/test generalization passed across folds and k values.
- The output report labels the claim state as
  `core_structural_result_passed`.

Later interpretation:

This is the first-order mechanism behind the compact retinal-translation
geometry. It should be described structurally: first-order tangents occupy a
compact shared subspace and generalize across images. It should not be turned
into a behavioral, optimality, or decoder claim.

## 2026-06-03: Shared Transformation Geometry

Status: `Historical / partially superseded`.

Primary docs and outputs:

- `archive/superseded_handoffs/shared_transformation_geometry_handoff.md`
- `shared_transformation_geometry_handoff_v2.md`
- `shared_transformation_geometry/README.md`
- `outputs/twin_covariance_structure/shared_transformation_geometry/`

Motivation:

STG asked whether recorded V1 contains a conserved retinal-transformation
geometry across images beyond trivial displacement magnitude and image
similarity. It was an ambitious recorded/twin bridge for signed tangent maps,
twin-template matching, and residual RDM geometry.

At-the-time correction:

The early RDM framing was demoted because an RDM is symmetric and cannot
distinguish signed displacement direction. Signed tangent-map comparison became
the primary signed test; residual RDM geometry became secondary/diagnostic.

Outcome / lessons:

- The infrastructure produced support census, tangent-map, template-match,
  residual RDM, and aggregation runners.
- It established useful patterns: session-level inference, support census
  first, image-similarity controls, drift-only masking, and explicit
  `control_not_evaluable` labels.
- It also exposed fragility: clean signed image-specific recorded derivative
  manifolds and exact recorded/twin signed-axis matches were not robust enough
  to headline.

Later interpretation:

STG became a reference layer rather than the final claim vehicle. Its safer
ideas were absorbed into direct recorded derivative Tier 1 and compact
covariance closure. Do not resurrect the strong signed-axis STG claim without
new evidence.

## 2026-06-03: Twin Covariance Structure

Status: `Supportive / framing pivot`.

Primary docs and outputs:

- `Twin_Covariance_Structure_Prescription.md`
- `twin_covariance_analysis_plan.md`
- `twin_covariance_structure/README.md`
- `outputs/twin_covariance_structure/`

Motivation:

This prescription separated what the deterministic twin can answer from what
requires a noise model. The twin is a good instrument for structure: low rank,
signal alignment, image specificity, occupancy dependence, translation tangent
alignment, and single-neuron-to-population bridges. It is not a good standalone
instrument for whether FEM covariance helps or hurts coding.

At-the-time interpretation:

The recording proves reafferent covariance exists and dominates positive noise
correlations; the twin explains why that reafferent covariance has its
structure.

Core conceptual outcomes:

- Signal alignment is not automatically a catastrophe; moving the image and
  changing the image drive should overlap in response space.
- Low rank should be tied to 2D translation plus finite-cloud curvature.
- Image specificity distinguishes reafference from global state.
- Occupancy, not trajectory order, governs second-moment covariance structure.

Later interpretation:

This prescription was important because it stopped the twin from being treated
as a contested performance oracle. Its structural pieces feed Figure 4 and
compact geometry. Functional/information claims are kept separate and require
explicit noise/readout assumptions.

## 2026-06-01: Keystone / Geometry-Crossover Link

Status: `Open / adjudication plan`.

Primary docs:

- `archive/superseded_handoffs/Keystone_Geometry_Crossover_handoff_v2.md`
- `Keystone_Geometry_Crossover_handoff_v3.md`
- `archive/early_bigpicture/bigpicture_fem_v1_high_impact_analysis_plan_v2.md`
- `archive/early_bigpicture/bigpicture_phase1_fem_v1_coding_agent_plan_v2.md`

Motivation:

After the E-optotype crossover and translation geometry were both in hand, the
keystone thread asked for the missing link: does a decoder-free geometry
quantity predict the sign, transition LogMAR, and magnitude of the FEM accuracy
advantage?

At-the-time design:

- Tier 1: cloud-separability gain `G_sep`, computed from deterministic mean
  responses over real versus stabilized position clouds.
- Tier 2: Jacobian/tangent mechanism `DeltaM`, asking whether translation
  mimicry/tangent escape tracks the same transition.
- The firewall: geometry observables cannot use decoder outputs or noise models,
  otherwise the test is circular.

Current interpretation:

This is an adjudication plan, not a closed result in `declan/`. It is valuable
because it formalized a clean distinction:

- `geometry_predicts_global_crossover`: geometry predicts and explains the
  functional crossover via tangent mechanism.
- `geometry_predicts_crossover_via_sampling_not_tangent`: cloud sampling
  predicts the crossover, but the equivariant tangent story does not.
- `geometry_tracks_difficulty_not_mechanism`: geometry is just difficulty.

Given later Figure 4/Figure 5 separation, this plan is less central than it was
when Figure 4 was trying to carry the active-sensing crossover.

## 2026-05-29: Jacobian Audit / Predictive Framework

Status: `Historical -> partially rescued as structure`.

Primary docs and outputs:

- `jacobian_results/results_and_interpretation.md`
- `archive/jacobian_early/jacobian_predictive_framework_progress_summary.md`
- `archive/jacobian_early/jacobian_predictive_framework_handoff_revised.md`
- `eoptotype_jacobian_field_smoothness_handoff.md`
- `fem_path_integrated_separability_handoff.md`
- `outputs/stats/eoptotype_jacobian_field_*`
- `outputs/stats/fem_step_jacobian_*`

Motivation:

The original Jacobian hypothesis was that FEM-induced covariance might be
predicted by a first-order pushforward:

```text
C_FEM ~= J Sigma_eye J.T
```

The appeal was strong: it would connect eye motion, local image translation,
and population covariance in one equation.

Outcome:

- Direction worked robustly. The image-translation Jacobian captured the
  leading FEM covariance subspace with alignment roughly `0.40-0.60`, 2-4x
  above null.
- Magnitude was fragile. Naive `J_static x Sigma_frame` overpredicted by
  `6-490x`; `J_eff x Sigma_trial` underpredicted by `0.003-0.053x`.
- Position-histogram integrated `J_int x Sigma_total` got near scale agreement
  at `lm=-0.20` for three of four orientations, but remained off at
  `lm=-0.40`, consistent with grid resolution being too coarse for tiny E
  strokes.
- Representational intervention with stimulus-specific J could raise decoding
  to 100%, but the class-specific nature of that manipulation made it too easy
  to overinterpret. Pooled-J controls were safer and less dramatic.

Later interpretation:

The magnitude identity is a closed/failed branch at full cloud scale. The
directional/subspace result survived and became the right way to use Jacobians:
translation tangents define the geometry, but do not by themselves provide a
full covariance magnitude identity.

The smoothness work also changed the story: the issue was not a wildly rough
Jacobian field. Instead, finite-cloud scale, phase, curvature, and resolution
explain much of the magnitude mismatch.

## 2026-05-26: E-Optotype Hyperacuity, Crossover, and Covariance Ablations

Status: `Closed`, with a narrowed mechanism.

Primary docs and outputs:

- `archive/early_bigpicture/revised_analysis_plan.md`
- `FEM_population_coding_writeup.md`
- `archive/eoptotype/fem_eoptotype_hyperacuity_results.md`
- `fem_covariance_geometry.py`
- `fem_global_intervention.py`
- `fem_differential_intervention.py`
- `eoptotype_continuous_pass.py`
- `declan/fem_covariance_geometry_results/`
- `declan/fem_global_intervention_results/`
- `declan/fem_differential_intervention_results/`
- `declan/continuous_pass_results/`
- `declan/gru_passthrough_figures/`

Motivation:

This was the first functional active-sensing arc: real FEMs appeared to hurt
orientation decoding at larger E sizes but help near the hyperacuity regime.
The exciting hypothesis was that information might migrate from mean rates into
FEM covariance geometry or temporal trajectory structure.

At-the-time established facts:

- The twin has real temporal processing within its window, and model
  correlations are purely reafferent by architecture.
- D1 time-averaged rate showed a real-vs-stabilized crossover around
  `LogMAR ~ -0.32`.
- Real FEM hurt at `-0.20/-0.25`, became roughly neutral near `-0.30`, and
  helped around `-0.35` to `-0.40` under the windowed pipeline.
- Spatial SSI on E-optotype at `-0.20` increased under real FEM despite
  stabilized outperforming real in orientation decoding, implying a readout/task
  distinction.

Closed outcomes:

- Covariance-code migration was false. FEM subspaces did not rotate with E
  orientation; off-diagonal overlap was ~1.0. Covariance decoders were near
  chance, and combined covariance features added essentially nothing over D1.
- Alignment transition was real: alpha was higher near `-0.20` than `-0.40` in
  real FEM, with stabilized showing the opposite ordering.
- Signal geometry likely moved: `C_signal` eigenvalues were larger at `-0.40`,
  and overlap with translation nuisance directions fell.
- Pooled FEM-subspace ablation improved real at `-0.20` and was null at
  `-0.40`, matching the alpha pattern, but the stabilized control also
  improved. Therefore it removed a generic positional nuisance, not a uniquely
  dynamic-FEM covariance component.
- Differential `C_real - C_stabilized` ablation also failed to isolate a
  real-specific causal covariance mode.
- Temporal coding remained null. D3/temporal residual features did not rescue
  orientation information; continuous forward pass degraded performance and
  stayed below the windowed pipeline.
- Fixed-center was exposed as a deterministic oracle, not a biological static
  baseline.
- Among nonzero FEM amplitudes, larger movements reduced E-orientation decoding
  in this model/readout; no inverted-U or optimal biological amplitude emerged.
- `-0.40/-0.45/-0.50` formed a model-native retinal saturation plateau, so the
  smallest nominal sizes are not independent hyperacuity measurements.

Final interpretation:

The E-optotype crossover is real in the windowed pipeline but the mechanism is
not a temporal code and not a covariance-code migration. It is best read as
first-order spatial sampling in the time-averaged rate code, relative to
trial-mean stabilization. Dynamic FEM can help near the model's resolution
limit by sampling useful nearby retinal phases, but it does not beat a
deterministic fixed-position oracle and should not be framed as optimal active
trajectory selection.

## 2026-05-21: Early FEM / Temporal Decoding / COM Dynamics

Status: `Historical`, with some durable findings.

Primary docs and outputs:

- `archive/early_bigpicture/results_summary.md`
- `archive/temporal_decoding/temporal_decoding_analysis_plan_consolidated_v2.md`
- `archive/temporal_decoding/temporal_decoding_analysis_implementation_plan.md`
- `archive/temporal_decoding/temporal_decoding_diagnostic_plan.md`
- `com_dynamics.py`
- `transformation_dynamics.py`
- `displacement_decoding.py`
- `eoptotype_continuous_pass.py`
- `translation_covariance.py`
- `declan/displacement_decoding_figures/`
- `declan/transformation_dynamics_figures/`
- `declan/transformation_dynamics_figures/com/`

Motivation:

This was the broad exploration phase: try temporal decoding, velocity
readouts, displacement decoding, COM/spatial moments, transformation dynamics,
and translation covariance to see what FEM-driven population dynamics encode.

Durable outcomes:

- Temporal residual features did not improve over time-averaged rates in the
  orientation task.
- Velocity/transformation variables were not decodable from the tested latent
  or spatial-moment representations under independent-window processing.
- Within-image displacement decoding was near-perfect (`R2 ~0.998-0.999`).
- Cross-image displacement decoding failed badly (`R2 ~ -1.3`), which was
  initially a null for universal displacement decoding but later became a
  positive control for image-specific reafferent geometry.
- CoM/moment features did not beat scalar rates for small displacement
  decoding.

Later interpretation:

The early "temporal dynamics encode transformation" branch did not survive as
an active mechanism. But the displacement-decoding result became crucial: V1
encodes retinal displacement exquisitely within an image, and that code is
content-specific rather than universal. That fact directly feeds the later
TFTS/compact-geometry story.

## 2026-01 to 2026-04: Backimage, Translation Covariance, and Generated Diagnostics

Status: `Historical / artifact base`.

Primary artifacts:

- `translation_covariance/`
- `overnight_backimage_sweeps/`
- `overnight_backimage_long_sweeps_20s/`
- `overnight_backimage_long_sweeps_20s_re/`
- `test_sweeps/`
- `E_diagnostics_human_240ppd/`
- `E_diagnostics_model_37ppd_resnet_none_convgru/`
- `backimage_*`, `hybrid_eye_trace_*`, `fixrsvp_*`, and `spatial_info_*`
  caches.

Motivation:

These were the early data/caching/sweep artifacts that made later work
possible: backimage fixation pools, hybrid eye traces, natural-image sweeps,
E-optotype retinal/model diagnostics, and January translation-covariance
products.

Outcome:

They produced many useful caches and figures, but most are generated artifacts
rather than current analysis entry points.

Later interpretation:

Keep them as provenance and source material. Do not use them as current
manuscript claims without checking which later plan or README superseded their
interpretation.

## Open Claim Boundaries

Current safe claims:

- Recorded FEM-linked covariance is a major, low-dimensional reafferent
  component of V1 shared variability.
- The deterministic twin is useful for explaining the structure of that
  covariance, not for proving perception or optimality.
- Image translation tangents are image-specific but compact and
  image-generalizing as a family.
- Finite-difference fitted-twin translation sources, including compact
  restricted sources, predict a reliable component of recorded FEM covariance
  above strong nulls.
- Real retinal motion improves a V1-model natural-image spatial-information
  efficiency endpoint relative to stabilization.
- The next safe Figure 5 direction is a pose-conditioning story: retinal motion
  can create useful pose-aware signal and pose-blind nuisance covariance, but
  those must be separated explicitly. The cache-first recorded GLM ladder did
  not provide a positive recorded spike-prediction anchor for this claim.
- The corrected gain-orthogonal structured decoder did not pass its promotion
  gate. Treat decoder-level compact displacement readout as currently
  unsupported, while keeping the compact covariance/geometry claim intact.

Claims to avoid unless new evidence lands:

- Real FEM trajectories are optimal.
- A covariance-aware Fisher peak near empirical FEM scale proves optimization.
- V1 has a literal universal 2D eye-position coordinate map.
- FEM covariance fully explains all recorded shared variability.
- Absolute eye-position decoding by a permissive MLP proves compact
  translation geometry.
- A split-specific `candidate_positive` structured-decoder audit proves a
  compact displacement readout. The pooled six-session v2 result was null.
- Raw residual variance reduction proves forward-twin denoising; only
  controlled excess over shuffled-eye/gain/random-subspace nulls is meaningful.
- Forward-twin compact denoising is eye-trace-specific. The corrected matched
  preview beat random/unit-shuffled compact controls but did not beat
  compact-projected shuffled-eye controls.
- The E-optotype crossover is caused by a temporal code or covariance-code
  migration.
- The deterministic twin alone proves whether FEM covariance helps or hurts
  biological visual coding.
- The cache-first recorded pose-aware GLM null disproves the recorded FEM
  covariance result; it only bounds a simple content-blind prediction endpoint.
- A coarse time-by-eye GLM interaction with catastrophic held-out likelihood is
  evidence against retinal-translation coding; treat it as estimator pathology
  unless a geometry-constrained model reproduces it.

## Fast Resume Pointers

If resuming Figure 4 compact geometry:

- Read `compact_retinal_translation_geometry/README.md`.
- Then read `outputs/compact_retinal_translation_geometry/tables/acceptance_matrix.csv`.
- Then check whether `relative_displacement_decoding_prod_gpu1` has completed
  and whether its decoding results pass leakage and null checks.

If resuming Figure 5 active sensing:

- Read `active_sensing_movie_information/README.md`.
- Then read `active_sensing_movie_information/figure5_additional_checks_prep.md`.
- For the new non-circular direction, read
  `Non_circular_FEM_information_tests_prescription.md` before adding new model
  optimality claims.
- For the covariance-aware implemented path, read
  `Covariance_aware_FEM_optimality_analysis_prescription.md`, then
  `jake/twininfo/run_covariance_optimality.py`, then
  `active_sensing_movie_information/summarize_covariance_optimality.py`.
- Treat the natural-image-only population checks as current; treat cached
  e-optotype checks as historical scaffolding.
- Treat `recorded_pose_aware_prediction_multisession_6pilot` as a controlled
  null for the simple recorded GLM bridge. Do not promote it as a main positive
  panel.

If resuming structured decoding or denoising:

- Read `structured_translation_decoder_analysis.md` before interpreting
  `run_windowed_siamese_relative_decoding.py` outputs.
- For the latest structured decoder result, read
  `outputs/compact_retinal_translation_geometry/gain_orth_structured_prod_v2_gpu0/`
  and `outputs/compact_retinal_translation_geometry/gain_orth_structured_prod_v2_gpu1/`;
  pool across both before interpreting, because the GPU1 split alone was
  misleadingly positive.
- Treat `run_tejas_style_eyepos_decoder.py` as a sanity check unless a later
  note explicitly promotes it.
- Read `forward_twin_reafferent_denoising_analysis.md`; promote only controlled
  held-out denoising excess, not raw variance reduction.
- For the current forward-denoising outcome, read
  `outputs/forward_twin_reafferent_denoising_preview_patched_matched/` and the
  three diagnostic folders. Treat the result as useful but not promoted:
  compact denoising beats random/unit-shuffled geometry, not shuffled-eye.

If resuming recorded derivative alignment:

- Read `outputs/direct_recorded_derivative_twin_alignment_prod/README.md`.
- Keep Tier 1 as compact-basis enrichment only; do not headline signed axes.

If resuming the old E-optotype crossover:

- Read `archive/eoptotype/fem_eoptotype_hyperacuity_results.md` before
  `archive/early_bigpicture/revised_analysis_plan.md`.
- Assume the final mechanism is mean-rate spatial sampling unless you are
  explicitly testing a new control.
