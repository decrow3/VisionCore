# Claim-Critical Failure-Mode Diagnostics Queue

Status: consolidated queue, 2026-06-21.

Purpose: keep anticipated failure modes visible for the analyses that can
change the main Figure 4 claims or the decision to launch long canonical runs.
This is not a list of every possible supplement. It is the set of diagnostics
that should be checked before a claim is locked, or that should trigger a
narrow methods-style deep dive if the result fails in an informative way.

Reference style: the `inhomogenous stimuli writeup.pdf` is the model for a
deep dive. It starts from hidden assumptions, shows how the data violate them,
derives or implements a correction/diagnostic, validates on controlled and real
data, then gates the claim. The diagnostics below are organized the same way:

```text
claim -> hidden assumption -> anticipated failure mode -> diagnostic -> action
```

## Short Answer

We have partially queued these diagnostics, but until now they were scattered.

Already explicit:

- raw-edge roadblock residual adjudication;
- feature/readout adjudication across `k`, feature family, and readout;
- aggregate motion-family QC and absolute-gain guardrails;
- corrected static-mean baseline plus all-readout/nested-alpha audit;
- OU trace-control audit before using OU as a headline negative control;
- joint observer matched-static, scale, and posterior guardrails;
- axis-conditioned hard-negative and edge-parallel preservation audits;
- behavior metric-convention and endpoint-null diagnostics.

Previously under-consolidated:

- model-objective failure modes as a first-class deep-dive trigger;
- global/screen-axis artifact checks for apparent objective wins;
- same-window residual explanation beyond raw edge geometry;
- population/readout dependence of objective-axis conclusions;
- a single promotion/demotion map for claim-critical diagnostics.

## Triage Levels

### Level 1: Claim-Locking Gates

These must be reviewed before the corresponding claim is promoted into the
main text or before a long canonical run is treated as final.

### Level 2: Deep-Dive Triggers

These do not necessarily block the current paper, but if they fail or produce a
surprising sign, they should become a narrow methods note rather than another
ad hoc result panel.

### Level 3: Supplement/Interpretation Guardrails

These should remain visible in captions, supplement, or atlas prose, but they
do not by themselves block the core claim.

## Module A: Retinal-Movie Premise

Core claim:

```text
During fixation, a fixed screen image becomes a retinal movie with measurable
temporal contrast/motion structure.
```

Claim-locking diagnostics:

| failure mode | diagnostic already queued | current artifact | action |
| --- | --- | --- | --- |
| FEM and stabilized movies differ because of rendering/crop artifacts rather than retinal motion | stabilized-vs-FEM movie QC | `panel_A/A2_movie_transform_qc.*` | Gate Panel A claim on QC, not on downstream decoding. |
| FEM movie variance is a denominator or covariance-accounting artifact | covariance bridge guardrail | `panel_A/A5_covariance_bridge_guardrail.*` | Keep as bridge/supplement unless denominator convention is fully harmonized. |
| crop/border/image-window selection drives downstream effects | image/window manifest provenance | canonical geometry figure-pack input checks | Treat any border-distance imbalance as a nuisance covariate in model-objective and behavior residual analyses. |

Deep-dive trigger:

- If retinal movie QC fails by image class, crop position, or border distance,
  write a narrow rendering/windowing note before interpreting downstream model
  failures.

## Module B: Aggregate FEM Feature Utility

Core claim:

```text
Empirical drift-like response movies add feature-decodable structure beyond
static responses in the V1 twin.
```

Claim-locking diagnostics:

| failure mode | diagnostic already queued | current artifact | action |
| --- | --- | --- | --- |
| "more motion is better" rather than FEM-like motion | Brownian/rotated controls, absolute-gain guardrail, and OU audit | `panel_B/B2`, `B4`, `B5`; `declan/ou_trace_control_and_readout_audit_handoff.md` | Claim empirical specificity only at scales/readouts where controls remain separated and the control family is validated. |
| feature/readout choice is post hoc | `k=2/4/8/16/32`, Gabor-vs-pyramid, corrected static-mean v6 adjudication, and all-readout/nested-alpha audit | canonical active-sensing provenance; `backimage_feature_decomposition_adjudication_v6_staticmean_corrected_power_rerun_primary_scales`; `incremental_staticmean_plus_motion_allreadouts_v1/readout_atlas_figures/` | Current role split is supported but not write-locked: `mean`/`delta_mean` for absolute gain, `delta_mean` for local pairing, temporal PCA/DCT for order-sensitive diagnostics. |
| clipping/RMS/path-length mismatch explains gains | motion-family QC | `aggregate_motion_summary.csv`, `B2_motion_family_qc.*` | Fail closed to "motion can help" if motion family matching is broken. |
| OU is pathological rather than a valid matched null | exact trace replay, RMS/path/autocorr/PSD/centering audit, response-space geometry, nested-alpha decoder behavior | pending `backimage_ou_trace_control_audit_n384_power_v1/` from `declan/ou_trace_control_and_readout_audit_handoff.md` | Do not use OU as the headline negative control until classified as valid primary, diagnostic-only, invalid, or inconclusive. |
| train/test leakage or image identity leakage | grouped image CV / cache contracts | aggregate posthoc outputs and canonical wrappers | Canonical figure pack must report split convention. |
| single seed/sample cohort drives result | aggregate n384 power rerun plus local seed7/seed11 power reruns and posthocs | `canonical_active_sensing` power config and v6 adjudication | Aggregate still has one power seed; local sensitivity now has two seeds. Treat effect sizes as production candidates, with aggregate seed11 optional if reviewers demand seed replication. |

Deep-dive trigger:

- If Brownian/rotated controls erase empirical specificity at the chosen
  canonical feature/readout, write a focused "what motion property matters?"
  note before framing this as FEM-specific active sensing.

## Module C: Joint Image-And-Eye Observer

Core claim:

```text
When retinal motion matters, a joint image-and-eye observer recovers
information lost by a zero-eye observer.
```

Claim-locking diagnostics:

| failure mode | diagnostic already queued | current artifact | action |
| --- | --- | --- | --- |
| joint observer wins by static response strength, not motion-aware inference | matched-static distractors | `panel_C/C3_matched_static_rescue.*` | Required for main observer claim. |
| scale rescue is just zero-eye failure | scale-gap guardrail | `panel_C/C5_scale_gap_guardrail.*` | State scale/candidate dependence. |
| posterior is not meaningful or collapses to a trivial candidate | posterior concentration / Neff | `panel_C/C4_posterior_concentration.*` | Use as interpretive guardrail, not exact trajectory identification. |
| candidate/prior/hardness imbalance drives axis effects | hard-negative candidate controls | Panel D/C provenance and raw-edge audit inputs | Axis conclusions require shared-source or hard-negative runs. |
| compact-subspace necessity is inferred from the wrong endpoint | feature-space compact-only / compact-removed / addback decomposition | pending; current compact-removal audit is joint image-decoding accuracy | Do not use compact removal to explain feature-recovery cosine until the same decomposition is scored in feature space. |
| model-selection remains incomplete | joint `rel_0p25x` completion plus corrected v6/all-readout adjudication | canonical active-sensing provenance | Closed for the current primary-scale role split; keep `canonical_run_allowed=false` until OU/readout audit and an explicit write-lock/promotion pass are requested. |

Deep-dive trigger:

- If joint-minus-zero recovery is strong but axis-specific behavior alignment is
  weak, separate "motion-aware inference" from "behavioral axis prediction" in
  the main story rather than forcing one objective to do both jobs.

## Module D: Axis Geometry And Preservation

Core claim:

```text
Local image geometry defines meaningful motion axes; edge-parallel motion
preserves local image and V1-twin responses better than edge-orthogonal motion.
```

Claim-locking diagnostics:

| failure mode | diagnostic already queued | current artifact | action |
| --- | --- | --- | --- |
| axis preference flips with candidate set or scale | axis-conditioned hard-negative sweep | `panel_D/D3_axis_preference_guardrail.*` | Do not claim one universal useful axis. |
| preservation is true locally but not a policy objective | edge-parallel stability panel and scope text | `panel_D/D4_edge_parallel_stability.*` | Claim preservation, not full optimality. |
| model objective does not beat raw edge geometry | objective-alignment guardrail | `panel_D/D5_objective_alignment_guardrail.*` | Treat as unresolved mechanism. |
| raw edge confidence absorbs model variables | raw-edge residual adjudication | `canonical_geometry/run_raw_edge_audit.py` | Promote model bridge only if incremental residual explanation survives. |
| patch-average orientation misses a behaviorally salient contour | paired average-orientation vs winner-take-all local-orientation axis catalogs | behavior screen script `scripts/run_panel_d_wta_behavior_diagnostic.py`; one-session output `backimage_wta_orientation_behavior_diagnostic_allen_2022_02_16/`; future 4D larger-cache decoding rerun still pending | Define WTA from image-only features before scoring; treat as estimator sensitivity unless decoding and behavior both improve. |

Deep-dive trigger:

- If a V1-twin objective prefers edge-orthogonal movement while behavior is
  edge-parallel, write a narrow objective-landscape note: response modulation,
  preservation, and inference may be different objectives.

## Module E: Behavioral Contour Following

Core claim:

```text
Measured free-viewing FEM/fixation-cloud axes align modestly but reliably with
local image geometry.
```

Claim-locking diagnostics:

| failure mode | diagnostic already queued | current artifact | action |
| --- | --- | --- | --- |
| metric convention changes the effect size | weighted vs unweighted guardrail | `panel_E/E4_metric_convention_guardrail.*` | State convention in caption/prose. |
| endpoint-heavy `cos(2 delta)` histogram is misread | endpoint/null diagnostic | `panel_E/E8_endpoint_null_diagnostic.*` | Pair E8 with E3 whenever explaining endpoint enrichment. |
| effect is only visible after confidence filtering | full distribution/session and confidence diagnostics | `panel_E/E6`, `E7` | Report all/reliable/high-confidence subsets together. |
| within-session/global axis bias explains behavior | image-edge shuffle, session bootstrap, nuisance controls | existing `orientation_alignment_summary.csv`; residual audit queue | Treat global screen-axis predictors as nuisance. |
| current model objectives do not explain behavior beyond raw edge | raw-edge residual adjudication | `raw_edge_roadblock_handoff.md`; canonical geometry wrapper | Keep behavior claim separate from objective mechanism unless residual gate passes. |

Deep-dive trigger:

- If behavior remains positive but residual model variables fail, the main
  claim should be "animals follow local image geometry" plus "V1 twin explains
  possible utility/preservation," not "animals optimize the tested objective."

## Model-Objective Deep-Dive Queue

This is the branch most likely to deserve a dedicated note like the
inhomogeneous-stimuli writeup.

Question:

```text
Why do current model-objective axes fail to beat raw edge geometry, and is that
a real scientific negative or an estimator/objective mismatch?
```

Hidden assumptions to test:

1. Objective axes, raw edge axes, and behavioral drift axes are measured on the
   same windows with comparable uncertainty.
2. A predicted model axis is a local image-dependent variable, not a global
   screen-axis or trajectory-grid artifact.
3. The objective being optimized is the right biological quantity: response
   modulation, feature recovery, preservation, posterior concentration, and
   pose cost may prefer different axes.
4. The result is stable across population size/readout: sampled 64/256-unit
   diagnostics should not be mixed with canonical 756-unit claims.
5. Candidate hardness, border distance, source overlap, and image anisotropy do
   not explain apparent model wins.

Queued diagnostics:

| priority | diagnostic | concrete output | promotion/demotion action |
| --- | --- | --- | --- |
| L1 | same-window objective-vs-raw master table | one row per image/window with raw edge, behavior, objective axes, confidence, border distance, candidate hardness, source flags | Required before any objective bridge claim. |
| L1 | within-session residual regression | `Delta R2`, session-bootstrap CI, sign count after raw edge confidence and drift anisotropy | Promote only if model variables add residual explanation. |
| L1 | global-axis nuisance audit | predicted-axis histograms, screen-axis regressors, all-zero/all-90-degree detector | Any all-windows/global-axis predictor is nuisance, not mechanism. |
| L1 | shared-source overlap and candidate-hardness audit | source-overlap table and hardness-stratified effect table | Fail closed if objective advantage follows candidate/source imbalance. |
| L2 | objective-landscape maps around local edge axis | per-window objective score as a function of axis angle for representative edge/texture/isotropic patches | Deep-dive if objective geometry explains sign reversals. |
| L2 | synthetic image sanity suite | ideal edge, curved contour, texture, isotropic noise, border patch | Deep-dive if objective fails the toy cases expected from its interpretation. |
| L2 | population/readout sensitivity | sampled 64/256 vs canonical 756; pyramid/Gabor; `temporal_pca`/`delta_mean`; `k=16` centered | Demote sampled-objective conclusions if canonical population flips. |
| L2 | preservation-vs-modulation decomposition | compare pixel/V1 preservation, response-change magnitude, feature recovery, posterior concentration on the same windows | Split objectives if each explains different signs. |
| L3 | representative failure panels | selected windows where raw edge predicts behavior and model objective fails, plus the reverse | Use in supplement or diagnostic note, not main claim by itself. |

Recommended note outline if triggered:

```text
1. Define the axis-prediction problem on a shared window table.
2. State the competing objectives and their implied predicted axes.
3. Show the raw-edge baseline and behavior metric.
4. Show objective-axis distributions and global-axis artifacts.
5. Decompose residual behavior after raw edge confidence.
6. Validate objective signs on synthetic edge/texture/isotropic images.
7. Decide whether the objective is wrong, incomplete, or simply not behavioral.
```

Current expected outcome:

```text
Raw edge geometry remains the behavior baseline to beat. If model-derived
variables add residual explanation, promote them as a mechanistic bridge. If
not, keep the main paper honest: behavior follows image geometry, and the V1
twin shows utility/preservation/inference consequences without proving the
animal optimizes the tested objective.
```

## Canonical Run Preflight Diagnostics

Before launching or interpreting long canonical runs:

- `validate_configs` must pass for `canonical_active_sensing` and
  `canonical_geometry`.
- `--print-command` should be reviewed for every long job.
- wrappers must refuse existing non-empty output dirs unless the refresh is
  intentional and documented.
- output provenance must be updated before figure-pack promotion.
- feature target is supported by joint `rel_0p25x` completion and corrected
  static-mean/all-readout adjudication, but remains provisional until an
  explicit write-lock/promotion pass, all-readout Panel B review, and OU
  trace-control verdict.
- the current role split should be framed as:

```text
absolute aggregate candidates: pyramid_local_field k16 mean, delta_mean
local mechanistic sensitivity: pyramid_local_field k16 delta_mean
order-sensitive diagnostics: pyramid_local_field k16 temporal_pca / temporal_dct variants
```

## Promotion Rules

Promote to main claim only if:

- the diagnostic tests the failure mode that would otherwise explain the
  result;
- the effect survives the relevant session/bootstrap or image-disjoint split;
- nuisance variables and global-axis artifacts are explicitly checked;
- the result has a single-sentence claim boundary that does not borrow support
  from a different model.

Demote or reroute if:

- the effect is absorbed by raw edge geometry, candidate hardness, source
  overlap, or a global-axis nuisance;
- the sign depends on scale/candidate set in a way that changes the biological
  interpretation;
- the analysis proves a different claim than the panel prose wants to make.
