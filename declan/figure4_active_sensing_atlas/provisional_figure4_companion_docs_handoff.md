# Handoff: Provisional Figure 4 and Companion Model Documents

Status: handoff for a coding/writing agent  
Scope: use existing results while canonical power reruns are still running  
Primary output area: `declan/figure4_active_sensing_atlas/`

## Objective

Build a provisional Figure 4 package from the existing audited results, and draft companion methods/logic documents for the three main active-sensing models:

1. Aggregate FEM information model
2. Local `I_z` pairing model
3. Joint posterior / trajectory-table observer model

The figure and companion documents should be usable for scientific review now, but must remain clearly marked provisional until the large canonical reruns finish and are promoted.

The companion documents should move toward the style of:

`declan/inhomogenous stimuli writeup.pdf`

That means they should be explanatory argument documents, not merely result ledgers. Each should describe motivation, assumptions, estimator logic, controls, diagnostics, known failure modes, and the current claim boundary.

## Start Here

Read these first:

- `AGENT_CONTEXT.md`
- `declan/figure4_active_sensing_atlas/README.md`
- `declan/figure4_active_sensing_atlas/main_figure_compression_v0.md`
- `declan/figure4_active_sensing_atlas/working_results_draft.md`
- `declan/figure4_active_sensing_atlas/panel_source_map.md`
- `declan/figure4_active_sensing_atlas/panel_manifest.csv`
- `declan/figure4_active_sensing_atlas/provenance_ledger.md`
- `declan/figure4_active_sensing_atlas/claim_critical_diagnostics_queue.md`
- `declan/FEM_active_sensing_methods_status_note_review.md`
- `declan/canonical_active_sensing/provenance/current_outputs.md`
- `declan/MANIFEST.md`
- `declan/ANALYSIS_NARRATIVE.md`

For the target explanatory style, inspect the PDF with:

```bash
pdftotext -layout declan/inhomogenous\ stimuli\ writeup.pdf - | less
```

The important stylistic pattern is:

- Begin from the scientific tension or broken simplifying assumption.
- Introduce notation before results.
- State assumptions explicitly.
- Define estimator contracts mathematically.
- Explain what each control rules out.
- Separate supported claims from attractive but not-yet-supported interpretations.
- End with production implications and remaining checks.

## Current Scientific Framing

The provisional Figure 4 should tell this story:

Measured fixational eye movements create structured retinal image motion. Across three related model tests, that motion can increase feature-decodable information, support image/trajectory inference when eye position is latent, and aligns modestly but reliably with local image geometry in behavior. The strongest behavioral bridge is the agreement between contour-aligned movements being useful in the model objective and measured eye movements showing contour-following geometry.

Do not overclaim that the animal is proven to optimize the model objective. There is no direct behavioral intervention test here. The correct claim is a convergence of model-predicted useful motion geometry and measured FEM geometry.

## Claim Boundaries

Keep these distinctions explicit:

- Aggregate FEM model: supports a readout-split ensemble/distributional claim:
  mean/delta-mean for absolute gain beyond static mean, temporal PCA/DCT for
  order-sensitive empirical-vs-control diagnostics.
- Local `I_z` pairing model: supports a mechanistic sensitivity analysis of actual paired traces versus matched controls, but is less stable than the aggregate readout.
- Joint posterior model: supports recovery of image information when eye trajectory is latent, especially through joint image/trajectory inference; axis-specific edge-parallel evidence remains weaker and objective-dependent.
- Behavior/geometry panels: support modest but reliable contour/edge alignment in measured drift/fixation clouds; this is the behavioral bridge, not a causal behavioral test.
- Raw-edge roadblock: model objectives have not yet cleanly beaten raw edge orientation as a behavioral predictor. Treat this as a known hard baseline, not a failure to hide.

## Current Feature-Decomposition Status

The current corrected candidate spec is a role split:

- Absolute aggregate candidates: `pyramid_local_field`, `k=16`, `mean` and `delta_mean`
- Local mechanistic sensitivity readout: `pyramid_local_field`, `k=16`, `delta_mean`
- Order-sensitive diagnostics: `pyramid_local_field`, `k=16`, temporal PCA/DCT variants

This is not yet a final canonical lock. Existing adjudication favors the
role-split framing, but the OU trace-control/readout audit is still pending.
Use corrected static-mean/all-readout outputs for the provisional figure, and
flag the final promotion/write-lock as pending.

## Provisional Main Figure Contract

Create or update a concrete panel contract in:

- `declan/figure4_active_sensing_atlas/provisional_panel_contract_v0.csv`
- `declan/figure4_active_sensing_atlas/provisional_figure4_v0.md`

Recommended main figure panels:

### 4A. Retinal Movie Premise

Purpose: show that drift creates real temporal retinal input structure, not an abstract trajectory variable.

Use existing Module A assets and numbers:

- Temporal contrast RMS: real `11.245`, stabilized `0.000`
- Motion power versus stabilized: real `1462.431`, stabilized `0.000`
- Movie power mean: real `15178.177`, stabilized `15185.182`
- Provenance: 256 images, 29 sessions, 151 drift-only trace sources, canonical 756-unit twin

Possible panels: A1/A2/A4 from the atlas. Put extra rendering/QC in supplement.

### 4B. Aggregate FEM Information

Purpose: show that biological-like image motion adds feature-decodable response structure.

Primary existing source:

`outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/incremental_static_plus_motion_relids/`

Existing provisional numbers to include:

- Gabor `k=4`, temporal PCA:
  - `0.25x`: `+14.31`, CI `[7.45, 21.79]`
  - `0.5x`: `+13.04`, CI `[6.81, 20.89]`
  - `1x`: `+9.10`, CI `[3.73, 14.86]`
  - `1.5x`: `+9.98`
  - `2x`: `+9.07`
- Pyramid `k=8`, temporal PCA:
  - `0.25x`: `+5.20`
  - `0.5x`: `+4.89`
  - `1x`: `+3.93`
  - `1.5x`: `+4.44`
  - `2x`: `+4.21`

Controls to show if space permits:

- Gabor `k=4`, empirical minus controls:
  - `0.25x`: OU `+21.24`, Brownian `+10.52`, rotated `+15.27`
  - `0.5x`: OU `+19.59`, Brownian `+7.89`, rotated `+11.21`
  - `1x`: OU `+17.16`, Brownian `+0.51`, rotated `+5.63`

Important caveat: the current canonical candidate is `pyramid_local_field k16 temporal_pca`, so these existing `k=4/k=8` results are provisional support, not the final production panel if the rerun replaces them.

### 4C. Joint Image/Trajectory Observer

Purpose: show that when eye position is latent, a joint observer recovers pose-lost information better than a zero-eye observer.

Primary source:

`outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/observer_summary.csv`

Include the matched-static rescue result:

- `matched_static_response`, `1.0x`:
  - known-eye: `1.000`
  - zero-eye: `0.328`
  - joint-eye empirical prior: `0.766`
  - joint-eye OU prior: `0.797`

This is one of the cleanest model-objective results and should be prominent.

### 4D. Image-Axis / Edge-Parallel Mechanism

Purpose: show why contour-aligned motion is a plausible useful axis in the model.

Existing useful facts:

- Pixel preservation edge-parallel advantage mean `300.54`, CI `[172.789, 408.961]`, positive sessions `26/29`
- Twin preservation edge-parallel advantage mean `0.000454497`, CI `[0.000371047, 0.000536519]`, positive sessions `29/29`

Keep the claim narrow: edge-parallel motion is a stable image/twin preservation axis. Axis preference in the full joint objective is weaker and objective-dependent.

Estimator caveat to keep visible: the "local image axis" is not necessarily a
single perceptual object. There may be behaviorally meaningful differences
between an average orientation-energy estimate over the whole local patch and a
prominent orientation feature selected by a more winner-take-all rule. The
row-17/18 rail example is the current reference case: raw BackImage rows 17/18
from `Allen_2022-02-16`, trial `184`, have stored aggregate
`image_edge_axis_deg = -31.4 deg`, while a visible bright-rail fit gives
`-37.6 deg`. The asset
`figures/panel_D/story_options/4D_row17_row18_visible_rail_fit_orientation.png`
and CSV
`figures/panel_D/story_options/4D_row17_row18_visible_rail_fit_orientation_values.csv`
should be used as the provenance example. This should be framed as an open
axis-estimator question, not as a correction to the quantitative D readout.

Future larger-cache 4D rerun option: include a second, pre-specified
winner-take-all local-orientation axis catalog alongside the current
average-orientation-energy catalog. This is appropriate if it is defined from
the image alone before looking at decoding or behavior. The WTA catalog should
select the strongest/salient local orientation mode in the patch, not the axis
closest to the measured drift or the axis that improves the decoder. Run both
axis catalogs on the same matched-static response tables, windows, candidate
sets, scales, features, seeds, and bootstrap/permutation summaries. Report:

- average-energy along/across decoding contrast;
- WTA-orientation along/across decoding contrast;
- paired difference between the two axis estimators;
- distribution of average-vs-WTA axis angular disagreement;
- behavior alignment against both axis estimators.

If WTA strengthens both the hidden-eye decoding contrast and the behavior
alignment, it would support the idea that active sensing may use a prominent
contour feature rather than a patch-average orientation-energy estimate. If it
only improves one side, keep it as an estimator sensitivity analysis rather
than a promoted mechanism.

Quick behavior-side screen now exists:
`declan/figure4_active_sensing_atlas/scripts/run_panel_d_wta_behavior_diagnostic.py`.
This does not rerun model-response decoding. It reconstructs local BackImage
patches, computes an image-only WTA orientation mode, and compares recorded
drift alignment to the stored average-orientation axis versus the WTA axis.
Single-session runs are quick enough for exploratory use. Example:
`outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_wta_orientation_behavior_diagnostic_allen_2022_02_16/`.
For `Allen_2022-02-16`, the reliable subset gave a small WTA advantage
(`+0.0251` cos-2-delta alignment over average axis), with a larger descriptive
advantage on windows where the two estimators disagreed by at least `10 deg`
(`+0.2149`). Because this is one session, it should be treated as a useful
screening result, not an across-session claim. A sampled all-session attempt
was stopped because image reconstruction across many trials/sessions was not a
quick interaction-scale command; budget a longer run for a session-bootstrap
answer.

### 4E. Behavior Contour-Following Bridge

Purpose: show measured FEM geometry aligns with local image geometry.

Use more than the single behavior headline panel. The existing atlas already added extra behavior provenance panels because the earlier E3-only version was hard to parse.

Include:

- E2: behavior alignment strength
- E3: endpoint-zone enrichment
- E6: full drift-edge distribution/session diagnostic
- E7: confidence/signed-delta diagnostic
- E8: endpoint/null diagnostic

Headline result:

- all-window weighted `cos2 = 0.181`, CI `[0.124, 0.241]`
- reliable weighted `cos2 = 0.201`

This panel should make clear that the behavioral signal is modest but reliable and that null/session diagnostics agree with the headline.

### Optional 4F. Claim Summary / Boundary Panel

If the figure needs a sixth slot, use it to summarize the three model results and the behavioral bridge:

- aggregate FEM: information gain
- local pairing: mechanistic sensitivity
- joint posterior: latent trajectory inference
- behavior: contour-following geometry
- caveat: raw edge remains a hard baseline; canonical reruns pending

This can also be a compact supplement-facing panel instead of a main panel.

## Companion Documents To Draft

Create these files:

- `declan/figure4_active_sensing_atlas/4a_companion_retinal_movie_premise.md`
- `declan/figure4_active_sensing_atlas/4b_companion_aggregate_fem_model.md`
- `declan/figure4_active_sensing_atlas/4b_companion_local_Iz_pairing_model.md`
- `declan/figure4_active_sensing_atlas/4c_companion_joint_posterior_observer_model.md`
- `declan/figure4_active_sensing_atlas/4d_companion_along_edge_model_feature_encoding.md`
- `declan/figure4_active_sensing_atlas/4e_companion_behavior_geometry_bridge.md`

Each companion should be self-contained and written as a reasoning document. Do
not make them into flat inventories of runs. Each current companion now also
has a plain-English methods section that describes the implementation without
requiring the reader to jump directly into the code.

## Shared Notation

Use the same notation across all companion documents:

- `I`: image or image window
- `tau`: eye trajectory
- `y = f_theta(I, tau)`: response movie from the V1 twin/model
- `phi(I)`: image feature target
- `D(y, phi)`: feature decoding or feature-alignment score

Where helpful, define the response summary `s(y)` separately, because the current two-readout story depends on response summaries:

- `mean`: absolute aggregate readout candidate
- `delta_mean`: static-subtracted absolute/local mechanistic readout candidate
- `temporal_pca` / `temporal_dct`: order-sensitive diagnostic candidates

## Companion Structure Template

Use this structure for each companion document:

1. Title, date, status
2. Summary
3. Motivation
4. Notation and estimator contract
5. Assumptions
6. Controls
7. Existing evidence
8. Diagnostics and failure modes
9. Current claim boundary
10. Production rerun implications

The “Assumptions” section should use explicit labels like A1, A2, A3. The “Estimator contract” section should include simple equations, even if approximate.

## Companion 1: Aggregate FEM Information Model

Main question:

Do biological-like FEM trajectories increase image-feature-decodable structure in the model response, relative to static or motion-control baselines?

Estimator contract:

```text
E_{I, tau ~ family} D(f_theta(I, tau), phi(I))
```

Compare empirical FEM-like trajectories to static, OU, Brownian, rotated, and scale controls.

Key assumptions:

- A1: The V1 twin response is a useful proxy for early visual population response.
- A2: The chosen feature decomposition `phi(I)` captures relevant local image structure.
- A3: Image-grouped decoding avoids trivial image memorization.
- A4: Motion-family controls isolate biological-like trajectory structure rather than generic motion energy.
- A5: The response summary is part of the model hypothesis; `temporal_pca` and `delta_mean` are not interchangeable.

Known risks:

- The existing provisional figure evidence is not yet from the final `pyramid_local_field k16 temporal_pca` power rerun.
- Brownian/rotated specificity narrows at larger scales.
- Strong aggregate performance does not by itself prove behavior.

Production implication:

Use the pending large aggregate rerun to replace provisional `k=4/k=8` atlas panels if it confirms the `pyramid_local_field k16 temporal_pca` target.

## Companion 2: Local `I_z` Pairing Model

Main question:

Does the actual image/trace pairing produce a stronger local feature-response change than matched unpaired or motion-control traces?

Estimator contract:

```text
D(f_theta(I, tau_actual), phi(I))
  - E_{tau ~ matched controls} D(f_theta(I, tau), phi(I))
```

Key controls:

- matched unpaired traces
- rotated traces
- OU controls
- Brownian controls
- scale sweep
- seed and K sensitivity

Key assumptions:

- A1: Actual trace/image pairing is meaningful at the local window scale.
- A2: Matched-unpaired traces remove session/motion statistics while breaking image-specific pairing.
- A3: `delta_mean` is the more biologically local/mechanistic readout.
- A4: A stable claim requires resistance to sign reversal and sentinel accounting.

Current interpretation:

`pyramid_local_field k16 delta_mean` is the local mechanistic sensitivity candidate. It better captures paired-trace feature-response changes, but it is less stable than the aggregate `temporal_pca` readout under matched-unpaired and sign-reversal accounting.

Claim boundary:

This should not be the sole canonical headline until seed/power reruns confirm stability. It is a mechanistic sensitivity panel or companion result, not the current aggregate winner.

## Companion 3: Joint Posterior / Trajectory-Table Observer

Main question:

When the observer does not know the eye trajectory, can it recover image information by marginalizing over possible trajectories?

Estimator contract:

```text
p(I | y) proportional to sum_tau p(y | I, tau) p(tau)
```

Compare:

- known-eye observer
- zero-eye observer
- joint-eye observer with empirical prior
- joint-eye observer with OU prior
- matched-static distractor control

Key assumptions:

- A1: The candidate image and trajectory tables approximate the relevant posterior support.
- A2: The response likelihood is calibrated well enough for relative comparisons.
- A3: Matched-static controls rule out trivial static image separability.
- A4: Prior choice matters and must be reported.

Existing evidence:

The matched-static observer result is strong and clean:

- known-eye `1.000`
- zero-eye `0.328`
- joint-eye empirical prior `0.766`
- joint-eye OU prior `0.797`

Claim boundary:

This supports joint image/eye inference as a useful computation. It does not, by itself, prove that the animal performs this exact posterior computation. The axis-specific edge-parallel posterior term is weaker than the generic joint-minus-zero recovery.

## Optional Companion 4: Behavior Geometry Bridge

Main question:

Do measured FEMs follow local image geometry in the direction predicted to preserve or improve visual information?

Core logic:

Model diagnostics say contour/edge-aligned motion can preserve image/twin responses and is plausibly useful. Behavior diagnostics show measured FEMs align modestly but reliably with local image geometry.

Use the expanded behavior panel set:

- E2
- E3
- E6
- E7
- E8

Key caveat:

This is the closest available behavioral evidence, but it is not a direct behavioral test of the model objective. Raw edge remains the hard baseline.

## Do Not Disturb Running Jobs

Large reruns may be active. Do not move, delete, or rewrite any files in active output directories or `background_logs`.

Expected active/pending jobs at the time this handoff was written:

- aggregate power rerun
- local pairing seed7
- queued local pairing seed11

Use existing completed outputs for the provisional figure. Treat active reruns as pending updates only.

## Implementation Steps

1. Check repo state.

```bash
git status --short
```

2. Confirm active jobs without interrupting them.

```bash
ps -o pid,etime,cmd -p 2556842,2559487,2557972
```

3. Read the atlas and provenance files listed above.

4. Draft `provisional_panel_contract_v0.csv`.

Minimum columns:

```text
panel_id,title,claim,source_output,source_asset_or_table,status,provisional_reason,next_update_trigger
```

5. Draft `provisional_figure4_v0.md`.

Include:

- panel-by-panel story
- source paths
- exact numbers used
- caveat labels
- replacement plan after canonical reruns finish

6. Build or assemble a provisional composite only from existing assets.

Prefer existing atlas scripts if available. First inspect usage:

```bash
rg "build.*composite|panel_manifest|figure4" declan/figure4_active_sensing_atlas scripts
```

If a script exists, run its help before changing anything:

```bash
python path/to/script.py --help
```

If no reliable script exists, create a small audited builder under:

`declan/figure4_active_sensing_atlas/scripts/`

The builder should read the panel contract or manifest and emit:

- `declan/figure4_active_sensing_atlas/figures/provisional_figure4_v0.png`
- optionally `declan/figure4_active_sensing_atlas/figures/provisional_figure4_v0.pdf`

7. Draft the three companion documents.

Keep them concise but substantive. They should read like miniature methods essays with assumptions and estimator logic, not like a catalog of outputs.

8. Update provenance docs.

At minimum update:

- `declan/figure4_active_sensing_atlas/provenance_ledger.md`
- `declan/figure4_active_sensing_atlas/README.md`

Only update `declan/ANALYSIS_NARRATIVE.md` and `declan/MANIFEST.md` if the new provisional figure or companion docs are ready enough to become canonical navigation entries.

9. Run lightweight checks.

```bash
git diff --check
rg "TODO|FIXME|PLACEHOLDER" declan/figure4_active_sensing_atlas
```

If scripts were edited:

```bash
python -m py_compile path/to/edited_script.py
```

10. Summarize what changed and what remains blocked by reruns.

## Definition of Done

This handoff is complete when:

- A provisional Figure 4 panel contract exists.
- A provisional Figure 4 markdown narrative exists.
- A composite figure exists, or there is a clear documented reason why it could not yet be built.
- The three model companion docs exist and share notation/claim boundaries.
- Behavior provenance panels E2/E3/E6/E7/E8 are incorporated or explicitly queued.
- The provenance ledger lists every source output/table/asset used.
- The figure is clearly marked provisional.
- The pending reruns are not disturbed.

## Update Plan After Reruns Finish

When the large reruns complete:

1. Replace aggregate panel B with the production `pyramid_local_field k16 temporal_pca` output if confirmed.
2. Update local pairing companion and any local panel with seed7/seed11/power rerun results.
3. Re-run the feature-decomposition adjudication if new caches materially change the ranking.
4. Promote or revise the two-readout candidate spec.
5. Remove or soften provisional language only after source outputs, manifests, and diagnostics agree.
