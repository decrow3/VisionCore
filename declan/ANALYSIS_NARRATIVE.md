# declan Analysis Narrative

Last curated: 2026-06-30.

This is the brief synthesis companion to `MANIFEST.md`. The manifest answers
"where is it?" This file answers "what did we learn, what should we not claim,
and where should someone resume without retreading old ground?"

The detailed pre-compression narrative was archived at:

```text
declan/ANALYSIS_NARRATIVE_DETAILED_2026-06-22.md
```

Use that archive when exact output paths, historical values, or implementation
lineage are needed. Use this file for current interpretation.

## Reading Rules

- `Promoted`: can plausibly carry a figure or manuscript claim with stated
  guardrails.
- `Supportive`: useful evidence, but not a standalone headline.
- `Diagnostic`: technically useful and interpretation-shaping, but not a
  promoted claim.
- `Historical`: superseded or absorbed into later analyses.
- `Open`: planned, incomplete, or not yet interpreted.

## Current Synthesis

The project has converged on one structural story and one unified Figure 4
active-sensing atlas, rather than a separate later active-sensing figure.

1. Translation/Jacobian work rescued the structural idea. Retinal translations
   produce image-specific response tangents; those tangents are not universal
   signed x/y axes, but they occupy compact, image-generalizing response
   geometry and predict a reliable component of recorded FEM covariance.
2. Current Figure 3/4 structural claims should be about compact,
   content-routed retinal-translation geometry, covariance closure, and
   trajectory-aware feature recovery. They should not claim a unique compact
   mechanism over static response PCs, a universal eye-position map, or
   behavioral optimality.
3. Current Figure 4 active-sensing panels are claim-specific:
   - 4B: motion-rendered responses add feature information over the stabilized
     baseline. The trajectory renders the response movie but is not an explicit
     aggregate ridge-decoder input; the strict source-trial grouped cache is now
     the primary estimate, with the image-group cache retained as the optimistic
     provenance comparison. The same-axis pose-unaware hidden-sample proxy has
     negative point estimates relative to static, and OU remains audit-pending
     rather than a headline null.
   - 4C: compact subspace supports joint eye/image decoding, in the guarded
     sense that compact-only retains much of full joint feature recovery and
     compact removal collapses toward zero-eye. Static PCs remain a serious
     specificity guardrail. The joint decoder now has a completed
     inherited-decoder audit gate: promoted calibration uses source-row-heldout
     CV when available, supervised 4C feature decoders are source-row disjoint,
     and cache interval integrity passes for the validated contrast files.
   - 4D: along-edge motion benefits model feature encoding in the scoped
     matched-static feature-posterior observer. Hard-negative controls prevent a
     universal along-edge policy claim.
   - 4E: real drift follows clear edges. This is the behavioral bridge, not a
     proof that animals optimize the tested model objective.
4. The older active-sensing movie-information plan is now folded into Figure 4.
   Its surviving role is to support the premise that retinal motion creates
   useful movie structure, to separate pose-aware from pose-blind accounting,
   and to keep random, phase-cloud, scale, covariance, and whitening controls
   visible. It does not establish exact real-trajectory optimality.
5. Early E-optotype work is now a historical guardrail. It found a real
   FEM-related crossover, but later controls narrowed the mechanism to
   first-order spatial sampling in the mean-rate code near the model's
   resolution limit, not a temporal code or covariance-code migration story.

## Recent Additions Since 2026-06-22

- 4C now has a promoted continuous no-anchor feature-recovery observer, plus
  candidate-free continuous feature-embedding diagnostics. The nonlinear MLP
  and residual-tau branches show that compact moving responses contain
  recoverable feature information above zero-eye and stabilized baselines, but
  they are upper-bound diagnostics rather than downstream linear-readout
  claims.
- Static-response-PC controls have hardened the non-uniqueness caveat. The 4C
  static-PC tests and one-session covariance-closure static-PC predictor run
  show compact and static-PC bases are close contenders. The safe wording is
  that eye movements push responses through a compact part of the ordinary
  image-response manifold, not a unique translation-only channel.
- 4D is more bounded. The newer known-axis posterior diagnostic favors
  across-contour motion, the model-family check gives weak/mixed 1x along-edge
  support with wide intervals, and the linear-Gaussian check is positive only
  for some cosine readouts. Keep 4D as a scoped latent-eye prior result, not an
  intrinsic known-eye along-edge optimality claim.
- BackImage feature-family posthocs expand 4B support to Gabor/pyramid energy
  and orientation-covariance targets, but several are point-estimate-only or
  registered-target pilots. Treat them as supportive target-space/provenance
  evidence, not new promoted claims.
- A redundancy-resolved V1-twin population substrate now exists, with portable
  loaders and validation scripts. It is useful for faster or cleaner reduced
  population reruns, but current QC still shows some weak merges and
  reconstruction loss; it is infrastructure, not a new scientific conclusion.
- The Vernier walkthrough script and executed PDF are reproducibility aids. They
  make the old Vernier analysis legible but do not reopen the E-optotype or
  covariance-code interpretation.

## Current Main Claims

### Compact Translation Geometry

Status: `Promoted with specificity caveats`.

Small retinal translations move V1-twin responses through a compact,
image-conditioned response geometry. Local translation tangents are
image-specific, but the family of tangents is compact and generalizes across
images. Finite-difference fitted-twin translation sources, including
compact-restricted sources, predict a reliable component of recorded
FEM-linked covariance above strong nulls.

Do not claim:

- a universal literal 2D eye-position map;
- compact uniqueness over static response PCs;
- behavioral optimality from the geometry alone;
- full explanation of all recorded shared variability.

Resume from:

- `declan/compact_retinal_translation_geometry/static_pc_control_adjudication_note.md`
- `outputs/compact_retinal_translation_geometry/tables/acceptance_matrix.csv`
- `declan/fig3/generate_figure3_combined.py`

### Figure 4 Active-Sensing Atlas

Status: `Provisional figure package`.

The current Figure 4 package is the home for the active-sensing movie,
feature-encoding, compact-observer, axis-geometry, and behavior ideas. It is
organized around five panel claims:

```text
4A: One image becomes a retinal movie.
4B: Motion enhances feature encoding.
4C: Compact subspace supports joint eye/image decoding.
4D: Along-edge motion benefits model feature encoding.
4E: Real drift follows clear edges.
```

The panel companion docs are now the source of truth for claim-specific
evidence and caveats:

```text
declan/figure4_active_sensing_atlas/4b_companion_aggregate_fem_model.md
declan/figure4_active_sensing_atlas/4b_companion_local_Iz_pairing_model.md
declan/figure4_active_sensing_atlas/4c_companion_joint_posterior_observer_model.md
declan/figure4_active_sensing_atlas/4d_companion_along_edge_model_feature_encoding.md
declan/figure4_active_sensing_atlas/4e_companion_behavior_geometry_bridge.md
```

Do not retread the older panel ordering. The current mapping is:

- 4B is the aggregate motion-rendered feature-information result, with local
  pairing as a mechanistic companion.
- 4C is the continuous no-anchor feature-recovery observer plus compact
  feature-posterior intervention, not only the older matched-static
  image-identity observer.
- 4D is the along/across model feature-posterior contrast, not merely the older
  edge-parallel preservation audit.
- 4E is behavior following raw/local edge geometry, not model-objective
  optimization.

Recent results that should be represented in the companion docs:

- 4B: the corrected n384 source-trial grouped information-axis posthoc supports
  empirical motion gains for the static-subtracted local bridge, while the
  same-axis pose-unaware hidden-sample proxy has negative point estimates.
  Mean/readout absolute-gain summaries, `delta_mean`
  local-pairing sensitivity, and temporal PCA/DCT diagnostics should not be
  collapsed into one generic "motion helps" result.
- 4B local: pairing and power-seed follow-ups make the strongest mechanistic
  point through local `I_z` sensitivity and `delta_mean`, not through every
  feature/readout combination.
- 4C: the current endpoint combines the continuous no-anchor feature-recovery
  observer with the feature-posterior compact-only / compact-removed /
  compact-addback decomposition. Older matched-static image-identity rescue is
  supporting history; newer continuous feature-embedding and MLP branches are
  diagnostic/upper-bound support unless explicitly promoted in the companion.
  The inherited-decoder audit now passes with no failures: the promoted
  calibration gate is source-row-heldout feature cosine, while trial-id/table
  splits are retained only as provenance context.
- 4D: matched-static axis-conditioned feature posterior supports an along-edge
  gain at the selected scale/readout; known-axis, linear-Gaussian, model-family,
  hard-negative, and edge-preservation analyses define the boundary.
- 4E: raw/local edge geometry is the behavioral baseline to beat. Model
  objectives should be framed as possible mechanisms only if raw-edge residual
  gates pass.
- Canonical active-sensing and canonical geometry wrappers are guarded
  production routes, not new claims by themselves. The OU/readout audit and
  raw-edge residual adjudication remain claim-critical gates.

Folded-in active-sensing movie-information work should be cited here when it
helps explain the premise or controls: real-vs-stabilized retinal movies,
pose-aware versus pose-blind information accounting, input whitening, phase or
random motion controls, covariance-aware scale checks, and recorded-cortex
anchors. It should not be split into a separate figure story or used to claim
that the exact recorded trajectory is optimal.

Resume from:

- `declan/figure4_active_sensing_atlas/provisional_panel_contract_v0.csv`
- `declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v5.png`
- `declan/figure4_active_sensing_atlas/claim_critical_diagnostics_queue.md`

## Analyses That Closed Important Doors

### Jacobian / Translation-Covariance Identity

Status: `Historical -> structurally rescued`.

The first-order pushforward idea was right in direction and wrong in full
magnitude identity. Translation Jacobians captured leading FEM covariance
subspaces well above nulls, but naive magnitude predictions could over- or
under-shoot badly depending on estimator and scale.

Keep:

- translation tangents define useful response geometry;
- finite-cloud scale, phase, curvature, and resolution matter.

Avoid:

- a literal global covariance magnitude identity from `J Sigma_eye J.T`.

### Recorded Pose-Aware GLM

Status: `Closed / controlled null`.

The cache-first recorded GLM ladder did not show that simple additive or coarse
interaction eye-state covariates improve held-out recorded spike prediction
beyond valid-aware shuffled-eye controls. This does not refute covariance
closure or compact geometry; it bounds a simple content-blind prediction
endpoint.

Do not promote it as a main negative panel. Use it to avoid claiming that
recorded V1 information trivially improves when eye state is known.

### Structured Displacement Decoders

Status: `Diagnostic / current compact-specific decoder unsupported`.

Corrected structured decoders found some matched-context displacement signal,
but it weakens sharply under global-rate and top-target-PC projection controls.
The compact chart did not robustly beat gain-only or chart-shuffle controls on
the gain-orthogonal displacement endpoint.

Keep:

- recorded responses contain some matched-context displacement information;
- compact geometry remains strong at covariance/structural levels.

Avoid:

- a promoted compact-specific single-trial displacement decoder claim.

### Forward Twin Denoising

Status: `Diagnostic / not promoted`.

Forward-twin compact corrections beat random and unit-shuffled compact geometry
controls, but did not beat matched shuffled-eye controls. The effect is not yet
eye-trace-specific under the current tests.

Use this as a boundary result: the current twin/metric captures second-moment
geometry more robustly than precise single-trial eye-trace phase.

### Content-Routed Correct-Chart Alignment

Status: `Diagnostic / machinery validated, recorded effect fragile`.

Chart-swap machinery works and pseudo positive controls pass. Recorded
true-chart effects are split-, subset-, and session-sensitive. The gain-bottom
positive is a targeted hint, not a stable bridge.

Do not use this as a rescue route unless a preregistered targeted rerun
survives.

### Direct Recorded Derivative Alignment

Status: `Supportive`.

Recorded eye-position sensitivity is enriched in the compact fitted-twin
tangent subspace. This supports compact geometry but should not be rewritten as
signed horizontal/vertical axis recovery.

### Matched Twin Covariance Closure

Status: `Promoted`.

Finite-difference fitted-twin retinal translation sources predict a reliable
component of recorded FEM-linked covariance in matched recorded/twin unit
space. RF/readout-preserving nulls are the strongest reviewer-facing version.

Use strict wording: reliable component, above nulls. Do not claim full
covariance reproduction.

### E-Optotype Hyperacuity And Covariance Code

Status: `Historical guardrail / narrowed`.

The E-optotype crossover is real in the windowed pipeline, but it is not a main
current figure headline. The covariance-code and temporal-code interpretations
did not survive: FEM-related covariance subspaces did not rotate with E
orientation, covariance decoders were near chance, temporal residual features
did not rescue orientation decoding, and continuous forward-pass variants did
not improve the story.

Keep:

- dynamic FEM can help near the model's resolution limit through spatial
  sampling;
- fixed-center is an oracle, not a biological static baseline.

Avoid:

- covariance-code migration;
- temporal-code rescue;
- biological trajectory optimality from E-optotype results.

## Current Guardrails

Claims to avoid unless new evidence lands:

- real FEM trajectories are optimal;
- a covariance-aware Fisher peak near empirical FEM scale proves optimization;
- V1 has a literal universal 2D eye-position coordinate map;
- compact geometry is unique over static response PCs;
- FEM covariance fully explains recorded shared variability;
- permissive absolute eye-position decoding proves compact translation
  geometry;
- the E-optotype crossover is caused by a temporal code or covariance-code
  migration;
- raw denoising or raw decoder performance is meaningful without matched nulls;
- behavior follows a model objective beyond raw edge geometry;
- along-edge priors imply that known along-edge motion is intrinsically
  feature-optimal.

Current safe claims:

- recorded FEM-linked covariance has a reliable low-dimensional reafferent
  component;
- the deterministic twin is useful for explaining structure, not by itself
  proving perception or optimality;
- image translation tangents are image-specific but compact as a family;
- compact response components are functionally important for latent-eye
  feature recovery, but not unique over static response manifolds;
- retinal motion can improve model feature/information endpoints when pose is
  known or marginalized appropriately;
- measured drift/fixation geometry follows clear local edges modestly and
  reliably.

## Fast Resume

For current Figure 4 active-sensing:

1. Read `declan/figure4_active_sensing_atlas/provisional_panel_contract_v0.csv`.
2. Read the relevant `4*_companion_*.md` file.
3. Check `declan/figure4_active_sensing_atlas/claim_critical_diagnostics_queue.md`.
4. Treat `figure4_selected_v5.*` as the current provisional composite.
5. Use `declan/active_sensing_movie_information/README.md` and related
   prescriptions only as folded-in support/control history for Figure 4.
6. For 4C/4D post-2026-06-22 details, check the companion docs and diagnostics
   under `declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/` and
   `declan/figure4_active_sensing_atlas/figures/panel_D/diagnostics/`.

For compact geometry / covariance closure:

1. Read `declan/compact_retinal_translation_geometry/static_pc_control_adjudication_note.md`.
2. Check `outputs/compact_retinal_translation_geometry/tables/acceptance_matrix.csv`.
3. Keep covariance closure promoted; keep decoder bridges diagnostic.
4. Treat redundancy-resolved population specs under
   `outputs/redundancy_resolved_v1_twin/` as rerun infrastructure until a
   downstream analysis explicitly validates the reduced population endpoint.

For old E-optotype work:

1. Assume the final mechanism is mean-rate spatial sampling unless a new
   control proves otherwise.
2. Do not reopen covariance-code or temporal-code interpretations without a new
   explicit hypothesis.

For full historical details:

```text
declan/ANALYSIS_NARRATIVE_DETAILED_2026-06-22.md
```
