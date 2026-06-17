# FEM-V1 roadmap: current status and ways forward

Last updated: 2026-06-17

## Purpose of this note

This note is meant to clarify where the FEM-V1 paper stands now, what analyses are viable, and how to decide whether the compact reafferent geometry plus Wu-style observer branch belongs in the main manuscript or becomes its own paper.

The immediate goal is not to choose the final paper structure prematurely. The better plan is to keep two routes alive, develop each to the point where it is either figure-ready or clearly bounded, and then decide what strengthens the manuscript while it is being written.

The two routes are:

1. **Main-paper route without compact geometry as a required pillar**
2. **Compact geometry plus Wu-style observer route, either folded in as a mechanistic module or held for a separate paper**

The core judgment is that the compact geometry and Wu-style observer results should probably go in or stay out together. Compact geometry alone is interesting, but it risks landing as a descriptive covariance result. The Wu-style observer gives the geometry a plausible role: natural-image structure and trajectory-aware inference can turn FEM-induced response variability from nuisance into useful evidence. Conversely, the Wu-style observer is more compelling if it is tied to the compact reafferent geometry rather than presented as an isolated decoding analysis.

## Current high-level status

The paper can be strong without the compact geometry story.

The core paper already has a coherent thread:

1. During fixation, a large component of foveal V1 shared variability is linked to measured fixational eye movements.
2. This component is better understood as sensory reafference from self-generated retinal motion than as internally generated noise.
3. The FEM-linked covariance is low-dimensional and substantially aligned with stimulus-driven population structure.
4. Digital-twin analyses can test whether the effects are consistent with retinal translation rather than a generic extra-retinal or behavioral-state signal.
5. BackImage analyses provide an active-sensing and ecological link: FEMs during natural-image viewing are shaped by local image geometry, with the clearest current behavioral signature being motion biased along local oriented structure and edge-parallel preservation.

That story is already enough for a manuscript about V1 shared variability, active sensing, and reafference.

The compact geometry branch is deeper but more complicated. It could become the conceptual payoff of the paper if it also explains the along-contour behavior or predicts when edge-parallel trajectories are useful. If it does not connect to behavior, it may be better as its own paper.

## Route A: main paper without compact geometry as a required pillar

### Core claim

A large fraction of classical V1 shared variability during fixation is reafferent. The animal's eyes move, the retinal image moves, and V1 responses change accordingly. In screen coordinates, this appears as noise correlation. In eye-conditioned or retinal coordinates, much of it is structured sensory variability.

### Figure logic

A possible main-thread figure structure is:

#### Figure 1 or 2: FEM-linked variability in recorded V1

Show that conditioning on measured eye position reduces classical shared variability, including Fano factors and mean pairwise noise correlations. The removed component is low-dimensional and aligned with stimulus-locked response structure.

The key message:

```text
Classical noise correlations in foveal V1 are inflated by unmodeled retinal motion.
```

#### Figure 3: Reafference rather than extra-retinal signal

Use the twin and recorded/twin bridge to show that the eye-linked component is consistent with retinal translation through visual response geometry, not simply an extra-retinal eye-position signal or global gain.

The key message:

```text
Measured eye movements explain V1 shared variability because they change the retinal input.
```

This figure does not need the full compact-geometry story. It can focus on the reafferent covariance bridge, control analyses, and matched-unit prediction.

#### Figure 4: Active-sensing and natural-image behavior

Use BackImage aggregate FEM-information and local-image geometry results.

Candidate ingredients:

1. Real or empirical-like motion distributions improve feature-readout or trajectory-marginalized image identity relative to zero-eye or static-wrong observers.
2. Observed drift axes are biased toward local oriented image structure.
3. Edge-parallel motion perturbs pixels and V1-twin responses less than edge-orthogonal motion.
4. The strongest current behavioral prediction is along-contour or edge-parallel preservation, not global infomax optimality.

The key message:

```text
FEMs are not arbitrary motor noise. During natural-image viewing, their geometry is linked to local image structure and supports a useful sampling regime.
```

### Strengths of this route

This route is simpler, more coherent, and less dependent on the most complicated digital-twin claims. It keeps the paper centered on recorded V1 and on the reafferent reinterpretation of shared variability.

It also avoids overloading the manuscript with a second conceptual system: compact tangent bases, content-routed Jacobians, trajectory-table observers, and posterior diagnostics.

### Weaknesses of this route

The active-sensing interpretation may remain somewhat suggestive rather than mechanistically complete. The paper would show that FEMs explain shared variability and are related to natural-image structure, but it would not fully explain what the compact reafferent geometry is good for.

## Route B: compact geometry plus Wu-style observer as a mechanistic module

### Core claim

V1 does not represent fixational eye position with a universal signed coordinate axis. Instead, small retinal translations produce image-specific response changes that are routed through a shared low-dimensional population geometry.

Locally:

```text
r(I, tau) ≈ r0(I) + J(I) tau
```

and the compact-geometry claim is:

```text
J(I) ≈ U_trans A(I)
```

Here:

- `U_trans` is a shared low-dimensional reafferent population channel.
- `A(I)` is an image-dependent routing matrix.
- The same physical displacement can mean different activity patterns depending on the image.
- But translation effects are not arbitrary across images, they occupy a shared envelope.

The Wu-style observer gives this geometry a possible function. An observer that knows the V1 likelihood, a natural-image prior, and an eye-motion prior can treat eye trajectory as a latent variable:

```text
p(I, tau_1:T | y_1:T) ∝ p(y_1:T | I, tau_1:T) p_nat(I) p_FEM(tau_1:T)
```

The practical finite-table version scores image identity by marginalizing trajectory:

```text
log p(y | I) = log sum_tau p(y | I, tau) p(tau)
```

The key observer comparison is:

```text
known-eye >= joint-eye > zero-eye
```

### Why this is conceptually important

The same V1 activity can be noise or signal depending on the observer.

For a pose-blind observer, eye-induced response changes are nuisance covariance. For a pose-aware or trajectory-aware observer, those changes are structured evidence about image content under retinal motion.

This reframes the covariance result:

```text
FEM-linked variability is not merely noise to be removed. It is latent-variable structure.
```

### Current compact-geometry status

The compact-geometry result is already strong as a representation result:

- Local translation tangents are image-specific.
- The pooled tangent family is compact relative to unit-shuffled controls.
- Cross-image train/test tangent bases generalize to held-out images.
- A compact tangent basis captures a meaningful fraction of FEM-related local displacement sensitivity.
- Matched recorded/twin finite-difference translation covariances capture a reliable component of recorded FEM covariance.
- The controlled recorded-covariance bridge is preserved when finite-difference predictions are restricted to the compact tangent subspace.

This is already an interesting result, but it is still partly descriptive unless tied to a function or behavior.

### Current Wu-style observer status

The Vernier trajectory-table observer provided a useful negative diagnostic:

```text
known-eye high
zero-eye weak
joint-eye near chance
trajectory posterior diffuse
```

Interpretation:

```text
The Vernier stimulus contains fine-position information when pose is known, but it is too impoverished to support pose-free trajectory marginalization.
```

The BackImage trajectory-table observer is now directionally positive:

```text
known-eye high
zero-eye lower, especially at larger motion scale
joint-eye improves over zero-eye
posterior trajectory concentration increases with motion scale
```

In the n64 Option C pilot:

```text
0.25x: zero 0.797, joint 0.844 to 0.875, gain +0.047 to +0.078
0.50x: zero 0.719, joint 0.797 to 0.844, gain +0.078 to +0.125
1.00x: zero 0.484, joint 0.781 to 0.812, gain +0.297 to +0.328
```

The mechanistic diagnostic is also in the right direction:

```text
lower N_eff / K is associated with larger joint-minus-zero score gain
```

Current interpretation:

```text
Natural-image structure can support trajectory-marginalized image identification in the V1 twin, unlike the impoverished Vernier stimulus.
```

Current caveat:

```text
Empirical priors do not clearly beat OU priors in the larger run, so this is not yet evidence that real FEM statistics are uniquely optimal.
```

## The key integration question

The compact geometry and Wu-style observer branch should fold into the main paper only if they help explain the observed along-contour behavior.

The along-contour behavior is currently the most biologically grounded active-sensing result. Observed BackImage drift is biased toward local image orientation, and edge-parallel motion perturbs pixels and V1-twin responses less than edge-orthogonal motion.

If the compact/Wu-style module predicts the same behavioral geometry, it can become the functional payoff:

```text
The compact reafferent geometry not only explains FEM-linked covariance, it predicts which natural-image motions are useful or safe, and those predicted directions align with observed fixational drift.
```

If it does not predict the along-contour result beyond raw edge geometry, it may be too complicated for the main paper.

## Why along-contour prediction is the gating test

Pure information maximization may not predict along-contour drift. Motion orthogonal to an edge often produces stronger visual modulation than motion along the edge. If the objective is simply to maximize response change or local Fisher information, across-contour motion may look better.

Along-contour motion is more naturally explained by a tradeoff:

```text
move enough to sample or refresh the retinal input,
but move preferentially along directions that preserve local structure and limit pose-induced nuisance.
```

That fits the current behavioral result better than a pure infomax story.

So the integrated mechanism should not be:

```text
FEMs move along contours because that maximizes V1 response modulation.
```

It should be:

```text
FEMs move along contours because edge-parallel trajectories preserve image identity and reduce pose cost while still allowing useful trajectory-aware inference.
```

## The decisive analysis: axis-conditioned trajectory observer

The next analysis that could unify the branches is an axis-conditioned BackImage trajectory observer.

For each local BackImage patch, construct matched trajectory families:

```text
edge_parallel
edge_orthogonal
real_empirical
rotated_empirical
OU_matched
Brownian_matched
static
```

Match them on:

```text
RMS displacement
path length
duration
number of time bins
clipping fraction
```

Then compute the usual observers:

```text
known-eye
zero-eye
joint-eye
```

Primary metrics:

```text
joint_eye_accuracy
zero_eye_accuracy
joint_minus_zero_accuracy
known_minus_joint_gap
N_eff / K
nearest_trajectory_rank
joint_vs_best_dilution_gap
pixel perturbation
V1 response perturbation
observed drift-axis alignment
```

The strongest result would be:

```text
edge-parallel or real-like trajectories:
  preserve image identity better under joint marginalization
  produce lower pose-induced nuisance than edge-orthogonal trajectories
  retain enough motion signal for posterior concentration
  align with observed drift axes
```

This would let the compact/Wu-style module fold into the main paper, because it would explain the behavioral result rather than merely elaborating the covariance result.

### Critical baseline

The model-derived objective must be compared against raw edge geometry.

If raw edge orientation predicts observed drift as well as the model-derived compact/Wu objective, then the model objective may be biologically redundant for the main paper. That would still be useful, but it would favor a simpler main-paper story:

```text
Local image geometry predicts drift orientation, and edge-parallel motion preserves pixels and V1 responses.
```

The compact/Wu-style observer would then be better saved for its own paper.

## Promotion criteria

### Include compact/Wu module in the current paper if all are true

1. The BackImage trajectory-observer result survives harder controls:
   - matched-static-response candidates
   - larger candidate sets
   - leave-one-out trajectory priors
   - posterior diagnostics

2. The axis-conditioned observer predicts along-contour or edge-parallel utility:
   - edge-parallel trajectories outperform edge-orthogonal trajectories under joint marginalization, or
   - real drift axes align with trajectories that optimize the joint observer's robustness/pose cost tradeoff.

3. The model-derived result adds something beyond raw edge geometry:
   - predicts residual drift-axis deviations beyond edge orientation, or
   - predicts image-dependent cases where edge-parallel motion should or should not help, or
   - links posterior concentration and image-identity preservation to the observed drift geometry.

4. The story can be explained in one concise Results section without derailing the main paper.

### Keep compact/Wu module as a separate paper if any are true

1. The result is robust but requires substantial explanation of:
   - compact tangent bases
   - finite trajectory catalogs
   - Poisson likelihood observers
   - posterior entropy diagnostics
   - candidate-set construction

2. The result supports a functional role for compact geometry but does not explain along-contour behavior beyond raw edge baselines.

3. The result is exciting but too methodologically distinct from the main covariance/reafference paper.

4. The main manuscript is stronger when focused on recorded shared variability, reafference, and natural-image behavior.

### Demote compact/Wu module if any are true

1. Joint-eye improvement collapses under matched-static-response controls.
2. Posterior concentration does not survive larger K or harder candidate sets.
3. Empirical/real-like trajectories never outperform generic synthetic priors and the result reduces to generic trajectory marginalization.
4. The model-derived axis objective fails to beat raw edge geometry and does not explain residual behavior.
5. The compact geometry cannot be sufficiently anchored to recorded V1 rather than the twin architecture.

## Immediate next steps

### Main manuscript writing

Write the paper without assuming the compact/Wu module will be included.

Default title-level thread:

```text
Fixational eye movements reveal reafferent structure underlying V1 shared variability.
```

Default main claims:

1. FEMs account for a large component of classical shared variability in foveal V1.
2. The FEM-linked component is low-dimensional and aligned with stimulus-driven structure.
3. Digital-twin analyses show that this component is consistent with retinal reafference.
4. Natural-image viewing reveals image-contingent FEM geometry, especially along-contour or edge-parallel behavior.
5. FEMs should be treated as active sensory sampling, not merely motor noise.

Keep compact geometry as an optional mechanism while drafting.

### BackImage aggregate active-sensing branch

Finalize the simpler main-paper active-sensing result:

1. Edge-parallel versus edge-orthogonal preservation in pixels and V1-twin responses.
2. Observed drift-axis alignment with raw local image orientation.
3. Aggregate FEM feature-information results, if stable, framed as support rather than the main mechanism.
4. Avoid claims that real FEMs uniquely maximize information unless controls clearly support it.

### BackImage trajectory-observer branch

Run the next confirmatory observer analysis:

```text
n_images = 64 or 128
n_candidates = 8
K = 8 or 16
scales = 0.5x, 1.0x
priors = empirical, OU, shuffled-position
candidate modes = hard_negative_structure, matched_static_response
likelihood scales = 0.5, 1.0
primary prior mode = leave-one-out
```

Primary success pattern:

```text
known-eye high
zero-eye impaired at larger motion
joint-eye > zero-eye
joint advantage associated with lower N_eff / K
effect survives matched_static_response
```

### Axis-conditioned observer branch

Design and run a small diagnostic:

```text
edge_parallel vs edge_orthogonal
matched RMS/path length/duration
n_images = 32 or 64
K = 4 or 8 per axis family
candidate mode = hard_negative_structure or matched_static_response
scale = 0.5x and 1.0x
```

Primary question:

```text
Do edge-parallel trajectories preserve image identity or improve joint robustness relative to edge-orthogonal trajectories?
```

Secondary question:

```text
Does the model-predicted useful/safe axis explain observed drift-axis alignment beyond raw edge orientation?
```

## Current recommended manuscript posture

The safest posture is modular.

Write the current paper as if the compact/Wu branch is not required. Keep the central thread simple and strong:

```text
FEMs transform the interpretation of V1 shared variability from internal noise to structured reafference.
```

Then develop the compact/Wu branch as a candidate mechanistic module.

If it explains along-contour behavior, it can fold in as the functional payoff. If it remains a rich but separate observer story, it becomes its own paper. If it fails under harder controls, the main paper is unaffected.

## Working decision tree

```text
Does the main paper stand without compact geometry?
  yes -> draft it that way.

Does compact geometry robustly predict recorded FEM covariance?
  yes -> keep as optional mechanistic bridge or supplement.

Does Wu-style BackImage observer survive hard controls?
  yes -> compact geometry has a plausible functional role.

Does the observer/geometry predict along-contour behavior beyond raw edge baseline?
  yes -> consider folding into main paper.
  no  -> keep compact/Wu as separate paper or supplement.

Does the compact/Wu module make the main paper harder to understand?
  yes -> separate paper unless it is the key result.
```

## Bottom line

There are two viable papers here.

The first paper is about recorded V1 variability:

```text
FEMs account for much of foveal V1 shared variability, and this variability is structured reafference rather than internal noise.
```

The second paper, if the branch holds, is about computation:

```text
V1's compact reafferent geometry and natural-image structure allow trajectory-aware observers to recover image information under unknown fixational motion.
```

The bridge between them is along-contour behavior. If compact geometry plus trajectory-aware inference explains why real drift follows local image structure, then it belongs in the main paper. If not, it should probably become its own story.
