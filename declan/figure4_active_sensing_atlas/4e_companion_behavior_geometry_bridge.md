# Companion: Behavior Geometry Bridge

Date: 2026-06-21
Status: provisional methods/logic companion for Figure 4E

## Panel Claim Under Test

```text
Real drift follows clear edges.
```

This is the result Panel 4E is there to show if the evidence supports it. The
claim is behavioral and geometric: measured drift/fixation-cloud axes should be
more edge-following when the local image supplies a clear orientation. The
motivation, assumptions, metrics, controls, diagnostics, and caveats below are
all there to decide how strongly this sentence can be stated, and to keep it
separate from the stronger claim that behavior optimizes a specific V1-twin
objective.

## Summary

The behavior geometry bridge asks whether measured FEM/fixation-cloud axes are
aligned with local image geometry. This is the necessary behavioral link for
the active-sensing story, but it is also the place where the claim must be most
carefully bounded. A behavioral alignment with image edges does not prove that
the animal optimizes any particular V1-twin objective.

The current supported claim is modest and useful: measured drift/fixation-cloud
axes are reliably contour-following, and the effect strengthens when local
axis estimates are more reliable. The unresolved claim is model-objective
specificity: raw edge geometry remains the hard baseline that current V1-twin
objectives have not yet cleanly beaten.

## Motivation

The model panels show that retinal motion can be useful and that local image
geometry defines plausible movement axes. The behavior panel asks whether the
animal's measured movement geometry points in the same broad direction. The
right bridge is geometric convergence:

```text
model side: contour-aligned motion can preserve local pixels/twin responses
behavior side: measured FEM axes are modestly contour-following
```

That bridge is weaker than a causal intervention and weaker than an
optimization proof. It is still important because it connects the model's local
axis vocabulary to measured free-viewing behavior.

## Notation And Estimator Contract

For each analyzed image/window:

```text
e_i: local edge or contour axis
d_i: measured drift/fixation-cloud axis
c_i: local edge/FEM confidence or reliability weight
delta_i = angle(d_i, e_i), modulo 180 degrees
```

The signed alignment score is:

```text
a_i = cos(2 * delta_i)
```

with interpretation:

```text
a_i = +1: drift axis parallel to local edge
a_i =  0: drift axis 45 degrees from edge
a_i = -1: drift axis orthogonal to local edge
```

The session-level contracts are:

```text
unweighted:
  A_s = mean_i a_i within session s

weighted:
  A_s^w = sum_i c_i a_i / sum_i c_i within session s
```

Uncertainty is reported by session bootstrap or session-level summaries. The
endpoint-zone enrichment panels use angle-bin fractions and must be read
against the transformed uniform-angle null, not against visually equal endpoint
bar heights.

## Plain-English Methods

The 4E analysis asks whether measured fixation motion tends to run along local
image edges. Each data point is an image window paired with the eye movement
measured during that window.

For each image window, the image side of the analysis estimates a local edge
axis. An axis has no arrow: 0 degrees and 180 degrees mean the same edge
orientation. The analysis also assigns a confidence or reliability value when
the local image has a clearer orientation.

For the behavior side, the eye trace is summarized as a drift or fixation-cloud
axis. This is the main direction of the small eye movements during the analyzed
window. Like the image edge, it is treated as an axis rather than an arrow, so
movement in opposite directions along the same line counts as the same
orientation.

The alignment between the two axes is measured by the angle between them. The
score is `cos(2 * angle)`. This transformation is used because axes repeat
every 180 degrees. A score near +1 means the drift axis is parallel to the edge.
A score near 0 means it is about 45 degrees away. A score near -1 means it is
orthogonal to the edge.

The analysis reports both window-level summaries and session-level summaries,
but inference is kept at the session level. This matters because thousands of
windows from the same animal/session are not thousands of independent animals.
The unweighted session mean gives each analyzed window equal weight within a
session. The weighted version gives more influence to windows with clearer
edge or movement-axis estimates.

Three subsets are reported. The all-window subset asks whether the effect is
present broadly. The reliable-axis subset keeps windows with more trustworthy
axis estimates. The high-confidence subset is stricter and asks whether the
effect grows when both the image and movement axes are especially clear.

The endpoint-zone panels count how often the alignment angle falls near
parallel or near orthogonal. These panels are intuitive, but they need a null
because equal-looking angle bins do not always imply equal expected fractions
after transforming an axis angle. The companion therefore keeps the endpoint
null diagnostic next to the endpoint enrichment result.

The raw-edge baseline is the main behavior benchmark. Model-derived objective
axes are only meaningful as behavioral mechanisms if they explain behavior
beyond this simple image-edge axis, on the same windows and with session-level
uncertainty. Current V1-twin objective axes have not yet passed that residual
gate.

## Assumptions

A1. The local image edge axis `e_i` is well-defined for the analyzed window or
is assigned a confidence/reliability score when it is not.

A2. The measured drift/fixation-cloud axis `d_i` is a stable local behavioral
summary rather than a dominated global screen-axis artifact.

A3. Session-level inference is the right unit for reliability; many windows
within a session do not constitute independent animals or independent sessions.

A4. Weighted and unweighted metrics answer slightly different questions and
must not be mixed without labeling.

A5. A positive raw-edge alignment is a behavior result, not a model-objective
win. Model-derived axes need residual tests beyond raw edge geometry.

## Controls

All-window, reliable-axis, and high-confidence subsets:

```text
Check whether the effect exists broadly and whether it strengthens when local
axis estimates are more trustworthy.
```

Endpoint-zone enrichment:

```text
Tests whether parallel-zone occupancy exceeds the uniform-angle expectation.
Must travel with the endpoint/null diagnostic.
```

Metric convention guardrail:

```text
Compares weighted headline-style metrics with unweighted session means.
```

Session bootstrap and null diagnostics:

```text
Keep the inference at the session level and guard against overreading the large
number of windows.
```

Raw-edge baseline:

```text
Any V1-twin objective must beat or explain residual behavior beyond raw edge
geometry before it can be called a behavioral mechanism.
```

## Existing Evidence

Primary source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_edge_alignment_distribution_inspection/
```

Headline weighted values:

```text
all-window weighted edge-axis cos2 = 0.181
session-bootstrap CI = [0.124, 0.241]
reliable-axis weighted edge-axis cos2 = 0.201
```

Unweighted session-level values:

```text
all windows:
  n_windows = 11749
  n_sessions = 30
  mean session cos2 = 0.105
  CI = [0.067, 0.145]
  median abs delta = 39.0 deg

reliable axes:
  n_windows = 6242
  n_sessions = 30
  mean session cos2 = 0.140
  CI = [0.089, 0.188]
  median abs delta = 36.4 deg

high confidence:
  n_windows = 1045
  n_sessions = 30
  mean session cos2 = 0.269
  CI = [0.138, 0.396]
  median abs delta = 25.6 deg
```

Endpoint-zone enrichment:

```text
Observed / uniform expected fraction in parallel <=15 deg zone:
  all windows = 1.304
  reliable axes = 1.427
  high confidence = 2.124

Observed / uniform expected fraction in orthogonal >=75 deg zone:
  all windows = 0.906
  reliable axes = 0.851
  high confidence = 0.833
```

Raw-edge baseline:

```text
raw_edge_axis:
  n_windows = 256
  n_sessions = 29
  mean session cos2 = 0.182
  weighted session cos2 = 0.218
  positive sessions = 23 / 29
```

Model-side geometric support:

```text
edge-parallel pixel preservation advantage:
  mean = 300.54, CI [172.789, 408.961], positive sessions 26/29

edge-parallel twin preservation advantage:
  mean = 0.000454497, CI [0.000371047, 0.000536519],
  positive sessions 29/29
```

Historical tests routed to this claim:

```text
BackImage scalar local-image features did not robustly predict RMS radius,
diffusion, speed, path length, anisotropy, return-to-center strength, or
high-frequency FEM fraction. The surviving behavior result is directional:
drift/fixation-cloud orientation aligns modestly with local edge and spectral
axes, especially in reliable-axis subsets.

The scaled BackImage twin drift-geometry adjudication found raw_edge_axis was
the strongest biological baseline: session mean cos2 +0.182, weighted +0.218,
23/29 positive sessions, random-axis p_ge = 0.0004. Optimized V1-twin PA/PB
and Pareto axes did not beat raw edge. This is why 4E should claim real drift
follows clear edges, not that behavior optimizes the tested model objective.
```

## Diagnostics And Failure Modes

The behavior bridge can fail or be overread in these ways:

```text
metric convention changes effect size;
endpoint-heavy cos(2 delta) histograms are read against the wrong null;
alignment appears only after aggressive confidence filtering;
within-session or global screen-axis bias explains the effect;
raw edge absorbs model-derived objective variables;
model objective axes are measured on different windows or populations than
behavior axes.
```

Current handling:

```text
Show E2 plus E3 and the provenance diagnostics E6/E7/E8.
State whether the figure uses weighted or unweighted metrics.
Keep raw edge as the baseline to beat.
Do not claim a V1-twin objective bridge until canonical geometry residual
adjudication closes.
```

## Current Claim Boundary

Supported:

```text
Measured free-viewing FEM/fixation-cloud axes are modestly but reliably aligned
with local image edge geometry, and the effect strengthens in reliable and
high-confidence subsets.
```

Not yet supported:

```text
The animal optimizes the aggregate FEM decoder, local I_z pairing objective, or
joint posterior observer objective.
Current V1-twin objective axes explain behavior beyond raw edge geometry.
The behavioral result is causal rather than correlational geometry.
```

## Production Rerun Implications

The canonical geometry surface should be the route for objective claims:

```text
declan/canonical_geometry/
```

Before promoting a model-objective behavioral bridge, require:

```text
same-window raw-edge/objective/behavior master table
within-session residual regression beyond raw edge confidence
global-axis nuisance audit
source-overlap and candidate-hardness audit
population/readout sensitivity
preservation-vs-modulation decomposition
```

If those gates fail, the main-paper wording should stay honest:

```text
behavior follows local image geometry, while the V1 twin shows possible
utility, preservation, and trajectory-aware inference consequences of such
geometry.
```
