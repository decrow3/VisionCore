# Structured Translation Decoder Analysis

## Motivation

The compact translation-geometry result says something stronger than "eye position matters." The law-of-total-covariance analysis already shows that conditioning on measured eye position removes a component of shared variability. A decoder analysis is only useful if it tests a more specific claim about the V1 population code:

> V1 responses contain a content-routed local translation code that cannot be reduced to global gain.

The current geometry suggests that retinal translations are neither represented by a universal x/y axis nor scattered arbitrarily across population space. Instead, for each stimulus-history object, small retinal translations produce an image-specific local response chart, and these local charts live in a shared compact population subspace.

This implies a discriminating decoder test. The goal is not to beat a flexible MLP on in-distribution prediction. A black-box MLP is an empirical ceiling, and beating it would mostly say that the MLP was underpowered or poorly regularized. The scientifically meaningful questions are:

1. Does a compact, structured decoder outperform a gain-only decoder on the displacement component that gain cannot explain?
2. Does the structured decoder approach the MLP ceiling, suggesting that the compact geometry is sufficient to explain what is decodable?
3. Under distribution shift or low-data conditions, does the structured decoder generalize better than the MLP because it uses the right inductive bias?

The key success criterion is:

> A compact structured decoder recovers local displacement direction above the gain-only null, especially for the displacement component orthogonal to the local gain axis.

## Coding Model

Let the population response be

```text
r(I, tau) in R^N
```

where `I` is the image or full stimulus-history object, and

```text
tau = (tau_x, tau_y)
```

is the local retinal translation. Around the current retinal pose,

```text
r(I, tau) ~= r0(I) + J(I) tau
```

where

```text
J(I) = [b_x(I)  b_y(I)]
```

contains the horizontal and vertical local translation tangents.

The compact-geometry result can be written as

```text
J(I) ~= U_trans A(I)
```

where:

- `U_trans` is a shared low-dimensional population subspace, approximately `N x k`, with `k ~ 10`.
- `A(I)` is an image- or history-specific routing matrix, approximately `k x 2`.

This is not a separable code of the form

```text
r(I, tau) = f(I) + g(tau)
```

because there is no universal population vector for rightward or upward retinal displacement. The direction within `U_trans` depends on image content through `A(I)`.

The useful framing is:

> Shared transformation channel, content-dependent routing.

## Local ML Decoder Implied By The Geometry

Under the local linear model with Poisson-like weighting,

```text
W(I) = diag(1 / r0(I))
```

the local maximum-likelihood / weighted least-squares estimator for displacement is

```text
tau_hat(I) = (J(I)^T W(I) J(I))^-1 J(I)^T W(I) (r - r0(I)).
```

Using the compact model,

```text
J(I) = U_trans A(I),
```

this becomes a structured, content-aware decoder. It is not a black box: the decoder is determined by the local tangent geometry.

## Decoder Ladder

Run a ladder of decoders rather than a head-to-head fight against the MLP.

### 1. Gain-Only Decoder

The global-gain null assumes eye-linked response changes are proportional to the baseline response:

```text
J_gain(I) propto r0(I).
```

Under this null, horizontal and vertical translation effects collapse onto a single local gain direction. The local Jacobian is effectively rank 1.

Consequence:

> Only one gain-like displacement component can be recovered. The displacement component orthogonal to this local gain axis should be formally unidentifiable and therefore at chance.

This is the critical null because global gain is the main interpretive confound.

### 2. Compact Structured Decoder

Use the model-derived compact translation geometry:

```text
J_struct(I) = U_trans A(I).
```

This decoder allows a rank-2 local translation chart inside the compact subspace. It should recover displacement direction components that the gain-only model cannot.

Important implementation detail:

- `U_trans` and `A(I)` should be built from twin tangents.
- Use image-disjoint or cross-fit construction wherever possible.
- Avoid estimating `A(I)` from the same recorded responses being decoded unless explicitly running a data-built comparison.

### 3. Content-Aware MLP

Use the existing MLP decoder as an empirical ceiling, not as the primary opponent.

Interpretation:

- If the structured decoder matches the MLP, the compact first-order geometry is close to sufficient for the decodable signal.
- If the MLP beats the structured decoder, the gap bounds the compact-geometry claim and suggests additional nonlinear, higher-order, or non-translational structure.
- If the structured decoder beats the MLP only in low-data or transfer settings, that is an inductive-bias result rather than an in-distribution accuracy claim.

## Headline Discriminator: Orthogonal-To-Gain Direction Decoding

The cleanest test is not raw eye-position decoding. It is:

> Can V1 responses decode the local displacement direction orthogonal to the gain axis?

Under pure global gain, translation tangents are parallel to `r0(I)`, so the local translation information is rank 1. The displacement component orthogonal to that gain direction is not recoverable.

Under the content-routed translation-chart model, `J(I)` is rank 2 within the compact translation subspace, so the second displacement dimension should be decodable.

Operationally:

1. For each image/history object, define the local gain-predicted displacement axis in the 2D displacement plane.
2. Decompose true displacement into:

```text
tau_parallel_gain
tau_orthogonal_gain
```

3. Evaluate decoders specifically on `tau_orthogonal_gain` or displacement direction after projecting out the gain-predicted component.

Prediction:

```text
gain-only decoder ~= chance on orthogonal component
compact structured decoder > chance
```

This is the strongest test against the global-gain explanation.

## Controls And Guardrails

### Remove Global Modes

Evaluate the key decoder comparisons after removing:

- global rate / population mean activity,
- target PC1 or known dominant target-aligned component,
- any other already identified global-rate confound.

The structured decoder should retain performance on the orthogonal-to-gain displacement component after these removals.

### Cross-Fit / Image-Disjoint Construction

The structured decoder must not learn the held-out image's recorded responses directly.

Preferred:

- learn `U_trans` from image-disjoint twin tangents,
- apply to held-out image responses,
- if estimating any mapping from twin to recorded units, cross-fit across sessions/images/trials.

Avoid:

- estimating the local chart from the same recorded data used for decoding,
- reporting same-image performance without a leakage audit,
- comparing a heavily regularized structured decoder against an undertrained MLP and interpreting "structured wins" as biology.

### Decode Direction Before Magnitude

Magnitude is easier to confound with gain. Direction, especially the component orthogonal to local gain, is the discriminating target.

Suggested targets:

- sign or angle of local displacement direction,
- orthogonal-to-gain displacement component,
- 2D direction class after removing gain axis,
- not absolute displacement amplitude as the headline.

### Twin-Built And Data-Built Variants

If feasible, report two structured decoders:

1. **Twin-built structured decoder:** uses twin tangents and compact geometry. This is the mechanistic test.
2. **Data-built structured decoder:** estimates a similar low-rank chart from recorded data with cross-validation. This separates a biological failure from a twin-mismatch failure.

If the data-built version succeeds but twin-built fails, the coding principle may be real but the twin mapping is incomplete.

### Null Decoders

Use nulls that isolate the relevant structure:

- gain-only rank-1 decoder,
- random low-rank basis matched to `k`,
- unit-shuffled tangent basis,
- image-label shuffled chart routing,
- content-blind global linear decoder,
- content-aware MLP ceiling.

## Metrics

Use metrics that match the target.

For direction:

- angular error,
- circular correlation,
- cosine similarity between true and decoded displacement direction,
- classification accuracy for direction bins,
- signed correlation on orthogonal-to-gain component.

For sufficiency:

- structured decoder performance / MLP performance,
- gap between structured and MLP,
- gap between gain-only and structured.

For transfer/data efficiency:

- performance as a function of training trial count,
- held-out image generalization,
- held-out session generalization,
- cross-session transfer after shared unit/readout alignment.

## Success Criteria

### Strong Success

```text
gain-only ~= chance on orthogonal-to-gain direction
compact structured > gain-only
compact structured > chance after global-rate/PC1 removal
compact structured approaches MLP ceiling
```

Interpretation:

> V1 contains local displacement information beyond global gain, and the compact content-routed translation geometry is largely sufficient to explain the decodable signal.

### Partial Success

```text
compact structured > gain-only
MLP > compact structured
```

Interpretation:

> The compact first-order translation geometry captures a real component of the decodable eye-movement signal, but additional nonlinear, higher-order, non-translational, or model-mismatch structure remains.

This does not break the story; it bounds it.

### Transfer Success

```text
MLP >= structured in-distribution
structured > MLP under low-data / cross-image / cross-session transfer
```

Interpretation:

> The compact geometry provides a useful inductive bias for generalizing the pose-related code.

This is the only setting where "better than the black box" is a clean win.

### Negative Result

```text
compact structured ~= gain-only
orthogonal-to-gain direction ~= chance
```

Interpretation:

> The recorded data do not support a rank-2 content-routed displacement code under the current analysis. This could mean the relevant signal is underpowered, the twin-to-recorded mapping is insufficient, or global gain accounts for the decodable component.

This should be treated as an interpretable bound, not tuned away.

## Why This Analysis Adds Value

This analysis is not another version of the law-of-total-covariance result.

The law-of-total-covariance analysis identifies that some recorded covariance is associated with eye state. The structured decoder asks a more specific question:

> Is the eye-linked signal shaped like local retinal translations of the image, rather than a generic global gain mode?

It turns the compact geometry into a falsifiable population-code prediction:

> V1 should contain displacement information in a rank-2, content-routed local chart, not only in a rank-1 gain axis.

That is a hard consequence of the geometry.

## Suggested Text If The Analysis Works

```text
The compact translation geometry also makes a decoder-level prediction. If FEM-linked activity were only a global gain fluctuation, the local displacement code would be effectively rank 1: only the component of retinal motion aligned with the gain axis would be recoverable. In contrast, the content-routed translation geometry predicts a rank-2 local chart within a shared compact subspace. We therefore compared a gain-only decoder, a compact structured decoder built from image-disjoint twin tangents, and a content-aware MLP ceiling. The structured decoder recovered displacement direction orthogonal to the local gain axis above chance and above the gain-only null, even after removing global-rate and target-PC1 components. Thus, the V1 population carries local retinal-displacement information that cannot be reduced to global eye-linked gain.
```

## Suggested Text If The MLP Beats The Structured Decoder

```text
The content-aware MLP exceeded the compact structured decoder, indicating that the first-order translation geometry is not a complete account of all decodable eye-movement-related structure. However, the structured decoder still outperformed the gain-only null on the displacement component orthogonal to local gain, showing that the recorded population contains rank-2 local translation information beyond a scalar gain signal. The gap to the MLP bounds the compact-geometry account and may reflect higher-order nonlinearities, non-translational signals, or mismatch between the digital twin and recorded population.
```

