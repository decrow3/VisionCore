# Recorded Pose-Aware Active-Sensing Analysis

## Motivation

The active-sensing figure currently makes its strongest information claim in the digital twin. The twin compares real retinal motion to stabilized counterfactual movies and shows that real fixational motion can increase pose-aware spatial information, especially for mid/high spatial-frequency content. This is valuable, but it leaves a natural reviewer question:

> Do recorded V1 spikes actually become more informative when retinal pose is known?

The recorded covariance analyses already show that measured eye position explains a large component of shared variability. The geometry analyses show that this variability has a compact translation-related structure. But the active-sensing claim needs a recorded-data bridge:

> In recorded V1, treating retinal pose as part of the sensory state should improve prediction or decoding relative to a pose-blind analysis.

This analysis should not try to prove that the animal's exact FEM trajectory is optimal. The more defensible claim is:

> Eye position is not merely a nuisance variable. It is part of the retinal sensory state, and including it improves the explanatory or decodable structure of recorded V1 responses.

This directly supports the central coordinate-frame thesis:

- Pose-blind view: FEM-driven modulation appears as noise/shared variability.
- Pose-aware view: the same modulation becomes predictable sensory structure.

## Core Question

Given recorded spike counts `Y`, nominal stimulus time or image identity `t`, and measured eye state `e`, compare:

```text
pose-blind model:   Y ~ t
pose-aware smoke:   Y ~ t + e
pose-aware sensory: Y ~ t + stimulus-by-eye interaction
```

or, for decoding:

```text
pose-blind decoder:   stimulus/retinal state decoded from Y without measured eye state
pose-aware decoder:   stimulus/retinal state decoded from Y with measured eye state or pose-conditioned labels
```

The first question is whether measured retinal pose improves held-out response prediction or stimulus/retinal-state decoding in recorded V1 beyond controls. The stronger active-sensing question is whether the improvement reflects stimulus-dependent retinal translation structure, rather than only additive eye-state modulation or global gain.

## Why This Adds Value Beyond Covariance Decomposition

The law-of-total-covariance analysis asks:

> How much response covariance is associated with measured eye state?

The recorded pose-aware active-sensing analysis asks:

> Does knowing eye state improve the encoding model or decoder of recorded V1 responses?

These are related but distinct. Covariance decomposition identifies a structured source of variability. Pose-aware prediction asks whether that variability carries useful sensory-state structure.

If positive, the result says:

> The same eye-linked variability that appears as shared noise in a stimulus-aligned analysis improves the description of the recorded neural code when retinal pose is included.

## Candidate Endpoints

There are several possible endpoints. They vary in ambition and risk.

### Endpoint 1: Held-Out Neural Log-Likelihood / Response Prediction

Fit encoding models to recorded spike counts and compare held-out prediction.

Models:

```text
M0: spike count ~ stimulus time / PSTH only
M1: M0 + additive eye position / eye trajectory
M2: M0 + gain-only or scalar eye-state modulation
M3: M0 + low-rank stimulus-by-eye / retinal-translation interaction
Mshuf: matched shuffled-eye versions of M1 and M3
```

Possible implementations:

- Poisson GLM per neuron.
- Negative-binomial GLM if overdispersion is severe.
- Low-rank population model with stimulus and eye terms.
- Nonparametric eye-bin conditioning if sample counts permit.

Primary metric:

```text
Delta log likelihood = LL(M1) - LL(M0)
translation-specific Delta = LL(M3) - max(LL(M1), LL(M2), LL(Mshuf))
```

or bits/spike:

```text
(LL_pose-aware - LL_pose-blind) / total spikes / log(2).
```

Strength:

- Direct recorded-data coding result.
- Naturally cross-validated.
- Does not require defining a stimulus decoder.

Risk:

- Can be dominated by firing-rate/gain effects unless controlled.
- Needs careful regularization and shuffle controls.

Best use:

> Primary recorded anchor if M1 is robust across sessions; stronger active-sensing anchor if M3 exceeds additive-eye, gain-only, and matched shuffled-eye controls.

### Endpoint 2: Decoding Nominal Stimulus State With And Without Eye State

Ask whether including eye state improves decoding of stimulus time/frame/image from population responses.

Pose-blind decoder:

```text
decode t or image from Y
```

Pose-aware decoder:

```text
decode t or retinal stimulus state from Y, conditioning on e
```

There are two versions:

1. **Conditioning version:** compare decoders trained/evaluated within eye-state bins or with eye-state covariates.
2. **Retinal-state relabeling version:** label each sample by the retinally shifted stimulus patch/state rather than nominal screen stimulus.

Primary metric:

- classification accuracy,
- cross-entropy,
- log likelihood of correct stimulus state,
- decoding improvement over shuffled-eye controls.

Strength:

- Intuitive "pose-aware decoding" result.

Risk:

- Defining retinal-state labels for natural images can be complex.
- Repeated stimulus sequence may make time priors strong.

Best use:

> Good if response-prediction endpoint is too abstract, but needs careful controls against time-only decoding.

### Endpoint 3: Pose-Aware Residual Independence

This is the closest to the existing covariance analysis but can be made predictive.

Fit:

```text
Y = stimulus component + pose component + residual.
```

Then ask whether the residual is closer to conditionally independent or less correlated on held-out data.

Metrics:

- residual noise correlations,
- Fano factor,
- covariance left in the FEM subspace,
- held-out likelihood under independent-Poisson residual model.

Strength:

- Directly links to the paper's covariance theme.

Risk:

- Can feel redundant with law-of-total-covariance unless explicitly held-out/predictive.

Best use:

> Supporting audit, not the only endpoint.

### Endpoint 4: Pose-Aware Information Bound From Recorded Responses

Estimate information under pose-blind vs pose-aware assumptions directly from recorded spikes.

Pose-blind:

```text
I(Y; stimulus time)
```

Pose-aware:

```text
I(Y; stimulus time, eye state)
```

or conditional:

```text
I(Y; stimulus time | eye state)
```

Possible estimators:

- cross-validated multinomial decoding information,
- Poisson encoding model likelihood converted to bits/spike,
- lower-bound mutual information from classifier cross-entropy.

Strength:

- Most directly addresses "recorded V1 information."

Risk:

- Harder statistically.
- Easy to overclaim absolute information values.

Best use:

> Use as a lower-bound or model-based comparison, not a definitive absolute information estimate.

## Recommended Primary Analysis

The safest first-pass primary analysis is:

> Cross-validated recorded response prediction with a model ladder: pose-blind PSTH, additive eye-state smoke test, gain-only eye-state control, and a stimulus-by-eye / retinal-translation interaction model.

This gives a recorded-data anchor for active sensing without needing to solve full natural-image decoding. The additive eye-state model should be treated as a sensitivity/smoke test. The strongest positive result is an interaction or translation-like term that exceeds both additive-eye and gain-only controls on held-out data.

## Basic Statistical Model

For neuron `n`, trial `i`, and time bin `t`, let

```text
y_{i,t,n}
```

be the spike count.

### Pose-Blind Model

```text
y_{i,t,n} ~ Poisson(lambda_{t,n})
log lambda_{t,n} = alpha_n + s_{t,n}
```

where `s_t` is the nominal stimulus-time/PSTH term.

### M1: Pose-Aware Additive Smoke Test

```text
log lambda_{i,t,n} = alpha_n + s_{t,n} + h_n(e_{i,t})
```

where `h_n(e)` is an eye-state modulation.

Eye-state features can include:

- horizontal eye position,
- vertical eye position,
- eye velocity,
- recent eye-position history,
- drift/microsaccade indicator,
- low-dimensional basis of recent eye trajectory.

Interpretation:

> M1 tests whether measured eye state predicts recorded spikes beyond the PSTH. It is not by itself sufficient for the sensory-coordinate claim because additive eye terms can absorb arousal, pupil/gain, slow behavioral state, or other non-translation effects.

### M2: Gain-Only / Scalar Eye-State Control

```text
log lambda_{i,t,n} = alpha_n + s_{t,n} + beta_n g(e_{i,t})
```

where `g(e)` is a scalar eye-state or behavioral-state drive shared across neurons up to cell-specific loadings. Variants include population-rate gain, low-dimensional global-rate factors, pupil/block terms if available, or projections onto global rate and target PC1.

Expected:

```text
M3 > max(M1, M2, shuffled controls)
```

for a translation-like active-sensing result. If `M2` explains most of the gain, the result should be interpreted as eye-linked state/gain prediction rather than evidence for retinal sensory coordinates.

### M3: Pose-Aware Retinal-Translation / Interaction Model

More mechanistic:

```text
lambda_{i,t} = lambda_t + B_t phi(e_{i,t})
```

where `B_t` is time-, image-, or stimulus-history-dependent and is constrained by model-derived translation geometry, low-rank recorded FEM covariance, or a cross-validated low-rank stimulus-by-eye interaction.

This is stronger but more complex.

Recommended path:

1. Start with a simple regularized pose-aware GLM.
2. Fit the gain-only/scalar eye-state control.
3. Promote the interaction or geometry-constrained version to the primary active-sensing rung if the simple model gives robust signal.
4. Use the active-sensing interpretation only for held-out improvement beyond additive-eye, gain-only, and shuffled-eye controls.

## Controls

### Shuffled-Eye Control

Permute eye traces across trials while preserving:

- stimulus sequence,
- marginal eye-position distribution,
- temporal eye dynamics if possible,
- session/block identity,
- stimulus time or local time-bin structure where possible.

Preferred first pass:

```text
shuffle eye traces within matched session/block and stimulus-time bins
```

or within narrow local windows that preserve slow time/block statistics. A global across-trial shuffle is useful as a permissive diagnostic but can change time/block structure and make the real-eye model look better for the wrong reason.

Expected:

```text
pose-aware real eye > pose-aware shuffled eye
```

This confirms the gain comes from trial-specific eye/neural coupling.

### Time-Only / Stimulus Prior Control

The repeated sequence creates strong time structure. Ensure the model does not gain simply from time priors.

Compare:

```text
stimulus/PSTH only
stimulus/PSTH + real eye
stimulus/PSTH + shuffled eye
eye-only / behavior-only
```

The eye-only model is important for decoding or information endpoints:

```text
Y or stimulus label ~ eye state only
```

or, for response prediction:

```text
spike count ~ eye state without stimulus time
```

This quantifies how much nominal stimulus time, image identity, or block structure is predictable from eye behavior alone.

### Global Gain Control

Add a model with only global population gain or scalar eye modulation:

```text
log lambda_{i,t,n} = alpha_n + s_{t,n} + g(e_{i,t}) beta_n
```

or remove:

- population mean activity,
- global rate,
- target PC1.

Expected:

```text
pose-aware model > gain-only model
```

if the effect is more than global gain. This should be a primary comparison, not only a post hoc audit.

### Slow Drift Control

Include slow time/session drift regressors or block effects. This prevents the eye model from absorbing slow nonstationarity.

### Trial-Disjoint Cross-Validation

Train/test splits must be trial-disjoint. For stronger tests, also hold out:

- image identities,
- time blocks,
- sessions.

### Eye-Trace Mismatch Control

Use eye traces from similar but wrong trials to test whether exact trajectory matters.

## Geometry-Linked Extensions

If the primary recorded pose-aware signal is positive, add one geometry-linked analysis.

### Translation-Subspace Prediction Improvement

Project recorded residuals into:

```text
U_trans
```

and into its orthogonal complement. Ask whether pose-aware prediction improves more in the translation subspace.

Expected:

```text
Delta LL or Delta R2 in U_trans > Delta in random matched subspaces.
```

This connects the recorded pose-aware prediction result to the compact geometry.

The preferred contrast is:

```text
excess = M3 - max(M1, M2, shuffled-eye M3)
```

and then test whether this excess is enriched in the recorded FEM covariance subspace, fitted-twin translation subspace, or compact translation basis relative to matched random, unit-shuffled, and RF/readout-preserving nulls.

### Gain-Orthogonal Variant

Within `U_trans`, remove global gain/PC1 and test whether pose-aware prediction survives.

Expected:

```text
pose-aware improvement survives global-mode removal.
```

## Relationship To The Active-Sensing Figure

This analysis would fit naturally beside the current model information panels.

Possible figure logic:

1. Real vs stabilized retinal movies in the twin.
2. Twin real-FEM motion increases pose-aware spatial information.
3. Gain depends on spatial frequency / spectral content.
4. Matched-motion controls bound optimality claims.
5. Recorded V1 pose-aware prediction: measured eye state improves held-out recorded response prediction or decoding beyond shuffled-eye controls.

The recorded panel would make the figure less model-only.

## Success Criteria

### Strong Success

```text
M1 improves held-out log likelihood across sessions
M3 exceeds max(M1, M2 gain-only, shuffled-eye controls)
real-eye improvement exceeds matched-time/block shuffled-eye controls
improvement survives global gain/PC1 controls
M3 excess is enriched in the recorded FEM or twin translation subspace
```

Interpretation:

> Recorded V1 responses are better described when retinal pose is treated as part of the sensory state, and the predictive gain is not reducible to additive eye state or global gain. The active-sensing model information result is therefore anchored in recorded neural coding.

### Partial Success

```text
pose-aware model improves LL but effect is mostly global/gain-like
```

Interpretation:

> Eye state carries predictive information about recorded responses, but the current result does not yet distinguish translation geometry from global modulation.

### Additive-Only Success

```text
M1 improves LL over M0
M3 does not exceed M1 or M2
```

Interpretation:

> Eye state improves held-out response prediction, but the current result should be described as pose-linked predictability rather than a translation-specific active-sensing bridge.

### Geometry-Specific Success

```text
pose-aware improvement is concentrated in U_trans or recorded Sigma_FEM subspace
```

Interpretation:

> The recorded pose-aware coding benefit is linked to the compact reafferent geometry.

### Negative Result

```text
pose-aware model ~= shuffled-eye control
```

Interpretation:

> Under the current estimator and data regime, measured eye state does not improve held-out recorded response prediction beyond time/stimulus structure. This would bound the active-sensing claim and may reflect limited repeats, noisy eye tracking, model mismatch, or insufficiently specific retinal-state labels.

## Important Interpretation Boundaries

Do not claim:

- the animal's exact eye trajectory is optimal,
- absolute information values are calibrated,
- downstream decoders use this exact model,
- all FEM covariance is useful.

Supported claim if positive:

> Recorded V1 responses contain pose-dependent structure that improves held-out prediction when retinal pose is known, supporting the view that FEM-linked variability is reafferent sensory structure rather than purely internal noise.

Stronger supported claim if the interaction/gain-null ladder succeeds:

> Recorded V1 responses contain stimulus-dependent pose structure that improves held-out prediction beyond additive eye-state and gain-only controls, linking the recorded predictive benefit to the compact retinal-translation geometry.

## Suggested Methods Sketch

```text
For each session, we fit cross-validated Poisson encoding models to spike counts using a ladder of predictors: nominal stimulus time alone, nominal stimulus time plus additive eye-state features, nominal stimulus time plus a gain-only eye-state factor, and nominal stimulus time plus a low-rank stimulus-by-eye interaction. Eye-state features included horizontal and vertical position, velocity, and recent trajectory terms. Models were fit on training trials and evaluated on held-out trials using log likelihood per spike. Shuffled-eye controls preserved the stimulus sequence, marginal eye statistics, and session/block or local stimulus-time structure while breaking trial-specific eye/neural coupling. Additional controls removed global population rate and target-PC1 components before fitting. We quantified the pose-aware gain as the held-out log-likelihood improvement of the real-eye model over the stimulus-only model, and the translation-specific gain as the excess of the interaction model over additive-eye, gain-only, and matched shuffled-eye controls.
```

## Suggested Results Text If Positive

```text
The model information analysis predicts that retinal pose should turn part of the apparent variability in V1 into predictable sensory structure. We therefore asked whether measured eye state improves held-out prediction of recorded V1 responses. Across sessions, an additive pose-aware encoding model that included measured eye position and velocity predicted held-out spike counts better than a pose-blind stimulus-time model. We then asked whether this improvement could be explained by global eye-linked gain. A low-rank stimulus-by-eye interaction model exceeded both the additive-eye and gain-only models, while matched-time shuffled-eye controls strongly reduced the effect. Thus, recorded V1 responses are better described when retinal pose is treated as part of the sensory state, and the predictive gain is not explained solely by global modulation.
```

## Suggested Results Text If Geometry-Enriched

```text
The pose-aware prediction gain was concentrated in the same compact reafferent dimensions identified by the covariance and twin analyses. Projecting responses into the model-derived translation subspace yielded larger pose-aware prediction improvements than matched random or unit-shuffled subspaces. Thus, the recorded predictive benefit of knowing retinal pose is linked to the compact translation geometry rather than to arbitrary low-dimensional modulation.
```
