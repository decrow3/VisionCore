# Compact Reafferent Geometry as a Joint Image-and-Eye Inference Framework

Last updated: 2026-06-15

## Executive summary

The compact reafferent geometry is more than a denoising observation. The strongest framing is that fixational eye movements (FEMs) drive V1 through a local response geometry that links image content, retinal pose, and population activity. In a stimulus-aligned analysis, this motion-induced activity appears as shared variability. For an observer that knows the geometry, natural-image statistics, and plausible FEM statistics, the same activity can become evidence for jointly estimating image content and retinal pose.

The key shift is:

```text
Denoising framing:
FEMs create structured shared variability, and compact geometry may make that variability easier to discount.

Joint-observer framing:
FEMs create response trajectories that supply extra constraints on the latent causes of V1 activity: image content and retinal pose.
```

This makes the pose-aware versus pose-blind asymmetry the central thesis rather than a complication. The same response component can be nuisance for a pose-blind observer and useful signal for a pose-aware or joint observer.

The compact geometry is not itself an observer, a denoiser, or an eye-position decoder. It is the likelihood structure that makes a joint image-and-eye observer plausible.

## 1. Why the denoising framing is incomplete

The law-of-total-covariance analysis separates recorded population covariance into stimulus-locked, FEM-linked, and residual components:

$$
\Sigma_{\mathrm{total}}
=
\Sigma_{\mathrm{PSTH}}
+
\Sigma_{\mathrm{FEM}}
+
\Sigma_{\mathrm{int}}.
$$

This is the statistical separation result. It shows that a large component of classical shared variability in foveal V1 is linked to measured eye position. After conditioning on eye position, the residual correlations collapse toward zero.

The compact-geometry result explains why the separated FEM component is structured. It says that the response changes induced by small retinal translations are not scattered randomly through the population. Instead, they occupy a low-dimensional population envelope.

That is important, but the word "denoising" can make the claim too narrow. If the story is only that compact geometry makes FEM-linked variance removable, then the analysis stays in a variance-accounting frame. It also leaves a conceptual tension: FEM covariance overlaps stimulus-driven dimensions, so removing it is not free. A fixed projection may reduce nuisance variance, but it may also remove stimulus signal.

A better framing is functional:

> FEM-induced response trajectories can be useful or harmful depending on the observer. They are harmful to an observer that ignores retinal pose, but useful to an observer that can use pose, infer pose, or combine the response trajectory with image and motion priors.

This is the role played by the joint-observer framework.

## 2. The Wu-style inference problem

Wu et al. provide the useful template. Their retinal analysis did not simply train a decoder from spikes to images. It built a generative observer: an encoding likelihood says which image and eye trajectory could have produced the spikes, while an external natural-image prior says which images are plausible. For jittered stimuli, the observer compares known-eye, zero-eye, and joint-estimated-eye cases.

The V1 analogue is:

$$
p(I,\tau_{1:T}\mid y_{1:T})
\propto
p(y_{1:T}\mid I,\tau_{1:T})
p_{\mathrm{nat}}(I)
p_{\mathrm{FEM}}(\tau_{1:T}).
$$

Each term has a simple interpretation.

### Neural likelihood

$$
p(y_{1:T}\mid I,\tau_{1:T})
$$

This asks: if the image were $I$, and the retinal pose trajectory were $\tau_{1:T}$, how likely would the observed V1 response $y_{1:T}$ be?

In the twin, this likelihood can be built from the predicted mean response:

$$
\mu_t = f_\theta(I,\tau_t),
$$

plus an explicit noise model. The noise model is not supplied by the noiseless twin. It must be specified separately, for example diagonal Gaussian, diagonal Poisson-like, or empirical residual covariance from recorded V1.

### Natural-image prior

$$
p_{\mathrm{nat}}(I)
$$

This says that not every image explanation is equally plausible. Natural images have edges, contours, textures, spatial correlations, and characteristic spectra. The prior prevents the observer from explaining V1 activity with arbitrary unnatural images.

This prior should be external to the V1 twin. That separation is crucial. If the image prior, likelihood, geometry, and scoring function all come from the same fitted model, the analysis risks becoming circular. The Wu logic is compelling because the encoding model and image prior are separable.

### FEM trajectory prior

$$
p_{\mathrm{FEM}}(\tau_{1:T})
$$

This says that not every eye trajectory is plausible. Fixational drift is smooth, confined, and has characteristic spatial and temporal statistics. This prior prevents the observer from explaining arbitrary response fluctuations by inventing arbitrary retinal motion.

Together, the observer asks:

> Which image and which eye trajectory jointly explain the V1 response, while remaining plausible under natural-image and FEM statistics?

## 3. Where compact geometry enters

For small retinal translations, the V1 response can be locally approximated as:

$$
\mu(I,\tau_t)
\approx
\mu_0(I) + J(I)\tau_t.
$$

Here:

- $\mu(I,\tau_t)$ is the mean V1 response to image $I$ at retinal pose $\tau_t$.
- $\mu_0(I)$ is the response to image $I$ at a reference pose, such as the stabilized or centered pose.
- $J(I)$ is the local translation Jacobian. It tells us how the population response changes for a tiny horizontal or vertical retinal shift.

In simple language, $J(I)$ is the image-specific map from small retinal motion to population activity.

If V1 had a separable eye-position code, the same displacement would add the same population pattern for every image:

$$
r(I,\tau) = f(I) + g(\tau).
$$

But that is not expected for V1. A rightward shift can increase a neuron for one image, decrease it for another, and do little for a third. The effect depends on local image structure.

So the translation Jacobian must be image-dependent.

The surprising result is that this image dependence appears structured:

$$
J(I) \approx U_{\mathrm{trans}}A(I).
$$

Here:

- $U_{\mathrm{trans}}$ is a shared low-dimensional population subspace.
- $A(I)$ is the image-specific routing matrix.
- Physical displacement $\tau_t$ is routed through $A(I)$ into the shared population channel $U_{\mathrm{trans}}$.

This is the central factorization.

It means V1 does not provide a universal coordinate system for retinal pose. The same direction inside $U_{\mathrm{trans}}$ need not mean the same physical displacement for every image. The meaning depends on $A(I)$.

But it also means translation effects are not arbitrary. They share a compact population envelope.

A concise phrase:

> V1 factorizes the transformation channel, but not the transformation coordinates.

## 4. Why this makes joint inference tractable

Project the response residual into the compact translation channel:

$$
z_t
=
U_{\mathrm{trans}}^\top
\left(y_t - \mu_0(I)\right).
$$

Using the local model,

$$
z_t \approx A(I)\tau_t + \epsilon_t.
$$

This is the key simplification. The observer does not need to fit an arbitrary high-dimensional population response change. Given a candidate image $I$, the response residual inside the compact channel becomes a low-dimensional pose-inference problem.

If the observer knows or can approximate $A(I)$, it can ask:

> Which small retinal displacement best explains the compact-channel residual?

But without $A(I)$, activity inside $U_{\mathrm{trans}}$ is ambiguous. It says that the response changed in a pose-sensitive channel, but it does not by itself say whether the eye moved right, left, up, or down.

This is the core asymmetry:

```text
To identify the pose-sensitive channel, the observer needs U_trans.
To recover physical retinal pose, the observer also needs the image-specific chart A(I).
```

That is why a simple global eye-position decoder can fail even when retinal pose is present in the population response.

## 5. Observer classes

This framework gives a cleaner set of observers than pose-aware, pose-blind, and a vague geometry-aware readout.

### 5.1 Known-eye observer

This observer is given the true eye trajectory $\tau_{1:T}$ and estimates the image or task variable.

For image inference:

$$
p(I\mid y_{1:T},\tau_{1:T})
\propto
p(y_{1:T}\mid I,\tau_{1:T})p_{\mathrm{nat}}(I).
$$

For a task variable $\theta$, such as Vernier offset or optotype orientation:

$$
p(\theta\mid y_{1:T},\tau_{1:T})
\propto
p(y_{1:T}\mid \theta,\tau_{1:T})p(\theta).
$$

This is the upper-bound observer. It asks what FEMs can provide if retinal pose is known.

### 5.2 Zero-eye observer

This observer assumes no eye movement:

$$
\tau_t = 0.
$$

It estimates image content or a task variable under the wrong retinal-pose model:

$$
p(I\mid y_{1:T},\tau=0).
$$

This is the pose-blind or wrong-model control. Motion-induced response changes become extra unexplained variability.

### 5.3 Joint geometry-aware observer

This observer does not know the eye trajectory. It knows, or has learned:

- the encoding likelihood,
- the compact translation channel $U_{\mathrm{trans}}$,
- the image-specific chart family $A(I)$,
- a natural-image or task prior,
- an FEM trajectory prior.

It estimates image content and retinal pose jointly:

$$
p(I,\tau_{1:T}\mid y_{1:T}).
$$

For a fine spatial task, it can estimate:

$$
p(\theta,\tau_{1:T}\mid y_{1:T}).
$$

The expected ordering is:

$$
\text{zero-eye}
<
\text{joint geometry-aware}
\lesssim
\text{known-eye}.
$$

The joint observer should not beat the known-eye observer. The key test is whether it recovers a substantial fraction of the known-eye advantage over the zero-eye observer.

## 6. Why not make full image reconstruction the main endpoint?

Full image reconstruction is natural for retina because retinal ganglion cells are the optic-nerve bottleneck. Reconstructing the image from retina asks what visual signal is transmitted to the brain.

V1 is different. V1 is already a transformed cortical representation. It builds selectivity, invariance, nonlinearities, and task-relevant distortions. A reviewer could reasonably argue that pixel reconstruction from V1 is the wrong endpoint.

The better V1 endpoints are fine-scale visual variables and feature-level information:

- Vernier offset,
- optotype orientation or position,
- local phase,
- edge position,
- high-spatial-frequency contrast,
- Gabor coefficients,
- steerable-pyramid coefficients,
- task-specific Fisher information or discriminability.

This preserves the Wu-style observer logic while avoiding the claim that V1 should invert back to pixels.

The strongest V1 question is:

> Does a joint geometry-aware observer recover fine spatial information from FEM-driven V1 response trajectories better than a zero-eye observer, and closer to a known-eye observer?

## 7. Why FEMs can help

A static image gives one response sample:

$$
\mu_0(I).
$$

A moving retinal image gives a response trajectory:

$$
\mu_0(I),
\quad
\mu_0(I)+J(I)\tau_1,
\quad
\mu_0(I)+J(I)\tau_2,
\quad
\ldots
$$

This trajectory can reveal local image structure. It samples how the V1 response changes under small retinal translations. For fine spatial variables, those derivatives can be informative.

In simple terms:

> FEMs expose local response derivatives of the image.

But this only helps if the observer can interpret the trajectory. If the observer knows the eye trace, interpretation is easier. If the observer does not know the eye trace, it needs priors and the compact geometry to infer image and pose jointly.

FEMs hurt when the observer cannot interpret the trajectory. Then the same motion-induced response changes appear as nuisance covariance around the stimulus-locked response.

This makes the pose-aware versus pose-blind asymmetry the point:

```text
The same activity is noise for a pose-blind observer and signal for a pose-aware or joint observer.
```

## 8. What is novel for V1

The non-derivative contribution is not simply applying the Wu observer to cortex. The V1-specific contribution is the compact, content-routed Jacobian:

$$
J(I) \approx U_{\mathrm{trans}}A(I).
$$

In retina, retinal jitter is close to a fixed geometric remapping across a relatively fixed sampling lattice. In V1, the same retinal displacement can have different effects depending on image content, nonlinear tuning, recurrent dynamics, and response history.

The important claim is that these image-dependent effects still share compact population support.

That gives a cortical coding principle:

> V1 exposes the consequences of self-generated retinal motion through a shared transformation channel whose coordinate meaning is image-specific.

This is not an explicit eye-position code. It is a compact likelihood geometry for self-motion-conditioned sensory inference.

## 9. Load-bearing risks and required gates

The joint-observer framing is stronger than the denoising framing, but it raises the evidentiary bar. The main risk is that compactness could be a property of the digital twin architecture rather than recorded V1.

Convolutional weight sharing and smooth model features can encourage compact translation effects. That does not make the result false, because V1 is itself retinotopic and locally filter-like. But it means the functional claim needs recorded-data anchors and controls.

### Gate 1: recorded compactness

Show that the FEM-linked component in recorded V1 is compact, reliable, and not just a model artifact.

Useful evidence:

- low participation ratio of recorded $\Sigma_{\mathrm{FEM}}$,
- split-half reliability of the FEM subspace,
- low-dimensional structure after eye conditioning,
- robustness across sessions.

### Gate 2: model-to-recorded alignment

Show that the model-derived translation geometry predicts the recorded FEM component better than controls.

Controls should include:

- random subspaces matched for dimension,
- gain-only or global-rate modes,
- unit-shuffled translation subspaces,
- eye-shuffled or image-shuffled controls,
- drift-only windows where the local linear approximation is valid.

### Gate 3: correct-chart versus wrong-chart

The key test of content routing is not merely that the correct image predicts better than the wrong image. A gain model can also be image-dependent because baseline responses change with image.

The clean chart-swap test holds the baseline image response fixed and swaps only the routing chart:

$$
\Delta \mu_{\mathrm{true}}
=
U_{\mathrm{trans}}A(I)\Delta\tau
$$

versus

$$
\Delta \mu_{\mathrm{swap}}
=
U_{\mathrm{trans}}A(I')\Delta\tau.
$$

If the correct chart predicts recorded response differences better than the swapped chart, while gain, norm, and subspace controls are matched, that supports content-dependent routing.

### Gate 4: external prior wall

The natural-image prior should not be trained from the V1 twin responses. It should be external, as in a denoiser trained on natural images or a hand-specified image/feature prior.

This wall is what prevents circularity.

### Gate 5: endpoint appropriate to V1

Use fine spatial information, not pixel reconstruction, as the main endpoint. Reconstruction can be a supplementary demonstration, but it should not be the central claim.

## 10. Practical analysis program

### Current manuscript

The current manuscript should establish the descriptive and geometric foundation:

1. FEMs account for a large fraction of foveal V1 shared variability.
2. Conditioning on eye position collapses classical noise correlations.
3. FEM-linked covariance is compact.
4. Digital-twin translation geometry predicts part of this compact recorded component.
5. The compact geometry is content-routed, not a universal eye-position coordinate.

This paper should avoid making strong claims that V1 denoises FEMs, that downstream circuits definitely use the geometry, or that real FEMs are globally optimal.

### Follow-up functional paper

The follow-up paper should ask what an observer can do with the geometry.

The central experiment:

```text
Compare zero-eye, known-eye, and joint geometry-aware observers on fine spatial variables.
```

Possible endpoints:

- Vernier offset discrimination,
- optotype orientation or position,
- local edge position,
- local phase,
- high-frequency image-feature recovery.

Success criterion:

```text
joint geometry-aware observer closes the gap between zero-eye and known-eye observers.
```

Stronger success criterion:

```text
the effect depends on the correct image-specific chart A(I), and disappears or weakens under wrong-chart, gain-only, and shuffled controls.
```

## 11. Simple algorithm sketch

The following is an abstract algorithm, not a final implementation.

1. Initialize an image or task-variable estimate.
2. Initialize an eye-trajectory estimate, perhaps zero or drawn from the FEM prior.
3. Use the twin to compute the predicted response and local chart:

   $$
   \mu_0(I), \quad A(I), \quad U_{\mathrm{trans}}.
   $$

4. Update the eye trajectory by fitting compact-channel residuals:

   $$
   z_t = U_{\mathrm{trans}}^\top(y_t-\mu_0(I))
   \approx
   A(I)\tau_t.
   $$

5. Penalize implausible eye trajectories using $p_{\mathrm{FEM}}(\tau_{1:T})$.
6. Update the image or task variable by maximizing the neural likelihood under the current eye trajectory.
7. Penalize implausible images using $p_{\mathrm{nat}}(I)$, or use a task prior $p(\theta)$ for fine spatial variables.
8. Iterate.
9. Score the final estimate against zero-eye and known-eye observers.

For a fine task variable $\theta$, the whole image need not be reconstructed. The observer can compute:

$$
p(\theta,\tau_{1:T}\mid y_{1:T})
\propto
p(y_{1:T}\mid \theta,\tau_{1:T})p(\theta)p_{\mathrm{FEM}}(\tau_{1:T}).
$$

This is often the more appropriate V1 endpoint.

## 12. Interpretation matrix

### Known-eye helps, joint helps

Interpretation:

> FEMs create useful samples, and V1 activity contains enough geometry to infer retinal pose well enough to use them.

This is the strongest functional result.

### Known-eye helps, joint fails

Interpretation:

> FEMs create useful samples when pose is known, but the tested observer cannot infer pose from V1 responses under the current noise model, prior, or geometry estimate.

This would still support the coordinate-frame story, but not the joint-observer claim.

### Known-eye does not help

Interpretation:

> The task, stimulus regime, motion scale, or readout is probably not in the FEM-beneficial regime.

This should push the analysis toward finer spatial variables, more naturalistic stable viewing, or different temporal windows.

### Zero-eye worsens with motion

Interpretation:

> The observer is treating reafferent structure as noise. This is expected and supports the pose-blind cost side of the framework.

### Joint works only with the correct chart

Interpretation:

> The compact geometry is not merely a gain or low-rank covariance effect. The image-specific routing matrix matters.

## 13. Wording to use

Strong but defensible:

> Fixational eye movements drive V1 through a compact, content-routed translation geometry. In screen coordinates, this activity appears as shared variability. For an observer with knowledge of the compact geometry, natural-image statistics, and plausible FEM statistics, the same response trajectory can provide evidence for jointly estimating image content and retinal pose.

Also safe:

> The compact geometry is not an explicit eye-position code. It is a low-dimensional likelihood structure for the sensory consequences of retinal motion.

Safe claim boundary:

> We do not claim that V1 globally optimizes fixational eye movements, or that downstream circuits explicitly implement this observer. The result identifies a representational geometry that makes such joint inference possible in principle and testable with known-eye, zero-eye, and joint-observer comparisons.

Avoid:

```text
V1 denoises FEMs.
V1 packages self-motion for downstream areas.
V1 explicitly encodes eye position.
The compact subspace solves the pose problem by itself.
Real FEMs are optimal.
```

## 14. Plain-language version

Every tiny eye movement slides the image across the retina. V1 responds to that retinally shifted image. If we ignore the eye movement, the resulting response changes look like neural noise. If we account for the eye movement, they become structured sensory reafference.

The compact-geometry result says that these reafferent effects are not arbitrary. The effect of a small image shift depends on the image, but across images those effects occupy a shared low-dimensional channel in the V1 population.

That shared channel is not an eye-position readout. The same activity pattern can mean different physical displacements for different images. To recover eye position, an observer needs the current image-specific chart. But the compact channel still matters because it makes the image-and-eye inference problem much smaller.

With a natural-image prior and an eye-movement prior, an observer can ask which image and which eye trajectory together best explain the V1 response. This is the V1 analogue of the Wu-style retinal observer, but with a cortex-appropriate endpoint such as fine spatial discriminability rather than full pixel reconstruction.

The core idea is that FEMs are not always signal and not always noise. They create response trajectories. Those trajectories are nuisance for an observer that ignores retinal pose, but they can become useful evidence for an observer that knows or infers retinal pose using compact geometry and natural priors.

## 15. One-sentence thesis

Compact reafferent geometry is the cortical likelihood structure that can turn FEM-induced V1 shared variability from pose-blind noise into pose-conditioned evidence for fine image structure.
