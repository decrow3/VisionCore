# Figure 4B working methods and interpretation

## Overview

The current Figure 4B analysis uses a held-out feature decoder. The decoder has a training and scoring rule that can be interpreted as an infomax-compatible estimate, but it is not a measure of total information in the population. A total population-information estimate was the original goal, but it repeatedly ran into implementation and interpretation problems.

The first attempt was a spatial single-spike information analysis over the model readout grid. This asked how spatially structured, or spatially peaky, the model activation maps became under different eye-movement conditions. A spatially peaky activation map means that a particular unit is responding strongly to a local image feature at a particular location.

The problem is that the simple population version of this estimator sums information across units as if their activation maps were independent. If two units have identical responses, the simple estimator counts both, even though the second unit adds no new information. Within a unit’s activation map, the sparsity measure penalizes uniform, nonselective activity. Across units, however, redundancy is not penalized. As a result, the population SSI score can be inflated when many units become active or spatially peaky in similar ways. In practice, this metric monotonically preferred larger-scale motions beyond the natural scale of fixational eye movements.

One possible fix would be to reduce the population to a nonredundant set of units, or to build a covariance-aware estimator that accounts for shared structure across units, analogous to accounting for noise correlations. I explored this direction, but it added complexity while leaving the encoded visual variable somewhat implicit. The simpler direction was to ask directly whether the model response helps recover a specified visual feature target.

That is the motivation for the feature decoder. Instead of asking whether activation maps contain more spatially organized activity, the decoder asks whether the population response contains information useful for predicting local image content. In principle, the target could be the local image patch itself, in pixels. In practice, I use a coarser and more statistically stable target: block-pooled pyramid features, including signed quadrature components and local energy across scale and orientation.

This also gives a cleaner information-theoretic framing. The decoder predicts the feature target from the model response, and the held-out residuals estimate how much uncertainty about that target remains. Under a linear-Gaussian decoder, this residual uncertainty can be converted into an infomax-compatible score: a variational lower bound on feature mutual information. Thus, the decoder does not merely ask whether motion makes responses larger or activation maps peakier. It asks whether motion makes the response more informative about a defined feature target.

There is an important technical caveat. The decoder gives a lower bound on feature information within each condition. The motion effect is a difference between these estimated bounds. That difference is best described as a motion-induced change in the estimated feature-information bound, not as a mathematically guaranteed lower bound on the true information gain, because the difference between two lower bounds need not itself be a lower bound.

Across the model variants tested so far, some implementations are brittle, but two conclusions have been fairly robust. First, motion-rendered responses add recoverable feature evidence relative to a counterfactual stabilized baseline when the response is summarized on the trajectory-family axis used by the aggregate decoder (Figure 4B). The same-axis pose-unaware hidden-sample proxy has negative point estimates relative to static, so the motion benefit should be interpreted as conditional on how trajectory-induced structure is made available to the readout. Second, motion along local contours appears to help more than motion across them, at least in the analyses summarized for Figure 4D. The endpoint here is recoverable feature evidence in a V1-twin response, not a behavioral claim that FEMs should move parallel to contours.

There are three broad implementations of these model analyses. This document focuses on the first, aggregate implementation of the feature decoder. In this analysis, fixation trajectories and backimage ROI windows are pooled across all backimage trials to build training and test sets. Real fixations are not paired with their original ROIs. Therefore, this analysis can only test whether the aggregate statistics of measured FEMs are well matched to the aggregate statistics of natural-image structure. It cannot test whether each specific trajectory is specially matched to the specific image patch that was viewed.

The second implementation addresses that limitation by testing whether real image-trajectory pairings outperform matched trajectory swaps. That analysis asks whether there is information in the specific coupling between a local image patch and the trajectory that sampled it. At present, this should be treated as diagnostic or supplemental unless it proves stable across seeds and manifests.

The third implementation is the joint decoder model, inspired by the Wu et al. observer framing. In that analysis, the model jointly infers image features and eye trajectory. This connects naturally to the compact subspace geometry in the lower row of Figure 3: if eye-position-dependent response changes live in a compact geometry, then a downstream observer may be able to jointly infer pose and content rather than requiring perfect external knowledge of eye position. That connection is potentially important, but the causal dependence of the joint decoder on the compact geometry still needs explicit ablation, so it may be out of scope for this paper.

## 1. Spatial SSI, for reference

Let the deterministic model rate map be

$$
r_{tnp}\ge 0,
$$

where $t$ indexes time, $n$ indexes neurons or model units, and $p$ indexes positions on the spatial readout grid. If the readout grid has $P$ positions, the mean rate over position is

$$
\hat{r}_{tn}=\frac{1}{P}\sum_{p=1}^{P}r_{tnp}.
$$

Define the normalized spatial gain map

$$
g_{tnp}=\frac{r_{tnp}}{\hat{r}_{tn}+\epsilon}.
$$

The spatial single-spike information for unit $n$ at time $t$ is

$$
I_{tn}^{SSI}=\frac{1}{P}\sum_{p=1}^{P}g_{tnp}\log_{2}(g_{tnp}+\epsilon).
$$

This is the expected information, in bits per spike, carried by a spike from unit $n$ about the grid index $p$, assuming a uniform prior over grid positions. This “position” reading is literal only in the classical case where one unit has a tuning map over a genuine position latent. For the dense multi-unit readout here, it is better understood as a peakedness functional over feature-tuned activation maps. A peak means that a feature-tuned unit responded strongly to some local image content at some location.

To get the population quantity, weight each unit by its expected number of spikes. For bin width $\Delta t$,

$$
m_{tn}=\hat{r}_{tn}\Delta t.
$$

The total expected spatial information over the response movie is

$$
B_{SSI}=\sum_{t,n}m_{tn}I_{tn}^{SSI}.
$$

The population bits per spike are

$$
I_{spike}^{SSI}=\frac{\sum_{t,n}m_{tn}I_{tn}^{SSI}}{\sum_{t,n}m_{tn}+\epsilon}.
$$

The corresponding information rate is

$$
I_{rate}^{SSI}=\frac{\sum_{t,n}m_{tn}I_{tn}^{SSI}}{T\Delta t}.
$$

Thus, SSI already has an efficiency form:

$$
I_{spike}^{SSI}=bits per expected spike.
$$

This matters for interpretation. A gain in $I_{rate}^{SSI}$ can reflect either better spatial selectivity or more total response. A gain in $I_{spike}^{SSI}$ is closer to an efficiency claim, but it still does not correct for redundancy across units.

## 2. What SSI does and does not test

The SSI calculation is valuable because it is simple, familiar, and computed straight from the model’s own rate maps. It asks whether the response map becomes sharper, or more peaked across the readout grid, under real eye motion.

However, SSI does not directly test whether eye movements improve recovery of image content. It is a peakedness functional of the map, not a decode of any explicitly named visual variable. The index it sums over is the grid coordinate:

$$
p=readout-grid index.
$$

The decoder instead treats the stimulus variable as an image-feature vector:

$$
\Phi =ϕ(I).
$$

So the two analyses differ in what is explicit:

$$
SSI: dispersion or peakedness over readout-grid locations in feature-tuned maps,Decoder: prediction of an explicit local image-feature vector \Phi.
$$

This is the main reason to move beyond SSI, but it is not the only one. Two assumptions are buried in the metric. Within a unit, $glog(g)$ rewards peakiness, which equates spatial concentration with informativeness, a sparse-coding premise that a trained, unregularized readout need not satisfy. Across units, the population form sums per-unit values with no cross term, so it cannot see redundancy: a duplicated unit contributes twice even though it adds nothing. The per-spike normalization caps the magnitude of the sum but does not remove redundant contributions. Nothing in the collapsed population SSI scalar makes redundant variance stop helping.

A held-out feature decoder addresses this by naming the target and crediting only variance that improves prediction on images it has not seen. This does not mean the decoder magically solves every correlation problem. Redundant or correlated response features can still affect the ridge geometry, especially if feature scaling and regularization are not handled carefully. In particular, ridge regression is not invariant to duplicated standardized columns: duplicated correlated units can effectively reweight a response direction under fixed $\lambda$. But the held-out score does not assume independence across units, and redundant activity only helps if it improves prediction of the target under the regularized readout.

An earlier E-optotype analysis exposed exactly this issue. A direct acuity claim requires an identity variable, for example E orientation or gap direction, not just spatial structure in the activation map. The pyramid decoder generalizes that identity-readout logic from optotypes to natural-image features.

$$
SSI→implicit feature-position structureE optotype→acuity-limit identity informationPyramid decoder→natural-image feature information.
$$

This also turns the comparison with SSI into a fair test rather than a dismissal. The current path does not pit the collapsed SSI scalar against the decoder. It feeds the per-unit SSI values, ssi_itn and its delta and rate-augmented variants, into the same population decoder, on the same bits axis, against the response summaries. All decoder inputs are z-scored using train-fold statistics, so SSI features are scaled comparably to the response summaries. Read this way, the redundancy objection is reduced because correlated SSI features are treated like any other correlated decoder input. What is left is a clean question: does the nonlinear peakedness transform expose image-feature content that the rate and temporal summaries miss? The current answer is that it mostly does not: in the $n=128$ SSI-combined run, adding high-dimensional ssi_itn features generally worsened held-out $-MSE$, while delta_ssi_unit_mean produced only weak or mixed incremental gains.

## 3. Decoder target: local pyramid features

For each local cropped image patch $I_{i}$, define a feature target

$$
\Phi_{i}=ϕ(I_{i})\in R^{d}.
$$

The target is a multiscale, multiorientation pyramid representation. Let

$$
P_{so}(I_{i},x)\in C
$$

be the complex pyramid coefficient at scale $s$, orientation $o$, and spatial location $x$. For each complex orientation plane, the analysis first computes three coefficient maps:

$$
ReP_{so}(I_{i},x),ImP_{so}(I_{i},x),∣P_{so}(I_{i},x)∣.
$$

The magnitude is therefore computed pointwise at each pyramid coefficient before any spatial pooling.

For each local pooling block $Ω_{b}$, the feature vector contains the block average of each map:

$$
ϕ_{isob}^{Re}=\frac{1}{∣Ω_{b}∣}\sum_{x\in Ω_{b}}ReP_{so}(I_{i},x),ϕ_{isob}^{Im}=\frac{1}{∣Ω_{b}∣}\sum_{x\in Ω_{b}}ImP_{so}(I_{i},x),ϕ_{isob}^{Mag}=\frac{1}{∣Ω_{b}∣}\sum_{x\in Ω_{b}}∣P_{so}(I_{i},x)∣.
$$

Equivalently, the target is

$$
\Phi_{i}=concat_{s,o,b}[mean_{x\in Ω_{b}}ReP_{so}(I_{i},x),mean_{x\in Ω_{b}}ImP_{so}(I_{i},x),mean_{x\in Ω_{b}}∣P_{so}(I_{i},x)∣].
$$

The important point is that the magnitude feature is

$$
mean_{x\in Ω_{b}}∣P_{so}(I_{i},x)∣,
$$

not

$$
∣mean_{x\in Ω_{b}}P_{so}(I_{i},x)∣.
$$

Thus, the magnitude channel preserves local oriented energy within each block, rather than allowing phase variation inside the block to cancel before the magnitude is computed.

This gives the target a clean V1-adjacent interpretation:

$$
ReP,ImPcapture phase-sensitive oriented structure,∣P∣captures phase-insensitive local oriented energy.
$$

The decoder target is therefore not a pixel reconstruction. It is a compact local-feature description of each image patch, organized by scale, orientation, phase-sensitive quadrature components, phase-invariant magnitude, and spatial pooling block.

### 3.1 Phase and pooling: what the target keeps, and the cost of finer grain

A caveat on “phase-sensitive,” since it is easy to over-read. The block average bites only the signed channels: the magnitude is pooled after the modulus, so it never carried local phase to begin with. Averaging signed Re and Im over a block averages complex coefficients, so phase that rotates within the block partially cancels. The target therefore keeps phase that is coherent at or above the block scale and discards the within-block phase layout. The loss is scale-dependent: a band’s coefficient phase varies on that band’s period, which is large relative to an $8\times 8$ block at coarse scales and small at fine scales. So the loss is mild at low spatial frequency and heaviest at high spatial frequency.

Two consequences follow. First, this is a ceiling on the target, not on the code: if the response carries fine within-block phase, the decoder cannot be credited for it because the target has already discarded it. A null on fine phase is uninterpretable here. Second, it compounds with the rendering ceiling: the finest scales are the least trustworthy on two independent counts.

Cross-band phase alignment is a separate matter, and it is a decoder limit rather than a binning limit. Per-band pooling does not remove cross-band alignment information: each band’s coarse phase is present in the concatenated vector. But local phase congruency is a relative-phase, multiplicative quantity, and a linear ridge cannot form the phase difference between two bands from their concatenated means. So if cross-band alignment is part of the claim, the fix is decoder capacity or explicit phase-congruency features, not smaller blocks. Shrinking the block recovers within-block phase; it does not, on its own, make cross-band alignment readable.

Recovering within-block phase means shrinking the block or dropping the pooling, and the cost is steep. The current target pyramid_local_field is $(384)$: 4 pyramid levels × 4 orientations × 3 summaries (Re, Im, magnitude) × 64 $8\times 8$ bins. The $8\times 8$ averaging is a post-review latent-target patch committed on 2026-06-15; earlier runs effectively used $4\times 4$. A full native downsampled pyramid at $128\times 128$ is 87,040 complex coefficients, which is 261,120 values under the current Re + Im + magnitude convention, approximately 85 times the present target, or 174,080 with invertible Re + Im alone, approximately 57 times the present target.

One nuance worth recording: downsample=True only helps if the pooling is actually relaxed. Keep the $8\times 8$ average after the pyramid and the count stays 3,072 regardless. So the savings are not free dimensionality reduction; they are conditional on giving up the block average.

The binding constraint is not compute but statistics. With 384 images, a roughly 100,000-dimensional target makes per-dimension held-out residuals noisy and the full-covariance log-det information form unusable, forcing the diagonal approximation and degrading the bits estimate. So finer pooling buys within-block phase but not cross-band alignment, and it trades a clean information estimate for a noisy one unless the image count grows with the target. If fine phase matters, the honest middle ground is to refine pooling only at the scales where it is lost, especially the finest bands where the period is small relative to the block, and to add explicit cross-band features if alignment is the target, rather than going to a full pixel or full-pyramid reconstruction that the image count cannot support.

### 3.2 Target scaling and covariance treatment

The target contains heterogeneous components: Re, Im, magnitude, scale, orientation, and spatial block. These components can have very different variances. In aggregate decoding, both decoder inputs $X$ and feature targets $Z$ are z-scored using train-fold statistics only. Feature PCA is then fit on the train fold and applied to the held-out fold. This makes the held-out decoder score much cleaner than raw, unstandardized MSE across heterogeneous pyramid components.

Formally, the preprocessing is

$$
\Phi →\hat{\Phi},
$$

where mean subtraction, z-scoring, and feature PCA are fit on the training set and applied to the held-out set.

This matters especially for the log-det interpretation. A scalar MSE score corresponds roughly to an isotropic Gaussian residual model. A per-feature variance ratio corresponds to a diagonal Gaussian residual model. A full log-det score requires a full residual covariance estimate.

The current main aggregate CSVs report scalar held-out $-MSE$ and $R^{2}$ in PCA feature space. The information posthoc promotes the diagonal Gaussian residual-variance ratio:

$$
\Delta \hat{I}_{bits}=\frac{1}{2}\sum_{j}\log_{2}(\frac{var_{0,j}}{var_{c,j}}).
$$

This is the appropriate headline bits form for the current implementation. The posthoc also writes a Ledoit-Wolf full-covariance log-det, but that should be treated as supplementary rather than headline.

Limitation: this diagonal form does not penalize correlated residual structure and is therefore not commensurable with the constrained discriminability metric (d-prime-squared with Sigma-inverse). See Section 17 for the covariance-whitened loss that bridges this gap, and Section 16 for why the gap matters for interpreting the along/across result.

At the raw target dimensionality, $d=3072$ and $n\approx 384$ images, a naive full residual covariance is rank deficient. In practice, the decoder uses reduced PCA target dimensions, for example $k=16$, so a full-covariance residual estimate is feasible only in this reduced target space and only with shrinkage. Full-covariance log-det over raw $d=3072$ should not be the headline.

## 4. Response summaries

Let the model response movie for image $I_{i}$ under trajectory $\tau_{i}$ be

$$
R_{i}(\tau_{i})=f_{\theta}(I_{i},\tau_{i}).
$$

We compare a stabilized reference to motion-driven responses:

$$
R_{i}^{0}=f_{\theta}(I_{i},\tau_{0}),R_{i}^{\tau}=f_{\theta}(I_{i},\tau_{i}).
$$

A response summary function $\psi (⋅)$ maps each response movie into a finite vector:

$$
S_{i}=\psi (R_{i}).
$$

Different summaries define different readout channels. Examples include:

$$
\psi_{mean}(R),\psi_{\Delta mean}(R),\psi_{PCA}(R),\psi_{DCT}(R).
$$

These summaries are intentionally lossy. A positive decoder result is therefore conservative for the existence of feature information in the full response movie. A null result is ambiguous because information may exist in response structure discarded by $\psi$.

### 4.1 Mean versus delta-mean: two readout hypotheses

The mean and delta-mean summaries should not be interpreted as competing

versions of the same claim. They answer different questions.

Let R0(I) be the mean response to the stabilized image, and let Rm(I, tau) be

the mean response to the same image rendered under a motion trajectory. The

mean readout uses Rm directly. This is an absolute response readout: it asks

whether the average response to the motion-rendered movie is useful for

recovering image features.

The delta-mean readout uses

Delta R(I, tau) = Rm(I, tau) - R0(I).

This is a static-subtracted motion component. It asks whether the change induced

by motion contains feature evidence beyond the static response.

This distinction matters for interpretation. A positive mean result can reflect

a good feature code in the motion-rendered average response, but it does not by

itself isolate the part of the code caused by motion. A positive delta-mean

result is more diagnostic for the active-sensing claim because the static

feature evidence has been subtracted before decoding.

The biological interpretation is also different. The mean readout is closer to

what a downstream area directly observes: absolute activity during the movie.

The delta-mean readout is an analysis contrast, or a possible normalized /

predictive readout, that isolates motion-induced response changes. Therefore,

mean is the stronger absolute readout candidate, while delta-mean is the cleaner

mechanistic bridge to the claim that motion adds feature information.

### 4.1 Response-summary leakage audit

Response summaries differ in leakage risk. Mean, delta-mean, and DCT summaries are cleaner checks because they do not require fitting a response PCA basis from the rendered response movies. Temporal response PCA is more delicate: in the current aggregate implementation, the temporal response PCA basis is fit globally across rendered response movies before decoder cross-validation. This means temporal PCA should not be the only load-bearing readout unless it is rerun with fold-local PCA.

For internal audit, record:

- whether PCA is fit within each training fold or globally;

- whether the same PCA basis is used across static and motion conditions;

- whether response features are standardized using train-only statistics;

- whether regularization $\lambda$ is chosen by inner cross-validation or fixed;

- whether the same folds and preprocessing are used across all motion conditions.

In the current aggregate runs, ridge $\alpha =10.0$ is fixed and shared across conditions, which is good: motion gains are not differential-alpha artifacts.

## 5. Two response conditions

The analysis separates two response conditions.

### 5.1 Stabilized reference

The stabilized condition asks how much feature information is recoverable without retinal motion:

$$
S_{i}^{0}=\psi (R_{i}^{0}).
$$

This is the baseline for asking whether motion adds information.

### 5.2 Motion-rendered / trajectory-conditioned response

The motion condition asks how much feature information is recoverable from responses rendered under empirical or empirical-like trajectories:

$$
S_{i}^{\tau}=\psi (R_{i}^{\tau}).
$$

The trajectory $\tau_{i}$ is used to render the model response movie, but in the aggregate Figure 4B implementation it is not an explicit input to the ridge decoder. The decoder receives only response summaries such as mean, delta-mean, temporal PCA, and DCT. Therefore, “pose-aware decoder” is too literal for aggregate Figure 4B.

The best description is:

$$
motion-rendered response
$$

or

$$
trajectory-conditioned twin response.
$$

The idealized pose-aware readout would be

$$
q(\Phi_{i}∣S_{i}^{\tau},\tau_{i}),
$$

but this is not what the aggregate decoder implements. The stronger pose-aware observer, where the readout explicitly conditions on or marginalizes over trajectories, belongs to the joint-decoder implementation.

See Section 15 for a formal treatment of the observer hierarchy (pose-aware, pose-hidden, pose-jointly-inferred) and why the aggregate decoder’s position within this hierarchy determines the interpretation of every number it produces.

Thus, the aggregate Figure 4B claim should be worded as:

empirical motion increases feature decoding in motion-rendered responses relative to stabilized input.

not as:

a pose-aware decoder recovers more feature information.

## 6. Ridge decoder

For each condition $c\in {0,\tau}$, we fit a linear decoder

$$
\hat{\Phi}_{i}^{c}=g_{c}(S_{i}^{c})=W_{c}S_{i}^{c}+b_{c}.
$$

The ridge objective is

$$
(\hat{W}_{c},\hat{b}_{c})=argmin_{W,b}\sum_{i\in D_{train}}∥\Phi_{i}-(WS_{i}^{c}+b)∥_{2}^{2}+\lambda ∥W∥_{F}^{2}.
$$

The current aggregate runs use fixed ridge $\alpha =10.0$, shared across conditions. This is important because it prevents the motion gain from being explained by condition-specific regularization choices.

Train and test splits are grouped by decode_group_mode=image. In the current implementation, this means selected window/source row, not necessarily original trial or full backimage canvas. Therefore, the split tests generalization to held-out selected windows, but not necessarily to fully independent source images.

This is the key methodological upgrade relative to the collapsed SSI scalar. SSI is computed directly from deterministic rate maps. The decoder only credits information that predicts held-out feature targets.

### 6.1 Train/test grouping and sample definition

A sample is approximately

$$
sample=(ROI window,source row,source trial,trajectory,motion scale,response summary,feature target).
$$

The current grouping is window/source-row based. It is not strict source-trial or backimage-canvas grouping.

This matters because crops from the same source trial can appear in both train and test. In the current $n=128$ aggregate run, 128 windows come from 110 unique $(session)$ keys; 12 source trials repeat, with a maximum of 5 crops from one trial. In the $n=384$ run, 384 windows come from 260 source trials, with up to 12 crops from one source trial. This can inflate apparent generalization if nearby crops share image statistics.

Robustness requirement: rerun the aggregate decoder with strict source-trial grouping before promotion. If the motion-induced information gain is stable under that split, report the current window/source-row grouped result together with the strict source-trial robustness check. If the gain shrinks materially, the strict source-trial split should become the primary estimate and the current grouping should be labeled as an optimistic/provenance analysis.

Implementation update: the strict source-trial grouped n=384 information-axis rerun is now complete and becomes the primary Panel B estimate. Empirical delta-mean information gain over stabilized is +1.09/+1.15/+0.98/+0.69/+0.72 bits at 0.25/0.5/1/1.5/2x, with point-centered decode-bootstrap CIs. The image/window grouped comparison is retained as optimistic provenance context.

Pose-unaware implementation update: the same source-trial grouped information-axis hidden-sample proxy is now complete. The pose-unaware hidden-sample trace is -0.53/-0.40/-0.60/-0.62/-0.49 bits at 0.25/0.5/1/1.5/2x, while the hidden-sample-minus-known penalty is -1.50/-1.54/-1.46/-1.34/-1.23 bits. This preserves the main conditional interpretation: motion-rendered responses carry feature evidence when trajectory-shaped structure is available through the rendered response summary, but hiding the trajectory sample converts much of that structure into nuisance variability. This is a same-axis proxy, not yet the full covariance-aware pose-hidden observer from Section 17.

The same trajectories may also appear across train and test at the trace-bank level. This is acceptable for the aggregate trajectory-family test because the question is not strict new-trajectory generalization. But it should be stated: aggregate Figure 4B tests the usefulness of trajectory-family statistics, not held-out image/trajectory pairing.

Within a run, static, empirical, Brownian, OU, and rotated conditions share the same selected image rows and latent targets. This is good: condition comparisons are same-manifest comparisons.

### 6.2 Matched decoder capacity

All condition comparisons should use matched folds, matched response dimensionality, matched target preprocessing, matched response preprocessing, and a matched regularization policy. In the current aggregate runs, $\alpha =10.0$ is fixed and shared across conditions, so the main motion gains are not differential-$\lambda$ artifacts.

All decoder input columns are train-fold z-scored. This makes SSI, mean, delta-mean, DCT, and PCA inputs more comparable. However, high-dimensional SSI concatenations can still change ridge behavior through dimensionality and correlation structure. Exact duplicates or strongly correlated columns are not automatically neutral under fixed ridge; they can reweight a response direction.

There is a subtle linear-algebra caveat. In an unregularized linear decoder,

the feature spaces [R0, Rm] and [R0, Delta R] span the same information, since

Delta R = Rm - R0. With finite data, train-fold z-scoring, correlated columns,

and fixed ridge regularization, these parameterizations are not equivalent.

Thus, differences between mean and delta-mean scores should be interpreted as

differences in readout parameterization and regularization geometry, not as

proof that one contains fundamentally different information from the other.

This is another reason to give them distinct roles rather than declare a single

winner.

## 7. Decoder as a variational information estimate

The decoder can be interpreted as a Gaussian variational approximation to the posterior over features:

$$
q_{c}(\Phi ∣S)=N(\Phi ;W_{c}S+b_{c},\Sigma_{c}).
$$

The residual covariance on held-out data is

$$
\Sigma_{c}=Cov_{i\in D_{test}}[\Phi_{i}-\hat{\Phi}_{i}^{c}].
$$

By the Barber-Agakov variational bound,

$$
I(\Phi ;S)\ge H(\Phi)+E_{\Phi,S}[logq(\Phi ∣S)].
$$

For the Gaussian decoder, this becomes

$$
\hat{I}_{c}=H(\Phi)-\frac{1}{2}logdet(2\pi e \Sigma_{c}).
$$

The entropy term $H(\Phi)$ is the same across decoder conditions, so information differences are determined by residual covariance:

$$
\Delta \hat{I}_{c}=\hat{I}_{c}-\hat{I}_{0}=\frac{1}{2}[logdet\Sigma_{0}-logdet\Sigma_{c}].
$$

For the current headline bits estimate, use the diagonal Gaussian residual variance ratio:

$$
\Delta \hat{I}_{bits,c}=\frac{1}{2}\sum_{j}\log_{2}(\frac{var(\epsilon_{0,j})}{var(\epsilon_{c,j})}).
$$

A scalar MSE approximation is cruder:

$$
\Delta \hat{I}_{c}\approx \frac{d}{2}\log(\frac{MSE_{0}}{MSE_{c}}),
$$

where $d$ is the dimensionality of the decoded feature space.

This is the clean infomax version of the decoding score. It turns improved held-out feature prediction into an estimated lower-bound change in feature information.

However, the motion-static comparison is a difference between estimated lower bounds. Each condition has its own approximation gap:

$$
I_{c}=\hat{I}_{c}^{LB}+gap_{c}.
$$

Therefore,

$$
I_{\tau}-I_{0}=(\hat{I}_{\tau}^{LB}-\hat{I}_{0}^{LB})+(gap_{\tau}-gap_{0}).
$$

The individual bounds are conservative estimates of feature information in each condition. The difference is best described as a motion-induced change in the estimated feature-information bound, not a guaranteed lower bound on the true information gain unless the approximation gaps are matched.

### 7.1 Deterministic rates versus sampled responses

The aggregate Figure 4B decoder uses deterministic mean rates from the digital twin. The CanonicalTwinScorer computes rate maps and spatially max-pools them; no Poisson sampling is used in the aggregate 4B path. Therefore, $\Sigma_{c}$ is not biological response noise. It is held-out feature-prediction error across images under the decoder. The Gaussian residual model is a model of decoder uncertainty about features, not spike-count noise.

If future variants use sampled rates, Poisson spike samples, or Gaussian-head samples, the residual will contain both feature-prediction error and sample noise, and the interpretation must be updated.

## 8. Main predicted pattern

The panel claim can be written as

$$
\Delta \hat{I}_{\tau}>0.
$$

The inequality means that motion-rendered responses contain more recoverable feature information than stabilized responses under matched decoder capacity.

In plain language:

$$
motion-rendered twin responses add recoverable feature evidence.
$$

For internal audit, define success as follows:

Primary success: positive held-out diagonal information gain over static for a V1-plausible feature target, fixed $\alpha =10.0$, strict source-trial grouped CV, same manifest, and train-fold target/input scaling. The image/window grouped result is retained as optimistic provenance context.

Secondary success: empirical motion beats Brownian, OU, rotated, or matched-random controls on the same information metric.

Efficiency success: the motion advantage remains interpretable after reporting response cost $\Delta C$. Use $\Delta I/\Delta C$ only with sign and denominator guardrails.

Specificity success: local pairing beats matched trajectory swaps, or the joint decoder recovers a pose-content advantage.

2026-06-30 update: the local pairing audit weakened the unconditional exact-pairing claim and exposed inherited decoder-contract issues. The joint decoder/observer branch now has a cache-level inherited-contract audit. The promoted 4C calibration gate is source-row-heldout feature cosine; posterior math, source identity, supervised source-row folds, and validated point/CI contrast files pass. A positive joint result should still be reported as 4C feature recovery unless it is recomputed on the same diagonal-information axis as 4B.

## 9. Relation to Fisher information

The same intuition can be written locally. Suppose the mean response depends on eye position $e$:

$$
\mu (I,e)\approx \mu (I,e_{0})+J_{e}(I)\delta e.
$$

If eye position is hidden, the movement-induced response covariance is approximately

$$
\Sigma_{FEM}(I)\approx J_{e}(I)\Sigma_{e}J_{e}(I)^{⊤}.
$$

For a feature parameter $\alpha$, define the feature-response sensitivity

$$
G_{\alpha}=\frac{∂\mu}{∂\alpha}.
$$

When pose is known, the local Fisher information is approximately

$$
F_{known}=G_{\alpha}^{⊤}\Sigma_{noise}^{-1}G_{\alpha}.
$$

Thus, the Fisher and decoder views express the same broader point: when trajectory is known or inferable, retinal motion can act as structured sampling rather than added covariance. But the aggregate Figure 4B decoder does not explicitly provide $\tau$ to the readout. Therefore, this Fisher section should be treated as conceptual scaffolding, not a literal description of the aggregate decoder.

See Section 16 for the connection between this Fisher framework and the constrained discriminability metric used in the Vernier analysis, and why the aggregate decoder’s ridge-MSE loss silently dropped the nuisance penalty (Sigma-inverse) that made the Vernier metric principled.

The complementary hidden-pose case, in which the motion-induced covariance enters the denominator,

$$
F_{hidden}=G_{\alpha}^{⊤}(\Sigma_{noise}+\Sigma_{FEM})^{-1}G_{\alpha},
$$

is modeled explicitly in the joint-observer analysis rather than in the aggregate Figure 4B decoder.

## 10. Efficiency: separating information from response drive

A positive absolute information gain is not by itself an efficiency claim. Motion could improve decoding simply by increasing total activity. Therefore, report both the absolute gain and the response cost.

The absolute gain is

$$
\Delta \hat{I}_{c}=\frac{1}{2}[logdet\Sigma_{0}-logdet\Sigma_{c}],
$$

or, in the headline diagonal-bits form,

$$
\Delta \hat{I}_{bits,c}=\frac{1}{2}\sum_{j}\log_{2}(\frac{var(\epsilon_{0,j})}{var(\epsilon_{c,j})}).
$$

Let the expected response cost be

$$
C_{c}=\sum_{i,t,n}\mu_{itn}^{c}\Delta t.
$$

The added response cost relative to the stabilized condition is

$$
\Delta C_{c}=C_{c}-C_{0}.
$$

The efficiency metric is

$$
\eta_{c}=\frac{\Delta \hat{I}_{c}}{\Delta C_{c}}.
$$

Depending on the model output, $C_{c}$ can be total expected spikes, total activation, or matched-rate response energy. The interpretation should be explicit:

$$
\Delta \hat{I}_{c}asks whether motion adds feature information,\Delta C_{c}asks whether motion changes response cost,\eta_{c}asks whether motion adds feature information per unit added response cost.
$$

Internal caveat: $\eta_{c}$ can be unstable if $\Delta C_{c}$ is near zero. If $\Delta C_{c}<0$, “per added spike” is not meaningful. Therefore, report $\Delta \hat{I}$ and $\Delta C$ first. Use $\Delta I/\Delta C$ only as a secondary efficiency axis with sign and denominator guardrails. Consider reporting $I/C$ as well if response cost changes weakly.

## 11. Motion-family and scale controls

The motion conditions need an explicit matching ledger. For each motion family, report the matching criterion and effective statistics after clipping or rendering.

Motion families to track:

- empirical FEM;

- Brownian;

- OU / confined drift;

- rotated drift;

- static / stabilized;

- drift-only versus drift plus microsaccades, if relevant.

Matching variables to report:

- RMS displacement;

- path length;

- speed;

- lag-1 autocorrelation;

- clipping fraction;

- source trajectory manifest;

- source ROI manifest.

The current aggregate runs write these values to aggregate_motion_summary.csv. They include RMS, path length, speed, lag-1 autocorrelation, and clipped fraction. They do not fully report velocity covariance or temporal spectrum in the main aggregate output.

The current $n=128$ and $n=384$ aggregate runs show clipped_fraction=0.0 and median effective/requested RMS equal to 1.0. Therefore, for these runs, nominal motion scale is not misleading due to clipping.

These aggregate runs are effectively drift-only. Source traces require max_trace_source_microsaccade_events=0 with z-threshold microsaccade detection, and all motion families use that same filtering.

The natural-scale claim should remain cautious. Aggregate gains do not cleanly peak at natural $1\times$; several motion-delta effects continue to grow at $2\times$. Therefore, do not claim a natural-scale optimum unless a specific panel shows it.

## 12. How this should be reported

The main figure should emphasize the simple contrast:

$$
motion-rendered / trajectory-conditioned response>stabilized response.
$$

The methods or supplement should report:

- the stabilized reference;

- the motion-rendered trajectory-family condition;

- that $\tau$ is used to render responses but is not an explicit aggregate-decoder input;

- the fixed ridge $\alpha =10.0$;

- the scalar $-MSE$/ $R^{2}$ score in the main aggregate CSVs;

- the diagonal Gaussian bits posthoc as the headline information score;

- the Ledoit-Wolf full-covariance log-det as supplemental only;

- target dimensionality and pooling details;

- target normalization and feature PCA using train-fold statistics;

- train/test grouping by strict source trial, with the selected image/window grouped result reported only as optimistic provenance context;

- response-summary preprocessing and leakage controls, especially global temporal PCA;

- deterministic mean-rate model outputs;

- motion-family matching and clipping statistics.

## 13. Literature grounding

This analysis sits at the intersection of four established traditions.

Spatial information. The SSI calculation follows the Skaggs-style single-spike information logic, where information is expressed as bits per spike and weighted by expected firing. This gives a familiar bits-per-spike form. As Section 2 notes, its position-information interpretation is licensed most cleanly for single tuned maps and weakens for a dense, redundant, feature-tuned population readout.

Decoding as an information assay. The ridge decoder is not meant to be a literal model of the animal’s readout. It is a controlled assay of recoverable information. This follows the standard population-coding practice of using decoders to estimate what stimulus variables are available in neural activity.

Pyramid features as V1-adjacent image variables. The pyramid target is biologically motivated because V1 neurons are tuned for local orientation, spatial scale, and phase. The signed quadrature components resemble phase-sensitive simple-cell channels. The magnitude components resemble phase-insensitive local energy or complex-cell channels.

Fixational eye movements as active sensing. The active-sensing interpretation is grounded in the idea that small eye movements transform spatial structure into temporal modulations. The decoder tests whether those temporal modulations improve recoverable feature information in motion-rendered twin responses. The stronger observer-level question, whether pose and content can be jointly inferred without explicit trajectory knowledge, is handled by the joint-decoder analysis.

## 14. Claim boundary

Supported claim:

In the fixed digital twin, motion-rendered responses add estimated lower-bounded feature information relative to stabilized responses under matched decoder capacity.

More precise wording:

In a fixed image-computable model, empirical drift-like motion increases held-out decoding of local pyramid features relative to a stabilized retinal input when responses are rendered under the empirical trajectory family. The aggregate decoder does not receive the trajectory as an explicit input, so this is not a literal pose-aware decoding claim. The complementary question of what an observer can recover without the trajectory is deferred to the joint-observer analysis.

Not supported by this analysis alone:

- The animal optimizes this decoder.

- The measured trajectory is globally optimal.

- The model result alone proves a biological active-sensing benefit.

- The aggregate decoder is explicitly pose-aware.

- A raw MSE gain is equal to mutual information.

- A positive absolute gain is automatically an efficiency gain.

- The motion-static difference is a rigorous lower bound on the true information gain.

- Fine within-block phase is recovered.

- Cross-band phase alignment is recovered under a linear decoder.

- Parallel-motion advantage has the same interpretation as the classic orthogonal-contour active-sensing prediction.

- The aggregate motion gain is tuned specifically to the natural FEM scale.

- The signal/nuisance partition is intrinsic to the code rather than observer-relative.

- The aggregate motion gain reflects a specific image-trajectory coupling rather than generic motion benefit (without the shuffle null).

- The along-edge preference reflects active per-patch trajectory shaping rather than passive statistical coincidence (without the image-swap null).

- The pyramid target is the uniquely correct feature set for evaluating motion benefit.

- The diagonal-bits metric is commensurable with the constrained discriminability metric used in the Vernier analysis.

The clean interpretation is:

This is a feature-restricted, model-internal, linear-Gaussian lower-bound estimate of recoverable feature information from motion-rendered V1-twin responses.

The cleanest active-sensing interpretation comes from delta_mean: empirical

drift-like motion induces response changes that add recoverable local-pyramid

feature evidence beyond the stabilized mean response. The mean readout is

reported as a complementary absolute-response readout, not as the isolated

motion-induced component.

## 15. Observer hierarchy and the signal/nuisance partition

### Prioritization

Of the analyses described in Sections 15–20, three are primary for the next revision. The remainder are supplemental or deferred to later work. The three primary analyses directly fix the interpretive weakness that 4B is currently a feature-decoder result without an explicit pose/nuisance penalty, while the behavior-facing story wants a constrained observer.

Primary analysis 1 (Section 17): Re-score 4B with covariance-whitened log-det in reduced target space. This is a re-scoring of existing results, not a new model run. It answers whether the motion gain survives when correlated feature-residual structure is penalized. Keep the diagonal bits as a reference, not the only headline.

Primary analysis 2 (Section 19.2): Run the trace-shuffle null on delta-R. This is the highest-value new analysis. It separates stimulus-specific sampling from generic motion benefit. It uses the existing decoder pipeline, image targets, folds, and response-summary machinery, but it requires new twin renders for permuted image-trajectory pairings unless those pairings have already been cached.

Primary analysis 3 (Section 16.1): Run along-contour versus across-contour under a constrained task-like metric with pose-marginal covariance in the denominator. This is the actual bridge to behavior. It requires computing tuning derivatives f_theta through the twin for a local task parameter, which is a new computation path.

Practical ordering: Analysis 1 first (re-score existing results, days). Analysis 2 next (new permuted-pairing renders plus the existing decoder pipeline, days to a week depending on cache state). Analysis 3 last (requires new twin computation for tuning derivatives, longer).

Supplemental (not blocking): MLP capacity ladder (Section 18), matched-static-response baseline (Section 19.1), passive/non-causal image-swap null (Section 19.3, but may become primary if the paper makes a trajectory-shaping claim), within-BackImage stratification (Section 19.4), target choice analysis (Section 20).

### Terminology

Throughout Sections 15–20, the term pose-marginal covariance refers to the covariance of the mean response over possible eye traces through the same image: Sigma_pose(I) = Cov_tau[mu(I, tau)]. This replaces the earlier wording “across-trajectory covariance,” which collides with “across-contour” (motion perpendicular to a local edge). Reserve “across-contour” exclusively for the movement direction relative to local image structure.

The aggregate Figure 4B decoder sits in an ambiguous position within a three-level observer hierarchy that must be made explicit, because the interpretation of every number the decoder produces depends on which level it occupies.

The three levels are defined by what the observer has access to about eye position:

Pose-aware oracle. The observer knows the trajectory that generated each response. Trajectory-induced response variance becomes explained variance rather than residual variance. The residual covariance Sigma shrinks, and discriminability can be higher than the static condition because each trajectory provides a distinct sample of the image. This is the active-sensing payoff: conditioned on knowing where you looked, looking was better than not looking.

Pose-hidden observer. The observer does not know the trajectory. Different trajectories through the same image produce different population responses, and this trajectory-induced variance enters the denominator of the discriminability metric as nuisance. The observer must treat motion-induced response modulation as noise.

Pose-jointly-inferred observer. The observer infers the trajectory (or marginalizes over it) from the response itself. This is the biologically realistic case: the animal does not have external access to its eye trace, but the response carries information about both image content and eye position. The discriminability of this observer is bounded by the data processing inequality: d-prime(pose-aware) >= d-prime(joint-infer) >= d-prime(pose-hidden). The gap between pose-aware and joint-infer measures how much trajectory information the response fails to carry; the gap between joint-infer and pose-hidden measures the value of inference.

Joint-decoder audit status: because the aggregate and local decoder branches both inherited estimator/provenance issues, the joint-inference observer now has a matching cache-level gate. The current 4C audit passes with no failures: calibration includes source-row CV and promotes the source-row-heldout row, supervised continuous feature decoders are source-row disjoint, posterior scores and posterior mass are internally consistent, and validated point estimates lie inside their own CIs. A positive joint result should still be reported as 4C feature recovery unless it is recomputed on the same diagonal-information axis as 4B.

This hierarchy is a mathematical identity for any target, because conditioning on additional information can only help. What depends on the target is the magnitude of each gap.

The aggregate 4B decoder does not fit cleanly into any one level. The trajectory renders the response (so the response benefits from trajectory-shaped input), but the trajectory is not an explicit decoder input (so the decoder cannot condition on it). The decoder pools across trajectories in training, so it learns a readout that works on average over the trajectory family. This is closer to a marginal-trajectory observer than to either the pose-aware oracle or the fully pose-hidden case. The practical consequence is that the aggregate ΔI reflects a mixture of genuine information gain and unresolved trajectory nuisance, and the relative proportions are not identifiable from the aggregate analysis alone.

The critical conceptual point is that signal and nuisance are not intrinsic properties of the neural code. They are defined relative to a target and an observer. The same response direction can carry image information and trajectory nuisance simultaneously: the Jacobian J(I) tells you, for this image, in which response directions trajectory variance is informative (because it samples new image content) and in which it is pure nuisance (because it adds variability to an already-measured feature). What the aggregate decoder reports as ΔI is conditional on the implicit observer choice, and that choice should be stated.

Minimum reporting requirement: a trajectory-augmented upper bracket and a pose-hidden lower bracket. Even if the joint-inference observer is deferred to the third implementation, the aggregate analysis should report what the decoder recovers when trajectory is provided as an explicit covariate or conditioning input, and what it recovers when trajectory-induced variance is propagated into target-space residual covariance. The trajectory-augmented bracket is not a full pose-aware oracle for a linear ridge decoder; it is a practical upper bracket whose validity depends on the decoder features and interactions. The aggregate Delta-I then sits between these brackets, and the bracket width is the observer-dependence that the aggregate analysis cannot resolve.

## 16. Objective-function lineage: connection to constrained discriminability

The Vernier/E-optotype analysis used constrained discriminability: d-prime-squared = Delta-mu-transpose Sigma-inverse Delta-mu, where Sigma is the pose-marginal covariance (covariance of the mean response over possible eye traces through the same image) for a fixed stimulus condition. This metric was chosen specifically because Sigma-inverse penalizes nuisance variance. Directions in response space that carry high trajectory-induced variability are downweighted in the discriminability calculation, regardless of whether they also carry stimulus information. The constrained metric therefore answers the question: how much stimulus information is accessible to an observer who must cope with trajectory noise?

The Figure 4B ridge-MSE decoder silently dropped that nuisance penalty. Ridge regression on deterministic rates with a fixed alpha does not incorporate pose-marginal response covariance into the loss function. The same word “linear” is doing weaker work: the Vernier metric is linear in the readout but constrained by the noise covariance; the ridge decoder is linear in the readout but unconstrained by trajectory-induced variance (since it operates on deterministic rates with no trial-to-trial variability from the trajectory).

This objective switch predicts a possible reconciliation of the along/across edge-motion result. Across-contour motion may dominate unconstrained metrics because it creates larger raw temporal modulation, whereas a constrained metric may favor along-contour motion once pose-marginal covariance enters the denominator. This remains a prediction until the constrained along/across analysis is run.

If confirmed, this reconciliation would be non-circular because the constrained metric is defined independently of behavior. It penalizes nuisance variance because the metric includes Sigma-inverse, not because the animal was observed to prefer along-edge motion. The animal could have evolved an unconstrained reader; the metric would still penalize nuisance. A confirmed match to behavior would therefore be evidence that the animal's readout approximates a nuisance-penalized computation.

Important caveat: “the brain is linear and cannot use across-edge information” is too strong a claim. Complex cells are exactly energy/quadrature readers, so V1 already has hardware for nonlinear phase-invariant readout. The defensible claim is about format and accessibility under the task-relevant noise structure, not about a ceiling on neural computation. The constrained metric penalizes nuisance variance, not algebraic nonlinearity.

### 16.1 Constrained along/across metric [PRIMARY ANALYSIS 3]

The reconciliation in Section 16 is currently an argument, not a computed result. To support the sentence “behavior aligns with nuisance-penalized task discriminability, not raw modulation,” the constrained metric must be computed explicitly.

The analysis uses a task-like constrained metric on a single local feature variable rather than the pyramid-feature ridge decoder. Pre-commit theta before running the analysis; the cleanest choice is signed local phase or signed position offset normal to the contour. Then compare drift direction d in {along-contour, across-contour} while keeping theta fixed. This avoids changing both the task variable and the movement direction at the same time.

For each drift direction d in {along-contour, across-contour}, compute the constrained Fisher information for the same local task variable theta: J_task(d) = f_theta(d)' (Sigma_count + Sigma_pose(d))^{-1} f_theta(d), where f_theta(d) = d mu / d theta is the tuning derivative of the mean response with respect to theta under drift direction d, Sigma_count is the spike-count noise covariance (Poisson or empirical), and Sigma_pose(d) is the pose-marginal covariance under drift direction d.

The key question is: for the same theta, which drift direction gives higher nuisance-penalized Fisher information? The prediction is: J_task(along) >= J_task(across) only after Sigma_pose enters the denominator. Without the pose-marginal penalty (i.e., using Sigma_count alone), across-contour may dominate because it produces larger tuning derivatives. With the penalty, the extra pose-marginal covariance of across-contour motion offsets the larger tuning derivatives.

Implementation note: this requires computing f_theta through the twin, which is a new computation path. The tuning derivative is obtained by finite-differencing the twin’s response with respect to small shifts in the task parameter. Sigma_pose(d) is computed from the existing rendered response trajectories, stratified by drift direction. This analysis is the slowest of the three primary analyses and should be scheduled last.

## 17. Covariance-whitened loss and the pose-known/pose-hidden bracket [PRIMARY ANALYSIS 1]

The current headline metric is a diagonal Gaussian bits estimate: ΔI_bits = (1/2) sum_j log2(var(epsilon_0,j) / var(epsilon_c,j)). This treats each target dimension independently and does not account for correlations in the residual structure. The Ledoit-Wolf full-covariance log-det is reported as supplementary. Neither of these connects to the constrained discriminability metric described in Section 16.

To make the decoder commensurable with the d-prime framework, the residual model should be upgraded to a covariance-whitened loss. Concretely, replace the diagonal residual variance with the full residual covariance Sigma_c, and report the log-det information as the headline. This means the information score accounts for correlated residual structure, and directions in target space with high residual covariance are implicitly penalized, analogous to the Sigma-inverse penalization in the d-prime metric.

The practical difficulty is that at the raw target dimensionality (d = 3072), a full residual covariance is rank-deficient with n approximately equal to 384 images. In PCA-reduced target space (k = 16), the full covariance is feasible with shrinkage. The recommendation is to report the covariance-whitened loss in the reduced target space as the primary metric, with the diagonal bits as a reference.

Separately, compute practical upper/lower brackets in this same metric. For the trajectory-augmented upper bracket, provide trajectory descriptors as explicit decoder inputs or conditioning variables. This is a linear pose-conditioned upper bracket, not a full pose-aware oracle: a ridge decoder with concatenated inputs can subtract only linear trajectory effects unless interaction terms are added. For the pose-hidden lower bracket, keep the covariance spaces aligned. The decoder residual covariance lives in target/PCA-feature space, while pose-marginal covariance first lives in response-summary space: Sigma_pose,S = Cov_tau[S(R(I, tau))]. With linear decoder phihat = W S, propagate trajectory nuisance into target space as Sigma_pose,phi approx W Sigma_pose,S W^T, and add that to the held-out residual covariance: Sigma_epsilon,hidden = Sigma_epsilon,count/fit + W Sigma_pose,S W^T. Better still, estimate Cov_tau[phihat(I, tau)] empirically by rendering multiple trajectories per image and measuring covariance of decoded feature predictions. The aggregate decoder's information score should then sit between these brackets. The width of the bracket is the interpretive ambiguity that the aggregate analysis cannot resolve.

Implementation for primary analysis 1: re-score the existing aggregate decoder residuals in PCA-reduced target space (k = 8, 16, 32) with Ledoit-Wolf shrinkage. Compute delta-I_logdet = (1/2) log2(|Sigma_epsilon,0| / |Sigma_epsilon,m|) for each k. If the motion gain is positive across k values, the headline result is strengthened. If it collapses at small k, the gain is carried by uncorrelated target dimensions and the result narrows to a diagonal-feature claim. This re-scoring uses existing held-out residuals and requires no new model runs.

## 18. Decoder capacity ladder and nonlinear extension [SUPPLEMENTAL]

The current analysis commits to a linear ridge decoder throughout. This is the right primary analysis because it connects to the linear-geometry thesis and produces an interpretable information score. However, the along/across result raises the question of whether the linear decoder is under-reading the response, and the MLP dissociation (across >> along under MLP, along > across under ridge) needs to be reported as a robustness check, not suppressed.

The principled approach is a capacity ladder with a pre-committed selection rule. The ladder runs from the ridge decoder (linear conditional mean, diagonal Gaussian residual) through an MLP with a diagonal Gaussian likelihood head (nonlinear conditional mean, diagonal Gaussian residual) to an MLP with a full Gaussian head (nonlinear conditional mean, full covariance residual). The key design constraint is that the likelihood head must remain Gaussian so that the Barber-Agakov variational bound is valid: for any decoder q(Phi | S), I(Phi; S) >= H(Phi) + E[log q(Phi | S)], and a tighter decoder gives a tighter bound. Making the conditional mean nonlinear does not cost infomax interpretability as long as the likelihood head stays Gaussian.

The selection rule should be pre-committed: use the decoder whose held-out log-likelihood is highest, subject to a complexity penalty (BIC or cross-validated log-likelihood). The principled endpoint is likely an energy/quadratic-form decoder rather than a generic MLP, because this connects to the complex-cell/energy-model readout that is biologically available in V1.

The real costs of the MLP extension are finite-sample optimism (the MLP can overfit with 384 images), condition-dependent gap-matching (the MLP may differentially overfit to the motion condition if it has more response variance to memorize), and coherence with the linear-geometry thesis. These should be reported as caveats, not used as reasons to suppress the result.

The MLP is not needed to confirm that motion information exists in the response. A shuffle null (Section 19.2) is the more targeted test. The MLP is needed to characterize what kind of information exists and whether the along/across result is decoder-class-dependent.

## 19. Missing controls and planned extensions

### 19.1 Matched-static-response baseline [SUPPLEMENTAL]

The current stabilized baseline R0 = f(I, tau_0) compares motion-rendered responses to a static input. Any absolute information gain could partly reflect increased total activity rather than improved encoding. Section 10 addresses this with the ΔI/ΔC efficiency metric, but this is a post-hoc correction.

A stronger design is a matched-static-response mode where the zero-eye baseline is forced to produce responses matched to the motion condition in a summary statistic (e.g., mean rate, total activation). This can be achieved by scaling the static input contrast or by selecting the stabilized response at a matched response level. The matched baseline is motion-limited by construction, isolating the information benefit from the response-drive confound. This mode was identified as load-bearing for the joint-observer analysis and should be implemented for the aggregate analysis as well.

### 19.2 Shuffle null for the conditional increment [PRIMARY ANALYSIS 2]

The motion-family controls (Section 11) test whether empirical FEMs are special relative to Brownian, OU, or rotated motion. But the more targeted question is whether the motion channel carries stimulus-specific information at all, as opposed to generic motion-induced response modulation that improves decoding through increased response variability.

The shuffle null tests this by permuting trajectory-image pairings within the aggregate pool. Each image is rendered under a trajectory drawn from a different image, preserving the marginal trajectory statistics and the marginal image statistics but destroying any coupling between trajectory and image content. If the motion increment survives the shuffle, it reflects generic motion benefit (more response variability happens to help decoding). If it collapses, the increment is stimulus-specific. This is the more direct test of whether the conditional increment (motion given static) is real.

Implementation for primary analysis 2: for each image I_i, re-render the twin response under a trajectory drawn from a different image: (I_i, tau_{pi(i)}). Compute delta-R_i(tau_{pi(i)}) = R_m(I_i, tau_{pi(i)}) - R_0(I_i). Score the decoder on shuffled delta-R against the original feature targets phi(I_i). Compare the information score under real pairings versus shuffled pairings. Multiple shuffle permutations provide a null distribution. This uses the existing decoder pipeline, image targets, folds, and response summaries, but it requires new twin renders for the permuted image-trajectory pairings unless all such pairings have already been cached. Once the shuffled responses exist, scoring is a standard decoder pass.

### 19.3 Passive/non-causal possibility and the image-swap null [SUPPLEMENTAL; PRIMARY if paper makes trajectory-shaping claim]

The along-edge preference in the decoder could reflect active per-patch trajectory shaping (the oculomotor system steers drift to align with local edge structure) or a passive statistical coincidence (a fixed drift policy happens to align with edges because natural images have oriented structure at all orientations, and the cos-squared alignment statistic is aggregate). The Vernier phase-cloud control already rules out online temporal-dynamics shaping (real approximately equals phase-cloud approximately equals order-shuffled), but it does not distinguish per-patch shaping from aggregate coincidence.

The marginal-orientation-matched image-swap null separates these. Replace each image patch with a different patch matched in marginal orientation content (dominant orientation, edge coherence, gradient energy) but drawn from a different image. Apply the original trajectory to the swapped patch. If the along-edge preference survives the swap, it reflects aggregate policy meeting aggregate image statistics. If it collapses, the original trajectory was tuned to the specific image patch that was viewed. This test is twin-free: the raw edge-alignment statistic does not require the twin, only the eye traces and the image patches.

This is arguably the most important missing control, because it determines whether the along-edge result supports an active-sensing claim or a statistical-coincidence claim.

Status note: this analysis is supplemental if the paper claims only that the constrained metric matches behavior (a claim about the objective, not the motor program). It becomes primary if the paper claims that drift is shaped to local image structure (a claim about active trajectory control). Since “active sensing” is in the paper’s framing, this test should remain on the list and be run once the three primary analyses are complete.

### 19.4 Within-BackImage stratification by image structure [SUPPLEMENTAL]

The aggregate analysis pools across all BackImage trials. This cannot test whether the FEM benefit scales with image structure. The twin makes it possible to stratify by image properties: local contrast, gradient energy, edge coherence, dominant orientation, spatial frequency content. If the motion benefit is larger for images with fine spatial structure (high-frequency content, strong edge coherence), this supports the phase-diversity mechanism. If the benefit is flat across image types, it reflects generic motion benefit that does not depend on image content.

This stratification is a refinement that Wu et al. could not demonstrate with their retinal model because they did not have the image-by-image resolution. The twin provides this resolution. The stratification does not require new model runs; it requires only partitioning existing results by image properties computed from the BackImage patches.

## 20. Target choice as a degree of freedom [SUPPLEMENTAL]

The pyramid target (Section 3) is a choice, and different target choices give different answers about whether motion helps. This is not an artifact; it reflects the fact that the signal/nuisance partition is target-relative. The document should be explicit about what the pyramid target assumes and what alternatives would change.

The pyramid target assumes that the downstream computation cares about local oriented structure at multiple scales, including phase-sensitive quadrature components and phase-insensitive energy. This is biologically motivated (V1 neurons are tuned for these features), but it is not the only defensible choice.

Three alternative targets occupy different positions on the commitment spectrum:

Target-agnostic: report the mutual information I(image; response | trajectory) versus I(image; response). The gap measures total trajectory nuisance. This is the most assumption-free but the hardest to compute honestly in high dimensions and the least connected to behavior.

Task-defined (constrained discriminability): pick a task parameter (orientation, position, phase) and compute d-prime-squared with Sigma-inverse penalizing trajectory-induced nuisance. This connects directly to the Vernier analysis and the along/across geometry. The advantage is that nuisance penalization is built into the metric. The disadvantage is that the task must be specified, and different tasks give different answers.

Pixel/full-image reconstruction: predict the raw image patch from the response. This is the maximum-commitment target. It subsumes all feature targets but requires far more decoder capacity and is dominated by low-frequency content that is already well-represented in the static response.

The current pyramid target sits between task-defined and target-agnostic. It names specific features (orientation, scale, phase, energy) but does not tie them to a behavioral task or penalize nuisance. The missing piece is either to add nuisance penalization (the covariance-whitened loss from Section 17) or to explicitly connect the pyramid features to a task that makes the signal/nuisance partition sharp. Without one of these, the ΔI the decoder reports is an unconstrained measure that does not distinguish between signal, nuisance, and redundancy.
