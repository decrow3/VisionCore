# Figure 4 Literature Methods Reset

Date: 2026-07-05

Purpose: identify where the current Figure 4 analyses have drifted into
bespoke exploratory machinery, and route the next tests back through established
computational-neuroscience methods. This note is aligned with `draft1.docx` and
the scientific-vetting checklist, with emphasis on feature decoders, latent-eye
observers, compact geometry, and along/across interpretation.

## Guiding Rule

For any claim from the feature-decoder steps onward, the default should be:

```text
known method family -> explicit observer/readout -> metric -> control -> claim
```

If an analysis cannot be placed in a known method family, it should be treated
as exploratory until we either replace it with a standard method or justify why
the standard method is inappropriate for the V1-twin setting.

## Biggest Method-Alignment Gaps

### 1. Joint Decoder / Latent-Eye Observer

Current issue:

The current 4C observer is useful and audited, but the theoretical story has
outgrown the cleanness of the endpoint. The promoted result is posterior
feature recovery with a continuous no-anchor calibration and compact
interventions. It is not yet the cleanest established observer: infer latent
eye trajectory, combine it with a visual likelihood, and reconstruct/estimate
image content under a specified prior.

Known method family to align with:

- Bayesian latent-state observer.
- Linear-Gaussian state-space model / Kalman smoother when the response is
  locally linear in eye displacement.
- Extended/unscented Kalman or particle filtering when the local-linear regime
  breaks.
- Wu-style known-eye / zero-eye / joint-eye reconstruction logic as the
  conceptual target, with exact citation and implementation details still to
  verify before manuscript use.

Target replacement test:

```text
response movie
  -> compact or static-PC coordinates
  -> continuous latent tau_hat under a declared process prior
  -> feature embedding z_hat or reconstruction-quality score
```

Minimal promotion gate:

- No candidate-image posterior as the feature endpoint.
- No MLP for the primary claim.
- No empirical trace replay endpoint.
- Source-row-disjoint fitting and calibration.
- Same score axis for all known-eye, zero-eye, hidden/joint-eye, static, compact,
  compact-removed, and static-PC controls.
- Report trajectory recovery error when tau_hat exists.

Current next action:

Use the existing candidate-free linear synthetic-prior observer script as a
starting point, but compare it against a standard Kalman/RTS smoother version
with compact and static-PC bases.

### 2. FEM Process Prior

Current issue:

OU, Brownian, rotated, and synthetic empirical-confined controls have been used
as analysis-specific nulls. That risks making the prior question look like a
local engineering choice, when FEM dynamics already have a literature: Brownian
behavior at short lags, confinement/anti-persistence at longer lags,
self-avoiding or self-repelling walks, and Bayesian parameter estimation.

Known method family to align with:

- Brownian/fractional Brownian descriptive checks for mean-square displacement.
- OU/AR(1) as the simplest confined Gaussian prior.
- Self-avoiding walk or self-repelling walk models with a confining potential.
- Posterior predictive checks of fitted FEM priors.

What to test:

Fit or calibrate candidate priors on held-out trajectories and compare them on:

- Step-size distribution.
- Velocity autocorrelation / reversal probability.
- Mean-square displacement across short and long lags.
- Radial confinement.
- Power spectrum or temporal autocovariance.
- Consequences for 4B feature information and 4C latent-eye recovery.

Interpretation rule:

OU should not be a straw-man negative control. It is the simplest member of a
confined-motion prior family. The newer reverse-step/random-walk prior should
be described in terms of established anti-persistence or self-avoidance
properties if it survives posterior predictive checks.

### 3. Feature Decoder Model

Current issue:

The feature decoders risk being asked to play too many roles: biological
readout, information lower-bound, natural-image reconstruction proxy, and
mechanistic FEM objective.

Known method family to align with:

- Encoding-model / decoding-model separation.
- Linear probe or ridge decoder for a declared feature target.
- Noise-ceiling and held-out cross-validation conventions from neural encoding
  models.
- Explicit target spaces motivated by V1 and natural-image statistics: Gabor,
  oriented energy, block-pyramids, CNN features, local contrast/edge structure.

Metric rule:

4B diagonal information-like scores, 4C feature cosine, negative MSE, and image
identity accuracy are different quantities. They can sit in the same paper, but
they should not be narrated as interchangeable evidence.

Minimal promotion gate:

- State whether the decoder is a measurement device, a biological readout
  candidate, or a reconstruction-quality observer.
- Keep source-row/source-trial splits.
- Include static/stabilized baselines on the same metric axis.
- Include feature-family sensitivity if the claim is about visual content in
  general rather than a single handpicked target.

### 4. Reconstruction and Natural-Image Priors

Current issue:

The joint-decoder discussion keeps leaning toward reconstruction logic. That is
not wrong, but it changes the scientific quantity. Once a natural-image prior
enters, the result is reconstruction quality under a prior, not response-carried
information in bits.

Known method family to align with:

- Representation inversion / feature inversion.
- MAP reconstruction with an explicit image prior.
- Plug-and-play, denoising, or deep-image-prior reconstruction methods for
  inverse problems.

Rule for Figure 4:

- Use feature decoders for 4B information/readout claims.
- Use reconstruction-quality observers for 4C latent-eye recovery claims.
- Do not import a natural-image prior into 4B and then call the result a V1
  information gain.

### 5. Compact Geometry Specificity

Current issue:

Compact geometry is real and useful, but static-response PCs are close
competitors. A literature-aligned framing should avoid treating compact
geometry as a unique eye-movement code unless the evidence beats ordinary
low-dimensional response-manifold controls.

Known method family to align with:

- Low-dimensional neural manifold / latent variable analysis.
- Compact vs static-PC basis comparison.
- Residualized compact after static PCs, and static-PC after compact.
- Covariance-aware controls rather than only diagonal or shuffle controls.

Minimal promotion gate:

- Compact-only near full recovery.
- Compact-removed near zero-eye recovery.
- Compact-addback restores full recovery.
- Static-PC controls reported as serious controls, not minor caveats.

Safe wording:

Eye movements push responses through a compact part of the ordinary
image-response manifold. Stronger claims require residual compact evidence over
static-PC and covariance controls.

### 6. Along vs Across Motion

Current issue:

The existing evidence should not be collapsed into "along motion helps." The
theoretical alternatives are distinct:

- Increased contour-parallel motion is useful.
- Reduced contour-normal motion is useful.
- Fixed-total anisotropy is useful.
- Edge-following behavior is a consequence of raw image geometry, not the model
  objective.

Known method family to align with:

- Active fixation as temporal encoding of spatial structure.
- Retinal motion as temporal whitening or refresh, not necessarily as an
  along-contour policy.
- Axis-conditioned observers with matched total RMS and separated normal vs
  parallel displacement.

Minimal promotion gate:

- Matched total displacement.
- Independent manipulation of contour-normal and contour-parallel components.
- Same observer and feature target across axis conditions.
- Raw-edge residual test before claiming the model objective explains behavior.

## Immediate Literature Reading Queue

Priority A: latent eye / image observer

- Verify the exact Wu Nature Communications 2024 paper and repository details
  before citing. Local notes describe known-eye, zero-eye, and joint-eye
  LNBRC-dCNN conditions, Bayesian MAP image reconstruction, and reconstruction
  quality metrics. Treat this as a citation lead until verified from the paper.
- Re-read standard state-space/Kalman observer material as the baseline method
  for continuous latent trajectory inference.
- Re-read Roweis and Ghahramani's linear-Gaussian model review and GPFA-style
  neural latent trajectory papers for the manifold/state-space vocabulary.

Priority B: FEM dynamics

- Rucci and Victor, "The unsteady eye: an information-processing stage, not a
  bug" (Trends in Neurosciences, 2015).
- Rucci and Poletti, "Control and Functions of Fixational Eye Movements"
  (Annual Review of Vision Science, 2015).
- Kuang, Poletti, Victor, and Rucci, "Temporal encoding of spatial information
  during active visual fixation" (Current Biology, 2012).
- Rucci, Iovin, Poletti, and Santini, "Miniature eye movements enhance fine
  spatial detail" (Nature, 2007).
- Engbert / Schwetlick line of self-avoiding-walk and Bayesian FEM modeling.

Priority C: V1 feature/encoding models

- Kindel, Christensen, and Zylberberg, "Using deep learning to reveal the neural
  code for images in primary visual cortex" (2017).
- Ecker/Sinz/Cadena/Tolias/Bethge line of CNN and rotation-equivariant V1
  encoding models.
- Classical Gabor/energy model and natural-image V1 encoding references for
  feature target justification.

Priority D: reconstruction and feature inversion

- Mahendran and Vedaldi, representation inversion.
- Deep Image Prior and plug-and-play / denoising-prior inverse-problem methods.
- These should be kept out of 4B information claims unless the metric is
  explicitly changed to reconstruction quality.

## Concrete Next Work Package

1. Literature verification table

Create a small table with columns:

```text
paper | method family | observer variables | prior | likelihood | metric |
what we can borrow | what we must not claim
```

2. FEM prior benchmark

Run or assemble a prior-only benchmark over empirical, Brownian, OU/AR(1),
self-avoiding/confined, and synthetic empirical-confined traces. Score
posterior predictive checks before using any prior as a 4B/4C control.

3. Standard continuous joint observer

Implement or promote the least bespoke version first:

```text
z_t = U^T (y_t - y_static)
z_t = A_I tau_t + eps_t
tau_t = alpha tau_{t-1} + eta_t
```

Use Kalman/RTS smoothing for each candidate image, with compact and static-PC
bases. If the linearization residual is large, move to EKF/UKF or a small
particle filter.

4. Candidate-free feature endpoint

Use the latent trajectory estimate only as a nuisance variable or covariate in a
source-row-disjoint linear-Gaussian feature observer. Promote only if it beats
zero/static baselines on the same feature-recovery axis.

5. Axis mechanism adjudication

Run matched-total, fixed-normal, and fixed-parallel axis tests so "along helps"
can be separated from "across is suppressed."

## Draft Consequences

- The current joint decoder should be described as an audited provisional
  observer, not the final theoretical endpoint.
- OU should be reframed as a member of the confined-motion prior family, not as
  a disposable negative control.
- RR100 should be described as a biologically motivated reduced V1
  representation candidate when it is the analysis substrate.
- Compact geometry should be presented as useful shared manifold structure
  unless residual compact evidence beats static-PC controls.
- Along/across language should stay mechanism-level until the component tests
  separate parallel increase from normal reduction.

## Sources Opened In This Reset

- Bayesian FEM dynamics / self-avoiding walk:
  https://arxiv.org/abs/2303.11941
- CNN V1 encoding model:
  https://arxiv.org/abs/1706.06208
- Rotation-equivariant CNN V1 model:
  https://arxiv.org/abs/1809.10504
- Representation inversion:
  https://arxiv.org/abs/1412.0035
- Deep Image Prior:
  https://arxiv.org/abs/1711.10925

Citation caveat: the exact Wu Nature Communications 2024 paper and any GPL
repository should be verified from the paper/repository before manuscript
citation or code-adjacent work. Use conceptual observer lessons only; do not
copy or mechanically adapt repository code.
