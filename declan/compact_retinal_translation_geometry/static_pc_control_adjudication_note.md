# Static-PC Control Adjudication Note

Date: 2026-06-22

## Short Answer

Yes, we previously tried removing a global first component, but that statement
needs to be kept precise.  In the recorded covariance / displacement branches,
we used `projection_control=global_rate+target_pc1`: a global-rate axis plus the
dominant fold-trained target/tangent PC.  In the new Figure 3 tangent-basis
adjudication, we also tested `static_response_pc_without_pc1`, which removes
the first static-response PC from the static-PC basis itself.

Those controls are not identical.  The older `global_rate+target_pc1` control
asks whether compact effects survive after projecting out broad low-dimensional
recorded/tangent covariance structure.  The new `static_response_pc_without_pc1`
control asks whether the static-response manifold's first PC is responsible for
static PCs matching the compact tangent basis on held-out translation tangent
capture.

## Current Bottom Line

The compact translation geometry is real, low-dimensional, image-generalizing,
and strongly above random, unit-shuffled, and global-rate controls.  The new
static-PC tangent-capture result does not overturn that.  It says something
narrower: this particular tangent-capture metric cannot establish uniqueness
over the static image-response manifold, because retinal translations are
tangents to that manifold in the first place.

To first order, shifting an image moves the response along a curve inside the
image-response manifold:

```text
M = { r(I) }
r(I, tau) ~= r(I) + J(I) tau
```

The vectors in `J(I)` are therefore local tangent vectors of `M`.  Static
response PCs span the highest-variance directions of `M`; if translations are a
large source of image-response variation, static PCs are expected to capture
translation tangents well.  A near tie between compact tangent PCs and
fold-disjoint static-response PCs is therefore not evidence against compact
geometry.  It is an inconclusive uniqueness test and, in a useful sense, a
manifold-consistency check: translation tangents sit in the high-variance image
response manifold.

The blast radius is contained.  This result wounds language like "unique",
"dedicated", "privileged", or "the compact subspace" when used as a
specificity claim over static PCs.  It does not reopen the core Figure 3/4
geometry result: compactness, image-disjoint generalization, above-null status,
and recorded covariance closure remain existence / above-null statements.

At `k=10` in
`outputs/compact_retinal_translation_geometry/static_pc_adjudication_v1/`:

| basis | held-out tangent capture |
|---|---:|
| compact tangent PCs | `0.488 [0.457, 0.513]` |
| static-response PCs | `0.506 [0.472, 0.536]` |
| compact residualized against static PCs | `0.116 [0.105, 0.127]` |
| static PCs residualized against compact | `0.133 [0.117, 0.152]` |
| static-response PCs without PC1 | `0.299 [0.282, 0.319]` |
| global-rate axis | `0.113 [0.098, 0.125]` |
| unit-shuffle compact | `0.069 [0.061, 0.076]` |
| random orthonormal | `0.014 [0.013, 0.014]` |

The paired compact-minus-static difference at `k=10` is
`-0.018 [-0.041, 0.004]`; across the k sweep it stays near zero:

| k | compact - static-response PC |
|---:|---:|
| 2 | `+0.002 [-0.014, +0.019]` |
| 5 | `-0.018 [-0.037, +0.004]` |
| 10 | `-0.018 [-0.041, +0.004]` |
| 20 | `-0.007 [-0.018, +0.003]` |
| 30 | `-0.009 [-0.019, +0.000]` |
| 40 | `-0.005 [-0.013, +0.003]` |
| 50 | `+0.007 [-0.001, +0.014]` |

The first static PC matters a lot: dropping it lowers static-PC capture from
`0.506` to `0.299` at `k=10`.  But even after dropping PC1, static PCs remain
well above random, unit-shuffle, and global-rate controls.  So the static
manifold overlap is not just a scalar global gain artifact, though PC1 carries a
large fraction of it.  This is exactly the useful middle ground: the channel is
multi-dimensional and manifold-tangent, not a single global gain axis.

The direct compact/static-PC overlap is also large.  At `k=10`, the bootstrapped
mean squared principal cosine is `0.551 [0.543, 0.564]`, the mean principal
cosine is `0.684 [0.674, 0.695]`, and the minimum principal cosine is
`0.092 [0.019, 0.164]`.  The residualized tangent-capture rows above are the
practical consequence: after either basis is stripped of the other, each retains
only a small amount of translation-tangent capture.

## Follow-Up Adjudication Completed

The requested compact/static-PC follow-up tests have now been run in three
places.

### A. Symmetric response-table removal, image-identity endpoint

Output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_symmetric_subspace_removal_prod_v1/
```

Across the six hard-negative prior/motion conditions at `k=10`,
likelihood scale `1.0`:

| response variant | mean joint image accuracy | mean joint-minus-zero accuracy |
|---|---:|---:|
| full exact | `0.770` | `+0.324` |
| zero static | `0.445` | `0.000` |
| compact only | `0.656` | `+0.211` |
| static-PC only | `0.634` | `+0.189` |
| compact removed | `0.432` | `-0.013` |
| static-PC removed | `0.454` | `+0.009` |
| compact residual only | `0.431` | `-0.014` |
| static-PC residual only | `0.430` | `-0.016` |
| compact residual removed | `0.659` | `+0.214` |
| static-PC residual removed | `0.714` | `+0.268` |
| random removed | `0.733` | `+0.288` |

Mean full-minus-removed accuracy losses:

| removed subspace | mean full-minus-removed accuracy |
|---|---:|
| compact | `0.337` |
| static PC | `0.315` |
| compact residual after static PCs | `0.111` |
| static-PC residual after compact | `0.056` |
| random | `0.037` |

Interpretation: compact and static-PC removals have very similar consequences
for the image-identity endpoint.  The residual-only bases are near zero-eye,
and removing only the residualized pieces has much smaller effects.  This
supports shared compact/static manifold overlap, not compact uniqueness.

### B. Symmetric response-table removal, feature-posterior endpoint

Output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_feature_symmetric_subspace_removal_prod_v1/
```

Across the six hard-negative prior/motion conditions at `k=10`, feature PCA
`k=8`, likelihood scale `1.0`:

| response variant | mean feature cosine | mean feature neg-MSE |
|---|---:|---:|
| known eye | `0.937` | `-9.729` |
| full exact | `0.872` | `-19.144` |
| zero static | `0.670` | `-59.031` |
| compact only | `0.838` | `-23.793` |
| static-PC only | `0.820` | `-25.803` |
| compact removed | `0.644` | `-64.186` |
| static-PC removed | `0.684` | `-58.044` |
| compact residual only | `0.676` | `-58.221` |
| static-PC residual only | `0.653` | `-61.881` |
| compact residual removed | `0.843` | `-22.814` |
| static-PC residual removed | `0.871` | `-20.106` |
| random removed | `0.867` | `-19.453` |

Mean full-minus-removed feature neg-MSE losses:

| removed subspace | mean full-minus-removed neg-MSE loss |
|---|---:|
| compact | `45.042` |
| static PC | `38.900` |
| compact residual after static PCs | `3.670` |
| static-PC residual after compact | `0.963` |
| random | `0.309` |

Interpretation: compact removal is somewhat worse than static-PC removal in the
feature endpoint, but both are large and both are far stronger than random
removal.  The residualized pieces explain very little.  This is evidence that
the useful feature-posterior response structure is mostly in the shared
compact/static manifold-overlap component.

### C. Covariance predictor comparison

Output:

```text
outputs/matched_twin_covariance_closure_static_pc_predictor_prod_v1/
```

This run matched the previous full-sample Allen `2022-02-16` finite-difference
closure scope, using full samples, `step_px=0.25`, PSD target, `k=2,10`, and
`n_nulls=100`.  The new source is a leave-fold/trial-disjoint static-PC
finite-difference covariance surrogate:

```text
fd_sample_eye_trace_xfit_static_pc_k10_cov
```

At `k=10`:

| projection control | finite-difference eye-trace source | compact xfit source | static-PC xfit source |
|---|---:|---:|---:|
| none | `0.747` | `0.742` | `0.748` |
| global_rate+target_pc1 | `0.426` | `0.417` | `0.425` |

Interpretation: in this covariance-closure predictor comparison, the static-PC
surrogate matches or slightly exceeds the compact cross-fit surrogate.  This is
not evidence against the finite-difference translation source itself; the
unprojected eye-trace source remains strong.  It is evidence against a
compact-specific covariance-predictor claim over static PCs.

## What Each Previous Control Showed

### 1. Figure 3 / compact geometry controls

Controls already passed:

- samplewise unit-shuffle null for the pooled tangent spectrum,
- image-disjoint compact-basis generalization,
- recorded covariance closure against unit-shuffle and RF/readout-preserving
  nulls,
- conservative `global_rate+target_pc1` projection in covariance closure.

Key old results:

- Tangent participation ratio at `delta=0.25 arcmin`: observed `9.04` versus
  unit-shuffle null around `31.03`.
- Image-disjoint `k=10` compact basis captured about `0.524` held-out tangent
  variance versus unit-shuffle null about `0.122`.
- Covariance closure survived `global_rate+target_pc1`: finite-difference
  source captured about `0.220` of PSD recorded FEM covariance, with
  unit-shuffle excess about `+0.177`, CI `[0.144, 0.212]`, positive in `24/24`
  sessions.

What these controls ruled out:

- compactness is not explained by arbitrary unit labels,
- compact axes generalize across held-out images,
- the finite-difference translation source predicts recorded FEM covariance
  above unit-shuffle and RF/readout-preserving nulls,
- the covariance result is not only global rate plus the dominant target PC.

What they missed:

- They did not ask whether a static-PC-derived predictor can predict recorded
  FEM covariance as well as the finite-difference translation source.
- They did not measure compact/static subspace overlap directly.
- The `target_pc1` projection is a strong low-dimensional nuisance control, but
  it is not the same as comparing the full static image-response manifold
  against the compact tangent manifold.
- A static-PC projection control is not the right covariance adjudication by
  itself, because projecting static PCs out of the target covariance can remove
  the signal and the proposed nuisance together.  The clean covariance test is a
  predictor comparison: build a static-PC-derived `J` surrogate and ask whether
  it predicts `Sigma_FEM` as well as the finite-difference translation `J`.

### 2. Relative displacement decoder controls

This branch explicitly tested projection controls:

| projection control | compact `R2_mean` | eye-shuffle excess |
|---|---:|---:|
| none | `0.0746` | `+0.1029 [0.0502, 0.1569]` |
| global_rate | `0.0506` | `+0.0789 [0.0350, 0.1202]` |
| target_pc1 | `0.0045` | `+0.0345 [0.0112, 0.0637]` |
| global_rate+target_pc1 | `-0.0019` | `+0.0284 [0.0083, 0.0539]` |

What this ruled out:

- matched-context responses contain displacement-related information;
- the raw displacement signal is not robustly compact-specific after removing
  global-rate and top-PC structure.

What it missed:

- It was a recorded single-trial displacement decoder, not a clean finite-
  difference tangent-energy adjudication.
- The top-PC projection showed that a broad component matters, but did not say
  whether static image-response PCs as a family can explain the compact tangent
  result.
- This is a limit on the current decoder, not a failure of the structural
  compact-geometry result.

### 3. Direct recorded derivative / covariance bridge

The direct recorded derivative alignment used the conservative condition
`projection_control=global_rate+target_pc1`, `target_variant=psd`,
`context_subset=reliability_qualified`, and `k=10`.

Key old results:

- capture mean `0.386`,
- RF/readout-null excess `+0.210`, CI `[0.178, 0.246]`,
- unit-shuffle excess `+0.288`,
- random-subspace excess `+0.284`,
- sign consistency `13/13` eligible sessions.

What this ruled out:

- recorded eye-position sensitivity is not merely arbitrary noise in unit
  space;
- it is enriched in the compact twin tangent subspace even after broad
  projection controls.

What it missed:

- It was an enrichment test, not a compact-versus-static-PC comparison.
- It does not establish signed horizontal/vertical coordinate recovery.
- It does not tell us whether static response PCs would be similarly enriched.
- The decisive static-PC version would be a matched predictor/enrichment test,
  not simply projecting the target away.

### 4. Compact mechanism / response-table ablations

The response-table mechanism analyzer tested:

```text
full_exact
zero_static
compact_only
compact_removed
log_compact_only
log_compact_removed
random_k
unit_shuffle_compact
gain_only
static_pc_k
```

In the hard-negative image-disjoint run at likelihood scale `1.0`:

| response variant | k | joint accuracy | median joint-zero true-score | rescue fraction |
|---|---:|---:|---:|---:|
| full_exact | 0 | `0.758` | `+6.116` | `1.000` |
| zero_static | 0 | `0.445` | `0.000` | `0.000` |
| compact_only | 10 | `0.645` | `+5.085` | `0.859` |
| compact_removed | 10 | `0.426` | `-2.328` | `-0.378` |
| log_compact_removed | 10 | `0.451` | `-0.333` | `0.012` |
| random_k | 10 | `0.422` | `-1.750` | `-0.299` |
| unit_shuffle_compact | 10 | `0.497` | `+1.739` | `0.328` |
| static_pc_k | 10 | `0.626` | `+4.560` | `0.794` |
| static_pc_k | 20 | `0.658` | `+4.650` | `0.828` |

What this ruled out:

- compact-only is sufficient to preserve much of the exact-table joint-eye
  image-identity rescue;
- compact-removed collapses most of that rescue;
- the collapse is not purely a negative-rate clipping artifact, because
  `log_compact_removed` also loses most of the rescue while keeping positive
  rates by construction;
- random and unit-shuffled compact controls are much weaker than compact.

What it missed:

- Static PCs were already competitive: `static_pc_k` recovers nearly as much
  rescue as compact at `k=10` and can exceed compact in accuracy at `k=20`.
- The analysis was "keep-only" for static PCs, not symmetric removal of
  static PCs versus compact PCs.
- It did not residualize compact against static PCs or static PCs against
  compact.
- It used image identity / table score rescue, not direct tangent-energy
  capture.
- Because static PCs may inherit compact translation tangents from the
  image-response manifold, strong static-PC performance weakens uniqueness
  wording but does not falsify compact geometry.

### 5. Feature-posterior compact-removal test

In the feature endpoint
`backimage_feature_posterior_compact_removed_pyramid_k8_n128_scales_0p5_1_2_v1`:

| response variant | mean feature neg-MSE | mean feature cosine | mean true mass |
|---|---:|---:|---:|
| full_exact / compact_addback | `-19.144` | `0.872` | `0.466` |
| compact_only | `-23.793` | `0.838` | `0.436` |
| compact_removed | `-64.186` | `0.644` | `0.348` |
| zero_static | `-59.031` | `0.670` | `0.348` |
| known_eye | `-9.729` | `0.937` | `0.574` |

What this ruled out:

- compact removal hurts feature-posterior recovery, and compact-only preserves
  a large fraction of the feature endpoint.

What it missed:

- It compared compact removal mainly to zero-static/full/compact-only, not to
  removing matched static PCs, gain axes, random subspaces, or residualized
  compact/static components.
- Thus it supports compact importance, but not compact specificity.

### 6. Compact-aware trajectory-prior tests

The image-independent compact prior is better described as a
leave-one-table-out catalog-statistic prior, not a universal biological
eye-motion prior.  It reweights trajectory slots using pooled response-space
statistics and stable trajectory IDs.

Hard-negative result:

| prior/control | mean feature neg-MSE gain over uniform |
|---|---:|
| unit-shuffle compact aware | `+0.857` |
| gain-axis aware | `+0.825` |
| image-independent compact prior | `+0.619` |
| static-PC aware | `+0.558` |
| candidate-conditioned compact weight | `-0.311` |
| random-subspace aware | `-0.696` |
| inverse compact control | `-1.312` |

Matched-static result:

| prior/control | mean feature neg-MSE gain over uniform |
|---|---:|
| inverse compact control | `+0.485` |
| random-subspace aware | `-0.050` |
| static-PC aware | `-0.475` |
| unit-shuffle compact aware | `-0.773` |
| image-independent compact prior | `-0.796` |
| gain-axis aware | `-0.877` |
| candidate-conditioned compact weight | `-3.050` |

Coverage caveat:

- In the hard-negative run, image-independent priors used nonmatching fallback
  for `5.84/16` trajectory slots on average, with no missing fallback slots.

What this ruled out:

- nonuniform response-space trajectory weighting can matter;
- the compact-prior implementation is not using `y_obs` directly.

What it missed:

- It did not show compact-specific prior usefulness.  Gain and unit-shuffle
  controls beat compact in hard-negative, compact hurt in matched-static, and
  fallback coverage remained incomplete.  That makes the endpoint
  artifact-dominated / inconclusive, not a clean negative against analytic
  compact priors.
- It used a finite catalog and leave-one-table-out lookup, not an analytic
  continuous eye-motion prior.
- It does not prove that prior knowledge of compact geometry is useful.
- It promotes `U_static_pc`, gain-axis, and unit-shuffle priors from routine
  controls to decisive controls for the next analytic observer.

### 7. Chart-swap / content-routing controls

The chart-swap analysis tests whether the correct image-conditioned chart gives
the true candidate a better score than wrong-image charts.

At `k=30`, correct-minus-wrong scores were positive:

| dataset / contrast | compact | static PCs | random | unit shuffle |
|---|---:|---:|---:|---:|
| hard-negative, correct - wrong roll | `+126.12` | `+140.78` | `+121.00` | `+106.33` |
| hard-negative, correct - wrong pool | `+65.14` | `+76.45` | `+63.07` | `+48.44` |
| matched-static, correct - wrong roll | `+73.30` | `+91.50` | `+64.01` | `+60.81` |
| matched-static, correct - wrong pool | `+42.76` | `+60.38` | `+34.53` | `+22.68` |

Correct-minus-global was weak or negative:

| dataset | compact correct - global at k=30 |
|---|---:|
| hard-negative | `-7.19` |
| matched-static | `+0.53` |

What this ruled out:

- image-conditioned charts matter relative to wrong charts;
- the content-routing idea is not just "one universal signed eye-position
  axis" in the simple separable sense.

What it missed:

- Static PCs again matched or exceeded compact on wrong-chart separation.
- Random bases were also strong in some wrong-chart contrasts, so the chart-
  swap score partly reflects generic high-dimensional image/template mismatch.
- Correct-versus-global did not provide strong compact-specific support.

### 8. Forward denoising controls

The forward denoising preview tested compact correction against random,
unit-shuffled compact, compact-projected shuffled-eye, full shuffled-eye, and
gain-only controls.

Key result:

- compact beat random and unit-shuffled compact controls;
- compact did not beat matched shuffled-eye controls;
- fixed-alpha diagnostics showed calibration mattered.

What this ruled out:

- compact corrections contain reproducible denoising signal relative to weak
  geometry controls.

What it missed:

- It did not establish specificity to the actual trial eye trace.
- It does not directly adjudicate compact versus static-response PCs.

## How The New Result Changes The Interpretation

Before the static-PC adjudication, the clean sentence was:

```text
Compact translation geometry is sufficient for much of the joint-eye rescue,
and compact removal strongly hurts.
```

That sentence still holds.  But the stronger sentence:

```text
The compact tangent subspace is the uniquely useful low-dimensional response
geometry for joint decoding.
```

is not supported by this comparison and should not be stated as a current
result.

The new result shows that fold-disjoint static-response PCs can capture the
same held-out tangent energy as compact tangent PCs.  This does not mean static
PCs explain the compact result away.  It means tangent capture is the wrong
metric for proving uniqueness over static PCs.  Local translation tangents are
embedded in, or strongly aligned with, the low-dimensional manifold of
image-evoked V1 responses.  That is theoretically expected: infinitesimal
retinal translations are tangents to the image-response manifold.

The static-PC result is therefore corroborative for the manifold-tangent frame:
it explains why global/rank-1 controls fail, why static-without-PC1 remains
above random, and why the compact subspace should be treated as a compact
manifold tangent channel rather than a dedicated translation-only channel.
The cost is honest wording.  We should give up "dedicated translation channel"
language, and describe `A(I)` as local chart coordinates of translation within a
shared manifold-aligned population geometry.

So the compact result should be described as:

```text
Small retinal translations occupy a compact, image-generalizing response
channel that is far above random/unit-shuffle/global-rate controls and predicts
recorded FEM covariance.  This channel substantially overlaps the static
image-response manifold, as expected if translations are manifold tangents; the
current tangent-capture test therefore does not adjudicate uniqueness over
fold-disjoint static-response PCs.
```

This should not reopen Figure 4.  The keystone claims are existence /
above-null claims: low-dimensionality, image-disjoint generalization,
above-null compact tangent capture, and recorded covariance prediction.  The
static-PC result only constrains specificity language and the compact-aware
prior / second-paper framing.

## What Remains For A Stronger Specificity Claim

The first five static-PC adjudications below are now complete for the current
production scopes:

1. Compact versus static-PC principal-angle overlap.
2. Residualized tangent capture.
3. Static-PC covariance-predictor comparison.
4. Symmetric response-table removal for image identity.
5. Symmetric response-table removal for feature recovery.

They converge on the same answer: compact is important, but its useful
component is highly shared with static-response PCs.  We should not promote a
compact-specific mechanism over static PCs from the current tests.

The remaining positive specificity tests would need to be stricter:

1. Broaden the covariance predictor comparison beyond the single full-sample
   Allen `2022-02-16` scope, if we want a population-level static-PC
   covariance statement.
2. Require an analytic compact-aware observer/prior to beat entropy-matched
   static-PC, gain, random, and unit-shuffle priors, and to lose that advantage
   when the compact/static-overlap component is removed.
3. Test whether content-conditioned chart models provide advantages that are
   not matched by static PCs, not just advantages over wrong/random charts.

## Practical Wording For The Manuscript Right Now

Use:

```text
The translation-linked response component is compact and image-generalizing,
and it lies in a response subspace that can support joint image/eye inference.
```

Avoid for now:

```text
The compact translation basis is uniquely better than static image-response PCs.
```

The better mechanistic framing is:

```text
V1 translations appear to reuse a compact population channel that is strongly
coupled to the static image-response manifold.  This explains why compact
geometry is useful, why global/rank-1 controls are insufficient, and why static
PCs are a serious contender rather than a disposable nuisance control.
```
