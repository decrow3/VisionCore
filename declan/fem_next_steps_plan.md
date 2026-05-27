# FEM Next Steps: Detailed Implementation Plan

*Written May 2026. References the existing code infrastructure in `scripts/temporal_decoding/` and `declan/`.*

---

## Recommended execution order

B → B.5 → A → C → D → E/F (conditional on B)

B comes first because it resolves whether the temporal null (C ≈ A) is a pipeline artifact — a finding that would change what the decoders are measuring but does not change the ablation story. A comes next; the fixed-position baseline is a rate-caching job that can sit in the queue while B runs. B.5 (matched position-distribution reweighting) sits between B and A because it requires no new rate caching — it reweights existing trials — and gives early signal on whether the non-specific ablation gains are a position-distribution artifact before committing to A's full caching run.

**Note on A's independence from B:** Even if B3 reveals a collapse artifact and temporal coding becomes the primary finding, A should still run. The ablation story (separating dynamic FEM from static position dispersion) is independent of whether the representation pipeline is adequate for temporal decoding. Do not treat B's decision gate as a reason to defer A.

---

## B. Spatial-map audit and collapse-mode comparison

**Goal:** Verify that `amax` spatial collapse is not silently destroying temporal information before reaching the decoders. Three sub-tasks, all scripts already exist.

### B1 — Hotspot visualisation (`inspect_maps.py`)

Script: `scripts/temporal_decoding/inspect_maps.py`

Run at the two key LogMARs for one real-FEM and one stabilized trial, orientation 0°. Also run for one high-motion trace (RMS ≈ 0.222°, from the motion-magnitude binning results) and one low-motion trace:

```bash
python scripts/temporal_decoding/inspect_maps.py --logmar -0.20 --orientation 0 --n_neurons 6 --trace_idx 0
python scripts/temporal_decoding/inspect_maps.py --logmar -0.40 --orientation 0 --n_neurons 6 --trace_idx 0
```

**Quantitative decision gate.** Visualisation alone is not sufficient — compute the following for each neuron and summarise in a table:

| Metric | How to compute | Threshold for concern |
|---|---|---|
| Pearson r (argmax displacement vs eye position) | correlate Δargmax(t) with Δeye(t) across frames | r < 0.3 in real condition = no coupling |
| RMS hotspot displacement (real) | RMS of argmax(t) − argmax(0) across frames | < 0.5 px = effectively flat |
| Fraction frames with >1 px movement (real) | fraction of t where ‖argmax(t) − argmax(t−1)‖ > 1 | < 10% = hotspot barely moves |
| Real vs stabilized RMS ratio | RMS_real / RMS_stabilized | < 1.5 = no meaningful difference |

**Interpretation branches:**
- If r > 0.3 and RMS ratio > 1.5 in real condition → hotspots track eye position, `amax` is destroying real signal → treat B2/B3 as urgent.
- If all four metrics are below threshold in both conditions → spatial collapse is not the bottleneck → B2/B3 can be brief confirmations.

---

### B2 — Step-shift sensitivity test (`step_shift_test.py`)

Script: `scripts/temporal_decoding/step_shift_test.py`

```bash
python scripts/temporal_decoding/step_shift_test.py --logmar -0.40 --n_traces 5 --orientation 0
python scripts/temporal_decoding/step_shift_test.py --logmar -0.20 --n_traces 5 --orientation 0
```

**The 1.0 arcmin shift is the ecologically relevant test** — this matches the FEM stroke amplitude. The 0.5 and 2.0 arcmin cases bracket it.

**Information-preservation chain.** Rather than only comparing raw maps to `amax`, measure sensitivity across a sequence of collapse modes to directly quantify how much signal each one preserves:

```
raw (H×W map) → flat (spatially concatenated) → CoM → amax
```

For each collapse mode, report the sensitivity ratio: mean response difference (shifted vs unshifted) divided by response variability. If sensitivity drops sharply at a specific step (e.g., flat preserves the shift but amax loses it), that step is the bottleneck.

**Pass criterion:** If amax sensitivity ≥ 10% of raw-map sensitivity, collapse is not the bottleneck. If amax sensitivity < 10% of raw-map sensitivity but flat sensitivity ≥ 50%, `amax` is the bottleneck and B3 becomes the critical test.

---

### B3 — Collapse-mode comparison (`collapse_comparison.py`)

Script: `scripts/temporal_decoding/collapse_comparison.py`

```bash
python scripts/temporal_decoding/collapse_comparison.py --logmar -0.20 --n_traces 60 --n_splits 5 --n_pca_flat 50
python scripts/temporal_decoding/collapse_comparison.py --logmar -0.40 --n_traces 60 --n_splits 5 --n_pca_flat 50
```

Use `n_traces=60` (not the default 20) for stable accuracy estimates.

**What to look for:** D1 and Model C accuracy under `max`, `mean`, `flat+PCA`, and CoM collapse modes.

| Outcome | Interpretation |
|---|---|
| C ≈ A across all modes | Temporal null is real; not a collapse artifact |
| C > A under `flat` or CoM but not `max` | `amax` is the bottleneck; rebuild temporal decoding with flat features before drawing further conclusions |
| C > A under `mean` but not `max` | Mean preserves some signal `max` does not; worth following up |

**Decision gate:** If `flat+PCA` rescues Model C at either LogMAR, the immediate implication is to rebuild temporal decoding with flat features — not to stop other steps. The ablation (step A), α sweep (C), and amplitude scaling (D) do not depend on the temporal null and should continue regardless. What changes is the framing: if temporal coding is rescued, it becomes a parallel finding rather than a closed question.

---

## B.5 — Matched position-distribution reweighting (pre-A check)

**Goal:** Before caching new rates for the `fixed_center` condition, get early signal on whether the non-specific ablation gains are driven by different position distributions between real and stabilized — using only existing cached rates.

**What to do:** For each LogMAR (−0.20, −0.40), compute the distribution of per-trial mean eye positions in the real and stabilized conditions. The real condition has a spread of mean positions across trials; stabilized also has a spread (each trial is held at its own mean). Reweight or subsample real trials so that their mean-position distribution matches the stabilized distribution (e.g., via importance weighting or nearest-neighbour subsampling). Then rerun the pooled and differential ablations on the reweighted real trials vs unweighted stabilized trials.

**No new code required beyond scipy/numpy reweighting.** Add this as a short script `declan/matched_position_ablation.py`.

**Interpretation branches:**
- If the ablation non-specificity dissolves under matched distributions (real improves, stabilized does not) → the effect was a position-distribution artifact → A becomes a cleaner confirmation of this.
- If the non-specificity persists under matched distributions → the effect is not explained by position-distribution mismatch → A tests something sharper (fixed vs per-trial mean), and the result will be more informative.

---

## A. True fixed-position stabilization baseline

**Goal:** Replace the current `stabilized` condition (each trial at its own mean position, leaving residual across-trial positional variance) with a single fixed retinal position for all trials. This makes the ablation results interpretable as specific to dynamic FEM rather than static position dispersion.

### A1 — Add `fixed_center` condition to `rate_computation.py`

File: `scripts/temporal_decoding/rate_computation.py`

The current `stabilized` condition calls `_scale_trace(eyepos, eye_scale=0)`, which collapses each frame's position to the per-trial mean. The new condition should collapse all positions to the grand mean across all 471 traces and all timepoints.

**Implementation — compute grand mean offline first:**

```python
import numpy as np
traces = np.load('scripts/temporal_decoding/data/eye_traces.npz')
# Inspect the array structure, then:
grand_mean = traces['eyepos'].reshape(-1, 2).mean(axis=0)  # shape (2,)
print(f"Grand mean eye position: {grand_mean}")
```

Run this once, record the printed value, and hardcode it as a constant in `rate_computation.py`:

```python
# Grand mean across all 471 traces (computed offline 2026-05-22)
_GRAND_MEAN_EYE_POS = torch.tensor([X.XXX, Y.YYY])  # degrees
```

Then in `build_counterfactual_stim`:

```python
elif condition == 'fixed_center':
    mean_pos = _GRAND_MEAN_EYE_POS.to(eyepos.device)
    scaled = torch.zeros_like(eyepos) + mean_pos  # (T, 2) broadcast
```

**Do not use `eyepos.mean()` without the reshape** — if `eyepos` is shape `(T, 2)`, that call silently averages x and y together into a scalar. Use `eyepos.reshape(-1, 2).mean(dim=0)` for the offline computation.

Also add `'fixed_center'` to the docstring condition list and include the hardcoded constant in a comment noting it was computed from the full trace set.

### A1 sanity check — before running the full cache

After implementing `fixed_center`, run a quick check that all trials are actually held at the same position:

```python
# In a notebook or short script:
for i in range(5):
    stim = build_counterfactual_stim(full_stack, eyepos[i], condition='fixed_center')
    print(f"Trial {i}: first-frame center = {stim[0].mean():.4f}")  # should be identical
```

Only proceed to A2 once this passes.

### A2 — Cache rates under `fixed_center`

Script: `scripts/temporal_decoding/cache_eoptotype_rates.py`

Add `fixed_center` to the conditions list and run for lm = −0.20 and lm = −0.40:

```bash
python scripts/temporal_decoding/cache_eoptotype_rates.py \
    --logmars -0.20,-0.40 --condition fixed_center --hires_threshold 2.0
```

### A3 — Rerun pooled ablation with `fixed_center` as the stabilized arm

Script: `declan/fem_global_intervention.py`

```bash
python declan/fem_global_intervention.py \
    --logmars -0.20,-0.40 --condition real --rate_file_tag allhires_fresh
python declan/fem_global_intervention.py \
    --logmars -0.20,-0.40 --condition fixed_center --rate_file_tag allhires_fresh
```

Compare the `D1 original` vs `D1 cleaned` delta for `real` vs `fixed_center`. If the `fixed_center` delta is near zero and the `real` delta at −0.20 remains +0.027, the ablation effect is genuinely dynamic-FEM-specific.

### A4 — Rerun differential ablation

Script: `declan/fem_differential_intervention.py`

Replace the stabilized arm with `fixed_center`. The differential covariance `C_real − C_fixed_center` now captures purely dynamic FEM variance. Rerun both LogMARs.

**Pass criterion:** Real condition improves at −0.20 and `fixed_center` control does not. If the pattern holds, the ablation story is closed. If `fixed_center` still improves similarly to real, the effect is not dynamic-FEM-specific even with a proper baseline — which is itself an important finding, not a failure.

---

## C. α reversal mechanism (LogMAR sweep)

**Goal:** Track how α (alignment of FEM covariance subspace with orientation-signal subspace) changes continuously across LogMARs for both real and fixed_center conditions.

**What exists:** `fem_global_intervention.py` already computes α at each requested LogMAR via `--logmars`.

### C1 — Check cached LogMARs

```bash
ls scripts/temporal_decoding/data/rates/ | grep allhires_fresh
```

Cache any missing intermediate LogMARs before running the sweep:

```bash
python scripts/temporal_decoding/cache_eoptotype_rates.py \
    --logmars -0.10,-0.15,-0.25,-0.30,-0.35,-0.45,-0.50 \
    --condition real --hires_threshold 2.0
# And same for fixed_center after A is done
```

### C2 — Run the sweep

```bash
python declan/fem_global_intervention.py \
    --logmars -0.10,-0.15,-0.20,-0.25,-0.30,-0.35,-0.40,-0.45,-0.50 \
    --condition real --rate_file_tag allhires_fresh

python declan/fem_global_intervention.py \
    --logmars -0.10,-0.15,-0.20,-0.25,-0.30,-0.35,-0.40,-0.45,-0.50 \
    --condition fixed_center --rate_file_tag allhires_fresh
```

### C3 — Extended geometry outputs

`fem_global_intervention.py` currently saves α and the C_signal eigenspectrum. Extend the save to include the following for each LogMAR, to make the alignment-transition story fully decomposable:

- **Principal angles** between the top-2 FEM subspace U_FEM and the top-2 signal subspace U_signal (the latter computed from C_signal). Use `scipy.linalg.subspace_angles(U_FEM, U_signal)`.
- **Overlap matrix** `U_FEM.T @ U_signal` (2×2): entries tell you which FEM mode aligns with which signal mode.
- **Projection norms of class means onto U_FEM**: for each orientation k, `‖U_FEM^T (μ_k − μ̄)‖`. This measures how much orientation-discriminative information lies in the FEM subspace per class, not just in aggregate.

These can be added to the existing save dict without changing the script interface.

### C4 — Plot

Create a three-panel figure: (1) α vs LogMAR, real and fixed_center; (2) leading C_signal eigenvalue vs LogMAR; (3) leading principal angle between U_FEM and U_signal vs LogMAR. Overlay the D1 crossover point (lm ≈ −0.35) as a vertical dashed line.

**What to look for:**
- Does real α decrease monotonically as LogMAR becomes more negative?
- Does fixed_center α move in the same direction (both conditions: signal moves away from FEM subspace) or opposite (fixed_center reversal was a position-distribution artifact)?
- Does the crossover in decoding accuracy correspond to a crossover in α or in the principal angle?

---

## D. FEM amplitude scaling

**Goal:** Test whether D1 accuracy at lm = −0.40 forms an inverted-U in FEM amplitude, with the biological amplitude (1.0×) near the optimum.

### D1 — Cache rates at 0.5× and 2.0× amplitude

The `scaled_0.5` and `scaled_2.0` conditions already exist in `rate_computation.py`:

```bash
python scripts/temporal_decoding/cache_eoptotype_rates.py \
    --logmars -0.40 --condition scaled_0.5 --hires_threshold 2.0
python scripts/temporal_decoding/cache_eoptotype_rates.py \
    --logmars -0.40 --condition scaled_2.0 --hires_threshold 2.0
```

### D2 — Check retinal excursion bounds before decoding

At 2× amplitude, some traces may push the rendered stimulus outside the model's effective RF support or the spatial region covered during training. Before running decoding, log the retinal occupancy for each condition:

```python
# For each condition, compute and print:
# - Histogram of eye positions across all frames (2D, in degrees)
# - Fraction of frames where |position| > some threshold (e.g., 0.1°)
# - Min/max position per axis
```

If a substantial fraction of 2× frames are out-of-distribution (e.g., >20% of frames beyond the 99th percentile of the training distribution), interpret a 2× accuracy drop with caution — it may reflect a rendering/support artifact rather than a genuine active-sensing ceiling.

### D3 — Run D1 decoding across amplitude conditions

Conditions: `fixed_center` (0×), `scaled_0.5` (0.5×), `real` (1.0×), `scaled_2.0` (2.0×)

```bash
for cond in fixed_center scaled_0.5 real scaled_2.0; do
    python declan/fem_global_intervention.py \
        --logmars -0.40 --condition $cond --rate_file_tag allhires_fresh
done
```

Tabulate D1 accuracy and α at lm = −0.40 for each condition.

**What to look for:** An inverted-U with peak near 1.0× supports the claim that biological FEM amplitude is tuned for hyperacuity orientation decoding. A monotone increase means more FEM is always better (sampling benefit dominates and the optimum is beyond 2×). A monotone decrease would contradict the crossover story and require explanation.

---

## E. GRU passthrough at hyperacuity LogMAR

**Goal:** Run the existing GRU passthrough test on the E-optotype at lm = −0.40, to test whether GRU temporal integration is differentially relevant for near-threshold stimuli relative to natural images.

**Run this step if B suggests temporal structure may be present; treat as secondary if B3 shows C ≈ A across all collapse modes.**

### E1 — Adapt `gru_passthrough_test.py` for optotype input

File: `declan/gru_passthrough_test.py`

Add a `--stimulus` argument with choices `natural` (default) and `eoptotype`, plus `--logmar` and `--orientation` for the optotype path. When `--stimulus eoptotype`, replace the image-loading block:

```python
from scripts.temporal_decoding.stimulus_hires import hires_counterfactual_stim
stim = hires_counterfactual_stim(orientation_deg=orientation, logmar=logmar, ...)
```

The dynamic-vs-static comparison (Test 1) is stimulus-agnostic.

### E2 — Run the test

```bash
python declan/gru_passthrough_test.py \
    --stimulus eoptotype --logmar -0.40 --orientation 0 --n-frames 200
python declan/gru_passthrough_test.py \
    --stimulus eoptotype --logmar -0.20 --orientation 0 --n-frames 200
```

Compare to the natural-image result (R² dynamic vs static ≈ −50.5, RSA = 0.82).

**Interpretation caveat.** The E-optotype is a static spatial pattern with FEM-driven drift on top; natural images have genuine temporal structure (luminance changes, motion). A difference in passthrough results between the two could mean either: (a) GRU temporal integration matters more near threshold, or (b) the GRU behaves differently on static vs temporally varying inputs. To distinguish these, a useful control would be to run the passthrough on a static natural image patch (freeze the image, let FEM drift over it) — this has the same temporal structure as the E-optotype task but uses natural image content. If the passthrough result matches the E-optotype result under this control, interpretation (b) is less likely.

---

## F. Continuous forward pass — full run

**Goal:** Confirm the 32-trace pilot result at full scale (all 471 traces). Secondary step; run after B, A, C, D are complete.

**Run this step if B suggests temporal structure may be present. If B3 shows C ≈ A across all collapse modes, treat F as a secondary validation rather than a project-critical step.**

### F1 — Run at full trace count

Script: `declan/eoptotype_continuous_pass.py`

```bash
python declan/eoptotype_continuous_pass.py \
    --logmar -0.40 --windows 1,24,60 --n_traces 471
python declan/eoptotype_continuous_pass.py \
    --logmar -0.20 --windows 1,24,60 --n_traces 471
```

**Expected runtime:** ~15× longer than the 32-trace pilot. Run in a `screen`/`tmux` session or submit to the cluster.

**Pass criterion (explicit).** The pilot showed real D1 W=60 ≈ 0.414, stabilized ≈ 0.400, D3 ≈ 0.264 — compared to the windowed pipeline at the same 32 traces (real D1 W=60 = 0.611). The full run confirms the out-of-training-distribution interpretation if:
1. Both real and stabilized continuous-pass D1 accuracies remain near 0.40 (not recovering toward 0.61).
2. D3 does not exceed D1 in the continuous pass.
3. The gap between real and stabilized does not re-emerge at n=471 (which would indicate the pilot was underpowered, not degraded).

If any of these three fail, the pilot was a sampling artifact and requires a targeted investigation before drawing conclusions.

---

## G. Position-conditioned D1 — already covered by D2a (no new work)

D2a (`decode_d2a_time_mean_plus_eye` in `scripts/temporal_decoding/eoptotype_decoder_controls.py`)
concatenates the trial's continuous mean eye position (2D float vector in degrees) to the
time-averaged rate vector before fitting logistic regression. This is exactly the position-
conditioned D1 described here.

D2a was already run and shows no improvement over D1 at both lm = −0.20 and −0.40.

**Conclusion:** Position conditioning with continuous x_eye does not help. This supports the
spatial sampling account: diverse retinal positions contribute independent orientation evidence
that integrates into a position-invariant signal in time-averaged rates, leaving no residual
position information that a downstream decoder could exploit. No new experiment needed.

---

## Branch conditions summary

| After completing B... | Implication for E/F |
|---|---|
| B3: C ≈ A across all collapse modes | Temporal null is credible. E and F are secondary architecture probes, not project-critical. Run after A, C, D. |
| B3: C > A under flat or CoM | Temporal null was a representation artifact. Rebuild temporal decoding with flat features. E and F become relevant again. |
| B1/B2: hotspots flat, amax not bottleneck | Spatial collapse is not the issue. Proceed directly to B3 as confirmation. |

A runs regardless of B outcome — the ablation story is independent of the temporal null.

---

## Summary table

| Step | Requires new rates? | New code? | Blocking for other steps? |
|---|---|---|---|
| B1 — Hotspot visualisation | No | No | No |
| B2 — Step-shift test | No | No | No |
| B3 — Collapse comparison | No | No | No |
| B.5 — Matched position reweighting | No | Yes (small) | Informs A interpretation |
| A1 — Add `fixed_center` condition | — | Yes (small) | A2, A3, A4, C |
| A1 sanity check | No | No | A2 |
| A2 — Cache `fixed_center` rates | Yes | No | A3, A4, C |
| A3 — Pooled ablation rerun | No | No | — |
| A4 — Differential ablation rerun | No | No | — |
| C1 — Cache intermediate LogMARs | Yes (7 missing) | No | C2 |
| C2 — α LogMAR sweep | No | No | — |
| C3/C4 — Extended geometry + plot | No | Yes (small additions) | — |
| D — Amplitude scaling | Yes (0.5×, 2.0×) | No | — |
| E — GRU passthrough (optotype) | No | Yes (small) | — |
| F — Continuous pass full run | No | No | — |
| G — Position-conditioned D1 | — | — | **Already done via D2a** (no new work) |



Addendum: Deferred subpixel-rendering and saturation audit

Status: Important, but deliberately deferred until after the current next-step sequence.

Motivation:
The C sweep revealed that several quantities become numerically identical or nearly identical across lm = −0.40, −0.45, and −0.50: D1 accuracy, ablation Δ, and α values. This suggests that, in the current rendering/model pipeline, the effective stimulus representation may saturate below approximately lm = −0.40. This does not invalidate the hyperacuity framing. Hyperacuity is expected to operate in a subpixel regime. The concern is narrower: below some scale, further nominal reductions in LogMAR may no longer produce distinct retinal/model inputs under the current discretized rendering and resampling pipeline.

These tests should therefore be treated as a rendering and model-input audit, not as a replacement for the current mechanistic analyses. They should be run after the fixed-position baseline, α sweep, amplitude scaling, and spatial-map collapse audit, to avoid conflating two issues:

Whether FEM helps through spatial sampling and signal-nuisance alignment.
Whether the current renderer/model input preserves distinct subpixel stimuli below lm ≈ −0.40.

The goal is to determine whether the −0.40/−0.45/−0.50 plateau reflects:

true model insensitivity to increasingly small E-optotypes,
quantization/saturation in the rendered retinal stimulus,
loss during retina resampling to model-native ppd,
loss during lag embedding or rate caching,
or downstream collapse/readout insensitivity.
G. Subpixel-rendering diagnostic using existing E-optotype checks

Goal: Determine whether E-optotype structure and FEM-induced shimmer remain visible in the retinal input below lm = −0.40.

Relevant existing infrastructure:
The repo already contains E-optotype diagnostic code designed to inspect retinal movies, frame-to-frame shimmer, and neural-map propagation across E sizes. The useful outputs include:

retina_stabilized.mp4
retina_fem.mp4
retina_mean.npy
retina_std.npy
retina_delta_energy.npy
retina_xt_slice.npy
neural_maps_fem.npy
neural_maps_stabilized.npy
neural_mean_map.npy
neural_std_map.npy
com_traj.npy
width_traj.npy

The key diagnostic idea is: in the subpixel regime, stabilized movies may collapse toward a blob, but real FEM movies should still show structured shimmer if the renderer is preserving useful subpixel information. If retina_std.npy and retina_delta_energy.npy become featureless or identical across lm = −0.40 to −0.50, the current pipeline has likely hit a rendering or sampling floor.

G1 — High-retina-PPD human-inspection render

Run the diagnostic at high retinal sampling resolution so that any subpixel shimmer is visible before model-native downsampling.

Suggested LogMARs:

--logmars -0.35,-0.40,-0.45,-0.50,-0.55

Suggested settings:

--world_ppd 240
--retina_ppd 240
--retina_size 256
--orientation_deg 0
--n_frames 240
--include_matched_null

What to look for:

Does retina_fem.mp4 show structured shimmer at lm = −0.40, −0.45, and −0.50?
Does retina_std.npy retain E-like structure across those sizes?
Does retina_delta_energy.npy decrease smoothly with LogMAR, or does it plateau/collapse?
Are −0.40, −0.45, and −0.50 visually and numerically distinct at high retina ppd?

Interpretation:

Outcome	Interpretation
High-PPD movies and retina_std remain distinct below −0.40	The high-resolution renderer can represent subpixel structure; any plateau likely occurs later in model-native sampling or model responses
High-PPD movies already become identical or featureless below −0.40	The rendering itself has saturated; below −0.40 values are not distinct stimuli in the current pipeline
Real FEM retains shimmer but stabilized does not	This is the expected hyperacuity-compatible pattern
Neither real nor stabilized retains structure	No usable subpixel information is present at this stage
H. Model-native retina sampling audit

Goal: Determine whether subpixel structure survives the conversion to the model-native retinal grid.

The high-PPD diagnostic asks whether the stimulus can be rendered. The model-native diagnostic asks whether the digital twin actually receives that structure.

Suggested settings:

--logmars -0.35,-0.40,-0.45,-0.50,-0.55
--world_ppd 240
--retina_ppd 37.50476617
--retina_size 101
--orientation_deg 0
--n_frames 240
--save_neural_maps

What to compute:

For each LogMAR and condition:

Pixelwise norm differences between retinal movies.
Correlation between retinal movies across LogMARs.
Number or fraction of pixels changing across LogMARs.
retina_delta_energy.npy across LogMAR.
retina_std.npy structure across LogMAR.
Difference between real FEM and stabilized inputs.
Difference between adjacent LogMARs, especially:
−0.35 vs −0.40
−0.40 vs −0.45
−0.45 vs −0.50

Critical comparison:

If −0.40, −0.45, and −0.50 are distinct at high retina ppd but nearly identical at model-native retina ppd, then the bottleneck is likely retina resampling / model input resolution, not the high-resolution stimulus construction.

Interpretation:

Outcome	Interpretation
Model-native retinal tensors differ smoothly across LogMAR	The plateau is probably not caused by input quantization
Model-native retinal tensors are identical or nearly identical from −0.40 downward	The decoding/covariance plateau likely reflects model-input saturation
Retinal tensors differ but final rates do not	The model or readout is insensitive to these subpixel differences
Real FEM tensors differ but stabilized tensors do not	FEM is preserving subpixel structure through motion, consistent with the active-sampling account
I. Early-model and neural-map propagation audit

Goal: Determine whether subpixel differences that survive the retinal input are propagated into model features and neural readout maps.

This should be run only after G/H show that the input tensors remain distinct below −0.40.

Use the existing --save_neural_maps path to save:

neural_maps_fem.npy
neural_maps_stabilized.npy
neural_mean_map.npy
neural_std_map.npy
com_traj.npy
width_traj.npy

What to compute:

For lm = −0.35, −0.40, −0.45, −0.50:

Norm difference between neural maps across LogMAR.
Correlation between neural maps across LogMAR.
FEM-vs-stabilized neural-map difference.
CoM trajectory amplitude.
Width trajectory.
Whether map-level differences survive but collapsed scalar rates saturate.

Key question:

Does the model spatial map still encode differences below −0.40 even when D1/α/Δ are flat?

Interpretation:

Outcome	Interpretation
Retinal inputs differ and neural maps differ, but scalar rates do not	Collapse/readout is the bottleneck
Retinal inputs differ but neural maps do not	Core/readout is insensitive to subpixel differences
Neural maps differ and CoM trajectories track FEM	Spatial-map temporal code remains plausible
Neural maps are identical from −0.40 downward	The plateau is already present before decoding
J. Step-shift survival test at subpixel sizes

Goal: Use a known imposed displacement to test whether subpixel shifts survive each stage of the pipeline.

This is related to the B2 step-shift test, but the focus here is specifically on the subpixel E-size regime and the −0.40 plateau.

Suggested runs:

python scripts/temporal_decoding/step_shift_test.py --logmar -0.35 --n_traces 5 --orientation 0
python scripts/temporal_decoding/step_shift_test.py --logmar -0.40 --n_traces 5 --orientation 0
python scripts/temporal_decoding/step_shift_test.py --logmar -0.45 --n_traces 5 --orientation 0
python scripts/temporal_decoding/step_shift_test.py --logmar -0.50 --n_traces 5 --orientation 0

The most relevant shifts are:

0.5 arcmin,
1.0 arcmin,
2.0 arcmin.

What to inspect:

world image difference,
retinal input difference,
pre-collapse spatial-map difference,
post-amax difference,
post-mean difference,
amax / pre sensitivity ratio,
mean / pre sensitivity ratio.

Interpretation:

Outcome	Interpretation
Shift survives in world image but not retinal input	Retina resampling is the bottleneck
Shift survives retinal input but not neural maps	Model core/readout is insensitive
Shift survives neural maps but not amax	Spatial collapse is the bottleneck
Shift survives all stages at −0.35 but not −0.40	The saturation threshold lies between −0.35 and −0.40
Shift survives all stages even at −0.50	The plateau in D1/α/Δ reflects decoder or signal-geometry saturation, not rendering failure
K. Explicit tensor identity / near-identity check for cached rates

Goal: Determine whether the cached rate files for lm = −0.40, −0.45, and −0.50 are identical because the inputs are identical, because rates are identical, or because of an accidental caching/file-tag issue.

This is a simple sanity check and should be run before drawing any strong conclusions from the plateau.

Checks:

Confirm filenames and metadata encode the correct LogMAR.
Load cached retinal/input tensors if saved.
Load cached rate tensors for real and stabilized conditions.
Compute:
np.max(abs(rate_lm1 - rate_lm2))
np.linalg.norm(rate_lm1 - rate_lm2)
correlation across flattened rate tensors
number of exactly equal elements
Repeat for:
−0.35 vs −0.40
−0.40 vs −0.45
−0.45 vs −0.50

Interpretation:

Outcome	Interpretation
Rates are exactly identical across LogMARs	Possible cache reuse, quantized identical input, or deterministic saturation; inspect metadata and stimulus tensors
Rates differ numerically but decoder/α are identical	Plateau is functional/statistical, not literal tensor identity
Rates differ at −0.35 vs −0.40 but not below	Saturation begins at −0.40
Metadata mismatch	Fix cache naming or rate-generation path before using these results
L. How these deferred tests affect interpretation

These tests should be used to decide how to report the hyperacuity regime.

If subpixel structure survives rendering and model-native sampling below −0.40:
Then the plateau in D1/α/Δ is not a rendering failure. It reflects either model/readout insensitivity or a genuine saturation of the task-relevant population signal. In that case, the hyperacuity story can include −0.40 and below, but should say that performance saturates across these smaller sizes.

If high-PPD rendering is distinct but model-native input saturates below −0.40:
Then the current digital-twin input resolution is the bottleneck. The hyperacuity story remains conceptually valid, but the current implementation only supports it down to approximately −0.35. Pushing lower requires higher model-native retinal sampling or anti-aliased subpixel rendering that survives downsampling.

If even high-PPD rendering saturates below −0.40:
Then the current E-rendering implementation itself cannot represent further size reductions. The below−0.40 conditions should be collapsed into a single rendering-floor condition in the write-up.

If neural maps preserve differences but scalar rates do not:
Then the hyperacuity signal may exist spatially but is lost during collapse/readout. This would connect the rendering audit back to the B spatial-map audit and motivate a spatial-map decoder.

If cached rates are exactly reused or metadata are inconsistent:
Then the plateau may be a pipeline artifact rather than a scientific result. Fix caching before interpreting the C sweep or D amplitude scaling.

M. Reporting guidance before these tests are run

Until this audit is complete, phrase the hyperacuity result carefully:

The FEM benefit is clearest at the transition into the subpixel regime, especially around lm = −0.35. In the current pipeline, lm = −0.40 to −0.50 produce nearly identical decoding and geometry metrics, suggesting either stimulus/rendering saturation or model/readout saturation. We therefore treat those conditions as a subpixel plateau rather than independent size points until a rendering and model-input audit is complete.

Avoid saying:

FEM helps from −0.35 to −0.50 as though these are independent hyperacuity measurements.

Prefer:

FEM helps at the onset of the subpixel regime, and the current implementation enters a plateau for smaller nominal sizes.