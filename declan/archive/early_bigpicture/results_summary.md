# FEM Population Dynamics: Results Summary

**Digital twin:** learned_resnet_none_convgru_gaussian, epoch 147, val BPS = 0.5702  
**Stimuli:** 6 natural images (BeachWave, Colony_Bonnie, Colony_PJ, Hawaii_trees, LakeOntario_Rocks1, SD_Orangatang)  
**Date compiled:** 2026-04-17

---

## Overview

Four analyses test what variables FEM-driven V1 population dynamics encode, across four representational formats.

| Analysis | Question | Representation tested | Result |
|----------|----------|-----------------------|--------|
| 1. Temporal decoding (C vs A) | Do temporal features add to rate code, and does FEM help? | Rate code + temporal residual features | Model C does not improve over A; FEM effect is regime-dependent (hurts near −0.20, helps for −0.35…−0.50 at long windows; both collapse by −0.55) |
| 2. Transformation dynamics (B ≈ 0) | Do latent dynamics encode eye velocity? | PCA latent space (z) | B ≈ 0 — null |
| 3. Displacement decoding (Phase 1–2) | Do response differences encode retinal displacement? | Scalar rates + spatial moments | Strong within-image; fails cross-image |
| 4. CoM dynamics | Do spatial map features track eye velocity? | CoM + width moments | In progress |

---

## Analysis 1: Temporal Decoding (C vs A)

**Task:** 4-way orientation classification (E-like stimuli, 4 orientations).  
**Conditions:** stabilized (no eye movement) vs real FEM eye traces.  
**Model A:** rate-only baseline (time-averaged rate per trial). **Model C:** Model A + single-trial temporal residual PCA features.

Neurometric curves and fitted thresholds (criterion = 0.625) are saved in:
- scripts/temporal_decoding/figures/fig_neurometric_cached_dual_regime.png
- scripts/temporal_decoding/data/results/neurometric_cached_dual_regime.npz

Staleness note (important): several downstream “mechanistic” caches in the temporal-decoding suite depend on the Phase-3 `--threshold_logmar` setting (e.g., covariances, intervention, budget/entropy-type summaries). Any such results generated at a higher `threshold_logmar` (often default 0.0, or the near-threshold −0.20 regime) should be treated as **stale** with respect to the FEM-help regime (≈ −0.35…−0.50) and should be rerun at a LogMAR in that regime if we want mechanistic conclusions to track “where FEM helps”.

Key fitted thresholds (LogMAR at criterion):
- stabilized / Model A: ≤ −0.500 within the cached sweep (still above criterion at −0.50)
- real FEM / Model C: −0.199
- ΔLogMAR (stabilized/A − real/C): ≤ −0.301 within this criterion-fit setup (interpret cautiously given the non-monotonic sweep and the −0.60 collapse evidence)

Important update (chance boundary): extending beyond −0.50 shows that performance collapses by −0.55 (and remains at chance at −0.60; see “1.0b”). This means the “≤ −0.500” statement is an artifact of stopping the sweep at −0.50; the true lower bound is between −0.50 and −0.55 in this rendering/decoder regime.

Per-LogMAR decoding accuracies (mean ± std across CV folds; dual-regime cached sweep):

| LogMAR | stabilized A | stabilized C | real A | real C |
|---:|---:|---:|---:|---:|
| +1.00 | 0.999 ± 0.001 | 0.999 ± 0.001 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| +0.80 | 0.998 ± 0.002 | 0.998 ± 0.002 | 0.998 ± 0.003 | 0.998 ± 0.003 |
| +0.60 | 0.997 ± 0.003 | 0.997 ± 0.003 | 0.998 ± 0.003 | 0.998 ± 0.003 |
| +0.50 | 0.998 ± 0.003 | 0.998 ± 0.002 | 0.997 ± 0.004 | 0.997 ± 0.004 |
| +0.40 | 0.998 ± 0.001 | 0.997 ± 0.001 | 0.997 ± 0.003 | 0.998 ± 0.001 |
| +0.30 | 0.994 ± 0.002 | 0.994 ± 0.001 | 0.998 ± 0.002 | 0.998 ± 0.001 |
| +0.20 | 0.992 ± 0.003 | 0.992 ± 0.003 | 0.997 ± 0.002 | 0.996 ± 0.001 |
| +0.10 | 0.989 ± 0.003 | 0.990 ± 0.004 | 0.991 ± 0.005 | 0.989 ± 0.005 |
| +0.00 | 0.973 ± 0.003 | 0.960 ± 0.003 | 0.981 ± 0.006 | 0.971 ± 0.007 |
| -0.10 | 0.955 ± 0.008 | 0.933 ± 0.009 | 0.938 ± 0.005 | 0.916 ± 0.009 |
| -0.15 | 0.948 ± 0.009 | 0.927 ± 0.003 | 0.930 ± 0.003 | 0.912 ± 0.009 |
| -0.20 | 0.758 ± 0.016 | 0.706 ± 0.015 | 0.658 ± 0.026 | 0.617 ± 0.007 |
| -0.25 | 0.761 ± 0.018 | 0.722 ± 0.013 | 0.658 ± 0.024 | 0.619 ± 0.006 |
| -0.30 | 0.865 ± 0.003 | 0.821 ± 0.012 | 0.813 ± 0.018 | 0.783 ± 0.016 |
| -0.35 | 0.837 ± 0.028 | 0.803 ± 0.024 | 0.890 ± 0.018 | 0.872 ± 0.023 |
| -0.40 | 0.842 ± 0.030 | 0.803 ± 0.023 | 0.892 ± 0.020 | 0.875 ± 0.021 |
| -0.45 | 0.840 ± 0.028 | 0.809 ± 0.032 | 0.892 ± 0.019 | 0.877 ± 0.022 |
| -0.50 | 0.842 ± 0.027 | 0.812 ± 0.025 | 0.892 ± 0.019 | 0.876 ± 0.022 |

### 1.0b How far negative until chance?

Because the neurometric sweep above ends at −0.50 (and is non-monotonic in this range), we ran standalone decode-only probes at more negative LogMAR using separately tagged rate caches (so we do not overwrite the main sweep caches).

Tagged boundary artifacts (n=100 traces):
- scripts/temporal_decoding/data/results/neurometric_cached_dual_regime_n100_boundary.npz
- scripts/temporal_decoding/figures/fig_neurometric_cached_dual_regime_n100_boundary.png

Probe results using 100 traces (hi-res caches tagged `_n100`; 4-class chance = 0.25):

| LogMAR | cache tag | stabilized A | stabilized C | real A | real C |
|---:|:---:|---:|---:|---:|---:|
| -0.50 | n100 | 0.672 ± 0.022 | 0.575 ± 0.053 | 0.800 ± 0.038 | 0.778 ± 0.034 |
| -0.55 | n100 | 0.250 ± 0.000 | 0.250 ± 0.000 | 0.250 ± 0.000 | 0.250 ± 0.000 |
| -0.60 | n100 | 0.250 ± 0.000 | 0.250 ± 0.000 | 0.250 ± 0.000 | 0.250 ± 0.000 |

Conclusion: for this stimulus generator / discretization regime, accuracy is still high at −0.50 but at chance by −0.55. There is no value in running more negative LogMAR without changing how the E is rendered (subpixel/anti-aliasing), because orientations become indistinguishable on the hi-res grid.

### 1.1 Accuracy at matched integration windows

Window = number of frames integrated before classification.

This integration-time sweep uses the accumulation-aligned time_mean featurization (mean rate over the last W frames) and compares conditions (real vs stabilized).

Artifacts:
- scripts/temporal_decoding/data/results/integration_time_time_mean.pkl
- scripts/temporal_decoding/figures/fig_integration_time_time_mean.png

| Window | Duration | Stabilized acc | Real FEM acc |
|--------|----------|--------------------------|-------------------|
| 1 frame | 8 ms | 0.768 ± 0.009 | 0.296 ± 0.005 |
| 6 frames | 50 ms | 0.770 ± 0.008 | 0.365 ± 0.020 |
| 24 frames | 200 ms | 0.767 ± 0.015 | 0.522 ± 0.014 |
| 60 frames | 500 ms | 0.769 ± 0.014 | 0.646 ± 0.029 |

Stabilized outperforms real FEM at all integration windows (no crossover) at this near-threshold LogMAR. Under stabilization the retinal input is nearly time-invariant, so increasing the averaging window W changes little (accuracy stays ~0.77). Under real FEM, a single-frame snapshot is near chance (0.296 at W=1), but longer averaging windows partially recover performance (0.646 at W=60).

Controls / provenance:
- The same pattern is reproduced by an independent decode-only script that loads the cached `.npz` rates directly: `scripts/temporal_decoding/integration_time_controls.py` at LogMAR −0.20.
- At an easier LogMAR (+0.40), the same time_mean decoder is near ceiling for both real and stabilized (~0.98–0.998), confirming that “0.7–0.8” values are not a universal ceiling but reflect the specific operating point and protocol.

Important caveat (validity vs reproducibility): these curves are internally consistent and reproducible, but by themselves they do not establish *why* stabilized > real here (e.g., whether this reflects a genuine representational limitation vs a mismatch between decoder/featurization and the temporal structure of the real-FEM response).

### 1.1b Spatial-information control on the E-optotype (representation diagnostic)

Because the temporal decoder suggests stabilized > real at near-threshold LogMAR (−0.20), but prior work on fixRSVP suggested FEM can *increase* representation-level spatial/Fisher-style information, we ran a direct spatial single-spike information (SSI) comparison on the E-optotype itself.

Method:
- Reuse the existing spatial-information machinery (`scripts/spatial_info.py`: `spatial_ssi_population`) and the same cached eye traces used by the temporal-decoding stack (`scripts/temporal_decoding/data/eye_traces.npz`).
- Use the dual-regime high-resolution E pipeline (`scripts/temporal_decoding/stimulus_hires.py:hires_counterfactual_stim`) at LogMAR −0.20 to preserve subpixel orientation structure.
- Compare real FEM vs stabilized (same mean position, no motion) over a 60-frame window (0.5 s). We ran both (i) a pilot at 0° using 5 traces and (ii) a full 4-orientation sweep (0/90/180/270°) averaging across 20 traces.

Result (population SSI; nanmean over traces, T = 60 frames):

| Orientation | cum_bits real | cum_bits stabilized | Δ cum_bits | mean bits/spike real | mean bits/spike stabilized | Δ bits/spike |
|---:|---:|---:|---:|---:|---:|---:|
| 0° | 0.543 | 0.463 | +0.080 | 0.0091 | 0.0076 | +0.0015 |
| 90° | 0.541 | 0.459 | +0.082 | 0.0090 | 0.0075 | +0.0015 |
| 180° | 0.541 | 0.462 | +0.079 | 0.0090 | 0.0076 | +0.0015 |
| 270° | 0.542 | 0.463 | +0.080 | 0.0091 | 0.0076 | +0.0015 |

Artifact:
- Orientation sweep (n=20):
	- `scripts/temporal_decoding/data/results/eoptotype_spatial_ssi_lm-0.20_ori0_T60_n20.npz`
	- `scripts/temporal_decoding/data/results/eoptotype_spatial_ssi_lm-0.20_ori90_T60_n20.npz`
	- `scripts/temporal_decoding/data/results/eoptotype_spatial_ssi_lm-0.20_ori180_T60_n20.npz`
	- `scripts/temporal_decoding/data/results/eoptotype_spatial_ssi_lm-0.20_ori270_T60_n20.npz`
- Pilot (0°, n=5): `scripts/temporal_decoding/data/results/eoptotype_spatial_ssi_lm-0.20_ori0_T60_n5.npz`

Interpretation: at least under this representation-level SSI metric (and using the same eye traces / near-threshold LogMAR regime), real FEM *increases* spatial information relative to stabilized. This suggests the stabilized>real temporal-decoding result may reflect a decoder/featurization mismatch (or task-specific readout limitations) rather than a simple “FEM removes information” story.

### 1.1c Decoder controls (D1–D3) (readout hypothesis test)

Goal: test whether the stabilized>real gap near threshold reflects (i) a poor readout/featurization choice for real FEM, or (ii) a genuine loss of orientation-readout performance under FEM even when the decoder is “eye-aware”.

Decoders:
- D1: time_mean rates (mean over last W frames).
- D2a: concat([rate_mean, eye_mean]) and decode with the same linear classifier.
- D2b: eye-conditioned response correction (not triggered here; see below).
- D3: supervised mean-trajectory subspace (triggered here).

LogMAR −0.20 artifacts:
- scripts/temporal_decoding/data/results/decoder_controls_lm-0.20.npz
- scripts/temporal_decoding/figures/fig_decoder_controls_D1_lm-0.20.png
- scripts/temporal_decoding/figures/fig_decoder_controls_D2_lm-0.20.png

Summary table (mean ± std across folds; grouped CV by trace; LogisticRegression L2, C=1.0):

| Decoder | W=1 | W=24 | W=60 |
|---|---:|---:|---:|
| D1 real | 0.296 ± 0.005 | 0.522 ± 0.014 | 0.646 ± 0.029 |
| D1 stabilized | 0.768 ± 0.009 | 0.767 ± 0.015 | 0.769 ± 0.014 |
| D2a real | 0.293 ± 0.009 | 0.513 ± 0.013 | 0.638 ± 0.030 |
| D3 real | 0.260 ± 0.004 | 0.273 ± 0.002 | 0.286 ± 0.012 |

Auto-selection logic:
- max(D2a_real − D1_real) = +0.007 → D2b not run.
- max(stabilized D1) − max(real best) = +0.124 → D3 run.

Interpretation: none of these readouts (D1 or the eye-aware D2a) bring real FEM close to stabilized at LogMAR −0.20, despite SSI increasing under FEM. This supports the “SSI gain is mostly position/spatial diversity rather than orientation-readout gain” interpretation at this operating point.

LogMAR −0.15 artifacts:
- scripts/temporal_decoding/data/results/decoder_controls_lm-0.15.npz
- scripts/temporal_decoding/figures/fig_decoder_controls_D1_lm-0.15.png
- scripts/temporal_decoding/figures/fig_decoder_controls_D2_lm-0.15.png

Summary table (mean ± std across folds; grouped CV by trace; LogisticRegression L2, C=1.0):

| Decoder | W=1 | W=24 | W=60 |
|---|---:|---:|---:|
| D1 real | 0.439 ± 0.017 | 0.859 ± 0.015 | 0.926 ± 0.011 |
| D1 stabilized | 0.956 ± 0.013 | 0.954 ± 0.013 | 0.954 ± 0.015 |
| D2a real | 0.439 ± 0.020 | 0.859 ± 0.015 | 0.923 ± 0.011 |
| D3 real | 0.317 ± 0.021 | 0.337 ± 0.014 | 0.400 ± 0.013 |

Auto-selection logic:
- max(D2a_real − D1_real) = +0.000 → D2b not run.
- max(stabilized D1) − max(real best) = +0.031 → D3 run.

Interpretation: at LogMAR −0.15, real FEM gets much closer to stabilized under the simple D1 time-mean readout (0.926 vs 0.956 at W=60), but still does not exceed stabilized, and “eye-aware” D2a does not help. This is consistent with the same qualitative conclusion as LogMAR −0.20, but with a much smaller remaining gap.

LogMAR −0.25 artifacts:
- scripts/temporal_decoding/data/results/decoder_controls_lm-0.25.npz
- scripts/temporal_decoding/figures/fig_decoder_controls_D1_lm-0.25.png
- scripts/temporal_decoding/figures/fig_decoder_controls_D2_lm-0.25.png

Summary table (mean ± std across folds; grouped CV by trace; LogisticRegression L2, C=1.0):

| Decoder | W=1 | W=24 | W=60 |
|---|---:|---:|---:|
| D1 real | 0.290 ± 0.006 | 0.525 ± 0.012 | 0.654 ± 0.033 |
| D1 stabilized | 0.769 ± 0.014 | 0.776 ± 0.012 | 0.773 ± 0.010 |
| D2a real | 0.291 ± 0.011 | 0.524 ± 0.015 | 0.650 ± 0.031 |
| D3 real | 0.260 ± 0.009 | 0.280 ± 0.007 | 0.284 ± 0.006 |

Auto-selection logic:
- max(D2a_real − D1_real) = +0.009 → D2b not run.
- max(stabilized D1) − max(real best) = +0.122 → D3 run.

Interpretation: LogMAR −0.25 is qualitatively similar to −0.20: the best real decoder (D1/D2a) remains far below stabilized across windows, and adding eye_mean (D2a) yields at most a negligible improvement.

LogMAR −0.30 artifacts:
- scripts/temporal_decoding/data/results/decoder_controls_lm-0.30.npz
- scripts/temporal_decoding/figures/fig_decoder_controls_D1_lm-0.30.png
- scripts/temporal_decoding/figures/fig_decoder_controls_D2_lm-0.30.png

Summary table (mean ± std across folds; grouped CV by trace; LogisticRegression L2, C=1.0):

| Decoder | W=1 | W=24 | W=60 |
|---|---:|---:|---:|
| D1 real | 0.358 ± 0.008 | 0.709 ± 0.022 | 0.828 ± 0.016 |
| D1 stabilized | 0.874 ± 0.034 | 0.876 ± 0.026 | 0.875 ± 0.029 |
| D2a real | 0.355 ± 0.007 | 0.708 ± 0.025 | 0.832 ± 0.016 |
| D3 real | 0.286 ± 0.009 | 0.306 ± 0.006 | 0.347 ± 0.024 |

Auto-selection logic:
- max(D2a_real − D1_real) = +0.004 → D2b not run.
- max(stabilized D1) − max(real best) = +0.050 → D3 run.

Interpretation: at LogMAR −0.30, both real and stabilized are still far above chance (4-class chance = 0.25) under the simple D1 time-mean readout, with real improving substantially with longer windows. Stabilized remains higher than real at all tested windows.

LogMAR −0.35 artifacts:
- scripts/temporal_decoding/data/results/decoder_controls_lm-0.35.npz
- scripts/temporal_decoding/figures/fig_decoder_controls_D1_lm-0.35.png
- scripts/temporal_decoding/figures/fig_decoder_controls_D2_lm-0.35.png

Summary table (mean ± std across folds; grouped CV by trace; LogisticRegression L2, C=1.0):

| Decoder | W=1 | W=24 | W=60 |
|---|---:|---:|---:|
| D1 real | 0.391 ± 0.010 | 0.787 ± 0.029 | 0.891 ± 0.012 |
| D1 stabilized | 0.842 ± 0.031 | 0.836 ± 0.030 | 0.838 ± 0.027 |
| D2a real | 0.386 ± 0.010 | 0.782 ± 0.026 | 0.886 ± 0.012 |

Auto-selection logic:
- max(D2a_real − D1_real) = +0.003 → D2b not run.
- max(stabilized D1) − max(real best) = −0.049 → D3 not run.

Interpretation: at LogMAR −0.35, real FEM surpasses stabilized at longer integration windows (W=60: 0.891 vs 0.838), yielding a clear crossover relative to the stabilized baseline. This is consistent with the “multi-position sampling adds usable orientation information” regime emerging at this harder (more negative) LogMAR.

LogMAR −0.40 artifacts:
- scripts/temporal_decoding/data/results/decoder_controls_lm-0.40.npz
- scripts/temporal_decoding/figures/fig_decoder_controls_D1_lm-0.40.png
- scripts/temporal_decoding/figures/fig_decoder_controls_D2_lm-0.40.png

Summary table (mean ± std across folds; grouped CV by trace; LogisticRegression L2, C=1.0):

| Decoder | W=1 | W=24 | W=60 |
|---|---:|---:|---:|
| D1 real | 0.394 ± 0.013 | 0.789 ± 0.023 | 0.894 ± 0.015 |
| D1 stabilized | 0.843 ± 0.033 | 0.834 ± 0.028 | 0.841 ± 0.030 |
| D2a real | 0.392 ± 0.010 | 0.786 ± 0.027 | 0.893 ± 0.013 |

Auto-selection logic:
- max(D2a_real − D1_real) = −0.001 → D2b not run.
- max(stabilized D1) − max(real best) = −0.051 → D3 not run.

Interpretation: at LogMAR −0.40, the crossover persists: real FEM exceeds stabilized at long windows (W=60: 0.894 vs 0.841). Adding eye_mean (D2a) does not improve over the simple D1 time-mean readout.

LogMAR −0.45 artifacts:
- scripts/temporal_decoding/data/results/decoder_controls_lm-0.45.npz
- scripts/temporal_decoding/figures/fig_decoder_controls_D1_lm-0.45.png
- scripts/temporal_decoding/figures/fig_decoder_controls_D2_lm-0.45.png

Summary table (mean ± std across folds; grouped CV by trace; LogisticRegression L2, C=1.0):

| Decoder | W=1 | W=24 | W=60 |
|---|---:|---:|---:|
| D1 real | 0.395 ± 0.013 | 0.791 ± 0.025 | 0.894 ± 0.015 |
| D1 stabilized | 0.845 ± 0.032 | 0.835 ± 0.031 | 0.840 ± 0.029 |
| D2a real | 0.392 ± 0.013 | 0.786 ± 0.028 | 0.891 ± 0.014 |

Auto-selection logic:
- max(D2a_real − D1_real) = −0.001 → D2b not run.
- max(stabilized D1) − max(real best) = −0.049 → D3 not run.

Interpretation: at LogMAR −0.45, real FEM again surpasses stabilized at long windows (W=60: 0.894 vs 0.840), with negligible benefit from including eye_mean.

LogMAR −0.50 artifacts:
- scripts/temporal_decoding/data/results/decoder_controls_lm-0.50.npz
- scripts/temporal_decoding/figures/fig_decoder_controls_D1_lm-0.50.png
- scripts/temporal_decoding/figures/fig_decoder_controls_D2_lm-0.50.png

Summary table (mean ± std across folds; grouped CV by trace; LogisticRegression L2, C=1.0):

| Decoder | W=1 | W=24 | W=60 |
|---|---:|---:|---:|
| D1 real | 0.393 ± 0.013 | 0.791 ± 0.023 | 0.895 ± 0.013 |
| D1 stabilized | 0.844 ± 0.031 | 0.837 ± 0.029 | 0.841 ± 0.029 |
| D2a real | 0.391 ± 0.012 | 0.784 ± 0.028 | 0.893 ± 0.012 |

Auto-selection logic:
- max(D2a_real − D1_real) = −0.002 → D2b not run.
- max(stabilized D1) − max(real best) = −0.051 → D3 not run.

Interpretation: at LogMAR −0.50, the pattern remains unchanged: real FEM exceeds stabilized at long windows (W=60: 0.895 vs 0.841), and “eye-aware” D2a is not better than D1.

#### Nonlinear D1 control: MLP on the same time-mean feature

Success criterion: does the nonlinear readout “rescue” real FEM more than stabilized?

Define:
- rescue_real(W) = acc_MLP_real(W) − acc_linear_real(W)
- rescue_stabilized(W) = acc_MLP_stabilized(W) − acc_linear_stabilized(W)
- key metric = max_W [ rescue_real(W) − rescue_stabilized(W) ]

Model:
- Input = same D1 feature (time-mean rate vector over last W frames)
- MLP: 128 → 64 (ReLU) → 4 logits; dropout = 0.15
- Loss: cross-entropy; optimizer: AdamW; weight decay = 1e-4
- Early stopping (within each CV fold): max_epochs=100, patience=10, val_frac=0.2 (split by groups)
- CV: same grouped CV by eye trace as before; StandardScaler per fold

LogMAR −0.20 artifacts:
- scripts/temporal_decoding/data/results/decoder_controls_mlp_only_lm-0.20.npz
- scripts/temporal_decoding/figures/fig_decoder_controls_D1_mlp_only_lm-0.20.png

| Decoder | W=1 | W=24 | W=60 |
|---|---:|---:|---:|
| D1 real (linear) | 0.297 ± 0.005 | 0.522 ± 0.014 | 0.647 ± 0.028 |
| D1 stabilized (linear) | 0.769 ± 0.009 | 0.768 ± 0.016 | 0.769 ± 0.015 |
| D1 real (MLP) | 0.281 ± 0.024 | 0.424 ± 0.020 | 0.528 ± 0.023 |
| D1 stabilized (MLP) | 0.784 ± 0.003 | 0.759 ± 0.022 | 0.772 ± 0.022 |

Rescue metric (at LogMAR −0.20):
- max(rescue_real) = −0.015
- max(rescue_stabilized) = +0.015
- max(rescue_real − rescue_stabilized) = −0.028

LogMAR −0.15 artifacts:
- scripts/temporal_decoding/data/results/decoder_controls_mlp_only_lm-0.15.npz
- scripts/temporal_decoding/figures/fig_decoder_controls_D1_mlp_only_lm-0.15.png

| Decoder | W=1 | W=24 | W=60 |
|---|---:|---:|---:|
| D1 real (linear) | 0.440 ± 0.018 | 0.859 ± 0.015 | 0.927 ± 0.012 |
| D1 stabilized (linear) | 0.956 ± 0.013 | 0.954 ± 0.013 | 0.956 ± 0.014 |
| D1 real (MLP) | 0.412 ± 0.027 | 0.788 ± 0.016 | 0.899 ± 0.007 |
| D1 stabilized (MLP) | 0.950 ± 0.017 | 0.948 ± 0.019 | 0.947 ± 0.017 |

Rescue metric (at LogMAR −0.15):
- max(rescue_real) = −0.028
- max(rescue_stabilized) = −0.006
- max(rescue_real − rescue_stabilized) = −0.019

LogMAR −0.25 artifacts:
- scripts/temporal_decoding/data/results/decoder_controls_mlp_only_lm-0.25.npz
- scripts/temporal_decoding/figures/fig_decoder_controls_D1_mlp_only_lm-0.25.png

| Decoder | W=1 | W=24 | W=60 |
|---|---:|---:|---:|
| D1 real (linear) | 0.292 ± 0.007 | 0.526 ± 0.011 | 0.654 ± 0.032 |
| D1 stabilized (linear) | 0.770 ± 0.013 | 0.776 ± 0.011 | 0.772 ± 0.010 |
| D1 real (MLP) | 0.275 ± 0.013 | 0.426 ± 0.015 | 0.538 ± 0.018 |
| D1 stabilized (MLP) | 0.762 ± 0.010 | 0.771 ± 0.013 | 0.761 ± 0.014 |

Rescue metric (at LogMAR −0.25):
- max(rescue_real) = −0.016
- max(rescue_stabilized) = +0.021
- max(rescue_real − rescue_stabilized) = −0.009

Interpretation: the nonlinear readout does *not* rescue real FEM more than stabilized in any of these regimes (rescue-difference metric is negative for −0.15/−0.20/−0.25). In fact, the MLP tends to degrade real-FEM accuracy relative to the linear D1 baseline while leaving stabilized unchanged or slightly improved.

### 1.2 Older intervention cache (stale / not the revised Priority 2 test)

This intervention result is cached separately (`intervention.pkl`) and is not directly comparable to the near-threshold integration-time sweep above: it is computed under the Phase 3 “threshold_logmar” setting used when the cache was generated (often the default LogMAR = 0.0), and it uses the ablation-ladder featurizations (Model A = mean rate over the full available window; Model C = Model A + temporal residual PCA features).

| Condition | Model A acc | Model C acc |
|-----------|-------------|-------------|
| Original (full response) | 0.950 ± 0.061 | 0.963 ± 0.065 |
| Cleaned (content regressed out) | 0.200 ± 0.094 | 0.300 ± 0.094 |
| Drop on cleaning | −0.750 | −0.663 |

Both models collapse near chance (4-class chance = 0.25) when content is regressed out. The slight residual advantage of C over A (0.30 vs 0.20) after cleaning is marginally above chance but not a reliable temporal signal.

### 1.2b Revised Priority 2: pooled translation-subspace ablation

We reran the causal intervention using the revised Priority 2 script on the fresh all-hires caches:

- `declan/fem_global_intervention.py`
- `declan/fem_global_intervention_results/fem_global_intervention_real.npz`
- `declan/fem_global_intervention_results/fem_global_intervention_stabilized.npz`

Method:
- Fit a pooled within-orientation covariance `C_FEM` from trial-mean responses.
- Remove its top-2 subspace `U_FEM` inside each CV fold.
- Rerun the D1 decoder before and after ablation.

Results:

| Condition | LogMAR | α | D1 original | D1 cleaned | Δ |
|-----------|---:|---:|---:|---:|---:|
| real | -0.20 | 0.689 | 0.747 ± 0.015 | 0.773 ± 0.013 | +0.027 |
| real | -0.40 | 0.559 | 0.936 ± 0.009 | 0.936 ± 0.011 | +0.000 |
| stabilized | -0.20 | 0.052 | 0.774 ± 0.020 | 0.803 ± 0.018 | +0.029 |
| stabilized | -0.40 | 0.666 | 0.840 ± 0.030 | 0.855 ± 0.029 | +0.015 |

Interpretation:
- The real-condition ordering is consistent with the alignment story: stronger effect at −0.20, null at −0.40.
- But the stabilized control also improves after the same ablation, so the removed subspace is not specific to dynamic FEM variability.
- In this pipeline, `stabilized` still varies across trials because each trace is held at its own mean eye position, preserving a low-rank translation nuisance. The intervention is therefore best interpreted as removing a shared positional / displacement subspace rather than a uniquely FEM-driven covariance mode.

Bottom line: the revised Priority 2 result supports a translation-alignment account, but it does **not** by itself prove that dynamic FEM covariance is the causal source of the crossover.

### 1.2c Differential covariance ablation: real minus stabilized

To test whether a covariance component unique to dynamic FEMs was being masked by the shared translation nuisance, we ran a second intervention using the positive eigenspace of `C_real - C_stabilized` fit inside each CV fold:

- `declan/fem_differential_intervention.py`
- `declan/fem_differential_intervention_results/fem_differential_intervention.npz`

Results:

| LogMAR | mean positive eigvals(`C_real - C_stabilized`) | real Δ | stabilized Δ |
|---:|---:|---:|---:|
| -0.20 | [0.003668, 0.001265] | +0.016 | +0.017 |
| -0.40 | [0.000859, 0.000293] | +0.002 | -0.002 |

Interpretation:
- At −0.20, removing the differential subspace helps real and stabilized by essentially the same amount.
- At −0.40, the effect is null in both directions.
- So even the real-minus-stabilized covariance component does not yield a clean real-specific causal intervention under the current representation and decoder.

Bottom line: the covariance-ablation branch is now substantially constrained. The data still support a first-order spatial-sampling account of the crossover, but not a strong claim that dynamic FEM covariance is the uniquely causal ingredient.

### 1.2d Priority 3 pilot: continuous forward pass at LogMAR -0.40

We implemented a dedicated continuous-pass analysis:

- `declan/eoptotype_continuous_pass.py`
- `declan/continuous_pass_results/continuous_pass_lm-0.40_n32.npz`

This feeds each trial as one continuous hi-res movie through the frontend, convnet, and GRU, then decodes from the resulting continuous population trajectory.

32-trace pilot results:

| Condition | D1 W=1 | D1 W=24 | D1 W=60 | D3 W=60 |
|---|---:|---:|---:|---:|
| real | 0.380 ± 0.062 | 0.355 ± 0.093 | 0.414 ± 0.057 | 0.264 ± 0.039 |
| stabilized | 0.415 ± 0.058 | 0.445 ± 0.042 | 0.400 ± 0.051 | 0.302 ± 0.052 |

Matched-subset control (same 32 traces, standard cached-window pipeline):

| Condition | cached D1 W=1 | cached D1 W=24 | cached D1 W=60 | cached D3 W=60 |
|---|---:|---:|---:|---:|
| real | 0.345 ± 0.033 | 0.538 ± 0.057 | 0.611 ± 0.082 | 0.311 ± 0.037 |
| stabilized | 0.500 ± 0.064 | 0.467 ± 0.076 | 0.461 ± 0.035 | 0.320 ± 0.038 |

Interpretation:
- The standard windowed pipeline still shows the expected crossover on this subset (`real > stabilized` at W=60).
- The continuous pass suppresses that advantage and lowers both conditions toward ~0.4 accuracy.
- D3 remains below D1 in both cases.

Current reading: the continuous forward pass does **not** uncover a stronger temporal code for orientation discrimination. In this pilot, it looks more like an out-of-training-regime degradation than a hidden temporal-integration benefit.

### 1.3 FEM gain by motion magnitude

FEM gain = acc(Model C) − acc(Model A). Stimuli binned by retinal motion RMS.

| Motion bin | RMS (mean) | n stimuli | acc(A) | acc(C) | Gain |
|------------|------------|-----------|--------|--------|------|
| High | 0.222 | 7 | 1.000 | 0.500 | −0.500 |
| Medium | 0.139 | 6 | 0.875 | 0.656 | −0.219 |
| Low | 0.057 | 7 | 0.938 | 0.594 | −0.344 |

Gain is **negative at all motion levels** — adding temporal residual features (Model C) does not help orientation decoding, regardless of how much the eye moves.

**Interpretation:** In this dataset, real FEM traces reduce acuity relative to stabilized (ΔLogMAR ≤ −0.30 for the stabilized/A vs real/C comparison), and temporal residual features (Model C) do not provide a consistent gain over the rate-only baseline (Model A). The integration-time sweep shows real FEM responses benefit from longer accumulation windows, but at the near-threshold LogMAR (−0.20) they still do not catch up to stabilized performance.

Refinement: “FEM vs stabilized” depends on regime. Near LogMAR −0.20 (the near-threshold decoder-controls regime), stabilized is substantially better than real FEM under time-mean decoding. For harder stimuli in the −0.35…−0.50 range, real FEM can surpass stabilized at long integration windows (decoder-controls), even though both are far above chance. By −0.60, both conditions are at chance (1.0b), so FEM provides no benefit.

---

## Analysis 2: Transformation Dynamics (B ≈ 0)

**Setup:** Fit a linear dynamical system z(t+1) = A·z(t) + B·v(t) + c to the 2D PCA-projected latent state z, where v(t) = eye velocity. Test whether the B (velocity-driven input) term contributes meaningful variance.

**Latent space:** top-2 PCA components of population rate vector, per stimulus.

### 2.1 Variance explained: full model vs velocity term alone

| Stimulus | R² (full: A + B + c) | R² (B only) | Spectral radius of A |
|----------|----------------------|-------------|----------------------|
| BeachWave.jpg | 0.910 | 0.000116 | 0.954 |
| Colony_Bonnie.JPG | 0.908 | 0.000205 | 0.958 |
| Colony_PJ.JPG | 0.934 | 0.000495 | 0.963 |
| Hawaii_trees.JPG | 0.939 | 0.000051 | 0.966 |
| LakeOntario_Rocks1.JPG | 0.934 | 0.000095 | 0.960 |
| SD_Orangatang.JPG | 0.922 | 0.000128 | 0.963 |
| **Mean** | **0.925** | **0.000182** | **0.961** |

The A matrix (autoregressive term) explains ~92% of variance in z. The B term (eye velocity input) explains < 0.05% — less than 1/5000th of total variance.

### 2.2 Direct velocity decoding from latent state

| Feature | Task | R² |
|---------|------|----|
| Δz (latent state change) | Decode eye velocity | ~0.000 |
| Mean rate | Decode eye velocity | ~0.000 |
| Mean rate | Decode stimulus identity (6-class) | 1.000 |
| Δz | Decode stimulus identity (6-class) | 0.220 |

Eye velocity is not decodable from the latent state (R² ≈ 0 in both directions). Mean rates perfectly decode stimulus identity, confirming the representation is rich — velocity information is simply absent.

**Interpretation:** The 2D latent dynamics are dominated by slow attractor dynamics intrinsic to the network (spectral radius ~0.96, time constant ~25 frames). Eye velocity produces no detectable perturbation to this trajectory. The scalar latent space discards the spatial structure needed to represent eye position or velocity.

---

## Analysis 3: Displacement Decoding

**Setup:** Compute population responses at a 11×11 grid of eye positions spanning ±0.05° in 0.01° steps (121 positions, 6 images). Train ridge regression to predict displacement (δx, δy) from response differences Δr = r(p₂) − r(p₁).

**Feature sets:**
- `feat_scalar`: Δ of amax-collapsed rates — (N,) per pair
- `feat_com`: Δ of CoM features only — (2N,) per pair
- `feat_width`: Δ of width features only — (2N,) per pair  
- `feat_moments`: Δ of all spatial moments (CoM + width) — (4N,) per pair

**Cross-validation:** Phase 1 = within-image 5-fold CV. Phase 2 = leave-one-image-out.

### 3.1 Phase 1: Within-image displacement decoding

Mean R² across 6 stimuli (predicting δx, δy jointly):

| Feature set | Mean R² | Min | Max | Permutation null 95th pct |
|-------------|---------|-----|-----|--------------------------|
| feat_scalar | 0.9975 | 0.9950 | 0.9998 | ~0.011 |
| feat_com | 0.9984 | 0.9975 | 0.9998 | ~0.011 |
| feat_width | 0.9988 | 0.9980 | 0.9999 | ~0.012 |
| feat_moments | 0.9991 | 0.9985 | 0.9999 | ~0.011 |

All feature sets achieve near-perfect displacement decoding within an image. Feature sets are nearly indistinguishable at ceiling; scalar rates perform as well as spatial moments.

Per-stimulus breakdown (feat_scalar R²_mean):

| Stimulus | R²_x | R²_y | R²_mean |
|----------|------|------|---------|
| BeachWave.jpg | 0.9990 | 0.9996 | 0.9993 |
| Colony_Bonnie.JPG | 0.9943 | 0.9992 | 0.9967 |
| Colony_PJ.JPG | 0.9997 | 0.9999 | 0.9998 |
| Hawaii_trees.JPG | 0.9941 | 0.9984 | 0.9963 |
| LakeOntario_Rocks1.JPG | 0.9909 | 0.9990 | 0.9950 |
| SD_Orangatang.JPG | 0.9976 | 0.9988 | 0.9982 |

### 3.2 Phase 2: Cross-image generalization (leave-one-image-out)

Train on 5 images, test on held-out 6th. R²_mean per held-out stimulus:

| Held-out stimulus | feat_scalar | feat_com | feat_width | feat_moments |
|-------------------|-------------|----------|------------|--------------|
| BeachWave.jpg | −1.097 | −1.506 | −1.291 | −1.402 |
| Colony_Bonnie.JPG | −0.918 | −1.914 | −0.944 | −0.617 |
| Colony_PJ.JPG | −2.482 | −4.780 | −2.014 | −1.843 |
| Hawaii_trees.JPG | −1.956 | −3.170 | −2.015 | −1.298 |
| LakeOntario_Rocks1.JPG | −0.272 | −1.938 | −1.002 | −1.134 |
| SD_Orangatang.JPG | −0.979 | −1.975 | +0.114 | −1.099 |
| **Mean** | **−1.284** | **−2.547** | **−1.192** | **−1.232** |

Cross-image R² is **uniformly negative** — worse than predicting the mean displacement. The decoder trained on 5 images actively anti-generalizes to the 6th. Spatial moments (feat_com) generalize *worse* than scalar rates, suggesting CoM features are more entangled with image-specific content.

### 3.3 Displacement magnitude sweep

R²_mean as a function of maximum displacement magnitude tested (within-image, single stimulus):

| Max displacement | feat_scalar | feat_com | feat_moments |
|------------------|-------------|----------|--------------|
| 0.01° | 0.9986 | 0.9995 | 0.9993 |
| 0.02° | 0.9735 | 0.9714 | 0.9849 |
| 0.03° | 0.9902 | 0.9892 | 0.9946 |
| 0.04° | 0.9947 | 0.9945 | 0.9976 |
| 0.05° | 0.9967 | 0.9966 | 0.9984 |

Decoding is reliable across the full FEM range (±0.05°). The slight dip at 0.02° likely reflects the transition from near-zero to non-trivial displacements rather than a real non-monotonicity.

**Interpretation:** The population has a high-fidelity displacement code within each image — every small retinal shift (down to 0.01°) produces a discriminable response change. However, this code is completely image-specific: the direction in population space that encodes displacement depends on which neurons are activated and how their RFs relate to local image structure. A downstream decoder cannot read out displacement without knowing the image, or without learning a nonlinear readout. Scalar rates encode displacement as well as spatial moments at this scale.

---

## Analysis 4: CoM Dynamics

*In progress — Phase 1 (moment trajectory computation across stimuli) running.*

---

## Cross-analysis Summary

| Finding | Null/Positive | Implication |
|---------|---------------|-------------|
| Temporal residual features (Model C) do not improve over rate-only (A) | Null | Orientation decoding is dominated by the mean-rate code |
| FEM effect is regime-dependent (near −0.20 hurts; −0.35…−0.50 helps at long windows; ≤ −0.55 is chance) | Mixed | FEM can help when multi-position sampling adds usable orientation evidence, but becomes irrelevant once the stimulus is below the discretization limit |
| Velocity not decodable from latent state (B ≈ 0) | Null | Scalar latent dynamics encode content, not transformation |
| A matrix explains 92% of latent variance; spectral radius ~0.96 | Descriptive | Slow attractor dynamics dominate; eye velocity is a negligible perturbation |
| Within-image displacement decoding: R² ≈ 0.997–0.999 | Positive | Population encodes fine-grained displacement; code is present |
| Cross-image displacement decoding: R² ≈ −1.3 (negative) | Null | Displacement code is image-specific; not a universal transformation signal |
| Scalar rates ≈ spatial moments for displacement (within-image) | Descriptive | CoM/width features don't add beyond scalar rates at this scale |

**Emerging picture:** The digital twin represents image *content* robustly (mean-rate features achieve high orientation accuracy on the original responses). Adding temporal residual features does not improve orientation decoding, and real FEM traces reduce acuity relative to stabilized. Velocity information is not present in the low-D latent state (B ≈ 0), and displacement decoding is strong within-image but fails to generalize across images (cross-image R² < 0). Transformation information, if present, is not in a format accessible to a simple linear decoder operating on population responses without image-specific calibration.

**Phase 3 (pending):** FEM vs static displacement decoding (Ahissar test) — does temporal integration of FEM trajectories improve displacement encoding beyond static snapshots at the same positions?
