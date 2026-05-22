# Implementation Plan: Temporal Decoding Analysis

## Granular Subgoals for Claude Code Agents in VisionCore

---

## Overview

This document translates the consolidated analysis plan (`analysis_plan_consolidated_v2.md`) into concrete coding tasks for the VisionCore repository. Each task is scoped for a single agent session, with explicit inputs, outputs, existing code to reference, and acceptance criteria.

**Repository structure:**
```
VisionCore/
├── scripts/           # Analysis scripts (our new code goes here)
│   ├── utils.py       # get_model_and_dataset_configs()
│   ├── spatial_info.py  # make_counterfactual_stim(), compute_rate_map_batched(), etc.
│   ├── mcfarland_sim.py -> DataYatesV1  # get_fixrsvp_stack(), eye_deg_to_norm(), shift_movie_with_eye()
│   ├── check_fixrsvp_model_fisherinfo.py  # DifferentiableStimulus, DifferentiableRetina, forward-AD
│   └── fixrsvp_digitaltwin_spatialinfo_declan.py  # eye trace extraction pattern
├── eval/              # Model evaluation utilities
├── models/            # Model architecture code
├── training/          # Training infrastructure
└── experiments/       # Experiment configs and shell scripts
```

**Key pre-existing assets:**
- Trained model: `resnet_none_convgru` loaded via `get_model_and_dataset_configs()`
- Population readout: `get_spatial_readout(model, outputs)` pools ~130 neurons with CC_norm > 0.5
- Eye trace library: extracted in `fixrsvp_digitaltwin_spatialinfo_declan.py` — hundreds of fixational traces, max 540 frames
- E optotype generator: `DifferentiableStimulus` class in `check_fixrsvp_model_fisherinfo.py`
- Counterfactual stimulus pipeline: `make_counterfactual_stim()` in `spatial_info.py`
- Precomputed unit quality: `mcfarland_outputs_mono.pkl`

**Constants:**
- Frame rate: 120 Hz (8.33 ms per frame)
- PPD (pixels per degree): 37.50476617
- n_lags (model temporal history): 32 frames
- Output spatial size: (151, 151) or (101, 101)

---

# PHASE 1: CORE INFRASTRUCTURE

## Task 1.1: E Optotype Stimulus Generator Module

**Goal:** Create a reusable module that generates Tumbling E stimuli at arbitrary orientations and LogMAR values, compatible with the existing counterfactual stimulus pipeline.

**Reference code:** `DifferentiableStimulus` class in `scripts/check_fixrsvp_model_fisherinfo.py` (line ~66017 in bundle). This class already renders E optotypes with continuous (x, y, orientation, LogMAR) control. However, it renders into a high-res world canvas for the differentiable retina pipeline. We need a version that renders into the same format as `get_fixrsvp_stack()` — a numpy array of shape (N_frames, H, W) that can be passed to `make_counterfactual_stim()`.

**Inputs:**
- Orientation: one of {0, 90, 180, 270} degrees
- LogMAR: float, range [−0.3, 1.0]
- Background: mean gray (127)
- Canvas size: 600×600 (matching `get_fixrsvp_stack` output)
- PPD: 37.50476617

**Outputs:**
- `e_optotype_stack(orientation, logmar, n_frames=540, ppd=37.50) -> np.ndarray (n_frames, H, W)`
- Returns a static image repeated n_frames times (the "stimulus stack" that `make_counterfactual_stim` expects)
- Pixel values in [0, 255] uint8 range, matching existing pipeline conventions

**File to create:** `scripts/temporal_decoding/stimulus.py`

**Acceptance criteria:**
- At LogMAR 0.0, the E gap subtends ~1 arcmin (~0.625 pixels at 37.5 ppd)
- At LogMAR 1.0, the E is clearly visible (~6.25 pixels gap)
- The 4 orientations produce visually correct rotations
- Output shape and dtype match `get_fixrsvp_stack()` output
- Include a simple visualization function that shows the E at several LogMAR values

---

## Task 1.2: Eye Trace Library Extraction and Caching

**Goal:** Extract all fixational eye traces from the dataset into a single cached file, with metadata (session, trial, duration, RMS, path length).

**Reference code:** The eye trace extraction loop in `fixrsvp_digitaltwin_spatialinfo_declan.py` (around line 57000 in bundle) already does this but is embedded in a script. Factor it out.

**Inputs:**
- Model loaded via `get_model_and_dataset_configs()`
- Precomputed outputs from `mcfarland_outputs_mono.pkl` (for session list)

**Outputs:**
- `scripts/temporal_decoding/data/eye_traces.npz` containing:
  - `traces`: (N_trials, max_T, 2) float32, NaN-padded
  - `durations`: (N_trials,) int — valid duration before NaN
  - `sessions`: (N_trials,) string — source session
  - `rms`: (N_trials,) float — RMS displacement from mean
  - `path_length`: (N_trials,) float — total path length in degrees
  - `velocity_rms`: (N_trials,) float — RMS velocity in deg/s

**File to create:** `scripts/temporal_decoding/extract_eye_traces.py`

**Acceptance criteria:**
- Extracts from all sessions listed in `mcfarland_outputs_mono.pkl`
- Filters to fixation periods only (eccentricity < 1 deg)
- Filters to traces > 60 frames duration
- Computes and stores kinematic statistics (RMS, path length, velocity)
- Saves to npz for fast reloading
- Print summary: N traces, duration distribution, RMS distribution

---

## Task 1.3: Rate Matrix Computation Pipeline

**Goal:** Given a stimulus stack and set of eye traces, compute population rate matrices through the digital twin. This is the core data generation step that all subsequent analyses depend on.

**Reference code:** `compute_rate_map_batched()` in `spatial_info.py`, `make_counterfactual_stim()` in `spatial_info.py`

**Inputs:**
- Stimulus stack: (N_frames, H, W) from Task 1.1 or `get_fixrsvp_stack()`
- Eye traces: (M, T, 2) from Task 1.2
- Model + readout (loaded via utils)
- FEM condition: one of {real, stabilized, scaled_0.5, scaled_2.0}

**Outputs:**
- Function `compute_population_rates(model, readout, stim_stack, eye_traces, condition, n_lags=32, out_size=(101,101), batch_size=32) -> dict`
- Returns dict with:
  - `rates`: (M, T_valid, N_neurons) float32 — population rate for each trial
  - `condition`: string
  - `stim_params`: dict (orientation, logmar, etc.)

**File to create:** `scripts/temporal_decoding/rate_computation.py`

**Key implementation details:**
- For "stabilized": set eye position to trial mean for all time points (eye_scale=0)
- For "scaled_X": scale eye trace around mean by factor X (use `rescale_fixations_only` pattern from `fixrsvp_digitaltwin_spatialinfo_declan.py`)
- Normalize stimulus: `(stim - 127.0) / 255.0` before passing to model (matches existing convention)
- The readout produces spatial rate maps (N, H_out, W_out); we need to **collapse spatial dims** to get (N,) per time bin. Use `model.model.activation(readout(core_output[:,:,-1]))` and then spatial mean or spatial max — check what the existing pipeline does.
- Handle variable-length traces by processing each trace individually or padding
- Move data to GPU in batches, collect on CPU to manage memory

**Acceptance criteria:**
- Produces rate matrices consistent with existing `compute_rate_map_batched` results
- Handles the 4 FEM conditions correctly
- Memory-efficient (processes in batches, moves to CPU after each)
- Includes a smoke test: run on 5 traces, 2 conditions, verify shapes and that stabilized < real FEM in variance

**IMPORTANT NOTE on spatial rate maps:** The existing `compute_rate_map` returns a spatial rate map (N_neurons, H_out, W_out) for each time step. For the decoding analysis, we need a single rate value per neuron per time step, not a spatial map. There are two options:
1. Take the spatial max of each neuron's rate map (the "maximum response" interpretation)
2. Take a weighted spatial average using the readout's Gaussian mask (the "expected response" interpretation)
Consult the existing analysis scripts to determine which convention is used. The `spatial_ssi_population` function treats the full spatial map, but for decoding we need to collapse. **Ask the user which approach to use if unclear.**

---

## Task 1.4: Matched-Budget Null Trace Generation

**Goal:** For each real eye trace, generate phase-randomized null traces that preserve velocity PSD but destroy specific trajectory structure.

**Inputs:**
- Eye trace library from Task 1.2

**Outputs:**
- Function `generate_phase_randomized_traces(traces, n_nulls=10) -> np.ndarray`
  - Input: (M, T, 2) real traces
  - Output: (M, n_nulls, T, 2) null traces
- Each null preserves: amplitude spectrum (= velocity PSD), RMS, approximate path length
- Each null destroys: specific trajectory, phase relationships between x and y

**File to create:** `scripts/temporal_decoding/null_traces.py`

**Implementation:**
```python
def phase_randomize_trace(trace, rng=None):
    """Phase-randomize a single (T, 2) eye trace."""
    rng = rng or np.random.default_rng()
    T = trace.shape[0]
    result = np.zeros_like(trace)
    for dim in range(2):
        ft = np.fft.rfft(trace[:, dim])
        phases = rng.uniform(0, 2*np.pi, size=ft.shape)
        phases[0] = 0  # preserve DC (mean position)
        ft_randomized = np.abs(ft) * np.exp(1j * phases)
        result[:, dim] = np.fft.irfft(ft_randomized, n=T)
    return result
```

**Acceptance criteria:**
- Null traces have same amplitude spectrum as originals (verify with PSD comparison)
- Null traces have different specific trajectories (verify visually)
- RMS displacement approximately preserved (within 10%)
- Generates reproducibly with fixed seed

---

# PHASE 2: PRIMARY DECODING PIPELINE

## Task 2.1: Decoding Infrastructure — Models A, B, C

**Goal:** Implement the core decoding pipeline with grouped cross-validation and the first three ablation models.

**Inputs:**
- Rate matrices from Task 1.3: dict mapping (stimulus_id, condition) -> (M, T, N) rate tensors
- Stimulus labels (orientation: 0, 90, 180, 270)

**Outputs:**
- Function `run_decoding_ladder(rates_by_stim, trace_ids, models=['A','B','C'], n_splits=5) -> dict`
- Returns dict with accuracy, MI, and std for each model

**File to create:** `scripts/temporal_decoding/decoding.py`

**Model A — Rate only:**
```python
X_A = rates.mean(axis=1)  # (M, N) — time-averaged rate per trial
```

**Model B — Temporal mean trajectory subspace:**
```python
# 1. Compute class-mean trajectories from TRAINING data only
class_means = {}  # stim_id -> (T, N) mean trajectory
for stim_id in train_stim_ids:
    class_means[stim_id] = rates_train[stim_id].mean(axis=0)  # avg over traces

# 2. Stack class means, PCA to get U_B
all_means = np.stack([cm.flatten() for cm in class_means.values()])  # (K, T*N)
from sklearn.decomposition import PCA
pca_B = PCA(n_components=min(K-1, 20))  # at most K-1 components for K classes
pca_B.fit(all_means)
U_B = pca_B  # fitted on class means only

# 3. Project SINGLE-TRIAL trajectories onto U_B
X_B = U_B.transform(trial_trajectory.flatten())  # (d_B,) per trial
```

**CRITICAL:** Verify that trace distributions are identical across stimulus classes. If not, resample to equalize before computing class means.

**Model C — Full single-trial temporal trajectory:**
```python
# Project onto broader PCA subspace (fitted on ALL training trajectories, not just means)
pca_C = PCA(n_components=50)  # captures mean + trace-specific variation
pca_C.fit(all_training_trajectories_flattened)
X_C = pca_C.transform(trial_trajectory.flatten())
```

**Cross-validation:**
```python
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def decode_with_cv(X, y, groups, n_splits=5, C=1.0):
    gkf = GroupKFold(n_splits=n_splits)
    accuracies = []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = LogisticRegression(penalty='l2', C=C, max_iter=2000, solver='lbfgs')
        clf.fit(X_tr, y[train_idx])
        accuracies.append(clf.score(X_te, y[test_idx]))
    return np.mean(accuracies), np.std(accuracies)
```

**Groups = eye trace IDs** (not time bins, not trials within a trace). This ensures the decoder generalizes across eye movements.

**Acceptance criteria:**
- A < C for real FEM condition at threshold LogMAR (temporal code helps)
- A ≈ C for stabilized condition (no temporal structure to exploit)
- B sits between A and C (captures mean dynamics but not trace-specific)
- Cross-validation splits by trace ID, verified by printing trace IDs in train/test
- PCA for U_B fitted on training data only within each fold

---

## Task 2.2: Model D — Residual Covariance Features

**Goal:** Implement the residual covariance feature extraction and the D negative control.

**Inputs:**
- Rate matrices and U_B projection from Task 2.1

**Outputs:**
- Function `compute_model_D_features(trajectory, U_B, n_lags=5) -> np.ndarray`
- Returns covariance features of the residual after projecting out the B subspace

**File to create:** add to `scripts/temporal_decoding/decoding.py`

**Primary D: low-rank lagged cross-covariance:**
```python
def compute_residual_covariance_features(trajectory, U_B, n_lags=5):
    """
    trajectory: (T, N) single-trial rate trajectory
    U_B: fitted PCA object from Model B
    n_lags: number of temporal lags for cross-covariance
    """
    # 1. Project out B subspace
    traj_flat = trajectory.flatten()
    b_proj = U_B.inverse_transform(U_B.transform(traj_flat.reshape(1,-1)))
    residual = (traj_flat - b_proj.flatten()).reshape(trajectory.shape)  # (T, N)
    
    # 2. Compute lagged cross-covariance of residual
    T, N = residual.shape
    cov_features = []
    for lag in range(n_lags + 1):
        if lag == 0:
            C = residual.T @ residual / T  # (N, N)
        else:
            C = residual[lag:].T @ residual[:-lag] / (T - lag)  # (N, N)
        # Take upper triangle (including diagonal)
        cov_features.append(C[np.triu_indices(N)])
    
    features = np.concatenate(cov_features)
    
    # 3. Reduce dimensionality (PCA on covariance features, fitted on training)
    return features
```

**Sensitivity D2: within-neuron autocovariance only:**
```python
def compute_autocov_features(trajectory, U_B, n_lags=10):
    residual = ...  # same as above
    T, N = residual.shape
    features = []
    for lag in range(1, n_lags + 1):
        autocov = np.mean(residual[lag:] * residual[:-lag], axis=0)  # (N,)
        features.append(autocov)
    return np.concatenate(features)
```

**Negative control (critical):**
```python
def compute_shuffled_D_features(trajectory, U_B, n_lags=5, rng=None):
    """Shuffle time independently per neuron, then compute D features."""
    rng = rng or np.random.default_rng()
    T, N = trajectory.shape
    shuffled = trajectory.copy()
    for n in range(N):
        rng.shuffle(shuffled[:, n])
    return compute_residual_covariance_features(shuffled, U_B, n_lags)
```

**Acceptance criteria:**
- D features have reasonable dimensionality after PCA reduction (verify variance explained)
- D negative control (shuffled) produces near-chance gain (if real D gain is significant, it must exceed shuffled D gain)
- D feature computation is fast enough for cross-validation (~seconds per trial)
- If D > B gain exists, verify it is stable across at least 3 LogMAR values

---

## Task 2.3: Integration Time Sweep

**Goal:** Train causal sliding-window decoders at multiple integration times and plot the accuracy-vs-window curve.

**Inputs:**
- Rate matrices from Task 1.3

**Outputs:**
- Function `integration_time_curve(rates_by_stim, trace_ids, windows, conditions) -> dict`
- Returns accuracy(W) for each condition

**Update (April 2026):** There are now two integration-time decoding *methods*:
- `flat_pca` (original): flatten the last W frames ($W \times N$), then PCA, then decode.
- `time_mean` (accumulation-aligned control): average the last W frames (N-dim), then decode.

In the hyperacuity regime, `flat_pca` can artificially obliterate real signal. Treat `time_mean` as the primary sanity check when the curve looks flat/near-chance.

**File to create:** `scripts/temporal_decoding/integration_time.py`

**Implementation:**
```python
windows = [1, 3, 6, 12, 24, 36, 48, 60]  # frames at 120 Hz

def decode_causal_window(rates_by_stim, trace_ids, W, **kwargs):
    """Decode using only the last W frames of each trial."""
    X_dict = {}
    for stim_id, rates in rates_by_stim.items():
        # rates: (M, T, N)
        X_dict[stim_id] = rates[:, -W:, :].reshape(rates.shape[0], -1)  # (M, W*N)
    # ... standard decoding with CV

def decode_causal_window_time_mean(rates_by_stim, trace_ids, W, **kwargs):
    """Decode from the mean rate over the last W frames (accumulation-aligned)."""
    X_dict = {}
    for stim_id, rates in rates_by_stim.items():
        X_dict[stim_id] = rates[:, -W:, :].mean(axis=1)  # (M, N)
    # ... standard decoding with CV
```

**Key: run for both real FEM and stabilized conditions on the same plot.**

**Acceptance criteria:**
- Curves are monotonically non-decreasing with W (more data never hurts with proper regularization)
- Stabilized condition shows flatter curve (less temporal structure to exploit)
- Real FEM curve rises more steeply
- Plot includes error bars (std across CV folds)
- Include both crossover and no-crossover annotations

---

## Task 2.4: Neurometric Curves

**Goal:** Sweep LogMAR and plot decoding accuracy vs. stimulus size for each condition and model.

**Inputs:**
- Tasks 1.1–1.3 (stimulus generation, eye traces, rate computation)
- Task 2.1 (decoding pipeline)

**Outputs:**
- `scripts/temporal_decoding/neurometric.py` — orchestration script
- Figure: accuracy vs. LogMAR for Models A and C, real FEM and stabilized
- Summary: ΔLogMAR threshold shift

**Implementation:**
```python
logmar_values = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.1, -0.2, -0.3]
orientations = [0, 90, 180, 270]
conditions = ['real', 'stabilized', 'matched_null']

for logmar in logmar_values:
    for ori in orientations:
        stim_stack = e_optotype_stack(ori, logmar)
        for condition in conditions:
            rates = compute_population_rates(model, readout, stim_stack, 
                                              eye_traces, condition)
            # store rates keyed by (ori, logmar, condition)
    
    # Decode orientation for this LogMAR
    accuracy_A = decode_model_A(rates_this_logmar, trace_ids)
    accuracy_C = decode_model_C(rates_this_logmar, trace_ids)
```

**Acceptance criteria:**
- Neurometric curves are sigmoid-shaped (high accuracy at large LogMAR, chance at small)
- Real FEM Model C curve is shifted left (better threshold) relative to stabilized Model A
- ΔLogMAR is a positive number (FEMs help)
- Threshold defined as LogMAR where accuracy crosses 62.5% (midpoint of chance-to-ceiling)

**IMPORTANT: This is the most computationally expensive task.** 9 LogMAR × 4 orientations × 3 conditions × ~200 traces = ~21,600 rate computations. At ~5 sec each = ~30 hours. Plan to:
- Start with a reduced set: 5 LogMAR values, 50 traces, 2 conditions
- Cache intermediate results aggressively
- Parallelize across GPU if possible

---

## Task 2.5: Sequential Entropy Reduction (Ideal Observer)

**Goal:** Implement a Bayesian observer that updates a posterior over stimulus classes as the temporal response unfolds.

**Inputs:**
- Rate matrices from Task 1.3 (for a fixed LogMAR at threshold)

**Outputs:**
- Function `sequential_entropy(rates_by_stim, trace_ids) -> dict`
- Returns H(S|r_{1:t}) vs. t for each condition

**File to create:** `scripts/temporal_decoding/entropy.py`

**Implementation:**
```python
def sequential_posterior(rates_by_stim, test_trace_idx):
    """
    Compute posterior p(S | r_{1:t}) using Gaussian class-conditional likelihoods.
    
    For each class k, estimate mean trajectory mu_k (T,N) and 
    shared covariance Sigma (N,N) from training traces.
    
    At each time t, update:
        log p(S=k | r_{1:t}) += log N(r(t) | mu_k(t), Sigma)
    """
    K = len(rates_by_stim)
    stim_ids = list(rates_by_stim.keys())
    
    # Fit class-conditional Gaussians from training data
    class_means = {}  # stim_id -> (T, N)
    for k in stim_ids:
        train_rates = rates_by_stim[k][train_mask]
        class_means[k] = train_rates.mean(axis=0)
    
    # Shared covariance (pooled across classes, at each time)
    # Use diagonal or low-rank for tractability
    
    # Sequential update for test trial
    test_rates = rates_by_stim[test_stim][test_trace_idx]  # (T, N)
    log_posterior = np.zeros(K)  # uniform prior
    
    entropy_over_time = []
    for t in range(T):
        for i, k in enumerate(stim_ids):
            # Gaussian log-likelihood at time t
            diff = test_rates[t] - class_means[k][t]
            log_posterior[i] += -0.5 * diff @ Sigma_inv @ diff
        
        # Normalize
        posterior = softmax(log_posterior)
        entropy_over_time.append(-np.sum(posterior * np.log2(posterior + 1e-12)))
    
    return np.array(entropy_over_time)
```

**Acceptance criteria:**
- Entropy starts at log2(4) = 2 bits (uniform over 4 orientations)
- Entropy decreases over time (evidence accumulates)
- Real FEM condition shows faster decrease than stabilized
- Averaged over held-out test traces with grouped CV

---

# PHASE 3: MECHANISTIC ANALYSES

## Task 3.1: Signal and Noise Covariance Computation

**Goal:** Compute C_signal and C_FEM from the rate matrices.

**Inputs:**
- Rate matrices from Task 1.3, for all stimulus conditions and FEM conditions

**Outputs:**
- Function `compute_covariances(rates_by_stim, mode='instantaneous') -> dict`
- Returns C_signal, C_FEM, eigenvalues, eigenvectors

**File to create:** `scripts/temporal_decoding/geometry.py`

**Implementation:** Follow Section 4.3 of the analysis plan. Use the corrected null: compute C_signal for both temporal and instantaneous representations under both real FEM and stabilized.

**Acceptance criteria:**
- C_signal is PSD (all eigenvalues ≥ 0)
- C_FEM is PSD
- Under stabilization, C_FEM ≈ 0 (verify: eigenvalues near machine epsilon)
- Task-relevant subspace SNR computed and reported
- Eigenvalue spectra plotted for visual inspection

---

## Task 3.2: Alignment Analysis and Representational Intervention

**Goal:** Compute alignment fraction α and perform the representational intervention (project out FEM-aligned subspace, rerun decoding).

**Inputs:**
- Covariances from Task 3.1
- Decoding pipeline from Task 2.1

**Outputs:**
- α for each condition
- Decoding accuracy before and after removing FEM-aligned subspace
- Figure: bar chart of accuracy with and without intervention

**File to create:** add to `scripts/temporal_decoding/geometry.py`

**Implementation:**
```python
def alignment_fraction(C_signal, C_FEM, d):
    eigvals, eigvecs = np.linalg.eigh(C_signal)
    U_s = eigvecs[:, -d:]  # top d signal eigenvectors
    C_FEM_proj = U_s.T @ C_FEM @ U_s
    alpha = np.trace(C_FEM_proj) / np.trace(C_FEM)
    return alpha, d / C_signal.shape[0]  # alpha, chance level

def representational_intervention(rates_by_stim, C_signal, C_FEM, d_remove=5):
    """Remove top FEM-aligned signal subspace and rerun decoding."""
    # 1. Find FEM-aligned signal components
    eigvals_s, eigvecs_s = np.linalg.eigh(C_signal)
    U_s = eigvecs_s[:, -d_remove:]  # top signal subspace
    
    eigvals_n, eigvecs_n = np.linalg.eigh(C_FEM)
    U_n = eigvecs_n[:, -d_remove:]  # top noise subspace
    
    # Find most aligned components via SVD of overlap
    _, sigmas, Vt = np.linalg.svd(U_s.T @ U_n)
    aligned_directions = U_s @ Vt.T[:, :d_remove]  # most aligned directions
    
    # 2. Project out from each trial
    P_remove = aligned_directions @ aligned_directions.T
    rates_cleaned = {}
    for stim_id, rates in rates_by_stim.items():
        M, T, N = rates.shape
        rates_flat = rates.reshape(M, -1)
        # Project out per-time-bin
        rates_clean = rates.copy()
        for t in range(T):
            rates_clean[:, t, :] -= (rates[:, t, :] @ P_remove.T)
        rates_cleaned[stim_id] = rates_clean
    
    # 3. Rerun decoding on cleaned rates
    return decode_model_C(rates_cleaned, ...)
```

**Acceptance criteria:**
- α > chance for real FEM condition (FEM noise has *some* overlap with signal)
- The intervention produces a clear result: accuracy drops significantly (signal-bearing) OR improves (nuisance)
- Report α and intervention result together

---

## Task 3.3: Trace-Budget Stratification

**Goal:** Bin real traces by oculomotor budget and test whether temporal coding gain scales with movement.

**Inputs:**
- Eye trace library with kinematics (Task 1.2)
- Decoding pipeline (Task 2.1)

**Outputs:**
- Plot: ΔAccuracy (Model C − Model A) vs. budget tercile

**File to create:** `scripts/temporal_decoding/budget_analysis.py`

**Implementation:**
- Bin traces into low/medium/high RMS terciles
- Run Model A and C decoding within each bin
- Plot gain vs. bin

**Acceptance criteria:**
- Clear trend (monotonic increase, or saturation, or peak)
- Error bars from cross-validation within each bin
- Sample size per bin reported

---

# PHASE 4: SECONDARY AND EXPLORATORY

## Task 4.1: Fisher Information Upgrade

**Goal:** Upgrade existing Fisher computation from J_indep to J_pop.

**Reference code:** `check_fixrsvp_model_fisherinfo.py` — the forward-AD infrastructure is already built.

**Key change:** Instead of:
```python
fisher_per_element = (d_rates_d_theta ** 2) / (rates_primal + epsilon)
chunk_fisher = fisher_per_element.sum()
```

Collect the full f' vector and compute:
```python
f_prime = d_rates_d_theta.reshape(-1)  # (N,)
# Σ computed from across-trace covariance + diag(rates)
L = torch.linalg.cholesky(Sigma)
v = torch.linalg.solve_triangular(L, f_prime, upper=False)
J_pop = v @ v
J_indep = (f_prime**2 / (rates_primal.reshape(-1) + epsilon)).sum()
eta = J_pop / J_indep
```

**File to modify:** `scripts/check_fixrsvp_model_fisherinfo.py` (or create new `scripts/temporal_decoding/fisher.py`)

**Acceptance criteria:**
- J_pop ≤ J_indep when noise correlations are present (sanity check)
- η reported under at least 2 noise models
- Fisher matrix (4×4) computed for x, y, orientation, LogMAR

---

## Task 4.2: Spatiotemporal Resonance (Conditional)

**Goal:** Characterize each neuron's SF/TF tuning and test whether FEM velocities shift stimulus SFs into the temporal passband.

**Reference code:** `eval/gratings_analysis.py` already computes grating responses. Use this to estimate SF_pref and TF_pref.

**File to create:** `scripts/temporal_decoding/resonance.py`

**Acceptance criteria:**
- SF_pref and TF_pref estimated for each neuron
- Resonance score correlates with decoder weight (from Model C)
- If correlation is weak (r < 0.2), report as inconclusive and do not include in main figures

---

# ORCHESTRATION

## Task 5.1: Main Pipeline Script

**Goal:** Create a single orchestration script that runs the full analysis pipeline end-to-end.

**File to create:** `scripts/temporal_decoding/run_analysis.py`

**Structure:**
```python
"""
Full temporal decoding analysis pipeline.
Usage: python run_analysis.py [--phase {1,2,3,4,all}] [--n_traces 200] [--logmar_subset]

Cached decode-only reruns (no model/sim imports):
    python run_analysis.py --phase 2 --decode_only --integration_method time_mean --threshold_logmar -0.20
"""

# Phase 1: Generate data
# 1.1 Load model, readout, eye traces
# 1.2 Generate E stimuli at LogMAR grid
# 1.3 Compute rate matrices for all conditions
# 1.4 Generate matched-budget null traces

# Phase 2: Primary decoding
# 2.1 Run ablation ladder (A/B/C/D) at threshold LogMAR
# 2.2 Run integration time sweep
# 2.3 Run neurometric curves (LogMAR sweep)
# 2.4 Run sequential entropy reduction

# Phase 3: Mechanistic
# 3.1 Compute covariances and eigenspectra
# 3.2 Run alignment analysis and intervention
# 3.3 Run budget stratification

# Phase 4: Figures
# Generate all figures from cached results
```

**Acceptance criteria:**
- Runs end-to-end with `--phase 1` through `--phase 4`
- Results cached at each phase (no recomputation)
- Figures saved to `figures/temporal_decoding/`
- Total runtime < 48 hours on a single GPU for full pipeline with 200 traces

**Notes (April 2026):**
- `run_analysis.py` now supports `--decode_only` to rerun Phase 2 from cached `.npz` rate files (useful if external data packages are unavailable or importing `rate_computation` fails).
- Integration-time results are saved with method-specific filenames, e.g. `integration_time_time_mean.pkl` and `fig_integration_time_time_mean.png`.

---

# FILE STRUCTURE

```
scripts/temporal_decoding/
├── __init__.py
├── stimulus.py           # Task 1.1: E optotype generation
├── extract_eye_traces.py # Task 1.2: Eye trace extraction and caching
├── rate_computation.py   # Task 1.3: Population rate matrix computation
├── null_traces.py        # Task 1.4: Phase-randomized null traces
├── decoding.py           # Tasks 2.1-2.2: Models A/B/C/D + CV
├── integration_time.py   # Task 2.3: Integration time sweep
├── neurometric.py        # Task 2.4: LogMAR sweep orchestration
├── entropy.py            # Task 2.5: Sequential ideal observer
├── geometry.py           # Tasks 3.1-3.2: Covariances, alignment, intervention
├── budget_analysis.py    # Task 3.3: Trace-budget stratification
├── fisher.py             # Task 4.1: J_pop upgrade
├── resonance.py          # Task 4.2: Spatiotemporal resonance (conditional)
├── run_analysis.py       # Task 5.1: Orchestration
├── plotting.py           # Shared plotting utilities
├── data/                 # Cached intermediate results
│   ├── eye_traces.npz
│   ├── rates/            # Cached rate matrices per condition
│   └── results/          # Cached decoding results
└── figures/              # Output figures
```

---

# DEPENDENCY ORDER

```
Task 1.1 (stimulus) ─────────┐
Task 1.2 (eye traces) ───────┤
                              ├──► Task 1.3 (rates) ──► Task 2.1 (A/B/C) ──► Task 2.2 (D)
Task 1.4 (null traces) ──────┘         │                      │
                                       │                      ├──► Task 2.3 (integration time)
                                       │                      ├──► Task 2.4 (neurometric)
                                       │                      └──► Task 2.5 (entropy)
                                       │
                                       └──► Task 3.1 (covariances) ──► Task 3.2 (intervention)
                                                                   └──► Task 3.3 (budget)
                                       
Task 4.1 (Fisher) ── independent, uses existing infrastructure
Task 4.2 (resonance) ── independent, uses eval/gratings_analysis.py
```

Tasks 1.1, 1.2, 1.4 can run in parallel.
Tasks 2.1–2.5 require Task 1.3.
Tasks 3.x require both Task 1.3 and Task 2.1.
Tasks 4.x are independent.

---

# AGENT INSTRUCTIONS TEMPLATE

When assigning a task to a Claude Code agent, use this template:

```
## Context
You are working in the VisionCore repository (a PyTorch-based digital twin of marmoset V1).
The analysis goal is described in `scripts/temporal_decoding/README.md` (the analysis plan).

## Your Task
[Copy the specific task section from above]

## Key Files to Read First
- `scripts/utils.py` — how to load the model
- `scripts/spatial_info.py` — existing rate computation functions
- `scripts/temporal_decoding/__init__.py` — any previously completed tasks

## Constraints
- All new code goes in `scripts/temporal_decoding/`
- Follow existing code conventions (numpy/torch, matplotlib for plotting)
- Cache intermediate results to `scripts/temporal_decoding/data/`
- Include docstrings and type hints
- Include a `if __name__ == '__main__':` smoke test

## When Stuck
- The model is loaded via `get_model_and_dataset_configs()` from `scripts/utils.py`
- The readout is created via `get_spatial_readout(model, outputs)` from `scripts/spatial_info.py`
- Precomputed outputs are in `mcfarland_outputs_mono.pkl`
- The model expects normalized input: `(stim - 127) / 255`
- Eye positions are in degrees; the model works in normalized [-1,1] coords via `eye_deg_to_norm()`
```
