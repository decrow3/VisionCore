"""
Transformation Dynamics Analysis
=================================

**ARCHITECTURAL CAVEAT — READ BEFORE INTERPRETING RESULTS**

This analysis was designed to test whether FEM-induced changes in population
state (Δz) encode eye velocity. The motivation came from C ≈ A at −0.20 dB,
which turned out to be the wrong operating point (valid crossover is −0.35 dB,
driven by spatial sampling). The entire pivot to "what do dynamics encode if
not content?" rested on that result.

More fundamentally: the model uses independent 32-frame windows with a fresh
GRU state each trial. It does not maintain recurrent state across windows or
frames in any cross-trial sense. The GRU operates within a single window and
produces one output — it is not being driven by eye velocity as an ongoing
input across the trial.

Consequence for interpretation
-------------------------------
- Δz ≈ 0 (B ≈ 0 in linear dynamics fit): **architecturally expected**. Eye
  velocity does not perturb within-window GRU state because the GRU processes
  a single chunk, not a continuous stream. This is not evidence that V1 lacks
  transformation encoding — it is evidence that the model lacks the temporal
  machinery needed to express it.
- The spectral radius ~0.96 and "slow attractor" language describe within-window
  GRU dynamics, not a continuous dynamical system evolving over the trial.
- All Δz-based velocity decoding nulls (Analyses A1–A6) are trivially expected
  under this architecture and are uninformative about biology.
- Analysis B (dissociation matrix) inherits the same confound.

If you report these results, qualify them as: "tested in a model with independent
temporal windows; nulls likely reflect architectural constraints rather than the
absence of biological transformation coding."

------------------------------------------------------------------------

Tests whether FEM-induced population dynamics (Δz) encode retinal translation
(eye velocity), and whether this is specific to the FEM-aligned subspace (U_pca2).

Analysis A (Phase 1 — killer test):
  A1. Core readout: z vs Δz vs [z,Δz] for velocity decoding
  A2. Control: time-shuffle (temporal order is necessary)
  A3. Control: random projection baseline (subspace specificity)
  A4. Control: shuffled-PCA baseline (population structure is necessary)
  A5. Specificity: Δz vs surrogate velocity targets
  A6. Cross-stimulus generalization curve (sweep shared basis dim d)

Analysis B (Phase 2 — dissociation):
  B1. 2×2 matrix: {static mean-rate, Δz} × {content, transformation}

Analysis C (Phase 3 — descriptive):
  C1. Linear dynamics: z(t+1) = Az(t) + Bu(t) + c; compare to B-only baseline
  C2. Input-conditioned flow fields

Data flow:
  backimage_fixation_results.pkl  →  eye traces (n_trials, T, 2) per stimulus
  all_cov_results.pkl             →  U_pca2 (N, 2) per stimulus
  model + readout                 →  δr(t) = r_real(t) − r_null(t)
  cache dir                       →  declan/transformation_dynamics_cache/*.npz
"""


# %% ── Imports ────────────────────────────────────────────────────────────────

# Patch DataYatesV1.enable_autoreload if missing, to avoid ImportError in mcfarland_sim
import sys
import types
try:
    import DataYatesV1
    if not hasattr(DataYatesV1, "enable_autoreload"):
        DataYatesV1.enable_autoreload = lambda: None
except ImportError:
    pass

import os
import sys
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from spatial_info import make_counterfactual_stim, get_spatial_readout, compute_rate_map_batched
from utils import get_model_and_dataset_configs

# %% ── Config ─────────────────────────────────────────────────────────────────

CACHE_DIR    = os.path.join(ROOT, 'declan', 'transformation_dynamics_cache')
COV_PKL      = os.path.join(ROOT, 'declan', 'translation_covariance', 'all_cov_results.pkl')
FIXATION_PKL = os.path.join(ROOT, 'declan', 'backimage_fixation_results.pkl')
FIGURES_DIR  = os.path.join(ROOT, 'declan', 'transformation_dynamics_figures')

N_LAGS              = 32      # temporal context frames for model
OUT_SIZE            = (151, 151)
SCALE               = 1.0
N_SPLITS_CV         = 5       # grouped CV folds (by trace index)
RIDGE_ALPHAS        = np.logspace(-2, 4, 13)   # ridge regularisation sweep
N_RANDOM_PROJ       = 200     # random-projection control repeats
N_SHUFFLED_PCA      = 100     # shuffled-covariance control repeats
D_VALUES            = [2, 4, 6, 8, 10, 15, 20]  # dim sweep for cross-stim curve

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# %% ── Device / model loading ─────────────────────────────────────────────────

def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model_and_readout():
    """Load digital twin model and spatial readout."""
    model, _ = get_model_and_dataset_configs()
    device = get_device()
    model = model.to(device).eval()

    outputs_candidates = [
        os.path.join(ROOT, 'scripts', 'mcfarland_outputs_mono.pkl'),
        os.path.join(ROOT, 'scripts', 'mcfarland_outputs.pkl'),
    ]
    import dill
    outputs = None
    for p in outputs_candidates:
        if os.path.exists(p):
            with open(p, 'rb') as f:
                outputs = dill.load(f)
            break
    if outputs is None:
        raise RuntimeError(
            'Could not find scripts/mcfarland_outputs_mono.pkl. '
            'Generate or symlink this file before running the analysis.'
        )
    readout = get_spatial_readout(model, outputs).to(device).eval()
    return model, readout


# %% ── Eye trace helpers ──────────────────────────────────────────────────────

def split_eye_traces(eyepos_flat: np.ndarray, n_trials: int) -> np.ndarray:
    """
    Split flat (n_trials*T + remainder, 2) array into (n_trials, T, 2).

    Trials are stored concatenated with a small leftover tail; we use
    T = total // n_trials and discard the tail.
    """
    T = eyepos_flat.shape[0] // n_trials
    return eyepos_flat[:n_trials * T].reshape(n_trials, T, 2).astype(np.float32)


def compute_velocity(eyepos: np.ndarray) -> np.ndarray:
    """
    Instantaneous eye velocity: vel[t] = pos[t] − pos[t-1].

    Args:
        eyepos: (T, 2) eye position trace in degrees

    Returns:
        vel: (T-1, 2) velocity in degrees/frame
    """
    return np.diff(eyepos, axis=0).astype(np.float32)


def velocity_direction_bins(vel: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Map (T, 2) velocity to (T,) direction bin indices [0, n_bins)."""
    angles = np.arctan2(vel[:, 1], vel[:, 0])  # (T,)
    bins = (np.floor((angles + np.pi) / (2 * np.pi / n_bins)) % n_bins).astype(int)
    return bins


def phase_randomize_velocity(vel: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Surrogate velocity with same power spectrum, randomised phase (per axis)."""
    out = np.empty_like(vel)
    for ax in range(vel.shape[1]):
        fft = np.fft.rfft(vel[:, ax])
        phases = rng.uniform(0, 2 * np.pi, len(fft))
        fft_rand = np.abs(fft) * np.exp(1j * phases)
        out[:, ax] = np.fft.irfft(fft_rand, n=vel.shape[0])
    return out.astype(np.float32)


# %% ── Neural response computation ────────────────────────────────────────────

def run_model_trial(
    image_gray: np.ndarray,
    eyepos: np.ndarray,
    model: torch.nn.Module,
    readout: torch.nn.Module,
    n_lags: int = N_LAGS,
    out_size: tuple = OUT_SIZE,
    scale: float = SCALE,
) -> np.ndarray:
    """
    Compute population rate vector for one trial.

    Args:
        image_gray: (H, W) float32 background image in [0, 255]
        eyepos: (T, 2) eye position trace
        model, readout: digital twin

    Returns:
        rates: (T, N) float32 population response
    """
    T = eyepos.shape[0]
    full_stack = np.repeat(image_gray[np.newaxis], T + n_lags, axis=0)

    eyepos_t = torch.from_numpy(eyepos).float().to(next(model.parameters()).device)
    stim = make_counterfactual_stim(full_stack, eyepos_t, out_size=out_size,
                                    n_lags=n_lags, scale_factor=scale)

    with torch.no_grad():
        y = compute_rate_map_batched(model, readout, (stim - 127.0) / 255.0)

    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    y = np.asarray(y, dtype=np.float32)
    # Compress spatial dims: (T, N, H, W) → (T, N)
    if y.ndim == 4:
        y = y.mean(axis=(2, 3))
    return y


def compute_delta_r_for_stimulus(
    stim_key: str,
    image_gray: np.ndarray,
    eyepos_3d: np.ndarray,
    model: torch.nn.Module,
    readout: torch.nn.Module,
    n_lags: int = N_LAGS,
) -> dict:
    """
    Compute δr(t) = r_real(t) − r_null(t) for all trials of one stimulus.

    The null response uses a constant trace at the per-trial mean position.

    Args:
        eyepos_3d: (M, T, 2) eye traces for this stimulus

    Returns:
        dict with keys:
          'delta_r'  : (M, T, N) float32 — FEM-modulated residuals
          'r_real'   : (M, T, N) float32 — raw rates
          'eyepos'   : (M, T, 2) float32 — eye positions (trimmed to valid T)
    """
    M, T, _ = eyepos_3d.shape
    device = next(model.parameters()).device

    # Null: constant trace at (0, 0) — origin position
    null_eyepos = np.zeros((T, 2), dtype=np.float32)
    r_null = run_model_trial(image_gray, null_eyepos, model, readout, n_lags=n_lags)
    T_out = r_null.shape[0]  # may differ from T due to n_lags alignment
    N = r_null.shape[1]

    delta_r_all = np.empty((M, T_out, N), dtype=np.float32)
    r_real_all  = np.empty((M, T_out, N), dtype=np.float32)
    ep_out      = np.empty((M, T_out, 2), dtype=np.float32)

    for i in range(M):
        trace = eyepos_3d[i]  # (T, 2)
        r = run_model_trial(image_gray, trace, model, readout, n_lags=n_lags)
        t_use = min(T_out, r.shape[0])
        # Align eye position with model output (prepend n_lags copies of first sample)
        pos_padded = np.vstack([np.repeat(trace[:1], n_lags, axis=0), trace])[:T_out]

        delta_r_all[i] = r[:t_use] - r_null[:t_use]
        r_real_all[i]  = r[:t_use]
        ep_out[i]      = pos_padded

        if (i + 1) % 10 == 0 or i == M - 1:
            print(f'  [{stim_key}] trial {i+1}/{M}', end='\r')

    print()
    return {
        'delta_r': delta_r_all,
        'r_real':  r_real_all,
        'eyepos':  ep_out,
    }


# %% ── Cache management ───────────────────────────────────────────────────────

def cache_path(stim_key: str) -> str:
    safe = stim_key.replace('/', '_').replace('.', '_')
    return os.path.join(CACHE_DIR, f'{safe}_delta_r.npz')


def load_or_compute_delta_r(
    stim_key: str,
    image_gray: np.ndarray,
    eyepos_3d: np.ndarray,
    model: torch.nn.Module,
    readout: torch.nn.Module,
    force_recompute: bool = False,
) -> dict:
    """Load cached δr if available, otherwise compute and save."""
    path = cache_path(stim_key)
    if os.path.exists(path) and not force_recompute:
        print(f'Loading cached δr for {stim_key}')
        d = np.load(path)
        return {k: d[k] for k in d.files}

    print(f'Computing δr for {stim_key} ({eyepos_3d.shape[0]} trials)...')
    data = compute_delta_r_for_stimulus(stim_key, image_gray, eyepos_3d, model, readout)
    np.savez_compressed(path, **data)
    print(f'  Saved to {path}')
    return data


# %% ── Image loading helper ───────────────────────────────────────────────────

def load_image_gray(stim_key: str) -> np.ndarray:
    """Locate and load background image as (H, W) float32 in [0, 255]."""
    data_yates = '/home/declan/DataYatesV1/DataYatesV1/exp/SupportData/Backgrounds'
    search_dirs = [
        data_yates,
        os.path.join(ROOT, 'declan'),
        os.path.join(ROOT, 'data'),
    ]
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            if stim_key in files:
                path = os.path.join(root, stim_key)
                try:
                    from PIL import Image
                    with Image.open(path) as im:
                        arr = np.array(im, dtype=np.float32)
                except ImportError:
                    import imageio.v2 as iio
                    arr = iio.imread(path).astype(np.float32)
                if arr.ndim == 3:
                    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
                    arr = 0.2989 * r + 0.5870 * g + 0.1140 * b
                if arr.max() <= 1.0:
                    arr *= 255.0
                return np.clip(arr, 0, 255)
    raise FileNotFoundError(f'Could not find {stim_key} under {search_dirs}')


# %% ── z / Δz / velocity extraction ──────────────────────────────────────────

def compute_z_trajectory(delta_r: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Project δr(t) onto d-dimensional basis U.

    Args:
        delta_r: (M, T, N) residual responses
        U: (N, d) orthonormal basis

    Returns:
        z: (M, T, d) projections
    """
    return (delta_r @ U).astype(np.float32)   # (M, T, N) × (N, d) → (M, T, d)


def compute_dz(z: np.ndarray) -> np.ndarray:
    """
    State increment Δz[t] = z[t+1] − z[t].

    Args:
        z: (M, T, d)

    Returns:
        dz: (M, T-1, d)
    """
    return np.diff(z, axis=1).astype(np.float32)


def extract_features_and_targets(
    delta_r: np.ndarray,   # (M, T, N)
    eyepos:  np.ndarray,   # (M, T, 2)
    U_pca2:  np.ndarray,   # (N, 2)
) -> dict:
    """
    Build all feature arrays and velocity targets for one stimulus.

    Returns dict with:
      'z'        : (M, T,   2)  — static projection
      'dz'       : (M, T-1, 2)  — state increment
      'z_dz'     : (M, T-1, 4)  — concatenated [z[:-1], dz]
      'mean_rate': (M, N)        — time-averaged δr (content code)
      'vel'      : (M, T-1, 2)  — instantaneous eye velocity
      'vel_dir'  : (M, T-1)     — 8-bin direction labels
      'trace_ids': (M,)          — trace index for grouped CV
    """
    M = delta_r.shape[0]
    z   = compute_z_trajectory(delta_r, U_pca2)        # (M, T, 2)
    dz  = compute_dz(z)                                 # (M, T-1, 2)
    z_t = z[:, :-1, :]                                  # align z to dz time axis
    vel = np.diff(eyepos, axis=1).astype(np.float32)   # (M, T-1, 2)

    return {
        'z':         z_t,
        'dz':        dz,
        'z_dz':      np.concatenate([z_t, dz], axis=-1),
        'mean_rate': delta_r.mean(axis=1),              # (M, N)
        'vel':       vel,
        'vel_dir':   np.stack([velocity_direction_bins(vel[i]) for i in range(M)]),
        'trace_ids': np.arange(M, dtype=int),
    }


# %% ── Ridge regression cross-validation ─────────────────────────────────────

def ridge_cv_by_trace(
    features: np.ndarray,   # (M*T', d_feat) or will be flattened
    targets:  np.ndarray,   # (M*T', d_out) continuous targets
    trace_ids: np.ndarray,  # (M*T',) group labels for CV split
    n_splits: int = N_SPLITS_CV,
    alphas: np.ndarray = RIDGE_ALPHAS,
) -> dict:
    """
    Grouped ridge regression CV. Groups = trace index (test on held-out traces).

    Returns:
        dict with 'r2_mean', 'r2_std', 'r2_folds': (n_splits,)
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_r2 = []

    for train_idx, test_idx in gkf.split(features, targets, groups=trace_ids):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(features[train_idx])
        X_te = scaler.transform(features[test_idx])
        y_tr = targets[train_idx]
        y_te = targets[test_idx]

        # Inner alpha selection via ridge CV on the training fold
        rcv = RidgeCV(alphas=alphas, fit_intercept=True)
        rcv.fit(X_tr, y_tr)
        y_pred = rcv.predict(X_te)

        # R² averaged over output dimensions
        ss_res = ((y_te - y_pred) ** 2).sum(axis=0)
        ss_tot = ((y_te - y_te.mean(axis=0)) ** 2).sum(axis=0)
        r2_per_dim = 1.0 - ss_res / (ss_tot + 1e-12)
        fold_r2.append(float(r2_per_dim.mean()))

    fold_r2 = np.array(fold_r2)
    return {
        'r2_mean': float(fold_r2.mean()),
        'r2_std':  float(fold_r2.std()),
        'r2_folds': fold_r2,
    }


def _flatten_trials(feat: np.ndarray, vel: np.ndarray, trace_ids: np.ndarray):
    """
    Flatten (M, T', d) features and (M, T', 2) targets to (M*T', d) / (M*T', 2).
    Expands trace_ids to match the time dimension.
    """
    M, T, d = feat.shape
    feat_flat = feat.reshape(M * T, d)
    vel_flat  = vel.reshape(M * T, vel.shape[-1])
    tid_flat  = np.repeat(trace_ids, T)
    return feat_flat, vel_flat, tid_flat


# %% ── Control: time-shuffle ──────────────────────────────────────────────────

def time_shuffle_dz(dz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Shuffle time indices independently per trial and per dimension.
    Preserves the marginal distribution of dz but destroys temporal order.
    """
    dz_shuf = dz.copy()
    M, T, d = dz.shape
    for i in range(M):
        idx = rng.permutation(T)
        dz_shuf[i] = dz[i, idx, :]
    return dz_shuf


# %% ── Control: random projection baseline ────────────────────────────────────

def random_projection_baseline(
    delta_r:   np.ndarray,   # (M, T, N)
    eyepos:    np.ndarray,   # (M, T, 2)
    trace_ids: np.ndarray,   # (M,)
    n_proj: int = N_RANDOM_PROJ,
    d: int = 2,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Decode velocity from Δz_rand for n_proj random d-dim projections of δr.

    Returns:
        r2_dist: (n_proj,) R² values under random projections
    """
    rng = rng or np.random.default_rng(0)
    N = delta_r.shape[2]
    r2_dist = np.empty(n_proj, dtype=np.float32)

    vel = np.diff(eyepos, axis=1).astype(np.float32)   # (M, T-1, 2)

    for k in range(n_proj):
        # Sample random orthonormal d-dim projection
        Q, _ = np.linalg.qr(rng.standard_normal((N, d)))
        U_rand = Q[:, :d]

        z_rand  = compute_z_trajectory(delta_r, U_rand)   # (M, T, d)
        dz_rand = compute_dz(z_rand)                       # (M, T-1, d)

        feat_flat, vel_flat, tid_flat = _flatten_trials(dz_rand, vel, trace_ids)
        res = ridge_cv_by_trace(feat_flat, vel_flat, tid_flat)
        r2_dist[k] = res['r2_mean']

        if (k + 1) % 50 == 0:
            print(f'  random proj {k+1}/{n_proj}, mean R²={r2_dist[:k+1].mean():.3f}', end='\r')

    print()
    return r2_dist


# %% ── Control: shuffled-PCA baseline ────────────────────────────────────────

def shuffled_pca_baseline(
    delta_r:   np.ndarray,   # (M, T, N)
    eyepos:    np.ndarray,
    trace_ids: np.ndarray,
    n_shuffles: int = N_SHUFFLED_PCA,
    d: int = 2,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    PCA on neuron-wise time-shuffled δr → Δz → decode velocity.
    Preserves per-neuron variance but destroys population covariance structure.

    Returns:
        r2_dist: (n_shuffles,) R² values
    """
    from numpy.linalg import eigh
    rng = rng or np.random.default_rng(1)
    M, T, N = delta_r.shape
    r2_dist = np.empty(n_shuffles, dtype=np.float32)
    vel = np.diff(eyepos, axis=1).astype(np.float32)

    X_all = delta_r.reshape(M * T, N)   # pool all time points for PCA

    for k in range(n_shuffles):
        X_shuf = X_all.copy()
        for n in range(N):
            rng.shuffle(X_shuf[:, n])   # shuffle time per neuron

        Xc = X_shuf - X_shuf.mean(axis=0)
        Sigma_shuf = (Xc.T @ Xc) / (M * T - 1)
        w, V = eigh(Sigma_shuf)
        order = np.argsort(w)[::-1]
        U_shuf = V[:, order][:, :d]

        z_shuf  = compute_z_trajectory(delta_r, U_shuf)
        dz_shuf = compute_dz(z_shuf)

        feat_flat, vel_flat, tid_flat = _flatten_trials(dz_shuf, vel, trace_ids)
        res = ridge_cv_by_trace(feat_flat, vel_flat, tid_flat)
        r2_dist[k] = res['r2_mean']

        if (k + 1) % 20 == 0:
            print(f'  shuffled PCA {k+1}/{n_shuffles}', end='\r')

    print()
    return r2_dist


# %% ── Cross-stimulus generalisation curve ────────────────────────────────────

def cross_stimulus_generalization_curve(
    all_stim_data: dict,   # stim_key → {'delta_r': (M,T,N), 'eyepos': (M,T,2)}
    all_U_pca2:    dict,   # stim_key → (N, 2)
    d_values: list = D_VALUES,
) -> dict:
    """
    Sweep shared-basis dimensionality d. For each d:
      - Learn a single shared d-dim PCA subspace on N-1 stimuli
      - Decode velocity on held-out stimulus using that shared subspace

    Returns:
        results: dict with keys being d values, each containing
                 {'within_r2': float, 'cross_r2': float, 'per_stim': dict}
    """
    from numpy.linalg import eigh

    stim_keys = sorted(all_stim_data.keys())
    results = {}

    for d in d_values:
        print(f'\n--- d={d} ---')
        cross_r2s  = []
        within_r2s = []

        for held_out in stim_keys:
            train_keys = [k for k in stim_keys if k != held_out]

            # Pool δr from training stimuli to fit shared subspace
            delta_r_train = []
            for k in train_keys:
                dr = all_stim_data[k]['delta_r']
                M, T, N = dr.shape
                delta_r_train.append(dr.reshape(M * T, N))
            X_train = np.concatenate(delta_r_train, axis=0)

            # PCA on pooled training data
            Xc = X_train - X_train.mean(axis=0)
            # Efficient: use SVD on (n_samples, N) matrix
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            U_shared = Vt[:d].T   # (N, d)

            # Decode on held-out stimulus using shared subspace
            ho_data = all_stim_data[held_out]
            dr_ho   = ho_data['delta_r']
            ep_ho   = ho_data['eyepos']
            tid_ho  = np.arange(dr_ho.shape[0])

            z_ho    = compute_z_trajectory(dr_ho, U_shared)
            dz_ho   = compute_dz(z_ho)
            vel_ho  = np.diff(ep_ho, axis=1).astype(np.float32)

            feat_flat, vel_flat, tid_flat = _flatten_trials(dz_ho, vel_ho, tid_ho)
            cross_res = ridge_cv_by_trace(feat_flat, vel_flat, tid_flat)
            cross_r2s.append(cross_res['r2_mean'])

            # Within-stimulus baseline: per-stimulus PCA
            U_stim = all_U_pca2[held_out]
            if d > 2:
                # Augment with additional PCs from this stimulus's covariance
                dr_ho_flat = dr_ho.reshape(dr_ho.shape[0] * dr_ho.shape[1], N)
                Xc_ho = dr_ho_flat - dr_ho_flat.mean(0)
                _, _, Vt_ho = np.linalg.svd(Xc_ho, full_matrices=False)
                U_stim = Vt_ho[:d].T

            z_w  = compute_z_trajectory(dr_ho, U_stim)
            dz_w = compute_dz(z_w)
            feat_w, vel_w, tid_w = _flatten_trials(dz_w, vel_ho, tid_ho)
            within_res = ridge_cv_by_trace(feat_w, vel_w, tid_w)
            within_r2s.append(within_res['r2_mean'])

            print(f'  hold-out={held_out}: cross R²={cross_res["r2_mean"]:.3f}, '
                  f'within R²={within_res["r2_mean"]:.3f}')

        results[d] = {
            'cross_r2_mean':  float(np.mean(cross_r2s)),
            'cross_r2_std':   float(np.std(cross_r2s)),
            'within_r2_mean': float(np.mean(within_r2s)),
            'within_r2_std':  float(np.std(within_r2s)),
            'per_stim_cross':  dict(zip(stim_keys, cross_r2s)),
            'per_stim_within': dict(zip(stim_keys, within_r2s)),
        }

    return results


# %% ── Analysis A: core readout ───────────────────────────────────────────────

def run_analysis_A_core(
    all_stim_data: dict,
    all_U_pca2:    dict,
    rng: np.random.Generator,
) -> dict:
    """
    For each stimulus: decode velocity from z, dz, [z,dz].
    Also run time-shuffle control.

    Returns nested dict: stim_key → feature_set → {'r2_mean', 'r2_std', 'r2_folds'}
    """
    results = {}

    for stim_key in sorted(all_stim_data.keys()):
        print(f'\n=== {stim_key} ===')
        data = all_stim_data[stim_key]
        feats = extract_features_and_targets(
            data['delta_r'], data['eyepos'], all_U_pca2[stim_key]
        )

        vel       = feats['vel']        # (M, T-1, 2)
        trace_ids = feats['trace_ids']  # (M,)
        stim_res  = {}

        for feat_name in ('z', 'dz', 'z_dz'):
            feat = feats[feat_name]   # (M, T-1, d)
            ff, vf, tf = _flatten_trials(feat, vel, trace_ids)
            stim_res[feat_name] = ridge_cv_by_trace(ff, vf, tf)
            print(f'  {feat_name}: R²={stim_res[feat_name]["r2_mean"]:.3f} '
                  f'± {stim_res[feat_name]["r2_std"]:.3f}')

        # Time-shuffle control on dz
        dz_shuf = time_shuffle_dz(feats['dz'], rng)
        ff_shuf, vf, tf = _flatten_trials(dz_shuf, vel, trace_ids)
        stim_res['dz_timeshuf'] = ridge_cv_by_trace(ff_shuf, vf, tf)
        print(f'  dz_timeshuf: R²={stim_res["dz_timeshuf"]["r2_mean"]:.3f}')

        results[stim_key] = stim_res

    return results


# %% ── Analysis A: specificity (Δz vs surrogate targets) ──────────────────────

def run_analysis_A_specificity(
    all_stim_data: dict,
    all_U_pca2:    dict,
    rng: np.random.Generator,
) -> dict:
    """
    Decode real velocity vs surrogate targets from Δz.
    Targets: real vel, phase-randomised vel, mismatched-trace vel.

    Returns: stim_key → target_name → r2 result
    """
    stim_keys = sorted(all_stim_data.keys())
    results = {}

    for stim_key in stim_keys:
        print(f'\n=== Specificity: {stim_key} ===')
        data  = all_stim_data[stim_key]
        feats = extract_features_and_targets(
            data['delta_r'], data['eyepos'], all_U_pca2[stim_key]
        )
        vel       = feats['vel']
        trace_ids = feats['trace_ids']
        dz        = feats['dz']

        ff, _, tf = _flatten_trials(dz, vel, trace_ids)
        stim_res  = {}

        # Real velocity
        _, vf, _ = _flatten_trials(dz, vel, trace_ids)
        stim_res['real_vel'] = ridge_cv_by_trace(ff, vf, tf)
        print(f'  real vel:   R²={stim_res["real_vel"]["r2_mean"]:.3f}')

        # Phase-randomised velocity
        vel_phase = np.stack([
            phase_randomize_velocity(vel[i], rng) for i in range(vel.shape[0])
        ])
        _, vf_phase, _ = _flatten_trials(dz, vel_phase, trace_ids)
        stim_res['phase_rand_vel'] = ridge_cv_by_trace(ff, vf_phase, tf)
        print(f'  phase-rand: R²={stim_res["phase_rand_vel"]["r2_mean"]:.3f}')

        # Mismatched-trace velocity: roll trace assignments by 1
        M = vel.shape[0]
        vel_mismatch = np.roll(vel, shift=1, axis=0)
        _, vf_mis, _ = _flatten_trials(dz, vel_mismatch, trace_ids)
        stim_res['mismatch_vel'] = ridge_cv_by_trace(ff, vf_mis, tf)
        print(f'  mismatch:   R²={stim_res["mismatch_vel"]["r2_mean"]:.3f}')

        results[stim_key] = stim_res

    return results


# %% ── Analysis B: content-transformation dissociation ────────────────────────

def run_analysis_B(
    all_stim_data: dict,
    all_U_pca2:    dict,
) -> dict:
    """
    2×2 dissociation matrix:
      Rows:    feature sets (mean_rate, dz)
      Columns: tasks (content=stimulus identity, transformation=velocity direction)

    Cross-validated within each stimulus for transformation task.
    Content task uses per-stimulus mean-rate features, grouped CV by trace.

    Returns:
        {
          'mean_rate_content':   r2 (classification accuracy reported as balanced acc)
          'mean_rate_transform': r2 (velocity direction decoding)
          'dz_content':          r2
          'dz_transform':        r2
        }
    """
    from sklearn.linear_model import LogisticRegression

    stim_keys = sorted(all_stim_data.keys())
    K = len(stim_keys)
    results = {}

    # ── Transformation task (continuous R²) ──────────────────────────────────
    print('\n--- Analysis B: transformation task ---')
    for feat_name in ('mean_rate', 'dz'):
        r2s = []
        for stim_key in stim_keys:
            data  = all_stim_data[stim_key]
            feats = extract_features_and_targets(
                data['delta_r'], data['eyepos'], all_U_pca2[stim_key]
            )
            vel       = feats['vel']
            trace_ids = feats['trace_ids']

            if feat_name == 'mean_rate':
                # Tile the static mean-rate feature across time steps
                M, T_1, _ = vel.shape
                mr = feats['mean_rate']  # (M, N)
                feat = np.repeat(mr[:, np.newaxis, :], T_1, axis=1)  # (M, T-1, N)
            else:
                feat = feats['dz']  # (M, T-1, 2)

            ff, vf, tf = _flatten_trials(feat, vel, trace_ids)
            res = ridge_cv_by_trace(ff, vf, tf)
            r2s.append(res['r2_mean'])
            print(f'  {feat_name} | {stim_key}: R²={res["r2_mean"]:.3f}')

        results[f'{feat_name}_transform_r2'] = float(np.mean(r2s))

    # ── Content task (accuracy) ───────────────────────────────────────────────
    # Pool mean-rate and dz features across stimuli for content classification
    print('\n--- Analysis B: content task ---')
    for feat_name in ('mean_rate', 'dz'):
        # Build (M_total, d) and (M_total,) label arrays
        X_list, y_list, g_list = [], [], []
        for label_idx, stim_key in enumerate(stim_keys):
            data  = all_stim_data[stim_key]
            feats = extract_features_and_targets(
                data['delta_r'], data['eyepos'], all_U_pca2[stim_key]
            )
            M = data['delta_r'].shape[0]

            if feat_name == 'mean_rate':
                X = feats['mean_rate']          # (M, N)
            else:
                X = feats['dz'].mean(axis=1)    # (M, 2) — trial-mean Δz

            X_list.append(X)
            y_list.append(np.full(M, label_idx, dtype=int))
            g_list.append(np.arange(M, dtype=int))

        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        g_all = np.concatenate(g_list, axis=0)

        M_min = min(len(g) for g in g_list)
        # Balance classes
        idx = np.concatenate([
            np.where(y_all == k)[0][:M_min] for k in range(K)
        ])
        X_bal = X_all[idx]; y_bal = y_all[idx]; g_bal = g_all[idx % M_min]

        gkf = GroupKFold(n_splits=N_SPLITS_CV)
        accs = []
        for tr, te in gkf.split(X_bal, y_bal, groups=g_bal):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_bal[tr])
            Xte = sc.transform(X_bal[te])
            clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs',
                                     random_state=42)
            clf.fit(Xtr, y_bal[tr])
            accs.append(clf.score(Xte, y_bal[te]))
        acc = float(np.mean(accs))
        results[f'{feat_name}_content_acc'] = acc
        print(f'  {feat_name} | content accuracy: {acc:.3f} (chance={1/K:.3f})')

    return results


# %% ── Analysis C: linear dynamics model ──────────────────────────────────────

def fit_linear_dynamics(
    z:   np.ndarray,   # (M, T, 2)
    vel: np.ndarray,   # (M, T, 2) — aligned to z
) -> dict:
    """
    Fit z(t+1) = A z(t) + B u(t) + c (full) and z(t+1) = B u(t) + c (B-only).

    Returns:
        dict with 'A', 'B', 'c', 'r2_full', 'r2_B_only', 'A_eigenvalues'
    """
    M, T, d = z.shape
    # Align: predict z(t+1) from z(t) and vel(t)
    # vel is (M, T-1, 2) — already temporal diff; use first T-1 time steps
    z_curr = z[:, :-1, :].reshape(-1, d)    # (M*(T-1), 2)
    z_next = z[:, 1:,  :].reshape(-1, d)    # (M*(T-1), 2)
    u      = vel[:, :T-1, :].reshape(-1, 2) # (M*(T-1), 2) — align velocity

    # Full model: [z_curr, u_curr, 1] → z_next
    X_full  = np.hstack([z_curr, u, np.ones((len(z_curr), 1))])
    # B-only: [u_curr, 1] → z_next
    X_Bonly = np.hstack([u, np.ones((len(u), 1))])

    def fit_and_r2(X, y):
        reg = Ridge(alpha=1e-3, fit_intercept=False)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        ss_res = ((y - y_pred) ** 2).sum(axis=0)
        ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
        r2 = float((1 - ss_res / (ss_tot + 1e-12)).mean())
        return reg.coef_, r2

    coef_full,  r2_full  = fit_and_r2(X_full,  z_next)
    coef_Bonly, r2_Bonly = fit_and_r2(X_Bonly, z_next)

    # Parse full model coefficients: coef is (d_out, d_in)
    A = coef_full[:, :d]            # (2, 2)
    B = coef_full[:, d:d+2]         # (2, 2)
    c = coef_full[:, -1]             # (2,)

    eigs = np.linalg.eigvals(A)

    return {
        'A':              A,
        'B':              B,
        'c':              c,
        'r2_full':        r2_full,
        'r2_B_only':      r2_Bonly,
        'A_eigenvalues':  eigs,
        'A_spectral_radius': float(np.abs(eigs).max()),
    }


def compute_flow_field(
    dz:       np.ndarray,   # (M, T-1, 2)
    z:        np.ndarray,   # (M, T-1, 2)
    vel:      np.ndarray,   # (M, T-1, 2)
    grid_res: int = 20,
    n_vel_bins: int = 8,
) -> dict:
    """
    Empirical Δz flow field, binned by z-state and (optionally) velocity direction.

    Returns:
        dict with 'grid_centers_x/y', 'mean_dz_unconditioned', 'mean_dz_per_vel_bin'
    """
    z_flat   = z.reshape(-1, 2)
    dz_flat  = dz.reshape(-1, 2)
    vel_flat = vel.reshape(-1, 2)
    vel_dirs = velocity_direction_bins(vel_flat, n_bins=n_vel_bins)

    x_edges = np.linspace(z_flat[:, 0].min(), z_flat[:, 0].max(), grid_res + 1)
    y_edges = np.linspace(z_flat[:, 1].min(), z_flat[:, 1].max(), grid_res + 1)
    x_ctr   = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_ctr   = 0.5 * (y_edges[:-1] + y_edges[1:])

    xi = np.searchsorted(x_edges[1:-1], z_flat[:, 0])
    yi = np.searchsorted(y_edges[1:-1], z_flat[:, 1])

    # Unconditioned
    mean_dz = np.zeros((grid_res, grid_res, 2))
    counts  = np.zeros((grid_res, grid_res))
    for t in range(len(z_flat)):
        mean_dz[xi[t], yi[t]] += dz_flat[t]
        counts[xi[t], yi[t]]  += 1
    with np.errstate(invalid='ignore'):
        mean_dz /= counts[..., np.newaxis].clip(1)

    # Conditioned on velocity direction
    mean_dz_per_dir = np.zeros((n_vel_bins, grid_res, grid_res, 2))
    counts_per_dir  = np.zeros((n_vel_bins, grid_res, grid_res))
    for t in range(len(z_flat)):
        v = vel_dirs[t]
        mean_dz_per_dir[v, xi[t], yi[t]] += dz_flat[t]
        counts_per_dir[v, xi[t], yi[t]]  += 1
    with np.errstate(invalid='ignore'):
        mean_dz_per_dir /= counts_per_dir[..., np.newaxis].clip(1)

    return {
        'x_centers':            x_ctr,
        'y_centers':            y_ctr,
        'mean_dz_unconditioned': mean_dz,
        'mean_dz_per_vel_dir':   mean_dz_per_dir,
        'counts':               counts,
    }


# %% ── Figures ─────────────────────────────────────────────────────────────────

def plot_figure1(core_results: dict, save: bool = True) -> None:
    """Figure 1: z vs Δz vs [z,Δz] for velocity decoding, + time-shuffle."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    stim_keys = sorted(core_results.keys())
    feat_names = ['z', 'dz', 'z_dz']
    feat_labels = ['z(t)\n(static)', 'Δz(t)\n(dynamic)', '[z,Δz](t)\n(combined)']
    colours = ['#4C72B0', '#DD8452', '#55A868']

    # Left: R² per feature per stimulus
    ax = axes[0]
    x_base = np.arange(len(feat_names))
    w = 0.8 / len(stim_keys)
    for si, stim in enumerate(stim_keys):
        r2s = [core_results[stim][fn]['r2_mean'] for fn in feat_names]
        errs = [core_results[stim][fn]['r2_std']  for fn in feat_names]
        ax.bar(x_base + si * w - 0.4 + w/2, r2s, w * 0.85, yerr=errs,
               label=stim, alpha=0.8, capsize=3)
    ax.set_xticks(x_base)
    ax.set_xticklabels(feat_labels)
    ax.set_ylabel('R² (velocity decoding)')
    ax.set_title('A: Δz > z for velocity prediction')
    ax.axhline(0, color='k', lw=0.5)
    ax.legend(fontsize=6, loc='upper left')

    # Right: time-shuffle control
    ax = axes[1]
    r2_real  = [np.mean([core_results[s]['dz']['r2_mean']         for s in stim_keys])]
    r2_shuf  = [np.mean([core_results[s]['dz_timeshuf']['r2_mean'] for s in stim_keys])]
    ax.bar([0, 1], [r2_real[0], r2_shuf[0]], color=['#DD8452', '#999999'],
           width=0.5, capsize=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Δz (real)', 'Δz (time-shuffled)'])
    ax.set_ylabel('R² (velocity decoding)')
    ax.set_title('C: Time-shuffle collapses Δz readout')
    ax.axhline(0, color='k', lw=0.5)

    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, 'fig1_dz_readout.pdf')
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved: {path}')
    return fig


def plot_figure2_random_projection(
    r2_pca:  float,
    r2_dist: np.ndarray,
    stim_key: str = '',
    save: bool = True,
) -> None:
    """Figure 2A: histogram of random-projection R² with U_pca2 marked."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r2_dist, bins=30, color='#4C72B0', alpha=0.7, label='Random projections')
    ax.axvline(r2_pca, color='#DD8452', lw=2, label=f'U_pca2 R²={r2_pca:.3f}')
    pctile = float(np.mean(r2_dist < r2_pca)) * 100
    ax.set_xlabel('R² (velocity decoding from Δz)')
    ax.set_ylabel('Count')
    ax.set_title(f'Random projection baseline — U_pca2 at {pctile:.0f}th percentile\n{stim_key}')
    ax.legend()
    plt.tight_layout()
    if save:
        safe = stim_key.replace('.', '_')
        path = os.path.join(FIGURES_DIR, f'fig2A_random_proj_{safe}.pdf')
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved: {path}')
    return fig


def plot_figure2_dim_curve(dim_curve: dict, save: bool = True) -> None:
    """Figure 2D: cross-stimulus R² as function of shared-basis dimensionality."""
    d_vals = sorted(dim_curve.keys())
    cross  = [dim_curve[d]['cross_r2_mean']  for d in d_vals]
    cross_err = [dim_curve[d]['cross_r2_std'] for d in d_vals]
    within = [dim_curve[d]['within_r2_mean'] for d in d_vals]
    within_err = [dim_curve[d]['within_r2_std'] for d in d_vals]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(d_vals, cross,  yerr=cross_err,  fmt='o-', label='Cross-stimulus',
                color='#DD8452', capsize=3)
    ax.errorbar(d_vals, within, yerr=within_err, fmt='s--', label='Within-stimulus',
                color='#4C72B0', capsize=3)
    ax.set_xlabel('Shared basis dimensionality d')
    ax.set_ylabel('R² (velocity decoding)')
    ax.set_title('Cross-stimulus generalisation vs dimensionality')
    ax.legend()
    ax.set_xticks(d_vals)
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, 'fig2D_dim_curve.pdf')
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved: {path}')
    return fig


def plot_figure3_dissociation(B_results: dict, n_stim: int, save: bool = True) -> None:
    """Figure 3A: 2×2 dissociation heatmap."""
    chance_content  = 1.0 / n_stim
    data = np.array([
        [B_results['mean_rate_content_acc'],        B_results['mean_rate_transform_r2']],
        [B_results['dz_content_acc'],               B_results['dz_transform_r2']],
    ])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=max(data.max(), 0.6), aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Content\n(accuracy)', 'Transformation\n(R²)'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Mean rate\n(static)', 'Δz\n(dynamic)'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{data[i,j]:.3f}', ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color='white' if data[i,j] > data.max()/2 else 'black')
    ax.set_title(f'Content–transformation dissociation\n(chance content={chance_content:.2f})')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, 'fig3_dissociation.pdf')
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved: {path}')
    return fig


def plot_figure4_flow_field(
    flow: dict,
    stim_key: str = '',
    n_dirs: int = 4,
    save: bool = True,
) -> None:
    """Figure 4A: Input-conditioned flow fields for n_dirs velocity directions."""
    fig, axes = plt.subplots(1, n_dirs, figsize=(4 * n_dirs, 4), sharey=True)
    dir_names = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
    cmap = plt.cm.RdBu_r

    X, Y = np.meshgrid(flow['x_centers'], flow['y_centers'], indexing='ij')
    for di in range(n_dirs):
        ax = axes[di]
        dz_d = flow['mean_dz_per_vel_dir'][di]   # (grid_res, grid_res, 2)
        magnitude = np.hypot(dz_d[..., 0], dz_d[..., 1])
        ax.quiver(X, Y, dz_d[..., 0], dz_d[..., 1], magnitude,
                  cmap=cmap, scale=None, alpha=0.8)
        ax.set_title(f'Eye velocity: {dir_names[di]}')
        ax.set_xlabel('PC1 (z₁)')
        if di == 0:
            ax.set_ylabel('PC2 (z₂)')

    fig.suptitle(f'Input-conditioned flow fields: {stim_key}', fontsize=12)
    plt.tight_layout()
    if save:
        safe = stim_key.replace('.', '_')
        path = os.path.join(FIGURES_DIR, f'fig4_flow_{safe}.pdf')
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved: {path}')
    return fig


# %% ── Main pipeline ──────────────────────────────────────────────────────────

def main(
    force_recompute: bool = False,
    run_random_proj: bool = True,
    run_cross_stim:  bool = True,
    run_B:  bool = True,
    run_analysis_C:  bool = True,
    stim_subset:     list = None,   # None = all 6; pass list to restrict
):
    rng = np.random.default_rng(42)

    # ── Load pre-computed subspaces ───────────────────────────────────────────
    with open(COV_PKL, 'rb') as f:
        cov_results = pickle.load(f)

    with open(FIXATION_PKL, 'rb') as f:
        fixation_results = pickle.load(f)

    # Restrict to stimuli with both U_pca2 and fixation data
    available = sorted(set(cov_results.keys()) & set(fixation_results.keys()))
    if stim_subset is not None:
        available = [s for s in available if s in stim_subset]
    print(f'Stimuli: {available}')

    all_U_pca2 = {k: cov_results[k]['U_pca2'] for k in available}

    # ── Load model (needed for cache miss) ────────────────────────────────────
    model, readout = None, None
    need_model = force_recompute or any(
        not os.path.exists(cache_path(k)) for k in available
    )
    if need_model:
        print('Loading model and readout...')
        model, readout = load_model_and_readout()

    # ── Generate / load δr(t) trajectories ────────────────────────────────────
    all_stim_data = {}
    for stim_key in available:
        entry = fixation_results[stim_key]
        ep_flat  = entry['eyepos'].astype(np.float32)
        n_trials = entry['n_trials']
        ep_3d    = split_eye_traces(ep_flat, n_trials)   # (M, T, 2)

        path = cache_path(stim_key)
        if not os.path.exists(path) or force_recompute:
            image_gray = load_image_gray(stim_key)
            data = load_or_compute_delta_r(
                stim_key, image_gray, ep_3d, model, readout,
                force_recompute=force_recompute,
            )
        else:
            data = load_or_compute_delta_r(
                stim_key, None, ep_3d, None, None,
                force_recompute=False,
            )

        all_stim_data[stim_key] = data

    # ── Analysis A: core readout ───────────────────────────────────────────────
    print('\n\n=== ANALYSIS A: CORE READOUT ===')
    core_results = run_analysis_A_core(all_stim_data, all_U_pca2, rng)

    # Print summary
    print('\nSummary (mean across stimuli):')
    for fn in ('z', 'dz', 'z_dz', 'dz_timeshuf'):
        r2s = [core_results[s][fn]['r2_mean'] for s in available]
        print(f'  {fn:12s}: R²={np.mean(r2s):.3f} ± {np.std(r2s):.3f}')

    plot_figure1(core_results)

    # Decision point check
    dz_r2  = np.mean([core_results[s]['dz']['r2_mean'] for s in available])
    z_r2   = np.mean([core_results[s]['z']['r2_mean']  for s in available])
    print(f'\nDecision: Δz R²={dz_r2:.3f} vs z R²={z_r2:.3f}')
    if dz_r2 <= z_r2 + 0.02:
        print('WARNING: Δz ≈ z. Transformation info is in static state. '
              'Check raw trajectories before proceeding to controls.')

    # ── Analysis A: specificity ───────────────────────────────────────────────
    print('\n=== ANALYSIS A: SPECIFICITY ===')
    spec_results = run_analysis_A_specificity(all_stim_data, all_U_pca2, rng)

    # ── Analysis A: random projection baseline ────────────────────────────────
    rand_proj_results = {}
    if run_random_proj:
        print('\n=== ANALYSIS A: RANDOM PROJECTION BASELINE ===')
        # Run on one representative stimulus (most trials)
        stim_for_rand = max(available, key=lambda k: all_stim_data[k]['delta_r'].shape[0])
        data_r   = all_stim_data[stim_for_rand]
        trace_r  = np.arange(data_r['delta_r'].shape[0])

        r2_dist = random_projection_baseline(
            data_r['delta_r'], data_r['eyepos'], trace_r, rng=rng
        )
        # U_pca2 Δz R² for this stimulus
        r2_pca = core_results[stim_for_rand]['dz']['r2_mean']
        pctile  = float(np.mean(r2_dist < r2_pca)) * 100
        print(f'\nU_pca2 R²={r2_pca:.3f} at {pctile:.0f}th percentile of random dist')
        print(f'Random dist: mean={r2_dist.mean():.3f}, 95th={np.percentile(r2_dist,95):.3f}')
        rand_proj_results = {
            'r2_dist':     r2_dist,
            'r2_pca':      r2_pca,
            'percentile':  pctile,
            'stim':        stim_for_rand,
        }
        plot_figure2_random_projection(r2_pca, r2_dist, stim_for_rand)

    # ── Analysis A: cross-stimulus generalisation curve ───────────────────────
    dim_curve = {}
    if run_cross_stim and len(available) >= 3:
        print('\n=== ANALYSIS A: CROSS-STIMULUS GENERALISATION CURVE ===')
        dim_curve = cross_stimulus_generalization_curve(
            all_stim_data, all_U_pca2, d_values=D_VALUES
        )
        plot_figure2_dim_curve(dim_curve)

    # ── Analysis B: dissociation ──────────────────────────────────────────────
    B_results = {}
    if run_B:
        print('\n=== ANALYSIS B: CONTENT-TRANSFORMATION DISSOCIATION ===')
        B_results = run_analysis_B(all_stim_data, all_U_pca2)
        plot_figure3_dissociation(B_results, n_stim=len(available))

    # ── Analysis C: flow fields + linear dynamics ─────────────────────────────
    C_results = {}
    if run_analysis_C:
        print('\n=== ANALYSIS C: FLOW FIELDS AND LINEAR DYNAMICS ===')
        for stim_key in available:
            data  = all_stim_data[stim_key]
            feats = extract_features_and_targets(
                data['delta_r'], data['eyepos'], all_U_pca2[stim_key]
            )
            z   = feats['z']    # (M, T-1, 2)
            dz  = feats['dz']   # (M, T-1, 2)
            vel = feats['vel']  # (M, T-1, 2)

            # z(t) aligned to dz time axis — feats['z'] is already (M, T-1, 2)
            z_full = compute_z_trajectory(data['delta_r'], all_U_pca2[stim_key])  # (M, T, 2)
            lin = fit_linear_dynamics(z_full, vel)
            print(f'\n{stim_key}:')
            print(f'  Full R²={lin["r2_full"]:.3f}, B-only R²={lin["r2_B_only"]:.3f}')
            print(f'  A eigenvalues: {lin["A_eigenvalues"]}')
            print(f'  A spectral radius: {lin["A_spectral_radius"]:.3f}')

            flow = compute_flow_field(dz, z, vel)
            plot_figure4_flow_field(flow, stim_key)
            C_results[stim_key] = {'linear': lin, 'flow': flow}

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        'core':      core_results,
        'specificity': spec_results,
        'rand_proj': rand_proj_results,
        'dim_curve': dim_curve,
        'B':         B_results,
        'C':         C_results,
    }
    np.save(os.path.join(FIGURES_DIR, 'analysis_summary.npy'),
            summary, allow_pickle=True)
    print(f'\nSummary saved to {FIGURES_DIR}/analysis_summary.npy')
    return summary


# %% ── Entry point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Transformation dynamics analysis')
    parser.add_argument('--recompute', action='store_true',
                        help='Force re-run model even if cache exists')
    parser.add_argument('--no-rand-proj', action='store_true')
    parser.add_argument('--no-cross-stim', action='store_true')
    parser.add_argument('--no-B', action='store_true')
    parser.add_argument('--no-C', action='store_true')
    parser.add_argument('--stim', nargs='+', default=None,
                        help='Restrict to specific stimulus keys')
    args = parser.parse_args()

    main(
        force_recompute=args.recompute,
        run_random_proj=not args.no_rand_proj,
        run_cross_stim=not args.no_cross_stim,
        run_B=not args.no_B,
        run_analysis_C=not args.no_C,
        stim_subset=args.stim,
    )
