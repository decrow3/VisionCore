"""
Displacement Decoding: Does the population encode retinal image displacement?
=============================================================================

**STATUS: RESULTS STAND — but note motivational context**

This analysis is purely spatial: it compares responses at different static
positions without requiring inter-frame temporal continuity. It is therefore
unaffected by the architectural issue that invalidates com_dynamics.py and
transformation_dynamics.py (independent 32-frame windows, fresh GRU state).

The analysis was originally framed as part of a broader "temporal dynamics"
investigation motivated by C ≈ A at −0.20 dB (which turned out to be the
wrong operating point). However, the displacement geometry findings are valid
regardless of that motivation:
  - Within-image R² ≈ 0.998: smooth, locally linear response manifold
  - Cross-image failure: displacement code is image-specific, not generalizable
  - Displacement magnitude sweep: characterizes the geometry of the spatial encoder

These results characterize what the model is — a spatial encoder with a
locally smooth but image-specific response manifold — and are interpretable
on those terms without reference to the temporal architecture question.

Phase 3 (FEM vs static comparison) uses FEM eye traces as position sequences
but treats each position as an independent spatial query. The comparison is
between spatial sampling patterns, not temporal dynamics, and is valid.

------------------------------------------------------------------------

Tests whether response *differences* between positions encode the displacement
between them — the Ahissar "figuring space by time" prediction.

Pipeline
--------
Phase 0  – Compute displacement grid responses (11×11 positions, ±0.05°)
Phase 1  – Within-image displacement decoding (scalar rates vs spatial moments)
Phase 2  – Cross-image generalization (leave-one-image-out)
Phase 3  – FEM vs static comparison (the Ahissar test)
Phase 4  – Cross-neuron lag analysis (supportive, optional)

Usage
-----
python declan/displacement_decoding.py                  # full run
python declan/displacement_decoding.py --phase0-only    # grid compute only
python declan/displacement_decoding.py --skip-phase4    # skip lag analysis
python declan/displacement_decoding.py --recompute      # ignore grid cache
python declan/displacement_decoding.py --stim Hawaii_trees.JPG  # one stimulus
"""

# %% ── Imports ────────────────────────────────────────────────────────────────

import os, sys, pickle, argparse
from typing import Optional
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

PCA_N_COMPONENTS = 50   # feature space has rank ≤ 120 (121 grid positions, differences)

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, '/home/declan/DataYatesV1')

import DataYatesV1  # noqa: F401  # type: ignore[import-untyped]
from spatial_info import get_spatial_readout, make_counterfactual_stim  # type: ignore[import-untyped]
from utils import get_model_and_dataset_configs  # type: ignore[import-untyped]

# Re-use moment computation from com_dynamics
sys.path.insert(0, os.path.join(ROOT, 'declan'))
from com_dynamics import (
    moments_from_maps_batch,
    load_image_gray,
    split_eye_traces,
)

# %% ── Config ─────────────────────────────────────────────────────────────────

CACHE_DIR    = os.path.join(ROOT, 'declan', 'displacement_decoding_cache')
COV_PKL      = os.path.join(ROOT, 'declan', 'translation_covariance', 'all_cov_results.pkl')
FIXATION_PKL = os.path.join(ROOT, 'declan', 'backimage_fixation_results.pkl')
FIGURES_DIR  = os.path.join(ROOT, 'declan', 'displacement_decoding_figures')

# Displacement grid: ±0.05° in steps of 0.01° → 11 × 11 = 121 positions
GRID_RANGE_DEG  = 0.05
GRID_STEP_DEG   = 0.01
GRID_STEPS      = int(round(2 * GRID_RANGE_DEG / GRID_STEP_DEG)) + 1  # 11

N_LAGS          = 32
OUT_SIZE        = (151, 151)
PPD             = 37.5
RIDGE_ALPHAS    = np.logspace(-2, 4, 13)
N_CV_SPLITS     = 5   # within-image CV folds

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# %% ── Model loading ──────────────────────────────────────────────────────────

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model_and_readout():
    model, _ = get_model_and_dataset_configs()
    device = get_device()
    model = model.to(device).eval()

    import dill
    candidates = [
        os.path.join(ROOT, 'scripts', 'mcfarland_outputs_mono.pkl'),
        os.path.join(ROOT, 'scripts', 'mcfarland_outputs.pkl'),
    ]
    outputs = None
    for p in candidates:
        if os.path.exists(p):
            with open(p, 'rb') as f:
                outputs = dill.load(f)
            break
    if outputs is None:
        raise RuntimeError('Could not find scripts/mcfarland_outputs_mono.pkl')

    readout = get_spatial_readout(model, outputs).to(device).eval()
    return model, readout


# %% ── Phase 0: Displacement grid computation ─────────────────────────────────

def grid_cache_path(stim_key: str) -> str:
    safe = stim_key.replace('/', '_').replace('.', '_')
    return os.path.join(CACHE_DIR, f'{safe}_grid.npz')


def compute_displacement_grid(
    model,
    readout,
    image_gray: np.ndarray,
    grid_range_deg: float = GRID_RANGE_DEG,
    grid_step_deg:  float = GRID_STEP_DEG,
    n_lags: int = N_LAGS,
    out_size: tuple = OUT_SIZE,
) -> dict:
    """
    Compute population responses at a regular grid of eye positions.

    For each position (sx, sy), runs the model with a constant (stationary)
    eye trace long enough for the GRU to warm up, then takes the last frame
    as the steady-state response at that position.

    Returns:
        scalar_rates : (n_pos, N)    — spatial-mean of rate map per neuron
        moments      : (n_pos, N, 5) — [com_x, com_y, σ_x, σ_y, σ_xy]
        positions    : (n_pos, 2)    — (x, y) in degrees
    """
    device = get_device()
    shifts = np.arange(-grid_range_deg, grid_range_deg + grid_step_deg / 2,
                       grid_step_deg, dtype=np.float32)
    T_warmup = n_lags + 10

    positions = []
    for sx in shifts:
        for sy in shifts:
            positions.append((float(sx), float(sy)))
    positions = np.array(positions, dtype=np.float32)  # (n_pos, 2)
    n_pos = len(positions)

    scalar_rates_list = []
    moments_list = []

    for i, (sx, sy) in enumerate(positions):
        eyepos = np.tile([sx, sy], (T_warmup, 1)).astype(np.float32)
        full_stack = np.repeat(image_gray[np.newaxis], T_warmup + n_lags, axis=0)
        eyepos_t = torch.from_numpy(eyepos).to(device)

        stim = make_counterfactual_stim(full_stack, eyepos_t,
                                        out_size=out_size, n_lags=n_lags)
        stim_norm = ((stim - 127.0) / 255.0).to(device)

        with torch.no_grad():
            x = model.model.core_forward(stim_norm, None)
            x_last = x[-1:, :, -1]             # (1, C, H_feat, W_feat)
            y = readout(x_last)                 # (1, N, H_out, W_out)
            y_act = model.model.activation(y)   # (1, N, H_out, W_out)
            maps = y_act[0]                     # (N, H_out, W_out)

        # Scalar rates: spatial mean
        rates_np = maps.cpu().float().numpy()          # (N, H, W)
        scalar = rates_np.mean(axis=(-2, -1))          # (N,)
        scalar_rates_list.append(scalar)

        # Spatial moments
        moms = moments_from_maps_batch(maps)            # (N, 5)
        moments_list.append(moms)

        if (i + 1) % 10 == 0 or i == n_pos - 1:
            print(f'  Grid position {i+1}/{n_pos} ({sx:.2f}°, {sy:.2f}°)', end='\r')

    print()
    return {
        'scalar_rates': np.stack(scalar_rates_list, axis=0).astype(np.float32),  # (n_pos, N)
        'moments':      np.stack(moments_list,      axis=0).astype(np.float32),  # (n_pos, N, 5)
        'positions':    positions,                                                  # (n_pos, 2)
    }


def load_or_compute_grid(
    stim_key: str,
    image_gray,
    model,
    readout,
    force_recompute: bool = False,
) -> dict:
    path = grid_cache_path(stim_key)
    if os.path.exists(path) and not force_recompute:
        print(f'  Loading cached grid for {stim_key}')
        d = np.load(path)
        return {k: d[k] for k in d.files}

    print(f'  Computing displacement grid for {stim_key}...')
    data = compute_displacement_grid(model, readout, image_gray)
    np.savez_compressed(path, **data)
    print(f'  Saved: {path}')
    return data


# %% ── Feature construction ───────────────────────────────────────────────────

def build_displacement_features(grid_data: dict, max_displacement_deg: Optional[float] = None) -> dict:
    """
    Build response-difference features and displacement targets from a grid.

    For each pair of positions (i, j), computes:
        Δr = r(j) - r(i)
        target = pos(j) - pos(i)

    If max_displacement_deg is given, only pairs within that distance are used.

    Returns:
        feat_scalar  : (n_pairs, N)    — Δ scalar rates
        feat_moments : (n_pairs, 4N)   — Δ [com_x, com_y, σ_x, σ_y] per neuron
        feat_com     : (n_pairs, 2N)   — Δ com only
        feat_width   : (n_pairs, 2N)   — Δ [σ_x, σ_y] only
        targets      : (n_pairs, 2)    — (δx, δy) in degrees
        pair_idx     : (n_pairs, 2)    — position indices (i, j)
    """
    scalar  = grid_data['scalar_rates']  # (n_pos, N)
    moments = grid_data['moments']       # (n_pos, N, 5)
    pos     = grid_data['positions']     # (n_pos, 2)
    n_pos   = pos.shape[0]

    # Build all ordered pairs (i, j) with i < j
    ii, jj = np.tril_indices(n_pos, k=-1)

    disp = pos[jj] - pos[ii]            # (n_pairs, 2)
    if max_displacement_deg is not None:
        dist = np.linalg.norm(disp, axis=1)
        keep = dist <= max_displacement_deg
        ii, jj, disp = ii[keep], jj[keep], disp[keep]

    N = scalar.shape[1]
    delta_scalar  = scalar[jj] - scalar[ii]                              # (n_pairs, N)
    delta_moments = moments[jj, :, :4] - moments[ii, :, :4]             # (n_pairs, N, 4)
    delta_com     = delta_moments[:, :, :2].reshape(len(ii), 2 * N)     # (n_pairs, 2N)
    delta_width   = delta_moments[:, :, 2:4].reshape(len(ii), 2 * N)   # (n_pairs, 2N)
    delta_all     = delta_moments.reshape(len(ii), 4 * N)               # (n_pairs, 4N)

    # Filter NaN rows (neurons with zero activation produce NaN moments)
    valid = (np.isfinite(delta_scalar).all(axis=1) &
             np.isfinite(delta_all).all(axis=1))
    return {
        'feat_scalar':  delta_scalar[valid],
        'feat_moments': delta_all[valid],
        'feat_com':     delta_com[valid],
        'feat_width':   delta_width[valid],
        'targets':      disp[valid],
        'pair_idx':     np.stack([ii[valid], jj[valid]], axis=1),
    }


# %% ── Decoding ───────────────────────────────────────────────────────────────

def ridge_r2(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_CV_SPLITS,
    n_permutations: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    RidgeCV with KFold CV. Returns mean R² across folds for x, y, combined,
    plus a permutation null (targets shuffled, geometry preserved).

    Permutation null: shuffle y rows within each fold independently, so the
    decoder sees the same feature distribution but broken target alignment.
    This gives a data-driven floor rather than an arbitrary threshold.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_x_list, r2_y_list = [], []
    null_r2_list = []

    for train_idx, test_idx in kf.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        # PCA pre-reduction: feature space has rank ≤ 120 regardless of column count.
        # Hard-thresholding at 50 components is equivalent to ridge (which soft-thresholds
        # the same near-zero singular values), but reduces SVD cost by ~3,600×.
        if Xtr_s.shape[1] > PCA_N_COMPONENTS:
            n_comp = min(PCA_N_COMPONENTS, Xtr_s.shape[0] - 1)
            pca = PCA(n_components=n_comp)
            Xtr_s = pca.fit_transform(Xtr_s)
            Xte_s = pca.transform(Xte_s)

        reg = RidgeCV(alphas=RIDGE_ALPHAS)
        reg.fit(Xtr_s, ytr)
        pred = reg.predict(Xte_s)

        ss_res = ((yte - pred) ** 2).sum(axis=0)
        ss_tot = ((yte - yte.mean(axis=0)) ** 2).sum(axis=0)
        r2 = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)
        r2_x_list.append(r2[0])
        r2_y_list.append(r2[1])

        # Permutation nulls: Xtr_s already PCA-reduced, reuse it directly
        for _ in range(n_permutations):
            perm = rng.permutation(len(ytr))
            reg_null = RidgeCV(alphas=RIDGE_ALPHAS)
            reg_null.fit(Xtr_s, ytr[perm])
            pred_null = reg_null.predict(Xte_s)
            ss_res_null = ((yte - pred_null) ** 2).sum(axis=0)
            r2_null = np.where(ss_tot > 0, 1 - ss_res_null / ss_tot, 0.0)
            null_r2_list.append(float(r2_null.mean()))

    return {
        'r2_x':    float(np.mean(r2_x_list)),
        'r2_y':    float(np.mean(r2_y_list)),
        'r2_mean': float(np.mean(r2_x_list + r2_y_list)),
        'null_r2': float(np.mean(null_r2_list)),
        'null_r2_95': float(np.percentile(null_r2_list, 95)),
    }


def decode_displacement(
    grid_data: dict,
    max_displacement_deg: Optional[float] = None,
    n_splits: int = N_CV_SPLITS,
) -> dict:
    """
    Within-image displacement decoding for all feature sets.
    """
    feats = build_displacement_features(grid_data, max_displacement_deg)
    targets = feats['targets']

    results = {}
    for name in ['feat_scalar', 'feat_moments', 'feat_com', 'feat_width']:
        X = feats[name]
        print(f'    Decoding {name} ({X.shape[1]} features, {len(X)} pairs)...')
        results[name] = ridge_r2(X, targets, n_splits=n_splits)

    return results


def sweep_displacement_magnitude(
    grid_data: dict,
    magnitudes_deg: Optional[list] = None,
    n_splits: int = N_CV_SPLITS,
) -> dict:
    """
    Local linearity check: decode displacement with only pairs within radius r.

    If the code is genuinely differential / local, R² should be highest for
    small displacements (where the response manifold is locally linear) and
    degrade for large ones (where nonlinearities dominate).

    Computes R² vs. max_displacement_deg for each feature set.
    """
    if magnitudes_deg is None:
        magnitudes_deg = [0.01, 0.02, 0.03, 0.04, 0.05]

    results = {mag: {} for mag in magnitudes_deg}
    for mag in magnitudes_deg:
        feats = build_displacement_features(grid_data, max_displacement_deg=mag)
        targets = feats['targets']
        n_pairs = len(targets)
        print(f'  Sweep r={mag:.2f}°: {n_pairs} pairs', end='')
        if n_pairs < n_splits * 5:
            print(' — too few pairs, skipping')
            for fname in ['feat_scalar', 'feat_moments', 'feat_com']:
                results[mag][fname] = {'r2_mean': np.nan, 'null_r2': np.nan}
            continue
        print()
        for fname in ['feat_scalar', 'feat_moments', 'feat_com']:
            X = feats[fname]
            results[mag][fname] = ridge_r2(X, targets, n_splits=n_splits,
                                            n_permutations=10)

    return results


# %% ── Bilinear interpolation of grid responses ───────────────────────────────

def build_grid_interpolators(grid_data: dict) -> dict:
    """
    Build RegularGridInterpolator objects for scalar rates and moments.

    The grid is assumed to be on a regular (GRID_STEPS × GRID_STEPS) lattice
    with equal spacing in x and y.  Any position within the grid bounds can
    then be looked up with sub-pixel accuracy.

    Returns:
        interp_scalar  : callable (n_pts, 2) → (n_pts, N)
        interp_moments : callable (n_pts, 2) → (n_pts, N, 4)
        x_vals, y_vals : 1-D grids (sorted unique x and y coordinates)
    """
    pos     = grid_data['positions']     # (n_pos, 2)
    scalar  = grid_data['scalar_rates']  # (n_pos, N)
    moments = grid_data['moments']       # (n_pos, N, 5)

    x_vals = np.sort(np.unique(pos[:, 0]))
    y_vals = np.sort(np.unique(pos[:, 1]))
    nx, ny = len(x_vals), len(y_vals)
    N = scalar.shape[1]

    # Reshape to (nx, ny, N/5) grids
    # positions are ordered: for sx in x_vals: for sy in y_vals (see compute_displacement_grid)
    scalar_grid  = scalar.reshape(nx, ny, N)                       # (nx, ny, N)
    moments_grid = moments[:, :, :4].reshape(nx, ny, N, 4)         # (nx, ny, N, 4)

    # Build per-neuron interpolators for scalar rates
    interp_scalar_list = [
        RegularGridInterpolator((x_vals, y_vals), scalar_grid[:, :, n],
                                method='linear', bounds_error=False,
                                fill_value=np.nan)
        for n in range(N)
    ]

    # Build per-neuron-per-moment interpolators for moments
    interp_moments_list = [
        [
            RegularGridInterpolator((x_vals, y_vals), moments_grid[:, :, n, k],
                                    method='linear', bounds_error=False,
                                    fill_value=np.nan)
            for k in range(4)
        ]
        for n in range(N)
    ]

    def interp_scalar(pts):
        """pts: (M, 2). Returns (M, N)."""
        return np.stack([f(pts) for f in interp_scalar_list], axis=1)

    def interp_moments(pts):
        """pts: (M, 2). Returns (M, N, 4)."""
        return np.stack(
            [np.stack([interp_moments_list[n][k](pts) for k in range(4)], axis=1)
             for n in range(N)],
            axis=1
        )

    return {
        'interp_scalar':  interp_scalar,
        'interp_moments': interp_moments,
        'x_vals':         x_vals,
        'y_vals':         y_vals,
    }


# %% ── Phase 2: Cross-image generalization ────────────────────────────────────

def cross_image_displacement_decoding(all_grid_data: dict) -> dict:
    """
    Leave-one-image-out: train on 5 images, test on 6th.
    Tests whether the displacement code generalises across image content.
    """
    stim_keys = list(all_grid_data.keys())
    all_feats = {k: build_displacement_features(all_grid_data[k]) for k in stim_keys}

    results = {}
    for held_out in stim_keys:
        train_keys = [k for k in stim_keys if k != held_out]

        test_feats = all_feats[held_out]
        y_test     = test_feats['targets']

        for fname in ['feat_scalar', 'feat_moments', 'feat_com', 'feat_width']:
            X_train = np.concatenate([all_feats[k][fname] for k in train_keys], axis=0)
            y_train = np.concatenate([all_feats[k]['targets'] for k in train_keys], axis=0)
            X_test_f = test_feats[fname]

            # Remove NaN columns (from moment features on held-out image)
            valid_cols = np.isfinite(X_train).all(axis=0) & np.isfinite(X_test_f).all(axis=0)
            X_train_f = X_train[:, valid_cols]
            X_test_f  = X_test_f[:, valid_cols]

            scaler = StandardScaler()
            X_train_f = scaler.fit_transform(X_train_f)
            X_test_f  = scaler.transform(X_test_f)

            if X_train_f.shape[1] > PCA_N_COMPONENTS:
                n_comp = min(PCA_N_COMPONENTS, X_train_f.shape[0] - 1)
                pca = PCA(n_components=n_comp)
                X_train_f = pca.fit_transform(X_train_f)
                X_test_f  = pca.transform(X_test_f)

            reg = RidgeCV(alphas=RIDGE_ALPHAS)
            reg.fit(X_train_f, y_train)
            pred = reg.predict(X_test_f)

            ss_res = ((y_test - pred) ** 2).sum(axis=0)
            ss_tot = ((y_test - y_test.mean(axis=0)) ** 2).sum(axis=0)
            r2 = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)

            if held_out not in results:
                results[held_out] = {}
            results[held_out][fname] = {
                'r2_x': float(r2[0]),
                'r2_y': float(r2[1]),
                'r2_mean': float(r2.mean()),
            }

        print(f'  Cross-image: held out {held_out}, '
              f'scalar R²={results[held_out]["feat_scalar"]["r2_mean"]:.3f}, '
              f'moments R²={results[held_out]["feat_moments"]["r2_mean"]:.3f}')

    return results


# %% ── Phase 3: FEM vs static comparison ─────────────────────────────────────

def _run_fem_grid_response(
    model,
    readout,
    image_gray: np.ndarray,
    eyepos_3d: np.ndarray,  # (M, T, 2)
    n_lags: int = N_LAGS,
    out_size: tuple = OUT_SIZE,
    batch_size: int = 32,
) -> dict:
    """
    Run model with real FEM eye traces and return instantaneous rate vectors.

    Returns:
        scalar_rates : (M*T, N)  — spatial-mean rate at each time step
        moments      : (M*T, N, 5)
        displacements: (M*T, 2)  — eyepos(t) - mean_eyepos per trial
        trial_ids    : (M*T,)
    """
    device = get_device()
    M, T, _ = eyepos_3d.shape

    all_rates = []
    all_moments = []
    all_disps = []
    all_trial_ids = []

    for i in range(M):
        ep = eyepos_3d[i]             # (T, 2)
        mean_ep = ep.mean(axis=0)     # (2,) — fixation center
        disp = ep - mean_ep           # (T, 2) — displacement from fixation

        full_stack = np.repeat(image_gray[np.newaxis], T + n_lags, axis=0)
        ep_t = torch.from_numpy(ep).float().to(device)
        stim = make_counterfactual_stim(full_stack, ep_t,
                                        out_size=out_size, n_lags=n_lags)
        stim_norm = ((stim - 127.0) / 255.0).to(device)

        trial_rates = []
        trial_moms = []

        for t_start in range(0, T, batch_size):
            t_end = min(t_start + batch_size, T)
            x_batch = stim_norm[t_start:t_end].to(device)
            with torch.no_grad():
                x = model.model.core_forward(x_batch, None)
                x_last = x[:, :, -1]            # (B, C, H, W)
                y = readout(x_last)             # (B, N, H_out, W_out)
                y_act = model.model.activation(y)  # (B, N, H_out, W_out)

            rates_np = y_act.cpu().float().numpy()  # (B, N, H, W)
            sc = rates_np.mean(axis=(-2, -1))        # (B, N)
            trial_rates.append(sc)

            for b in range(y_act.shape[0]):
                m = moments_from_maps_batch(y_act[b])  # (N, 5)
                trial_moms.append(m)

            del x_batch, x, x_last, y, y_act
            torch.cuda.empty_cache()

        trial_rates = np.concatenate(trial_rates, axis=0)  # (T, N)
        trial_moms = np.stack(trial_moms, axis=0)           # (T, N, 5)

        all_rates.append(trial_rates)
        all_moments.append(trial_moms)
        all_disps.append(disp)
        all_trial_ids.append(np.full(T, i, dtype=int))

        if (i + 1) % 5 == 0 or i == M - 1:
            print(f'  FEM trials: {i+1}/{M}', end='\r')

    print()
    return {
        'scalar_rates': np.concatenate(all_rates,    axis=0).astype(np.float32),  # (M*T, N)
        'moments':      np.concatenate(all_moments,  axis=0).astype(np.float32),  # (M*T, N, 5)
        'displacements': np.concatenate(all_disps,   axis=0).astype(np.float32),  # (M*T, 2)
        'trial_ids':    np.concatenate(all_trial_ids, axis=0),                    # (M*T,)
    }


def fem_cache_path(stim_key: str) -> str:
    safe = stim_key.replace('/', '_').replace('.', '_')
    return os.path.join(CACHE_DIR, f'{safe}_fem_rates.npz')


def fem_vs_static_displacement(
    stim_key: str,
    grid_data: dict,
    model,
    readout,
    image_gray: np.ndarray,
    eyepos_3d: np.ndarray,  # (M, T, 2)
    force_recompute: bool = False,
) -> dict:
    """
    Compare velocity decoding from static vs FEM response differences.

    Both conditions use the same *differential* framing as Phase 1, so the
    comparison is clean:

      Static baseline:
        Δr_static(t) = interp_grid(pos_{t+1}) - interp_grid(pos_t)
        Uses bilinear interpolation of the cached grid — represents a pure
        spatial code with no recurrent history.

      FEM trajectory:
        Δr_fem(t) = r_fem(t+1) - r_fem(t)
        Actual response difference under real FEM; the GRU has seen the full
        preceding trajectory.

      Both predict: v(t) = eyepos(t+1) - eyepos(t)  [instantaneous velocity]

    This framing is consistent with Phase 1 (Δr predicts Δpos) and isolates
    exactly the Ahissar question: does temporal integration of movement under
    real FEM add velocity information beyond what the static position code
    already provides through its local gradient?

    CV: leave-one-trial-out across trials. A permutation null is computed for
    each condition by shuffling v(t) targets within each training fold.
    """
    # ── Load or compute FEM rates ─────────────────────────────────────────────
    path = fem_cache_path(stim_key)
    if os.path.exists(path) and not force_recompute:
        print(f'  Loading cached FEM rates for {stim_key}')
        d = np.load(path)
        fem_data = {k: d[k] for k in d.files}
    else:
        print(f'  Computing FEM rates for {stim_key}...')
        fem_data = _run_fem_grid_response(model, readout, image_gray, eyepos_3d)
        np.savez_compressed(path, **fem_data)
        print(f'  Saved: {path}')

    fem_scalar = fem_data['scalar_rates']   # (M*T, N)
    fem_moms   = fem_data['moments']        # (M*T, N, 5)
    fem_pos    = fem_data['displacements']  # (M*T, 2) — eyepos relative to trial mean
    fem_trials = fem_data['trial_ids']      # (M*T,)

    # ── Build static comparator via bilinear interpolation ────────────────────
    print(f'  Building grid interpolators for {stim_key}...')
    interps = build_grid_interpolators(grid_data)

    # Clip FEM positions to grid bounds before interpolating
    x_min, x_max = interps['x_vals'][0], interps['x_vals'][-1]
    y_min, y_max = interps['y_vals'][0], interps['y_vals'][-1]
    pos_clipped = np.clip(fem_pos,
                          [x_min, y_min],
                          [x_max, y_max]).astype(np.float32)

    static_scalar_all  = interps['interp_scalar'](pos_clipped)         # (M*T, N)
    static_moments_all = interps['interp_moments'](pos_clipped)        # (M*T, N, 4)
    static_moments_all = static_moments_all.reshape(len(fem_pos), -1)  # (M*T, 4N)

    # ── Convert to frame-to-frame differences (velocity target) ───────────────
    # Work per trial so we don't diff across trial boundaries.
    unique_trials = np.unique(fem_trials)
    rng = np.random.default_rng(0)
    results = {tag: {'r2s_static': [], 'r2s_fem': [], 'nulls': []}
               for tag in ['scalar', 'moments']}

    for held_trial in unique_trials:
        test_mask = fem_trials == held_trial

        for tag, static_all, fem_feat_all in [
            ('scalar',  static_scalar_all,
             fem_scalar),
            ('moments', static_moments_all,
             fem_moms[:, :, :4].reshape(len(fem_pos), -1)),
        ]:
            def _diff_within_trial(arr, mask):
                """Frame-to-frame differences, keeping only valid (finite) rows."""
                # arr shape: (sum(mask), feat_dim)
                # Returns (sum(mask)-1, feat_dim), (sum(mask)-1, 2) for targets
                a = arr[mask]
                p = fem_pos[mask]
                delta_r = np.diff(a, axis=0)           # (T-1, feat_dim)
                delta_v = np.diff(p, axis=0)           # (T-1, 2)
                valid = (np.isfinite(delta_r).all(axis=1) &
                         np.isfinite(delta_v).all(axis=1))
                return delta_r[valid], delta_v[valid]

            # Collect training data across all non-held-out trials
            Xtr_s_list, Xtr_f_list, ytr_list = [], [], []
            for tr in unique_trials:
                if tr == held_trial:
                    continue
                mask_tr = fem_trials == tr
                dr_s, dv = _diff_within_trial(static_all, mask_tr)
                dr_f, _  = _diff_within_trial(fem_feat_all, mask_tr)
                Xtr_s_list.append(dr_s)
                Xtr_f_list.append(dr_f)
                ytr_list.append(dv)

            if not Xtr_s_list:
                continue

            Xtr_s = np.concatenate(Xtr_s_list, axis=0)
            Xtr_f = np.concatenate(Xtr_f_list, axis=0)
            ytr   = np.concatenate(ytr_list,   axis=0)

            Xte_s, yte = _diff_within_trial(static_all,   test_mask)
            Xte_f, _   = _diff_within_trial(fem_feat_all, test_mask)

            if len(yte) < 5:
                continue

            for X_tr, X_te, r2_list_key in [
                (Xtr_s, Xte_s, 'r2s_static'),
                (Xtr_f, Xte_f, 'r2s_fem'),
            ]:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)
                reg = RidgeCV(alphas=RIDGE_ALPHAS)
                reg.fit(X_tr_s, ytr)
                pred = reg.predict(X_te_s)
                ss_res = ((yte - pred) ** 2).sum(axis=0)
                ss_tot = ((yte - yte.mean(axis=0)) ** 2).sum(axis=0)
                r2 = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)
                results[tag][r2_list_key].append(float(r2.mean()))

            # Permutation null on FEM (shuffle velocity targets)
            perm = rng.permutation(len(ytr))
            scaler = StandardScaler()
            Xf_s = scaler.fit_transform(Xtr_f)
            Xfe_s = scaler.transform(Xte_f)
            reg_null = RidgeCV(alphas=RIDGE_ALPHAS)
            reg_null.fit(Xf_s, ytr[perm])
            pred_null = reg_null.predict(Xfe_s)
            ss_null = ((yte - pred_null) ** 2).sum(axis=0)
            ss_tot  = ((yte - yte.mean(axis=0)) ** 2).sum(axis=0)
            r2_null = np.where(ss_tot > 0, 1 - ss_null / ss_tot, 0.0)
            results[tag]['nulls'].append(float(r2_null.mean()))

    out = {}
    for tag in ['scalar', 'moments']:
        out[f'static_{tag}'] = float(np.nanmean(results[tag]['r2s_static']))
        out[f'fem_{tag}']    = float(np.nanmean(results[tag]['r2s_fem']))
        out[f'null_{tag}']   = float(np.nanmean(results[tag]['nulls']))

    return out


# %% ── Phase 4: Cross-neuron lag analysis ────────────────────────────────────

def compute_pairwise_lags(
    fem_data: dict,
    readout_positions: np.ndarray,  # (N, 2) — RF centres in pixels
    ppd: float = PPD,
    max_lag: int = 10,
    n_pairs: int = 500,
    min_speed_dps: float = 0.005,
) -> dict:
    """
    Cross-neuron lag analysis: does τ_ij ≈ RF_sep_ij / V_drift?

    For a random sample of neuron pairs, computes the cross-correlation
    peak lag and compares to the predicted lag from RF separation and drift speed.

    Returns:
        observed_lags    : (n_pairs,) — peak-correlation lag in frames
        predicted_lags   : (n_pairs,) — |RF_sep along drift| / V_drift
        rf_separations   : (n_pairs,) — projected RF separation (pixels)
        drift_speeds     : (n_pairs,) — mean drift speed during correlated windows (deg/frame)
        correlation      : float — Pearson r between observed and predicted lags
    """
    rates   = fem_data['scalar_rates']   # (M*T, N)
    disps   = fem_data['displacements']  # (M*T, 2)
    trials  = fem_data['trial_ids']      # (M*T,)
    vel     = np.diff(disps, axis=0, prepend=disps[:1])  # (M*T, 2) approx velocity

    N = rates.shape[1]
    rng = np.random.default_rng(0)

    # Sample random neuron pairs
    pair_i = rng.integers(0, N, size=n_pairs)
    pair_j = rng.integers(0, N, size=n_pairs)
    same = pair_i == pair_j
    pair_j[same] = (pair_j[same] + 1) % N

    # RF separations in degrees
    rf_sep_pix = readout_positions[pair_j] - readout_positions[pair_i]  # (n_pairs, 2)
    rf_sep_deg = rf_sep_pix / ppd                                         # (n_pairs, 2)

    observed_lags   = np.full(n_pairs, np.nan)
    predicted_lags  = np.full(n_pairs, np.nan)
    rf_sep_proj     = np.full(n_pairs, np.nan)
    drift_speeds    = np.full(n_pairs, np.nan)

    # Per-trial cross-correlations, then average
    unique_trials = np.unique(trials)

    for p in range(n_pairs):
        i, j = pair_i[p], pair_j[p]
        lags = np.arange(-max_lag, max_lag + 1)
        ccf = np.zeros(len(lags))
        n_valid = 0

        for tr in unique_trials:
            mask = trials == tr
            ri = rates[mask, i]
            rj = rates[mask, j]
            v  = vel[mask]

            if len(ri) < 2 * max_lag + 5:
                continue

            # Mean drift direction for this trial
            v_mean = v.mean(axis=0)
            v_speed = np.linalg.norm(v_mean)
            if v_speed < min_speed_dps:
                continue

            v_dir = v_mean / v_speed  # unit vector

            # Cross-correlation
            ri_c = ri - ri.mean()
            rj_c = rj - rj.mean()
            std_prod = ri_c.std() * rj_c.std()
            if std_prod < 1e-10:
                continue

            for k, lag in enumerate(lags):
                if lag >= 0:
                    ccf[k] += np.dot(ri_c[:len(ri)-lag], rj_c[lag:]) / (std_prod * len(ri))
                else:
                    ccf[k] += np.dot(ri_c[-lag:], rj_c[:len(rj)+lag]) / (std_prod * len(ri))
            n_valid += 1

            # Accumulate projected RF separation and speed
            proj = float(np.dot(rf_sep_deg[p], v_dir))
            rf_sep_proj[p] = proj if np.isnan(rf_sep_proj[p]) else rf_sep_proj[p] + proj
            drift_speeds[p] = v_speed if np.isnan(drift_speeds[p]) else drift_speeds[p] + v_speed

        if n_valid > 0:
            ccf /= n_valid
            peak_idx = int(np.argmax(np.abs(ccf)))
            observed_lags[p] = float(lags[peak_idx])
            rf_sep_proj[p] /= n_valid
            drift_speeds[p] /= n_valid
            if drift_speeds[p] > 0:
                predicted_lags[p] = rf_sep_proj[p] / drift_speeds[p]

    # Pearson r between observed and predicted (finite only)
    valid = np.isfinite(observed_lags) & np.isfinite(predicted_lags)
    if valid.sum() >= 3:
        from scipy.stats import pearsonr
        r, pval = pearsonr(observed_lags[valid], predicted_lags[valid])
    else:
        r, pval = np.nan, np.nan

    return {
        'observed_lags':  observed_lags,
        'predicted_lags': predicted_lags,
        'rf_sep_proj':    rf_sep_proj,
        'drift_speeds':   drift_speeds,
        'pearson_r':      float(r),  # type: ignore[arg-type]
        'pearson_p':      float(pval),  # type: ignore[arg-type]
    }


# %% ── Plotting ───────────────────────────────────────────────────────────────

def plot_phase1(results_by_stim: dict, save: bool = True):
    """Grouped bar chart of R² by feature set, per stimulus + mean, with null band."""
    feature_names = ['feat_scalar', 'feat_com', 'feat_width', 'feat_moments']
    labels = ['Scalar Δr', 'ΔCoM', 'ΔWidth', 'Δ All moments']
    stim_keys = list(results_by_stim.keys())
    n_stim = len(stim_keys)
    x = np.arange(len(feature_names))
    width = 0.7 / (n_stim + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for k, stim_key in enumerate(stim_keys):
        r2s = [results_by_stim[stim_key].get(fn, {}).get('r2_mean', np.nan)
               for fn in feature_names]
        ax.bar(x + k * width, r2s, width, label=stim_key, alpha=0.8)

    # Mean across stimuli
    mean_r2s, mean_null_95 = [], []
    for fn in feature_names:
        vals = [results_by_stim[s].get(fn, {}).get('r2_mean', np.nan) for s in stim_keys]
        null = [results_by_stim[s].get(fn, {}).get('null_r2_95', np.nan) for s in stim_keys]
        mean_r2s.append(np.nanmean(vals))
        mean_null_95.append(np.nanmean(null))
    ax.bar(x + n_stim * width, mean_r2s, width, label='Mean', color='black', alpha=0.9)

    # Permutation null (95th percentile) as a horizontal reference per group
    for xi, null_hi in zip(x + n_stim * width / 2, mean_null_95):
        ax.plot([xi - 0.3, xi + 0.3], [null_hi, null_hi], 'r--', lw=1.2,
                label='Null 95th pct' if xi == x[0] + n_stim * width / 2 else '')

    ax.set_xticks(x + width * n_stim / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel('R² (mean δx + δy)')
    ax.set_title('Phase 1: Within-image displacement decoding\n'
                 '(red dashes = permutation null 95th pct)')
    ax.legend(fontsize=7, ncol=2)
    ax.axhline(0, color='k', lw=0.5)
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, 'phase1_within_image.pdf')
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved: {path}')
    return fig


def plot_sweep(sweep_results: dict, save: bool = True):
    """R² vs max displacement magnitude — local linearity check."""
    mags = sorted(sweep_results.keys())
    feature_names = ['feat_scalar', 'feat_com', 'feat_moments']
    labels = ['Scalar', 'ΔCoM', 'All moments']
    colors = ['steelblue', 'forestgreen', 'tomato']

    fig, ax = plt.subplots(figsize=(6, 4))
    for fn, label, color in zip(feature_names, labels, colors):
        r2s  = [sweep_results[m].get(fn, {}).get('r2_mean', np.nan) for m in mags]
        nulls = [sweep_results[m].get(fn, {}).get('null_r2', np.nan) for m in mags]
        mags_deg = [m * 1000 for m in mags]  # convert to mdeg for readability
        ax.plot(mags_deg, r2s,  '-o', color=color, label=label)
        ax.plot(mags_deg, nulls, '--', color=color, alpha=0.4, lw=0.8)

    ax.set_xlabel('Max displacement radius (mdeg)')
    ax.set_ylabel('R² (mean δx + δy)')
    ax.set_title('Displacement magnitude sweep\n(dashed = permutation null)')
    ax.legend()
    ax.axhline(0, color='k', lw=0.5)
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, 'sweep_displacement_magnitude.pdf')
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved: {path}')
    return fig


def plot_phase2(within_results: dict, cross_results: dict, save: bool = True):
    """Scatter: within-image R² vs cross-image R², per feature set."""
    feature_names = ['feat_scalar', 'feat_com', 'feat_moments']
    labels = ['Scalar', 'CoM', 'All moments']
    stim_keys = list(cross_results.keys())

    fig, axes = plt.subplots(1, len(feature_names), figsize=(12, 4))
    for ax, fn, label in zip(axes, feature_names, labels):
        within = [within_results[s].get(fn, {}).get('r2_mean', np.nan) for s in stim_keys]
        cross  = [cross_results[s].get(fn, {}).get('r2_mean', np.nan) for s in stim_keys]
        ax.scatter(within, cross, s=60, zorder=3)
        for s, wx, cx in zip(stim_keys, within, cross):
            ax.annotate(s[:8], (wx, cx), fontsize=6, ha='left')
        lo = min(min(within), min(cross), 0)
        hi = max(max(within), max(cross), 0.05)
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, label='y=x')
        ax.set_xlabel('Within-image R²')
        ax.set_ylabel('Cross-image R²')
        ax.set_title(label)
        ax.set_aspect('equal')
    plt.suptitle('Phase 2: Cross-image generalization', fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, 'phase2_cross_image.pdf')
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved: {path}')
    return fig


def plot_phase3(phase3_by_stim: dict, save: bool = True):
    """
    Bar chart: static vs FEM velocity-decoding R² for scalar and moments.

    Both conditions use frame-to-frame Δr to predict Δpos (velocity):
      Static: Δr from bilinear-interpolated grid
      FEM:    Δr from real FEM trajectory (GRU has temporal history)
    Permutation null shown as horizontal dashed line per feature group.
    """
    stim_keys = list(phase3_by_stim.keys())
    tags = ['scalar', 'moments']
    x = np.arange(len(tags))
    width = 0.28

    fig, ax = plt.subplots(figsize=(7, 4))
    for k, tag in enumerate(tags):
        mean_s = np.nanmean([phase3_by_stim[s].get(f'static_{tag}', np.nan) for s in stim_keys])
        mean_f = np.nanmean([phase3_by_stim[s].get(f'fem_{tag}',    np.nan) for s in stim_keys])
        mean_n = np.nanmean([phase3_by_stim[s].get(f'null_{tag}',   np.nan) for s in stim_keys])
        ax.bar(k - width, mean_s, width, label='Static (interp)' if k == 0 else '', color='steelblue', alpha=0.85)
        ax.bar(k,         mean_f, width, label='FEM'              if k == 0 else '', color='tomato',    alpha=0.85)
        # Null line
        ax.plot([k - width * 1.5, k + width * 0.5], [mean_n, mean_n], 'k--', lw=1.2,
                label='Null' if k == 0 else '')

    ax.set_xticks(x)
    ax.set_xticklabels(['Scalar Δr → Δv', 'Moment Δr → Δv'])
    ax.set_ylabel('Mean R² (velocity decoding)')
    ax.set_title('Phase 3: Static (grid-interpolated) vs FEM trajectory\n'
                 'Ahissar test: does movement history add velocity information?\n'
                 'dashed = permutation null')
    ax.legend()
    ax.axhline(0, color='k', lw=0.5)
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, 'phase3_fem_vs_static.pdf')
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved: {path}')
    return fig


def plot_phase4(lag_results: dict, save: bool = True):
    """Scatter: observed lag vs predicted lag = RF_sep / V_drift."""
    obs  = lag_results['observed_lags']
    pred = lag_results['predicted_lags']
    r    = lag_results['pearson_r']
    p    = lag_results['pearson_p']
    speed = lag_results['drift_speeds']

    valid = np.isfinite(obs) & np.isfinite(pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(pred[valid], obs[valid], c=speed[valid], cmap='viridis',
                    s=15, alpha=0.5)
    plt.colorbar(sc, ax=ax, label='Mean drift speed (deg/frame)')
    lo = min(pred[valid].min(), obs[valid].min()) - 1
    hi = max(pred[valid].max(), obs[valid].max()) + 1
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('Predicted lag: RF_sep / V_drift (frames)')
    ax.set_ylabel('Observed cross-correlation lag (frames)')
    ax.set_title(f'Phase 4: Cross-neuron lag structure\nr={r:.3f}, p={p:.3g}, n={valid.sum()}')
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, 'phase4_pairwise_lags.pdf')
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved: {path}')
    return fig


# %% ── Main ───────────────────────────────────────────────────────────────────

def main(
    stim_subset:      Optional[list] = None,
    phase0_only:      bool  = False,
    skip_phase3:      bool  = False,
    skip_phase4:      bool  = False,
    force_recompute:  bool  = False,
):
    # ── Load metadata ─────────────────────────────────────────────────────────
    with open(COV_PKL, 'rb') as f:
        cov_results = pickle.load(f)
    with open(FIXATION_PKL, 'rb') as f:
        fixation_results = pickle.load(f)

    available = sorted(set(cov_results.keys()) & set(fixation_results.keys()))
    if stim_subset:
        available = [s for s in available if s in stim_subset]
    print(f'Stimuli ({len(available)}): {available}')

    # ── Load model ────────────────────────────────────────────────────────────
    print('\nLoading model...')
    model, readout = load_model_and_readout()

    # ── Phase 0: Displacement grid ────────────────────────────────────────────
    print(f'\n=== PHASE 0: Displacement grid computation '
          f'({GRID_STEPS}×{GRID_STEPS} = {GRID_STEPS**2} positions) ===')

    all_grid_data = {}
    for stim_key in available:
        print(f'\n[{stim_key}]')
        image_gray = load_image_gray(stim_key)
        all_grid_data[stim_key] = load_or_compute_grid(
            stim_key, image_gray, model, readout,
            force_recompute=force_recompute,
        )

    if phase0_only:
        print('\nPhase 0 complete. Exiting (--phase0-only).')
        return {'phase0': all_grid_data}

    # ── Phase 1: Within-image decoding ────────────────────────────────────────
    print('\n=== PHASE 1: Within-image displacement decoding ===')
    phase1 = {}
    for stim_key in available:
        print(f'\n[{stim_key}]')
        phase1[stim_key] = decode_displacement(all_grid_data[stim_key])

    print('\nPhase 1 summary (mean across stimuli):')
    for fn in ['feat_scalar', 'feat_com', 'feat_width', 'feat_moments']:
        r2s  = [phase1[s][fn]['r2_mean']    for s in available]
        null = [phase1[s][fn]['null_r2_95'] for s in available]
        print(f'  {fn:<16}: R²={np.mean(r2s):.3f} ± {np.std(r2s):.3f}'
              f'  (null 95th: {np.mean(null):.3f})')

    plot_phase1(phase1)

    # ── Phase 1b: Local linearity sweep ──────────────────────────────────────
    print('\n=== PHASE 1b: Displacement magnitude sweep ===')
    # Run on first stimulus only for speed; others available if needed
    sweep_results = sweep_displacement_magnitude(all_grid_data[available[0]])
    print('  Sweep R² (feat_moments) by radius:')
    for mag, res in sorted(sweep_results.items()):
        r2 = res.get('feat_moments', {}).get('r2_mean', np.nan)
        nu = res.get('feat_moments', {}).get('null_r2', np.nan)
        print(f'    {mag*1000:.0f} mdeg: R²={r2:.3f}  null={nu:.3f}')
    plot_sweep(sweep_results)

    # ── Phase 2: Cross-image generalization ──────────────────────────────────
    if len(available) >= 3:
        print('\n=== PHASE 2: Cross-image generalization ===')
        phase2 = cross_image_displacement_decoding(all_grid_data)

        print('\nPhase 2 summary (mean R² across held-out images):')
        for fn in ['feat_scalar', 'feat_com', 'feat_moments']:
            r2s = [phase2[s][fn]['r2_mean'] for s in available]
            within = [phase1[s][fn]['r2_mean'] for s in available]
            frac = np.nanmean(r2s) / max(np.nanmean(within), 1e-6)
            print(f'  {fn:<16}: cross R²={np.mean(r2s):.3f}, '
                  f'within R²={np.mean(within):.3f}, '
                  f'fraction={frac:.2f}')

        plot_phase2(phase1, phase2)
    else:
        phase2 = {}
        print('Skipping Phase 2 (need ≥3 stimuli).')

    # ── Phase 3: FEM vs static ────────────────────────────────────────────────
    phase3 = {}
    if not skip_phase3:
        print('\n=== PHASE 3: FEM vs static displacement decoding ===')
        for stim_key in available:
            print(f'\n[{stim_key}]')
            entry = fixation_results[stim_key]
            ep_flat  = entry['eyepos'].astype(np.float32)
            n_trials = entry['n_trials']
            ep_3d    = split_eye_traces(ep_flat, n_trials)
            image_gray = load_image_gray(stim_key)

            phase3[stim_key] = fem_vs_static_displacement(
                stim_key, all_grid_data[stim_key],
                model, readout, image_gray, ep_3d,
                force_recompute=force_recompute,
            )
            r = phase3[stim_key]
            print(f'  scalar:  static={r["static_scalar"]:.3f}  FEM={r["fem_scalar"]:.3f}'
                  f'  null={r["null_scalar"]:.3f}')
            print(f'  moments: static={r["static_moments"]:.3f}  FEM={r["fem_moments"]:.3f}'
                  f'  null={r["null_moments"]:.3f}')

        plot_phase3(phase3)
        print('\nPhase 3 summary — velocity decoding Δr→Δv (mean across stimuli):')
        print('  (Both static and FEM use differential framing: frame-to-frame Δr → velocity)')
        for tag in ['scalar', 'moments']:
            sv  = np.mean([phase3[k][f'static_{tag}'] for k in available])
            fv  = np.mean([phase3[k][f'fem_{tag}']    for k in available])
            nv  = np.mean([phase3[k][f'null_{tag}']   for k in available])
            diff = fv - sv
            direction = 'FEM > static' if diff > 0.05 else 'FEM ≈ static' if abs(diff) <= 0.05 else 'FEM < static'
            print(f'  {tag:<8}: static={sv:.3f}  FEM={fv:.3f}  null={nv:.3f} → {direction}')

    # ── Phase 4: Cross-neuron lags ────────────────────────────────────────────
    phase4 = {}
    if not skip_phase4 and phase3:
        print('\n=== PHASE 4: Cross-neuron lag analysis ===')
        # Need RF positions: use the readout weight centres
        # Approximate: use spatial moments at mean position from grid centre
        centre_idx = (GRID_STEPS * GRID_STEPS) // 2
        centre_moms = all_grid_data[available[0]]['moments'][centre_idx]  # (N, 5)
        rf_xy_pix = centre_moms[:, :2]  # (N, 2) — CoM at centre position

        # Load one stimulus's FEM data (already computed)
        first_stim = available[0]
        fem_path = fem_cache_path(first_stim)
        if os.path.exists(fem_path):
            d = np.load(fem_path)
            fem_data = {k: d[k] for k in d.files}
            phase4 = compute_pairwise_lags(fem_data, rf_xy_pix)
            print(f'  Pearson r(τ_obs, τ_pred) = {phase4["pearson_r"]:.3f} '
                  f'(p={phase4["pearson_p"]:.3g})')
            plot_phase4(phase4)
        else:
            print(f'  Phase 3 FEM data not found for {first_stim}, skipping Phase 4.')

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        'phase1':  phase1,
        'phase2':  phase2,
        'phase3':  phase3,
        'phase4':  phase4,
        'sweep':   sweep_results,
        'stimuli': available,
    }
    out_path = os.path.join(FIGURES_DIR, 'displacement_summary.npy')
    np.save(out_path, summary, allow_pickle=True)  # type: ignore[arg-type]
    print(f'\nSummary saved: {out_path}')

    # ── Decision table ────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('DECISION TABLE')
    print('  Note: Phases 1–2 test the steady-state displacement manifold.')
    print('  Phase 3 tests whether real FEM dynamics add velocity information')
    print('  beyond what the static spatial code already provides (Ahissar test).')
    print('=' * 70)
    for fn in ['feat_scalar', 'feat_moments']:
        vals = [phase1[s][fn]['r2_mean']    for s in available]
        null = [phase1[s][fn]['null_r2_95'] for s in available]
        r2  = np.mean(vals)
        n95 = np.mean(null)
        above_null = r2 > n95
        verdict = ('strong positive' if r2 > 0.3
                   else 'ambiguous' if r2 > 0.05
                   else 'negative')
        flag = '' if above_null else ' [≤ null!]'
        print(f'  Within-image {fn:<16}: R²={r2:.3f}  null95={n95:.3f} → {verdict}{flag}')

    if phase2:
        for fn in ['feat_scalar', 'feat_moments']:
            w = np.mean([phase1[s][fn]['r2_mean'] for s in available])
            c = np.mean([phase2[s][fn]['r2_mean'] for s in available])
            frac = c / max(w, 1e-6)
            verdict = ('strong positive' if frac > 0.5
                       else 'ambiguous' if frac > 0.2
                       else 'negative')
            print(f'  Cross-image  {fn:<16}: {frac:.0%} of within → {verdict}')

    if phase3:
        print('  --- Phase 3: velocity decoding (Δr → Δv) ---')
        for tag in ['scalar', 'moments']:
            sv   = np.mean([phase3[k][f'static_{tag}'] for k in available])
            fv   = np.mean([phase3[k][f'fem_{tag}']    for k in available])
            nv   = np.mean([phase3[k][f'null_{tag}']   for k in available])
            diff = fv - sv
            verdict = ('FEM improves (>0.05)' if diff > 0.05
                       else 'FEM degrades (<-0.05)' if diff < -0.05
                       else 'FEM ≈ static')
            above = 'static>null' if sv > nv else 'static≤null'
            print(f'  {tag:<8}: static={sv:.3f}  FEM={fv:.3f}  null={nv:.3f} '
                  f'  ΔFEM={diff:+.3f} → {verdict}  ({above})')

    if phase4:
        r = phase4.get('pearson_r', np.nan)
        verdict = ('strong' if abs(r) > 0.3 else 'weak' if abs(r) > 0.1 else 'absent')
        print(f'  Cross-neuron lag structure: r={r:.3f} → {verdict} (exploratory)')
    print('=' * 70)

    return summary


# %% ── Entry point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stim',          nargs='+', default=None)
    parser.add_argument('--phase0-only',   action='store_true')
    parser.add_argument('--skip-phase3',   action='store_true')
    parser.add_argument('--skip-phase4',   action='store_true')
    parser.add_argument('--recompute',     action='store_true')
    args = parser.parse_args()

    main(
        stim_subset=args.stim,
        phase0_only=args.phase0_only,
        skip_phase3=args.skip_phase3,
        skip_phase4=args.skip_phase4,
        force_recompute=args.recompute,
    )
