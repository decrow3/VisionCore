"""
Spatial Map Dynamics: CoM-Based Transformation Analysis
========================================================

**ARCHITECTURAL CAVEAT — READ BEFORE INTERPRETING RESULTS**

This analysis was designed to test whether inter-frame changes in the spatial
population map (ΔCoM) track FEM eye velocity. It was motivated by a result
(C ≈ A at −0.20 dB) that turned out to be the wrong operating point; the
valid crossover is at −0.35 dB, where spatial sampling rather than temporal
integration drives the FEM benefit.

More fundamentally: the model processes independent 32-frame windows with a
*fresh* GRU state at the start of every trial. There is no recurrent state
propagating from one stimulus window to the next. ΔCoM(t) between consecutive
output frames is therefore the difference between two nearly-independent spatial
computations, not a signal from a continuous dynamical system.

Consequence for interpretation
-------------------------------
- Phase 0 (static-shift sanity check): **valid** — tests a purely spatial
  property (does the model's map shift when you translate the image?).
- Phase 2 (does ΔCoM track velocity during FEM?): **architecturally confounded**.
  Negative R² values indicate the absence of inter-frame temporal continuity, not
  the absence of velocity encoding in V1. A null here is trivially expected and
  is uninformative about biology.
- Phase 3 (content-transformation dissociation): **depends on Phase 2** and
  inherits the same confound.

If you run this analysis, report Phase 2/3 results as: "tested in a model with
independent temporal windows; nulls likely reflect architectural constraints
rather than the absence of biological temporal coding."

------------------------------------------------------------------------

Tests whether the *rate of change of spatial map position* (ΔCoM) tracks
eye velocity, using the full (N, 51, 51) spatial rate maps that were discarded
in the scalar-collapse analyses.

Architecture confirmed: readout output is (T, N=756, H=51, W=51).

Pipeline
--------
Phase 0  – Static-shift sanity check: is CoM monotonic over the FEM range?
Phase 1  – Compute moment trajectories (on-the-fly, discard raw maps)
Phase 2  – Decode velocity from CoM, ΔCoM, width, Δwidth; controls
Phase 3  – Content-transformation dissociation (only if Phase 2 passes)

Usage
-----
python declan/com_dynamics.py                     # full run
python declan/com_dynamics.py --phase0-only       # sanity check only
python declan/com_dynamics.py --stim Hawaii_trees.JPG  # one stimulus
python declan/com_dynamics.py --recompute         # ignore cache
"""

# %% ── Imports ────────────────────────────────────────────────────────────────

import os, sys, pickle, argparse, time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MplFigure
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def _fmt(seconds: float) -> str:
    """Format elapsed seconds as e.g. '1h 23m 45s' or '2m 05s' or '45s'."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f'{h}h {m:02d}m {s:02d}s'
    if m:
        return f'{m}m {s:02d}s'
    return f'{s}s'

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, '/home/declan/DataYatesV1')

import DataYatesV1  # must be importable before spatial_info
from spatial_info import get_spatial_readout, make_counterfactual_stim
from utils import get_model_and_dataset_configs

# %% ── Config ─────────────────────────────────────────────────────────────────

CACHE_DIR    = os.path.join(ROOT, 'declan', 'transformation_dynamics_cache')
COV_PKL      = os.path.join(ROOT, 'declan', 'translation_covariance', 'all_cov_results.pkl')
FIXATION_PKL = os.path.join(ROOT, 'declan', 'backimage_fixation_results.pkl')
FIGURES_DIR  = os.path.join(ROOT, 'declan', 'transformation_dynamics_figures', 'com')

N_LAGS       = 32
OUT_SIZE     = (151, 151)   # stimulus output size (pixels)
PPD          = 37.5         # pixels per degree
N_SPLITS_CV  = 5
RIDGE_ALPHAS = np.logspace(-2, 4, 13)

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


# %% ── Spatial moment computation ────────────────────────────────────────────

def compute_com(rate_map: np.ndarray):
    """
    Center of mass of a 2D rate map.

    Args:
        rate_map: (H, W) non-negative array

    Returns:
        (com_x, com_y) in pixel coordinates, or (nan, nan) if empty
    """
    total = float(rate_map.sum())
    if total < 1e-10:
        return np.nan, np.nan
    H, W = rate_map.shape
    yy, xx = np.mgrid[0:H, 0:W]
    com_x = float((xx * rate_map).sum() / total)
    com_y = float((yy * rate_map).sum() / total)
    return com_x, com_y


def compute_spatial_moments(rate_map: np.ndarray):
    """
    First and second spatial moments of a 2D rate map.

    Returns:
        (com_x, com_y, sigma_x, sigma_y)
    """
    total = float(rate_map.sum())
    if total < 1e-10:
        return np.nan, np.nan, np.nan, np.nan
    H, W = rate_map.shape
    yy, xx = np.mgrid[0:H, 0:W]
    com_x = float((xx * rate_map).sum() / total)
    com_y = float((yy * rate_map).sum() / total)
    sigma_x = float(np.sqrt(((xx - com_x) ** 2 * rate_map).sum() / total))
    sigma_y = float(np.sqrt(((yy - com_y) ** 2 * rate_map).sum() / total))
    return com_x, com_y, sigma_x, sigma_y


def moments_from_maps_batch(rate_maps: torch.Tensor) -> np.ndarray:
    """
    Compute spatial moments for a batch of maps, in-place on GPU then return CPU array.

    Args:
        rate_maps: (N, H, W) float32 tensor (non-negative)

    Returns:
        moments: (N, 5) float32 — [com_x, com_y, sigma_x, sigma_y, sigma_xy] per neuron
    """
    return moments_from_maps_4d(rate_maps.unsqueeze(0))[0]


def moments_from_maps_4d(rate_maps: torch.Tensor) -> np.ndarray:
    """
    Compute spatial moments for a (B, N, H, W) batch without Python-level loops.

    Eliminates the per-sample Python loop in run_trial_moments by operating on the
    full batch dimension at once.

    Args:
        rate_maps: (B, N, H, W) float32 tensor (non-negative)

    Returns:
        moments: (B, N, 5) float32 — [com_x, com_y, sigma_x, sigma_y, sigma_xy]
    """
    rate_maps = rate_maps.float()
    B, N, H, W = rate_maps.shape
    device = rate_maps.device

    # Coordinate grids broadcast over (B, N, H, W)
    yy = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)

    total = rate_maps.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-10)  # (B, N, 1, 1)

    com_x = (xx * rate_maps).sum(dim=(-2, -1)) / total.squeeze(-1).squeeze(-1)  # (B, N)
    com_y = (yy * rate_maps).sum(dim=(-2, -1)) / total.squeeze(-1).squeeze(-1)

    dx = xx - com_x.view(B, N, 1, 1)
    dy = yy - com_y.view(B, N, 1, 1)
    t   = total.squeeze(-1).squeeze(-1)
    sigma_x  = torch.sqrt(((dx ** 2) * rate_maps).sum(dim=(-2, -1)) / t)
    sigma_y  = torch.sqrt(((dy ** 2) * rate_maps).sum(dim=(-2, -1)) / t)
    sigma_xy = ((dx * dy)  * rate_maps).sum(dim=(-2, -1)) / t

    moments = torch.stack([com_x, com_y, sigma_x, sigma_y, sigma_xy], dim=2)  # (B, N, 5)
    return moments.cpu().numpy().astype(np.float32)


# %% ── Phase 0: Static-shift sanity check ────────────────────────────────────

def phase0_static_shift(
    model,
    readout,
    image_gray: np.ndarray,
    shift_range_deg: float = 0.1,
    n_steps: int = 20,
    n_lags: int = N_LAGS,
    out_size: tuple = OUT_SIZE,
) -> dict:
    """
    Phase 0: verify that CoM responds reliably to sub-degree image translations.

    Generates a grid of eye positions spanning ±shift_range_deg in x and y,
    computes spatial moments at each, then checks:
      - local slope of CoM vs. shift at origin
      - sign consistency across neurons

    Returns:
        dict with 'pass', 'com_vs_shift', 'local_slope_x', 'local_slope_y',
                  'sign_consistency_x', 'sign_consistency_y',
                  'sigma_vs_shift'
    """
    device = get_device()
    shifts = np.linspace(-shift_range_deg, shift_range_deg, n_steps)
    T_warmup = n_lags + 10  # enough frames to warm up the GRU

    # Compute moment at each x-shift (y=0)
    com_x_vs_shift_x = []
    com_y_vs_shift_x = []
    sigma_x_vs_shift_x = []

    t0 = time.time()
    n_total = 2 * n_steps
    print(f'Phase 0: sweeping {n_steps} x-shifts + {n_steps} y-shifts ({n_total} forward passes)...')

    for i, sx in enumerate(shifts):
        eyepos = np.tile([sx, 0.0], (T_warmup, 1)).astype(np.float32)
        full_stack = np.repeat(image_gray[np.newaxis], T_warmup + n_lags, axis=0)
        eyepos_t = torch.from_numpy(eyepos).to(device)
        stim = make_counterfactual_stim(full_stack, eyepos_t,
                                        out_size=out_size, n_lags=n_lags)
        stim_norm = (stim - 127.0) / 255.0
        stim_norm = stim_norm.to(device)

        with torch.no_grad():
            x = model.model.core_forward(stim_norm, None)
            x_last = x[-1:, :, -1]             # last frame only: (1, C, H, W)
            y = readout(x_last)                 # (1, N, H_out, W_out)
            y_act = model.model.activation(y)   # (1, N, H_out, W_out)
            maps = y_act[0]                     # (N, H_out, W_out)

        moments = moments_from_maps_batch(maps)  # (N, 5)
        com_x_vs_shift_x.append(moments[:, 0])
        com_y_vs_shift_x.append(moments[:, 1])
        sigma_x_vs_shift_x.append(moments[:, 2])

        done = i + 1
        elapsed = time.time() - t0
        eta = elapsed / done * (n_total - done)
        print(f'  x-sweep {done}/{n_steps}  [{_fmt(elapsed)} elapsed, ~{_fmt(eta)} remaining]',
              end='\r', flush=True)

    print()
    com_x_arr   = np.stack(com_x_vs_shift_x,  axis=0)  # (n_steps, N)
    sigma_x_arr = np.stack(sigma_x_vs_shift_x, axis=0)

    # Y-sweep
    com_x_vs_shift_y = []
    com_y_vs_shift_y = []
    for i, sy in enumerate(shifts):
        eyepos = np.tile([0.0, sy], (T_warmup, 1)).astype(np.float32)
        full_stack = np.repeat(image_gray[np.newaxis], T_warmup + n_lags, axis=0)
        eyepos_t = torch.from_numpy(eyepos).to(device)
        stim = make_counterfactual_stim(full_stack, eyepos_t,
                                        out_size=out_size, n_lags=n_lags)
        stim_norm = ((stim - 127.0) / 255.0).to(device)

        with torch.no_grad():
            x = model.model.core_forward(stim_norm, None)
            x_last = x[-1:, :, -1]
            y = readout(x_last)
            y_act = model.model.activation(y)
            maps = y_act[0]

        moments = moments_from_maps_batch(maps)
        com_x_vs_shift_y.append(moments[:, 0])
        com_y_vs_shift_y.append(moments[:, 1])

        done = n_steps + i + 1
        elapsed = time.time() - t0
        eta = elapsed / done * (n_total - done)
        print(f'  y-sweep {i+1}/{n_steps}  [{_fmt(elapsed)} elapsed, ~{_fmt(eta)} remaining]',
              end='\r', flush=True)

    print()

    com_y_shift_y = np.stack(com_y_vs_shift_y, axis=0)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    # Local slope at origin (central difference over central ±2 steps)
    mid = n_steps // 2
    dshift = shifts[mid + 1] - shifts[mid - 1]

    slope_com_x_by_shift_x = (com_x_arr[mid + 1] - com_x_arr[mid - 1]) / dshift  # (N,)
    slope_com_y_by_shift_y = (com_y_shift_y[mid + 1] - com_y_shift_y[mid - 1]) / dshift

    # Sign consistency: fraction of neurons with positive slope
    # (CoM_x should increase when eye shifts right → image shifts left → CoM shifts right in visual space)
    # Whether sign is positive or negative depends on convention; check consistency of sign
    sign_x = np.sign(slope_com_x_by_shift_x[np.isfinite(slope_com_x_by_shift_x)])
    sign_y = np.sign(slope_com_y_by_shift_y[np.isfinite(slope_com_y_by_shift_y)])
    sign_consistency_x = float(max(np.mean(sign_x > 0), np.mean(sign_x < 0)))
    sign_consistency_y = float(max(np.mean(sign_y > 0), np.mean(sign_y < 0)))

    # Local slope magnitude (median across neurons, ignoring NaN)
    valid_x = slope_com_x_by_shift_x[np.isfinite(slope_com_x_by_shift_x)]
    valid_y = slope_com_y_by_shift_y[np.isfinite(slope_com_y_by_shift_y)]
    median_slope_x = float(np.median(np.abs(valid_x))) if len(valid_x) > 0 else 0.0
    median_slope_y = float(np.median(np.abs(valid_y))) if len(valid_y) > 0 else 0.0

    # Sigma change (width change vs shift)
    sigma_slope_x = (sigma_x_arr[mid + 1] - sigma_x_arr[mid - 1]) / dshift
    sigma_changes = np.abs(sigma_slope_x[np.isfinite(sigma_slope_x)])
    sigma_responsive = float(np.mean(sigma_changes > 0.05))  # fraction with measurable width change

    # ── Decision ──────────────────────────────────────────────────────────────
    SLOPE_THRESHOLD = 0.1    # pixels / 0.01 degree shift → 10 px/degree
    CONSISTENCY_THRESHOLD = 0.5

    com_reliable = (sign_consistency_x > CONSISTENCY_THRESHOLD and
                    sign_consistency_y > CONSISTENCY_THRESHOLD and
                    median_slope_x > SLOPE_THRESHOLD and
                    median_slope_y > SLOPE_THRESHOLD)

    result = {
        'pass':                 com_reliable,
        'sign_consistency_x':   sign_consistency_x,
        'sign_consistency_y':   sign_consistency_y,
        'median_slope_x':       median_slope_x,
        'median_slope_y':       median_slope_y,
        'sigma_responsive_frac': sigma_responsive,
        'shifts':               shifts,
        'com_x_vs_shift_x':     com_x_arr,      # (n_steps, N)
        'com_y_vs_shift_y':     com_y_shift_y,  # (n_steps, N)
        'sigma_x_vs_shift_x':   sigma_x_arr,
    }

    print(f'\nPhase 0 results:')
    print(f'  CoM_x slope (median, px/deg):  {median_slope_x:.3f}')
    print(f'  CoM_y slope (median, px/deg):  {median_slope_y:.3f}')
    print(f'  Sign consistency x: {sign_consistency_x:.2%}')
    print(f'  Sign consistency y: {sign_consistency_y:.2%}')
    print(f'  Width (sigma) responsive frac: {sigma_responsive:.2%}')
    print(f'  Decision: {"PASS → proceed to Phase 1" if com_reliable else "FAIL → check thresholds / use width features"}')

    return result


def plot_phase0(result: dict, n_neurons: int = 20, save: bool = True) -> MplFigure:
    """Plot CoM vs. shift for a sample of neurons."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    shifts = result['shifts']
    N = result['com_x_vs_shift_x'].shape[1]
    idx = np.linspace(0, N - 1, min(n_neurons, N), dtype=int)

    ax = axes[0]
    for i in idx:
        com_vals = result['com_x_vs_shift_x'][:, i]
        if np.isfinite(com_vals).all():
            ax.plot(shifts, com_vals - com_vals[len(shifts)//2], alpha=0.4, lw=0.8)
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('Eye shift x (degrees)')
    ax.set_ylabel('ΔCoM_x (pixels, mean-centered)')
    ax.set_title(f'CoM_x vs x-shift\n(sign consistency={result["sign_consistency_x"]:.0%}, '
                 f'slope={result["median_slope_x"]:.2f} px/deg)')

    ax = axes[1]
    for i in idx:
        com_vals = result['com_y_vs_shift_y'][:, i]
        if np.isfinite(com_vals).all():
            ax.plot(shifts, com_vals - com_vals[len(shifts)//2], alpha=0.4, lw=0.8)
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('Eye shift y (degrees)')
    ax.set_ylabel('ΔCoM_y (pixels, mean-centered)')
    ax.set_title(f'CoM_y vs y-shift\n(sign consistency={result["sign_consistency_y"]:.0%}, '
                 f'slope={result["median_slope_y"]:.2f} px/deg)')

    plt.suptitle('Phase 0: Static CoM vs. Image Shift', fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, 'phase0_com_vs_shift.pdf')
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved: {path}')
    return fig


# %% ── Phase 1: Moment trajectory computation ─────────────────────────────────

def run_trial_moments(
    model,
    readout,
    image_gray: np.ndarray,
    eyepos: np.ndarray,           # (T, 2)
    null_eyepos: np.ndarray,      # (T, 2) — per-trial mean position
    n_lags: int = N_LAGS,
    out_size: tuple = OUT_SIZE,
    batch_size: int = 32,
    full_stack: np.ndarray = None,  # pre-allocated (T+n_lags, H, W) — pass from outer loop
) -> tuple:
    """
    Compute spatial moments for one trial.

    Runs the model on both the real and null eye traces, computing moments
    on-the-fly and discarding the (T, N, 51, 51) rate maps immediately.

    Returns:
        moments_real: (T, N, 5) — [com_x, com_y, sigma_x, sigma_y, sigma_xy] per neuron
        moments_null: (T, N, 5) — same for the null (mean-position) trace
    """
    device = get_device()
    T = eyepos.shape[0]

    # Pre-allocate image stack once if not provided by caller
    _stack = full_stack if full_stack is not None else \
        np.repeat(image_gray[np.newaxis], T + n_lags, axis=0)

    def _run_moments(ep: np.ndarray, label: str):
        ep_t = torch.from_numpy(ep).float().to(device)
        stim = make_counterfactual_stim(_stack, ep_t, out_size=out_size, n_lags=n_lags)
        stim_norm = (stim - 127.0) / 255.0

        all_moments = []
        n_batches = (T + batch_size - 1) // batch_size
        t_start_run = time.time()
        for bi, t_start in enumerate(range(0, T, batch_size)):
            t_end = min(t_start + batch_size, T)
            x_batch = stim_norm[t_start:t_end].to(device)
            with torch.no_grad():
                x = model.model.core_forward(x_batch, None)
                x_last = x[:, :, -1]               # (B, C, H_feat, W_feat)
                y = readout(x_last)                 # (B, N, 51, 51)
                y_act = model.model.activation(y)   # (B, N, 51, 51)

            # Compute moments for the whole batch at once — no Python loop over B
            batch_moms = moments_from_maps_4d(y_act)  # (B, N, 5)
            all_moments.append(batch_moms)

            del x_batch, x, x_last, y, y_act
            torch.cuda.empty_cache()

            elapsed = time.time() - t_start_run
            eta = elapsed / (bi + 1) * (n_batches - bi - 1)
            print(f'    {label} batch {bi+1}/{n_batches} '
                  f'(frames {t_start}–{t_end-1})  '
                  f'[{_fmt(elapsed)} elapsed, ~{_fmt(eta)} remaining]',
                  end='\r', flush=True)

        print()
        return np.concatenate(all_moments, axis=0)  # (T, N, 5)

    moments_real = _run_moments(eyepos,      'real')
    moments_null = _run_moments(null_eyepos, 'null')
    return moments_real, moments_null


def compute_moment_trajectories(
    stim_key: str,
    image_gray: np.ndarray,
    eyepos_3d: np.ndarray,     # (M, T, 2)
    model,
    readout,
    n_lags: int = N_LAGS,
    out_size: tuple = OUT_SIZE,
) -> dict:
    """
    Compute moment trajectories for all trials of one stimulus.

    Caches raw real and null moments separately so any delta combination
    (e.g. delta_com, delta_width, delta_sigma_xy) can be formed without
    re-running the forward pass.

    Returns:
        {
            'moments_real': (M, T, N, 5) — [com_x, com_y, sigma_x, sigma_y, sigma_xy]
            'moments_null': (M, T, N, 5) — same for null (mean-position) trace
            'eyepos':       (M, T, 2)
        }
    Note: delta_moments = moments_real - moments_null is computed on the fly in
    extract_moment_features, not stored, so any combination can be built later.
    """
    M, T, _ = eyepos_3d.shape

    # Pre-allocate image stack once — same image repeated for all trials
    full_stack = np.repeat(image_gray[np.newaxis], T + n_lags, axis=0)

    moments_real_list = []
    moments_null_list = []

    t0_stim = time.time()
    for i in range(M):
        ep = eyepos_3d[i]                                              # (T, 2)
        null_ep = np.tile(ep.mean(axis=0), (T, 1)).astype(np.float32) # per-trial mean

        t0_trial = time.time()
        print(f'\n  [{stim_key}] trial {i+1}/{M}')
        m_real, m_null = run_trial_moments(
            model, readout, image_gray, ep, null_ep,
            n_lags=n_lags, out_size=out_size, full_stack=full_stack,
        )
        moments_real_list.append(m_real)
        moments_null_list.append(m_null)

        elapsed_trial = time.time() - t0_trial
        elapsed_stim  = time.time() - t0_stim
        trials_left   = M - (i + 1)
        eta_stim      = elapsed_stim / (i + 1) * trials_left
        print(f'  [{stim_key}] trial {i+1}/{M} done  '
              f'(trial: {_fmt(elapsed_trial)}, stim elapsed: {_fmt(elapsed_stim)}, '
              f'stim ETA: ~{_fmt(eta_stim)})')

    print()
    return {
        'moments_real': np.stack(moments_real_list, axis=0),  # (M, T, N, 5)
        'moments_null': np.stack(moments_null_list, axis=0),  # (M, T, N, 5)
        'eyepos':       eyepos_3d,
    }


def cache_path_com(stim_key: str) -> str:
    safe = stim_key.replace('/', '_').replace('.', '_')
    return os.path.join(CACHE_DIR, f'{safe}_moments.npz')


def load_or_compute_moments(
    stim_key: str,
    image_gray,
    eyepos_3d: np.ndarray,
    model,
    readout,
    force_recompute: bool = False,
) -> dict:
    path = cache_path_com(stim_key)
    if os.path.exists(path) and not force_recompute:
        print(f'Loading cached moments for {stim_key}')
        d = np.load(path)
        return {k: d[k] for k in d.files}

    print(f'Computing moments for {stim_key} ({eyepos_3d.shape[0]} trials)...')
    data = compute_moment_trajectories(stim_key, image_gray, eyepos_3d, model, readout)
    np.savez_compressed(path, **data)
    print(f'  Saved to {path}')
    return data


# %% ── Phase 2: Feature extraction and velocity decoding ──────────────────────

def extract_moment_features(data: dict) -> dict:
    """
    Build all feature arrays from moment trajectories.

    Accepts either:
      - new format: data has 'moments_real' and 'moments_null' (M,T,N,5), delta computed here
      - legacy format: data has 'delta_moments' (M,T,N,4), used directly

    Moment columns:
      0: com_x   1: com_y   2: sigma_x   3: sigma_y   4: sigma_xy

    Returns dict with keys:
      'com'       : (M, T,   2N)  — Δcom_x, Δcom_y per neuron
      'dcom'      : (M, T-1, 2N)  — Δ(com_x, com_y) time derivative
      'width'     : (M, T,   2N)  — Δsigma_x, Δsigma_y
      'dwidth'    : (M, T-1, 2N)  — time derivative of width
      'cross'     : (M, T,   N)   — Δsigma_xy (cross-covariance)
      'dcross'    : (M, T-1, N)   — time derivative of cross-covariance
      'all_mom'   : (M, T,   5N)  — all five delta moments
      'dall_mom'  : (M, T-1, 5N)  — time derivative of all five
      'com_dcom'  : (M, T-1, 4N)  — [Δcom(t), Δ(Δcom)(t)]
      'mean_rate' : (M, 5N)        — trial-mean delta moments (content proxy)
      'vel'       : (M, T-1, 2)   — eye velocity
      'trace_ids' : (M,)
    """
    if 'moments_real' in data:
        # New format: compute delta on-the-fly
        dm = data['moments_real'] - data['moments_null']   # (M, T, N, 5)
    else:
        # Legacy format: delta_moments is (M, T, N, 4), no sigma_xy
        dm = data['delta_moments']  # (M, T, N, 4)

    ep = data['eyepos']   # (M, T, 2)
    M, T, N, K = dm.shape  # K=5 (new) or 4 (legacy)

    com   = dm[:, :, :, :2].reshape(M, T, 2 * N)    # (M, T, 2N)
    width = dm[:, :, :, 2:4].reshape(M, T, 2 * N)   # (M, T, 2N)
    all_m = dm.reshape(M, T, K * N)                  # (M, T, K*N)

    dcom   = np.diff(com,   axis=1)   # (M, T-1, 2N)
    dwidth = np.diff(width, axis=1)   # (M, T-1, 2N)
    dall_m = np.diff(all_m, axis=1)   # (M, T-1, K*N)

    vel = np.diff(ep, axis=1).astype(np.float32)  # (M, T-1, 2)

    out = {
        'com':       com[:, :-1, :],
        'dcom':      dcom,
        'width':     width[:, :-1, :],
        'dwidth':    dwidth,
        'all_mom':   all_m[:, :-1, :],
        'dall_mom':  dall_m,
        'com_dcom':  np.concatenate([com[:, :-1, :], dcom], axis=-1),
        'mean_rate': dm.mean(axis=1).reshape(M, K * N),
        'vel':       vel,
        'trace_ids': np.arange(M, dtype=int),
    }

    if K == 5:
        cross  = dm[:, :, :, 4:5].reshape(M, T, N)   # (M, T, N)
        dcross = np.diff(cross, axis=1)               # (M, T-1, N)
        out['cross']  = cross[:, :-1, :]
        out['dcross'] = dcross

    return out


def _flatten(feat, vel, trace_ids):
    """Flatten (M, T', d) → (M*T', d), expand trace_ids."""
    M, Tp, d = feat.shape
    return (feat.reshape(M * Tp, d),
            vel.reshape(M * Tp, 2),
            np.repeat(trace_ids, Tp))


def ridge_cv_by_trace(features, targets, trace_ids, n_splits=N_SPLITS_CV):
    """Grouped ridge CV (by trace). Returns r2_mean, r2_std, r2_folds."""
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    for tr, te in gkf.split(features, targets, groups=trace_ids):
        sc = StandardScaler()
        Xtr = sc.fit_transform(features[tr])
        Xte = sc.transform(features[te])
        rcv = RidgeCV(alphas=RIDGE_ALPHAS)
        rcv.fit(Xtr, targets[tr])
        pred = rcv.predict(Xte)
        ss_res = ((targets[te] - pred) ** 2).sum(0)
        ss_tot = ((targets[te] - targets[te].mean(0)) ** 2).sum(0)
        folds.append(float((1 - ss_res / (ss_tot + 1e-12)).mean()))
    folds = np.array(folds)
    return {'r2_mean': float(folds.mean()), 'r2_std': float(folds.std()),
            'r2_folds': folds}


# %% ── Phase 2: Core decoding + controls ──────────────────────────────────────

def run_phase2_decoding(all_stim_data: dict) -> dict:
    """
    For each stimulus, decode velocity from each feature set.
    Also run time-shuffle and neuron-shuffle controls.

    Returns:
        stim_key → feature_name → r2_result
    """
    base_feature_names = ['com', 'dcom', 'width', 'dwidth', 'all_mom', 'dall_mom', 'com_dcom']
    results = {}

    for stim_key in sorted(all_stim_data.keys()):
        print(f'\n=== {stim_key} ===')
        feats = extract_moment_features(all_stim_data[stim_key])
        vel   = feats['vel']
        tids  = feats['trace_ids']
        stim_res = {}

        # Include cross-term features when available (new 5-moment format)
        feature_names = list(base_feature_names)
        if 'cross' in feats:
            feature_names += ['cross', 'dcross']

        for fn in feature_names:
            ff, vf, tf = _flatten(feats[fn], vel, tids)
            stim_res[fn] = ridge_cv_by_trace(ff, vf, tf)
            print(f'  {fn:<12}: R²={stim_res[fn]["r2_mean"]:.3f} ± {stim_res[fn]["r2_std"]:.3f}')

        # Time-shuffle control on dcom
        rng = np.random.default_rng(0)
        dcom_shuf = feats['dcom'].copy()
        for i in range(dcom_shuf.shape[0]):
            dcom_shuf[i] = dcom_shuf[i, rng.permutation(dcom_shuf.shape[1]), :]
        ff_s, vf_s, tf_s = _flatten(dcom_shuf, vel, tids)
        stim_res['dcom_timeshuf'] = ridge_cv_by_trace(ff_s, vf_s, tf_s)
        print(f'  {"dcom_timeshuf":<12}: R²={stim_res["dcom_timeshuf"]["r2_mean"]:.3f}')

        # Neuron-shuffle control: shuffle neuron identities at each time step
        M, Tp, dim = feats['dcom'].shape
        N = dim // 2
        dcom_nshuffle = feats['dcom'].reshape(M, Tp, N, 2).copy()
        for t in range(Tp):
            perm = rng.permutation(N)
            dcom_nshuffle[:, t, :, :] = dcom_nshuffle[:, t, perm, :]
        dcom_nshuffle = dcom_nshuffle.reshape(M, Tp, dim)
        ff_n, vf_n, tf_n = _flatten(dcom_nshuffle, vel, tids)
        stim_res['dcom_neuronshuffle'] = ridge_cv_by_trace(ff_n, vf_n, tf_n)
        print(f'  {"dcom_neurshuf":<12}: R²={stim_res["dcom_neuronshuffle"]["r2_mean"]:.3f}')

        results[stim_key] = stim_res

    return results


# %% ── Phase 2: Cross-stimulus generalization ─────────────────────────────────

def run_cross_stimulus(all_stim_data: dict) -> dict:
    """
    Leave-one-stimulus-out: train ΔCoM → velocity on 5, test on held-out 6th.

    CoM features are neuron-indexed (no PCA basis), so no basis-choice issue.

    Returns:
        stim_key → {'cross_r2': float, 'within_r2': float}
    """
    stim_keys = sorted(all_stim_data.keys())
    results = {}

    for held in stim_keys:
        train_keys = [k for k in stim_keys if k != held]

        # Pool training data
        feat_list, vel_list, tid_list = [], [], []
        offset = 0
        for k in train_keys:
            feats = extract_moment_features(all_stim_data[k])
            ff, vf, tf = _flatten(feats['dcom'], feats['vel'], feats['trace_ids'])
            feat_list.append(ff)
            vel_list.append(vf)
            tid_list.append(tf + offset)
            offset += feats['trace_ids'].max() + 1

        X_train = np.concatenate(feat_list, axis=0)
        y_train = np.concatenate(vel_list, axis=0)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        rcv = RidgeCV(alphas=RIDGE_ALPHAS)
        rcv.fit(X_tr_s, y_train)

        # Test on held-out
        feats_ho = extract_moment_features(all_stim_data[held])
        ff_ho, vf_ho, tf_ho = _flatten(feats_ho['dcom'], feats_ho['vel'],
                                        feats_ho['trace_ids'])
        X_te_s = scaler.transform(ff_ho)
        pred = rcv.predict(X_te_s)
        ss_res = ((vf_ho - pred) ** 2).sum(0)
        ss_tot = ((vf_ho - vf_ho.mean(0)) ** 2).sum(0)
        cross_r2 = float((1 - ss_res / (ss_tot + 1e-12)).mean())

        # Within-stimulus baseline via CV
        within_res = ridge_cv_by_trace(ff_ho, vf_ho, tf_ho)
        within_r2 = within_res['r2_mean']

        results[held] = {'cross_r2': cross_r2, 'within_r2': within_r2}
        print(f'  {held}: cross R²={cross_r2:.3f}, within R²={within_r2:.3f}')

    return results


# %% ── Phase 3: Content-transformation dissociation ───────────────────────────

def run_phase3_dissociation(all_stim_data: dict) -> dict:
    """
    2×2 dissociation matrix:
      Rows:    {mean rate (time-avg), ΔCoM (spatial dynamics)}
      Columns: {content (stimulus ID), transformation (velocity R²)}
    """
    stim_keys = sorted(all_stim_data.keys())
    K = len(stim_keys)
    results = {}

    print('\n--- Phase 3: Transformation task ---')
    for fn in ('mean_rate', 'dcom'):
        r2s = []
        for stim_key in stim_keys:
            feats = extract_moment_features(all_stim_data[stim_key])
            vel  = feats['vel']
            tids = feats['trace_ids']
            if fn == 'mean_rate':
                # Tile static feature across time
                M, Tp, _ = vel.shape
                feat = np.repeat(feats['mean_rate'][:, np.newaxis, :], Tp, axis=1)
            else:
                feat = feats['dcom']
            ff, vf, tf = _flatten(feat, vel, tids)
            res = ridge_cv_by_trace(ff, vf, tf)
            r2s.append(res['r2_mean'])
            print(f'  {fn} | {stim_key}: R²={res["r2_mean"]:.3f}')
        results[f'{fn}_transform_r2'] = float(np.mean(r2s))

    print('\n--- Phase 3: Content task ---')
    for fn in ('mean_rate', 'dcom'):
        X_list, y_list, g_list = [], [], []
        for label_idx, stim_key in enumerate(stim_keys):
            feats = extract_moment_features(all_stim_data[stim_key])
            M = all_stim_data[stim_key]['eyepos'].shape[0]
            if fn == 'mean_rate':
                X = feats['mean_rate']           # (M, 4N)
            else:
                X = feats['dcom'].mean(axis=1)   # (M, 2N) — trial-mean ΔCoM
            X_list.append(X)
            y_list.append(np.full(M, label_idx, dtype=int))
            g_list.append(np.arange(M, dtype=int))

        X_all = np.concatenate(X_list); y_all = np.concatenate(y_list)
        g_all = np.concatenate(g_list)
        M_min = min(len(g) for g in g_list)
        idx = np.concatenate([np.where(y_all == k)[0][:M_min] for k in range(K)])
        X_b = X_all[idx]; y_b = y_all[idx]; g_b = g_all[idx % M_min]

        gkf = GroupKFold(n_splits=N_SPLITS_CV)
        accs = []
        for tr, te in gkf.split(X_b, y_b, groups=g_b):
            sc = StandardScaler()
            clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=42)
            clf.fit(sc.fit_transform(X_b[tr]), y_b[tr])
            accs.append(clf.score(sc.transform(X_b[te]), y_b[te]))
        acc = float(np.mean(accs))
        results[f'{fn}_content_acc'] = acc
        print(f'  {fn} content accuracy: {acc:.3f} (chance={1/K:.3f})')

    return results


# %% ── Figures ─────────────────────────────────────────────────────────────────

def plot_phase2_summary(phase2: dict, save: bool = True) -> MplFigure:
    """Bar plot: R² per feature set, averaged across stimuli."""
    stims = sorted(phase2.keys())
    candidates = [
        ('com',                'CoM',             '#4C72B0'),
        ('dcom',               'ΔCoM',            '#DD8452'),
        ('width',              'Width',            '#55A868'),
        ('dwidth',             'ΔWidth',           '#C44E52'),
        ('cross',              'σ_xy',             '#8172B2'),
        ('dcross',             'Δσ_xy',            '#937860'),
        ('com_dcom',           '[CoM,ΔCoM]',       '#64B5CD'),
        ('dcom_timeshuf',      'ΔCoM\n(t-shuf)',   '#999999'),
        ('dcom_neuronshuffle', 'ΔCoM\n(n-shuf)',   '#BBBBBB'),
    ]
    # Only include features present in the results
    present = [(fn, lb, co) for fn, lb, co in candidates
               if fn in phase2[stims[0]]]
    fns     = [x[0] for x in present]
    labels  = [x[1] for x in present]
    colours = [x[2] for x in present]

    means = [np.mean([phase2[s][fn]['r2_mean'] for s in stims]) for fn in fns]
    stds  = [np.std( [phase2[s][fn]['r2_mean'] for s in stims]) for fn in fns]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(fns))
    ax.bar(x, means, yerr=stds, color=colours, width=0.65, capsize=4, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('R² (velocity decoding)')
    ax.set_title('Phase 2: Spatial moment features — velocity decoding\n(mean ± std across stimuli)')

    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, 'phase2_feature_comparison.pdf')
        fig.savefig(path, bbox_inches='tight'); print(f'Saved: {path}')
    return fig


def plot_cross_stim(cross: dict, save: bool = True) -> MplFigure:
    stims = sorted(cross.keys())
    within = [cross[s]['within_r2'] for s in stims]
    cross_ = [cross[s]['cross_r2']  for s in stims]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(stims))
    w = 0.35
    ax.bar(x - w/2, within, w, label='Within-stimulus', color='#4C72B0', alpha=0.8)
    ax.bar(x + w/2, cross_,  w, label='Cross-stimulus',  color='#DD8452', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.split('.')[0] for s in stims], rotation=20, ha='right', fontsize=8)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('R² (ΔCoM → velocity)')
    ax.set_title('Phase 2: Cross-stimulus generalisation of ΔCoM readout')
    ax.legend()
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, 'phase2_cross_stimulus.pdf')
        fig.savefig(path, bbox_inches='tight'); print(f'Saved: {path}')
    return fig


def plot_dissociation(phase3: dict, n_stim: int, save: bool = True) -> MplFigure:
    data = np.array([
        [phase3['mean_rate_content_acc'],  phase3['mean_rate_transform_r2']],
        [phase3['dcom_content_acc'],        phase3['dcom_transform_r2']],
    ])
    vmax = max(data.max(), 0.6)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=vmax, aspect='auto')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Content\n(accuracy)', 'Transformation\n(R²)'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Mean rate\n(static)', 'ΔCoM\n(dynamic)'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{data[i,j]:.3f}', ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color='white' if data[i, j] > vmax / 2 else 'black')
    ax.set_title(f'Phase 3: Content–transformation dissociation\n(chance={1/n_stim:.2f})')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, 'phase3_dissociation.pdf')
        fig.savefig(path, bbox_inches='tight'); print(f'Saved: {path}')
    return fig


# %% ── Helper: image loading and trace splitting ──────────────────────────────

def load_image_gray(stim_key: str) -> np.ndarray:
    search = [
        '/home/declan/DataYatesV1/DataYatesV1/exp/SupportData/Backgrounds',
        os.path.join(ROOT, 'declan'),
        os.path.join(ROOT, 'data'),
    ]
    for base in search:
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
    raise FileNotFoundError(f'Could not find {stim_key}')


def split_eye_traces(eyepos_flat: np.ndarray, n_trials: int) -> np.ndarray:
    T = eyepos_flat.shape[0] // n_trials
    return eyepos_flat[:n_trials * T].reshape(n_trials, T, 2).astype(np.float32)


# %% ── Main ───────────────────────────────────────────────────────────────────

def main(
    stim_subset:      list | None = None,
    phase0_only:      bool  = False,
    force_recompute:  bool  = False,
    skip_cross_stim:  bool  = False,
    skip_phase3:      bool  = False,
):
    # ── Load metadata ─────────────────────────────────────────────────────────
    with open(COV_PKL, 'rb') as f:
        cov_results = pickle.load(f)
    with open(FIXATION_PKL, 'rb') as f:
        fixation_results = pickle.load(f)

    available = sorted(set(cov_results.keys()) & set(fixation_results.keys()))
    if stim_subset:
        available = [s for s in available if s in stim_subset]
    print(f'Stimuli: {available}')

    # ── Load model ────────────────────────────────────────────────────────────
    print('\nLoading model...')
    model, readout = load_model_and_readout()

    # ── Phase 0: sanity check on first stimulus ───────────────────────────────
    stim0 = available[0]
    print(f'\n=== PHASE 0: Static-shift sanity check ({stim0}) ===')
    try:
        image0 = load_image_gray(stim0)
    except FileNotFoundError as e:
        print(f'Could not load image: {e}\nSkipping Phase 0.')
        image0 = None

    p0_result = None
    if image0 is not None:
        p0_result = phase0_static_shift(model, readout, image0)
        plot_phase0(p0_result)
        p0_path = os.path.join(FIGURES_DIR, 'phase0_data.pkl')
        with open(p0_path, 'wb') as _f:
            pickle.dump(p0_result, _f)
        print(f'Saved Phase 0 data: {p0_path}')

        if not p0_result['pass']:
            print('\nPhase 0 FAILED. Check thresholds or consider width-only features.')
            if phase0_only:
                return {'phase0': p0_result}
            print('Continuing with Phase 1 anyway (check plots manually).')

    if phase0_only:
        return {'phase0': p0_result}

    # ── Phase 1: compute moment trajectories ──────────────────────────────────
    print('\n=== PHASE 1: Moment trajectory computation ===')
    all_stim_data = {}
    for stim_key in available:
        entry = fixation_results[stim_key]
        ep_flat  = entry['eyepos'].astype(np.float32)
        n_trials = entry['n_trials']
        ep_3d    = split_eye_traces(ep_flat, n_trials)

        path = cache_path_com(stim_key)
        if not os.path.exists(path) or force_recompute:
            image_gray = load_image_gray(stim_key)
            data = load_or_compute_moments(
                stim_key, image_gray, ep_3d, model, readout,
                force_recompute=force_recompute,
            )
        else:
            data = load_or_compute_moments(
                stim_key, None, ep_3d, None, None, force_recompute=False
            )
        all_stim_data[stim_key] = data

    # ── Phase 2: velocity decoding ────────────────────────────────────────────
    print('\n=== PHASE 2: Velocity decoding ===')
    phase2 = run_phase2_decoding(all_stim_data)

    print('\nSummary (mean across stimuli):')
    for fn in ['com', 'dcom', 'width', 'dwidth', 'cross', 'dcross', 'com_dcom',
               'dcom_timeshuf', 'dcom_neuronshuffle']:
        r2s = [phase2[s].get(fn, {}).get('r2_mean', np.nan) for s in available]
        if not np.all(np.isnan(r2s)):
            print(f'  {fn:<18}: R²={np.nanmean(r2s):.3f} ± {np.nanstd(r2s):.3f}')

    plot_phase2_summary(phase2)

    # Decision point
    dcom_r2 = np.mean([phase2[s]['dcom']['r2_mean'] for s in available])
    print(f'\nDecision: ΔCoM R²={dcom_r2:.3f}')
    if dcom_r2 <= 0.02:
        print('ΔCoM R²≈0: spatial map dynamics do not encode velocity.')
        print('Reporting as clean negative result.')

    # Cross-stimulus generalisation
    cross_results = {}
    if not skip_cross_stim and len(available) >= 3:
        print('\n=== PHASE 2: Cross-stimulus generalisation ===')
        cross_results = run_cross_stimulus(all_stim_data)
        plot_cross_stim(cross_results)

    # ── Phase 3: dissociation ─────────────────────────────────────────────────
    phase3 = {}
    if not skip_phase3:
        print('\n=== PHASE 3: Content-transformation dissociation ===')
        phase3 = run_phase3_dissociation(all_stim_data)
        plot_dissociation(phase3, n_stim=len(available))

    # ── Save ──────────────────────────────────────────────────────────────────
    summary = {'phase0': p0_result, 'phase2': phase2,
               'cross': cross_results, 'phase3': phase3}
    out_path = os.path.join(FIGURES_DIR, 'com_summary.pkl')
    with open(out_path, 'wb') as _f:
        pickle.dump(summary, _f)
    print(f'\nSummary saved: {out_path}')
    return summary


# %% ── Entry point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stim', nargs='+', default=None)
    parser.add_argument('--phase0-only', action='store_true')
    parser.add_argument('--recompute',   action='store_true')
    parser.add_argument('--no-cross',    action='store_true')
    parser.add_argument('--no-phase3',   action='store_true')
    args = parser.parse_args()

    main(
        stim_subset=args.stim,
        phase0_only=args.phase0_only,
        force_recompute=args.recompute,
        skip_cross_stim=args.no_cross,
        skip_phase3=args.no_phase3,
    )
