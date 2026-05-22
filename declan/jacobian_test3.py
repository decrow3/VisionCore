"""
Test 3: Analytic Translation Jacobian for E-optotype via forward-AD
Test 6: Representational intervention via U_jac

Computes J = [∂λ/∂x, ∂λ/∂y] at the mean real-FEM eye trace as operating point,
then compares the Jacobian subspace U_jac to the empirical C_FEM subspace (U_pca2)
from cached rates.  Also runs the Test 6 representational intervention: project
out / project onto U_jac per orientation and report Δacc at both LogMARs.

Usage
-----
    python declan/jacobian_test3.py
    python declan/jacobian_test3.py --logmars -0.20,-0.40 --device cuda
    python declan/jacobian_test3.py --n_null_reps 200   # faster null controls

Output (declan/jacobian_results/)
-----
    test3_lm{logmar:.2f}.npz   – J, U_jac, alignment, capture, null distributions
    test3_summary.txt          – human-readable table
    test3_test6_plots.pdf      – figures
"""

from __future__ import annotations

import argparse
import os
import sys
import dill
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# ── Path setup ─────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
SCRIPTS_DIR = REPO_DIR / 'scripts'
TD_DIR = SCRIPTS_DIR / 'temporal_decoding'
RATES_DIR = TD_DIR / 'data' / 'rates'
DATA_DIR = TD_DIR / 'data'
OUT_DIR = SCRIPT_DIR / 'jacobian_results'

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(TD_DIR))

from spatial_info import get_spatial_readout, embed_time_lags  # noqa: E402
from stimulus_hires import HiResERenderer, HiResRetina          # noqa: E402
from utils import get_model_and_dataset_configs                  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────

ORIENTATIONS = [0, 90, 180, 270]
PKL_PATH = SCRIPTS_DIR / 'mcfarland_outputs_mono.pkl'
N_LAGS = 32


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_eye_traces(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        traces:   (M, T_max, 2) float32
        durations: (M,)  int
    """
    d = np.load(str(path), allow_pickle=True)
    return d['traces'].astype(np.float32), d['durations'].astype(int)


def compute_mean_trace(traces: np.ndarray, durations: np.ndarray,
                       n_frames: int) -> np.ndarray:
    """
    Element-wise mean of real FEM traces up to n_frames.

    Includes only trials whose valid length >= n_frames so each time point
    is averaged over the same set of trials.

    Returns: (n_frames, 2) float32
    """
    mask = durations >= n_frames
    assert mask.sum() > 0, (
        f"No trials have length >= {n_frames}. "
        f"Max duration: {durations.max()}"
    )
    valid = traces[mask, :n_frames, :]   # (M_valid, n_frames, 2)
    # Averaging suppresses trial-to-trial variability → the resulting trajectory
    # is a smooth, low-velocity drift that is not physically realizable.  This
    # may bias J toward low-spatial-frequency sensitivity.  Interpret J accordingly;
    # if alignment is marginal, consider re-running with a single representative trace.
    return valid.mean(axis=0)            # (n_frames, 2)


def load_rates_by_ori(logmar: float, condition: str,
                      rates_dir: Path = RATES_DIR) -> dict:
    """
    Load cached rates for all 4 orientations.
    Returns dict  ori_key → (M, N) time-averaged float64 array.
    """
    prefix = 'rates_hires'
    result = {}
    for ori in ORIENTATIONS:
        fname = f'{prefix}_lm{logmar:.2f}_ori{ori}_{condition}.npz'
        path = rates_dir / fname
        if not path.exists():
            # try without hires prefix
            fname_lo = f'rates_lm{logmar:.2f}_ori{ori}_{condition}.npz'
            path = rates_dir / fname_lo
        if not path.exists():
            raise FileNotFoundError(
                f"Missing cached rates: {rates_dir / fname}\n"
                "Run cache_eoptotype_rates.py first."
            )
        d = np.load(str(path), allow_pickle=True)
        rates_pad = d['rates']          # (M, T_max, N)
        lengths = d['lengths'].astype(int)
        # Time-average each trial
        ravg = np.stack([rates_pad[i, :lengths[i]].mean(0)
                         for i in range(len(lengths))], axis=0)
        result[f'ori{ori}'] = ravg.astype(np.float64)
    n_neurons = next(iter(result.values())).shape[1]
    print(f"  Loaded lm={logmar:+.2f} {condition:10s} | "
          f"N={n_neurons} neurons | "
          f"trials/ori: {[v.shape[0] for v in result.values()]}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Forward-AD compatible embedding
# ═══════════════════════════════════════════════════════════════════════════════

def embed_time_lags_pure(movie: torch.Tensor, n_lags: int = 32) -> torch.Tensor:
    """
    Forward-AD compatible version of embed_time_lags.
    Uses torch.stack instead of in-place assignment so dual tangents propagate.

    Input:  movie (T, H, W)
    Output: (T - n_lags + 1, 1, n_lags, H, W)
    """
    T = movie.shape[0]
    out_frames = T - n_lags + 1
    assert out_frames >= 1, f"Need T >= n_lags, got T={T}, n_lags={n_lags}"
    lags = []
    for lag in range(n_lags):
        # lag 0 = current frame, lag 1 = 1 frame ago, …
        lags.append(movie[n_lags - 1 - lag: T - lag])   # (out_frames, H, W)
    lagged = torch.stack(lags, dim=1)                    # (out_frames, n_lags, H, W)
    return lagged.unsqueeze(1)                           # (out_frames, 1, n_lags, H, W)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Jacobian computation
# ═══════════════════════════════════════════════════════════════════════════════

def _rates_from_stim(model, readout, stim: torch.Tensor) -> torch.Tensor:
    """
    Run (B, 1, n_lags, H, W) stim through model → (B, N) scalar rates.

    Uses spatial mean (instead of max) so forward-AD tangents flow correctly.
    """
    feats = model.model.core_forward(stim, None)     # (B, C, T_f, H_f, W_f)
    y = readout(feats[:, :, -1])                     # (B, N, H_r, W_r)
    rates_map = model.model.activation(y)            # (B, N, H_r, W_r)
    return rates_map.mean(dim=(-2, -1))              # (B, N)


def compute_jacobian(
    model,
    readout,
    renderer: HiResERenderer,
    retina: HiResRetina,
    mean_trace: np.ndarray,
    orientation_deg: float,
    logmar: float,
    device: str,
    n_lags: int = N_LAGS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the translation Jacobian J ∈ ℝ^{N×2} at the mean FEM eye trace.

    The Jacobian is ∂λ/∂p where p = (x, y) is a uniform displacement applied
    to all n_lags frames of the mean trace (i.e., a global translation of the
    visual scene relative to the eye).

    Forward-AD is applied to the eye trace tensor; tangents propagate through
    HiResRetina → embed_time_lags_pure → model → spatial mean → rates.

    Args:
        mean_trace: (T, 2) float32 array; first n_lags frames are used.
        orientation_deg: E orientation in degrees.
        logmar: E size in LogMAR units.
        device: torch device string.
        n_lags: number of temporal history frames to use.

    Returns:
        J:      (N, 2) numpy array — columns are ∂λ/∂x and ∂λ/∂y
        U_jac:  (N, 2) numpy array — orthonormal basis via QR
    """
    dev = torch.device(device)

    # Stage 1: render world image (no differentiation needed)
    renderer.eval()
    retina.eval()
    model.model.eval()
    readout.eval()

    with torch.no_grad():
        world_img = renderer(orientation_deg, logmar).to(dev)  # (1,1,H_w,W_w) ∈ [0,1]
        world_gray = 127.0 * (1.0 - world_img)                # E=0, bg=127

    # Base eye trace: first n_lags frames of mean_trace
    base_trace = torch.tensor(mean_trace[:n_lags], dtype=torch.float32)  # (n_lags, 2)
    base_trace = base_trace.to(dev)

    # This computes the translation-invariant Jacobian: a uniform shift Δp is
    # applied to ALL n_lags frames simultaneously.  The resulting J captures the
    # subspace that the model is sensitive to when the ENTIRE position history
    # moves by (δx, δy).  It is NOT the full temporal Jacobian ∑_t ∂λ/∂p_t δp_t,
    # which would require separate tangent vectors per lag.  Discrepancies between
    # U_jac and U_pca2 may reflect temporal-weighting differences across lags
    # (View B / GRU mixing) rather than a failure of the Jacobian model.
    J_cols = []
    for axis in range(2):           # 0 = x, 1 = y
        tangent = torch.zeros_like(base_trace)
        tangent[:, axis] = 1.0      # uniform unit shift in this axis

        with torch.no_grad():
            with fwAD.dual_level():
                dual_trace = fwAD.make_dual(base_trace, tangent)

                # Stage 2: sample retinal movie with dual eye trace
                # HiResRetina.forward expects (T, 2)
                movie_dual = retina(world_gray, dual_trace)   # (1,1,n_lags,H_r,W_r)
                movie_dual = movie_dual[0, 0]                  # (n_lags, H_r, W_r)
                movie_dual = movie_dual / 127.0               # normalise as in pipeline

                # Stage 3: embed time lags → (1, 1, n_lags, H_r, W_r)
                stim_dual = embed_time_lags_pure(movie_dual, n_lags)

                # Stage 4: model forward pass
                rates_dual = _rates_from_stim(model, readout, stim_dual)  # (1, N)
                rates_dual = rates_dual[0]                                  # (N,)

                # Extract tangent = ∂λ/∂axis
                tangent_out = fwAD.unpack_dual(rates_dual).tangent         # (N,) or None
                if tangent_out is None:
                    raise RuntimeError(
                        "Forward-AD tangent is None — check that the pipeline "
                        "is fully differentiable (no .detach() or numpy calls)."
                    )
                J_cols.append(tangent_out.cpu().numpy())

    J = np.stack(J_cols, axis=1)     # (N, 2)
    U_jac, _ = np.linalg.qr(J)      # (N, 2), orthonormal basis of Jacobian column space
    return J, U_jac


# ═══════════════════════════════════════════════════════════════════════════════
# 3b. Effective temporal Jacobian (Test 4b)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_jacobian_eff(
    model,
    readout,
    renderer: HiResERenderer,
    retina: HiResRetina,
    traces: np.ndarray,
    durations: np.ndarray,
    orientation_deg: float,
    logmar: float,
    device: str,
    eps: float = 0.005,
    n_lags: int = N_LAGS,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Effective temporal Jacobian via finite differences over the empirical
    FEM trace distribution.

    For each valid trace the last n_lags frames are shifted by ±eps along
    each axis.  Rates are averaged over all M_valid traces before differencing:

        J_eff[:, axis] = E_trace[(λ(p+ε·eₐ) − λ(p−ε·eₐ)) / (2ε)]

    This is the expected local sensitivity under the actual FEM distribution
    (rather than the sensitivity at the single mean trace used by compute_jacobian).

    J_eff should be paired with Sigma_trial — the covariance of per-trace mean
    positions — not with per-frame Sigma_eye.  Together they give the first-order
    between-trial covariance prediction: C_pred = J_eff Σ_trial J_effᵀ.

    Args:
        traces:     (M, T_max, 2) float32 NaN-padded traces
        durations:  (M,) int  valid frame counts
        eps:        finite-difference step size in degrees (default 0.005)
        batch_size: traces per model forward pass

    Returns:
        J_eff: (N, 2)  — columns ∂⟨λ⟩/∂x and ∂⟨λ⟩/∂y averaged over traces
        U_eff: (N, 2)  — orthonormal basis via QR
    """
    dev = torch.device(device)
    renderer.eval()
    retina.eval()
    model.model.eval()
    readout.eval()

    with torch.no_grad():
        world_img = renderer(orientation_deg, logmar).to(dev)
        world_gray = 127.0 * (1.0 - world_img)

    # Use the last n_lags frames of each valid trace as the operating window
    valid_windows = []
    for i in range(len(durations)):
        T_i = int(durations[i])
        if T_i < n_lags:
            continue
        valid_windows.append(traces[i, T_i - n_lags: T_i])   # (n_lags, 2)
    windows = np.stack(valid_windows).astype(np.float32)      # (M_val, n_lags, 2)
    M_val = windows.shape[0]
    print(f"    J_eff: {M_val} valid traces, eps={eps} deg")

    J_eff_cols = []
    for axis in range(2):
        delta = np.zeros(2, dtype=np.float32)
        delta[axis] = eps

        rates_plus_all  = []
        rates_minus_all = []

        for b_start in range(0, M_val, batch_size):
            b_end   = min(b_start + batch_size, M_val)
            batch   = windows[b_start:b_end]              # (B, n_lags, 2)

            for sign, store in [(+1, rates_plus_all), (-1, rates_minus_all)]:
                shifted = batch + sign * delta            # broadcast over (B, n_lags)
                stims   = []
                with torch.no_grad():
                    for b in range(shifted.shape[0]):
                        trace_t = torch.tensor(shifted[b], dtype=torch.float32,
                                               device=dev)
                        movie = retina(world_gray, trace_t)[0, 0] / 127.0
                        stims.append(embed_time_lags_pure(movie, n_lags))
                    stim_batch = torch.cat(stims, dim=0)  # (B, 1, n_lags, H, W)
                    r = _rates_from_stim(model, readout, stim_batch)  # (B, N)
                store.append(r.cpu().numpy())

        r_plus  = np.concatenate(rates_plus_all,  axis=0).mean(axis=0)  # (N,)
        r_minus = np.concatenate(rates_minus_all, axis=0).mean(axis=0)  # (N,)
        J_eff_cols.append((r_plus - r_minus) / (2.0 * eps))

    J_eff = np.stack(J_eff_cols, axis=1)                  # (N, 2)
    U_eff, _ = np.linalg.qr(J_eff)
    return J_eff, U_eff


def compute_jacobian_integrated(
    model,
    readout,
    renderer: HiResERenderer,
    retina: HiResRetina,
    traces: np.ndarray,
    durations: np.ndarray,
    orientation_deg: float,
    logmar: float,
    device: str,
    eps: float = 0.005,
    n_lags: int = N_LAGS,
    n_grid: int = 7,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Position-distribution-weighted Jacobian:

        J_integrated = E_{p ~ P_FEM}[J(p)]

    where P_FEM is the marginal distribution of eye positions across all
    valid trace frames, and each J(p) is evaluated at a STATIC null trace
    (all n_lags frames held at position p).

    Using a static trace decouples spatial sensitivity from temporal dynamics:
    J(p) is the pure spatial derivative of the model's settled response at p,
    with no GRU transients.  The weighted average over the FEM position histogram
    gives the effective Jacobian that accounts for the model's spatial nonlinearity
    over the positions actually visited during fixation.

    Paired with Sigma_trial for C_pred = J_int Σ_trial J_intᵀ — the cleanest
    test of whether position-distribution curvature explains the scale discrepancy.

    Args:
        n_grid: side length of the 2D histogram grid (n_grid×n_grid bins)
        eps:    finite-difference step size in degrees

    Returns:
        J_int: (N, 2)  — position-histogram-weighted Jacobian columns
        U_int: (N, 2)  — orthonormal basis via QR
    """
    dev = torch.device(device)
    renderer.eval(); retina.eval(); model.model.eval(); readout.eval()

    with torch.no_grad():
        world_img = renderer(orientation_deg, logmar).to(dev)
        world_gray = 127.0 * (1.0 - world_img)

    # Pool all valid frame positions across traces
    pos_list = []
    for i in range(len(durations)):
        T_i = int(durations[i])
        if T_i < n_lags:
            continue
        pos_list.append(traces[i, :T_i])
    all_pos = np.concatenate(pos_list, axis=0).astype(np.float64)   # (M*T, 2)

    # 2D histogram over the marginal position distribution
    x_min, x_max = all_pos[:, 0].min(), all_pos[:, 0].max()
    y_min, y_max = all_pos[:, 1].min(), all_pos[:, 1].max()
    H, xe, ye = np.histogram2d(
        all_pos[:, 0], all_pos[:, 1],
        bins=[np.linspace(x_min, x_max, n_grid + 1),
              np.linspace(y_min, y_max, n_grid + 1)],
    )
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    XX, YY = np.meshgrid(xc, yc, indexing='ij')
    grid_pos = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)  # (K, 2)
    weights  = H.ravel().astype(np.float64)

    # Keep only non-empty bins
    mask     = weights > 0
    grid_pos = grid_pos[mask]
    weights  = weights[mask]
    weights /= weights.sum()
    K = len(grid_pos)
    print(f"    J_int: {K}/{n_grid**2} non-empty bins, "
          f"position range x=[{x_min:.3f},{x_max:.3f}] y=[{y_min:.3f},{y_max:.3f}] deg")

    J_int_cols = []
    for axis in range(2):
        delta = np.zeros(2, dtype=np.float32)
        delta[axis] = eps

        rates_weighted = np.zeros(readout.n_units, dtype=np.float64)

        for b_start in range(0, K, batch_size):
            b_end    = min(b_start + batch_size, K)
            batch_p  = grid_pos[b_start:b_end]   # (B, 2)
            batch_w  = weights[b_start:b_end]     # (B,)

            stims_plus  = []
            stims_minus = []
            with torch.no_grad():
                for b in range(batch_p.shape[0]):
                    for sign, stims_list in [(+1, stims_plus), (-1, stims_minus)]:
                        # Static null trace: all n_lags frames fixed at position p
                        pos_shifted = batch_p[b] + sign * delta
                        null_trace  = torch.tensor(
                            np.tile(pos_shifted, (n_lags, 1)),
                            dtype=torch.float32, device=dev,
                        )
                        movie = retina(world_gray, null_trace)[0, 0] / 127.0
                        stims_list.append(embed_time_lags_pure(movie, n_lags))

                r_plus  = _rates_from_stim(
                    model, readout, torch.cat(stims_plus,  dim=0)
                ).cpu().numpy()   # (B, N)
                r_minus = _rates_from_stim(
                    model, readout, torch.cat(stims_minus, dim=0)
                ).cpu().numpy()

            drdelta = (r_plus - r_minus) / (2.0 * eps)          # (B, N)
            rates_weighted += (batch_w[:, None] * drdelta).sum(0)  # (N,)

        J_int_cols.append(rates_weighted)

    J_int = np.stack(J_int_cols, axis=1).astype(np.float64)   # (N, 2)
    U_int, _ = np.linalg.qr(J_int)
    return J_int, U_int


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FEM covariance
# ═══════════════════════════════════════════════════════════════════════════════

def compute_fem_covariance(ravg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    C_FEM = covariance of time-averaged real rates across M trials.
    Returns (C_FEM: (N,N), U_pca2: (N,2)) — top 2 eigenvectors.
    """
    R = ravg - ravg.mean(0, keepdims=True)
    C = (R.T @ R) / max(R.shape[0] - 1, 1)
    C = (C + C.T) / 2
    eigvals, eigvecs = np.linalg.eigh(C)
    U_pca2 = eigvecs[:, -2:]     # (N, 2) top eigenvectors (ascending order from eigh)
    return C, U_pca2


def compute_sigma_eye(traces: np.ndarray, durations: np.ndarray,
                      n_lags: int) -> np.ndarray:
    """
    Per-frame eye-position covariance (Σ_eye for Test 4).

    Uses per-frame deviations from each trial's centroid, pooled across all
    (M × T) frames — mechanistically correct because the GRU weights recent
    frames rather than the trial mean.

    Returns: (2, 2) float64.
    """
    devs = []
    for i in range(len(durations)):
        T = int(durations[i])
        if T < n_lags:
            continue
        ep = traces[i, :T]                    # (T, 2)
        centroid = ep.mean(0, keepdims=True)   # (1, 2)
        devs.append(ep - centroid)             # (T, 2)
    all_devs = np.concatenate(devs, axis=0)    # (M*T, 2)
    return np.cov(all_devs.T).astype(np.float64)


def compute_sigma_trial(traces: np.ndarray, durations: np.ndarray,
                        n_lags: int) -> np.ndarray:
    """
    Covariance of per-trace mean positions (Σ_trial for Test 4b).

    Paired with J_eff: the remaining variance source after temporal integration
    is between-trial drift, not within-trial rapid FEM.

    Returns: (2, 2) float64.
    """
    means = []
    for i in range(len(durations)):
        T_i = int(durations[i])
        if T_i < n_lags:
            continue
        means.append(traces[i, :T_i].mean(0))
    return np.cov(np.stack(means).T).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def alignment_score(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    Mean squared cosine of principal angles between column spaces of U1 and U2.
    = 1 if identical subspace, ≈ d/N if independent.
    """
    Q1, _ = np.linalg.qr(U1)
    Q2, _ = np.linalg.qr(U2)
    sv = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(np.mean(np.clip(sv, 0.0, 1.0) ** 2))


def capture_fraction(U: np.ndarray, C: np.ndarray) -> float:
    """
    Fraction of C's trace captured by the column space of U.
    = tr(U^T C U) / tr(C).
    """
    Q, _ = np.linalg.qr(U)
    num = float(np.trace(Q.T @ C @ Q))
    denom = float(np.trace(C))
    return num / (denom + 1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Null controls
# ═══════════════════════════════════════════════════════════════════════════════

def random_rank2_nulls(N: int, C: np.ndarray, n_reps: int = 500,
                       rng: np.random.Generator = None) -> np.ndarray:
    """
    Distribution of capture_fraction for 500 random rank-2 subspaces.
    Returns (n_reps,) array of capture fractions.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    fracs = np.empty(n_reps)
    for i in range(n_reps):
        U_rand, _ = np.linalg.qr(rng.standard_normal((N, 2)))
        fracs[i] = capture_fraction(U_rand, C)
    return fracs


def matched_energy_nulls(J: np.ndarray, C: np.ndarray, n_reps: int = 500,
                         rng: np.random.Generator = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Row-shuffle null: permute neuron identities in J, preserving column norms.

    For each rep: shuffle rows of J → J_null → U_null via QR.
    Returns two (n_reps,) arrays: (alignment_scores, capture_fractions).
    """
    if rng is None:
        rng = np.random.default_rng(1)
    N = J.shape[0]
    align_null = np.empty(n_reps)
    capt_null = np.empty(n_reps)
    # U_pca2 is not available here; alignment is computed externally against it
    # — caller should call alignment_score(U_null, U_pca2) for each rep.
    # We just return the null U subspaces via capture_fractions.
    U_null_list = []
    for i in range(n_reps):
        perm = rng.permutation(N)
        J_null = J[perm]
        U_null, _ = np.linalg.qr(J_null)
        capt_null[i] = capture_fraction(U_null, C)
        U_null_list.append(U_null)
    return capt_null, U_null_list


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Test 6: Representational intervention
# ═══════════════════════════════════════════════════════════════════════════════

def _project_out(ravg: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Remove U's column space from each row of ravg. (M, N) → (M, N)."""
    Q, _ = np.linalg.qr(U)
    return ravg - (ravg @ Q) @ Q.T


def _project_onto(ravg: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Keep only U's column space in each row of ravg. (M, N) → (M, N)."""
    Q, _ = np.linalg.qr(U)
    return (ravg @ Q) @ Q.T


def decode_accuracy(ravg_by_ori: dict, n_splits: int = 5) -> float:
    """
    Orientation decoding accuracy (D1: logistic regression on time-averaged rates).

    ravg_by_ori: dict ori_key → (M, N) float64
    Returns mean CV accuracy across n_splits GroupKFold splits.
    """
    stim_ids = sorted(ravg_by_ori.keys())
    # Equalise classes by truncating to min trial count across orientations
    min_M = min(ravg_by_ori[k].shape[0] for k in stim_ids)
    X_parts, y_parts, g_parts = [], [], []
    for label, key in enumerate(stim_ids):
        R = ravg_by_ori[key][:min_M]
        X_parts.append(R)
        y_parts.append(np.full(min_M, label, dtype=int))
        g_parts.append(np.arange(min_M))
    X = np.concatenate(X_parts, axis=0).astype(np.float64)
    y = np.concatenate(y_parts)
    groups = np.concatenate(g_parts)

    # GroupKFold by trial index means the decoder is tested on held-out TRACES,
    # i.e., it must generalise across different FEM realisations.  This is
    # cross-trace generalisation, not just classification accuracy.
    gkf = GroupKFold(n_splits=min(n_splits, min_M))
    accs = []
    for tr_idx, te_idx in gkf.split(X, y, groups):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr_idx])
        X_te = sc.transform(X[te_idx])
        clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        clf.fit(X_tr, y[tr_idx])
        accs.append(clf.score(X_te, y[te_idx]))
    return float(np.mean(accs))


def run_test6(
    U_jac_by_ori: dict,
    ravg_by_ori: dict,
) -> dict:
    """
    Test 6: project out / project onto U_jac per orientation, run D1 decoder.

    U_jac_by_ori: dict ori_key → (N, 2) Jacobian subspace
    ravg_by_ori:  dict ori_key → (M, N) time-averaged real rates

    Returns dict with:
        acc_full:            baseline accuracy
        acc_subtract:        after projecting out U_jac_k per class
        acc_isolate_jac:     decoding from U_jac component only
        acc_isolate_perp:    decoding from complement only
        acc_subtract_pooled: subtract with pooled U_jac (6c)
        acc_isolate_pooled:  isolate with pooled U_jac (6c)
    """
    # Full accuracy baseline
    acc_full = decode_accuracy(ravg_by_ori)

    # 6a: subtract orientation-specific U_jac
    ravg_sub = {k: _project_out(ravg_by_ori[k], U_jac_by_ori[k])
                for k in ravg_by_ori}
    acc_subtract = decode_accuracy(ravg_sub)

    # 6b: isolate Jacobian and complement
    ravg_jac = {k: _project_onto(ravg_by_ori[k], U_jac_by_ori[k])
                for k in ravg_by_ori}
    ravg_perp = {k: _project_out(ravg_by_ori[k], U_jac_by_ori[k])
                 for k in ravg_by_ori}
    acc_isolate_jac = decode_accuracy(ravg_jac)
    acc_isolate_perp = decode_accuracy(ravg_perp)

    # 6c: pooled U_jac — top 2 left singular vectors of the concatenated U_jac matrices
    # These are the 2 directions most consistently aligned with U_jac across orientations.
    U_all = np.column_stack([U_jac_by_ori[f'ori{o}'] for o in ORIENTATIONS])  # (N, 8)
    U_s, _, _ = np.linalg.svd(U_all, full_matrices=False)  # (N, 8)
    U_pooled = U_s[:, :2]  # (N, 2)

    ravg_sub_pool = {k: _project_out(ravg_by_ori[k], U_pooled) for k in ravg_by_ori}
    ravg_jac_pool = {k: _project_onto(ravg_by_ori[k], U_pooled) for k in ravg_by_ori}
    acc_subtract_pooled = decode_accuracy(ravg_sub_pool)
    acc_isolate_pooled = decode_accuracy(ravg_jac_pool)

    # Specific-vs-pooled delta: if large, the orientation-specific result is partly
    # a geometric artifact of modifying each class in a different basis.
    delta_specific_vs_pooled = acc_subtract - acc_subtract_pooled

    # Trial-shuffled control: shuffle trial order independently per neuron, destroying
    # population correlations while preserving marginal statistics.
    # If acc_full >> acc_trial_shuffled, the decoder uses population structure.
    # If they are similar, individual neuron marginals carry most of the information.
    rng_shuf = np.random.default_rng(seed=0)
    ravg_trial_shuffled = {}
    for k, R in ravg_by_ori.items():
        R_shuf = R.copy()
        for n in range(R.shape[1]):
            rng_shuf.shuffle(R_shuf[:, n])
        ravg_trial_shuffled[k] = R_shuf
    acc_trial_shuffled = decode_accuracy(ravg_trial_shuffled)

    return {
        'acc_full': acc_full,
        'acc_subtract': acc_subtract,
        'delta_acc_subtract': acc_subtract - acc_full,
        'acc_isolate_jac': acc_isolate_jac,
        'acc_isolate_perp': acc_isolate_perp,
        'acc_subtract_pooled': acc_subtract_pooled,
        'delta_acc_subtract_pooled': acc_subtract_pooled - acc_full,
        'delta_specific_vs_pooled': delta_specific_vs_pooled,
        'acc_isolate_pooled': acc_isolate_pooled,
        'acc_trial_shuffled': acc_trial_shuffled,
        'delta_trial_shuffled': acc_trial_shuffled - acc_full,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Main analysis loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis(
    logmars: list[float],
    device: str,
    n_null_reps: int,
    n_lags: int,
    n_mean_frames: int,
    run_eff_jacobian: bool = False,
    run_int_jacobian: bool = False,
    int_n_grid: int = 7,
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model and readout ─────────────────────────────────────────────────
    print("Loading model...")
    model, _ = get_model_and_dataset_configs()
    model.model.eval()
    model = model.to(device)

    with open(str(PKL_PATH), 'rb') as f:
        outputs = dill.load(f)
    readout = get_spatial_readout(model, outputs).to(device)
    readout.eval()

    print(f"  N = {readout.n_units} neurons")

    # ── Build HiRes renderer + retina ─────────────────────────────────────────
    renderer = HiResERenderer(device=device).to(device)
    retina_module = HiResRetina().to(device)

    # ── Load eye traces and compute mean trace ────────────────────────────────
    traces, durations = load_eye_traces(DATA_DIR / 'eye_traces.npz')
    print(f"Loaded {len(traces)} eye traces; computing mean over first "
          f"{n_mean_frames} frames...")
    mean_trace = compute_mean_trace(traces, durations, n_mean_frames)  # (n_mean_frames, 2)

    sigma_eye = compute_sigma_eye(traces, durations, n_lags)
    print(f"  Σ_eye (per-frame):\n{sigma_eye}")
    sigma_trial = compute_sigma_trial(traces, durations, n_lags)
    print(f"  Σ_trial (between-trial):\n{sigma_trial}")

    # ── Per-logmar analysis ────────────────────────────────────────────────────
    all_results = {}
    summary_lines = []
    summary_lines.append(
        f"{'LogMAR':>8} {'Ori':>5} {'Align_jac':>10} {'Capt_jac':>10} "
        f"{'Capt_rand_p95':>14} {'Capt_null_p95':>14} {'Capt_null_mean':>14}"
    )
    summary_lines.append('-' * 80)

    for logmar in logmars:
        print(f"\n{'='*60}")
        print(f"LogMAR = {logmar:+.2f}")
        print('='*60)

        # Load real rates per orientation
        print("Loading cached rates (real)...")
        ravg_by_ori = load_rates_by_ori(logmar, 'real')

        # ── Test 3: Jacobian per orientation ─────────────────────────────────
        print("Computing Jacobians via forward-AD...")
        J_by_ori = {}
        U_jac_by_ori = {}
        J_eff_by_ori = {}
        U_eff_by_ori = {}
        J_int_by_ori = {}
        U_int_by_ori = {}
        align_by_ori = {}
        align_eff_by_ori = {}
        align_int_by_ori = {}
        capt_by_ori = {}
        capt_null_by_ori = {}
        capt_rand_by_ori = {}
        U_pca2_by_ori = {}
        C_FEM_by_ori = {}

        for ori in ORIENTATIONS:
            key = f'ori{ori}'
            print(f"  Orientation {ori}°...", end=' ', flush=True)

            # Jacobian
            J, U_jac = compute_jacobian(
                model, readout, renderer, retina_module,
                mean_trace, float(ori), logmar, device, n_lags
            )
            J_by_ori[key] = J
            U_jac_by_ori[key] = U_jac

            # Test 4d: integrated Jacobian (optional, ~1 min per condition on GPU)
            if run_int_jacobian:
                print(f"  J_int orientation {ori}°...", end=' ', flush=True)
                J_int, U_int = compute_jacobian_integrated(
                    model, readout, renderer, retina_module,
                    traces, durations, float(ori), logmar, device,
                    n_lags=n_lags, n_grid=int_n_grid,
                )
                J_int_by_ori[key] = J_int
                U_int_by_ori[key] = U_int

            # Test 4b: effective Jacobian (optional, ~5 min per condition on GPU)
            if run_eff_jacobian:
                print(f"  J_eff orientation {ori}°...", end=' ', flush=True)
                J_eff, U_eff = compute_jacobian_eff(
                    model, readout, renderer, retina_module,
                    traces, durations, float(ori), logmar, device, n_lags=n_lags
                )
                J_eff_by_ori[key] = J_eff
                U_eff_by_ori[key] = U_eff

            # C_FEM and U_pca2 from real rates
            C_FEM, U_pca2 = compute_fem_covariance(ravg_by_ori[key])
            C_FEM_by_ori[key] = C_FEM
            U_pca2_by_ori[key] = U_pca2

            # Metrics
            aln = alignment_score(U_jac, U_pca2)
            cap = capture_fraction(U_jac, C_FEM)
            align_by_ori[key] = aln
            capt_by_ori[key] = cap

            # Null controls
            N_neurons = J.shape[0]
            rng0 = np.random.default_rng(42 + ori)
            rand_caps = random_rank2_nulls(N_neurons, C_FEM, n_reps=n_null_reps, rng=rng0)

            rng1 = np.random.default_rng(99 + ori)
            null_caps, null_U_list = matched_energy_nulls(
                J, C_FEM, n_reps=n_null_reps, rng=rng1)
            null_aligns = np.array([alignment_score(U_n, U_pca2)
                                    for U_n in null_U_list])

            capt_rand_by_ori[key] = rand_caps
            capt_null_by_ori[key] = (null_caps, null_aligns)

            p95_rand = float(np.percentile(rand_caps, 95))
            p95_null = float(np.percentile(null_caps, 95))
            mean_null = float(null_caps.mean())

            print(f"align={aln:.3f}  capture={cap:.3f}  "
                  f"rand_p95={p95_rand:.3f}  null_p95={p95_null:.3f}")
            if run_int_jacobian:
                aln_int = alignment_score(U_int_by_ori[key], U_pca2)
                align_int_by_ori[key] = aln_int
                print(f"    J_int align={aln_int:.3f}")
            if run_eff_jacobian:
                aln_eff = alignment_score(U_eff_by_ori[key], U_pca2)
                align_eff_by_ori[key] = aln_eff
                print(f"    J_eff align={aln_eff:.3f}")

            summary_lines.append(
                f"{logmar:>8.2f} {ori:>5}° "
                f"{aln:>10.3f} {cap:>10.3f} "
                f"{p95_rand:>14.3f} {p95_null:>14.3f} {mean_null:>14.3f}"
            )

        # ── Orientation-invariance check ─────────────────────────────────────
        print("\nOrientation-invariance of U_jac:")
        ori_pairs = [(0, 90), (0, 180), (0, 270), (90, 180), (90, 270), (180, 270)]
        ori_cross_align = {}
        for o1, o2 in ori_pairs:
            a = alignment_score(U_jac_by_ori[f'ori{o1}'], U_jac_by_ori[f'ori{o2}'])
            ori_cross_align[(o1, o2)] = a
            print(f"  U_jac({o1}°) vs U_jac({o2}°): alignment = {a:.3f}")

        # ── Test 6: Representational intervention ────────────────────────────
        print("\nTest 6: Representational intervention...")
        test6 = run_test6(U_jac_by_ori, ravg_by_ori)
        print(f"  Full accuracy:          {test6['acc_full']:.3f}")
        print(f"  After subtract (spec):  {test6['acc_subtract']:.3f}  "
              f"(Δ = {test6['delta_acc_subtract']:+.3f})")
        print(f"  After subtract (pool):  {test6['acc_subtract_pooled']:.3f}  "
              f"(Δ = {test6['delta_acc_subtract_pooled']:+.3f})")
        print(f"  Spec vs pooled Δ:       {test6['delta_specific_vs_pooled']:+.3f}  "
              f"({'OK' if abs(test6['delta_specific_vs_pooled']) < 0.05 else 'CHECK — may be geometric artifact'})")
        print(f"  Isolate Jac only:       {test6['acc_isolate_jac']:.3f}")
        print(f"  Isolate perp only:      {test6['acc_isolate_perp']:.3f}")
        print(f"  Trial-shuffled control: {test6['acc_trial_shuffled']:.3f}  "
              f"(Δ vs full = {test6['delta_trial_shuffled']:+.3f})")

        # ── IC-A: signal geometry ─────────────────────────────────────────────
        means = np.stack([ravg_by_ori[f'ori{o}'].mean(0) for o in ORIENTATIONS])
        means -= means.mean(0, keepdims=True)
        C_signal = (means.T @ means) / (len(ORIENTATIONS) - 1)
        signal_capture_by_ori = {
            key: capture_fraction(U_jac_by_ori[key], C_signal)
            for key in U_jac_by_ori
        }
        print("\nIC-A: Jacobian capture of C_signal:")
        for key, sc in signal_capture_by_ori.items():
            print(f"  {key}: alpha_signal = {sc:.3f}")

        # ── IC-C: task-relevant SNR ───────────────────────────────────────────
        # Use trace complement: tr(C) = tr(Q^T C Q) + tr(Q_perp^T C Q_perp)
        # so complement power = total_trace - jac_power, avoiding an N×N QR.
        snr_jac_list, snr_perp_list = [], []
        eps = 1e-12
        for ori in ORIENTATIONS:
            key = f'ori{ori}'
            Q, _ = np.linalg.qr(U_jac_by_ori[key])
            sig_total = float(np.trace(C_signal))
            noi_total = float(np.trace(C_FEM_by_ori[key]))
            sig_jac = float(np.trace(Q.T @ C_signal @ Q))
            noi_jac = float(np.trace(Q.T @ C_FEM_by_ori[key] @ Q))
            sig_perp = sig_total - sig_jac
            noi_perp = noi_total - noi_jac
            snr_jac_list.append(sig_jac / (noi_jac + eps))
            snr_perp_list.append(sig_perp / (noi_perp + eps))

        print(f"\nIC-C SNR: Jacobian = {np.mean(snr_jac_list):.3f}, "
              f"Complement = {np.mean(snr_perp_list):.3f}")

        # ── IC-E: ensemble C_FEM averaged over orientations ───────────────────
        C_FEM_avg = np.mean(list(C_FEM_by_ori.values()), axis=0)
        w_avg, V_avg = np.linalg.eigh(C_FEM_avg)
        w_avg_sorted = w_avg[::-1]
        print(f"\nIC-E: C_FEM_avg top-5 eigenvalues: "
              f"{w_avg_sorted[:5].round(5)}")
        top2_pct = w_avg_sorted[:2].sum() / (w_avg_sorted.sum() + 1e-12)
        print(f"  Top-2 eigenvalues capture {top2_pct:.1%} of total variance")

        # ── Save results ──────────────────────────────────────────────────────
        save_path = OUT_DIR / f'test3_lm{logmar:.2f}.npz'
        np.savez_compressed(
            str(save_path),
            logmar=logmar,
            orientations=np.array(ORIENTATIONS),
            J_ori0=J_by_ori['ori0'],
            J_ori90=J_by_ori['ori90'],
            J_ori180=J_by_ori['ori180'],
            J_ori270=J_by_ori['ori270'],
            U_jac_ori0=U_jac_by_ori['ori0'],
            U_jac_ori90=U_jac_by_ori['ori90'],
            U_jac_ori180=U_jac_by_ori['ori180'],
            U_jac_ori270=U_jac_by_ori['ori270'],
            U_pca2_ori0=U_pca2_by_ori['ori0'],
            U_pca2_ori90=U_pca2_by_ori['ori90'],
            U_pca2_ori180=U_pca2_by_ori['ori180'],
            U_pca2_ori270=U_pca2_by_ori['ori270'],
            C_FEM_ori0=C_FEM_by_ori['ori0'],
            C_FEM_ori90=C_FEM_by_ori['ori90'],
            C_FEM_ori180=C_FEM_by_ori['ori180'],
            C_FEM_ori270=C_FEM_by_ori['ori270'],
            C_signal=C_signal,
            sigma_eye=sigma_eye,
            align_ori0=align_by_ori['ori0'],
            align_ori90=align_by_ori['ori90'],
            align_ori180=align_by_ori['ori180'],
            align_ori270=align_by_ori['ori270'],
            capt_ori0=capt_by_ori['ori0'],
            capt_ori90=capt_by_ori['ori90'],
            capt_ori180=capt_by_ori['ori180'],
            capt_ori270=capt_by_ori['ori270'],
            rand_caps_ori0=capt_rand_by_ori['ori0'],
            rand_caps_ori90=capt_rand_by_ori['ori90'],
            rand_caps_ori180=capt_rand_by_ori['ori180'],
            rand_caps_ori270=capt_rand_by_ori['ori270'],
            null_capt_ori0=capt_null_by_ori['ori0'][0],
            null_capt_ori90=capt_null_by_ori['ori90'][0],
            null_capt_ori180=capt_null_by_ori['ori180'][0],
            null_capt_ori270=capt_null_by_ori['ori270'][0],
            null_align_ori0=capt_null_by_ori['ori0'][1],
            null_align_ori90=capt_null_by_ori['ori90'][1],
            null_align_ori180=capt_null_by_ori['ori180'][1],
            null_align_ori270=capt_null_by_ori['ori270'][1],
            # Test 6
            t6_acc_full=test6['acc_full'],
            t6_acc_subtract=test6['acc_subtract'],
            t6_delta_subtract=test6['delta_acc_subtract'],
            t6_acc_isolate_jac=test6['acc_isolate_jac'],
            t6_acc_isolate_perp=test6['acc_isolate_perp'],
            t6_acc_subtract_pooled=test6['acc_subtract_pooled'],
            t6_delta_subtract_pooled=test6['delta_acc_subtract_pooled'],
            t6_delta_specific_vs_pooled=test6['delta_specific_vs_pooled'],
            t6_acc_trial_shuffled=test6['acc_trial_shuffled'],
            t6_delta_trial_shuffled=test6['delta_trial_shuffled'],
            # IC-A
            signal_capture=np.array([signal_capture_by_ori[f'ori{o}']
                                      for o in ORIENTATIONS]),
            # IC-C
            snr_jac=np.array(snr_jac_list),
            snr_perp=np.array(snr_perp_list),
            # IC-E
            C_FEM_avg=C_FEM_avg,
            C_FEM_avg_eigvals=w_avg,
            sigma_trial=sigma_trial,
            # Test 4b: effective Jacobian (only present when --run_eff_jacobian)
            **({f'J_eff_ori{o}':  J_eff_by_ori[f'ori{o}']  for o in ORIENTATIONS}
               if run_eff_jacobian else {}),
            **({f'U_eff_ori{o}':  U_eff_by_ori[f'ori{o}']  for o in ORIENTATIONS}
               if run_eff_jacobian else {}),
            **({f'align_eff_ori{o}': align_eff_by_ori[f'ori{o}'] for o in ORIENTATIONS}
               if run_eff_jacobian else {}),
            **({f'J_int_ori{o}':  J_int_by_ori[f'ori{o}']  for o in ORIENTATIONS}
               if run_int_jacobian else {}),
            **({f'U_int_ori{o}':  U_int_by_ori[f'ori{o}']  for o in ORIENTATIONS}
               if run_int_jacobian else {}),
            **({f'align_int_ori{o}': align_int_by_ori[f'ori{o}'] for o in ORIENTATIONS}
               if run_int_jacobian else {}),
        )
        print(f"  Saved: {save_path}")

        all_results[logmar] = {
            'J_by_ori': J_by_ori,
            'U_jac_by_ori': U_jac_by_ori,
            'U_pca2_by_ori': U_pca2_by_ori,
            'C_FEM_by_ori': C_FEM_by_ori,
            'align_by_ori': align_by_ori,
            'capt_by_ori': capt_by_ori,
            'capt_rand_by_ori': capt_rand_by_ori,
            'capt_null_by_ori': capt_null_by_ori,
            'ori_cross_align': ori_cross_align,
            'test6': test6,
            'signal_capture_by_ori': signal_capture_by_ori,
            'snr_jac': np.array(snr_jac_list),
            'snr_perp': np.array(snr_perp_list),
            'C_FEM_avg': C_FEM_avg,
        }

    # ── Write summary ──────────────────────────────────────────────────────────
    summary_path = OUT_DIR / 'test3_summary.txt'
    with open(str(summary_path), 'w') as f:
        f.write('\n'.join(summary_lines) + '\n\n')
        f.write('Test 6 summary\n' + '-' * 40 + '\n')
        for lm, res in all_results.items():
            t6 = res['test6']
            f.write(
                f"LogMAR {lm:+.2f}: "
                f"full={t6['acc_full']:.3f}  "
                f"sub-spec={t6['acc_subtract']:.3f} (Δ={t6['delta_acc_subtract']:+.3f})  "
                f"sub-pool={t6['acc_subtract_pooled']:.3f} (Δ={t6['delta_acc_subtract_pooled']:+.3f})  "
                f"spec-vs-pool={t6['delta_specific_vs_pooled']:+.3f}  "
                f"Jac-only={t6['acc_isolate_jac']:.3f}  "
                f"perp-only={t6['acc_isolate_perp']:.3f}  "
                f"shuffled={t6['acc_trial_shuffled']:.3f}\n"
            )
    print(f"\nSummary saved: {summary_path}")
    print('\n'.join(summary_lines))

    # ── Figures ────────────────────────────────────────────────────────────────
    _make_plots(all_results, logmars)
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def _make_plots(all_results: dict, logmars: list[float]):
    n_lm = len(logmars)
    fig, axes = plt.subplots(3, n_lm, figsize=(5 * n_lm, 13))
    if n_lm == 1:
        axes = axes[:, np.newaxis]

    for col, logmar in enumerate(logmars):
        res = all_results[logmar]

        # Row 0: capture fraction by orientation + null distributions
        ax = axes[0, col]
        caps_jac = [res['capt_by_ori'][f'ori{o}'] for o in ORIENTATIONS]
        rand_p95 = [np.percentile(res['capt_rand_by_ori'][f'ori{o}'], 95)
                    for o in ORIENTATIONS]
        null_p95 = [np.percentile(res['capt_null_by_ori'][f'ori{o}'][0], 95)
                    for o in ORIENTATIONS]
        x = np.arange(4)
        ax.bar(x, caps_jac, color='steelblue', label='U_jac capture')
        ax.plot(x, rand_p95, 'k--', label='random null p95')
        ax.plot(x, null_p95, 'r--', label='matched-energy null p95')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{o}°' for o in ORIENTATIONS])
        ax.set_ylabel('C_FEM capture fraction')
        ax.set_title(f'LM={logmar:+.2f}: Test 3 — Capture')
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1)

        # Row 1: alignment score by orientation + null distribution
        ax = axes[1, col]
        aligns = [res['align_by_ori'][f'ori{o}'] for o in ORIENTATIONS]
        null_align_p95 = [
            np.percentile(res['capt_null_by_ori'][f'ori{o}'][1], 95)
            for o in ORIENTATIONS
        ]
        ax.bar(x, aligns, color='coral', label='U_jac alignment')
        ax.plot(x, null_align_p95, 'r--', label='matched-energy null p95')
        ax.axhline(2 / next(iter(res['C_FEM_by_ori'].values())).shape[0],
                   color='gray', linestyle=':', label='chance (2/N)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{o}°' for o in ORIENTATIONS])
        ax.set_ylabel('Alignment score (mean cos²)')
        ax.set_title(f'LM={logmar:+.2f}: Test 3 — Alignment')
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1)

        # Row 2: Test 6 accuracy bars
        ax = axes[2, col]
        t6 = res['test6']
        labels = ['Full', 'Sub-spec', 'Sub-pool', 'Jac-only', 'Perp-only', 'Shuffled']
        accs = [t6['acc_full'], t6['acc_subtract'], t6['acc_subtract_pooled'],
                t6['acc_isolate_jac'], t6['acc_isolate_perp'], t6['acc_trial_shuffled']]
        colors = ['steelblue', 'darkorange', 'salmon', 'seagreen', 'mediumpurple', 'gray']
        ax.bar(range(6), accs, color=colors)
        ax.axhline(0.25, color='gray', linestyle='--', label='chance')
        ax.axhline(t6['acc_full'], color='steelblue', linestyle=':', alpha=0.6)
        ax.set_xticks(range(6))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Orientation decoding accuracy')
        ax.set_title(f'LM={logmar:+.2f}: Test 6 — Intervention')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plot_path = OUT_DIR / 'test3_test6_plots.pdf'
    fig.savefig(str(plot_path), bbox_inches='tight')
    plt.close(fig)
    print(f"Plots saved: {plot_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--logmars', type=str, default='-0.20,-0.40',
                   help='Comma-separated LogMAR values (default: -0.20,-0.40)')
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--n_null_reps', type=int, default=500,
                   help='Number of null control resamples (default: 500)')
    p.add_argument('--n_lags', type=int, default=N_LAGS,
                   help='Model temporal context frames (default: 32)')
    p.add_argument('--n_mean_frames', type=int, default=120,
                   help='Number of frames to use for mean trace (default: 120)')
    p.add_argument('--run_eff_jacobian', action='store_true',
                   help='Compute J_eff via finite differences over all traces (Test 4b; slow)')
    p.add_argument('--run_int_jacobian', action='store_true',
                   help='Compute J_int: position-histogram-weighted Jacobian (Test 4d; fast)')
    p.add_argument('--int_n_grid', type=int, default=7,
                   help='Histogram grid side length for J_int (default: 7 → 49 bins)')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    logmars = [float(x) for x in args.logmars.split(',')]
    print(f"Running Tests 3 + 6 on LogMARs: {logmars}")
    print(f"Device: {args.device}  |  null reps: {args.n_null_reps}  |  "
          f"n_lags: {args.n_lags}  |  mean_frames: {args.n_mean_frames}")
    run_analysis(
        logmars=logmars,
        device=args.device,
        n_null_reps=args.n_null_reps,
        n_lags=args.n_lags,
        n_mean_frames=args.n_mean_frames,
        run_eff_jacobian=args.run_eff_jacobian,
        run_int_jacobian=args.run_int_jacobian,
        int_n_grid=args.int_n_grid,
    )
