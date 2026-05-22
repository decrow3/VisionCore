"""
Tasks 3.1–3.2: Signal and Noise Covariance, Alignment Analysis,
               Representational Intervention

Computes C_signal (signal covariance driven by stimulus identity) and
C_FEM (noise covariance driven by eye movement variability), their
eigenspectra, and the alignment fraction α.

Also implements the representational intervention: project out the
FEM-aligned signal subspace and rerun decoding to test whether the
FEM structure is signal-bearing (accuracy drops) or nuisance (improves).
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_signal_covariance(
    rates_by_stim: dict,
    mode: str = 'instantaneous',
) -> np.ndarray:
    """
    Compute the signal covariance C_signal.

    C_signal = covariance of class-mean rate vectors across stimulus classes.
    High eigenvalues = directions that separate stimulus representations.

    Args:
        rates_by_stim: dict stim_id → (M, T, N) rate arrays
        mode: 'instantaneous' (use time-averaged rates, shape N×N)
              'temporal' (use full trajectories, shape T*N × T*N)

    Returns:
        C_signal: (N, N) or (T*N, T*N) PSD covariance matrix
    """
    stim_ids = sorted(rates_by_stim.keys())

    if mode == 'instantaneous':
        # Each class vector = time-averaged rate per class
        class_vecs = []
        for sid in stim_ids:
            X = rates_by_stim[sid]  # (M, T, N)
            class_vecs.append(X.mean(axis=(0, 1)))  # (N,) — mean over trials and time
        V = np.stack(class_vecs, axis=0)  # (K, N)
        V -= V.mean(axis=0, keepdims=True)
        C_signal = V.T @ V / (len(stim_ids) - 1)  # (N, N)

    elif mode == 'temporal':
        # Each class vector = flattened mean trajectory
        class_vecs = []
        for sid in stim_ids:
            X = rates_by_stim[sid]  # (M, T, N)
            class_vecs.append(X.mean(axis=0).flatten())  # (T*N,)
        V = np.stack(class_vecs, axis=0)  # (K, T*N)
        V -= V.mean(axis=0, keepdims=True)
        C_signal = V.T @ V / (len(stim_ids) - 1)  # (T*N, T*N)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Symmetrize for numerical stability
    return (C_signal + C_signal.T) / 2


def compute_fem_covariance(
    rates_by_stim: dict,
    mode: str = 'instantaneous',
) -> np.ndarray:
    """
    Compute the FEM noise covariance C_FEM.

    C_FEM = covariance of single-trial deviations from the class mean.
    This captures variability driven by different eye movement trajectories.

    Args:
        rates_by_stim: dict stim_id → (M, T, N) rate arrays
        mode: 'instantaneous' or 'temporal' (see compute_signal_covariance)

    Returns:
        C_FEM: (N, N) or (T*N, T*N) PSD covariance matrix
    """
    stim_ids = sorted(rates_by_stim.keys())
    all_residuals = []

    for sid in stim_ids:
        X = rates_by_stim[sid]  # (M, T, N)
        if mode == 'instantaneous':
            # Residual from class mean (time-averaged)
            class_mean = X.mean(axis=(0, 1), keepdims=True)  # (1, 1, N)
            residuals = (X - class_mean).mean(axis=1)  # (M, N) — average over time
            all_residuals.append(residuals)
        elif mode == 'temporal':
            class_mean = X.mean(axis=0, keepdims=True)  # (1, T, N)
            residuals = (X - class_mean).reshape(X.shape[0], -1)  # (M, T*N)
            all_residuals.append(residuals)

    R = np.concatenate(all_residuals, axis=0)  # (M_total, ...)
    R -= R.mean(axis=0, keepdims=True)
    C_FEM = R.T @ R / (len(R) - 1)
    return (C_FEM + C_FEM.T) / 2


def compute_covariances(
    rates_by_stim: dict,
    mode: str = 'instantaneous',
    _verify_temporal_richer: bool = False,
) -> dict:
    """
    Compute C_signal and C_FEM along with their eigendecompositions.

    Args:
        rates_by_stim: dict stim_id → (M, T, N) rate arrays
        mode: 'instantaneous' or 'temporal'

    Returns:
        dict with:
            C_signal: (D, D) signal covariance
            C_FEM: (D, D) FEM noise covariance
            eigvals_signal, eigvecs_signal: eigendecomposition of C_signal
            eigvals_fem, eigvecs_fem: eigendecomposition of C_FEM
            mode: str
    """
    C_signal = compute_signal_covariance(rates_by_stim, mode=mode)
    C_FEM = compute_fem_covariance(rates_by_stim, mode=mode)

    # Eigendecompositions (eigh = symmetric matrix, sorted ascending)
    eigvals_s, eigvecs_s = np.linalg.eigh(C_signal)
    eigvals_f, eigvecs_f = np.linalg.eigh(C_FEM)

    # Sanity: both should be PSD
    if eigvals_s.min() < -1e-8:
        print(f"Warning: C_signal has negative eigenvalues (min={eigvals_s.min():.2e})")
    if eigvals_f.min() < -1e-8:
        print(f"Warning: C_FEM has negative eigenvalues (min={eigvals_f.min():.2e})")

    # Stabilized condition check: C_FEM should be near zero (twin is deterministic)
    # The twin produces the same output for the same eye position, so with no
    # FEM variability the only C_FEM contributions come from floating-point noise.
    fem_trace = float(np.trace(C_FEM))
    signal_trace = float(np.trace(C_signal))
    if fem_trace < 1e-10 * signal_trace:
        print(f"  C_FEM ≈ 0 (trace={fem_trace:.2e}) — consistent with stabilized condition")

    result = {
        'C_signal': C_signal,
        'C_FEM': C_FEM,
        'eigvals_signal': eigvals_s,
        'eigvecs_signal': eigvecs_s,
        'eigvals_fem': eigvals_f,
        'eigvecs_fem': eigvecs_f,
        'mode': mode,
    }

    # When called with instantaneous mode, compare to temporal mode signal rank.
    # Temporal mode should have higher effective rank (richer C_signal) because it
    # captures trajectory shape differences, not just mean-rate differences.
    if mode == 'instantaneous' and _verify_temporal_richer:
        C_signal_temporal = compute_signal_covariance(rates_by_stim, mode='temporal')
        eigvals_t = np.linalg.eigvalsh(C_signal_temporal)
        # Effective rank: exp(entropy of normalized eigenvalue distribution)
        def effective_rank(ev):
            ev = ev[ev > 0]
            p = ev / ev.sum()
            return float(np.exp(-np.sum(p * np.log(p + 1e-30))))
        rank_inst = effective_rank(eigvals_s[eigvals_s > 0])
        rank_temp = effective_rank(eigvals_t[eigvals_t > 0])
        print(f"  C_signal effective rank: instantaneous={rank_inst:.1f}, "
              f"temporal={rank_temp:.1f} "
              f"({'temporal richer' if rank_temp > rank_inst else 'WARNING: temporal not richer — check T is > 1'})")
        result['temporal_rank_check'] = {'instantaneous': rank_inst, 'temporal': rank_temp}

    return result


def alignment_fraction(
    C_signal: np.ndarray,
    C_FEM: np.ndarray,
    d: int = 5,
) -> tuple:
    """
    Compute the alignment fraction α: how much of the FEM noise variance
    lies in the top-d signal subspace.

    α = trace(U_s^T C_FEM U_s) / trace(C_FEM)
    α_chance = d / D (if FEM noise were isotropic)

    Args:
        C_signal: (D, D) signal covariance
        C_FEM: (D, D) FEM noise covariance
        d: number of top signal eigenvectors to use

    Returns:
        alpha: alignment fraction (0 ≤ α ≤ 1)
        alpha_chance: expected α under isotropic noise
    """
    eigvals_s, eigvecs_s = np.linalg.eigh(C_signal)
    U_s = eigvecs_s[:, -d:]  # (D, d) top d signal eigenvectors

    C_FEM_proj = U_s.T @ C_FEM @ U_s  # (d, d) FEM cov in signal subspace
    denom = np.trace(C_FEM)

    alpha = np.trace(C_FEM_proj) / (denom + 1e-12)
    alpha_chance = d / C_signal.shape[0]

    return float(alpha), float(alpha_chance)


def representational_intervention(
    rates_by_stim: dict,
    covariance_result: dict,
    d_remove: int = 5,
) -> dict:
    """
    Remove the top FEM-aligned signal subspace from each trial's rates
    and return cleaned rate arrays.

    This tests whether the FEM-aligned subspace is signal-bearing (removing
    it hurts decoding accuracy) or a nuisance (removing it improves accuracy).

    Args:
        rates_by_stim: dict stim_id → (M, T, N) rate arrays
        covariance_result: output of compute_covariances()
        d_remove: number of FEM-aligned directions to remove

    Returns:
        rates_cleaned: dict stim_id → (M, T, N) rates with FEM-aligned dims removed
        aligned_directions: (N, d_remove) most FEM-aligned signal directions
    """
    C_signal = covariance_result['C_signal']
    C_FEM = covariance_result['C_FEM']
    D = C_signal.shape[0]

    # Step 1: Find most aligned directions between signal and FEM subspaces
    eigvals_s, eigvecs_s = np.linalg.eigh(C_signal)
    eigvals_f, eigvecs_f = np.linalg.eigh(C_FEM)

    U_s = eigvecs_s[:, -d_remove:]  # (D, d) top signal directions
    U_f = eigvecs_f[:, -d_remove:]  # (D, d) top FEM noise directions

    # SVD of the overlap matrix to find most aligned directions
    _, _, Vt = np.linalg.svd(U_s.T @ U_f)
    aligned_directions = U_s @ Vt.T[:, :d_remove]  # (D, d_remove)

    # Step 2: Projection matrix for the aligned subspace
    P_remove = aligned_directions @ aligned_directions.T  # (D, D)

    # Step 3: Project out from each trial at each time step
    rates_cleaned = {}
    for sid, X in rates_by_stim.items():
        M, T, N = X.shape
        X_clean = X.copy()
        for t in range(T):
            X_clean[:, t, :] = X[:, t, :] - X[:, t, :] @ P_remove.T
        rates_cleaned[sid] = X_clean

    return rates_cleaned, aligned_directions


def compute_subspace_snr(
    C_signal: np.ndarray,
    C_FEM: np.ndarray,
    d: int = 5,
) -> float:
    """
    Compute the task-relevant subspace SNR:
    SNR = trace(U_s^T C_signal U_s) / trace(U_s^T C_FEM U_s)

    High SNR: signal subspace is clean relative to FEM noise.
    Low SNR: FEM noise contaminates the signal subspace.
    """
    eigvals_s, eigvecs_s = np.linalg.eigh(C_signal)
    U_s = eigvecs_s[:, -d:]

    signal_power = np.trace(U_s.T @ C_signal @ U_s)
    noise_power = np.trace(U_s.T @ C_FEM @ U_s)
    return float(signal_power / (noise_power + 1e-12))


def plot_eigenspectra(
    covariance_results: dict,
    n_show: int = 20,
    figsize=(10, 5),
) -> plt.Figure:
    """
    Plot eigenvalue spectra of C_signal and C_FEM.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, (cov_name, ev_key) in zip(axes, [
        ('Signal covariance (C_signal)', 'eigvals_signal'),
        ('FEM noise covariance (C_FEM)', 'eigvals_fem'),
    ]):
        for label, data in covariance_results.items():
            ev = data[ev_key][-n_show:][::-1]  # top eigenvalues, descending
            ax.plot(np.arange(1, len(ev) + 1), ev, 'o-', label=label, linewidth=1.5)
        ax.set_xlabel('Component', fontsize=11)
        ax.set_ylabel('Eigenvalue', fontsize=11)
        ax.set_title(cov_name, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.suptitle('Eigenvalue Spectra', fontsize=13)
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    print("Testing geometry analysis with synthetic data...")
    np.random.seed(42)

    K, M, T, N = 4, 40, 60, 20
    stim_ids = ['ori0', 'ori90', 'ori180', 'ori270']

    # Synthetic data: stimulus-driven signal + FEM noise
    rates_by_stim = {}
    for k, sid in enumerate(stim_ids):
        signal = np.zeros((1, T, N))
        signal[0, :, k] = 2.0 * np.sin(np.linspace(0, np.pi, T))
        noise = np.random.randn(M, T, N) * 0.5
        rates_by_stim[sid] = (signal + noise).astype(np.float32)

    cov_result = compute_covariances(rates_by_stim, mode='instantaneous')
    C_signal = cov_result['C_signal']
    C_FEM = cov_result['C_FEM']

    print(f"C_signal shape: {C_signal.shape}")
    print(f"C_signal min eigenvalue: {cov_result['eigvals_signal'].min():.4e}")
    print(f"C_FEM min eigenvalue: {cov_result['eigvals_fem'].min():.4e}")

    assert cov_result['eigvals_signal'].min() > -1e-6, "C_signal should be PSD"
    assert cov_result['eigvals_fem'].min() > -1e-6, "C_FEM should be PSD"
    print("PSD checks: OK")

    alpha, alpha_chance = alignment_fraction(C_signal, C_FEM, d=5)
    print(f"Alignment fraction α = {alpha:.4f} (chance = {alpha_chance:.4f})")

    snr = compute_subspace_snr(C_signal, C_FEM, d=5)
    print(f"Subspace SNR = {snr:.4f}")

    rates_cleaned, directions = representational_intervention(
        rates_by_stim, cov_result, d_remove=3
    )
    print(f"Cleaned rates shape: {rates_cleaned['ori0'].shape}")
    assert rates_cleaned['ori0'].shape == (M, T, N), "Shape mismatch after intervention"
    print("Representational intervention: OK")

    fig = plot_eigenspectra(
        {'real': cov_result},
        n_show=10
    )
    fig.savefig('test_geometry.png', dpi=100, bbox_inches='tight')
    print("Saved test_geometry.png")
    print("All tests passed!")
