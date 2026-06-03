from __future__ import annotations

import numpy as np


def center_response(R: np.ndarray, axis: int) -> np.ndarray:
    """Mean-center a response array along a given axis."""
    arr = np.asarray(R, dtype=np.float64)
    return arr - np.mean(arr, axis=axis, keepdims=True)


def compute_cfem_for_image(R: np.ndarray, return_per_t: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Compute deterministic reafferent covariance for one image.

    Parameters
    ----------
    R : array, shape (n_eye, n_time, n_units)
        Model responses under multiple eye samples/traces.
    return_per_t : bool
        Whether to return per-time covariance matrices.

    Returns
    -------
    C : array, shape (n_units, n_units)
        C_FEM = E_t[Cov_eye(r | t)]
    per_t_covs : array or None
        shape (n_time, n_units, n_units) if requested, else None.
    """
    arr = np.asarray(R, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Expected R with ndim=3, got shape {arr.shape}")

    n_eye, n_time, n_units = arr.shape
    if n_eye < 2:
        raise ValueError("Need at least 2 eye samples to compute covariance")

    per_t_covs = np.zeros((n_time, n_units, n_units), dtype=np.float64)
    for t in range(n_time):
        x = arr[:, t, :]
        x = x - np.mean(x, axis=0, keepdims=True)
        denom = max(n_eye - 1, 1)
        c_t = (x.T @ x) / denom
        per_t_covs[t] = 0.5 * (c_t + c_t.T)

    C = np.mean(per_t_covs, axis=0)
    C = 0.5 * (C + C.T)
    return C, per_t_covs if return_per_t else None


def compute_signal_covariance(mu_images: np.ndarray) -> np.ndarray:
    """
    Covariance across image means.

    Parameters
    ----------
    mu_images : array, shape (n_images, n_units)

    Returns
    -------
    C_signal : array, shape (n_units, n_units)
    """
    x = np.asarray(mu_images, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D mu_images, got shape {x.shape}")
    x = x - np.mean(x, axis=0, keepdims=True)
    denom = max(x.shape[0] - 1, 1)
    C = (x.T @ x) / denom
    return 0.5 * (C + C.T)


def eigensystem(C: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric eigendecomposition in descending order."""
    x = np.asarray(C, dtype=np.float64)
    x = 0.5 * (x + x.T)
    evals, evecs = np.linalg.eigh(x)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    evals[np.abs(evals) < eps] = 0.0
    return evals, evecs


def participation_ratio(evals: np.ndarray, eps: float = 1e-12) -> float:
    """PR = (sum lambda)^2 / sum(lambda^2)."""
    lam = np.asarray(evals, dtype=np.float64)
    lam = lam[lam > 0]
    if lam.size == 0:
        return 0.0
    num = float(np.sum(lam) ** 2)
    den = float(np.sum(lam ** 2)) + eps
    return num / den


def top_subspace(evecs: np.ndarray, k: int) -> np.ndarray:
    """Return first k eigenvectors as a basis."""
    if k <= 0:
        return np.zeros((evecs.shape[0], 0), dtype=np.float64)
    k = min(k, evecs.shape[1])
    return np.asarray(evecs[:, :k], dtype=np.float64)
