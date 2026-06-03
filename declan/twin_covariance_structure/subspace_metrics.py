from __future__ import annotations

import numpy as np


def projection_matrix(U: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=np.float64)
    return U @ U.T


def _orthonormal_columns(U: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    U = np.asarray(U, dtype=np.float64)
    if U.size == 0:
        return np.zeros((U.shape[0], 0), dtype=np.float64)
    Q, R = np.linalg.qr(U)
    keep = np.abs(np.diag(R)) > eps
    return Q[:, keep]


def subspace_overlap(U: np.ndarray, V: np.ndarray, eps: float = 1e-12) -> float:
    """
    trace(P_U P_V) / min(dim(U), dim(V)) in [0, 1].
    """
    Qu = _orthonormal_columns(U, eps=eps)
    Qv = _orthonormal_columns(V, eps=eps)
    du = Qu.shape[1]
    dv = Qv.shape[1]
    if du == 0 or dv == 0:
        return float("nan")
    Pu = Qu @ Qu.T
    Pv = Qv @ Qv.T
    return float(np.trace(Pu @ Pv) / min(du, dv))


def variance_captured(C: np.ndarray, U: np.ndarray, eps: float = 1e-12) -> float:
    """trace(U.T C U) / trace(C)."""
    x = np.asarray(C, dtype=np.float64)
    x = 0.5 * (x + x.T)
    Q = _orthonormal_columns(U)
    if Q.shape[1] == 0:
        return 0.0
    num = float(np.trace(Q.T @ x @ Q))
    den = float(np.trace(x)) + eps
    return num / den


def directional_variance_capture(C_source: np.ndarray, U_target: np.ndarray) -> float:
    return variance_captured(C_source, U_target)


def principal_angles(U: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Qu = _orthonormal_columns(U)
    Qv = _orthonormal_columns(V)
    if Qu.shape[1] == 0 or Qv.shape[1] == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    s = np.linalg.svd(Qu.T @ Qv, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    angles = np.arccos(s)
    return angles, s
