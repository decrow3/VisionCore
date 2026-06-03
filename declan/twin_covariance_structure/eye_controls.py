from __future__ import annotations

import numpy as np


def occupancy_matched_shuffle(eye_positions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle time order while preserving occupancy samples."""
    ep = np.asarray(eye_positions, dtype=np.float64)
    idx = np.arange(ep.shape[0])
    rng.shuffle(idx)
    return ep[idx]


def amplitude_matched_gaussian(
    eye_positions: np.ndarray,
    rng: np.random.Generator,
    isotropic: bool = True,
) -> np.ndarray:
    """
    Match RMS amplitude, replace occupancy shape with Gaussian cloud.
    """
    ep = np.asarray(eye_positions, dtype=np.float64)
    center = np.mean(ep, axis=0, keepdims=True)
    centered = ep - center
    if isotropic:
        rms = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
        out = rng.normal(0.0, rms / np.sqrt(2.0), size=ep.shape)
    else:
        std = np.std(centered, axis=0, ddof=0)
        out = rng.normal(0.0, std, size=ep.shape)
    return out + center


def x_only(eye_positions: np.ndarray) -> np.ndarray:
    ep = np.asarray(eye_positions, dtype=np.float64)
    out = ep.copy()
    out[:, 1] = np.mean(ep[:, 1])
    return out


def y_only(eye_positions: np.ndarray) -> np.ndarray:
    ep = np.asarray(eye_positions, dtype=np.float64)
    out = ep.copy()
    out[:, 0] = np.mean(ep[:, 0])
    return out


def line_random_angle(eye_positions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Project occupancy onto a random 1D axis through the cloud center."""
    ep = np.asarray(eye_positions, dtype=np.float64)
    center = np.mean(ep, axis=0, keepdims=True)
    centered = ep - center
    theta = rng.uniform(0.0, 2.0 * np.pi)
    u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    coeff = centered @ u
    projected = coeff[:, None] * u[None, :]
    return projected + center


def isotropic_radius_rescale(eye_positions: np.ndarray, scale: float) -> np.ndarray:
    ep = np.asarray(eye_positions, dtype=np.float64)
    center = np.mean(ep, axis=0, keepdims=True)
    return center + (ep - center) * float(scale)
