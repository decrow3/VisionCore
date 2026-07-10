"""Shared feature-recovery scores for Figure 4 observer analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


R2_CV_METHOD = "pooled_multioutput_out_of_fold_sse_sst_train_mean_baseline"


@dataclass(frozen=True)
class PooledR2Result:
    """Pooled multi-output R2 summary."""

    r2: float
    sse: float
    sst: float
    n_samples: int
    n_outputs: int
    method: str = R2_CV_METHOD

    def as_dict(self, *, prefix: str = "") -> dict[str, Any]:
        return {
            f"{prefix}r2": self.r2,
            f"{prefix}sse": self.sse,
            f"{prefix}sst": self.sst,
            f"{prefix}n_samples": self.n_samples,
            f"{prefix}n_outputs": self.n_outputs,
            f"{prefix}method": self.method,
        }


def _as_2d(arr: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim == 1:
        out = out[:, None]
    if out.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array, got shape {out.shape}")
    return out


def _as_train_mean(train_mean: np.ndarray | float, n_outputs: int) -> np.ndarray:
    mean = np.asarray(train_mean, dtype=np.float64)
    if mean.ndim == 0:
        mean = np.full((1, int(n_outputs)), float(mean), dtype=np.float64)
    elif mean.ndim == 1:
        mean = mean[None, :]
    elif mean.ndim == 2 and mean.shape[0] == 1:
        mean = mean.astype(np.float64, copy=False)
    else:
        raise ValueError(f"train_mean must be scalar, 1D, or shape (1, d); got {mean.shape}")
    if mean.shape[1] != int(n_outputs):
        raise ValueError(f"train_mean output dimension {mean.shape[1]} does not match {n_outputs}")
    return mean


def per_sample_sse_sst(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    train_mean: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-sample SSE/SST for the locked train-baseline R2 convention.

    SSE and SST are summed across output dimensions for each sample. The
    returned validity mask is false for rows containing non-finite true,
    predicted, or train-mean values.
    """

    true = _as_2d(y_true, name="y_true")
    pred = _as_2d(y_pred, name="y_pred")
    if pred.shape != true.shape and pred.size == true.size:
        pred = pred.reshape(true.shape)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: y_true={true.shape}, y_pred={pred.shape}")
    mean = _as_train_mean(train_mean, true.shape[1])
    valid = np.isfinite(true).all(axis=1) & np.isfinite(pred).all(axis=1) & np.isfinite(mean).all()
    sse = np.full(true.shape[0], np.nan, dtype=np.float64)
    sst = np.full(true.shape[0], np.nan, dtype=np.float64)
    if np.any(valid):
        diff = true[valid] - pred[valid]
        baseline = true[valid] - mean
        sse[valid] = np.sum(diff * diff, axis=1)
        sst[valid] = np.sum(baseline * baseline, axis=1)
    return sse, sst, valid


def pooled_multioutput_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    train_mean: np.ndarray | float,
) -> PooledR2Result:
    """Compute pooled multi-output R2 using a train-fold mean baseline."""

    true = _as_2d(y_true, name="y_true")
    pred = _as_2d(y_pred, name="y_pred")
    sse_by_row, sst_by_row, valid = per_sample_sse_sst(true, pred, train_mean=train_mean)
    sse = float(np.nansum(sse_by_row[valid])) if np.any(valid) else float("nan")
    sst = float(np.nansum(sst_by_row[valid])) if np.any(valid) else float("nan")
    r2 = float(1.0 - sse / sst) if np.isfinite(sst) and sst > 1e-12 else float("nan")
    return PooledR2Result(
        r2=r2,
        sse=sse,
        sst=sst,
        n_samples=int(np.sum(valid)),
        n_outputs=int(true.shape[1]),
    )


def pooled_multioutput_r2_from_sse_sst(
    sse: np.ndarray | list[float],
    sst: np.ndarray | list[float],
    *,
    n_outputs: int | None = None,
) -> PooledR2Result:
    """Aggregate already-computed per-fold or per-row SSE/SST values."""

    sse_arr = np.asarray(sse, dtype=np.float64)
    sst_arr = np.asarray(sst, dtype=np.float64)
    if sse_arr.shape != sst_arr.shape:
        raise ValueError(f"SSE/SST shape mismatch: {sse_arr.shape} vs {sst_arr.shape}")
    valid = np.isfinite(sse_arr) & np.isfinite(sst_arr)
    total_sse = float(np.sum(sse_arr[valid])) if np.any(valid) else float("nan")
    total_sst = float(np.sum(sst_arr[valid])) if np.any(valid) else float("nan")
    r2 = float(1.0 - total_sse / total_sst) if np.isfinite(total_sst) and total_sst > 1e-12 else float("nan")
    return PooledR2Result(
        r2=r2,
        sse=total_sse,
        sst=total_sst,
        n_samples=int(np.sum(valid)),
        n_outputs=int(n_outputs) if n_outputs is not None else -1,
    )
