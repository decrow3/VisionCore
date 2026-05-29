#!/usr/bin/env python3
"""
Step 2: Real-data scalar bridge.

For each fixRSVP session, test whether the model-predicted Jacobian FEM drive

    g_t = tr(J_t  Sigma_eye,t  J_t^T)

predicts empirical FEM-dependent neural variance across images:

    E_FEM,t = tr(B_emp,t  Sigma_eye,t  B_emp,t^T)

where B_emp,t is the (NC x 2) empirical eye-sensitivity matrix estimated by OLS:
    R_centered ~ E_centered @ B_emp^T.

Analysis unit: per natural-image identity — all valid fixation samples for that
image pooled across trials and time bins.  No phase or radius sub-binning.

Decision gate (handoff doc):
  Win:   g_t predicts E_FEM above controls and above image-shuffled null.
  Partial: model tr(C_model_FEM) predicts E_FEM even if g_t alone is noisy.
  Fail:  prediction fully explained by eye amplitude, Jacobian norm, or mean rate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval.eval_stack_multidataset import load_model
from eval.eval_stack_utils import run_model
from eval.fixrsvp import get_fixrsvp_data
from VisionCore.paths import FIGURES_DIR


ROOT_OUTPUT_DIR = Path("outputs/jacobian_predictive_framework")
DEFAULT_JACOBIAN_STEP_PX = 0.5
REFERENCE_RATE_HZ = 240.0
DEFAULT_N_SPLIT_REPEATS = 100


# ---------------------------------------------------------------------------
# Stimulus helpers (duplicated from run_fixrsvp_steps01 to stay self-contained)
# ---------------------------------------------------------------------------

def _normalize_stim_dims(stim: np.ndarray) -> torch.Tensor:
    t = torch.as_tensor(stim, dtype=torch.float32)
    if t.dim() == 4:
        t = t.unsqueeze(1)
    if t.dim() != 5:
        raise ValueError(f"Expected 4D or 5D stim, got {tuple(t.shape)}")
    return t


def _shift_stimulus_batch(stim: torch.Tensor, displacements_px: np.ndarray) -> torch.Tensor:
    batch, channels, lags, height, width = stim.shape
    merged = stim.reshape(batch, channels * lags, height, width)
    ys = torch.linspace(-1.0, 1.0, height, device=stim.device, dtype=stim.dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=stim.device, dtype=stim.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = (
        torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    )
    d = torch.as_tensor(displacements_px, device=stim.device, dtype=stim.dtype)
    base_grid[..., 0] -= 2.0 * d[:, 0, None, None] / max(width - 1, 1)
    base_grid[..., 1] -= 2.0 * d[:, 1, None, None] / max(height - 1, 1)
    shifted = F.grid_sample(merged, base_grid, mode="bilinear",
                            padding_mode="zeros", align_corners=True)
    return shifted.reshape(batch, channels, lags, height, width)


def _shift_stimulus(stim: torch.Tensor, dx_px: float, dy_px: float) -> torch.Tensor:
    return _shift_stimulus_batch(stim, np.array([[dx_px, dy_px]], dtype=np.float64))


def _predict_responses(model, stim: torch.Tensor, dataset_idx: int,
                       batch_size: int = 64) -> np.ndarray:
    outputs = []
    for start in range(0, stim.shape[0], batch_size):
        batch = run_model(model, {"stim": stim[start: start + batch_size]},
                          dataset_idx=dataset_idx)
        outputs.append(batch["rhat"].detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float64)


def _compute_jacobian(model, stim: torch.Tensor, dataset_idx: int,
                      step_px: float) -> np.ndarray:
    """Return (n_neurons, 2) Jacobian via central finite differences."""
    jx = (
        _predict_responses(model, _shift_stimulus(stim,  step_px, 0.0), dataset_idx)[0]
        - _predict_responses(model, _shift_stimulus(stim, -step_px, 0.0), dataset_idx)[0]
    ) / (2.0 * step_px)
    jy = (
        _predict_responses(model, _shift_stimulus(stim, 0.0,  step_px), dataset_idx)[0]
        - _predict_responses(model, _shift_stimulus(stim, 0.0, -step_px), dataset_idx)[0]
    ) / (2.0 * step_px)
    return np.stack((jx, jy), axis=-1)  # (NC, 2)


def _choose_baseline(stim: torch.Tensor, eyepos: np.ndarray
                     ) -> tuple[torch.Tensor, np.ndarray]:
    """Return (baseline_stim, baseline_eye_deg) = sample nearest median eye."""
    center = np.nanmedian(eyepos, axis=0)
    idx = int(np.nanargmin(np.linalg.norm(eyepos - center[None, :], axis=1)))
    return stim[idx: idx + 1].clone(), eyepos[idx].copy()


def _resolve_pixels_per_degree(data: dict) -> float:
    ds = data.get("dataset")
    if ds is not None:
        try:
            dset_idx = int(ds.inds[:, 0].unique().item())
            ppd = ds.dsets[dset_idx].metadata.get("pixels_per_degree")
            if ppd is not None:
                return float(ppd)
        except Exception:
            pass
    return 37.5


# ---------------------------------------------------------------------------
# Per-image window collection
# ---------------------------------------------------------------------------

class ImageWindow:
    __slots__ = ("image_id", "trial_indices", "time_indices",
                 "stim", "eyepos_deg", "robs_rates")

    def __init__(self, image_id: int, trial_indices: np.ndarray,
                 time_indices: np.ndarray, stim: torch.Tensor,
                 eyepos_deg: np.ndarray, robs_rates: np.ndarray) -> None:
        self.image_id = image_id
        self.trial_indices = trial_indices
        self.time_indices = time_indices
        self.stim = stim
        self.eyepos_deg = eyepos_deg
        self.robs_rates = robs_rates  # (N, NC) spike rates


def _collect_image_windows(data: dict, min_samples: int, dt: float) -> list[ImageWindow]:
    robs_raw: np.ndarray = data["robs"]           # (NT, T, NC) spike counts
    eyepos_raw: np.ndarray = data["eyepos"]       # (NT, T, 2) degrees
    image_ids_raw: np.ndarray = data["image_ids"] # (NT, T)
    stim_raw: np.ndarray = data["stim"]           # (NT, T, ...)
    dfs_raw: np.ndarray | None = data.get("dfs")  # (NT, T, NC)

    NC = robs_raw.shape[2]
    # Primary validity: eye position must be finite and image must be valid.
    # Neurons are allowed to have NaN — handled per-neuron in the regression.
    valid = (
        np.isfinite(eyepos_raw).all(axis=2)
        & (image_ids_raw >= 0)
    )

    windows: list[ImageWindow] = []
    for img_id in np.unique(image_ids_raw[valid]):
        mask = valid & (image_ids_raw == img_id)
        tri, ti = np.where(mask)
        if len(tri) < min_samples:
            continue
        stim_img = _normalize_stim_dims(stim_raw[tri, ti])
        eyepos_img = eyepos_raw[tri, ti].astype(np.float64)
        robs_img = robs_raw[tri, ti].astype(np.float64) / dt  # spike rates
        if dfs_raw is not None:
            dfs_img = dfs_raw[tri, ti].astype(np.float64)
            robs_img = np.where(dfs_img > 0, robs_img, np.nan)
        windows.append(ImageWindow(
            image_id=int(img_id),
            trial_indices=tri,
            time_indices=ti,
            stim=stim_img,
            eyepos_deg=eyepos_img,
            robs_rates=robs_img,
        ))
    return windows


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _empirical_fem_drive(
    robs: np.ndarray,       # (N, NC) spike rates
    eyepos_px: np.ndarray,  # (N, 2)  centered on baseline
    sigma_eye: np.ndarray,  # (2, 2)
    min_samples: int = 20,
) -> dict:
    """
    OLS: R_c ~ E_c @ B^T  →  B is (NC, 2)
    E_FEM = tr(B @ sigma_eye @ B^T)

    Also reports B norm and median per-neuron R^2.
    """
    N, NC = robs.shape
    if N < min_samples:
        return {"e_fem": float("nan"), "b_emp_norm": float("nan"),
                "r2_eye_median": float("nan"), "n_valid_neurons": 0}
    eye_c = eyepos_px - eyepos_px.mean(axis=0, keepdims=True)

    # Fit each neuron separately so NaN responses don't contaminate others.
    B = np.zeros((NC, 2), dtype=np.float64)
    r2_per_neuron = np.full(NC, float("nan"))
    n_valid = 0
    for ni in range(NC):
        col = robs[:, ni]
        ok = np.isfinite(col)
        if ok.sum() < min_samples:
            continue
        col_c = col[ok] - col[ok].mean()
        x = eye_c[ok]
        b_i, _, _, _ = np.linalg.lstsq(x, col_c, rcond=None)  # (2,)
        B[ni] = b_i
        ss_tot = float(col_c @ col_c)
        if ss_tot > 1e-12:
            resid = col_c - x @ b_i
            r2_per_neuron[ni] = 1.0 - float(resid @ resid) / ss_tot
        n_valid += 1

    if n_valid == 0:
        return {"e_fem": float("nan"), "b_emp_norm": float("nan"),
                "r2_eye_median": float("nan"), "n_valid_neurons": 0}

    e_fem = float(np.trace(B @ sigma_eye @ B.T))
    return {
        "e_fem": e_fem,
        "b_emp_norm": float(np.linalg.norm(B, "fro")),
        "r2_eye_median": float(np.nanmedian(np.clip(r2_per_neuron, 0.0, 1.0))),
        "n_valid_neurons": n_valid,
    }


def _alignment_score(U1: np.ndarray, U2: np.ndarray) -> float:
    Q1, _ = np.linalg.qr(U1)
    Q2, _ = np.linalg.qr(U2)
    singular_values = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(np.mean(np.clip(singular_values, 0.0, 1.0) ** 2))


def _capture_fraction(U: np.ndarray, cov: np.ndarray) -> float:
    Q, _ = np.linalg.qr(U)
    return float(np.trace(Q.T @ cov @ Q) / (np.trace(cov) + 1e-12))


def _fit_empirical_B(
    robs: np.ndarray,
    eyepos_px: np.ndarray,
    min_samples: int = 10,
) -> tuple[np.ndarray, int]:
    N, NC = robs.shape
    if N < min_samples:
        return np.zeros((NC, 2), dtype=np.float64), 0

    eye_c = eyepos_px - eyepos_px.mean(axis=0, keepdims=True)
    B = np.zeros((NC, 2), dtype=np.float64)
    n_valid = 0
    for ni in range(NC):
        col = robs[:, ni]
        ok = np.isfinite(col)
        if ok.sum() < min_samples:
            continue
        col_c = col[ok] - col[ok].mean()
        b_i, _, _, _ = np.linalg.lstsq(eye_c[ok], col_c, rcond=None)
        B[ni] = b_i
        n_valid += 1
    return B, n_valid


def _eye_permuted_empirical_fem_null(
    robs: np.ndarray,
    eyepos_px: np.ndarray,
    sigma_eye: np.ndarray,
    min_samples: int,
    n_shuffles: int,
    rng: np.random.Generator,
) -> float:
    N, _ = robs.shape
    if N < min_samples:
        return float("nan")

    eye_c_full = eyepos_px - eyepos_px.mean(axis=0, keepdims=True)
    shuf_vals: list[float] = []
    for _ in range(n_shuffles):
        perm = rng.permutation(N)
        B_s, n_valid = _fit_empirical_B(robs, eye_c_full[perm], min_samples=min_samples)
        if n_valid == 0:
            continue
        shuf_vals.append(float(np.trace(B_s @ sigma_eye @ B_s.T)))
    return float(np.nanmedian(shuf_vals)) if shuf_vals else float("nan")


def _principal_angles_deg(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    Q1, _ = np.linalg.qr(U1)
    Q2, _ = np.linalg.qr(U2)
    singular_values = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    singular_values = np.clip(singular_values, -1.0, 1.0)
    return np.degrees(np.arccos(singular_values))


def _stimulus_locked_variance(
    robs: np.ndarray,
    time_indices: np.ndarray,
    min_bins: int = 2,
) -> dict:
    unique_bins = np.unique(time_indices)
    if unique_bins.size < min_bins:
        return {
            "stimulus_locked_variance": float("nan"),
            "stimulus_locked_n_bins": int(unique_bins.size),
        }

    psth = []
    for time_bin in unique_bins:
        mask = time_indices == time_bin
        if int(mask.sum()) == 0:
            continue
        psth.append(np.nanmean(robs[mask], axis=0))
    if len(psth) < min_bins:
        return {
            "stimulus_locked_variance": float("nan"),
            "stimulus_locked_n_bins": len(psth),
        }
    psth_arr = np.asarray(psth, dtype=np.float64)
    stim_locked = float(np.nansum(np.nanvar(psth_arr, axis=0, ddof=1)))
    return {
        "stimulus_locked_variance": stim_locked,
        "stimulus_locked_n_bins": int(psth_arr.shape[0]),
    }


def _model_fem_drive(J: np.ndarray, sigma_eye: np.ndarray) -> dict:
    if not np.isfinite(J).all():
        return {"g_t": float("nan"), "jacobian_fro_norm": float("nan"),
                "jacobian_rank_ratio": float("nan")}
    g_t = float(np.trace(J @ sigma_eye @ J.T))
    jac_fro = float(np.linalg.norm(J, "fro"))
    try:
        sv = np.linalg.svd(J, compute_uv=False)
        rank_ratio = float(sv[1] / sv[0]) if sv[0] > 1e-12 else float("nan")
    except np.linalg.LinAlgError:
        rank_ratio = float("nan")
    return {"g_t": g_t, "jacobian_fro_norm": jac_fro,
            "jacobian_rank_ratio": rank_ratio}


def _stim_stats(stim_np: np.ndarray) -> dict:
    frame = stim_np
    while frame.ndim > 2:
        frame = frame[-1] if frame.shape[0] > 1 else frame[0]
    rms = float(np.sqrt(np.mean(frame ** 2)))
    gx = np.diff(frame, axis=1)
    gy = np.diff(frame, axis=0)
    grad_energy = float(np.mean(gx ** 2) + np.mean(gy ** 2))
    return {"stim_rms": rms, "stim_grad_energy": grad_energy}


# ---------------------------------------------------------------------------
# Nonlinear model FEM covariance trace
# ---------------------------------------------------------------------------

def _model_fem_trace_nonlinear(
    model,
    baseline_stim: torch.Tensor,  # (1, C, L, H, W)
    eyepos_px: np.ndarray,         # (N, 2) centered
    dataset_idx: int,
    rng: np.random.Generator,
    max_samples: int = 200,
    min_samples: int = 8,
) -> dict:
    """
    Compute trace(Cov_s[r(I + Δp_s)]) by shifting the baseline stimulus by
    each observed eye offset and running the model.

    This is the nonlinear analogue of g_t = tr(J Σ_eye J^T): it does not use
    the linearization assumption, so it succeeds even when the eye-position
    cloud spans many pixels.
    """
    N = len(eyepos_px)
    nan_result = {"trace_cov_model_fem": float("nan"), "n_model_trace_samples": 0}
    if N > max_samples:
        idx = rng.choice(N, max_samples, replace=False)
        offsets = eyepos_px[idx]
    else:
        offsets = eyepos_px.copy()

    M = len(offsets)
    if M < min_samples:
        return nan_result

    # Shift the baseline stim by each offset and run through model.
    stim_rep = baseline_stim.repeat(M, 1, 1, 1, 1)  # (M, C, L, H, W)
    shifted = _shift_stimulus_batch(stim_rep, offsets.astype(np.float64))

    try:
        resp = _predict_responses(model, shifted, dataset_idx)  # (M, NC)
    except Exception:
        return nan_result

    if not np.isfinite(resp).all():
        resp = np.where(np.isfinite(resp), resp, np.nan)

    # trace(Cov) = sum of per-neuron sample variances (ddof=1).
    var_per_neuron = np.nanvar(resp, axis=0, ddof=1)  # (NC,)
    trace = float(np.nansum(var_per_neuron))
    return {"trace_cov_model_fem": trace, "n_model_trace_samples": M}


# ---------------------------------------------------------------------------
# Split-half B_emp reliability
# ---------------------------------------------------------------------------

def _split_half_fem_drive(
    robs: np.ndarray,            # (N, NC) spike rates, may contain NaN
    eyepos_px: np.ndarray,       # (N, 2) eye offsets in px (not pre-centered)
    trial_indices: np.ndarray,   # (N,) integer trial index per sample
    sigma_eye: np.ndarray,       # (2, 2) from full window
    min_samples: int = 10,
    n_shuffles: int = 10,
    n_split_repeats: int = DEFAULT_N_SPLIT_REPEATS,
    rng: np.random.Generator | None = None,
    return_split_bundles: bool = False,
) -> dict:
    """
    Split-half validation of B_emp and an eye-permutation shuffle for E_FEM.

    Split is by trial (odd/even alternating) to avoid temporal autocorrelation
    within a trial. Reports:
      e_fem_cv_*         : repeated symmetric cross-validated FEM drive summary
      b_split_corr_*     : repeated vec(B_A) vs vec(B_B) Pearson correlation summary
      e_fem_shuffle_med  : median E_FEM under within-image eye permutation
    """
    nan_result = {
        "e_fem_cv": float("nan"),
        "e_fem_cv_mean": float("nan"),
        "e_fem_cv_median": float("nan"),
        "e_fem_cv_std": float("nan"),
        "e_fem_cv_ci_low": float("nan"),
        "e_fem_cv_ci_high": float("nan"),
        "b_split_corr": float("nan"),
        "b_split_corr_mean": float("nan"),
        "b_split_corr_median": float("nan"),
        "b_split_corr_ci_low": float("nan"),
        "b_split_corr_ci_high": float("nan"),
        "n_split_repeats_valid": 0,
        "emp_split_alignment_mean": float("nan"),
        "emp_split_alignment_median": float("nan"),
        "emp_split_principal_angle_1_median_deg": float("nan"),
        "emp_split_principal_angle_2_median_deg": float("nan"),
        "emp_split_variance_capture_median": float("nan"),
        "e_fem_shuffle_med": float("nan"),
    }
    N, NC = robs.shape
    if N < 2 * min_samples:
        return nan_result
    if rng is None:
        rng = np.random.default_rng()

    unique_trials = np.unique(trial_indices)
    e_vals: list[float] = []
    b_corr_vals: list[float] = []
    align_vals: list[float] = []
    capture_vals: list[float] = []
    angle1_vals: list[float] = []
    angle2_vals: list[float] = []
    split_a: list[np.ndarray] = []
    split_b: list[np.ndarray] = []

    for _ in range(n_split_repeats):
        if len(unique_trials) >= 2:
            perm_trials = rng.permutation(unique_trials)
            half = len(perm_trials) // 2
            if half == 0 or half == len(perm_trials):
                continue
            trials_a = perm_trials[:half]
            trials_b = perm_trials[half:]
            idx_a = np.where(np.isin(trial_indices, trials_a))[0]
            idx_b = np.where(np.isin(trial_indices, trials_b))[0]
        else:
            perm = rng.permutation(N)
            idx_a, idx_b = perm[: N // 2], perm[N // 2 :]

        if len(idx_a) < min_samples or len(idx_b) < min_samples:
            continue

        B_a, n_valid_a = _fit_empirical_B(robs[idx_a], eyepos_px[idx_a], min_samples=min_samples)
        B_b, n_valid_b = _fit_empirical_B(robs[idx_b], eyepos_px[idx_b], min_samples=min_samples)
        if n_valid_a == 0 or n_valid_b == 0:
            continue

        e_fem_cv = 0.5 * (
            float(np.trace(B_a @ sigma_eye @ B_b.T))
            + float(np.trace(B_b @ sigma_eye @ B_a.T))
        )
        va, vb = B_a.ravel(), B_b.ravel()
        b_split_corr = (
            float(np.corrcoef(va, vb)[0, 1])
            if np.std(va) > 1e-12 and np.std(vb) > 1e-12
            else float("nan")
        )
        cov_b = B_b @ B_b.T
        angles = _principal_angles_deg(B_a, B_b)

        e_vals.append(e_fem_cv)
        b_corr_vals.append(b_split_corr)
        align_vals.append(_alignment_score(B_a, B_b))
        capture_vals.append(_capture_fraction(B_a, cov_b))
        angle1_vals.append(float(angles[0]) if angles.size >= 1 else float("nan"))
        angle2_vals.append(float(angles[1]) if angles.size >= 2 else float("nan"))
        if return_split_bundles:
            split_a.append(B_a)
            split_b.append(B_b)

    if not e_vals:
        return nan_result

    e_arr = np.asarray(e_vals, dtype=np.float64)
    b_arr = np.asarray(b_corr_vals, dtype=np.float64)
    align_arr = np.asarray(align_vals, dtype=np.float64)
    capture_arr = np.asarray(capture_vals, dtype=np.float64)
    angle1_arr = np.asarray(angle1_vals, dtype=np.float64)
    angle2_arr = np.asarray(angle2_vals, dtype=np.float64)

    result = {
        "e_fem_cv": float(np.nanmedian(e_arr)),
        "e_fem_cv_mean": float(np.nanmean(e_arr)),
        "e_fem_cv_median": float(np.nanmedian(e_arr)),
        "e_fem_cv_std": float(np.nanstd(e_arr, ddof=1)) if e_arr.size > 1 else 0.0,
        "e_fem_cv_ci_low": float(np.nanpercentile(e_arr, 2.5)),
        "e_fem_cv_ci_high": float(np.nanpercentile(e_arr, 97.5)),
        "b_split_corr": float(np.nanmedian(b_arr)),
        "b_split_corr_mean": float(np.nanmean(b_arr)),
        "b_split_corr_median": float(np.nanmedian(b_arr)),
        "b_split_corr_ci_low": float(np.nanpercentile(b_arr, 2.5)),
        "b_split_corr_ci_high": float(np.nanpercentile(b_arr, 97.5)),
        "n_split_repeats_valid": int(e_arr.size),
        "emp_split_alignment_mean": float(np.nanmean(align_arr)),
        "emp_split_alignment_median": float(np.nanmedian(align_arr)),
        "emp_split_principal_angle_1_median_deg": float(np.nanmedian(angle1_arr)),
        "emp_split_principal_angle_2_median_deg": float(np.nanmedian(angle2_arr)),
        "emp_split_variance_capture_median": float(np.nanmedian(capture_arr)),
        "e_fem_shuffle_med": _eye_permuted_empirical_fem_null(
            robs=robs,
            eyepos_px=eyepos_px,
            sigma_eye=sigma_eye,
            min_samples=min_samples,
            n_shuffles=n_shuffles,
            rng=rng,
        ),
    }
    if return_split_bundles:
        result["_split_bundle"] = {
            "B_a": np.asarray(split_a, dtype=np.float64),
            "B_b": np.asarray(split_b, dtype=np.float64),
        }
    return result


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata

    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 4:
        return float("nan")
    xr = rankdata(x[mask]).astype(np.float64)
    yr = rankdata(y[mask]).astype(np.float64)
    if np.std(xr) <= 1e-12 or np.std(yr) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _residualize_vector(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return y - np.nanmean(y)
    design = np.column_stack([np.ones(X.shape[0], dtype=np.float64), X])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def _rank_residual_corr(x: np.ndarray, y: np.ndarray, covariates: np.ndarray) -> float:
    from scipy.stats import rankdata

    mask = np.isfinite(x) & np.isfinite(y)
    if covariates.size > 0:
        mask &= np.isfinite(covariates).all(axis=1)
    if int(mask.sum()) < max(4, covariates.shape[1] + 2 if covariates.ndim == 2 else 4):
        return float("nan")

    x_rank = rankdata(x[mask]).astype(np.float64)
    y_rank = rankdata(y[mask]).astype(np.float64)
    if covariates.size == 0:
        xr = x_rank - x_rank.mean()
        yr = y_rank - y_rank.mean()
    else:
        cov_rank = np.column_stack([
            rankdata(covariates[mask, idx]).astype(np.float64)
            for idx in range(covariates.shape[1])
        ])
        xr = _residualize_vector(x_rank, cov_rank)
        yr = _residualize_vector(y_rank, cov_rank)
    if np.std(xr) <= 1e-12 or np.std(yr) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _build_reliable_subset_summary(results: list[dict]) -> dict:
    trace_cov = np.array([row["trace_cov_model_fem"] for row in results], dtype=np.float64)
    e_cv_mean = np.array([row["e_fem_cv_mean"] for row in results], dtype=np.float64)
    e_cv_med = np.array([row["e_fem_cv_median"] for row in results], dtype=np.float64)
    b_split = np.array([row["b_split_corr_median"] for row in results], dtype=np.float64)
    e_fem = np.array([row["e_fem"] for row in results], dtype=np.float64)
    e_shuffle = np.array([row["e_fem_shuffle_med"] for row in results], dtype=np.float64)
    n_samples = np.array([row["n_samples"] for row in results], dtype=np.float64)

    sample_cut = float(np.nanpercentile(n_samples, 25)) if np.isfinite(n_samples).any() else float("nan")
    split_cut = float(np.nanpercentile(b_split, 25)) if np.isfinite(b_split).any() else float("nan")
    subset_masks = {
        "all_images": np.ones(len(results), dtype=bool),
        "b_split_corr_pos": b_split > 0.0,
        "e_fem_gt_shuffle": e_fem > e_shuffle,
        "both_criteria": (b_split > 0.0) & (e_fem > e_shuffle),
        "top75_n_samples": n_samples >= sample_cut,
        "top75_b_split": b_split >= split_cut,
    }

    summary: dict[str, dict] = {}
    for name, mask in subset_masks.items():
        summary[name] = {
            "n": int(mask.sum()),
            "rho_trace_vs_Ecv_mean": _spearman_corr(trace_cov[mask], e_cv_mean[mask]),
            "rho_trace_vs_Ecv_median": _spearman_corr(trace_cov[mask], e_cv_med[mask]),
            "median_trace_cov_model_fem": float(np.nanmedian(trace_cov[mask])) if mask.any() else float("nan"),
            "median_e_fem_cv_mean": float(np.nanmedian(e_cv_mean[mask])) if mask.any() else float("nan"),
            "median_e_fem_cv_median": float(np.nanmedian(e_cv_med[mask])) if mask.any() else float("nan"),
        }
    return summary


def _build_residualized_summary(results: list[dict]) -> dict:
    trace_cov = np.array([row["trace_cov_model_fem"] for row in results], dtype=np.float64)
    e_cv = np.array([row["e_fem_cv_mean"] for row in results], dtype=np.float64)
    eye_amp = np.array([row["eye_amplitude_px2"] for row in results], dtype=np.float64)
    mean_emp_rate = np.array([row["mean_emp_rate"] for row in results], dtype=np.float64)
    mean_model_rate = np.array([row["mean_model_rate"] for row in results], dtype=np.float64)
    stim_rms = np.array([row["stim_rms"] for row in results], dtype=np.float64)
    stim_grad = np.array([row["stim_grad_energy"] for row in results], dtype=np.float64)
    n_samples = np.array([row["n_samples"] for row in results], dtype=np.float64)
    stim_locked = np.array([row["stimulus_locked_variance"] for row in results], dtype=np.float64)
    model_resp_norm = np.array([row["model_baseline_response_norm"] for row in results], dtype=np.float64)

    basic_cov = np.column_stack([eye_amp, mean_emp_rate, mean_model_rate])
    full_cov = np.column_stack([
        eye_amp,
        mean_emp_rate,
        mean_model_rate,
        stim_rms,
        stim_grad,
        n_samples,
        stim_locked,
        model_resp_norm,
    ])
    return {
        "rho_trace_vs_Ecv_raw": _spearman_corr(trace_cov, e_cv),
        "rho_trace_vs_Ecv_resid_basic": _rank_residual_corr(trace_cov, e_cv, basic_cov),
        "rho_trace_vs_Ecv_resid_full": _rank_residual_corr(trace_cov, e_cv, full_cov),
        "rho_eye_amp_vs_Ecv": _spearman_corr(eye_amp, e_cv),
        "rho_rate_vs_Ecv": _spearman_corr(mean_emp_rate, e_cv),
        "rho_grad_vs_Ecv": _spearman_corr(stim_grad, e_cv),
        "rho_stimulus_locked_vs_Ecv": _spearman_corr(stim_locked, e_cv),
    }


# ---------------------------------------------------------------------------
# Image-shuffled null
# ---------------------------------------------------------------------------

def _compute_shuffled_g(
    results: list[dict],
    n_matches: int = 5,
) -> dict[int, float]:
    """
    For each image, compute g_t using Jacobians from matched other images.
    Matching on jacobian_fro_norm + stim_rms (same strategy as Step 1).
    Returns dict: image_id → median shuffled g_t.
    """
    ids = np.array([r["image_id"] for r in results])
    J_by_id = {r["image_id"]: r["_J"] for r in results}
    sigma_by_id = {r["image_id"]: r["_sigma_eye"] for r in results}
    fro = np.array([r["jacobian_fro_norm"] for r in results])
    rms = np.array([r["stim_rms"] for r in results])

    def _norm01(x: np.ndarray) -> np.ndarray:
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 0 else np.zeros_like(x)

    fro_n = _norm01(fro)
    rms_n = _norm01(rms)

    shuf_g: dict[int, float] = {}
    for i, row in enumerate(results):
        img_id = row["image_id"]
        sigma_i = sigma_by_id[img_id]
        dists = np.abs(fro_n - fro_n[i]) + np.abs(rms_n - rms_n[i])
        dists[ids == img_id] = np.inf
        candidate_ids = ids[np.argsort(dists)[:n_matches]]
        g_vals = [
            float(np.trace(J_by_id[int(cid)] @ sigma_i @ J_by_id[int(cid)].T))
            for cid in candidate_ids
            if int(cid) in J_by_id
        ]
        shuf_g[img_id] = float(np.nanmedian(g_vals)) if g_vals else float("nan")
    return shuf_g


# ---------------------------------------------------------------------------
# Main per-session analysis
# ---------------------------------------------------------------------------

def run_step2(
    subject: str,
    date: str,
    dataset_configs_path: str,
    output_dir: Path,
    checkpoint_path: str | None,
    dataset_idx: int,
    jacobian_step_px: float = DEFAULT_JACOBIAN_STEP_PX,
    min_samples: int = 50,
    n_shuffle_matches: int = 5,
    n_split_repeats: int = DEFAULT_N_SPLIT_REPEATS,
    residualized_controls: bool = False,
    reliable_subset_summary: bool = False,
    model_type: str | None = None,
    model_index: int | None = None,
    model_device: str = "cuda",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- data
    print(f"Loading fixRSVP data ({subject} {date})...")
    data = get_fixrsvp_data(
        subject=subject,
        date=date,
        dataset_configs_path=dataset_configs_path,
        use_cached_data=True,
    )
    pixels_per_degree = _resolve_pixels_per_degree(data)
    dt = 1.0 / REFERENCE_RATE_HZ

    windows = _collect_image_windows(data, min_samples=min_samples, dt=dt)
    if not windows:
        print(f"  No image windows with >= {min_samples} samples. Exiting.")
        return {"n_windows": 0}
    print(f"  {len(windows)} image windows  (pixels_per_degree = {pixels_per_degree:.3f})")

    # ---------------------------------------------------------------- model
    print("Loading model...")
    # model_config_dict override: pass the architecture config as a dict so load_model
    # does not need to resolve the stale relative path stored in old checkpoint hparams.
    _model_config_dict = None
    try:
        from models.config_loader import load_config as _load_cfg
        _model_config_dict = _load_cfg(
            "experiments/model_configs/learned_resnet_none_convgru_gaussian.yaml"
        )
    except Exception:
        pass

    # Pass the legacy config directly — it contains the session ordering and cids
    # that exactly match this checkpoint's readout architecture.
    load_kwargs: dict = {
        "model_type": model_type,
        "model_index": model_index,
        "cfg_dir_override": dataset_configs_path,
        "model_config_dict": _model_config_dict,
        "device": model_device,
    }
    if checkpoint_path is not None:
        load_kwargs["checkpoint_path"] = checkpoint_path
    model, _ = load_model(**load_kwargs)
    model.model.eval()

    # Auto-discover dataset_idx from session name if possible.
    # The session name (e.g. "Allen_2022-02-16") is stored in model.names.
    session_name = f"{subject}_{date}"
    if dataset_idx is not None and hasattr(model, "names"):
        try:
            _auto_idx = model.names.index(session_name)
            if _auto_idx != dataset_idx:
                print(
                    f"  Note: auto-detected dataset_idx={_auto_idx} for {session_name}"
                    f" (overrides --dataset-idx {dataset_idx})"
                )
                dataset_idx = _auto_idx
        except ValueError:
            print(f"  Warning: session '{session_name}' not in model.names; "
                  f"using dataset_idx={dataset_idx}")

    # Sanity: confirm dataset_idx maps to the expected session.
    if hasattr(model, "names") and dataset_idx < len(model.names):
        expected_nc = (
            model.model.readouts[dataset_idx].bias.shape[0]
            if hasattr(model.model, "readouts")
            else "?"
        )
        print(f"  dataset_idx={dataset_idx}  →  model.names[{dataset_idx}]="
              f"'{model.names[dataset_idx]}'  NC={expected_nc}")

    # Session-level RNG so each image gets independent but reproducible samples.
    session_rng = np.random.default_rng(seed=abs(hash(f"{subject}_{date}")) % (2**31))

    # ---------------------------------------------------------------- compute
    print("Computing per-image metrics...")
    results: list[dict] = []
    split_bundle_rows: list[dict] = []

    for win in windows:
        baseline_stim, baseline_eye_deg = _choose_baseline(win.stim, win.eyepos_deg)
        baseline_resp = _predict_responses(model, baseline_stim, dataset_idx)[0]

        eyepos_px = (win.eyepos_deg - baseline_eye_deg[None, :]) * pixels_per_degree
        sigma_eye = np.cov(eyepos_px.T).astype(np.float64)  # (2, 2)
        eye_amplitude = float(np.trace(sigma_eye))

        # Eye-radius percentiles — diagnose whether large offsets dominate.
        radii = np.linalg.norm(eyepos_px, axis=1)
        eye_radius_p50 = float(np.percentile(radii, 50))
        eye_radius_p90 = float(np.percentile(radii, 90))
        eye_radius_p99 = float(np.percentile(radii, 99))

        J = _compute_jacobian(model, baseline_stim, dataset_idx, jacobian_step_px)
        model_m = _model_fem_drive(J, sigma_eye)
        emp_m = _empirical_fem_drive(win.robs_rates, eyepos_px, sigma_eye)
        sh_m = _split_half_fem_drive(
            win.robs_rates, eyepos_px, win.trial_indices, sigma_eye,
            n_split_repeats=n_split_repeats,
            rng=session_rng,
            return_split_bundles=True,
        )
        ss = _stim_stats(baseline_stim.squeeze(0).cpu().numpy())
        stim_locked = _stimulus_locked_variance(win.robs_rates, win.time_indices)

        # Nonlinear model trace (does not use linearization)
        trace_m = _model_fem_trace_nonlinear(
            model, baseline_stim, eyepos_px, dataset_idx, session_rng
        )

        # Local g_t: Jacobian × Sigma_eye restricted to nearby samples.
        # Checks whether linearization-scale prediction (≤1 px, ≤2 px cloud)
        # predicts a local empirical FEM drive.
        local_g: dict = {}
        for thresh_px, col_g, col_n in [
            (1.0, "g_t_local_1px", "n_local_1px"),
            (2.0, "g_t_local_2px", "n_local_2px"),
        ]:
            mask = radii <= thresh_px
            n_local = int(mask.sum())
            local_g[col_n] = n_local
            if n_local >= 4 and np.isfinite(J).all():
                sig_loc = np.cov(eyepos_px[mask].T).astype(np.float64)
                local_g[col_g] = float(np.trace(J @ sig_loc @ J.T))
            else:
                local_g[col_g] = float("nan")

        row = {
            "image_id": win.image_id,
            "window_id": f"image_{win.image_id}",
            "n_samples": len(win.trial_indices),
            "pixels_per_degree": pixels_per_degree,
            "eye_amplitude_px2": eye_amplitude,
            "eye_radius_p50_px": eye_radius_p50,
            "eye_radius_p90_px": eye_radius_p90,
            "eye_radius_p99_px": eye_radius_p99,
            "mean_model_rate": float(np.mean(baseline_resp)),
            "mean_emp_rate": float(np.nanmean(win.robs_rates)),
            "model_baseline_response_norm": float(np.linalg.norm(baseline_resp)),
            **model_m,
            **trace_m,
            **local_g,
            **emp_m,
            **sh_m,
            **ss,
            **stim_locked,
            # Internal fields for null computation (stripped before CSV write)
            "_J": J,
            "_sigma_eye": sigma_eye,
        }
        results.append(row)

        split_bundle = sh_m.get("_split_bundle")
        if split_bundle is not None:
            split_bundle_rows.append({
                "image_id": win.image_id,
                "window_id": row["window_id"],
                "B_a": split_bundle["B_a"],
                "B_b": split_bundle["B_b"],
            })

    # -------------------------------------------------------------- null
    print("Computing image-shuffled null...")
    shuf_g = _compute_shuffled_g(results, n_matches=n_shuffle_matches)
    for row in results:
        sg = shuf_g.get(row["image_id"], float("nan"))
        row["g_t_shuffled_median"] = sg
        row["g_t_delta"] = (
            row["g_t"] - sg
            if math.isfinite(row["g_t"]) and math.isfinite(sg)
            else float("nan")
        )

    # ----------------------------------------------------------------- CSV
    scalar_keys = [k for k in results[0] if not k.startswith("_")]
    csv_path = output_dir / "step2_image_windows.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=scalar_keys)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in scalar_keys})
    print(f"  CSV: {csv_path}")

    bundle_path = output_dir / "step2_split_half_bundles.npz"
    if split_bundle_rows:
        np.savez_compressed(
            bundle_path,
            image_ids=np.asarray([row["image_id"] for row in split_bundle_rows], dtype=np.int64),
            window_ids=np.asarray([row["window_id"] for row in split_bundle_rows]),
            B_a=np.asarray([row["B_a"] for row in split_bundle_rows], dtype=np.float64),
            B_b=np.asarray([row["B_b"] for row in split_bundle_rows], dtype=np.float64),
        )
        print(f"  Split bundles: {bundle_path}")

    # --------------------------------------------------------------- stats
    g_arr = np.array([r["g_t"] for r in results])
    e_arr = np.array([r["e_fem"] for r in results])
    g_shuf = np.array([r["g_t_shuffled_median"] for r in results])
    jnorm = np.array([r["jacobian_fro_norm"] for r in results])
    eye_amp = np.array([r["eye_amplitude_px2"] for r in results])
    mean_rate = np.array([r["mean_emp_rate"] for r in results])
    tr_model = np.array([r["trace_cov_model_fem"] for r in results])
    g_local_1 = np.array([r["g_t_local_1px"] for r in results])
    g_local_2 = np.array([r["g_t_local_2px"] for r in results])
    e_fem_cv = np.array([r["e_fem_cv_median"] for r in results])
    e_fem_cv_mean = np.array([r["e_fem_cv_mean"] for r in results])
    b_split = np.array([r["b_split_corr_median"] for r in results])
    e_fem_shuf = np.array([r["e_fem_shuffle_med"] for r in results])

    fin = np.isfinite(g_arr) & np.isfinite(e_arr)
    n_finite = int(fin.sum())

    # B_emp reliability: how many images have reliable empirical eye sensitivity.
    b_reliable_frac = (
        float(np.nanmean(b_split > 0.0)) if np.isfinite(b_split).any() else float("nan")
    )
    b_split_median = float(np.nanmedian(b_split)) if np.isfinite(b_split).any() else float("nan")
    # E_FEM signal-to-noise: fraction of images where E_FEM > shuffle.
    e_snr_frac = float(np.nanmean(e_arr > e_fem_shuf)) if np.isfinite(e_fem_shuf).any() else float("nan")

    summary: dict = {
        "subject": subject,
        "date": date,
        "n_windows": len(results),
        "n_finite": n_finite,
        "dataset_idx": dataset_idx,
        # Empirical target reliability
        "b_split_median": b_split_median,
        "b_reliable_frac_pos": b_reliable_frac,
        "e_fem_snr_frac": e_snr_frac,
        # Main bridge
        "spearman_g_t_vs_e_fem": float("nan"),
        "spearman_g_t_shuffled_vs_e_fem": float("nan"),
        "spearman_g_t_delta": float("nan"),
        "spearman_model_trace_vs_e_fem": float("nan"),
        "spearman_g_t_vs_model_trace": float("nan"),
        # Cross-validated target
        "spearman_e_fem_vs_e_fem_cv": float("nan"),
        "spearman_g_t_vs_e_fem_cv": float("nan"),
        "spearman_model_trace_vs_e_fem_cv": float("nan"),
        "spearman_eye_amp_vs_e_fem_cv": float("nan"),
        # Local variants
        "spearman_g_t_local_1px_vs_e_fem": float("nan"),
        "spearman_g_t_local_2px_vs_e_fem": float("nan"),
        "spearman_g_t_local_1px_vs_model_trace": float("nan"),
        "spearman_g_t_local_2px_vs_model_trace": float("nan"),
        # Confounds
        "spearman_jnorm_vs_e_fem": float("nan"),
        "spearman_eye_amp_vs_e_fem": float("nan"),
        "spearman_mean_rate_vs_e_fem": float("nan"),
    }

    if n_finite >= 4:
        summary["spearman_g_t_vs_e_fem"] = _spearman_corr(g_arr, e_arr)
        summary["spearman_g_t_shuffled_vs_e_fem"] = _spearman_corr(g_shuf, e_arr)
        summary["spearman_g_t_delta"] = (
            summary["spearman_g_t_vs_e_fem"]
            - summary["spearman_g_t_shuffled_vs_e_fem"]
        )
        summary["spearman_model_trace_vs_e_fem"] = _spearman_corr(tr_model, e_arr)
        summary["spearman_g_t_vs_model_trace"] = _spearman_corr(g_arr, tr_model)
        summary["spearman_g_t_local_1px_vs_e_fem"] = _spearman_corr(g_local_1, e_arr)
        summary["spearman_g_t_local_2px_vs_e_fem"] = _spearman_corr(g_local_2, e_arr)
        summary["spearman_g_t_local_1px_vs_model_trace"] = _spearman_corr(g_local_1, tr_model)
        summary["spearman_g_t_local_2px_vs_model_trace"] = _spearman_corr(g_local_2, tr_model)
        summary["spearman_jnorm_vs_e_fem"] = _spearman_corr(jnorm, e_arr)
        summary["spearman_eye_amp_vs_e_fem"] = _spearman_corr(eye_amp, e_arr)
        summary["spearman_mean_rate_vs_e_fem"] = _spearman_corr(mean_rate, e_arr)
        # Cross-validated E_FEM correlations (empirical target reliability check)
        summary["spearman_e_fem_vs_e_fem_cv"] = _spearman_corr(e_arr, e_fem_cv)
        summary["spearman_g_t_vs_e_fem_cv"] = _spearman_corr(g_arr, e_fem_cv)
        summary["spearman_model_trace_vs_e_fem_cv"] = _spearman_corr(tr_model, e_fem_cv)
        summary["spearman_eye_amp_vs_e_fem_cv"] = _spearman_corr(eye_amp, e_fem_cv)
        summary["spearman_model_trace_vs_e_fem_cv_mean"] = _spearman_corr(tr_model, e_fem_cv_mean)

    summary["n_split_repeats"] = int(n_split_repeats)
    summary["median_emp_split_alignment"] = float(np.nanmedian([r["emp_split_alignment_median"] for r in results]))
    summary["median_emp_split_variance_capture"] = float(np.nanmedian([r["emp_split_variance_capture_median"] for r in results]))
    summary["median_stimulus_locked_variance"] = float(np.nanmedian([r["stimulus_locked_variance"] for r in results]))

    summary_path = output_dir / "step2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    if reliable_subset_summary:
        subset_summary = _build_reliable_subset_summary(results)
        (output_dir / "step2_reliable_subset_summary.json").write_text(
            json.dumps(subset_summary, indent=2) + "\n"
        )
    if residualized_controls:
        residual_summary = _build_residualized_summary(results)
        (output_dir / "step2_residualized_summary.json").write_text(
            json.dumps(residual_summary, indent=2) + "\n"
        )

    print(
        f"\n  Spearman r(g_t,         E_FEM) = {summary['spearman_g_t_vs_e_fem']:.3f}"
        f"  (N={n_finite})"
    )
    print(f"  Spearman r(g_shuf,      E_FEM) = {summary['spearman_g_t_shuffled_vs_e_fem']:.3f}")
    print(f"  Spearman r(tr_model,    E_FEM) = {summary['spearman_model_trace_vs_e_fem']:.3f}  [nonlinear bridge]")
    print(f"  Spearman r(g_t,    tr_model)   = {summary['spearman_g_t_vs_model_trace']:.3f}  [linearization check]")
    print(f"  Spearman r(g_local_1px, E_FEM) = {summary['spearman_g_t_local_1px_vs_e_fem']:.3f}  [local ≤1 px]")
    print(f"  Spearman r(g_local_2px, E_FEM) = {summary['spearman_g_t_local_2px_vs_e_fem']:.3f}  [local ≤2 px]")
    print(f"  Spearman r(g_local_1px, tr_model) = {summary['spearman_g_t_local_1px_vs_model_trace']:.3f}")
    print(f"  Spearman r(|J|_F,       E_FEM) = {summary['spearman_jnorm_vs_e_fem']:.3f}  [confound]")
    print(f"  Spearman r(eye_amp,     E_FEM) = {summary['spearman_eye_amp_vs_e_fem']:.3f}  [confound]")
    print(f"  Spearman r(rate,        E_FEM) = {summary['spearman_mean_rate_vs_e_fem']:.3f}  [confound]")
    print(f"\n  --- Split-half B_emp reliability ---")
    print(f"  Median B_split_corr             = {b_split_median:.3f}  [B_emp reliability]")
    print(f"  Frac images B_split_corr > 0    = {b_reliable_frac:.3f}  [fraction reliable]")
    print(f"  Frac images E_FEM > shuffle     = {e_snr_frac:.3f}  [E_FEM SNR]")
    print(f"  Spearman r(E_FEM,    E_FEM_cv)  = {summary['spearman_e_fem_vs_e_fem_cv']:.3f}  [target self-consistency]")
    print(f"  Spearman r(g_t,      E_FEM_cv)  = {summary['spearman_g_t_vs_e_fem_cv']:.3f}  [cv bridge]")
    print(f"  Spearman r(tr_model, E_FEM_cv)  = {summary['spearman_model_trace_vs_e_fem_cv']:.3f}  [cv nonlinear bridge]")
    print(f"  Spearman r(eye_amp,  E_FEM_cv)  = {summary['spearman_eye_amp_vs_e_fem_cv']:.3f}  [cv confound]")

    # --------------------------------------------------------------- figure
    _write_step2_figure(results, summary, output_dir)

    # --------------------------------------------------------------- README
    _write_readme(output_dir, subject, date, dataset_configs_path, summary,
                  jacobian_step_px, min_samples, n_shuffle_matches,
                  pixels_per_degree, dt)

    return summary


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _write_step2_figure(results: list[dict], summary: dict,
                        output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    g = np.array([r["g_t"] for r in results])
    e = np.array([r["e_fem_cv_mean"] for r in results])
    g_shuf = np.array([r["g_t_shuffled_median"] for r in results])
    jnorm = np.array([r["jacobian_fro_norm"] for r in results])
    eye_amp = np.array([r["eye_amplitude_px2"] for r in results])
    tr_model = np.array([r["trace_cov_model_fem"] for r in results])
    stim_locked = np.array([r["stimulus_locked_variance"] for r in results])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    date = summary.get("date", "?")
    n = summary.get("n_windows", 0)
    fig.suptitle(
        f"Step 2 CV bridge  |  {date}  |  N={n} images",
        fontsize=11,
    )

    def _scatter(ax, x: np.ndarray, y: np.ndarray,
                 xlabel: str, ylabel: str, title: str,
                 color: str = "#2c7bb6") -> None:
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=45, alpha=0.85, color=color, edgecolors="none")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)

    rho_g = summary.get("spearman_g_t_vs_e_fem", float("nan"))
    rho_gs = summary.get("spearman_g_t_shuffled_vs_e_fem", float("nan"))
    rho_tr = summary.get("spearman_model_trace_vs_e_fem", float("nan"))
    rho_gt = summary.get("spearman_g_t_vs_model_trace", float("nan"))
    rho_j = summary.get("spearman_jnorm_vs_e_fem", float("nan"))
    rho_e = summary.get("spearman_eye_amp_vs_e_fem", float("nan"))

    # Row 0: model-predicted vs empirical
    _scatter(axes[0, 0], g, e,
             "g_t  (linear FEM drive)",
             "E_FEM_cv mean  (empirical FEM drive)",
             f"g_t vs E_FEM_cv  (ρ={summary.get('spearman_g_t_vs_e_fem_cv', float('nan')):.2f})",
             color="#2c7bb6")

    _scatter(axes[0, 1], tr_model, e,
             "tr(Cov model)  (nonlinear FEM trace)",
             "E_FEM_cv mean  (empirical FEM drive)",
             f"Model trace vs E_FEM_cv  (ρ={summary.get('spearman_model_trace_vs_e_fem_cv_mean', float('nan')):.2f})",
             color="#1a9641")

    _scatter(axes[0, 2], g - g_shuf, e,
             "g_t − g_shuf  (matched advantage)",
             "E_FEM_cv mean",
             f"Matched − shuffled  (Δρ={rho_g - rho_gs:.2f})" if math.isfinite(rho_g) and math.isfinite(rho_gs) else "Matched − shuffled",
             color="#d7191c")
    axes[0, 2].axvline(0, color="gray", linewidth=0.8, linestyle="--")

    # Row 1: linearization check + confounds
    _scatter(axes[1, 0], g, tr_model,
             "g_t  (linear Jacobian trace)",
             "tr(Cov model)  (nonlinear)",
             f"Linearization: g_t vs model trace  (ρ={rho_gt:.2f})",
             color="#fd8d3c")

    _scatter(axes[1, 1], stim_locked, e,
             "Stimulus-locked variance",
             "E_FEM_cv mean",
             f"Decisive control  (ρ={summary.get('spearman_e_fem_vs_e_fem_cv', float('nan')):.2f})",
             color="#756bb1")

    _scatter(axes[1, 2], eye_amp, e,
             "tr(Σ_eye)  (eye amplitude)",
             "E_FEM_cv mean",
             f"Confound: eye amplitude  (ρ={rho_e:.2f})",
             color="#74c476")

    fig.tight_layout()
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig_path = fig_dir / "step2_cv_bridge.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    canonical = FIGURES_DIR / "jacobian_predictive_framework"
    canonical.mkdir(parents=True, exist_ok=True)
    shutil.copy(fig_path,
                canonical / f"step2_cv_bridge_{summary.get('date', 'unknown')}.png")
    print(f"  Figure: {fig_path}")


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

def _write_readme(
    output_dir: Path, subject: str, date: str, dataset_configs_path: str,
    summary: dict, jacobian_step_px: float, min_samples: int,
    n_shuffle_matches: int, pixels_per_degree: float, dt: float,
) -> None:
    (output_dir / "step2_readme.md").write_text(f"""# Step 2: Real-data scalar bridge

## Session
- subject: {subject}
- date: {date}
- dataset config: {dataset_configs_path}
- repeated split-half repeats: {summary.get('n_split_repeats', 0)}

## Analysis unit
Per natural-image identity.  All valid fixation bins pooled across trials.
Minimum {min_samples} samples per image.

## Model-predicted FEM drive (g_t)
    g_t = tr(J_t @ Sigma_eye,t @ J_t^T)
J_t is (NC x 2), computed at the baseline eye position via central finite
differences (step = {jacobian_step_px:.3f} px).

## Empirical FEM drive (E_FEM)
    B_emp = OLS( R_centered ~ E_centered @ B^T )
    E_FEM = tr(B_emp @ Sigma_eye,t @ B_emp^T)
Neural responses in sp/s (dt = {dt:.6f} s).

## Shuffled null
Median g_t from {n_shuffle_matches} matched images (matched on jacobian_fro_norm + stim_rms).

## Cross-validated target
- repeated random trial-level split-halves
- saved split bundles: `step2_split_half_bundles.npz`
- primary bridge figure: `figures/step2_cv_bridge.png`

## Confounds reported
- ||J||_F (Jacobian norm)
- tr(Sigma_eye) (eye amplitude)
- mean empirical rate

## Key results
{json.dumps({k: round(v, 4) if isinstance(v, float) and math.isfinite(v) else v
              for k, v in summary.items() if not k.startswith('_')}, indent=2)}
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", required=True)
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--dataset-configs-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--checkpoint-path", default=None)
    p.add_argument("--dataset-idx", type=int, required=True)
    p.add_argument("--jacobian-step-px", type=float, default=DEFAULT_JACOBIAN_STEP_PX)
    p.add_argument("--min-samples", type=int, default=50)
    p.add_argument("--n-shuffle-matches", type=int, default=5)
    p.add_argument("--n-split-repeats", type=int, default=DEFAULT_N_SPLIT_REPEATS)
    p.add_argument("--reliable-subset-summary", action="store_true")
    p.add_argument("--residualized-controls", action="store_true")
    p.add_argument("--model-type", default=None)
    p.add_argument("--model-index", type=int, default=None)
    p.add_argument("--model-device", default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_step2(
        subject=args.subject,
        date=args.date,
        dataset_configs_path=args.dataset_configs_path,
        output_dir=Path(args.output_dir),
        checkpoint_path=args.checkpoint_path,
        dataset_idx=args.dataset_idx,
        jacobian_step_px=args.jacobian_step_px,
        min_samples=args.min_samples,
        n_shuffle_matches=args.n_shuffle_matches,
        n_split_repeats=args.n_split_repeats,
        residualized_controls=args.residualized_controls,
        reliable_subset_summary=args.reliable_subset_summary,
        model_type=args.model_type,
        model_index=args.model_index,
        model_device=args.model_device,
    )
    print(f"\nSaved Step 2 outputs to {args.output_dir}")
