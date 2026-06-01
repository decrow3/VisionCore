"""
Phase 1 FEM/V1 covariance analysis: FEM covariance estimators.

Implements two estimators:
  1. B_emp regression covariance (sensitivity estimator; close to prior empirical bridge)
  2. McFarland-style trajectory-similarity covariance (primary estimator)

Both produce C_FEM[unit, unit] and the underlying projection matrices.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Utility: valid-pair flattening
# ---------------------------------------------------------------------------

def _flatten_valid(R: np.ndarray, eye: np.ndarray, valid_mask: np.ndarray):
    """
    Flatten (NT, T, NC) arrays to (N_valid, NC) and (N_valid, 2) by keeping
    only bins where valid_mask is True and both R and eye are finite.
    """
    combined_valid = (
        valid_mask
        & np.all(np.isfinite(R), axis=-1)
        & np.all(np.isfinite(eye), axis=-1)
    )
    idx = np.where(combined_valid)
    R_flat = R[idx[0], idx[1], :]    # (N_valid, NC)
    E_flat = eye[idx[0], idx[1], :]  # (N_valid, 2)
    return R_flat, E_flat, combined_valid


# ---------------------------------------------------------------------------
# Estimator 1: B_emp regression covariance (sensitivity estimator)
# ---------------------------------------------------------------------------

def fit_b_emp(
    R_resid: np.ndarray,
    eye: np.ndarray,
    valid_mask: np.ndarray,
    min_samples: int = 50,
) -> dict:
    """
    Fit the eye-sensitivity matrix B_emp via OLS pooled across all valid bins.

        R_resid[i,t] ≈ B_emp.T @ eye[i,t] + noise

    Parameters
    ----------
    R_resid : (NT, T, NC)  LOTO residuals
    eye     : (NT, T, 2)   eye position in degrees
    valid_mask : (NT, T)

    Returns
    -------
    dict with:
        B_emp       (NC, 2)  sensitivity matrix
        R_eye_pred  (NT, T, NC)  eye-predicted component (NaN where invalid)
        C_eye       (NC, NC) eye-induced covariance B_emp @ Sigma_eye @ B_emp.T
        sigma_eye   (2, 2)   eye position covariance
        n_samples   int
        fit_ok      bool
    """
    R_flat, E_flat, combined_valid = _flatten_valid(R_resid, eye, valid_mask)
    n_samples = R_flat.shape[0]

    if n_samples < min_samples:
        warnings.warn(f"fit_b_emp: only {n_samples} valid samples (min={min_samples})")
        NC = R_resid.shape[-1]
        nan_mat = np.full((NC, NC), np.nan)
        return {
            "B_emp": np.full((NC, 2), np.nan),
            "R_eye_pred": np.full_like(R_resid, np.nan),
            "C_eye": nan_mat,
            "sigma_eye": np.full((2, 2), np.nan),
            "n_samples": n_samples,
            "fit_ok": False,
        }

    # Sanity check: residuals should be approximately zero-mean if proper
    # residualization was applied upstream.
    resid_mean = float(np.nanmean(R_flat))
    resid_std = float(np.nanstd(R_flat))
    if np.isfinite(resid_std) and resid_std > 0 and abs(resid_mean) > 0.1 * resid_std:
        warnings.warn(
            "fit_b_emp: input residuals appear non-centered; "
            "expected residualized activity (possible raw-rate input)"
        )

    # Center eye positions
    E_mean = E_flat.mean(axis=0)   # (2,)
    E_c = E_flat - E_mean          # (N_valid, 2)

    # OLS: B_emp.T = pinv(E_c) @ R_flat  → shape (2, NC)
    B_T, _, _, _ = np.linalg.lstsq(E_c, R_flat, rcond=None)  # (2, NC)
    B_emp = B_T.T  # (NC, 2)

    # Eye covariance
    sigma_eye = np.cov(E_flat.T)   # (2, 2)
    C_eye = B_emp @ sigma_eye @ B_emp.T  # (NC, NC)

    # Full predicted array (NaN where invalid)
    NT, T, NC = R_resid.shape
    R_eye_pred = np.full((NT, T, NC), np.nan)
    combined_valid_t = (
        valid_mask
        & np.all(np.isfinite(eye), axis=-1)
    )
    idx = np.where(combined_valid_t)
    E_c_all = eye[idx[0], idx[1], :] - E_mean
    R_eye_pred[idx[0], idx[1], :] = E_c_all @ B_T  # (N, NC)

    return {
        "B_emp": B_emp,
        "R_eye_pred": R_eye_pred,
        "C_eye": C_eye,
        "sigma_eye": sigma_eye,
        "E_mean": E_mean,
        "n_samples": n_samples,
        "fit_ok": True,
    }


def b_emp_covariance(
    R_resid: np.ndarray,
    eye: np.ndarray,
    valid_mask: np.ndarray,
    **kwargs,
) -> dict:
    """
    Convenience wrapper that returns the full B_emp estimator result.
    Label: 'B_emp_regression'.
    """
    result = fit_b_emp(R_resid, eye, valid_mask, **kwargs)
    result["method"] = "B_emp_regression"
    return result


# ---------------------------------------------------------------------------
# Estimator 2: McFarland-style trajectory-similarity covariance (primary)
# ---------------------------------------------------------------------------

def mcfarland_fem_covariance(
    R_resid: np.ndarray,
    eye: np.ndarray,
    image_ids: np.ndarray,
    valid_mask: np.ndarray,
    min_repeats: int = 3,
    n_similarity_bins: int = 4,
    seed: int = 0,
) -> dict:
    """
    McFarland-style trajectory-similarity FEM covariance estimator.

    For each image context with ≥ min_repeats valid trials:
      1. Compute mean eye-position per trial window (after averaging over the
         time bins belonging to that image in that trial).
      2. For all trial pairs (i, j): compute eye trajectory distance
             d_ij = ||e_i - e_j||_2
         and neural outer product
             q_ij = outer(r_i - r_mean, r_j - r_mean)
      3. Regress q_ij ~ slope * d_ij + C_intercept via OLS over all pairs.
      4. C_intercept (d=0) is the FEM covariance for this image context.

    Average C_intercept across all image contexts to produce C_FEM_mcfarland.

    Returns
    -------
    dict with:
        C_FEM       (NC, NC)  averaged McFarland FEM covariance
        n_contexts  int       number of image contexts contributing
        context_records  list[dict]  per-context metrics
        method      str
        fit_ok      bool
    """
    NT, T, NC = R_resid.shape
    rng = np.random.default_rng(seed)

    C_sum = np.zeros((NC, NC))
    n_contexts = 0
    context_records = []

    unique_images = np.unique(image_ids[valid_mask & (image_ids >= 0)])

    for img in unique_images:
        # Gather trials that have at least one valid bin for this image
        trial_has_img = np.any(valid_mask & (image_ids == img), axis=1)  # (NT,)
        trial_inds = np.where(trial_has_img)[0]

        if len(trial_inds) < min_repeats:
            continue

        # For each trial, average residuals and eye over its image bins
        r_list = []
        e_list = []
        for tr in trial_inds:
            t_mask = valid_mask[tr, :] & (image_ids[tr, :] == img)
            if not t_mask.any():
                continue
            r_list.append(R_resid[tr, t_mask, :].mean(axis=0))  # (NC,)
            e_list.append(eye[tr, t_mask, :].mean(axis=0))      # (2,)

        if len(r_list) < min_repeats:
            continue

        r_mat = np.array(r_list)  # (n, NC)
        e_mat = np.array(e_list)  # (n, 2)

        # Skip if any NaN
        if not (np.isfinite(r_mat).all() and np.isfinite(e_mat).all()):
            continue

        n = len(r_mat)
        r_mean = r_mat.mean(axis=0)

        # Pairwise quantities
        d_ij = []
        q_ij = []  # vectorised outer products as flattened upper triangles

        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(e_mat[i] - e_mat[j]))
                q = np.outer(r_mat[i] - r_mean, r_mat[j] - r_mean)
                d_ij.append(d)
                q_ij.append(q.ravel())

        if len(d_ij) < 2:
            continue

        d_arr = np.array(d_ij)                  # (n_pairs,)
        q_arr = np.array(q_ij)                  # (n_pairs, NC*NC)

        # OLS: q ~ slope * d + intercept  (per element of the NC×NC matrix)
        D = np.column_stack([d_arr, np.ones_like(d_arr)])  # (n_pairs, 2)
        coefs, _, _, _ = np.linalg.lstsq(D, q_arr, rcond=None)  # (2, NC*NC)
        C_intercept = coefs[1, :].reshape(NC, NC)  # d=0 intercept
        C_intercept = 0.5 * (C_intercept + C_intercept.T)

        C_sum += C_intercept
        n_contexts += 1
        context_records.append({
            "image_id": int(img),
            "n_repeats": n,
            "mean_d": float(np.mean(d_arr)),
            "std_d": float(np.std(d_arr)),
        })

    fit_ok = n_contexts >= 5
    C_FEM = C_sum / n_contexts if n_contexts > 0 else np.full((NC, NC), np.nan)

    if not fit_ok:
        warnings.warn(
            f"mcfarland_fem_covariance: only {n_contexts} contexts; "
            "result may be unreliable"
        )

    return {
        "C_FEM": C_FEM,
        "n_contexts": n_contexts,
        "context_records": context_records,
        "method": "mcfarland",
        "fit_ok": fit_ok,
    }


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def compare_estimators(result_mcfarland: dict, result_b_emp: dict) -> dict:
    """
    Compute alignment between the McFarland C_FEM and the B_emp C_eye.

    Returns dict with:
        cov_correlation   upper-triangle correlation between the two matrices
        subspace_overlap  subspace overlap of top-2 eigenvectors (k=2)
    """
    C1 = result_mcfarland.get("C_FEM")
    C2 = result_b_emp.get("C_eye")

    if C1 is None or C2 is None or not (np.isfinite(C1).all() and np.isfinite(C2).all()):
        return {
            "cov_correlation": float("nan"),
            "subspace_overlap": float("nan"),
            "status": "one_or_both_estimators_failed",
        }

    NC = C1.shape[0]
    idx = np.triu_indices(NC, k=1)
    c1_vec = C1[idx]
    c2_vec = C2[idx]
    cov_corr = float(stats.pearsonr(c1_vec, c2_vec)[0]) if len(c1_vec) > 10 else float("nan")

    # Top-2 subspace overlap
    overlap = _top_k_subspace_overlap(C1, C2, k=2)

    return {
        "cov_correlation": cov_corr,
        "subspace_overlap": overlap,
        "status": "ok",
    }


def _top_k_subspace_overlap(C1: np.ndarray, C2: np.ndarray, k: int = 2) -> float:
    """
    Subspace overlap (Grassmann similarity) between top-k eigenspaces of C1, C2.

        overlap = trace(U1^T U2 U2^T U1) / k

    where U1, U2 are (NC, k) matrices of top-k eigenvectors.
    """
    try:
        vals1, vecs1 = np.linalg.eigh(C1)
        vals2, vecs2 = np.linalg.eigh(C2)
        U1 = vecs1[:, -k:]  # top-k eigenvectors (NC, k)
        U2 = vecs2[:, -k:]
        M = U1.T @ U2        # (k, k)
        return float(np.trace(M @ M.T) / k)
    except np.linalg.LinAlgError:
        return float("nan")
