"""
Phase 1 FEM/V1 covariance analysis: Analysis 2 noise-correlation metrics.

Implements:
- Raw / PSTH-corrected / eye-corrected pairwise correlations
- Session-level correlation summaries
- Synthetic control diagnostics
- Residual correlation structure binned by mean-rate product
"""
from __future__ import annotations

import numpy as np

from .estimators import fit_b_emp


def _flatten_valid_data(R: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Flatten (NT,T,U) to (N,U), setting invalid rows to NaN."""
    NT, T, U = R.shape
    X = R.reshape(NT * T, U).astype(np.float64)
    vm = valid_mask.reshape(NT * T)
    X[~vm, :] = np.nan
    return X


def _pair_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Return (corr, cov, n) for finite-overlap samples."""
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 5:
        return float("nan"), float("nan"), n
    xv = x[m]
    yv = y[m]
    sx = np.std(xv)
    sy = np.std(yv)
    if sx < 1e-12 or sy < 1e-12:
        return float("nan"), float("nan"), n
    c = float(np.corrcoef(xv, yv)[0, 1])
    cov = float(np.cov(xv, yv)[0, 1])
    return c, cov, n


def _median_finite(v: list[float]) -> float:
    if not v:
        return float("nan")
    arr = np.array(v, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _fit_b_emp_flat(X: np.ndarray, E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Flat OLS fit for synthetic controls.

    Returns (B_emp [U,2], E_mean [2]).
    """
    E_mean = np.mean(E, axis=0)
    E_c = E - E_mean
    B_T, _, _, _ = np.linalg.lstsq(E_c, X, rcond=None)
    return B_T.T, E_mean


def _median_pair_corr_dense(X: np.ndarray) -> float:
    """Median upper-triangle correlation for dense finite matrix (N,U)."""
    if X.shape[0] < 5 or X.shape[1] < 2:
        return float("nan")
    C = np.corrcoef(X, rowvar=False)
    idx = np.triu_indices(C.shape[0], k=1)
    vals = C[idx]
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float("nan")


def run_noise_correlations(
    session: str,
    R_raw: np.ndarray,
    R_resid_psth: np.ndarray,
    eye: np.ndarray,
    valid_mask: np.ndarray,
    unit_mask: np.ndarray,
    seed: int = 0,
    min_pair_samples: int = 100,
) -> dict:
    """
    Run Analysis 2 for a single session.

    Returns dict with keys:
      - session_metrics
      - pair_metrics_rows
      - synthetic_control_rows
      - residual_structure_rows
    """
    rng = np.random.default_rng(seed)

    Rr = R_raw[:, :, unit_mask]
    Rp = R_resid_psth[:, :, unit_mask]
    U = Rr.shape[2]

    # Eye correction from PSTH residuals.
    bfit = fit_b_emp(Rp, eye, valid_mask)
    R_eye_pred = bfit["R_eye_pred"]
    R_eye_corr = Rp - R_eye_pred if bfit["fit_ok"] else np.full_like(Rp, np.nan)

    X_raw = _flatten_valid_data(Rr, valid_mask)
    X_psth = _flatten_valid_data(Rp, valid_mask)
    X_corr = _flatten_valid_data(R_eye_corr, valid_mask)

    # Pairwise metrics.
    pair_rows: list[dict] = []
    raw_vals = []
    psth_vals = []
    corr_vals = []

    unit_mean_rate = np.nanmean(X_raw, axis=0)

    for i in range(U):
        for j in range(i + 1, U):
            c_raw, cov_raw, n_raw = _pair_corr(X_raw[:, i], X_raw[:, j])
            c_psth, cov_psth, n_psth = _pair_corr(X_psth[:, i], X_psth[:, j])
            c_corr, cov_corr, n_corr = _pair_corr(X_corr[:, i], X_corr[:, j])

            n_eff = min(n_raw, n_psth, n_corr)
            if n_eff < min_pair_samples:
                continue

            if np.isfinite(c_raw):
                raw_vals.append(c_raw)
            if np.isfinite(c_psth):
                psth_vals.append(c_psth)
            if np.isfinite(c_corr):
                corr_vals.append(c_corr)

            z_raw = float(np.arctanh(np.clip(c_raw, -0.999999, 0.999999))) if np.isfinite(c_raw) else float("nan")
            z_psth = float(np.arctanh(np.clip(c_psth, -0.999999, 0.999999))) if np.isfinite(c_psth) else float("nan")
            z_corr = float(np.arctanh(np.clip(c_corr, -0.999999, 0.999999))) if np.isfinite(c_corr) else float("nan")

            pair_rows.append(
                {
                    "session": session,
                    "unit_i": i,
                    "unit_j": j,
                    "n_samples": n_eff,
                    "corr_raw": c_raw,
                    "corr_psth": c_psth,
                    "corr_eye_corrected": c_corr,
                    "fisher_z_raw": z_raw,
                    "fisher_z_psth": z_psth,
                    "fisher_z_eye_corrected": z_corr,
                    "cov_raw": cov_raw,
                    "cov_psth": cov_psth,
                    "cov_eye_corrected": cov_corr,
                    "correlation_delta_raw_to_psth": c_psth - c_raw if np.isfinite(c_psth) and np.isfinite(c_raw) else float("nan"),
                    "correlation_delta_psth_to_eye_corrected": c_corr - c_psth if np.isfinite(c_corr) and np.isfinite(c_psth) else float("nan"),
                    "correlation_delta_raw_to_eye_corrected": c_corr - c_raw if np.isfinite(c_corr) and np.isfinite(c_raw) else float("nan"),
                    "mean_rate_product": float(unit_mean_rate[i] * unit_mean_rate[j]) if np.isfinite(unit_mean_rate[i]) and np.isfinite(unit_mean_rate[j]) else float("nan"),
                }
            )

    n_pairs_valid = len(pair_rows)
    n_pairs_total = U * (U - 1) // 2

    med_raw = _median_finite(raw_vals)
    med_psth = _median_finite(psth_vals)
    med_corr = _median_finite(corr_vals)

    session_metrics = {
        "session": session,
        "n_units": U,
        "n_pairs_total": n_pairs_total,
        "n_pairs_valid": n_pairs_valid,
        "median_corr_raw": med_raw,
        "median_corr_psth": med_psth,
        "median_corr_eye_corrected": med_corr,
        "median_corr_residual": med_corr,
        "median_delta_raw_to_corrected": med_corr - med_raw if np.isfinite(med_corr) and np.isfinite(med_raw) else float("nan"),
        "fraction_pairs_positive_raw": float(np.mean(np.array(raw_vals) > 0)) if raw_vals else float("nan"),
        "fraction_pairs_positive_corrected": float(np.mean(np.array(corr_vals) > 0)) if corr_vals else float("nan"),
        "fraction_pairs_negative_corrected": float(np.mean(np.array(corr_vals) < 0)) if corr_vals else float("nan"),
    }

    # Residual structure: binned by mean-rate product.
    residual_structure_rows: list[dict] = []
    if pair_rows:
        mrp = np.array([r["mean_rate_product"] for r in pair_rows], dtype=np.float64)
        cc = np.array([r["corr_eye_corrected"] for r in pair_rows], dtype=np.float64)
        ok = np.isfinite(mrp) & np.isfinite(cc)
        if ok.sum() >= 20:
            q = np.quantile(mrp[ok], np.linspace(0.0, 1.0, 6))
            for b in range(5):
                lo, hi = q[b], q[b + 1]
                if b < 4:
                    m = ok & (mrp >= lo) & (mrp < hi)
                else:
                    m = ok & (mrp >= lo) & (mrp <= hi)
                residual_structure_rows.append(
                    {
                        "session": session,
                        "bin_index": b,
                        "mean_rate_product_lo": float(lo),
                        "mean_rate_product_hi": float(hi),
                        "n_pairs": int(m.sum()),
                        "median_corr_eye_corrected": float(np.median(cc[m])) if m.sum() else float("nan"),
                    }
                )

    # Synthetic controls.
    vm = valid_mask.reshape(-1)
    E = eye.reshape(-1, 2)[vm]
    Xp = X_psth[vm]
    # Keep finite rows only for synth generation.
    mfin = np.all(np.isfinite(E), axis=1) & np.all(np.isfinite(Xp), axis=1)
    E = E[mfin]
    Xp = Xp[mfin]

    synth_rows = []
    if E.shape[0] >= 200 and Xp.shape[1] >= 2:
        U_s = Xp.shape[1]
        mu = np.maximum(np.mean(Xp, axis=0), 0.1)

        # Build a synthetic eye component aligned with empirical B.
        B_emp_flat, E_mean = _fit_b_emp_flat(Xp - np.mean(Xp, axis=0), E)
        E_c = E - E_mean
        eye_comp = E_c @ B_emp_flat.T
        ec_std = np.std(eye_comp, axis=0)
        ec_std[ec_std < 1e-6] = 1.0
        eye_comp = eye_comp / ec_std[None, :] * np.std(mu)

        # Control 1: independent Poisson + eye component.
        lam1 = np.clip(mu[None, :] + 0.25 * eye_comp, 1e-3, None)
        X1 = rng.poisson(lam1).astype(np.float64)
        med_raw_1 = _median_pair_corr_dense(X1)
        B1, e1 = _fit_b_emp_flat(X1 - np.mean(X1, axis=0), E)
        X1_corr = (X1 - np.mean(X1, axis=0)) - (E - e1) @ B1.T
        med_corr_1 = _median_pair_corr_dense(X1_corr)
        synth_rows.append(
            {
                "session": session,
                "control_name": "independent_poisson_plus_eye",
                "ground_truth_intrinsic_median_corr": 0.0,
                "median_corr_raw": med_raw_1,
                "median_corr_corrected": med_corr_1,
                "expected": "near_zero_after_correction",
                "status": "ok",
            }
        )

        # Control 2: positive latent covariance + eye component.
        z = rng.standard_normal((E.shape[0], 1))
        a = rng.uniform(0.05, 0.15, size=(1, U_s))
        intrinsic = z @ a
        lam2 = np.clip(mu[None, :] + 0.25 * eye_comp + intrinsic, 1e-3, None)
        X2 = rng.poisson(lam2).astype(np.float64)
        med_raw_2 = _median_pair_corr_dense(X2)
        B2, e2 = _fit_b_emp_flat(X2 - np.mean(X2, axis=0), E)
        X2_corr = (X2 - np.mean(X2, axis=0)) - (E - e2) @ B2.T
        med_corr_2 = _median_pair_corr_dense(X2_corr)
        synth_rows.append(
            {
                "session": session,
                "control_name": "positive_latent_plus_eye",
                "ground_truth_intrinsic_median_corr": 0.04,
                "median_corr_raw": med_raw_2,
                "median_corr_corrected": med_corr_2,
                "expected": "remains_positive_after_correction",
                "status": "ok",
            }
        )

        # Control 3: zero intrinsic + estimated eye component (bias check).
        lam3 = np.clip(mu[None, :] + 0.25 * eye_comp, 1e-3, None)
        X3 = rng.poisson(lam3).astype(np.float64)
        B3, e3 = _fit_b_emp_flat(X3 - np.mean(X3, axis=0), E)
        X3_corr = (X3 - np.mean(X3, axis=0)) - (E - e3) @ B3.T
        synth_rows.append(
            {
                "session": session,
                "control_name": "zero_intrinsic_estimated_eye",
                "ground_truth_intrinsic_median_corr": 0.0,
                "median_corr_raw": _median_pair_corr_dense(X3),
                "median_corr_corrected": _median_pair_corr_dense(X3_corr),
                "expected": "near_zero_not_negative",
                "status": "ok",
            }
        )

        # Control 4: shuffled-eye correction on empirical residuals.
        perm = rng.permutation(E.shape[0])
        B4, e4 = _fit_b_emp_flat(Xp - np.mean(Xp, axis=0), E[perm])
        X4_corr = (Xp - np.mean(Xp, axis=0)) - (E - e4) @ B4.T
        synth_rows.append(
            {
                "session": session,
                "control_name": "shuffled_eye_correction_empirical",
                "ground_truth_intrinsic_median_corr": float("nan"),
                "median_corr_raw": _median_pair_corr_dense(Xp),
                "median_corr_corrected": _median_pair_corr_dense(X4_corr),
                "expected": "little_reduction_vs_matched",
                "status": "ok",
            }
        )

        # Control 5: cross-validated correction on empirical residuals.
        n = E.shape[0]
        idx = rng.permutation(n)
        a_idx = idx[: n // 2]
        b_idx = idx[n // 2 :]
        B5a, e5a = _fit_b_emp_flat(Xp[a_idx] - np.mean(Xp[a_idx], axis=0), E[a_idx])
        B5b, e5b = _fit_b_emp_flat(Xp[b_idx] - np.mean(Xp[b_idx], axis=0), E[b_idx])
        Xa_corr = (Xp[a_idx] - np.mean(Xp[a_idx], axis=0)) - (E[a_idx] - e5b) @ B5b.T
        Xb_corr = (Xp[b_idx] - np.mean(Xp[b_idx], axis=0)) - (E[b_idx] - e5a) @ B5a.T
        X5_corr = np.vstack([Xa_corr, Xb_corr])
        synth_rows.append(
            {
                "session": session,
                "control_name": "cross_validated_eye_correction_empirical",
                "ground_truth_intrinsic_median_corr": float("nan"),
                "median_corr_raw": _median_pair_corr_dense(Xp),
                "median_corr_corrected": _median_pair_corr_dense(X5_corr),
                "expected": "sign_flip_should_survive_if_real",
                "status": "ok",
            }
        )

    return {
        "session_metrics": session_metrics,
        "pair_metrics_rows": pair_rows,
        "synthetic_control_rows": synth_rows,
        "residual_structure_rows": residual_structure_rows,
    }
