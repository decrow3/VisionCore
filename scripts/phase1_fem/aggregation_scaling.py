"""
Phase 1 FEM/V1 covariance analysis: Analysis 3 aggregation-scaling diagnostic.

Supports multiple sampling units and estimator objects so scaling can be tested
on both the sensitivity estimator (B_emp regression) and the primary
trajectory-similarity covariance estimator (McFarland).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .estimators import fit_b_emp, mcfarland_fem_covariance, _top_k_subspace_overlap


def _combined_valid_mask(
    R_resid: np.ndarray,
    eye: np.ndarray,
    image_ids: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Mask of rows valid for both eye-linked estimators."""
    return (
        valid_mask
        & (image_ids >= 0)
        & np.all(np.isfinite(eye), axis=-1)
        & np.all(np.isfinite(R_resid), axis=-1)
    )


def _make_mask_from_flat_indices(shape: tuple[int, int], flat_indices: np.ndarray) -> np.ndarray:
    """Create (NT,T) bool mask with True at given flattened row indices."""
    NT, T = shape
    m = np.zeros(NT * T, dtype=bool)
    m[flat_indices] = True
    return m.reshape(NT, T)


def _build_sampling_units(
    sampling_unit: str,
    combined_valid: np.ndarray,
    image_ids: np.ndarray,
) -> list[np.ndarray]:
    """
    Build unit-index groups in flattened (trial,time) space.

    Each element of the returned list is a 1D array of flattened row indices
    that belong to one sampling unit.
    """
    NT, T = combined_valid.shape
    flat_valid = np.flatnonzero(combined_valid.reshape(NT * T))
    if flat_valid.size == 0:
        return []

    tr_idx, t_idx = np.unravel_index(flat_valid, (NT, T))
    img_idx = image_ids[tr_idx, t_idx]

    if sampling_unit == "valid_rows":
        return [np.array([int(i)], dtype=np.int64) for i in flat_valid]

    groups: dict[tuple[int, int], list[int]] = {}

    if sampling_unit == "image_time_cells":
        for k, flat_i in enumerate(flat_valid):
            key = (int(img_idx[k]), int(t_idx[k]))
            groups.setdefault(key, []).append(int(flat_i))
    elif sampling_unit == "image_windows":
        for k, flat_i in enumerate(flat_valid):
            key = (int(tr_idx[k]), int(img_idx[k]))
            groups.setdefault(key, []).append(int(flat_i))
    elif sampling_unit == "trials":
        for k, flat_i in enumerate(flat_valid):
            key = (int(tr_idx[k]), 0)
            groups.setdefault(key, []).append(int(flat_i))
    else:
        raise ValueError(f"Unknown sampling_unit '{sampling_unit}'")

    out = [np.array(v, dtype=np.int64) for _, v in groups.items() if len(v) > 0]
    return out


def _upper_tri_corr(C1: np.ndarray, C2: np.ndarray) -> float:
    """Pearson correlation across upper-triangle entries of two covariance matrices."""
    if C1.shape != C2.shape or C1.ndim != 2:
        return float("nan")
    idx = np.triu_indices(C1.shape[0], k=1)
    v1 = C1[idx]
    v2 = C2[idx]
    ok = np.isfinite(v1) & np.isfinite(v2)
    if int(ok.sum()) < 10:
        return float("nan")
    s1 = np.std(v1[ok])
    s2 = np.std(v2[ok])
    if s1 < 1e-12 or s2 < 1e-12:
        return float("nan")
    return float(np.corrcoef(v1[ok], v2[ok])[0, 1])


def _vector_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between flattened finite vectors."""
    av = a.ravel()
    bv = b.ravel()
    ok = np.isfinite(av) & np.isfinite(bv)
    if int(ok.sum()) < 10:
        return float("nan")
    av = av[ok]
    bv = bv[ok]
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(av, bv) / (na * nb))


def _quantile_finite(v: np.ndarray, q: float) -> float:
    vv = v[np.isfinite(v)]
    if vv.size == 0:
        return float("nan")
    return float(np.quantile(vv, q))


def _default_n_windows_schedule(n_available: int, min_fit_samples: int) -> list[int]:
    min_n = max(2 * int(min_fit_samples), 2)
    base = [10, 20, 40, 80, 160, n_available]
    out = sorted(set(int(x) for x in base if min_n <= int(x) <= n_available))
    return out


def _resolve_n_schedule(
    n_available: int,
    min_fit_samples: int,
    n_windows_schedule: list[int] | None,
) -> list[int]:
    min_n = max(2 * int(min_fit_samples), 2)
    if n_windows_schedule is None:
        return _default_n_windows_schedule(n_available, min_fit_samples)
    return sorted(set(int(x) for x in n_windows_schedule if min_n <= int(x) <= n_available))


def _estimate_covariance(
    estimator: str,
    R: np.ndarray,
    eye: np.ndarray,
    image_ids: np.ndarray,
    valid_mask: np.ndarray,
    min_fit_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Fit requested estimator and return a common dict with covariance object."""
    if estimator == "B_emp_regression":
        fit = fit_b_emp(R, eye, valid_mask, min_samples=min_fit_samples)
        return {
            "fit_ok": bool(fit.get("fit_ok", False)),
            "covariance": fit.get("C_eye"),
            "B_emp": fit.get("B_emp"),
            "n_samples": int(fit.get("n_samples", 0)),
            "reference_n_samples": int(fit.get("n_samples", 0)),
        }

    if estimator == "mcfarland":
        fit = mcfarland_fem_covariance(
            R_resid=R,
            eye=eye,
            image_ids=image_ids,
            valid_mask=valid_mask,
            min_repeats=3,
            seed=seed,
        )
        n_contexts = int(fit.get("n_contexts", 0))
        cov = fit.get("C_FEM")
        cov_ok = cov is not None and np.all(np.isfinite(cov))
        return {
            "fit_ok": bool(cov_ok and n_contexts >= 1),
            "covariance": cov,
            "B_emp": None,
            "n_samples": int(valid_mask.sum()),
            "reference_n_samples": n_contexts,
            "fit_ok_strict": bool(fit.get("fit_ok", False)),
        }

    raise ValueError(f"Unknown estimator '{estimator}'")


def run_aggregation_scaling(
    session: str,
    R_resid: np.ndarray,
    eye: np.ndarray,
    image_ids: np.ndarray,
    valid_mask: np.ndarray,
    unit_mask: np.ndarray,
    sampling_unit: str = "image_time_cells",
    estimator: str = "mcfarland",
    seed: int = 0,
    n_bootstrap: int = 100,
    n_windows_schedule: list[int] | None = None,
    min_fit_samples: int = 5,
    run_eye_shuffle_null: bool = True,
) -> dict[str, Any]:
    """
    Run Analysis 3 aggregation scaling for one session.

    Returns dict with keys:
            - session_metrics
      - aggregation_curve_rows
      - reliability_vs_n_rows
      - stage4_comparison_rows
    """
    R = R_resid[:, :, unit_mask]
    NT, T, U = R.shape

    if U < 2:
        return {
            "session_metrics": {
                "session": session,
                "status": "failed",
                "reason": "fewer_than_two_primary_units",
            },
            "aggregation_curve_rows": [],
            "reliability_vs_n_rows": [],
            "stage4_comparison_rows": [],
        }

    combined_valid = _combined_valid_mask(R, eye, image_ids, valid_mask)
    flat_combined_valid = np.flatnonzero(combined_valid.reshape(NT * T))
    n_valid_windows = int(combined_valid.sum())
    if n_valid_windows < max(2 * min_fit_samples, 8):
        return {
            "session_metrics": {
                "session": session,
                "status": "failed",
                "reason": "insufficient_valid_windows",
                "n_valid_windows": n_valid_windows,
                "n_units": U,
            },
            "aggregation_curve_rows": [],
            "reliability_vs_n_rows": [],
            "stage4_comparison_rows": [],
        }

    sampling_units = _build_sampling_units(sampling_unit, combined_valid, image_ids)
    n_available_units = int(len(sampling_units))
    if n_available_units < max(2 * min_fit_samples, 2):
        return {
            "session_metrics": {
                "session": session,
                "status": "failed",
                "reason": "insufficient_sampling_units",
                "sampling_unit": sampling_unit,
                "n_sampling_units": n_available_units,
                "n_valid_windows": n_valid_windows,
                "n_units": U,
            },
            "aggregation_curve_rows": [],
            "reliability_vs_n_rows": [],
            "stage4_comparison_rows": [],
        }

    # Full-data estimate used as a reference for scale-dependent similarity.
    full_fit = _estimate_covariance(
        estimator=estimator,
        R=R,
        eye=eye,
        image_ids=image_ids,
        valid_mask=combined_valid,
        min_fit_samples=max(min_fit_samples, 5),
        seed=seed,
    )
    if not full_fit.get("fit_ok", False):
        return {
            "session_metrics": {
                "session": session,
                "status": "failed",
                "reason": "full_fit_failed",
                "sampling_unit": sampling_unit,
                "estimator": estimator,
                "n_valid_windows": n_valid_windows,
                "n_units": U,
            },
            "aggregation_curve_rows": [],
            "reliability_vs_n_rows": [],
            "stage4_comparison_rows": [],
        }

    full_B = full_fit.get("B_emp")
    full_C = full_fit.get("covariance")

    schedule = _resolve_n_schedule(n_available_units, min_fit_samples, n_windows_schedule)
    rng = np.random.default_rng(seed)

    reliability_rows: list[dict[str, Any]] = []

    for N in schedule:
        n_half_units = N // 2
        if n_half_units < 1:
            continue

        for b in range(int(n_bootstrap)):
            chosen_units = rng.choice(n_available_units, size=N, replace=False)
            perm_units = rng.permutation(chosen_units)
            # Use equal-size halves; if N is odd we drop one sampled unit.
            used_units = perm_units[: 2 * n_half_units]
            a_units = used_units[:n_half_units]
            b_units = used_units[n_half_units:]

            if a_units.size == 0 or b_units.size == 0:
                continue

            rows_a = np.concatenate([sampling_units[int(i)] for i in a_units])
            rows_b = np.concatenate([sampling_units[int(i)] for i in b_units])

            vm_a = _make_mask_from_flat_indices((NT, T), rows_a)
            vm_b = _make_mask_from_flat_indices((NT, T), rows_b)

            fit_a = _estimate_covariance(
                estimator=estimator,
                R=R,
                eye=eye,
                image_ids=image_ids,
                valid_mask=vm_a,
                min_fit_samples=min_fit_samples,
                seed=int(seed + b + 17),
            )
            fit_b = _estimate_covariance(
                estimator=estimator,
                R=R,
                eye=eye,
                image_ids=image_ids,
                valid_mask=vm_b,
                min_fit_samples=min_fit_samples,
                seed=int(seed + b + 53),
            )
            if not (fit_a.get("fit_ok", False) and fit_b.get("fit_ok", False)):
                continue

            B_a = fit_a.get("B_emp")
            B_b = fit_b.get("B_emp")
            C_a = fit_a.get("covariance")
            C_b = fit_b.get("covariance")
            if C_a is None or C_b is None:
                continue

            # Split-half agreement metrics.
            m_cos = _vector_cosine(B_a, B_b) if (B_a is not None and B_b is not None) else float("nan")
            m_overlap = _top_k_subspace_overlap(C_a, C_b, k=2)
            m_covcorr = _upper_tri_corr(C_a, C_b)

            # Similarity to full-data estimate.
            if full_C is not None and np.all(np.isfinite(full_C)):
                m_to_full_a = _top_k_subspace_overlap(C_a, full_C, k=2)
                m_to_full_b = _top_k_subspace_overlap(C_b, full_C, k=2)
                m_to_full = float(np.nanmean([m_to_full_a, m_to_full_b]))
            else:
                m_to_full = float("nan")

            shuf_overlap = float("nan")
            shuf_covcorr = float("nan")
            shuf_cos = float("nan")
            if run_eye_shuffle_null:
                eye_flat = eye.reshape(NT * T, 2)
                eye_shuf_flat = eye_flat.copy()
                perm_rows = rng.permutation(flat_combined_valid)
                eye_shuf_flat[flat_combined_valid] = eye_flat[perm_rows]
                eye_shuf = eye_shuf_flat.reshape(NT, T, 2)

                fit_a_shuf = _estimate_covariance(
                    estimator=estimator,
                    R=R,
                    eye=eye_shuf,
                    image_ids=image_ids,
                    valid_mask=vm_a,
                    min_fit_samples=min_fit_samples,
                    seed=int(seed + b + 173),
                )
                fit_b_shuf = _estimate_covariance(
                    estimator=estimator,
                    R=R,
                    eye=eye_shuf,
                    image_ids=image_ids,
                    valid_mask=vm_b,
                    min_fit_samples=min_fit_samples,
                    seed=int(seed + b + 223),
                )
                if fit_a_shuf.get("fit_ok", False) and fit_b_shuf.get("fit_ok", False):
                    C_a_shuf = fit_a_shuf.get("covariance")
                    C_b_shuf = fit_b_shuf.get("covariance")
                    if C_a_shuf is not None and C_b_shuf is not None:
                        shuf_overlap = _top_k_subspace_overlap(C_a_shuf, C_b_shuf, k=2)
                        shuf_covcorr = _upper_tri_corr(C_a_shuf, C_b_shuf)
                    B_a_shuf = fit_a_shuf.get("B_emp")
                    B_b_shuf = fit_b_shuf.get("B_emp")
                    if B_a_shuf is not None and B_b_shuf is not None:
                        shuf_cos = _vector_cosine(B_a_shuf, B_b_shuf)

            reliability_rows.append(
                {
                    "session": session,
                    "sampling_unit": sampling_unit,
                    "estimator": estimator,
                    "n_windows": int(N),
                    "bootstrap_iter": int(b),
                    "split_half_B_emp_cosine": m_cos,
                    "split_half_subspace_overlap": m_overlap,
                    "split_half_covariance_correlation": m_covcorr,
                    "similarity_to_full_data_estimate": m_to_full,
                    "eye_shuffle_split_half_B_emp_cosine": shuf_cos,
                    "eye_shuffle_split_half_subspace_overlap": shuf_overlap,
                    "eye_shuffle_split_half_covariance_correlation": shuf_covcorr,
                    "true_minus_eye_shuffle_reliability": (
                        m_overlap - shuf_overlap
                        if np.isfinite(m_overlap) and np.isfinite(shuf_overlap)
                        else float("nan")
                    ),
                    "model_alignment_at_N": float("nan"),
                    "image_shuffle_alignment_at_N": float("nan"),
                }
            )

    curve_rows: list[dict[str, Any]] = []
    for N in schedule:
        rows_n = [r for r in reliability_rows if int(r["n_windows"]) == int(N)]
        if not rows_n:
            continue

        ov = np.array([r["split_half_subspace_overlap"] for r in rows_n], dtype=np.float64)
        cos = np.array([r["split_half_B_emp_cosine"] for r in rows_n], dtype=np.float64)
        cov = np.array([r["split_half_covariance_correlation"] for r in rows_n], dtype=np.float64)
        to_full = np.array([r["similarity_to_full_data_estimate"] for r in rows_n], dtype=np.float64)
        ov_shuf = np.array([r["eye_shuffle_split_half_subspace_overlap"] for r in rows_n], dtype=np.float64)
        delta_true_shuf = np.array([r["true_minus_eye_shuffle_reliability"] for r in rows_n], dtype=np.float64)

        curve_rows.append(
            {
                "session": session,
                "sampling_unit": sampling_unit,
                "estimator": estimator,
                "n_windows": int(N),
                "n_bootstrap_success": int(len(rows_n)),
                "split_half_subspace_overlap_median": _quantile_finite(ov, 0.5),
                "split_half_subspace_overlap_ci_lo": _quantile_finite(ov, 0.05),
                "split_half_subspace_overlap_ci_hi": _quantile_finite(ov, 0.95),
                "eye_shuffle_split_half_subspace_overlap_median": _quantile_finite(ov_shuf, 0.5),
                "eye_shuffle_split_half_subspace_overlap_ci_lo": _quantile_finite(ov_shuf, 0.05),
                "eye_shuffle_split_half_subspace_overlap_ci_hi": _quantile_finite(ov_shuf, 0.95),
                "true_minus_eye_shuffle_reliability_median": _quantile_finite(delta_true_shuf, 0.5),
                "true_minus_eye_shuffle_reliability_ci_lo": _quantile_finite(delta_true_shuf, 0.05),
                "true_minus_eye_shuffle_reliability_ci_hi": _quantile_finite(delta_true_shuf, 0.95),
                "split_half_B_emp_cosine_median": _quantile_finite(cos, 0.5),
                "split_half_B_emp_cosine_ci_lo": _quantile_finite(cos, 0.05),
                "split_half_B_emp_cosine_ci_hi": _quantile_finite(cos, 0.95),
                "split_half_covariance_correlation_median": _quantile_finite(cov, 0.5),
                "split_half_covariance_correlation_ci_lo": _quantile_finite(cov, 0.05),
                "split_half_covariance_correlation_ci_hi": _quantile_finite(cov, 0.95),
                "similarity_to_full_data_estimate_median": _quantile_finite(to_full, 0.5),
                "similarity_to_full_data_estimate_ci_lo": _quantile_finite(to_full, 0.05),
                "similarity_to_full_data_estimate_ci_hi": _quantile_finite(to_full, 0.95),
                "model_alignment_at_N": float("nan"),
                "image_shuffle_alignment_at_N": float("nan"),
            }
        )

    # Session-level summary metrics.
    curve_sorted = sorted(curve_rows, key=lambda r: int(r["n_windows"]))
    rel_curve = np.array(
        [r.get("split_half_subspace_overlap_median", float("nan")) for r in curve_sorted],
        dtype=np.float64,
    )
    n_curve = np.array([int(r["n_windows"]) for r in curve_sorted], dtype=np.int64)

    rel_at_max_n = float(rel_curve[-1]) if rel_curve.size else float("nan")
    rel_max = float(np.nanmax(rel_curve)) if rel_curve.size else float("nan")
    rel_target = 0.5 * rel_max if np.isfinite(rel_max) else float("nan")
    max_n_windows = float(n_curve[-1]) if n_curve.size else float("nan")

    n_half = float("nan")
    if np.isfinite(rel_target) and rel_curve.size:
        reached = np.where(rel_curve >= rel_target)[0]
        if reached.size:
            n_half = float(n_curve[int(reached[0])])

    stage4_rows = [
        {
            "session": session,
            "stage4_metric_name": "single_window_reliability",
            "stage4_value": float("nan"),
            "aggregation_metric_name": "split_half_subspace_overlap",
            "aggregation_value_at_max_N": rel_at_max_n,
            "max_N_windows": max_n_windows,
            "comparison_status": "stage4_not_available",
            "note": "Populate from Stage 4 summary when available.",
        }
    ]

    analysis_variant = (
        "B_emp_row_sampling"
        if estimator == "B_emp_regression" and sampling_unit == "valid_rows"
        else f"{estimator}_{sampling_unit}"
    )

    session_metrics = {
        "session": session,
        "status": "ok",
        "analysis_variant": analysis_variant,
        "sampling_unit": sampling_unit,
        "estimator": estimator,
        "n_units": U,
        "n_valid_windows": n_valid_windows,
        "n_sampling_units": n_available_units,
        "n_curve_points": int(len(curve_sorted)),
        "n_bootstrap_requested": int(n_bootstrap),
        "n_bootstrap_effective": int(len(reliability_rows)),
        "aggregation_reliability_at_max_N": rel_at_max_n,
        "aggregation_reliability_full": rel_at_max_n,
        "aggregation_reliability_max": rel_max,
        "aggregation_n_half": n_half,
        "max_N_windows": max_n_windows,
        "stage4_explained_by_aggregation": "not_available",
        "primary_metric": "split_half_subspace_overlap",
        "full_data_reference_fit_n_samples": int(full_fit.get("reference_n_samples", 0)),
        "full_fit_n_samples": int(full_fit.get("n_samples", 0)),
        "full_fit_b_norm": (
            float(np.linalg.norm(full_B))
            if (full_B is not None and np.all(np.isfinite(full_B)))
            else float("nan")
        ),
        "full_fit_cov_trace": (
            float(np.trace(full_C))
            if (full_C is not None and np.all(np.isfinite(full_C)))
            else float("nan")
        ),
    }

    return {
        "session_metrics": session_metrics,
        "aggregation_curve_rows": curve_rows,
        "reliability_vs_n_rows": reliability_rows,
        "stage4_comparison_rows": stage4_rows,
    }
