"""
Phase 1 FEM/V1 covariance analysis: covariance geometry metrics.

Provides:
  - Eigenspectrum metrics (top-k variance fraction, participation ratio, etc.)
  - Subspace alignment metrics (overlap, covariance correlation, projection fraction)
  - Split-half reliability of B_emp and FEM covariance
  - Shared vs image-specific geometry (Models 1–3 from plan)
  - Control distributions (eye shuffle, image shuffle, random subspace)
  - Reliability-normalized alignment with reliability ceiling gate
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

from .estimators import fit_b_emp, _top_k_subspace_overlap, _flatten_valid
from .model_alignment import load_cached_model_alignment


# ---------------------------------------------------------------------------
# Eigenspectrum metrics
# ---------------------------------------------------------------------------

def eigenspectrum_metrics(C: np.ndarray) -> dict:
    """
    Compute standard eigenspectrum metrics for a covariance matrix C (NC, NC).

    Returns dict with:
        eigenvalues                 array of sorted (descending) positive eigenvalues
        total_variance
        top1_variance_fraction
        top2_variance_fraction
        top5_variance_fraction
        participation_ratio         (sum λ)² / sum(λ²)
        effective_dimensionality    exp(entropy of normalised spectrum)
        n_positive_eigenvalues
    """
    C_sym = 0.5 * (C + C.T)
    denom = float(np.linalg.norm(C_sym))
    asym = float(np.linalg.norm(C - C.T) / denom) if denom > 0 else 0.0
    if asym > 1e-8:
        warnings.warn(
            f"eigenspectrum_metrics: covariance not symmetric (relative asymmetry={asym:.3e}); "
            "using symmetrized matrix"
        )

    vals = np.linalg.eigvalsh(C_sym)     # ascending
    vals = vals[::-1].copy()         # descending
    vals = vals[vals > 0]

    if len(vals) == 0:
        return {k: float("nan") for k in (
            "total_variance", "top1_variance_fraction", "top2_variance_fraction",
            "top5_variance_fraction", "participation_ratio", "effective_dimensionality",
        )} | {"eigenvalues": np.array([]), "n_positive_eigenvalues": 0}

    total = float(vals.sum())
    top1 = float(vals[0] / total)
    top2 = float(vals[:2].sum() / total) if len(vals) >= 2 else top1
    top5 = float(vals[:5].sum() / total) if len(vals) >= 5 else float(vals.sum() / total)
    pr = float(total ** 2 / (vals ** 2).sum())
    p = vals / total
    eff_dim = float(np.exp(-np.sum(p * np.log(p + 1e-12))))

    return {
        "eigenvalues": vals,
        "total_variance": total,
        "top1_variance_fraction": top1,
        "top2_variance_fraction": top2,
        "top5_variance_fraction": top5,
        "participation_ratio": pr,
        "effective_dimensionality": eff_dim,
        "n_positive_eigenvalues": len(vals),
    }


# ---------------------------------------------------------------------------
# Alignment metrics
# ---------------------------------------------------------------------------

def subspace_overlap(C_emp: np.ndarray, C_ref: np.ndarray, k: int = 2) -> float:
    """Grassmann subspace overlap between top-k eigenspaces of C_emp and C_ref."""
    return _top_k_subspace_overlap(C_emp, C_ref, k=k)


def covariance_correlation(C1: np.ndarray, C2: np.ndarray) -> float:
    """Upper-triangle Pearson correlation between two covariance matrices."""
    NC = C1.shape[0]
    idx = np.triu_indices(NC, k=1)
    v1, v2 = C1[idx], C2[idx]
    if len(v1) < 5 or not (np.isfinite(v1).all() and np.isfinite(v2).all()):
        return float("nan")
    return float(stats.pearsonr(v1, v2)[0])


def projection_fraction(C_emp: np.ndarray, C_ref: np.ndarray, k: int = 2) -> float:
    """
    Fraction of C_emp variance captured by the top-k subspace of C_ref.

        frac = tr(U_ref^T C_emp U_ref) / tr(C_emp)
    """
    try:
        vals_ref, vecs_ref = np.linalg.eigh(C_ref)
        U_ref = vecs_ref[:, -k:]   # (NC, k) top-k eigenvectors
        num = float(np.trace(U_ref.T @ C_emp @ U_ref))
        denom = float(np.trace(C_emp))
        return num / denom if abs(denom) > 1e-12 else float("nan")
    except np.linalg.LinAlgError:
        return float("nan")


def alignment_metrics(C_emp: np.ndarray, C_ref: np.ndarray, k: int = 2) -> dict:
    """Compute all alignment metrics between C_emp and a reference C_ref."""
    return {
        f"subspace_overlap_k{k}": subspace_overlap(C_emp, C_ref, k),
        "covariance_correlation": covariance_correlation(C_emp, C_ref),
        f"projection_fraction_k{k}": projection_fraction(C_emp, C_ref, k),
    }


# ---------------------------------------------------------------------------
# Reliability ceiling and normalised alignment
# ---------------------------------------------------------------------------

def split_half_reliability_covariance(
    R_resid: np.ndarray,
    eye: np.ndarray,
    valid_mask: np.ndarray,
    n_splits: int = 100,
    k: int = 2,
    seed: int = 0,
) -> dict:
    """
    Split-half reliability of the B_emp covariance estimate.

    Splits trials into two random halves, fits B_emp independently on each,
    computes the Grassmann overlap and cosine similarity of B columns.

    Returns dict with:
        split_half_subspace_overlap  median across splits
        split_half_b_cosine          median signed cosine of top-2 columns
        reliability_ceiling          same as split_half_subspace_overlap
        split_half_values            array of per-split overlap values
    """
    NT = R_resid.shape[0]
    rng = np.random.default_rng(seed)

    overlaps = []
    cosines = []

    for _ in range(n_splits):
        perm = rng.permutation(NT)
        ha, hb = perm[:NT // 2], perm[NT // 2:]

        res_a = fit_b_emp(R_resid[ha], eye[ha], valid_mask[ha])
        res_b = fit_b_emp(R_resid[hb], eye[hb], valid_mask[hb])

        if not (res_a["fit_ok"] and res_b["fit_ok"]):
            continue

        B_a, B_b = res_a["B_emp"], res_b["B_emp"]  # (NC, 2)

        # Subspace overlap of the 2D subspace spanned by columns of B
        C_a = res_a["C_eye"]
        C_b = res_b["C_eye"]
        overlaps.append(_top_k_subspace_overlap(C_a, C_b, k=k))

        # Column cosines (sign-invariant)
        for col in range(B_a.shape[1]):
            n_a = np.linalg.norm(B_a[:, col])
            n_b = np.linalg.norm(B_b[:, col])
            if n_a > 1e-12 and n_b > 1e-12:
                cosines.append(abs(float(B_a[:, col] @ B_b[:, col]) / (n_a * n_b)))

    if not overlaps:
        return {
            "split_half_subspace_overlap": float("nan"),
            "split_half_b_cosine": float("nan"),
            "reliability_ceiling": float("nan"),
            "split_half_values": np.array([]),
            "status": "insufficient_data",
        }

    median_overlap = float(np.median(overlaps))
    return {
        "split_half_subspace_overlap": median_overlap,
        "split_half_b_cosine": float(np.median(cosines)) if cosines else float("nan"),
        "reliability_ceiling": median_overlap,
        "split_half_values": np.array(overlaps),
        "status": "ok",
    }


def reliability_normalized_excess(
    observed_metric: float,
    control_metric: float,
    reliability_ceiling: float,
    min_ceiling: float = 0.20,
) -> tuple[float, str]:
    """
    Compute ceiling-normalised excess (observed - control) and a status flag.

        norm = (observed_metric - control_metric) / reliability_ceiling

    Returns (norm_value, status_str).
    """
    if not np.isfinite(reliability_ceiling) or reliability_ceiling < min_ceiling:
        return float("nan"), "low_reliability_ceiling"
    if not (np.isfinite(observed_metric) and np.isfinite(control_metric)):
        return float("nan"), "metric_not_finite"
    norm = (observed_metric - control_metric) / reliability_ceiling
    return float(norm), "ok"


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def eye_shuffle_control(
    R_resid: np.ndarray,
    eye: np.ndarray,
    valid_mask: np.ndarray,
    n_shuffles: int = 200,
    k: int = 2,
    seed: int = 0,
) -> dict:
    """
    Eye-shuffle null distribution of B_emp subspace.

    For each shuffle: permute trial indices of eye data, refit B_emp, compute
    C_eye eigenspectrum and top2 variance fraction.

    Returns dict with:
        shuffle_top2_fractions  (n_shuffles,)
        shuffle_participation_ratios  (n_shuffles,)
        shuffle_subspace_overlaps  list of overlaps vs observed (if C_emp provided)
        p95_top2_fraction
        p95_participation_ratio
    """
    from .estimators import fit_b_emp as _fit_b_emp
    NT = R_resid.shape[0]
    rng = np.random.default_rng(seed)

    top2_fracs = []
    prs = []

    # Fit observed once for reference
    res_obs = _fit_b_emp(R_resid, eye, valid_mask)

    for _ in range(n_shuffles):
        perm = rng.permutation(NT)
        eye_perm = eye[perm]
        res = _fit_b_emp(R_resid, eye_perm, valid_mask)
        if not res["fit_ok"]:
            continue
        C = res["C_eye"]
        m = eigenspectrum_metrics(C)
        top2_fracs.append(m["top2_variance_fraction"])
        prs.append(m["participation_ratio"])

    if not top2_fracs:
        return {"status": "failed", "n_successful_shuffles": 0}

    arr_t2 = np.array(top2_fracs)
    arr_pr = np.array(prs)

    return {
        "shuffle_top2_fractions": arr_t2,
        "shuffle_participation_ratios": arr_pr,
        "p95_top2_fraction": float(np.percentile(arr_t2, 95)),
        "p95_participation_ratio": float(np.percentile(arr_pr, 95)),
        "n_successful_shuffles": len(top2_fracs),
        "status": "ok",
    }


def image_shuffle_control(
    R_raw: np.ndarray,
    eye: np.ndarray,
    image_ids: np.ndarray,
    valid_mask: np.ndarray,
    n_shuffles: int = 200,
    seed: int = 0,
) -> dict:
    """
    Image-shuffle null: permute image labels, recompute LOTO residuals from raw
    rates/counts, refit B_emp.

    Quantifies how much of the B_emp geometry depends on image identity
    (vs just eye statistics).

    Returns dict with shuffle distribution of top2 variance fraction and PR.
    """
    from .preprocessing import compute_loto_residuals
    from .estimators import fit_b_emp as _fit_b_emp
    rng = np.random.default_rng(seed)

    NT, T = image_ids.shape
    top2_fracs = []
    prs = []

    for _ in range(n_shuffles):
        # Permute image IDs per trial (preserves eye-image co-structure within trials
        # but breaks cross-trial image identity alignment)
        perm = rng.permutation(NT)
        img_shuf = image_ids[perm, :]

        R_resid_shuf = compute_loto_residuals(R_raw, img_shuf, valid_mask)
        res = _fit_b_emp(R_resid_shuf, eye, valid_mask)
        if not res["fit_ok"]:
            continue
        m = eigenspectrum_metrics(res["C_eye"])
        top2_fracs.append(m["top2_variance_fraction"])
        prs.append(m["participation_ratio"])

    if not top2_fracs:
        return {"status": "failed", "n_successful_shuffles": 0}

    arr_t2 = np.array(top2_fracs)
    arr_pr = np.array(prs)
    return {
        "shuffle_top2_fractions": arr_t2,
        "shuffle_participation_ratios": arr_pr,
        "p95_top2_fraction": float(np.percentile(arr_t2, 95)),
        "p95_participation_ratio": float(np.percentile(arr_pr, 95)),
        "n_successful_shuffles": len(top2_fracs),
        "status": "ok",
    }


def random_subspace_control(NC: int, k: int = 2, n_samples: int = 1000, seed: int = 0) -> dict:
    """
    Expected subspace overlap between two random k-dimensional subspaces in NC dimensions.
    Provides a theoretical floor for overlap comparisons.
    """
    rng = np.random.default_rng(seed)
    overlaps = []
    for _ in range(n_samples):
        A = rng.standard_normal((NC, k))
        B_ = rng.standard_normal((NC, k))
        Q_a, _ = np.linalg.qr(A)
        Q_b, _ = np.linalg.qr(B_)
        M = Q_a.T @ Q_b
        overlaps.append(float(np.trace(M @ M.T) / k))
    return {
        "random_subspace_mean": float(np.mean(overlaps)),
        "random_subspace_std": float(np.std(overlaps)),
        "random_subspace_p95": float(np.percentile(overlaps, 95)),
        "NC": NC,
        "k": k,
    }


# ---------------------------------------------------------------------------
# Shared vs image-specific geometry (Plan Section 6.2, Models 1-3)
# ---------------------------------------------------------------------------

def fit_shared_basis(
    R_resid: np.ndarray,
    eye: np.ndarray,
    valid_mask: np.ndarray,
) -> dict:
    """
    Model 1: global shared eye basis. R_resid = B_shared @ eye + error.
    Equivalent to global B_emp.
    """
    from .estimators import fit_b_emp as _fit
    res = _fit(R_resid, eye, valid_mask)
    res["model"] = "shared_basis"
    return res


def fit_image_specific_basis(
    R_resid: np.ndarray,
    eye: np.ndarray,
    image_ids: np.ndarray,
    valid_mask: np.ndarray,
    min_samples_per_image: int = 30,
) -> dict:
    """
    Model 2: per-image B_image fitted for each image with sufficient samples.

    Returns:
        B_images        dict[image_id -> (NC, 2)]
        C_eye_images    dict[image_id -> (NC, NC)]
        C_eye_mean      (NC, NC) mean across image contexts
        n_images_fit    int
        model           str
    """
    NT, T, NC = R_resid.shape
    B_images: dict[int, np.ndarray] = {}
    C_eye_images: dict[int, np.ndarray] = {}

    unique_images = np.unique(image_ids[valid_mask & (image_ids >= 0)])

    for img in unique_images:
        img_mask = valid_mask & (image_ids == img)  # (NT, T)
        n_valid = int(img_mask.sum())
        if n_valid < min_samples_per_image:
            continue

        # Build per-image data arrays: keep only trials/bins for this image
        trial_has = np.any(img_mask, axis=1)
        trial_inds = np.where(trial_has)[0]

        # Fit B_emp on this image's data only
        res = fit_b_emp(
            R_resid[trial_inds],
            eye[trial_inds],
            img_mask[trial_inds],
            min_samples=min_samples_per_image,
        )
        if not res["fit_ok"]:
            continue

        B_images[int(img)] = res["B_emp"]
        C_eye_images[int(img)] = res["C_eye"]

    n_fit = len(B_images)
    if n_fit == 0:
        return {
            "B_images": {},
            "C_eye_images": {},
            "C_eye_mean": np.full((NC, NC), np.nan),
            "n_images_fit": 0,
            "model": "image_specific_basis",
            "fit_ok": False,
        }

    C_stack = np.array(list(C_eye_images.values()))  # (n_fit, NC, NC)
    C_mean = C_stack.mean(axis=0)

    return {
        "B_images": B_images,
        "C_eye_images": C_eye_images,
        "C_eye_mean": C_mean,
        "n_images_fit": n_fit,
        "model": "image_specific_basis",
        "fit_ok": True,
    }


def compare_shared_vs_image_specific(
    R_resid: np.ndarray,
    eye: np.ndarray,
    image_ids: np.ndarray,
    valid_mask: np.ndarray,
    min_samples_per_image: int = 30,
) -> dict:
    """
    Compare shared basis and image-specific basis via explained variance.

    For each image with enough samples, compute:
      - R² of global B_shared prediction
      - R² of per-image B_image prediction (fitted on held-out half to avoid circularity)

    Returns summary dict.
    """
    NT, T, NC = R_resid.shape
    rng = np.random.default_rng(0)

    unique_images = np.unique(image_ids[valid_mask & (image_ids >= 0)])

    r2_shared_list = []
    r2_image_list = []
    subspace_delta_list = []

    for img in unique_images:
        img_mask = valid_mask & (image_ids == img)
        n_valid = int(img_mask.sum())
        if n_valid < min_samples_per_image * 2:
            continue

        trial_has = np.any(img_mask, axis=1)
        ti = np.where(trial_has)[0]
        if len(ti) < 6:
            continue

        # Split into train/test halves for image-specific model
        perm = rng.permutation(len(ti))
        train_i, test_i = ti[perm[: len(ti) // 2]], ti[perm[len(ti) // 2 :]]

        # Shared model fit on training half (across all images in training trials).
        res_shared_train = fit_b_emp(
            R_resid[train_i], eye[train_i], valid_mask[train_i],
            min_samples=min_samples_per_image,
        )
        res_img_train = fit_b_emp(
            R_resid[train_i], eye[train_i], img_mask[train_i],
            min_samples=min_samples_per_image,
        )
        if not res_img_train["fit_ok"] or not res_shared_train["fit_ok"]:
            continue

        # Evaluate on test half
        test_mask = img_mask[test_i]
        R_test, E_test, _ = _flatten_valid(R_resid[test_i], eye[test_i], test_mask)
        if R_test.shape[0] < 5:
            continue

        # Shared model prediction
        E_mean_shared = res_shared_train.get("E_mean")
        if E_mean_shared is None or not np.all(np.isfinite(E_mean_shared)):
            continue
        E_c_test = E_test - E_mean_shared
        B_T_shared = res_shared_train["B_emp"].T   # (2, NC)
        pred_shared = E_c_test @ B_T_shared  # (N_test, NC)

        # Image-specific model prediction (using train-fitted B, eval on test)
        B_T_img = res_img_train["B_emp"].T
        E_mean_img = res_img_train.get("E_mean")
        if E_mean_img is None or not np.all(np.isfinite(E_mean_img)):
            continue
        pred_img = (E_test - E_mean_img) @ B_T_img

        r2_shared_list.append(_r_squared(R_test, pred_shared))
        r2_image_list.append(_r_squared(R_test, pred_img))

        # Subspace delta against held-out image covariance.
        res_img_test = fit_b_emp(
            R_resid[test_i], eye[test_i], test_mask,
            min_samples=min_samples_per_image,
        )
        if res_img_test["fit_ok"]:
            C_test = res_img_test["C_eye"]
            C_shared = res_shared_train["C_eye"]
            C_img = res_img_train["C_eye"]
            delta = subspace_overlap(C_test, C_img, k=2) - subspace_overlap(C_test, C_shared, k=2)
            subspace_delta_list.append(delta)

    n_images = len(r2_shared_list)
    return {
        "n_images_evaluated": n_images,
        "mean_r2_shared": float(np.mean(r2_shared_list)) if r2_shared_list else float("nan"),
        "mean_r2_image_specific": float(np.mean(r2_image_list)) if r2_image_list else float("nan"),
        "r2_delta_image_vs_shared": float(
            np.mean(np.array(r2_image_list) - np.array(r2_shared_list))
        ) if r2_image_list else float("nan"),
        "mean_subspace_delta": float(np.mean(subspace_delta_list)) if subspace_delta_list else float("nan"),
        "r2_shared_list": r2_shared_list,
        "r2_image_list": r2_image_list,
        "model_1_vs_2_status": "ok" if n_images >= 3 else "insufficient_images",
    }

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Explained variance fraction (averaged over units)."""
    ss_res = np.nansum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.nansum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    ratio = np.divide(ss_res, ss_tot, out=np.full_like(ss_res, np.nan), where=ss_tot > 1e-12)
    r2_per_unit = 1.0 - ratio
    return float(np.nanmean(r2_per_unit))


# ---------------------------------------------------------------------------
# Full covariance geometry analysis for one session
# ---------------------------------------------------------------------------

def run_covariance_geometry(
    session: str,
    R_raw: np.ndarray,
    R_resid: np.ndarray,
    eye: np.ndarray,
    image_ids: np.ndarray,
    valid_mask: np.ndarray,
    unit_mask: np.ndarray,
    model_alignment_base_dir,
    n_splits: int = 100,
    n_shuffles: int = 200,
    seed: int = 0,
) -> dict:
    """
    Run the full Analysis 1 covariance geometry pipeline for one session.

    Steps:
      1. Fit B_emp (sensitivity estimator) and compute C_eye
      2. Eigenspectrum metrics on C_eye
      3. Split-half reliability (reliability ceiling)
      4. Shared vs image-specific geometry comparison
      5. Eye-shuffle and image-shuffle controls
    6. Cached model alignment loading (if available)
    7. Reliability-normalised empirical excess vs eye-shuffle p95
    8. Random subspace floor

    All downstream analyses use primary unit set (unit_mask).

    Returns a nested dict with all metrics.
    """
    from .estimators import b_emp_covariance, mcfarland_fem_covariance

    # Restrict to primary units
    R_p = R_resid[:, :, unit_mask]   # (NT, T, n_units)
    NC = R_p.shape[2]

    # ------------------------------------------------------------------
    # 1. B_emp (sensitivity estimator)
    # ------------------------------------------------------------------
    res_b_emp = b_emp_covariance(R_p, eye, valid_mask)
    C_eye = res_b_emp["C_eye"] if res_b_emp["fit_ok"] else np.full((NC, NC), np.nan)

    eigen_b_emp = eigenspectrum_metrics(C_eye) if res_b_emp["fit_ok"] else {
        k: float("nan") for k in ("total_variance", "top1_variance_fraction",
                                   "top2_variance_fraction", "top5_variance_fraction",
                                   "participation_ratio", "effective_dimensionality")
    }

    # ------------------------------------------------------------------
    # 2. McFarland estimator (primary)
    # ------------------------------------------------------------------
    res_mcfarland = mcfarland_fem_covariance(
        R_p,
        eye,
        image_ids,
        valid_mask,
        min_repeats=3,
        seed=seed,
    )
    mc_fit_mode = "strict_min_repeats_3"

    # If strict fit is unavailable because context support is low, run a
    # documented relaxed pass to distinguish data-support limits from geometry nulls.
    if not res_mcfarland["fit_ok"] and int(res_mcfarland.get("n_contexts", 0)) > 0:
        res_mc_relaxed = mcfarland_fem_covariance(
            R_p,
            eye,
            image_ids,
            valid_mask,
            min_repeats=2,
            seed=seed,
        )
        if int(res_mc_relaxed.get("n_contexts", 0)) >= int(res_mcfarland.get("n_contexts", 0)):
            res_mcfarland = res_mc_relaxed
            mc_fit_mode = "relaxed_min_repeats_2"

    mc_n_contexts = int(res_mcfarland.get("n_contexts", 0) or 0)
    if res_mcfarland["fit_ok"] and mc_fit_mode == "strict_min_repeats_3":
        mc_context_support_tier = "strong"
    elif res_mcfarland["fit_ok"] and mc_fit_mode == "relaxed_min_repeats_2":
        mc_context_support_tier = "limited_but_usable"
    elif mc_n_contexts > 0:
        mc_context_support_tier = "limited_insufficient"
    else:
        mc_context_support_tier = "absent"

    C_fem_mc = res_mcfarland["C_FEM"] if res_mcfarland["fit_ok"] else np.full((NC, NC), np.nan)
    eigen_mc = eigenspectrum_metrics(C_fem_mc) if res_mcfarland["fit_ok"] else {
        k: float("nan") for k in ("total_variance", "top1_variance_fraction",
                                   "top2_variance_fraction", "top5_variance_fraction",
                                   "participation_ratio", "effective_dimensionality")
    }

    # Estimator comparison
    from .estimators import compare_estimators
    estimator_cmp = compare_estimators(res_mcfarland, res_b_emp)

    # ------------------------------------------------------------------
    # 3. Split-half reliability (on B_emp; McFarland reliability deferred)
    # ------------------------------------------------------------------
    reliability = split_half_reliability_covariance(
        R_p, eye, valid_mask, n_splits=n_splits, seed=seed
    )

    # ------------------------------------------------------------------
    # 4. Shared vs image-specific geometry
    # ------------------------------------------------------------------
    shared_vs_img = compare_shared_vs_image_specific(R_p, eye, image_ids, valid_mask)

    # ------------------------------------------------------------------
    # 5. Controls
    # ------------------------------------------------------------------
    eye_ctrl = eye_shuffle_control(R_p, eye, valid_mask, n_shuffles=n_shuffles, seed=seed)
    R_raw_p = R_raw[:, :, unit_mask]
    img_ctrl = image_shuffle_control(R_raw_p, eye, image_ids, valid_mask, n_shuffles=n_shuffles, seed=seed)
    rand_ctrl = random_subspace_control(NC, k=2, n_samples=1000, seed=seed)

    # ------------------------------------------------------------------
    # 6. Model alignment from cached jacobian_predictive_framework outputs
    # ------------------------------------------------------------------
    model_align = load_cached_model_alignment(session, model_alignment_base_dir)

    # ------------------------------------------------------------------
    # 7. Reliability-normalised empirical excess (B_emp top2 vs eye-shuffle p95)
    # ------------------------------------------------------------------
    ceiling = reliability.get("reliability_ceiling", float("nan"))
    obs_top2 = eigen_b_emp.get("top2_variance_fraction", float("nan"))
    shuffle_p95_top2 = eye_ctrl.get("p95_top2_fraction", float("nan"))
    norm_align, norm_status = reliability_normalized_excess(
        obs_top2, shuffle_p95_top2, ceiling
    )

    # ------------------------------------------------------------------
    # Assemble session summary row
    # ------------------------------------------------------------------
    session_metrics = {
        "session": session,
        "n_units_primary": NC,
        "n_valid_bins": int(valid_mask.sum()),
        # B_emp eigenspectrum
        "b_emp_top2_fraction": eigen_b_emp.get("top2_variance_fraction"),
        "b_emp_participation_ratio": eigen_b_emp.get("participation_ratio"),
        "b_emp_total_variance": eigen_b_emp.get("total_variance"),
        "b_emp_effective_dim": eigen_b_emp.get("effective_dimensionality"),
        "b_emp_fit_ok": res_b_emp["fit_ok"],
        "b_emp_n_samples": res_b_emp.get("n_samples"),
        # McFarland eigenspectrum
        "mc_top2_fraction": eigen_mc.get("top2_variance_fraction"),
        "mc_participation_ratio": eigen_mc.get("participation_ratio"),
        "mc_n_contexts": mc_n_contexts,
        "mc_fit_ok": res_mcfarland["fit_ok"],
        "mc_fit_mode": mc_fit_mode,
        "mc_context_support_tier": mc_context_support_tier,
        # Estimator comparison
        "primary_vs_sensitivity_cov_corr": estimator_cmp.get("cov_correlation"),
        "primary_vs_sensitivity_subspace_overlap": estimator_cmp.get("subspace_overlap"),
        # Reliability
        "reliability_ceiling": ceiling,
        "split_half_subspace_overlap": reliability.get("split_half_subspace_overlap"),
        "split_half_b_cosine": reliability.get("split_half_b_cosine"),
        # Model alignment (primary basis selected from cached summary)
        "model_alignment": model_align.get("model_alignment", float("nan")),
        "model_shuffle_alignment": model_align.get("model_shuffle_alignment", float("nan")),
        "model_alignment_basis": model_align.get("primary_basis", ""),
        "model_alignment_source": model_align.get("source_path", ""),
        "model_alignment_n_windows": model_align.get("n_windows_model_alignment", float("nan")),
        "model_reliability_ceiling": model_align.get("reliability_ceiling_model", float("nan")),
        "ceiling_normalized_alignment": model_align.get("ceiling_normalized_alignment", float("nan")),
        "alignment_norm_status": model_align.get("alignment_norm_status", "not_available"),
        # Normalised empirical-vs-control excess (separate diagnostic)
        "ceiling_normalized_empirical_excess": norm_align,
        "empirical_excess_status": norm_status,
        "model_alignment_status": "ok" if model_align.get("available", False) else model_align.get("status", "not_available"),
        # Controls
        "eye_shuffle_p95_top2": eye_ctrl.get("p95_top2_fraction"),
        "image_shuffle_p95_top2": img_ctrl.get("p95_top2_fraction"),
        "random_subspace_mean_overlap": rand_ctrl.get("random_subspace_mean"),
        # Shared vs image-specific
        "mean_r2_shared": shared_vs_img.get("mean_r2_shared"),
        "mean_r2_image_specific": shared_vs_img.get("mean_r2_image_specific"),
        "shared_vs_img_r2_delta": shared_vs_img.get("r2_delta_image_vs_shared"),
        "n_images_evaluated": shared_vs_img.get("n_images_evaluated"),
        "model_1_vs_2_status": shared_vs_img.get("model_1_vs_2_status"),
    }

    return {
        "session_metrics": session_metrics,
        "eigenspectrum_b_emp": eigen_b_emp,
        "eigenspectrum_mcfarland": eigen_mc,
        "reliability": reliability,
        "shared_vs_image_specific": shared_vs_img,
        "eye_shuffle_control": eye_ctrl,
        "image_shuffle_control": img_ctrl,
        "random_subspace_control": rand_ctrl,
        "b_emp_result": res_b_emp,
        "mcfarland_result": res_mcfarland,
        "model_alignment": model_align,
        "estimator_comparison": estimator_cmp,
    }
