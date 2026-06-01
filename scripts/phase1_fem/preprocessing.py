"""
Phase 1 FEM/V1 covariance analysis: shared preprocessing and QC.

Handles data loading, unit selection, residual construction, and session-level
QC metrics. All downstream analyses (covariance geometry, noise correlations,
aggregation scaling) call into this module.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from eval.fixrsvp import get_fixrsvp_data


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_session(
    subject: str,
    date: str,
    dataset_configs_path: str,
    use_cached: bool = True,
    fixation_degree_radius: float = 1.0,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Load fixRSVP session data via eval.fixrsvp.get_fixrsvp_data.

    Returns a dict with keys:
        robs            (NT, T, NC)  spike counts (NaN for invalid bins)
        dfs             (NT, T, NC)  data validity flags
        eyepos          (NT, T, 2)   eye position in degrees
        image_ids       (NT, T)      0-indexed image ID (-1 = invalid)
        fix_dur         (NT,)        fixation duration in bins
        trial_t_bins    list[array]  time bin centers per trial (seconds)
        cids            list[int]    cluster IDs
        bin_ms          float        inferred bin size in milliseconds
        session         str          e.g. 'Allen_2022-02-16'
    """
    data = get_fixrsvp_data(
        subject,
        date,
        dataset_configs_path,
        use_cached_data=use_cached,
        fixation_degree_radius=fixation_degree_radius,
        verbose=verbose,
    )

    robs = data["robs"]          # (NT, T, NC)
    dfs = data["dfs"]            # (NT, T, NC)
    eyepos = data["eyepos"]      # (NT, T, 2)
    image_ids = data["image_ids"]  # (NT, T)
    fix_dur = data["fix_dur"]    # (NT,)
    trial_t_bins = data.get("trial_t_bins")

    bin_ms = _infer_bin_ms(trial_t_bins)

    return {
        "robs": robs,
        "dfs": dfs,
        "eyepos": eyepos,
        "image_ids": image_ids,
        "fix_dur": fix_dur,
        "trial_t_bins": trial_t_bins,
        "cids": data.get("cids", []),
        "bin_ms": bin_ms,
        "session": f"{subject}_{date}",
    }


def _infer_bin_ms(trial_t_bins) -> float:
    if trial_t_bins is None:
        return float("nan")
    diffs = []
    for arr in trial_t_bins:
        v = np.asarray(arr, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size >= 3:
            dv = np.diff(v)
            dv = dv[(dv > 0) & np.isfinite(dv)]
            if dv.size:
                diffs.append(dv)
    if not diffs:
        return float("nan")
    return float(np.median(np.concatenate(diffs)) * 1000.0)


# ---------------------------------------------------------------------------
# Valid-bin mask
# ---------------------------------------------------------------------------

def build_valid_mask(robs: np.ndarray, dfs: np.ndarray, eyepos: np.ndarray, image_ids: np.ndarray) -> np.ndarray:
    """
    Boolean mask (NT, T): True where eye position is finite, image_id >= 0,
    and at least one unit has finite response and dfs > 0.

    This mask is intentionally unit-agnostic at the trial-time level; per-unit
    validity is handled downstream during unit selection and model fitting.
    """
    r_finite_any = np.any(np.isfinite(robs), axis=-1)        # (NT, T)
    e_finite = np.all(np.isfinite(eyepos), axis=-1)      # (NT, T)
    d_valid_any = np.any(dfs > 0, axis=-1)                   # (NT, T)
    img_valid = image_ids >= 0                            # (NT, T)
    return r_finite_any & e_finite & d_valid_any & img_valid


# ---------------------------------------------------------------------------
# Unit selection
# ---------------------------------------------------------------------------

def select_primary_units(
    robs: np.ndarray,
    dfs: np.ndarray,
    valid_mask: np.ndarray,
    bin_ms: float,
    min_rate_hz: float = 0.5,
    min_valid_fraction: float = 0.2,
) -> tuple[np.ndarray, dict]:
    """
    Select units that meet minimum firing rate and data quality thresholds.

    Returns:
        unit_mask   (NC,) boolean
        unit_stats  dict with per-unit metrics
    """
    NT, T, NC = robs.shape
    if np.isfinite(bin_ms):
        bin_s = bin_ms / 1000.0
    else:
        warnings.warn(
            "select_primary_units: bin_ms is NaN; using 120Hz fallback (8.33ms) "
            "for rate conversion"
        )
        bin_s = 1.0 / 120.0

    # Mean rate per unit (over valid bins only)
    valid_3d = valid_mask[:, :, np.newaxis]  # (NT, T, 1)
    n_valid_per_unit = (valid_3d * np.isfinite(robs)).sum(axis=(0, 1))  # (NC,)
    total_spikes = np.nansum(robs * valid_3d, axis=(0, 1))              # (NC,)
    mean_rate_hz = np.where(n_valid_per_unit > 0,
                            total_spikes / (n_valid_per_unit * bin_s),
                            0.0)

    # Valid bin fraction per unit
    total_bins = NT * T
    valid_fraction = n_valid_per_unit / max(total_bins, 1)

    unit_mask = (mean_rate_hz >= min_rate_hz) & (valid_fraction >= min_valid_fraction)

    return unit_mask, {
        "mean_rate_hz": mean_rate_hz,
        "valid_fraction": valid_fraction,
        "n_valid_bins": n_valid_per_unit,
    }


# ---------------------------------------------------------------------------
# Residual construction
# ---------------------------------------------------------------------------

def compute_loto_residuals(
    robs: np.ndarray,
    image_ids: np.ndarray,
    valid_mask: np.ndarray,
    min_repeats: int = 2,
) -> np.ndarray:
    """
    Leave-one-trial-out image/time residuals.

    For each (trial, time) pair (i, t):
        mean = mean of robs[j, t] over all j ≠ i where
               image_ids[j, t] == image_ids[i, t] and valid_mask[j, t]
        R_resid[i, t] = robs[i, t] - mean

    Pairs where there are fewer than min_repeats OTHER valid trials with the
    same image stay NaN (no reliable mean to subtract).

    Returns R_resid (NT, T, NC), NaN where undefined.
    """
    NT, T, NC = robs.shape
    R_resid = np.full_like(robs, np.nan)

    for t in range(T):
        vm_t = valid_mask[:, t]          # (NT,)
        if vm_t.sum() < min_repeats:
            continue
        imgs_t = image_ids[:, t]         # (NT,)

        unique_imgs = np.unique(imgs_t[vm_t & (imgs_t >= 0)])
        for img in unique_imgs:
            mask = vm_t & (imgs_t == img)
            n = int(mask.sum())
            if n < min_repeats:
                continue
            idx = np.where(mask)[0]
            R_sub = robs[idx, t, :]  # (n, NC)
            total = R_sub.sum(axis=0)

            # LOTO: subtract mean of the OTHER (n-1) trials
            loto_mean = (total - R_sub) / (n - 1)
            R_resid[idx, t, :] = R_sub - loto_mean

    return R_resid


def compute_raw_normalized(
    robs: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Raw activity normalized by each unit's mean and std across valid bins.
    Returns R_norm (NT, T, NC), NaN where not valid.
    """
    NT, T, NC = robs.shape
    v3 = valid_mask[:, :, np.newaxis]
    # Per-unit mean and std over valid bins
    R_valid = np.where(v3, robs, np.nan)
    mu = np.nanmean(R_valid.reshape(-1, NC), axis=0)   # (NC,)
    sd = np.nanstd(R_valid.reshape(-1, NC), axis=0)    # (NC,)
    sd = np.where(sd < eps, eps, sd)

    R_norm = np.where(v3, (robs - mu) / sd, np.nan)
    return R_norm


# ---------------------------------------------------------------------------
# Image repeat support
# ---------------------------------------------------------------------------

def compute_image_repeat_support(
    image_ids: np.ndarray,
    valid_mask: np.ndarray,
) -> list[dict]:
    """
    For each (image_id, time_bin) cell, count how many valid trials repeat it.

    Returns a list of dicts with: image_id, time_bin, n_repeats.
    """
    NT, T = image_ids.shape
    records: list[dict] = []
    for t in range(T):
        vm_t = valid_mask[:, t]
        imgs_t = image_ids[:, t]
        unique_imgs = np.unique(imgs_t[vm_t & (imgs_t >= 0)])
        for img in unique_imgs:
            n = int((vm_t & (imgs_t == img)).sum())
            records.append({"image_id": int(img), "time_bin": t, "n_repeats": n})
    return records


# ---------------------------------------------------------------------------
# Session QC
# ---------------------------------------------------------------------------

def compute_session_qc(
    session: str,
    robs: np.ndarray,
    eyepos: np.ndarray,
    image_ids: np.ndarray,
    dfs: np.ndarray,
    valid_mask: np.ndarray,
    unit_mask: np.ndarray,
    unit_stats: dict,
    bin_ms: float,
    image_repeat_records: list[dict] | None = None,
    analysis_recommendation: str = "ok",
) -> dict:
    """Return a flat dict of session-level QC metrics."""
    NT, T, NC = robs.shape

    n_valid_bins_total = int(valid_mask.sum())
    valid_bin_fraction = n_valid_bins_total / max(NT * T, 1)

    # Image-level statistics
    if image_repeat_records is None:
        image_repeat_records = compute_image_repeat_support(image_ids, valid_mask)

    n_repeats_arr = np.array([r["n_repeats"] for r in image_repeat_records], dtype=float)
    n_unique_images = int(len(set(r["image_id"] for r in image_repeat_records)))
    frac_ge2 = float(np.mean(n_repeats_arr >= 2)) if len(n_repeats_arr) else float("nan")
    frac_ge3 = float(np.mean(n_repeats_arr >= 3)) if len(n_repeats_arr) else float("nan")
    median_repeats = float(np.median(n_repeats_arr)) if len(n_repeats_arr) else float("nan")

    # Eye statistics
    eye_valid = np.isfinite(eyepos[:, :, 0])
    eye_x_std = float(np.nanstd(eyepos[:, :, 0]))
    eye_y_std = float(np.nanstd(eyepos[:, :, 1]))
    eye_pos_valid_frac = float(eye_valid.mean())

    # Image valid fraction
    image_valid_fraction = float((image_ids >= 0).mean())

    # Firing rate summary across primary units
    rates = unit_stats["mean_rate_hz"][unit_mask]
    mean_rate_median = float(np.median(rates)) if len(rates) else float("nan")
    mean_rate_p10 = float(np.percentile(rates, 10)) if len(rates) else float("nan")
    mean_rate_p90 = float(np.percentile(rates, 90)) if len(rates) else float("nan")

    return {
        "session": session,
        "n_trials": NT,
        "n_time_bins": T,
        "n_units_total": NC,
        "n_units_primary": int(unit_mask.sum()),
        "valid_bin_fraction": valid_bin_fraction,
        "image_valid_fraction": image_valid_fraction,
        "n_unique_images": n_unique_images,
        "median_repeats_per_image_time_cell": median_repeats,
        "frac_image_time_cells_ge2_repeats": frac_ge2,
        "frac_image_time_cells_ge3_repeats": frac_ge3,
        "mean_rate_median": mean_rate_median,
        "mean_rate_p10": mean_rate_p10,
        "mean_rate_p90": mean_rate_p90,
        "eye_x_std": eye_x_std,
        "eye_y_std": eye_y_std,
        "eye_position_valid_fraction": eye_pos_valid_frac,
        "bin_ms": bin_ms,
        "analysis_recommendation": analysis_recommendation,
    }


def compute_unit_qc(
    session: str,
    unit_stats: dict,
    unit_mask: np.ndarray,
    cids: list,
) -> list[dict]:
    """Return per-unit QC rows."""
    NC = len(unit_mask)
    rows = []
    for n in range(NC):
        rows.append({
            "session": session,
            "unit_index": n,
            "cid": cids[n] if n < len(cids) else n,
            "mean_rate_hz": float(unit_stats["mean_rate_hz"][n]),
            "valid_fraction": float(unit_stats["valid_fraction"][n]),
            "n_valid_bins": int(unit_stats["n_valid_bins"][n]),
            "in_primary_set": bool(unit_mask[n]),
        })
    return rows
