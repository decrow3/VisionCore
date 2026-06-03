from __future__ import annotations

import argparse
import csv
import itertools
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from eval.fixrsvp import get_fixrsvp_data

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.shared_transformation_geometry.utils import DEFAULT_OUT_ROOT, harmonize_fixrsvp_arrays  # type: ignore
    from declan.twin_covariance_structure.run_a3_fixrsvp_audit import _predict_twin_rates  # type: ignore
else:
    from .utils import DEFAULT_OUT_ROOT, harmonize_fixrsvp_arrays
    from declan.twin_covariance_structure.run_a3_fixrsvp_audit import _predict_twin_rates  # type: ignore


CONTROL_METRICS = ("pixel_correlation", "rms_contrast_difference", "fourier_amplitude_similarity")


def _infer_bin_ms(data: dict[str, Any], fallback_ms: float = 10.0) -> float:
    trial_t_bins = data.get("trial_t_bins")
    if trial_t_bins is None or not isinstance(trial_t_bins, (list, tuple)):
        return float(fallback_ms)
    diffs: list[np.ndarray] = []
    for arr in trial_t_bins:
        vec = np.asarray(arr, dtype=np.float64)
        vec = vec[np.isfinite(vec)]
        if vec.size < 3:
            continue
        dv = np.diff(vec)
        dv = dv[(np.isfinite(dv)) & (dv > 0)]
        if dv.size:
            diffs.append(dv)
    if not diffs:
        return float(fallback_ms)
    return float(np.median(np.concatenate(diffs)) * 1000.0)


def _nan_gauss_time(arr: np.ndarray, sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return np.asarray(arr, dtype=np.float64)
    arrf = np.asarray(arr, dtype=np.float64)
    try:
        from scipy.ndimage import gaussian_filter1d  # local import to keep module import light
    except Exception:
        return arrf
    arr0 = np.where(np.isnan(arrf), 0.0, arrf)
    w = (~np.isnan(arrf)).astype(np.float64)
    f_arr = gaussian_filter1d(arr0, sigma_bins, axis=1, mode="nearest")
    f_w = gaussian_filter1d(w, sigma_bins, axis=1, mode="nearest")
    return np.divide(f_arr, f_w, out=np.full_like(f_arr, np.nan), where=f_w > 1e-8)


def _detect_saccade_like_events(
    *,
    eyepos_xy: np.ndarray,
    valid_bin_mask: np.ndarray,
    image_ids: np.ndarray,
    trial_ids: np.ndarray,
    bin_s: float,
    smooth_sigma_bins: float,
    speed_percentile: float,
    amp_thresh_deg: float,
    refrac_ms: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ex = _nan_gauss_time(eyepos_xy[:, :, 0], smooth_sigma_bins)
    ey = _nan_gauss_time(eyepos_xy[:, :, 1], smooth_sigma_bins)
    dx = np.diff(ex, axis=1)
    dy = np.diff(ey, axis=1)
    speed = np.sqrt(dx * dx + dy * dy) / max(bin_s, 1e-6)
    step_valid = valid_bin_mask[:, 1:] & valid_bin_mask[:, :-1]
    speed_valid = speed[step_valid & np.isfinite(speed)]
    if speed_valid.size == 0:
        return [], {"n_detected": 0, "speed_threshold_deg_s": float("nan")}

    speed_thr = float(np.percentile(speed_valid, speed_percentile))
    refrac_bins = max(1, int(round((refrac_ms / 1000.0) / max(bin_s, 1e-6))))
    events: list[dict[str, Any]] = []

    for tr in range(eyepos_xy.shape[0]):
        s = speed[tr]
        if not np.isfinite(s).any():
            continue
        local_max = (s > np.roll(s, 1)) & (s >= np.roll(s, -1))
        cand = np.where((s >= speed_thr) & local_max)[0]
        if cand.size == 0:
            continue

        keep: list[int] = []
        for t0 in cand:
            if keep and (t0 - keep[-1] < refrac_bins):
                continue
            left = int(t0)
            while left > 0 and np.isfinite(s[left]) and s[left] > 0.5 * speed_thr:
                left -= 1
            right = int(t0)
            while right < s.shape[0] - 1 and np.isfinite(s[right]) and s[right] > 0.5 * speed_thr:
                right += 1
            onset_bin = max(left, 0)
            offset_bin = min(right + 1, eyepos_xy.shape[1] - 1)
            if onset_bin >= offset_bin:
                continue
            if not (valid_bin_mask[tr, onset_bin] and valid_bin_mask[tr, offset_bin]):
                continue
            disp = float(np.hypot(ex[tr, offset_bin] - ex[tr, onset_bin], ey[tr, offset_bin] - ey[tr, onset_bin]))
            if (not np.isfinite(disp)) or (disp < amp_thresh_deg):
                continue
            peak_bin = int(t0 + 1)
            image_id = int(image_ids[tr, peak_bin]) if peak_bin < image_ids.shape[1] else -1
            events.append(
                {
                    "trial_index": int(tr),
                    "trial_id": int(trial_ids[tr]),
                    "image_id": int(image_id),
                    "onset_bin": int(onset_bin),
                    "peak_bin": int(peak_bin),
                    "offset_bin": int(offset_bin),
                    "amplitude_deg": float(disp),
                    "peak_speed_deg_s": float(s[t0]),
                }
            )
            keep.append(int(t0))

    return events, {"n_detected": int(len(events)), "speed_threshold_deg_s": float(speed_thr)}


def _image_support_counts(valid_mask: np.ndarray, image_ids: np.ndarray) -> dict[int, int]:
    if valid_mask.size == 0:
        return {}
    ids = np.asarray(image_ids[valid_mask], dtype=np.int64)
    if ids.size == 0:
        return {}
    uniq, counts = np.unique(ids, return_counts=True)
    return {int(i): int(c) for i, c in zip(uniq, counts, strict=False)}


def build_drift_only_valid_mask(
    *,
    rates: np.ndarray,
    eyepos: np.ndarray,
    image_ids: np.ndarray,
    data: dict[str, Any],
    drift_only: bool,
    eye_smooth_sigma_bins: float,
    speed_percentile: float,
    amp_thresh_deg: float,
    refrac_ms: float,
    exclusion_pre_ms: float,
    exclusion_post_ms: float,
    fallback_bin_ms: float = 10.0,
) -> tuple[np.ndarray, dict[str, object]]:
    base_valid = np.isfinite(rates).all(axis=2) & np.isfinite(eyepos).all(axis=2) & (image_ids >= 0)
    pre_counts = _image_support_counts(base_valid, image_ids)

    n_total = int(np.sum(base_valid))
    n_excluded = 0
    events: list[dict[str, Any]] = []
    speed_threshold = float("nan")
    inferred_bin_ms = _infer_bin_ms(data, fallback_ms=float(fallback_bin_ms))
    bin_s = max(float(inferred_bin_ms) / 1000.0, 1e-6)

    if drift_only:
        trial_ids = np.arange(image_ids.shape[0], dtype=np.int64)
        events, detect_meta = _detect_saccade_like_events(
            eyepos_xy=np.asarray(eyepos, dtype=np.float64),
            valid_bin_mask=base_valid,
            image_ids=np.asarray(image_ids, dtype=np.int64),
            trial_ids=trial_ids,
            bin_s=bin_s,
            smooth_sigma_bins=float(eye_smooth_sigma_bins),
            speed_percentile=float(speed_percentile),
            amp_thresh_deg=float(amp_thresh_deg),
            refrac_ms=float(refrac_ms),
        )
        speed_threshold = float(detect_meta.get("speed_threshold_deg_s", float("nan")))
        exclude = np.zeros_like(base_valid, dtype=bool)
        pre_bins = max(1, int(round((float(exclusion_pre_ms) / 1000.0) / bin_s)))
        post_bins = max(1, int(round((float(exclusion_post_ms) / 1000.0) / bin_s)))
        for e in events:
            tr = int(e["trial_index"])
            pk = int(e["peak_bin"])
            st = max(0, pk - pre_bins)
            en = min(base_valid.shape[1], pk + post_bins)
            exclude[tr, st:en] = True
        final_valid = base_valid & (~exclude)
        n_excluded = int(np.sum(base_valid & exclude))
    else:
        final_valid = base_valid

    post_counts = _image_support_counts(final_valid, image_ids)
    support_meta: dict[str, object] = {
        "drift_only": bool(drift_only),
        "inferred_bin_ms": float(inferred_bin_ms),
        "drift_speed_threshold_deg_s": float(speed_threshold),
        "drift_n_events_detected": int(len(events)),
        "n_valid_samples_before_exclusion": int(n_total),
        "n_valid_samples_excluded": int(n_excluded),
        "n_valid_samples_after_exclusion": int(np.sum(final_valid)),
        "fraction_valid_samples_after_exclusion": (float(np.sum(final_valid)) / float(n_total)) if n_total > 0 else float("nan"),
        "n_images_with_samples_before_exclusion": int(len(pre_counts)),
        "n_images_with_samples_after_exclusion": int(len(post_counts)),
        "image_support_before_exclusion_min": float(np.min(list(pre_counts.values()))) if pre_counts else float("nan"),
        "image_support_before_exclusion_median": float(np.median(list(pre_counts.values()))) if pre_counts else float("nan"),
        "image_support_after_exclusion_min": float(np.min(list(post_counts.values()))) if post_counts else float("nan"),
        "image_support_after_exclusion_median": float(np.median(list(post_counts.values()))) if post_counts else float("nan"),
        "image_support_counts_before_exclusion": pre_counts,
        "image_support_counts_after_exclusion": post_counts,
    }
    return final_valid, support_meta


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fit_tangent_map(rates: np.ndarray, dxdy: np.ndarray, ridge_alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(dxdy, dtype=np.float64)
    y = np.asarray(rates, dtype=np.float64)
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    xc = x - x_mean

    xtx = xc.T @ xc
    reg = ridge_alpha * np.eye(xtx.shape[0], dtype=np.float64)
    b = np.linalg.solve(xtx + reg, xc.T @ (y - y_mean))
    pred = y_mean + xc @ b

    ss_res = np.sum((y - pred) ** 2, axis=0)
    ss_tot = np.sum((y - y_mean) ** 2, axis=0) + 1e-12
    r2 = 1.0 - (ss_res / ss_tot)

    bx = b[0, :].copy()
    by = b[1, :].copy()
    j = np.stack([bx, by], axis=1)
    return bx, by, j, r2


def _fit_unitwise_nuisance_residual(y: np.ndarray, nuisance_cols: list[np.ndarray]) -> np.ndarray:
    if not nuisance_cols:
        return np.asarray(y, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    cols = [np.ones(yy.shape[0], dtype=np.float64)]
    for c in nuisance_cols:
        cc = np.asarray(c, dtype=np.float64).ravel()
        if cc.shape[0] != yy.shape[0]:
            continue
        if not np.all(np.isfinite(cc)):
            continue
        if float(np.std(cc)) <= 1e-12:
            continue
        cols.append(cc)
    if len(cols) == 1:
        return yy
    x = np.stack(cols, axis=1)
    beta, *_ = np.linalg.lstsq(x, yy, rcond=None)
    return yy - (x @ beta)


def _project_responses_out_axes(y: np.ndarray, axes: list[np.ndarray]) -> np.ndarray:
    yy = np.asarray(y, dtype=np.float64)
    if yy.ndim != 2 or not axes:
        return yy

    good_axes: list[np.ndarray] = []
    for a in axes:
        aa = np.asarray(a, dtype=np.float64).ravel()
        if aa.shape[0] != yy.shape[1]:
            continue
        n = float(np.linalg.norm(aa))
        if (not np.isfinite(n)) or n <= 1e-12:
            continue
        good_axes.append(aa / n)
    if not good_axes:
        return yy

    u = np.stack(good_axes, axis=1)  # (n_units, k)
    q, _ = np.linalg.qr(u)
    q = np.asarray(q, dtype=np.float64)
    return yy - (yy @ q) @ q.T


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 0.0 or bn <= 0.0:
        return float("nan")
    return float(np.dot(a, b) / (an * bn))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    keep = np.isfinite(aa) & np.isfinite(bb)
    if int(np.sum(keep)) < 2:
        return float("nan")
    aa = aa[keep] - float(np.mean(aa[keep]))
    bb = bb[keep] - float(np.mean(bb[keep]))
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den <= 0.0:
        return float("nan")
    return float(np.dot(aa, bb) / den)


def _j_diagnostics(j: np.ndarray, rank_tol_rel: float) -> tuple[np.ndarray, int, np.ndarray, float, float]:
    u, s, _ = np.linalg.svd(np.asarray(j, dtype=np.float64), full_matrices=False)
    if s.size == 0 or s[0] <= 0.0:
        return np.zeros((j.shape[0], 0), dtype=np.float64), 0, s, 0.0, 0.0
    tol = float(rank_tol_rel) * float(s[0])
    rank = int(np.sum(s > tol))
    basis = u[:, :rank].copy() if rank > 0 else np.zeros((j.shape[0], 0), dtype=np.float64)
    e = s * s
    e_sum = float(np.sum(e)) + 1e-12
    frac1 = float(e[0] / e_sum) if e.size >= 1 else 0.0
    frac2 = float((e[0] + (e[1] if e.size >= 2 else 0.0)) / e_sum)
    return basis, rank, s, frac1, frac2


def _subspace_overlap(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    if basis_a.shape[1] == 0 or basis_b.shape[1] == 0:
        return float("nan")
    s = np.linalg.svd(basis_a.T @ basis_b, compute_uv=False)
    return float(np.mean(s**2))


def _random_map_with_norms(n_units: int, norm_x: float, norm_y: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bx = rng.standard_normal(n_units)
    by = rng.standard_normal(n_units)
    bx *= (norm_x / (np.linalg.norm(bx) + 1e-12))
    by *= (norm_y / (np.linalg.norm(by) + 1e-12))
    j = np.stack([bx, by], axis=1)
    return bx, by, j


def _bootstrap_ci(vals: np.ndarray, qlo: float = 0.025, qhi: float = 0.975) -> tuple[float, float]:
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(vals, qlo)), float(np.quantile(vals, qhi))


def _compute_image_similarity(img_a: np.ndarray, img_b: np.ndarray) -> dict[str, float]:
    a = np.asarray(img_a, dtype=np.float64)
    b = np.asarray(img_b, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    af = a.ravel()
    bf = b.ravel()
    pixel_corr = _pearson(af, bf)
    rms_diff = float(abs(np.std(af) - np.std(bf)))

    fa = np.abs(np.fft.rfft2(a))
    fb = np.abs(np.fft.rfft2(b))
    fourier_amp_corr = _pearson(fa.ravel(), fb.ravel())
    return {
        "pixel_correlation": pixel_corr,
        "rms_contrast_difference": rms_diff,
        "fourier_amplitude_similarity": fourier_amp_corr,
    }


def _image_bootstrap_weighted_mean(
    pair_rows: list[dict[str, object]],
    value_key: str,
    image_ids: list[int],
    *,
    seed: int,
    n_bootstrap: int,
) -> np.ndarray:
    if not pair_rows or not image_ids:
        return np.asarray([], dtype=np.float64)

    vals = np.asarray([float(r[value_key]) for r in pair_rows], dtype=np.float64)
    img_i = np.asarray([int(r["image_i"]) for r in pair_rows], dtype=np.int64)
    img_j = np.asarray([int(r["image_j"]) for r in pair_rows], dtype=np.int64)

    rng = np.random.default_rng(seed)
    boots = []
    base_images = np.asarray(image_ids, dtype=np.int64)
    for _ in range(int(n_bootstrap)):
        sampled = rng.choice(base_images, size=base_images.size, replace=True)
        uniq, counts = np.unique(sampled, return_counts=True)
        count_map = {int(u): int(c) for u, c in zip(uniq, counts, strict=False)}

        weights = np.asarray(
            [float(count_map.get(int(i), 0) * count_map.get(int(j), 0)) for i, j in zip(img_i, img_j, strict=False)],
            dtype=np.float64,
        )
        keep = np.isfinite(vals) & (weights > 0)
        if not np.any(keep):
            continue
        boots.append(float(np.sum(vals[keep] * weights[keep]) / np.sum(weights[keep])))
    return np.asarray(boots, dtype=np.float64)


def _image_bootstrap_adjusted_intercept(
    pair_rows: list[dict[str, object]],
    value_key: str,
    control_key: str,
    image_ids: list[int],
    *,
    seed: int,
    n_bootstrap: int,
) -> np.ndarray:
    if not pair_rows or not image_ids:
        return np.asarray([], dtype=np.float64)

    y = np.asarray([float(r[value_key]) for r in pair_rows], dtype=np.float64)
    x = np.asarray([float(r[control_key]) for r in pair_rows], dtype=np.float64)
    img_i = np.asarray([int(r["image_i"]) for r in pair_rows], dtype=np.int64)
    img_j = np.asarray([int(r["image_j"]) for r in pair_rows], dtype=np.int64)
    keep0 = np.isfinite(y) & np.isfinite(x)
    if not np.any(keep0):
        return np.asarray([], dtype=np.float64)

    y = y[keep0]
    x = x[keep0]
    img_i = img_i[keep0]
    img_j = img_j[keep0]

    rng = np.random.default_rng(seed)
    boots = []
    base_images = np.asarray(image_ids, dtype=np.int64)
    for _ in range(int(n_bootstrap)):
        sampled = rng.choice(base_images, size=base_images.size, replace=True)
        uniq, counts = np.unique(sampled, return_counts=True)
        count_map = {int(u): int(c) for u, c in zip(uniq, counts, strict=False)}

        w = np.asarray(
            [float(count_map.get(int(i), 0) * count_map.get(int(j), 0)) for i, j in zip(img_i, img_j, strict=False)],
            dtype=np.float64,
        )
        keep = w > 0
        if int(np.sum(keep)) < 3:
            continue
        yk = y[keep]
        xk = x[keep]
        wk = w[keep]
        xbar = float(np.sum(wk * xk) / np.sum(wk))
        xc = xk - xbar
        den = float(np.sum(wk * xc * xc))
        if den > 0.0:
            b1 = float(np.sum(wk * xc * yk) / den)
        else:
            b1 = 0.0
        b0 = float(np.sum(wk * (yk - b1 * xc)) / np.sum(wk))
        boots.append(b0)

    return np.asarray(boots, dtype=np.float64)


def _fit_adjusted_intercept(pair_rows: list[dict[str, object]], value_key: str, control_key: str) -> tuple[float, float, int]:
    vals = [
        (float(r[value_key]), float(r[control_key]))
        for r in pair_rows
        if np.isfinite(float(r[value_key])) and np.isfinite(float(r[control_key]))
    ]
    if len(vals) < 3:
        return float("nan"), float("nan"), 0
    y = np.asarray([v[0] for v in vals], dtype=np.float64)
    x = np.asarray([v[1] for v in vals], dtype=np.float64)
    xc = x - float(np.mean(x))
    X = np.stack([np.ones_like(xc), xc], axis=1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[0]), float(beta[1]), int(y.size)


def _fit_low_similarity(pair_rows: list[dict[str, object]], value_key: str, control_key: str) -> tuple[list[dict[str, object]], float, int]:
    vals = np.asarray([float(r[control_key]) for r in pair_rows if np.isfinite(float(r[control_key]))], dtype=np.float64)
    if vals.size == 0:
        return [], float("nan"), 0
    thr = float(np.median(vals))
    low_rows = [
        r
        for r in pair_rows
        if np.isfinite(float(r[control_key])) and np.isfinite(float(r[value_key])) and float(r[control_key]) <= thr
    ]
    if not low_rows:
        return [], thr, 0
    arr = np.asarray([float(r[value_key]) for r in low_rows], dtype=np.float64)
    return low_rows, thr, int(arr.size)


def _safe_unit_axis(a: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float64).ravel()
    n = float(np.linalg.norm(aa))
    if (not np.isfinite(n)) or n <= 0.0:
        return np.zeros_like(aa, dtype=np.float64)
    return aa / n


def _extract_template_2d(stim_slice: np.ndarray) -> np.ndarray | None:
    st = np.asarray(stim_slice)
    if st.size == 0:
        return None
    if st.ndim == 2:
        return np.asarray(st, dtype=np.float64)
    if st.ndim == 3:
        # (samples, H, W)
        return np.asarray(np.nanmean(st, axis=0), dtype=np.float64)
    if st.ndim == 4:
        # (samples, C/T, H, W) -> first channel/frame
        return np.asarray(np.nanmean(st[:, 0, :, :], axis=0), dtype=np.float64)
    if st.ndim == 5:
        # (samples, T, C, H, W) or (samples, T, H, W, C) layouts are reduced conservatively
        s0 = st[:, 0, ...]
        while s0.ndim > 3:
            s0 = s0[:, 0, ...]
        if s0.ndim == 3:
            return np.asarray(np.nanmean(s0, axis=0), dtype=np.float64)
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 1 STG signed tangent-map analysis")
    p.add_argument("--subject", type=str, required=True)
    p.add_argument("--date", type=str, required=True)
    p.add_argument("--dataset-configs-path", type=Path, default=Path("experiments") / "dataset_configs" / "multi_basic_240_rsvp.yaml")
    p.add_argument("--source", choices=("recorded", "twin"), required=True)
    p.add_argument("--image-set", type=str, default="high_support")
    p.add_argument("--sample-mode", choices=("all_available", "fixed_n"), default="fixed_n")
    p.add_argument("--n-samples-threshold", type=int, default=320)
    p.add_argument("--min-samples", type=int, default=160)
    p.add_argument("--ridge-alpha", type=float, default=1e-3)
    p.add_argument("--recorded-nuisance", choices=("none", "time", "time_global"), default="time_global")
    p.add_argument("--recorded-axis-projection", choices=("none", "global_rate", "pc1", "both"), default="none")
    p.add_argument("--recorded-shared-mode-projection-k", type=str, default="0")
    p.add_argument("--rank-tol-rel", type=float, default=1e-6)
    p.add_argument("--n-nulls", type=int, default=200)
    p.add_argument("--bootstrap-repeats", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--predict-batch-size", type=int, default=64)
    p.add_argument("--model-device", type=str, default="cuda")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--use-cached-data", action="store_true", default=True)
    p.add_argument("--drift-only", action="store_true", help="Restrict analysis to drift-only samples by excluding peri-saccadic windows")
    p.add_argument("--drift-eye-smooth-sigma-bins", type=float, default=1.0)
    p.add_argument("--drift-speed-percentile", type=float, default=99.5)
    p.add_argument("--drift-amp-thresh-deg", type=float, default=0.12)
    p.add_argument("--drift-refrac-ms", type=float, default=20.0)
    p.add_argument("--drift-exclusion-pre-ms", type=float, default=50.0)
    p.add_argument("--drift-exclusion-post-ms", type=float, default=150.0)
    return p


def _build_source_rates(args: argparse.Namespace, data: dict[str, Any]) -> tuple[np.ndarray, int | None]:
    if args.source == "recorded":
        return np.asarray(data["robs"], dtype=np.float64), None

    twin_rates, dataset_idx = _predict_twin_rates(
        data=data,
        subject=args.subject,
        date=args.date,
        dataset_configs_path=str(args.dataset_configs_path),
        model_type=None,
        model_index=None,
        checkpoint_path=None,
        dataset_idx=None,
        model_device=str(args.model_device),
        predict_batch_size=int(args.predict_batch_size),
    )
    return np.asarray(twin_rates, dtype=np.float64), int(dataset_idx)


def main() -> None:
    args = build_parser().parse_args()
    if args.source == "recorded" and str(args.recorded_shared_mode_projection_k).strip() not in {"", "0"}:
        import subprocess
        import sys

        cmd = [
            sys.executable,
            "-m",
            "declan.shared_transformation_geometry.run_stg_shared_mode_projection_sweep",
            "--subject",
            str(args.subject),
            "--date",
            str(args.date),
            "--dataset-configs-path",
            str(args.dataset_configs_path),
            "--sample-mode",
            str(args.sample_mode),
            "--n-samples-threshold",
            str(args.n_samples_threshold),
            "--min-samples",
            str(args.min_samples),
            "--recorded-nuisance",
            str(args.recorded_nuisance),
            "--recorded-axis-projection",
            str(args.recorded_axis_projection),
            "--recorded-shared-mode-projection-k",
            str(args.recorded_shared_mode_projection_k),
            "--ridge-alpha",
            str(args.ridge_alpha),
            "--rank-tol-rel",
            str(args.rank_tol_rel),
            "--n-nulls",
            str(args.n_nulls),
            "--bootstrap-repeats",
            str(args.bootstrap_repeats),
            "--seed",
            str(args.seed),
            "--out-dir",
            str(args.out_dir),
            "--drift-eye-smooth-sigma-bins",
            str(args.drift_eye_smooth_sigma_bins),
            "--drift-speed-percentile",
            str(args.drift_speed_percentile),
            "--drift-amp-thresh-deg",
            str(args.drift_amp_thresh_deg),
            "--drift-refrac-ms",
            str(args.drift_refrac_ms),
            "--drift-exclusion-pre-ms",
            str(args.drift_exclusion_pre_ms),
            "--drift-exclusion-post-ms",
            str(args.drift_exclusion_post_ms),
        ]
        if args.use_cached_data:
            cmd.append("--use-cached-data")
        if args.drift_only:
            cmd.append("--drift-only")
        subprocess.run(cmd, check=True)
        return
    rng = np.random.default_rng(int(args.seed))

    data = get_fixrsvp_data(
        subject=args.subject,
        date=args.date,
        dataset_configs_path=str(args.dataset_configs_path),
        use_cached_data=bool(args.use_cached_data),
    )
    data = harmonize_fixrsvp_arrays(data)

    rates, twin_dataset_idx = _build_source_rates(args, data)
    eyepos = np.asarray(data["eyepos"], dtype=np.float64)
    image_ids = np.asarray(data["image_ids"], dtype=np.int64)
    stim = np.asarray(data["stim"], dtype=np.float64)
    n_trials = int(image_ids.shape[0])
    n_time = int(image_ids.shape[1])
    time_grid = np.broadcast_to(np.arange(n_time, dtype=np.float64)[None, :], (n_trials, n_time))

    valid, drift_support = build_drift_only_valid_mask(
        rates=rates,
        eyepos=eyepos,
        image_ids=image_ids,
        data=data,
        drift_only=bool(args.drift_only),
        eye_smooth_sigma_bins=float(args.drift_eye_smooth_sigma_bins),
        speed_percentile=float(args.drift_speed_percentile),
        amp_thresh_deg=float(args.drift_amp_thresh_deg),
        refrac_ms=float(args.drift_refrac_ms),
        exclusion_pre_ms=float(args.drift_exclusion_pre_ms),
        exclusion_post_ms=float(args.drift_exclusion_post_ms),
    )
    img_ids = sorted(int(i) for i in np.unique(image_ids[valid]))

    per_image: dict[int, dict[str, Any]] = {}
    image_metric_rows: list[dict[str, object]] = []
    image_templates: dict[int, np.ndarray] = {}

    global_rate_axis = np.zeros(rates.shape[2], dtype=np.float64)
    global_pc1_axis = np.zeros(rates.shape[2], dtype=np.float64)
    global_mean_by_time = np.zeros(n_time, dtype=np.float64)
    global_pc1_by_time = np.zeros(n_time, dtype=np.float64)
    if args.source == "recorded":
        y_valid = rates[valid]
        if y_valid.ndim == 2 and y_valid.shape[0] >= 2:
            global_rate_axis = _safe_unit_axis(np.nanmean(y_valid, axis=0))
            yc = y_valid - np.nanmean(y_valid, axis=0, keepdims=True)
            _, _, vh = np.linalg.svd(yc, full_matrices=False)
            if vh.size:
                global_pc1_axis = _safe_unit_axis(vh[0])

        for t in range(n_time):
            mt = valid[:, t]
            if int(np.sum(mt)) > 0:
                global_mean_by_time[t] = float(np.nanmean(rates[:, t, :][mt]))
                rtm = rates[:, t, :][mt]
                if rtm.shape[0] >= 2:
                    rt_centered = rtm - np.mean(rtm, axis=0, keepdims=True)
                    _, _, vh_t = np.linalg.svd(rt_centered, full_matrices=False)
                    if vh_t.size:
                        score = rt_centered @ vh_t[0]
                        global_pc1_by_time[t] = float(np.mean(score))

    for img in img_ids:
        mask = valid & (image_ids == img)
        y_all = rates[mask]
        x_all = eyepos[mask]
        t_all = time_grid[mask]
        n_available = int(y_all.shape[0])

        if args.sample_mode == "fixed_n":
            n_used = int(args.n_samples_threshold)
            if n_available < n_used:
                continue
            img_rng = np.random.default_rng(int(args.seed) + (10007 * int(img)))
            pick = img_rng.permutation(n_available)[:n_used]
            y = y_all[pick]
            x = x_all[pick]
            tt = t_all[pick]
        else:
            if n_available < int(args.min_samples):
                continue
            y = y_all
            x = x_all
            tt = t_all
            n_used = int(y.shape[0])

        dxdy = x - np.mean(x, axis=0, keepdims=True)

        y_for_fit = np.asarray(y, dtype=np.float64)
        if args.source == "recorded" and args.recorded_axis_projection != "none":
            proj_axes: list[np.ndarray] = []
            if args.recorded_axis_projection in ("global_rate", "both"):
                proj_axes.append(global_rate_axis)
            if args.recorded_axis_projection in ("pc1", "both"):
                proj_axes.append(global_pc1_axis)
            y_for_fit = _project_responses_out_axes(y_for_fit, proj_axes)

        nuisance_cols: list[np.ndarray] = []
        if args.source == "recorded" and args.recorded_nuisance != "none":
            t_z = (tt - float(np.mean(tt))) / (float(np.std(tt)) + 1e-12)
            nuisance_cols.extend([t_z, t_z * t_z])
            if args.recorded_nuisance == "time_global":
                t_int = np.clip(np.rint(tt).astype(np.int64), 0, n_time - 1)
                nuisance_cols.append(global_mean_by_time[t_int])
                nuisance_cols.append(global_pc1_by_time[t_int])

        y_fit = _fit_unitwise_nuisance_residual(y_for_fit, nuisance_cols)
        bx, by, j, r2 = _fit_tangent_map(y_fit, dxdy, ridge_alpha=float(args.ridge_alpha))
        basis, rank_j, svals, frac1, frac2 = _j_diagnostics(j, rank_tol_rel=float(args.rank_tol_rel))
        cond = float(np.linalg.cond(j)) if np.all(np.isfinite(j)) else float("inf")
        cos_xy = _cos(bx, by)
        ang = float(np.degrees(np.arccos(np.clip(cos_xy, -1.0, 1.0)))) if np.isfinite(cos_xy) else float("nan")
        align_bx_global_rate = _cos(bx, global_rate_axis) if args.source == "recorded" else float("nan")
        align_bx_global_pc1 = _cos(bx, global_pc1_axis) if args.source == "recorded" else float("nan")

        per_image[img] = {
            "bx": bx,
            "by": by,
            "j": j,
            "basis": basis,
            "rank_j": int(rank_j),
            "singular_values_j": np.asarray(svals, dtype=np.float64),
            "frac_energy_colspace_top1": float(frac1),
            "frac_energy_colspace_top2": float(frac2),
            "n_samples_available": n_available,
            "n_samples_used": int(n_used),
            "n_units": int(y.shape[1]),
            "dxdy": dxdy,
            "y": y_fit,
        }
        image_metric_rows.append(
            {
                "session_id": f"{args.subject}_{args.date}",
                "subject": args.subject,
                "date": args.date,
                "source": args.source,
                "image_set": args.image_set,
                "analysis_representation": "raw_samples",
                "sample_mode": args.sample_mode,
                "image_id": int(img),
                "n_samples_available": int(n_available),
                "n_samples_used": int(n_used),
                "n_units": int(y.shape[1]),
                "r2_mean": float(np.mean(r2)),
                "r2_median": float(np.median(r2)),
                "norm_bx": float(np.linalg.norm(bx)),
                "norm_by": float(np.linalg.norm(by)),
                "angle_between_bx_by": ang,
                "condition_number_J": cond,
                "rank_J": int(rank_j),
                "align_bx_global_rate_axis": float(align_bx_global_rate),
                "align_bx_global_pc1_axis": float(align_bx_global_pc1),
                "singular_values_J": json.dumps([float(v) for v in svals.tolist()]),
                "frac_energy_colspace_top1": float(frac1),
                "frac_energy_colspace_top2": float(frac2),
            }
        )

        st = stim[mask]
        tpl = _extract_template_2d(st)
        if tpl is not None and tpl.size > 0:
            image_templates[img] = tpl

    usable_ids = sorted(per_image.keys())
    out_dir = Path(args.out_dir) / f"{args.subject}_{args.date}" / f"source_{args.source}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "stg_tangent_maps.pkl").open("wb") as handle:
        pickle.dump(
            {
                "session_id": f"{args.subject}_{args.date}",
                "subject": args.subject,
                "date": args.date,
                "source": args.source,
                "image_set": args.image_set,
                "analysis_representation": "raw_samples",
                "sample_mode": args.sample_mode,
                "n_samples_threshold": int(args.n_samples_threshold),
                "dataset_idx": twin_dataset_idx,
                "images": per_image,
            },
            handle,
        )
    _write_csv(out_dir / "stg_tangent_map_image_metrics.csv", image_metric_rows)

    alignment_rows: list[dict[str, object]] = []
    pair_rows_for_boot: list[dict[str, object]] = []

    for img_i, img_j in itertools.combinations(usable_ids, 2):
        a = per_image[img_i]
        b = per_image[img_j]

        cos_bx = _cos(a["bx"], b["bx"])
        cos_by = _cos(a["by"], b["by"])
        mean_signed = float(np.nanmean([cos_bx, cos_by]))
        subspace = _subspace_overlap(a["basis"], b["basis"])

        sim = {
            "pixel_correlation": float("nan"),
            "rms_contrast_difference": float("nan"),
            "fourier_amplitude_similarity": float("nan"),
        }
        if img_i in image_templates and img_j in image_templates:
            sim = _compute_image_similarity(image_templates[img_i], image_templates[img_j])

        eye_null_signed: list[float] = []
        eye_null_subspace: list[float] = []
        rand_null_signed: list[float] = []
        rand_null_subspace: list[float] = []
        for _ in range(int(args.n_nulls)):
            perm_i = rng.permutation(a["dxdy"].shape[0])
            perm_j = rng.permutation(b["dxdy"].shape[0])
            bx_i_s, by_i_s, j_i_s, _ = _fit_tangent_map(a["y"], a["dxdy"][perm_i], ridge_alpha=float(args.ridge_alpha))
            bx_j_s, by_j_s, j_j_s, _ = _fit_tangent_map(b["y"], b["dxdy"][perm_j], ridge_alpha=float(args.ridge_alpha))
            basis_i_s, _, _, _, _ = _j_diagnostics(j_i_s, rank_tol_rel=float(args.rank_tol_rel))
            basis_j_s, _, _, _, _ = _j_diagnostics(j_j_s, rank_tol_rel=float(args.rank_tol_rel))
            eye_null_signed.append(float(np.nanmean([_cos(bx_i_s, bx_j_s), _cos(by_i_s, by_j_s)])))
            eye_null_subspace.append(_subspace_overlap(basis_i_s, basis_j_s))

            rbx_i, rby_i, rj_i = _random_map_with_norms(
                a["n_units"], float(np.linalg.norm(a["bx"])), float(np.linalg.norm(a["by"])), rng
            )
            rbx_j, rby_j, rj_j = _random_map_with_norms(
                b["n_units"], float(np.linalg.norm(b["bx"])), float(np.linalg.norm(b["by"])), rng
            )
            rbasis_i, _, _, _, _ = _j_diagnostics(rj_i, rank_tol_rel=float(args.rank_tol_rel))
            rbasis_j, _, _, _, _ = _j_diagnostics(rj_j, rank_tol_rel=float(args.rank_tol_rel))
            rand_null_signed.append(float(np.nanmean([_cos(rbx_i, rbx_j), _cos(rby_i, rby_j)])))
            rand_null_subspace.append(_subspace_overlap(rbasis_i, rbasis_j))

        eye_null_signed_arr = np.asarray(eye_null_signed, dtype=np.float64)
        rand_null_signed_arr = np.asarray(rand_null_signed, dtype=np.float64)
        eye_null_subspace_arr = np.asarray(eye_null_subspace, dtype=np.float64)
        rand_null_subspace_arr = np.asarray(rand_null_subspace, dtype=np.float64)

        eye_ci_low, eye_ci_high = _bootstrap_ci(eye_null_signed_arr)
        rand_ci_low, rand_ci_high = _bootstrap_ci(rand_null_signed_arr)
        eye_sub_ci_low, eye_sub_ci_high = _bootstrap_ci(eye_null_subspace_arr)
        rand_sub_ci_low, rand_sub_ci_high = _bootstrap_ci(rand_null_subspace_arr)

        row = {
            "session_id": f"{args.subject}_{args.date}",
            "subject": args.subject,
            "date": args.date,
            "source": args.source,
            "image_set": args.image_set,
            "analysis_representation": "raw_samples",
            "sample_mode": args.sample_mode,
            "image_i": int(img_i),
            "image_j": int(img_j),
            "n_samples_available_i": int(a["n_samples_available"]),
            "n_samples_available_j": int(b["n_samples_available"]),
            "n_samples_used_i": int(a["n_samples_used"]),
            "n_samples_used_j": int(b["n_samples_used"]),
            "n_units": int(a["n_units"]),
            "rank_J_i": int(a["rank_j"]),
            "rank_J_j": int(b["rank_j"]),
            "cos_bx": cos_bx,
            "cos_by": cos_by,
            "mean_signed_column_alignment": mean_signed,
            "subspace_overlap_k2": subspace,
            "null_eyeshuffle_mean": float(np.mean(eye_null_signed_arr)),
            "null_eyeshuffle_ci_low": eye_ci_low,
            "null_eyeshuffle_ci_high": eye_ci_high,
            "null_random_mean": float(np.mean(rand_null_signed_arr)),
            "null_random_ci_low": rand_ci_low,
            "null_random_ci_high": rand_ci_high,
            "null_eyeshuffle_subspace_mean": float(np.mean(eye_null_subspace_arr)),
            "null_eyeshuffle_subspace_ci_low": eye_sub_ci_low,
            "null_eyeshuffle_subspace_ci_high": eye_sub_ci_high,
            "null_random_subspace_mean": float(np.mean(rand_null_subspace_arr)),
            "null_random_subspace_ci_low": rand_sub_ci_low,
            "null_random_subspace_ci_high": rand_sub_ci_high,
            "exceeds_eyeshuffle_null": bool(mean_signed > float(np.mean(eye_null_signed_arr))),
            "exceeds_random_null": bool(mean_signed > float(np.mean(rand_null_signed_arr))),
            "exceeds_eyeshuffle_subspace_null": bool(subspace > float(np.mean(eye_null_subspace_arr))),
            "exceeds_random_subspace_null": bool(subspace > float(np.mean(rand_null_subspace_arr))),
            "pixel_correlation": float(sim["pixel_correlation"]),
            "rms_contrast_difference": float(sim["rms_contrast_difference"]),
            "fourier_amplitude_similarity": float(sim["fourier_amplitude_similarity"]),
        }
        alignment_rows.append(row)
        pair_rows_for_boot.append(
            {
                "image_i": int(img_i),
                "image_j": int(img_j),
                "signed": float(mean_signed),
                "subspace": float(subspace),
                "diff_signed_eye": float(mean_signed - float(np.mean(eye_null_signed_arr))),
                "diff_signed_random": float(mean_signed - float(np.mean(rand_null_signed_arr))),
                "diff_subspace_eye": float(subspace - float(np.mean(eye_null_subspace_arr))),
                "diff_subspace_random": float(subspace - float(np.mean(rand_null_subspace_arr))),
                "eye_null_signed": float(np.mean(eye_null_signed_arr)),
                "rand_null_signed": float(np.mean(rand_null_signed_arr)),
                "eye_null_subspace": float(np.mean(eye_null_subspace_arr)),
                "rand_null_subspace": float(np.mean(rand_null_subspace_arr)),
                "pixel_correlation": float(sim["pixel_correlation"]),
                "rms_contrast_difference": float(sim["rms_contrast_difference"]),
                "fourier_amplitude_similarity": float(sim["fourier_amplitude_similarity"]),
            }
        )

    _write_csv(out_dir / "stg_tangent_map_alignment.csv", alignment_rows)

    signed_vals = np.asarray([float(r["signed"]) for r in pair_rows_for_boot], dtype=np.float64)
    subspace_vals = np.asarray([float(r["subspace"]) for r in pair_rows_for_boot], dtype=np.float64)
    eye_null_signed = np.asarray([float(r["eye_null_signed"]) for r in pair_rows_for_boot], dtype=np.float64)
    rand_null_signed = np.asarray([float(r["rand_null_signed"]) for r in pair_rows_for_boot], dtype=np.float64)

    diff_signed_eye = np.asarray([float(r["diff_signed_eye"]) for r in pair_rows_for_boot], dtype=np.float64)
    diff_signed_rand = np.asarray([float(r["diff_signed_random"]) for r in pair_rows_for_boot], dtype=np.float64)
    diff_sub_eye = np.asarray([float(r["diff_subspace_eye"]) for r in pair_rows_for_boot], dtype=np.float64)
    diff_sub_rand = np.asarray([float(r["diff_subspace_random"]) for r in pair_rows_for_boot], dtype=np.float64)

    boot_diff_signed_eye = _image_bootstrap_weighted_mean(
        pair_rows_for_boot,
        "diff_signed_eye",
        usable_ids,
        seed=int(args.seed) + 1,
        n_bootstrap=int(args.bootstrap_repeats),
    )
    boot_diff_signed_rand = _image_bootstrap_weighted_mean(
        pair_rows_for_boot,
        "diff_signed_random",
        usable_ids,
        seed=int(args.seed) + 2,
        n_bootstrap=int(args.bootstrap_repeats),
    )
    boot_diff_sub_eye = _image_bootstrap_weighted_mean(
        pair_rows_for_boot,
        "diff_subspace_eye",
        usable_ids,
        seed=int(args.seed) + 3,
        n_bootstrap=int(args.bootstrap_repeats),
    )
    boot_diff_sub_rand = _image_bootstrap_weighted_mean(
        pair_rows_for_boot,
        "diff_subspace_random",
        usable_ids,
        seed=int(args.seed) + 4,
        n_bootstrap=int(args.bootstrap_repeats),
    )

    ci_signed_eye = _bootstrap_ci(boot_diff_signed_eye)
    ci_signed_rand = _bootstrap_ci(boot_diff_signed_rand)
    ci_sub_eye = _bootstrap_ci(boot_diff_sub_eye)
    ci_sub_rand = _bootstrap_ci(boot_diff_sub_rand)

    p_signed_eye = float(np.mean(boot_diff_signed_eye <= 0.0)) if boot_diff_signed_eye.size else float("nan")
    p_signed_rand = float(np.mean(boot_diff_signed_rand <= 0.0)) if boot_diff_signed_rand.size else float("nan")
    p_sub_eye = float(np.mean(boot_diff_sub_eye <= 0.0)) if boot_diff_sub_eye.size else float("nan")
    p_sub_rand = float(np.mean(boot_diff_sub_rand <= 0.0)) if boot_diff_sub_rand.size else float("nan")

    controlled: dict[str, float] = {}
    for idx, control_key in enumerate(CONTROL_METRICS, start=1):
        int_eye, slope_eye, n_eye = _fit_adjusted_intercept(pair_rows_for_boot, "diff_signed_eye", control_key)
        int_rand, slope_rand, n_rand = _fit_adjusted_intercept(pair_rows_for_boot, "diff_signed_random", control_key)
        boot_int_eye = _image_bootstrap_adjusted_intercept(
            pair_rows_for_boot,
            "diff_signed_eye",
            control_key,
            usable_ids,
            seed=int(args.seed) + 200 + idx,
            n_bootstrap=int(args.bootstrap_repeats),
        )
        boot_int_rand = _image_bootstrap_adjusted_intercept(
            pair_rows_for_boot,
            "diff_signed_random",
            control_key,
            usable_ids,
            seed=int(args.seed) + 300 + idx,
            n_bootstrap=int(args.bootstrap_repeats),
        )
        ci_eye = _bootstrap_ci(boot_int_eye)
        ci_rand = _bootstrap_ci(boot_int_rand)
        p_eye = float(np.mean(boot_int_eye <= 0.0)) if boot_int_eye.size else float("nan")
        p_rand = float(np.mean(boot_int_rand <= 0.0)) if boot_int_rand.size else float("nan")

        controlled[f"controlled_effect_minus_eye_shuffle_{control_key}"] = float(int_eye)
        controlled[f"controlled_alpha1_minus_eye_shuffle_{control_key}"] = float(slope_eye)
        controlled[f"controlled_n_pairs_minus_eye_shuffle_{control_key}"] = int(n_eye)
        controlled[f"controlled_ci_low_minus_eye_shuffle_{control_key}"] = float(ci_eye[0])
        controlled[f"controlled_ci_high_minus_eye_shuffle_{control_key}"] = float(ci_eye[1])
        controlled[f"controlled_p_minus_eye_shuffle_{control_key}_le_0"] = float(p_eye)

        controlled[f"controlled_effect_minus_random_map_{control_key}"] = float(int_rand)
        controlled[f"controlled_alpha1_minus_random_map_{control_key}"] = float(slope_rand)
        controlled[f"controlled_n_pairs_minus_random_map_{control_key}"] = int(n_rand)
        controlled[f"controlled_ci_low_minus_random_map_{control_key}"] = float(ci_rand[0])
        controlled[f"controlled_ci_high_minus_random_map_{control_key}"] = float(ci_rand[1])
        controlled[f"controlled_p_minus_random_map_{control_key}_le_0"] = float(p_rand)

        low_eye_rows, low_thr, low_n_eye = _fit_low_similarity(pair_rows_for_boot, "diff_signed_eye", control_key)
        low_rand_rows, _, low_n_rand = _fit_low_similarity(pair_rows_for_boot, "diff_signed_random", control_key)
        low_eye_boot = _image_bootstrap_weighted_mean(
            low_eye_rows,
            "diff_signed_eye",
            usable_ids,
            seed=int(args.seed) + 400 + idx,
            n_bootstrap=int(args.bootstrap_repeats),
        )
        low_rand_boot = _image_bootstrap_weighted_mean(
            low_rand_rows,
            "diff_signed_random",
            usable_ids,
            seed=int(args.seed) + 500 + idx,
            n_bootstrap=int(args.bootstrap_repeats),
        )
        low_eye_ci = _bootstrap_ci(low_eye_boot)
        low_rand_ci = _bootstrap_ci(low_rand_boot)

        low_eye_arr = np.asarray([float(r["diff_signed_eye"]) for r in low_eye_rows], dtype=np.float64)
        low_rand_arr = np.asarray([float(r["diff_signed_random"]) for r in low_rand_rows], dtype=np.float64)

        controlled[f"low_similarity_threshold_{control_key}"] = float(low_thr)
        controlled[f"low_similarity_n_pairs_minus_eye_shuffle_{control_key}"] = int(low_n_eye)
        controlled[f"low_similarity_effect_minus_eye_shuffle_{control_key}"] = float(np.nanmean(low_eye_arr)) if low_eye_arr.size else float("nan")
        controlled[f"low_similarity_ci_low_minus_eye_shuffle_{control_key}"] = float(low_eye_ci[0])
        controlled[f"low_similarity_ci_high_minus_eye_shuffle_{control_key}"] = float(low_eye_ci[1])
        controlled[f"low_similarity_p_minus_eye_shuffle_{control_key}_le_0"] = (
            float(np.mean(low_eye_boot <= 0.0)) if low_eye_boot.size else float("nan")
        )

        controlled[f"low_similarity_n_pairs_minus_random_map_{control_key}"] = int(low_n_rand)
        controlled[f"low_similarity_effect_minus_random_map_{control_key}"] = float(np.nanmean(low_rand_arr)) if low_rand_arr.size else float("nan")
        controlled[f"low_similarity_ci_low_minus_random_map_{control_key}"] = float(low_rand_ci[0])
        controlled[f"low_similarity_ci_high_minus_random_map_{control_key}"] = float(low_rand_ci[1])
        controlled[f"low_similarity_p_minus_random_map_{control_key}_le_0"] = (
            float(np.mean(low_rand_boot <= 0.0)) if low_rand_boot.size else float("nan")
        )

    primary_control = "pixel_correlation"
    low_ci_eye_primary = float(controlled[f"low_similarity_ci_low_minus_eye_shuffle_{primary_control}"])
    low_ci_rand_primary = float(controlled[f"low_similarity_ci_low_minus_random_map_{primary_control}"])
    low_n_eye_primary = int(controlled[f"low_similarity_n_pairs_minus_eye_shuffle_{primary_control}"])
    low_n_rand_primary = int(controlled[f"low_similarity_n_pairs_minus_random_map_{primary_control}"])

    control_is_evaluable = (
        np.isfinite(low_ci_eye_primary)
        and np.isfinite(low_ci_rand_primary)
        and (low_n_eye_primary > 0)
        and (low_n_rand_primary > 0)
    )

    if control_is_evaluable:
        controlled_label = (
            "tangent_shared_geometry"
            if (low_ci_eye_primary > 0.0) and (low_ci_rand_primary > 0.0)
            else "not_supported"
        )
    else:
        controlled_label = "control_not_evaluable"

    summary_rows = [
        {
            "session_id": f"{args.subject}_{args.date}",
            "subject": args.subject,
            "date": args.date,
            "source": args.source,
            "image_set": args.image_set,
            "analysis_representation": "raw_samples",
            "sample_mode": args.sample_mode,
            "drift_only": bool(args.drift_only),
            "bootstrap_unit": "image",
            "controlled_label_basis": "low_similarity_pairs_pixel_correlation",
            "controlled_primary_metric": primary_control,
            "control_is_evaluable": bool(control_is_evaluable),
            "n_pairs_control_evaluable": int(min(low_n_eye_primary, low_n_rand_primary) if control_is_evaluable else 0),
            "n_images": int(len(usable_ids)),
            "n_pairs": int(len(pair_rows_for_boot)),
            "n_images_with_samples_before_exclusion": int(drift_support["n_images_with_samples_before_exclusion"]),
            "n_images_with_samples_after_exclusion": int(drift_support["n_images_with_samples_after_exclusion"]),
            "n_valid_samples_before_exclusion": int(drift_support["n_valid_samples_before_exclusion"]),
            "n_valid_samples_excluded": int(drift_support["n_valid_samples_excluded"]),
            "n_valid_samples_after_exclusion": int(drift_support["n_valid_samples_after_exclusion"]),
            "fraction_valid_samples_after_exclusion": float(drift_support["fraction_valid_samples_after_exclusion"]),
            "drift_n_events_detected": int(drift_support["drift_n_events_detected"]),
            "drift_speed_threshold_deg_s": float(drift_support["drift_speed_threshold_deg_s"]),
            "n_samples_threshold": int(args.n_samples_threshold),
            "recorded_nuisance": str(args.recorded_nuisance),
            "recorded_axis_projection": str(args.recorded_axis_projection),
            "n_units": int(next(iter(per_image.values()))["n_units"]) if per_image else 0,
            "mean_n_samples_available": float(np.mean([int(v["n_samples_available"]) for v in per_image.values()])) if per_image else float("nan"),
            "mean_n_samples_used": float(np.mean([int(v["n_samples_used"]) for v in per_image.values()])) if per_image else float("nan"),
            "mean_r2_per_image": float(np.nanmean([float(r["r2_mean"]) for r in image_metric_rows])) if image_metric_rows else float("nan"),
            "median_r2_per_image": float(np.nanmedian([float(r["r2_median"]) for r in image_metric_rows])) if image_metric_rows else float("nan"),
            "mean_norm_bx": float(np.nanmean([float(r["norm_bx"]) for r in image_metric_rows])) if image_metric_rows else float("nan"),
            "mean_norm_by": float(np.nanmean([float(r["norm_by"]) for r in image_metric_rows])) if image_metric_rows else float("nan"),
            "n_low_norm_images": int(np.sum([float(r["norm_bx"]) < 1e-6 or float(r["norm_by"]) < 1e-6 for r in image_metric_rows])) if image_metric_rows else 0,
            "mean_align_bx_global_rate_axis": (
                float(np.nanmean([float(r["align_bx_global_rate_axis"]) for r in image_metric_rows]))
                if (image_metric_rows and args.source == "recorded")
                else float("nan")
            ),
            "mean_align_bx_global_pc1_axis": (
                float(np.nanmean([float(r["align_bx_global_pc1_axis"]) for r in image_metric_rows]))
                if (image_metric_rows and args.source == "recorded")
                else float("nan")
            ),
            "mean_cos_bx": float(np.nanmean([float(r["cos_bx"]) for r in alignment_rows])) if alignment_rows else float("nan"),
            "mean_cos_by": float(np.nanmean([float(r["cos_by"]) for r in alignment_rows])) if alignment_rows else float("nan"),
            "mean_signed_column_alignment": float(np.nanmean(signed_vals)) if signed_vals.size else float("nan"),
            "mean_subspace_overlap_k2": float(np.nanmean(subspace_vals)) if subspace_vals.size else float("nan"),
            "mean_eye_shuffle_null": float(np.nanmean(eye_null_signed)) if eye_null_signed.size else float("nan"),
            "mean_random_map_null": float(np.nanmean(rand_null_signed)) if rand_null_signed.size else float("nan"),
            "effect_minus_eye_shuffle": float(np.nanmean(diff_signed_eye)) if diff_signed_eye.size else float("nan"),
            "effect_minus_random_map": float(np.nanmean(diff_signed_rand)) if diff_signed_rand.size else float("nan"),
            "bootstrap_ci_low_minus_eye_shuffle": ci_signed_eye[0],
            "bootstrap_ci_high_minus_eye_shuffle": ci_signed_eye[1],
            "bootstrap_p_minus_eye_shuffle_le_0": p_signed_eye,
            "bootstrap_ci_low_minus_random": ci_signed_rand[0],
            "bootstrap_ci_high_minus_random": ci_signed_rand[1],
            "bootstrap_p_minus_random_le_0": p_signed_rand,
            "mean_eye_shuffle_null_subspace": float(np.nanmean([float(r["eye_null_subspace"]) for r in pair_rows_for_boot])) if pair_rows_for_boot else float("nan"),
            "mean_random_map_null_subspace": float(np.nanmean([float(r["rand_null_subspace"]) for r in pair_rows_for_boot])) if pair_rows_for_boot else float("nan"),
            "effect_minus_eye_shuffle_subspace": float(np.nanmean(diff_sub_eye)) if diff_sub_eye.size else float("nan"),
            "effect_minus_random_map_subspace": float(np.nanmean(diff_sub_rand)) if diff_sub_rand.size else float("nan"),
            "bootstrap_ci_low_minus_eye_shuffle_subspace": ci_sub_eye[0],
            "bootstrap_ci_high_minus_eye_shuffle_subspace": ci_sub_eye[1],
            "bootstrap_p_minus_eye_shuffle_subspace_le_0": p_sub_eye,
            "bootstrap_ci_low_minus_random_subspace": ci_sub_rand[0],
            "bootstrap_ci_high_minus_random_subspace": ci_sub_rand[1],
            "bootstrap_p_minus_random_subspace_le_0": p_sub_rand,
            "tangent_sanity_label": (
                "non_degenerate"
                if image_metric_rows
                and (float(np.nanmean([float(r["r2_mean"]) for r in image_metric_rows])) > 0.0)
                and (int(np.sum([float(r["norm_bx"]) < 1e-6 or float(r["norm_by"]) < 1e-6 for r in image_metric_rows])) == 0)
                else "potentially_degenerate"
            ),
            "interpretation_label_uncontrolled": (
                "tangent_shared_geometry"
                if np.isfinite(ci_signed_eye[0])
                and np.isfinite(ci_signed_rand[0])
                and ci_signed_eye[0] > 0.0
                and ci_signed_rand[0] > 0.0
                else "not_supported"
            ),
            "interpretation_label": (
                controlled_label
            ),
            **controlled,
        }
    ]
    _write_csv(out_dir / "stg_tangent_summary.csv", summary_rows)

    metadata = {
        "session_id": f"{args.subject}_{args.date}",
        "source": args.source,
        "analysis_representation": "raw_samples",
        "sample_mode": args.sample_mode,
        "drift_only": bool(args.drift_only),
        "drift_parameters": {
            "eye_smooth_sigma_bins": float(args.drift_eye_smooth_sigma_bins),
            "speed_percentile": float(args.drift_speed_percentile),
            "amp_thresh_deg": float(args.drift_amp_thresh_deg),
            "refrac_ms": float(args.drift_refrac_ms),
            "exclusion_pre_ms": float(args.drift_exclusion_pre_ms),
            "exclusion_post_ms": float(args.drift_exclusion_post_ms),
        },
        "drift_support": drift_support,
        "n_samples_threshold": int(args.n_samples_threshold),
        "min_samples": int(args.min_samples),
        "n_nulls": int(args.n_nulls),
        "bootstrap_repeats": int(args.bootstrap_repeats),
        "bootstrap_unit": "image",
        "recorded_nuisance": str(args.recorded_nuisance),
        "recorded_axis_projection": str(args.recorded_axis_projection),
        "image_similarity_controls": list(CONTROL_METRICS),
        "controlled_label_basis": "low_similarity_pairs_pixel_correlation",
        "controlled_primary_metric": primary_control,
        "control_not_evaluable_label": "control_not_evaluable",
    }
    (out_dir / "stg_tangent_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(str(out_dir / "stg_tangent_maps.pkl"))
    print(str(out_dir / "stg_tangent_map_image_metrics.csv"))
    print(str(out_dir / "stg_tangent_map_alignment.csv"))
    print(str(out_dir / "stg_tangent_summary.csv"))
    print(str(out_dir / "stg_tangent_metadata.json"))


if __name__ == "__main__":
    main()
