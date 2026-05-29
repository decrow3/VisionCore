#!/usr/bin/env python3
"""Empirical split-half geometry ceiling for fixRSVP image windows.

This script measures whether the empirical eye-sensitivity geometry is
recoverable above explicit eye-response decoupling and random-subspace nulls.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from eval.fixrsvp import get_fixrsvp_data
from VisionCore.paths import FIGURES_DIR
from scripts.jacobian_predictive_framework.run_fixrsvp_step2 import (
    REFERENCE_RATE_HZ,
    _choose_baseline,
    _collect_image_windows,
    _resolve_pixels_per_degree,
    _split_half_fem_drive,
)
from scripts.jacobian_predictive_framework.run_fixrsvp_steps01 import _paired_delta_summary


DEFAULT_OUTPUT_DIR = Path("outputs/jacobian_predictive_framework/empirical_geometry_ceiling")


def _orth_basis(matrix: np.ndarray, n_dim: int) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64)
    safe = np.nan_to_num(np.asarray(matrix, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        u, _s, _vh = np.linalg.svd(safe, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    if u.size == 0:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    return u[:, : min(n_dim, u.shape[1])]


def _subspace_alignment(B_a: np.ndarray, B_b: np.ndarray, n_dim: int = 2) -> float:
    U1 = _orth_basis(B_a, n_dim=n_dim)
    U2 = _orth_basis(B_b, n_dim=n_dim)
    if U1.size == 0 or U2.size == 0:
        return float("nan")
    singular_values = np.linalg.svd(U1.T @ U2, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return float(np.mean(singular_values ** 2))


def _principal_angles_deg(B_a: np.ndarray, B_b: np.ndarray, n_dim: int = 2) -> np.ndarray:
    U1 = _orth_basis(B_a, n_dim=n_dim)
    U2 = _orth_basis(B_b, n_dim=n_dim)
    if U1.size == 0 or U2.size == 0:
        return np.empty(0, dtype=np.float64)
    singular_values = np.linalg.svd(U1.T @ U2, compute_uv=False)
    singular_values = np.clip(singular_values, -1.0, 1.0)
    return np.degrees(np.arccos(singular_values))


def _capture_fraction(B_source: np.ndarray, B_target: np.ndarray, n_dim: int = 2) -> float:
    U = _orth_basis(B_source, n_dim=n_dim)
    if U.size == 0:
        return float("nan")
    cov_target = B_target @ B_target.T
    denom = np.trace(cov_target) + 1e-12
    return float(np.trace(U.T @ cov_target @ U) / denom)


def _vec_corr(B_a: np.ndarray, B_b: np.ndarray) -> float:
    va = np.asarray(B_a, dtype=np.float64).ravel()
    vb = np.asarray(B_b, dtype=np.float64).ravel()
    if np.std(va) <= 1e-12 or np.std(vb) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(va, vb)[0, 1])


def _matrix_stats(B: np.ndarray) -> dict:
    safe = np.nan_to_num(np.asarray(B, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        sv = np.linalg.svd(safe, compute_uv=False)
    except np.linalg.LinAlgError:
        sv = np.array([float("nan"), float("nan")], dtype=np.float64)
    s1 = float(sv[0]) if sv.size >= 1 else float("nan")
    s2 = float(sv[1]) if sv.size >= 2 else float("nan")
    rank_ratio = float(s2 / s1) if np.isfinite(s1) and s1 > 1e-12 and np.isfinite(s2) else float("nan")
    return {
        "fro_norm": float(np.linalg.norm(safe, ord="fro")),
        "singular_1": s1,
        "singular_2": s2,
        "rank_ratio": rank_ratio,
    }


def _bundle_alignment_metrics(B_a: np.ndarray, B_b: np.ndarray) -> dict:
    if B_a.ndim != 3 or B_b.ndim != 3 or B_a.shape != B_b.shape or B_a.shape[0] == 0:
        return {
            "alignment2_values": np.empty(0, dtype=np.float64),
            "alignment1_values": np.empty(0, dtype=np.float64),
            "capture2_values": np.empty(0, dtype=np.float64),
            "capture1_values": np.empty(0, dtype=np.float64),
            "angle1_values": np.empty(0, dtype=np.float64),
            "angle2_values": np.empty(0, dtype=np.float64),
            "vec_corr_values": np.empty(0, dtype=np.float64),
            "B_a_norm_values": np.empty(0, dtype=np.float64),
            "B_b_norm_values": np.empty(0, dtype=np.float64),
            "B_a_rank_ratio_values": np.empty(0, dtype=np.float64),
            "B_b_rank_ratio_values": np.empty(0, dtype=np.float64),
        }

    align2 = np.empty(B_a.shape[0], dtype=np.float64)
    align1 = np.empty(B_a.shape[0], dtype=np.float64)
    capture2 = np.empty(B_a.shape[0], dtype=np.float64)
    capture1 = np.empty(B_a.shape[0], dtype=np.float64)
    angle1 = np.empty(B_a.shape[0], dtype=np.float64)
    angle2 = np.empty(B_a.shape[0], dtype=np.float64)
    vec_corr = np.empty(B_a.shape[0], dtype=np.float64)
    B_a_norm = np.empty(B_a.shape[0], dtype=np.float64)
    B_b_norm = np.empty(B_a.shape[0], dtype=np.float64)
    B_a_rank_ratio = np.empty(B_a.shape[0], dtype=np.float64)
    B_b_rank_ratio = np.empty(B_a.shape[0], dtype=np.float64)
    for idx in range(B_a.shape[0]):
        angles = _principal_angles_deg(B_a[idx], B_b[idx], n_dim=2)
        stats_a = _matrix_stats(B_a[idx])
        stats_b = _matrix_stats(B_b[idx])
        align2[idx] = _subspace_alignment(B_a[idx], B_b[idx], n_dim=2)
        align1[idx] = _subspace_alignment(B_a[idx], B_b[idx], n_dim=1)
        capture2[idx] = _capture_fraction(B_a[idx], B_b[idx], n_dim=2)
        capture1[idx] = _capture_fraction(B_a[idx], B_b[idx], n_dim=1)
        angle1[idx] = float(angles[0]) if angles.size >= 1 else float("nan")
        angle2[idx] = float(angles[1]) if angles.size >= 2 else float("nan")
        vec_corr[idx] = _vec_corr(B_a[idx], B_b[idx])
        B_a_norm[idx] = stats_a["fro_norm"]
        B_b_norm[idx] = stats_b["fro_norm"]
        B_a_rank_ratio[idx] = stats_a["rank_ratio"]
        B_b_rank_ratio[idx] = stats_b["rank_ratio"]
    return {
        "alignment2_values": align2,
        "alignment1_values": align1,
        "capture2_values": capture2,
        "capture1_values": capture1,
        "angle1_values": angle1,
        "angle2_values": angle2,
        "vec_corr_values": vec_corr,
        "B_a_norm_values": B_a_norm,
        "B_b_norm_values": B_b_norm,
        "B_a_rank_ratio_values": B_a_rank_ratio,
        "B_b_rank_ratio_values": B_b_rank_ratio,
    }


def _concat_metric_dicts(metric_runs: list[dict]) -> dict:
    if not metric_runs:
        return _bundle_alignment_metrics(np.empty((0, 0, 0), dtype=np.float64), np.empty((0, 0, 0), dtype=np.float64))
    keys = metric_runs[0].keys()
    return {
        key: np.concatenate([np.asarray(run[key], dtype=np.float64) for run in metric_runs])
        for key in keys
    }


def _random_alignment_null(
    B_b: np.ndarray,
    n_random_subspaces: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if B_b.ndim != 3 or B_b.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    n_neurons = B_b.shape[1]
    random_alignments = np.empty(n_random_subspaces, dtype=np.float64)
    medians = np.empty(B_b.shape[0], dtype=np.float64)
    for rep in range(B_b.shape[0]):
        target_basis = _orth_basis(B_b[rep], n_dim=2)
        if target_basis.size == 0:
            medians[rep] = float("nan")
            continue
        for ridx in range(n_random_subspaces):
            U_rand, _ = np.linalg.qr(rng.standard_normal((n_neurons, 2)))
            singular_values = np.linalg.svd(U_rand.T @ target_basis, compute_uv=False)
            random_alignments[ridx] = float(np.mean(np.clip(singular_values, 0.0, 1.0) ** 2))
        medians[rep] = float(np.nanmedian(random_alignments))
    return medians


def _permute_eye_positions(eyepos_px: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return eyepos_px[rng.permutation(len(eyepos_px))]


def _circular_shift_eye_positions_within_trials(
    eyepos_px: np.ndarray,
    trial_indices: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    shifted = eyepos_px.copy()
    for trial_id in np.unique(trial_indices):
        idx = np.where(trial_indices == trial_id)[0]
        if idx.size <= 1:
            continue
        shift = int(rng.integers(1, idx.size))
        shifted[idx] = eyepos_px[idx[np.roll(np.arange(idx.size), shift)]]
    return shifted


def _subset_summary(rows: list[dict], subset_name: str, mask: np.ndarray) -> dict:
    selected = [row for row, keep in zip(rows, mask) if keep]
    if not selected:
        return {
            "subset": subset_name,
            "n": 0,
            "median_alignment2": float("nan"),
            "median_alignment1": float("nan"),
            "median_alignment2_minus_eyeperm": float("nan"),
            "median_alignment1_minus_eyeperm": float("nan"),
        }

    paired_2d = [{"matched": row["emp_split_alignment_2d"], "shuffled": row["eye_perm_alignment_2d"]} for row in selected]
    paired_1d = [{"matched": row["emp_split_alignment_top1"], "shuffled": row["eye_perm_alignment_top1"]} for row in selected]
    return {
        "subset": subset_name,
        "n": len(selected),
        "median_alignment2": float(np.nanmedian([row["emp_split_alignment_2d"] for row in selected])),
        "median_alignment1": float(np.nanmedian([row["emp_split_alignment_top1"] for row in selected])),
        "median_vec_corr": float(np.nanmedian([row["vec_corr_BA_BB"] for row in selected])),
        "median_rank_ratio": float(np.nanmedian([row["min_B_rank_ratio"] for row in selected])),
        "paired_alignment2_vs_eyeperm": _paired_delta_summary(paired_2d, "matched", "shuffled"),
        "paired_alignment1_vs_eyeperm": _paired_delta_summary(paired_1d, "matched", "shuffled"),
    }


def _safe_pair_min(a: float, b: float) -> float:
    vals = np.array([a, b], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.min(vals))


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    boot = np.empty(5000, dtype=np.float64)
    for idx in range(boot.size):
        sample_idx = rng.integers(0, values.size, size=values.size)
        boot[idx] = float(np.nanmedian(values[sample_idx]))
    return float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))


def run_empirical_geometry_ceiling(
    subject: str,
    date: str,
    dataset_configs_path: str,
    output_dir: Path,
    min_samples: int,
    n_split_repeats: int,
    n_random_subspaces: int,
    n_eye_permutations: int,
    rank_ratio_threshold: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / REFERENCE_RATE_HZ
    data = get_fixrsvp_data(
        subject=subject,
        date=date,
        dataset_configs_path=dataset_configs_path,
        use_cached_data=True,
    )
    pixels_per_degree = _resolve_pixels_per_degree(data)
    windows = _collect_image_windows(data, min_samples=min_samples, dt=dt)
    if not windows:
        return {"n_windows": 0, "subject": subject, "date": date}

    rng = np.random.default_rng(seed=abs(hash((subject, date, "emp_geom"))) % (2**31))
    rows: list[dict] = []

    for win in windows:
        _baseline_stim, baseline_eye_deg = _choose_baseline(win.stim, win.eyepos_deg)
        eyepos_px = (win.eyepos_deg - baseline_eye_deg[None, :]) * pixels_per_degree
        sigma_eye = np.cov(eyepos_px.T).astype(np.float64)

        actual = _split_half_fem_drive(
            robs=win.robs_rates,
            eyepos_px=eyepos_px,
            trial_indices=win.trial_indices,
            sigma_eye=sigma_eye,
            min_samples=10,
            n_shuffles=0,
            n_split_repeats=n_split_repeats,
            rng=rng,
            return_split_bundles=True,
        )
        eye_perm_runs = []
        eye_perm_metric_runs = []
        for _ in range(n_eye_permutations):
            eye_perm = _split_half_fem_drive(
                robs=win.robs_rates,
                eyepos_px=_permute_eye_positions(eyepos_px, rng),
                trial_indices=win.trial_indices,
                sigma_eye=sigma_eye,
                min_samples=10,
                n_shuffles=0,
                n_split_repeats=n_split_repeats,
                rng=rng,
                return_split_bundles=True,
            )
            eye_perm_runs.append(eye_perm)
            eye_perm_bundle = eye_perm.get("_split_bundle", {})
            eye_perm_metric_runs.append(_bundle_alignment_metrics(
                eye_perm_bundle.get("B_a", np.empty((0, 0, 0), dtype=np.float64)),
                eye_perm_bundle.get("B_b", np.empty((0, 0, 0), dtype=np.float64)),
            ))
        split_label_shuffle = _split_half_fem_drive(
            robs=win.robs_rates,
            eyepos_px=eyepos_px,
            trial_indices=rng.permutation(win.trial_indices),
            sigma_eye=sigma_eye,
            min_samples=10,
            n_shuffles=0,
            n_split_repeats=n_split_repeats,
            rng=rng,
            return_split_bundles=True,
        )
        eye_shift = _split_half_fem_drive(
            robs=win.robs_rates,
            eyepos_px=_circular_shift_eye_positions_within_trials(eyepos_px, win.trial_indices, rng),
            trial_indices=win.trial_indices,
            sigma_eye=sigma_eye,
            min_samples=10,
            n_shuffles=0,
            n_split_repeats=n_split_repeats,
            rng=rng,
            return_split_bundles=True,
        )

        actual_bundle = actual.get("_split_bundle", {})
        split_label_bundle = split_label_shuffle.get("_split_bundle", {})
        eye_shift_bundle = eye_shift.get("_split_bundle", {})
        actual_metrics = _bundle_alignment_metrics(
            actual_bundle.get("B_a", np.empty((0, 0, 0), dtype=np.float64)),
            actual_bundle.get("B_b", np.empty((0, 0, 0), dtype=np.float64)),
        )
        eye_perm_metrics = _concat_metric_dicts(eye_perm_metric_runs)
        split_label_metrics = _bundle_alignment_metrics(
            split_label_bundle.get("B_a", np.empty((0, 0, 0), dtype=np.float64)),
            split_label_bundle.get("B_b", np.empty((0, 0, 0), dtype=np.float64)),
        )
        eye_shift_metrics = _bundle_alignment_metrics(
            eye_shift_bundle.get("B_a", np.empty((0, 0, 0), dtype=np.float64)),
            eye_shift_bundle.get("B_b", np.empty((0, 0, 0), dtype=np.float64)),
        )
        random_align = _random_alignment_null(
            actual_bundle.get("B_b", np.empty((0, 0, 0), dtype=np.float64)),
            n_random_subspaces=n_random_subspaces,
            rng=rng,
        )

        actual_align2 = actual_metrics["alignment2_values"]
        actual_align1 = actual_metrics["alignment1_values"]
        eye_perm_align2 = eye_perm_metrics["alignment2_values"]
        eye_perm_align1 = eye_perm_metrics["alignment1_values"]
        split_label_align2 = split_label_metrics["alignment2_values"]
        actual_capture2 = actual_metrics["capture2_values"]
        actual_capture1 = actual_metrics["capture1_values"]
        eye_perm_capture2 = eye_perm_metrics["capture2_values"]
        eye_perm95 = float(np.nanpercentile(eye_perm_align2, 95)) if eye_perm_align2.size else float("nan")
        eye_perm95_top1 = float(np.nanpercentile(eye_perm_align1, 95)) if eye_perm_align1.size else float("nan")
        vec_corr_vals = actual_metrics["vec_corr_values"]
        B_a_norm_vals = actual_metrics["B_a_norm_values"]
        B_b_norm_vals = actual_metrics["B_b_norm_values"]
        B_a_rank_vals = actual_metrics["B_a_rank_ratio_values"]
        B_b_rank_vals = actual_metrics["B_b_rank_ratio_values"]

        min_rank_ratio = float(np.nanmedian(np.minimum(B_a_rank_vals, B_b_rank_vals))) if B_a_rank_vals.size and B_b_rank_vals.size else float("nan")
        use_top1 = bool(np.isfinite(min_rank_ratio) and min_rank_ratio < rank_ratio_threshold)

        rows.append({
            "image_id": int(win.image_id),
            "window_id": f"image_{win.image_id}",
            "n_samples": int(len(win.trial_indices)),
            "n_split_repeats_valid": int(actual.get("n_split_repeats_valid", 0)),
            "e_fem_cv_median": float(actual.get("e_fem_cv_median", float("nan"))),
            "b_split_corr_median": float(actual.get("b_split_corr_median", float("nan"))),
            "emp_split_alignment_2d": float(np.nanmedian(actual_align2)) if actual_align2.size else float("nan"),
            "emp_split_alignment_top1": float(np.nanmedian(actual_align1)) if actual_align1.size else float("nan"),
            "eye_perm_alignment_2d": float(np.nanmedian(eye_perm_align2)) if eye_perm_align2.size else float("nan"),
            "eye_perm_alignment_top1": float(np.nanmedian(eye_perm_align1)) if eye_perm_align1.size else float("nan"),
            "split_label_alignment_2d": float(np.nanmedian(split_label_align2)) if split_label_align2.size else float("nan"),
            "eye_shift_alignment_2d": float(np.nanmedian(eye_shift_metrics["alignment2_values"])) if eye_shift_metrics["alignment2_values"].size else float("nan"),
            "random_subspace_alignment_median": float(np.nanmedian(random_align)) if random_align.size else float("nan"),
            "emp_minus_eyeperm_alignment_2d": (
                float(np.nanmedian(actual_align2) - np.nanmedian(eye_perm_align2))
                if actual_align2.size and eye_perm_align2.size else float("nan")
            ),
            "emp_minus_eyeperm_alignment_top1": (
                float(np.nanmedian(actual_align1) - np.nanmedian(eye_perm_align1))
                if actual_align1.size and eye_perm_align1.size else float("nan")
            ),
            "emp_split_capture_2d": float(np.nanmedian(actual_capture2)) if actual_capture2.size else float("nan"),
            "emp_split_capture_top1": float(np.nanmedian(actual_capture1)) if actual_capture1.size else float("nan"),
            "eye_perm_capture_2d": float(np.nanmedian(eye_perm_capture2)) if eye_perm_capture2.size else float("nan"),
            "emp_split_principal_angle_1_deg": float(np.nanmedian(actual_metrics["angle1_values"])) if actual_metrics["angle1_values"].size else float("nan"),
            "emp_split_principal_angle_2_deg": float(np.nanmedian(actual_metrics["angle2_values"])) if actual_metrics["angle2_values"].size else float("nan"),
            "vec_corr_BA_BB": float(np.nanmedian(vec_corr_vals)) if vec_corr_vals.size else float("nan"),
            "B_A_norm_median": float(np.nanmedian(B_a_norm_vals)) if B_a_norm_vals.size else float("nan"),
            "B_B_norm_median": float(np.nanmedian(B_b_norm_vals)) if B_b_norm_vals.size else float("nan"),
            "B_A_rank_ratio_median": float(np.nanmedian(B_a_rank_vals)) if B_a_rank_vals.size else float("nan"),
            "B_B_rank_ratio_median": float(np.nanmedian(B_b_rank_vals)) if B_b_rank_vals.size else float("nan"),
            "min_B_rank_ratio": min_rank_ratio,
            "rank_adaptive_alignment": float(np.nanmedian(actual_align1 if use_top1 else actual_align2)) if (actual_align1.size if use_top1 else actual_align2.size) else float("nan"),
            "rank_adaptive_eye_perm_alignment": float(np.nanmedian(eye_perm_align1 if use_top1 else eye_perm_align2)) if (eye_perm_align1.size if use_top1 else eye_perm_align2.size) else float("nan"),
            "rank_adaptive_delta": (
                float(np.nanmedian((actual_align1 if use_top1 else actual_align2)) - np.nanmedian((eye_perm_align1 if use_top1 else eye_perm_align2)))
                if (actual_align1.size if use_top1 else actual_align2.size) and (eye_perm_align1.size if use_top1 else eye_perm_align2.size)
                else float("nan")
            ),
            "rank_adaptive_mode": "top1" if use_top1 else "2d",
            "eye_perm_alignment_2d_p95": eye_perm95,
            "eye_perm_alignment_top1_p95": eye_perm95_top1,
            "emp_alignment_gt_eyeperm_p95": bool(np.nanmedian(actual_align2) > eye_perm95) if actual_align2.size and np.isfinite(eye_perm95) else False,
        })

    csv_path = output_dir / "empirical_geometry_ceiling.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    actual_vals_2d = np.array([row["emp_split_alignment_2d"] for row in rows], dtype=np.float64)
    actual_vals_1d = np.array([row["emp_split_alignment_top1"] for row in rows], dtype=np.float64)
    eye_perm_vals_2d = np.array([row["eye_perm_alignment_2d"] for row in rows], dtype=np.float64)
    eye_perm_vals_1d = np.array([row["eye_perm_alignment_top1"] for row in rows], dtype=np.float64)
    split_label_vals = np.array([row["split_label_alignment_2d"] for row in rows], dtype=np.float64)
    eye_shift_vals = np.array([row["eye_shift_alignment_2d"] for row in rows], dtype=np.float64)
    random_vals = np.array([row["random_subspace_alignment_median"] for row in rows], dtype=np.float64)
    capture_vals = np.array([row["emp_split_capture_2d"] for row in rows], dtype=np.float64)
    delta_vals_2d = np.array([row["emp_minus_eyeperm_alignment_2d"] for row in rows], dtype=np.float64)
    delta_vals_1d = np.array([row["emp_minus_eyeperm_alignment_top1"] for row in rows], dtype=np.float64)
    rank_adaptive_vals = np.array([row["rank_adaptive_alignment"] for row in rows], dtype=np.float64)
    rank_adaptive_eye_perm_vals = np.array([row["rank_adaptive_eye_perm_alignment"] for row in rows], dtype=np.float64)
    rank_adaptive_delta_vals = np.array([row["rank_adaptive_delta"] for row in rows], dtype=np.float64)
    summary_rng = np.random.default_rng(0)
    ci2_low, ci2_high = _bootstrap_ci(actual_vals_2d, summary_rng)
    ci1_low, ci1_high = _bootstrap_ci(actual_vals_1d, summary_rng)
    delta2_low, delta2_high = _bootstrap_ci(delta_vals_2d, summary_rng)
    delta1_low, delta1_high = _bootstrap_ci(delta_vals_1d, summary_rng)
    rank_adapt_low, rank_adapt_high = _bootstrap_ci(rank_adaptive_delta_vals, summary_rng)

    norm_vals = np.array([_safe_pair_min(row["B_A_norm_median"], row["B_B_norm_median"]) for row in rows], dtype=np.float64)
    rank_vals = np.array([row["min_B_rank_ratio"] for row in rows], dtype=np.float64)
    e_cv_vals = np.array([row["e_fem_cv_median"] for row in rows], dtype=np.float64)
    subset_summaries = {
        "all_images": _subset_summary(rows, "all_images", np.ones(len(rows), dtype=bool)),
        "high_norm": _subset_summary(rows, "high_norm", norm_vals >= np.nanmedian(norm_vals)),
        "rank_ratio_gt_0p1": _subset_summary(rows, "rank_ratio_gt_0p1", rank_vals > 0.1),
        "rank_ratio_gt_0p2": _subset_summary(rows, "rank_ratio_gt_0p2", rank_vals > 0.2),
        "e_fem_cv_pos": _subset_summary(rows, "e_fem_cv_pos", e_cv_vals > 0.0),
        "high_samples": _subset_summary(rows, "high_samples", np.array([row["n_samples"] for row in rows], dtype=np.float64) >= np.nanpercentile(np.array([row["n_samples"] for row in rows], dtype=np.float64), 50)),
    }

    paired_2d = _paired_delta_summary(
        [{"matched": row["emp_split_alignment_2d"], "shuffled": row["eye_perm_alignment_2d"]} for row in rows],
        "matched",
        "shuffled",
    )
    paired_1d = _paired_delta_summary(
        [{"matched": row["emp_split_alignment_top1"], "shuffled": row["eye_perm_alignment_top1"]} for row in rows],
        "matched",
        "shuffled",
    )
    paired_rank_adaptive = _paired_delta_summary(
        [{"matched": row["rank_adaptive_alignment"], "shuffled": row["rank_adaptive_eye_perm_alignment"]} for row in rows],
        "matched",
        "shuffled",
    )

    summary = {
        "subject": subject,
        "date": date,
        "n_windows": len(rows),
        "median_emp_split_alignment_2d": float(np.nanmedian(actual_vals_2d)),
        "median_emp_split_alignment_top1": float(np.nanmedian(actual_vals_1d)),
        "median_eye_perm_alignment_2d": float(np.nanmedian(eye_perm_vals_2d)),
        "median_eye_perm_alignment_top1": float(np.nanmedian(eye_perm_vals_1d)),
        "median_eye_perm_alignment_2d_p95": float(np.nanmedian([row["eye_perm_alignment_2d_p95"] for row in rows])),
        "median_eye_perm_alignment_top1_p95": float(np.nanmedian([row["eye_perm_alignment_top1_p95"] for row in rows])),
        "median_split_label_alignment_2d": float(np.nanmedian(split_label_vals)),
        "median_eye_shift_alignment_2d": float(np.nanmedian(eye_shift_vals)),
        "median_random_alignment": float(np.nanmedian(random_vals)),
        "median_emp_split_capture_2d": float(np.nanmedian(capture_vals)),
        "median_alignment_delta_2d": float(np.nanmedian(delta_vals_2d)),
        "median_alignment_delta_top1": float(np.nanmedian(delta_vals_1d)),
        "median_rank_adaptive_alignment": float(np.nanmedian(rank_adaptive_vals)),
        "median_rank_adaptive_eye_perm_alignment": float(np.nanmedian(rank_adaptive_eye_perm_vals)),
        "median_rank_adaptive_delta": float(np.nanmedian(rank_adaptive_delta_vals)),
        "median_emp_split_alignment_2d_ci_low": ci2_low,
        "median_emp_split_alignment_2d_ci_high": ci2_high,
        "median_emp_split_alignment_top1_ci_low": ci1_low,
        "median_emp_split_alignment_top1_ci_high": ci1_high,
        "median_alignment_delta_2d_ci_low": delta2_low,
        "median_alignment_delta_2d_ci_high": delta2_high,
        "median_alignment_delta_top1_ci_low": delta1_low,
        "median_alignment_delta_top1_ci_high": delta1_high,
        "median_rank_adaptive_delta_ci_low": rank_adapt_low,
        "median_rank_adaptive_delta_ci_high": rank_adapt_high,
        "fraction_images_gt_eyeperm_p95": float(np.nanmean([row["emp_alignment_gt_eyeperm_p95"] for row in rows])),
        "median_vec_corr_BA_BB": float(np.nanmedian([row["vec_corr_BA_BB"] for row in rows])),
        "median_min_B_rank_ratio": float(np.nanmedian(rank_vals)),
        "paired_alignment2_vs_eyeperm": paired_2d,
        "paired_alignment1_vs_eyeperm": paired_1d,
        "paired_rank_adaptive_vs_eyeperm": paired_rank_adaptive,
        "n_split_repeats": int(n_split_repeats),
        "n_random_subspaces": int(n_random_subspaces),
        "n_eye_permutations": int(n_eye_permutations),
        "rank_ratio_threshold": float(rank_ratio_threshold),
    }
    (output_dir / "empirical_geometry_ceiling_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "empirical_geometry_ceiling_reliability_summary.json").write_text(json.dumps(subset_summaries, indent=2) + "\n")
    _write_distribution_figure(rows, output_dir, summary)
    return summary


def _write_distribution_figure(rows: list[dict], output_dir: Path, summary: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    actual = np.array([row["emp_split_alignment_2d"] for row in rows], dtype=np.float64)
    eye_perm = np.array([row["eye_perm_alignment_2d"] for row in rows], dtype=np.float64)
    split_label = np.array([row["split_label_alignment_2d"] for row in rows], dtype=np.float64)
    random = np.array([row["random_subspace_alignment_median"] for row in rows], dtype=np.float64)
    delta = np.array([row["emp_minus_eyeperm_alignment_2d"] for row in rows], dtype=np.float64)
    top1_actual = np.array([row["emp_split_alignment_top1"] for row in rows], dtype=np.float64)
    top1_eye_perm = np.array([row["eye_perm_alignment_top1"] for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].boxplot([actual, eye_perm, split_label, random], tick_labels=["emp 2D", "eye-perm", "split-label", "random"])
    axes[0].set_ylabel("Split-half alignment")
    axes[0].set_title("2D geometry ceiling")

    axes[1].boxplot([top1_actual, top1_eye_perm], tick_labels=["emp top1", "eye-perm top1"])
    axes[1].set_ylabel("Top-1 alignment")
    axes[1].set_title("Top-1 geometry ceiling")

    axes[2].hist(delta[np.isfinite(delta)], bins=12, color="#2c7bb6", alpha=0.85)
    axes[2].axvline(0.0, color="#c44e52", linestyle="--", linewidth=1.0)
    axes[2].set_xlabel("Empirical - eye-perm alignment")
    axes[2].set_ylabel("Image windows")
    axes[2].set_title(
        "Median delta = {delta:.3f}".format(delta=summary.get("median_alignment_delta_2d", float("nan")))
    )

    fig.tight_layout()
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "empirical_geometry_ceiling_distribution.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")

    canonical = FIGURES_DIR / "jacobian_predictive_framework"
    canonical.mkdir(parents=True, exist_ok=True)
    fig.savefig(canonical / f"empirical_geometry_ceiling_distribution_{summary.get('date', 'unknown')}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--dataset-configs-path", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--n-split-repeats", type=int, default=100)
    parser.add_argument("--n-random-subspaces", type=int, default=100)
    parser.add_argument("--n-eye-permutations", type=int, default=25)
    parser.add_argument("--rank-ratio-threshold", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_empirical_geometry_ceiling(
        subject=args.subject,
        date=args.date,
        dataset_configs_path=args.dataset_configs_path,
        output_dir=Path(args.output_dir),
        min_samples=args.min_samples,
        n_split_repeats=args.n_split_repeats,
        n_random_subspaces=args.n_random_subspaces,
        n_eye_permutations=args.n_eye_permutations,
        rank_ratio_threshold=args.rank_ratio_threshold,
    )
    print(json.dumps(result, indent=2))