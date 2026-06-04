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

from . import run_stg_tangent_stage1 as stg1
from .utils import DEFAULT_OUT_ROOT, harmonize_fixrsvp_arrays


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_k_list(raw: str) -> list[int]:
    parts = [piece.strip() for piece in str(raw).split(",") if piece.strip()]
    return [int(piece) for piece in parts] if parts else [0]


def _source_rates(args: argparse.Namespace, data: dict[str, Any]) -> tuple[np.ndarray, int | None]:
    return stg1._build_source_rates(args, data)


def _sample_images(
    rates: np.ndarray,
    eyepos: np.ndarray,
    image_ids: np.ndarray,
    stim: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    sample_mode: str,
    n_samples_threshold: int,
    min_samples: int,
    seed: int,
) -> tuple[dict[int, dict[str, Any]], np.ndarray]:
    if valid_mask is None:
        valid = np.isfinite(rates).all(axis=2) & np.isfinite(eyepos).all(axis=2) & (image_ids >= 0)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
    n_trials = int(image_ids.shape[0])
    n_time = int(image_ids.shape[1])
    time_grid = np.broadcast_to(np.arange(n_time, dtype=np.float64)[None, :], (n_trials, n_time))
    sampled: dict[int, dict[str, Any]] = {}
    for img in sorted(int(i) for i in np.unique(image_ids[valid])):
        mask = valid & (image_ids == img)
        y_all = rates[mask]
        x_all = eyepos[mask]
        t_all = time_grid[mask]
        s_all = stim[mask]
        n_available = int(y_all.shape[0])
        if sample_mode == "fixed_n":
            if n_available < n_samples_threshold:
                continue
            img_rng = np.random.default_rng(int(seed) + (10007 * int(img)))
            pick = img_rng.permutation(n_available)[:n_samples_threshold]
            y = y_all[pick]
            x = x_all[pick]
            tt = t_all[pick]
            ss = s_all[pick]
            n_used = int(n_samples_threshold)
        else:
            if n_available < min_samples:
                continue
            y = y_all
            x = x_all
            tt = t_all
            ss = s_all
            n_used = int(y.shape[0])
        sampled[img] = {
            "y": np.asarray(y, dtype=np.float64),
            "x": np.asarray(x, dtype=np.float64),
            "t": np.asarray(tt, dtype=np.float64),
            "stim": np.asarray(ss, dtype=np.float64),
            "n_available": n_available,
            "n_used": n_used,
            "n_units": int(y.shape[1]),
        }
    return sampled, time_grid


def _global_axes(rates: np.ndarray, valid: np.ndarray, time_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_time = int(time_grid.shape[1])
    global_rate_axis = np.zeros(rates.shape[2], dtype=np.float64)
    global_pc1_axis = np.zeros(rates.shape[2], dtype=np.float64)
    global_mean_by_time = np.zeros(n_time, dtype=np.float64)
    global_pc1_by_time = np.zeros(n_time, dtype=np.float64)
    y_valid = rates[valid]
    if y_valid.ndim == 2 and y_valid.shape[0] >= 2:
        global_rate_axis = stg1._safe_unit_axis(np.nanmean(y_valid, axis=0))
        yc = y_valid - np.nanmean(y_valid, axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(yc, full_matrices=False)
        if vh.size:
            global_pc1_axis = stg1._safe_unit_axis(vh[0])

    for t in range(n_time):
        mt = valid[:, t]
        if int(np.sum(mt)) > 0:
            global_mean_by_time[t] = float(np.nanmean(rates[:, t, :][mt]))
            rtm = rates[:, t, :][mt]
            if rtm.shape[0] >= 2:
                rt_centered = rtm - np.mean(rtm, axis=0, keepdims=True)
                _, _, vh_t = np.linalg.svd(rt_centered, full_matrices=False)
                if vh_t.size:
                    global_pc1_by_time[t] = float(np.mean(rt_centered @ vh_t[0]))
    return global_rate_axis, global_pc1_axis, global_mean_by_time, global_pc1_by_time


def _shared_basis(sampled: dict[int, dict[str, Any]], global_axes: list[np.ndarray], k: int) -> tuple[np.ndarray, np.ndarray, float]:
    pooled = []
    for img in sorted(sampled):
        y = np.asarray(sampled[img]["y"], dtype=np.float64)
        y = stg1._project_responses_out_axes(y, global_axes) if global_axes else y
        pooled.append(y)
    pooled_arr = np.concatenate(pooled, axis=0) if pooled else np.zeros((0, 0), dtype=np.float64)
    if k <= 0 or pooled_arr.size == 0:
        return pooled_arr, np.zeros((pooled_arr.shape[1] if pooled_arr.ndim == 2 else 0, 0), dtype=np.float64), 0.0
    centered = pooled_arr - np.mean(pooled_arr, axis=0, keepdims=True)
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    kk = int(min(k, vh.shape[0]))
    basis = np.asarray(vh[:kk].T, dtype=np.float64)
    var_expl = float(np.sum(s[:kk] ** 2) / (np.sum(s ** 2) + 1e-12))
    return pooled_arr, basis, var_expl


def _project_with_shared_basis(y: np.ndarray, pooled_mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if basis.size == 0:
        return np.asarray(y, dtype=np.float64)
    centered = np.asarray(y, dtype=np.float64) - pooled_mean
    return pooled_mean + centered - (centered @ basis) @ basis.T


def _template_2d(stim_slice: np.ndarray) -> np.ndarray | None:
    return stg1._extract_template_2d(stim_slice)


def _control_metrics() -> tuple[str, ...]:
    return stg1.CONTROL_METRICS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Recorded-only shared-mode projection sweep for Stage 1")
    p.add_argument("--subject", type=str, required=True)
    p.add_argument("--date", type=str, required=True)
    p.add_argument("--dataset-configs-path", type=Path, default=Path("experiments") / "dataset_configs" / "multi_basic_240_rsvp.yaml")
    p.add_argument("--source", choices=("recorded", "twin"), default="recorded")
    p.add_argument("--sample-mode", choices=("all_available", "fixed_n"), default="fixed_n")
    p.add_argument("--n-samples-threshold", type=int, default=320)
    p.add_argument("--min-samples", type=int, default=320)
    nuisance_choices = ("none", "time", "time_global", "time_fixed_effect", "time_spline", "time_global_fixed_effect", "time_global_spline")
    p.add_argument("--nuisance", choices=nuisance_choices, default="time_global")
    p.add_argument("--axis-projection", choices=("none", "global_rate", "pc1", "both"), default="both")
    p.add_argument("--shared-mode-projection-k", type=str, default="0,1,2,3,5,10")
    p.add_argument("--recorded-nuisance", choices=nuisance_choices, default=None)
    p.add_argument("--recorded-axis-projection", choices=("none", "global_rate", "pc1", "both"), default=None)
    p.add_argument("--recorded-shared-mode-projection-k", type=str, default=None)
    p.add_argument("--ridge-alpha", type=float, default=1e-3)
    p.add_argument("--rank-tol-rel", type=float, default=1e-6)
    p.add_argument("--n-nulls", type=int, default=200)
    p.add_argument("--bootstrap-repeats", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--predict-batch-size", type=int, default=64)
    p.add_argument("--model-device", type=str, default="cuda")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--use-cached-data", action="store_true", default=True)
    p.add_argument("--drift-only", action="store_true", help="Restrict recorded samples to drift-only windows")
    p.add_argument("--drift-eye-smooth-sigma-bins", type=float, default=1.0)
    p.add_argument("--drift-speed-percentile", type=float, default=99.5)
    p.add_argument("--drift-amp-thresh-deg", type=float, default=0.12)
    p.add_argument("--drift-refrac-ms", type=float, default=20.0)
    p.add_argument("--drift-exclusion-pre-ms", type=float, default=50.0)
    p.add_argument("--drift-exclusion-post-ms", type=float, default=150.0)
    return p


def _resolve_source_options(args: argparse.Namespace) -> tuple[str, str, str]:
    nuisance = str(args.recorded_nuisance) if args.recorded_nuisance is not None else str(args.nuisance)
    axis_projection = str(args.recorded_axis_projection) if args.recorded_axis_projection is not None else str(args.axis_projection)
    shared_k = str(args.recorded_shared_mode_projection_k) if args.recorded_shared_mode_projection_k is not None else str(args.shared_mode_projection_k)
    return nuisance, axis_projection, shared_k


def _fit_one_projection(
    *,
    args: argparse.Namespace,
    k: int,
    sampled: dict[int, dict[str, Any]],
    global_rate_axis: np.ndarray,
    global_pc1_axis: np.ndarray,
    global_mean_by_time: np.ndarray,
    global_pc1_by_time: np.ndarray,
    drift_support: dict[str, object],
    n_images_meeting_sampling_before: int,
    n_images_meeting_sampling_after: int,
    source: str,
    nuisance_model: str,
    axis_projection: str,
) -> dict[str, object]:
    global_axes = []
    if axis_projection in ("global_rate", "both"):
        global_axes.append(global_rate_axis)
    if axis_projection in ("pc1", "both"):
        global_axes.append(global_pc1_axis)

    pooled_arr, shared_basis, var_expl = _shared_basis(sampled, global_axes, k)
    pooled_mean = np.mean(pooled_arr, axis=0, keepdims=True) if pooled_arr.size else np.zeros((1, next(iter(sampled.values()))["n_units"]), dtype=np.float64)

    per_image: dict[int, dict[str, Any]] = {}
    image_metric_rows: list[dict[str, object]] = []
    image_templates: dict[int, np.ndarray] = {}
    for img in sorted(sampled):
        y = np.asarray(sampled[img]["y"], dtype=np.float64)
        x = np.asarray(sampled[img]["x"], dtype=np.float64)
        tt = np.asarray(sampled[img]["t"], dtype=np.float64)
        stim = np.asarray(sampled[img]["stim"], dtype=np.float64)

        if global_axes:
            y = stg1._project_responses_out_axes(y, global_axes)
        if k > 0 and shared_basis.size:
            y = _project_with_shared_basis(y, pooled_mean, shared_basis)

        dxdy = x - np.mean(x, axis=0, keepdims=True)
        nuisance_cols: list[np.ndarray] = []
        if nuisance_model != "none":
            t_z = (tt - float(np.mean(tt))) / (float(np.std(tt)) + 1e-12)
            if nuisance_model in {"time", "time_global"}:
                nuisance_cols.extend([t_z, t_z * t_z])
            elif nuisance_model in {"time_fixed_effect", "time_global_fixed_effect"}:
                t_int = np.clip(np.rint(tt).astype(np.int64), 0, global_mean_by_time.shape[0] - 1)
                one_hot = np.eye(global_mean_by_time.shape[0], dtype=np.float64)[t_int]
                if one_hot.shape[1] > 1:
                    nuisance_cols.append(one_hot[:, 1:])
            elif nuisance_model in {"time_spline", "time_global_spline"}:
                t_unit = np.asarray(tt, dtype=np.float64)
                t_unit = (t_unit - np.nanmin(t_unit)) / (np.nanmax(t_unit) - np.nanmin(t_unit) + 1e-12)
                nuisance_cols.extend([t_unit, t_unit * t_unit, t_unit * t_unit * t_unit])

            if nuisance_model in {"time_global", "time_global_fixed_effect", "time_global_spline"}:
                t_int = np.clip(np.rint(tt).astype(np.int64), 0, global_mean_by_time.shape[0] - 1)
                nuisance_cols.append(global_mean_by_time[t_int])
                nuisance_cols.append(global_pc1_by_time[t_int])

        y_fit = stg1._fit_unitwise_nuisance_residual(y, nuisance_cols)
        bx, by, j, r2 = stg1._fit_tangent_map(y_fit, dxdy, ridge_alpha=float(args.ridge_alpha))
        basis, rank_j, svals, frac1, frac2 = stg1._j_diagnostics(j, rank_tol_rel=float(args.rank_tol_rel))
        image_metric_rows.append(
            {
                "session_id": f"{args.subject}_{args.date}",
                "subject": args.subject,
                "date": args.date,
                "source": str(source),
                "image_set": "high_support",
                "analysis_representation": "raw_samples",
                "sample_mode": args.sample_mode,
                "axis_projection": str(axis_projection),
                "nuisance_model": str(nuisance_model),
                "recorded_axis_projection": str(axis_projection),
                "recorded_nuisance": str(nuisance_model),
                "recorded_shared_mode_projection_k": int(k),
                "image_id": int(img),
                "n_samples_available": int(sampled[img]["n_available"]),
                "n_samples_used": int(sampled[img]["n_used"]),
                "n_units": int(sampled[img]["n_units"]),
                "r2_mean": float(np.mean(r2)),
                "r2_median": float(np.median(r2)),
                "norm_bx": float(np.linalg.norm(bx)),
                "norm_by": float(np.linalg.norm(by)),
                "angle_between_bx_by": float(np.degrees(np.arccos(np.clip(stg1._cos(bx, by), -1.0, 1.0)))),
                "condition_number_J": float(np.linalg.cond(j)) if np.all(np.isfinite(j)) else float("inf"),
                "rank_J": int(rank_j),
                "align_bx_global_rate_axis": float(stg1._cos(bx, global_rate_axis)),
                "align_bx_global_pc1_axis": float(stg1._cos(bx, global_pc1_axis)),
                "singular_values_J": json.dumps([float(v) for v in svals.tolist()]),
                "frac_energy_colspace_top1": float(frac1),
                "frac_energy_colspace_top2": float(frac2),
                "shared_mode_basis_source": "global_response_pca",
                "n_modes_projected": int(k),
                "variance_explained_by_projected_modes": float(var_expl),
            }
        )
        per_image[img] = {
            "bx": bx,
            "by": by,
            "basis": basis,
            "dxdy": dxdy,
            "y": y_fit,
            "n_units": int(sampled[img]["n_units"]),
            "n_samples_available": int(sampled[img]["n_available"]),
            "n_samples_used": int(sampled[img]["n_used"]),
            "rank_j": int(rank_j),
        }
        tpl = _template_2d(stim)
        if tpl is not None:
            image_templates[img] = tpl

    alignment_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(int(args.seed) + int(k) * 101)
    usable_ids = sorted(per_image)
    for img_i, img_j in itertools.combinations(usable_ids, 2):
        a = per_image[img_i]
        b = per_image[img_j]
        cos_bx = stg1._cos(a["bx"], b["bx"])
        cos_by = stg1._cos(a["by"], b["by"])
        signed = float(np.nanmean([cos_bx, cos_by]))
        subspace = stg1._subspace_overlap(a["basis"], b["basis"])
        sim = {"pixel_correlation": float("nan"), "rms_contrast_difference": float("nan"), "fourier_amplitude_similarity": float("nan")}
        if img_i in image_templates and img_j in image_templates:
            sim = stg1._compute_image_similarity(image_templates[img_i], image_templates[img_j])

        eye_null_signed = []
        eye_null_subspace = []
        rand_null_signed = []
        rand_null_subspace = []
        for _ in range(int(args.n_nulls)):
            perm_i = rng.permutation(a["dxdy"].shape[0])
            perm_j = rng.permutation(b["dxdy"].shape[0])
            bx_i_s, by_i_s, j_i_s, _ = stg1._fit_tangent_map(a["y"], a["dxdy"][perm_i], ridge_alpha=float(args.ridge_alpha))
            bx_j_s, by_j_s, j_j_s, _ = stg1._fit_tangent_map(b["y"], b["dxdy"][perm_j], ridge_alpha=float(args.ridge_alpha))
            basis_i_s, _, _, _, _ = stg1._j_diagnostics(j_i_s, rank_tol_rel=float(args.rank_tol_rel))
            basis_j_s, _, _, _, _ = stg1._j_diagnostics(j_j_s, rank_tol_rel=float(args.rank_tol_rel))
            eye_null_signed.append(float(np.nanmean([stg1._cos(bx_i_s, bx_j_s), stg1._cos(by_i_s, by_j_s)])))
            eye_null_subspace.append(stg1._subspace_overlap(basis_i_s, basis_j_s))
            rbx_i, rby_i, rj_i = stg1._random_map_with_norms(a["n_units"], float(np.linalg.norm(a["bx"])), float(np.linalg.norm(a["by"])), rng)
            rbx_j, rby_j, rj_j = stg1._random_map_with_norms(b["n_units"], float(np.linalg.norm(b["bx"])), float(np.linalg.norm(b["by"])), rng)
            rbasis_i, _, _, _, _ = stg1._j_diagnostics(rj_i, rank_tol_rel=float(args.rank_tol_rel))
            rbasis_j, _, _, _, _ = stg1._j_diagnostics(rj_j, rank_tol_rel=float(args.rank_tol_rel))
            rand_null_signed.append(float(np.nanmean([stg1._cos(rbx_i, rbx_j), stg1._cos(rby_i, rby_j)])))
            rand_null_subspace.append(stg1._subspace_overlap(rbasis_i, rbasis_j))

        eye_null_signed_arr = np.asarray(eye_null_signed, dtype=np.float64)
        rand_null_signed_arr = np.asarray(rand_null_signed, dtype=np.float64)
        eye_null_subspace_arr = np.asarray(eye_null_subspace, dtype=np.float64)
        rand_null_subspace_arr = np.asarray(rand_null_subspace, dtype=np.float64)
        alignment_rows.append(
            {
                "session_id": f"{args.subject}_{args.date}",
                "subject": args.subject,
                "date": args.date,
                "source": "recorded",
                "recorded_shared_mode_projection_k": int(k),
                "image_i": int(img_i),
                "image_j": int(img_j),
                "mean_signed_column_alignment": signed,
                "subspace_overlap_k2": subspace,
                "null_eyeshuffle_mean": float(np.mean(eye_null_signed_arr)),
                "null_random_mean": float(np.mean(rand_null_signed_arr)),
                "null_eyeshuffle_subspace_mean": float(np.mean(eye_null_subspace_arr)),
                "null_random_subspace_mean": float(np.mean(rand_null_subspace_arr)),
                "pixel_correlation": float(sim["pixel_correlation"]),
                "rms_contrast_difference": float(sim["rms_contrast_difference"]),
                "fourier_amplitude_similarity": float(sim["fourier_amplitude_similarity"]),
            }
        )
        pair_rows.append(
            {
                "image_i": int(img_i),
                "image_j": int(img_j),
                "signed": float(signed),
                "subspace": float(subspace),
                "diff_signed_eye": float(signed - float(np.mean(eye_null_signed_arr))),
                "diff_signed_random": float(signed - float(np.mean(rand_null_signed_arr))),
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

    signed_vals = np.asarray([float(r["signed"]) for r in pair_rows], dtype=np.float64)
    subspace_vals = np.asarray([float(r["subspace"]) for r in pair_rows], dtype=np.float64)
    diff_signed_eye = np.asarray([float(r["diff_signed_eye"]) for r in pair_rows], dtype=np.float64)
    diff_signed_rand = np.asarray([float(r["diff_signed_random"]) for r in pair_rows], dtype=np.float64)
    diff_sub_eye = np.asarray([float(r["diff_subspace_eye"]) for r in pair_rows], dtype=np.float64)
    diff_sub_rand = np.asarray([float(r["diff_subspace_random"]) for r in pair_rows], dtype=np.float64)
    boot_diff_signed_eye = stg1._image_bootstrap_weighted_mean(pair_rows, "diff_signed_eye", usable_ids, seed=int(args.seed) + 1, n_bootstrap=int(args.bootstrap_repeats))
    boot_diff_signed_rand = stg1._image_bootstrap_weighted_mean(pair_rows, "diff_signed_random", usable_ids, seed=int(args.seed) + 2, n_bootstrap=int(args.bootstrap_repeats))
    boot_diff_sub_eye = stg1._image_bootstrap_weighted_mean(pair_rows, "diff_subspace_eye", usable_ids, seed=int(args.seed) + 3, n_bootstrap=int(args.bootstrap_repeats))
    boot_diff_sub_rand = stg1._image_bootstrap_weighted_mean(pair_rows, "diff_subspace_random", usable_ids, seed=int(args.seed) + 4, n_bootstrap=int(args.bootstrap_repeats))
    ci_signed_eye = stg1._bootstrap_ci(boot_diff_signed_eye)
    ci_signed_rand = stg1._bootstrap_ci(boot_diff_signed_rand)
    ci_sub_eye = stg1._bootstrap_ci(boot_diff_sub_eye)
    ci_sub_rand = stg1._bootstrap_ci(boot_diff_sub_rand)

    controlled: dict[str, object] = {}
    control_keys = _control_metrics()
    for idx, control_key in enumerate(control_keys, start=1):
        int_eye = stg1._image_bootstrap_adjusted_intercept(pair_rows, "diff_signed_eye", control_key, usable_ids, seed=int(args.seed) + 200 + idx, n_bootstrap=int(args.bootstrap_repeats))
        int_rand = stg1._image_bootstrap_adjusted_intercept(pair_rows, "diff_signed_random", control_key, usable_ids, seed=int(args.seed) + 300 + idx, n_bootstrap=int(args.bootstrap_repeats))
        low_eye_rows, low_thr, low_n_eye = stg1._fit_low_similarity(pair_rows, "diff_signed_eye", control_key)
        low_rand_rows, _, low_n_rand = stg1._fit_low_similarity(pair_rows, "diff_signed_random", control_key)
        controlled[f"controlled_effect_minus_eye_shuffle_{control_key}"] = float(np.mean(int_eye)) if int_eye.size else float("nan")
        controlled[f"controlled_effect_minus_random_map_{control_key}"] = float(np.mean(int_rand)) if int_rand.size else float("nan")
        controlled[f"low_similarity_threshold_{control_key}"] = float(low_thr)
        controlled[f"low_similarity_n_pairs_minus_eye_shuffle_{control_key}"] = int(low_n_eye)
        controlled[f"low_similarity_effect_minus_eye_shuffle_{control_key}"] = float(np.nanmean(np.asarray([float(r["diff_signed_eye"]) for r in low_eye_rows], dtype=np.float64))) if low_eye_rows else float("nan")
        controlled[f"low_similarity_n_pairs_minus_random_map_{control_key}"] = int(low_n_rand)
        controlled[f"low_similarity_effect_minus_random_map_{control_key}"] = float(np.nanmean(np.asarray([float(r["diff_signed_random"]) for r in low_rand_rows], dtype=np.float64))) if low_rand_rows else float("nan")

    ctrl_eye_vals = [float(controlled[f"controlled_effect_minus_eye_shuffle_{c}"]) for c in control_keys if np.isfinite(float(controlled[f"controlled_effect_minus_eye_shuffle_{c}"]))]
    ctrl_rand_vals = [float(controlled[f"controlled_effect_minus_random_map_{c}"]) for c in control_keys if np.isfinite(float(controlled[f"controlled_effect_minus_random_map_{c}"]))]
    controlled_regression_status = "suspect_identical_controls" if ctrl_eye_vals and ctrl_rand_vals and (max(ctrl_eye_vals) - min(ctrl_eye_vals) < 1e-12) and (max(ctrl_rand_vals) - min(ctrl_rand_vals) < 1e-12) else "ok"

    summary = {
        "session_id": f"{args.subject}_{args.date}",
        "subject": args.subject,
        "date": args.date,
        "source": str(source),
        "image_set": "high_support",
        "analysis_representation": "raw_samples",
        "sample_mode": args.sample_mode,
        "drift_only": bool(args.drift_only),
        "bootstrap_unit": "image",
        "nuisance_model": str(nuisance_model),
        "axis_projection": str(axis_projection),
        "recorded_nuisance": str(nuisance_model),
        "recorded_axis_projection": str(axis_projection),
        "recorded_shared_mode_projection_k": int(k),
        "projection_k": int(k),
        "shared_mode_basis_source": "global_response_pca",
        "n_modes_projected": int(k),
        "variance_explained_by_projected_modes": float(var_expl),
        "n_images": int(len(usable_ids)),
        "n_pairs": int(len(pair_rows)),
        "n_images_with_samples_before_exclusion": int(drift_support.get("n_images_with_samples_before_exclusion", 0)),
        "n_images_with_samples_after_exclusion": int(drift_support.get("n_images_with_samples_after_exclusion", 0)),
        "n_images_meeting_sampling_before_exclusion": int(n_images_meeting_sampling_before),
        "n_images_meeting_sampling_after_exclusion": int(n_images_meeting_sampling_after),
        "n_valid_samples_before_exclusion": int(drift_support.get("n_valid_samples_before_exclusion", 0)),
        "n_valid_samples_excluded": int(drift_support.get("n_valid_samples_excluded", 0)),
        "n_valid_samples_after_exclusion": int(drift_support.get("n_valid_samples_after_exclusion", 0)),
        "fraction_valid_samples_after_exclusion": float(drift_support.get("fraction_valid_samples_after_exclusion", float("nan"))),
        "drift_n_events_detected": int(drift_support.get("drift_n_events_detected", 0)),
        "drift_speed_threshold_deg_s": float(drift_support.get("drift_speed_threshold_deg_s", float("nan"))),
        "n_samples_threshold": int(args.n_samples_threshold),
        "n_units": int(next(iter(per_image.values()))["n_units"]) if per_image else 0,
        "mean_signed_column_alignment": float(np.nanmean(signed_vals)) if signed_vals.size else float("nan"),
        "mean_subspace_overlap_k2": float(np.nanmean(subspace_vals)) if subspace_vals.size else float("nan"),
        "effect_minus_eye_shuffle": float(np.nanmean(diff_signed_eye)) if diff_signed_eye.size else float("nan"),
        "effect_minus_random_map": float(np.nanmean(diff_signed_rand)) if diff_signed_rand.size else float("nan"),
        "bootstrap_ci_low_minus_eye_shuffle": float(ci_signed_eye[0]),
        "bootstrap_ci_high_minus_eye_shuffle": float(ci_signed_eye[1]),
        "ci_low_minus_eye_shuffle": float(ci_signed_eye[0]),
        "ci_high_minus_eye_shuffle": float(ci_signed_eye[1]),
        "bootstrap_ci_low_minus_random": float(ci_signed_rand[0]),
        "bootstrap_ci_high_minus_random": float(ci_signed_rand[1]),
        "ci_low_minus_random_map": float(ci_signed_rand[0]),
        "ci_high_minus_random_map": float(ci_signed_rand[1]),
        "effect_minus_eye_shuffle_subspace": float(np.nanmean(diff_sub_eye)) if diff_sub_eye.size else float("nan"),
        "effect_minus_random_map_subspace": float(np.nanmean(diff_sub_rand)) if diff_sub_rand.size else float("nan"),
        "bootstrap_ci_low_minus_eye_shuffle_subspace": float(ci_sub_eye[0]),
        "bootstrap_ci_high_minus_eye_shuffle_subspace": float(ci_sub_eye[1]),
        "ci_low_minus_eye_shuffle_subspace": float(ci_sub_eye[0]),
        "ci_high_minus_eye_shuffle_subspace": float(ci_sub_eye[1]),
        "bootstrap_ci_low_minus_random_subspace": float(ci_sub_rand[0]),
        "bootstrap_ci_high_minus_random_subspace": float(ci_sub_rand[1]),
        "ci_low_minus_random_subspace": float(ci_sub_rand[0]),
        "ci_high_minus_random_subspace": float(ci_sub_rand[1]),
        "mean_align_bx_global_rate_axis": float(np.nanmean([float(v["align_bx_global_rate_axis"]) for v in image_metric_rows])) if image_metric_rows else float("nan"),
        "mean_align_bx_global_pc1_axis": float(np.nanmean([float(v["align_bx_global_pc1_axis"]) for v in image_metric_rows])) if image_metric_rows else float("nan"),
        "controlled_regression_status": controlled_regression_status,
        "control_is_evaluable": bool(np.isfinite(ci_signed_eye[0]) and np.isfinite(ci_signed_rand[0])),
        "interpretation_label": "tangent_shared_geometry" if np.isfinite(ci_signed_eye[0]) and np.isfinite(ci_signed_rand[0]) and ci_signed_eye[0] > 0.0 and ci_signed_rand[0] > 0.0 else "not_supported",
        **controlled,
    }

    out_dir = Path(args.out_dir) / f"{args.subject}_{args.date}" / f"source_{source}" / f"projection_k{int(k)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "stg_tangent_maps.pkl").open("wb") as handle:
        pickle.dump(
            {
                "session_id": f"{args.subject}_{args.date}",
                "subject": args.subject,
                "date": args.date,
                "source": str(source),
                "sample_mode": args.sample_mode,
                "nuisance_model": str(nuisance_model),
                "axis_projection": str(axis_projection),
                "recorded_nuisance": str(nuisance_model),
                "recorded_axis_projection": str(axis_projection),
                "recorded_shared_mode_projection_k": int(k),
                "shared_mode_basis_source": "global_response_pca",
                "variance_explained_by_projected_modes": float(var_expl),
                "images": per_image,
            },
            handle,
        )
    _write_csv(out_dir / "stg_tangent_map_image_metrics.csv", image_metric_rows)
    _write_csv(out_dir / "stg_tangent_map_alignment.csv", alignment_rows)
    _write_csv(out_dir / "stg_tangent_summary.csv", [summary])
    (out_dir / "stg_tangent_metadata.json").write_text(
        json.dumps(
            {
                "session_id": f"{args.subject}_{args.date}",
                "source": "recorded",
                "drift_only": bool(args.drift_only),
                "recorded_nuisance": str(args.recorded_nuisance),
                "recorded_axis_projection": str(args.recorded_axis_projection),
                "recorded_shared_mode_projection_k": int(k),
                "shared_mode_basis_source": "global_response_pca",
                "n_modes_projected": int(k),
                "variance_explained_by_projected_modes": float(var_expl),
                "drift_parameters": {
                    "eye_smooth_sigma_bins": float(args.drift_eye_smooth_sigma_bins),
                    "speed_percentile": float(args.drift_speed_percentile),
                    "amp_thresh_deg": float(args.drift_amp_thresh_deg),
                    "refrac_ms": float(args.drift_refrac_ms),
                    "exclusion_pre_ms": float(args.drift_exclusion_pre_ms),
                    "exclusion_post_ms": float(args.drift_exclusion_post_ms),
                },
                "drift_support": drift_support,
                "controlled_regression_status": controlled_regression_status,
                "image_similarity_controls": list(control_keys),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = build_parser().parse_args()
    nuisance_model, axis_projection, shared_mode_k = _resolve_source_options(args)
    data = harmonize_fixrsvp_arrays(
        get_fixrsvp_data(
            subject=args.subject,
            date=args.date,
            dataset_configs_path=str(args.dataset_configs_path),
            use_cached_data=bool(args.use_cached_data),
        )
    )
    rates, _ = _source_rates(args, data)
    eyepos = np.asarray(data["eyepos"], dtype=np.float64)
    image_ids = np.asarray(data["image_ids"], dtype=np.int64)
    stim = np.asarray(data["stim"], dtype=np.float64)

    base_valid = np.isfinite(rates).all(axis=2) & np.isfinite(eyepos).all(axis=2) & (image_ids >= 0)
    analysis_valid, drift_support = stg1.build_drift_only_valid_mask(
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

    n_trials = int(image_ids.shape[0])
    n_time = int(image_ids.shape[1])
    time_grid = np.broadcast_to(np.arange(n_time, dtype=np.float64)[None, :], (n_trials, n_time))

    sampled, _ = _sample_images(
        rates,
        eyepos,
        image_ids,
        stim,
        valid_mask=analysis_valid,
        sample_mode=args.sample_mode,
        n_samples_threshold=int(args.n_samples_threshold),
        min_samples=int(args.min_samples),
        seed=int(args.seed),
    )
    global_rate_axis, global_pc1_axis, global_mean_by_time, global_pc1_by_time = _global_axes(rates, analysis_valid, time_grid)

    before_counts = stg1._image_support_counts(base_valid, image_ids)
    after_counts = stg1._image_support_counts(analysis_valid, image_ids)
    min_needed = int(args.n_samples_threshold) if args.sample_mode == "fixed_n" else int(args.min_samples)
    n_images_meeting_sampling_before = int(sum(v >= min_needed for v in before_counts.values()))
    n_images_meeting_sampling_after = int(sum(v >= min_needed for v in after_counts.values()))

    sweep_rows: list[dict[str, object]] = []
    source_dir = Path(args.out_dir) / f"{args.subject}_{args.date}" / f"source_{args.source}"
    source_dir.mkdir(parents=True, exist_ok=True)

    for k in _parse_k_list(shared_mode_k):
        sweep_rows.append(
            _fit_one_projection(
                args=args,
                k=int(k),
                sampled=sampled,
                global_rate_axis=global_rate_axis,
                global_pc1_axis=global_pc1_axis,
                global_mean_by_time=global_mean_by_time,
                global_pc1_by_time=global_pc1_by_time,
                drift_support=drift_support,
                n_images_meeting_sampling_before=n_images_meeting_sampling_before,
                n_images_meeting_sampling_after=n_images_meeting_sampling_after,
                source=str(args.source),
                nuisance_model=str(nuisance_model),
                axis_projection=str(axis_projection),
            )
        )

    session_root = Path(args.out_dir) / f"{args.subject}_{args.date}"
    _write_csv(session_root / "stg_shared_mode_projection_sweep.csv", sweep_rows)
    if str(args.source) == "twin":
        _write_csv(session_root / "stg_twin_symmetric_processing_sweep.csv", sweep_rows)
        print(str(session_root / "stg_twin_symmetric_processing_sweep.csv"))
    else:
        print(str(session_root / "stg_shared_mode_projection_sweep.csv"))


if __name__ == "__main__":
    main()