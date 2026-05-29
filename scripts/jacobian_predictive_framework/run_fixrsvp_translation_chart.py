#!/usr/bin/env python3
"""Phase 1 translation-chart test for fixRSVP image windows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch

from eval.eval_stack_multidataset import load_model
from eval.fixrsvp import get_fixrsvp_data
from VisionCore.paths import FIGURES_DIR
from scripts.jacobian_predictive_framework.run_fixrsvp_step2 import (
    DEFAULT_JACOBIAN_STEP_PX,
    REFERENCE_RATE_HZ,
    _choose_baseline,
    _collect_image_windows,
    _compute_jacobian,
    _predict_responses,
    _resolve_pixels_per_degree,
    _shift_stimulus_batch,
    _stim_stats,
)
from scripts.jacobian_predictive_framework.run_fixrsvp_steps01 import _paired_delta_summary


ROOT_OUTPUT_DIR = Path("outputs/jacobian_predictive_framework")
DEFAULT_OUTPUT_DIR = ROOT_OUTPUT_DIR / "translation_chart"
DEFAULT_GRID_PX = (-4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0)
DEFAULT_RADIUS_BINS = (1.0, 2.0, 4.0, 8.0)


def _parse_grid_px(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _orthonormalize_basis(basis: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(basis)
    return Q[:, :2]


def _alignment_score(U1: np.ndarray, U2: np.ndarray) -> float:
    Q1, _ = np.linalg.qr(U1)
    Q2, _ = np.linalg.qr(U2)
    singular_values = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(np.mean(np.clip(singular_values, 0.0, 1.0) ** 2))


def _capture_fraction(U: np.ndarray, cov: np.ndarray) -> float:
    Q, _ = np.linalg.qr(U)
    return float(np.trace(Q.T @ cov @ Q) / (np.trace(cov) + 1e-12))


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


def _union_fieldnames(rows: list[dict]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    return sorted(keys)


def _load_step2_metrics(step2_image_windows_csv: str | None) -> tuple[dict[str, dict], dict]:
    if not step2_image_windows_csv:
        return {}, {
            "step2_image_windows_csv": None,
            "n_step2_rows_loaded": 0,
        }
    path = Path(step2_image_windows_csv)
    if not path.exists():
        return {}, {
            "step2_image_windows_csv": str(path),
            "n_step2_rows_loaded": 0,
        }

    joined: dict[str, dict] = {}
    n_loaded = 0
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            window_id = row.get("window_id")
            if not window_id and row.get("image_id") not in (None, ""):
                window_id = f"image_{row['image_id']}"
            if not window_id:
                continue
            n_loaded += 1
            joined[window_id] = {
                "trace_cov_model_fem": float(row.get("trace_cov_model_fem", "nan")),
                "e_fem_cv_median": float(row.get("e_fem_cv_median", "nan")),
                "e_fem_cv_mean": float(row.get("e_fem_cv_mean", "nan")),
            }
    return joined, {
        "step2_image_windows_csv": str(path),
        "n_step2_rows_loaded": int(n_loaded),
    }


def _correlation_summary(rows: list[dict], target_field: str, predictor_fields: list[str]) -> dict:
    summary: dict[str, float] = {}
    target = np.array([row.get(target_field, float("nan")) for row in rows], dtype=np.float64)
    for predictor in predictor_fields:
        x = np.array([row.get(predictor, float("nan")) for row in rows], dtype=np.float64)
        summary[f"spearman_{predictor}_vs_{target_field}"] = _spearman_corr(x, target)
    return summary


def _paired_rows(rows: list[dict], matched_basis: str, shuffled_basis: str, metric_field: str) -> list[dict]:
    matched = {row["window_id"]: row for row in rows if row["chart_basis"] == matched_basis}
    shuffled = {row["window_id"]: row for row in rows if row["chart_basis"] == shuffled_basis}
    paired: list[dict] = []
    for window_id, matched_row in matched.items():
        shuffled_row = shuffled.get(window_id)
        if shuffled_row is None:
            continue
        matched_val = float(matched_row.get(metric_field, float("nan")))
        shuffled_val = float(shuffled_row.get(metric_field, float("nan")))
        paired.append({
            "window_id": window_id,
            "matched": matched_val,
            "shuffled": shuffled_val,
            "delta": matched_val - shuffled_val if np.isfinite(matched_val) and np.isfinite(shuffled_val) else float("nan"),
            "trace_cov_model_fem": float(matched_row.get("trace_cov_model_fem", float("nan"))),
            "e_fem_cv_median": float(matched_row.get("e_fem_cv_median", float("nan"))),
            "e_fem_cv_mean": float(matched_row.get("e_fem_cv_mean", float("nan"))),
        })
    return paired


def _paired_correlation_summary(rows: list[dict], metric_name: str) -> dict:
    if not rows:
        return {}
    return {
        f"spearman_{metric_name}_matched_vs_e_fem_cv_median": _spearman_corr(
            np.array([row["matched"] for row in rows], dtype=np.float64),
            np.array([row["e_fem_cv_median"] for row in rows], dtype=np.float64),
        ),
        f"spearman_{metric_name}_delta_vs_e_fem_cv_median": _spearman_corr(
            np.array([row["delta"] for row in rows], dtype=np.float64),
            np.array([row["e_fem_cv_median"] for row in rows], dtype=np.float64),
        ),
        f"spearman_{metric_name}_delta_vs_trace_cov_model_fem": _spearman_corr(
            np.array([row["delta"] for row in rows], dtype=np.float64),
            np.array([row["trace_cov_model_fem"] for row in rows], dtype=np.float64),
        ),
    }


def generate_translation_grid(grid_px: tuple[float, ...]) -> np.ndarray:
    return np.array([(dx, dy) for dy in grid_px for dx in grid_px], dtype=np.float64)


def compute_grid_responses(
    model,
    baseline_stim: torch.Tensor,
    offsets_px: np.ndarray,
    dataset_idx: int,
) -> np.ndarray:
    stim_batch = baseline_stim.repeat(len(offsets_px), 1, 1, 1, 1)
    shifted = _shift_stimulus_batch(stim_batch, offsets_px)
    return _predict_responses(model, shifted, dataset_idx)


def _pca_basis(delta_r_grid: np.ndarray) -> np.ndarray:
    centered = delta_r_grid - delta_r_grid.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2:
        return np.eye(centered.shape[1], 2, dtype=np.float64)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    return _orthonormalize_basis(vh[:2].T)


def _matched_candidate_ids(rows: list[dict], target_row: dict, n_matches: int) -> list[str]:
    feature_names = [
        "jacobian_fro_norm",
        "stim_rms",
        "stim_grad_energy",
        "mean_model_rate",
        "eye_amplitude_px2",
    ]
    feature_matrix = np.array([[row[name] for name in feature_names] for row in rows], dtype=np.float64)
    center = np.nanmedian(feature_matrix, axis=0)
    scale = np.nanmedian(np.abs(feature_matrix - center[None, :]), axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    target_features = np.array([target_row[name] for name in feature_names], dtype=np.float64)
    distances = []
    for row in rows:
        if row["window_id"] == target_row["window_id"]:
            continue
        if row["image_id"] == target_row["image_id"]:
            continue
        row_features = np.array([row[name] for name in feature_names], dtype=np.float64)
        distances.append((float(np.linalg.norm((row_features - target_features) / scale)), row["window_id"]))
    distances.sort(key=lambda item: item[0])
    return [window_id for _distance, window_id in distances[:n_matches]]


def _fit_inverse_map(z_train: np.ndarray, offsets_train: np.ndarray) -> np.ndarray:
    design = np.column_stack([z_train, np.ones(z_train.shape[0], dtype=np.float64)])
    weights, _, _, _ = np.linalg.lstsq(design, offsets_train, rcond=None)
    return weights


def _predict_inverse_map(z_test: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.column_stack([z_test, np.ones(z_test.shape[0], dtype=np.float64)])
    return design @ weights


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    resid = np.sum((y_true - y_pred) ** 2)
    tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2)
    if tot <= 1e-12:
        return float("nan")
    return float(1.0 - resid / tot)


def coordinate_recovery_metrics(z: np.ndarray, offsets_px: np.ndarray) -> dict:
    signs = np.sign(offsets_px)
    quadrant_masks = [
        (signs[:, 0] > 0) & (signs[:, 1] > 0),
        (signs[:, 0] > 0) & (signs[:, 1] < 0),
        (signs[:, 0] < 0) & (signs[:, 1] > 0),
        (signs[:, 0] < 0) & (signs[:, 1] < 0),
    ]
    pred = np.full_like(offsets_px, np.nan, dtype=np.float64)
    any_test = False
    n_test_points = 0
    for test_mask in quadrant_masks:
        train_mask = ~test_mask
        if int(test_mask.sum()) < 2 or int(train_mask.sum()) < 4:
            continue
        weights = _fit_inverse_map(z[train_mask], offsets_px[train_mask])
        pred[test_mask] = _predict_inverse_map(z[test_mask], weights)
        any_test = True
        n_test_points += int(test_mask.sum())

    cv_mode_used = "quadrant_cv"
    if not any_test:
        weights = _fit_inverse_map(z, offsets_px)
        pred = _predict_inverse_map(z, weights)
        cv_mode_used = "in_sample_fallback"
        n_test_points = int(offsets_px.shape[0])

    mask = np.isfinite(pred).all(axis=1) & np.isfinite(offsets_px).all(axis=1)
    if int(mask.sum()) < 4:
        return {
            "coord_R2_dx": float("nan"),
            "coord_R2_dy": float("nan"),
            "coord_R2_total": float("nan"),
            "angular_error_deg": float("nan"),
            "radial_error_px": float("nan"),
            "cv_mode_used": cv_mode_used,
            "n_test_points": int(n_test_points),
        }

    y_true = offsets_px[mask]
    y_pred = pred[mask]
    dx_r2 = _r2_score(y_true[:, [0]], y_pred[:, [0]])
    dy_r2 = _r2_score(y_true[:, [1]], y_pred[:, [1]])
    total_r2 = _r2_score(y_true, y_pred)
    true_norm = np.linalg.norm(y_true, axis=1)
    pred_norm = np.linalg.norm(y_pred, axis=1)
    nz = (true_norm > 1e-9) & (pred_norm > 1e-9)
    if int(nz.sum()) == 0:
        angular_error = float("nan")
    else:
        cosang = np.sum(y_true[nz] * y_pred[nz], axis=1) / (true_norm[nz] * pred_norm[nz])
        angular_error = float(np.nanmean(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))))
    radial_error = float(np.nanmean(np.abs(pred_norm - true_norm)))
    return {
        "coord_R2_dx": dx_r2,
        "coord_R2_dy": dy_r2,
        "coord_R2_total": total_r2,
        "angular_error_deg": angular_error,
        "radial_error_px": radial_error,
        "cv_mode_used": cv_mode_used,
        "n_test_points": int(n_test_points),
    }

def _coordinate_map_from_grid(z: np.ndarray, offsets_px: np.ndarray) -> np.ndarray:
    return _fit_inverse_map(z, offsets_px)

def fem_sampling_metrics(
    responses_fem: np.ndarray,
    offsets_fem: np.ndarray,
    basis: np.ndarray,
    coord_weights: np.ndarray,
    baseline_resp: np.ndarray,
    grid_offsets: np.ndarray,
) -> dict:
    delta_fem = responses_fem - baseline_resp[None, :]
    z_fem = delta_fem @ basis
    pred_offsets = _predict_inverse_map(z_fem, coord_weights)
    cov_fem = np.cov(delta_fem, rowvar=False).astype(np.float64) if delta_fem.shape[0] >= 2 else np.zeros((delta_fem.shape[1], delta_fem.shape[1]), dtype=np.float64)
    metrics = coordinate_recovery_metrics(z_fem, offsets_fem)
    true_norm = np.linalg.norm(offsets_fem, axis=1)
    pred_norm = np.linalg.norm(pred_offsets, axis=1)
    nz = (true_norm > 1e-9) & (pred_norm > 1e-9)
    if int(nz.sum()) == 0:
        angular_error = float("nan")
    else:
        cosang = np.sum(offsets_fem[nz] * pred_offsets[nz], axis=1) / (true_norm[nz] * pred_norm[nz])
        angular_error = float(np.nanmean(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))))
    grid_box_limit = float(np.max(np.abs(grid_offsets))) if grid_offsets.size else 0.0
    grid_radius_limit = float(np.max(np.linalg.norm(grid_offsets, axis=1))) if grid_offsets.size else 0.0
    within_box = (np.abs(offsets_fem[:, 0]) <= grid_box_limit) & (np.abs(offsets_fem[:, 1]) <= grid_box_limit)
    within_radius = np.linalg.norm(offsets_fem, axis=1) <= grid_radius_limit
    in_grid_metrics = coordinate_recovery_metrics(z_fem[within_box], offsets_fem[within_box]) if int(within_box.sum()) >= 4 else {
        "coord_R2_total": float("nan"),
        "cv_mode_used": "insufficient_points",
        "n_test_points": int(within_box.sum()),
    }
    return {
        "fem_chart_capture": _capture_fraction(basis, cov_fem),
        "fem_coord_R2_dx": _r2_score(offsets_fem[:, [0]], pred_offsets[:, [0]]),
        "fem_coord_R2_dy": _r2_score(offsets_fem[:, [1]], pred_offsets[:, [1]]),
        "fem_coord_R2_total": _r2_score(offsets_fem, pred_offsets),
        "fem_coord_R2_total_all_offsets": _r2_score(offsets_fem, pred_offsets),
        "fem_coord_R2_total_in_grid_only": float(in_grid_metrics.get("coord_R2_total", float("nan"))),
        "fem_coord_angular_error": angular_error,
        "fem_coord_radial_error": float(np.nanmean(np.abs(pred_norm - true_norm))),
        "fem_trace_cov": float(np.trace(cov_fem)),
        "n_fem_samples": int(offsets_fem.shape[0]),
        "fraction_fem_offsets_within_grid_box": float(np.mean(within_box)) if within_box.size else float("nan"),
        "fraction_fem_offsets_within_grid_radius": float(np.mean(within_radius)) if within_radius.size else float("nan"),
        "fem_coord_cv_mode_used_in_grid_only": in_grid_metrics.get("cv_mode_used", "insufficient_points"),
        "fem_coord_n_test_points_in_grid_only": int(in_grid_metrics.get("n_test_points", 0)),
        **metrics,
    }


def chart_smoothness_metrics(z: np.ndarray, offsets_px: np.ndarray) -> dict:
    retinal_d = np.linalg.norm(offsets_px[:, None, :] - offsets_px[None, :, :], axis=-1)
    chart_d = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    tri = np.triu_indices_from(retinal_d, k=1)
    rd = retinal_d[tri]
    cd = chart_d[tri]
    if rd.size < 4:
        return {"smoothness_rho": float("nan")}
    rd_rank = np.argsort(np.argsort(rd)).astype(np.float64)
    cd_rank = np.argsort(np.argsort(cd)).astype(np.float64)
    if np.std(rd_rank) <= 1e-12 or np.std(cd_rank) <= 1e-12:
        return {"smoothness_rho": float("nan")}
    return {"smoothness_rho": float(np.corrcoef(rd_rank, cd_rank)[0, 1])}


def _radius_bin_label(radius: float, radius_bins: tuple[float, ...]) -> str:
    if radius <= radius_bins[0]:
        return f"<= {radius_bins[0]:g}"
    for lo, hi in zip(radius_bins[:-1], radius_bins[1:]):
        if lo < radius <= hi:
            return f"{lo:g}-{hi:g}"
    return f"> {radius_bins[-1]:g}"


def _distance_breakdown(
    z: np.ndarray,
    offsets_px: np.ndarray,
    radius_bins: tuple[float, ...],
) -> list[dict]:
    radii = np.linalg.norm(offsets_px, axis=1)
    rows = []
    for label in sorted({_radius_bin_label(radius, radius_bins) for radius in radii}):
        if label.startswith("<="):
            hi = radius_bins[0]
            mask = radii <= hi
        elif label.startswith(">"):
            lo = radius_bins[-1]
            mask = radii > lo
        else:
            lo_str, hi_str = label.split("-")
            lo = float(lo_str)
            hi = float(hi_str)
            mask = (radii > lo) & (radii <= hi)
        if int(mask.sum()) < 4:
            continue
        rows.append({"radius_bin": label, **coordinate_recovery_metrics(z[mask], offsets_px[mask])})
    return rows


def _build_global_basis(jacobians: list[np.ndarray]) -> np.ndarray:
    finite_jacobians = [jac for jac in jacobians if np.isfinite(jac).all()]
    if not finite_jacobians:
        n_neurons = jacobians[0].shape[0]
        return _orthonormalize_basis(np.eye(n_neurons, 2, dtype=np.float64))
    stack = np.concatenate(finite_jacobians, axis=1)
    stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
    U, _s, _vh = np.linalg.svd(stack, full_matrices=False)
    return _orthonormalize_basis(U[:, :2])


def _load_model_for_session(
    dataset_configs_path: str,
    checkpoint_path: str | None,
    model_type: str | None,
    model_index: int | None,
    model_device: str,
) -> tuple[object, dict | None]:
    model_config_dict = None
    try:
        from models.config_loader import load_config
        model_config_dict = load_config("experiments/model_configs/learned_resnet_none_convgru_gaussian.yaml")
    except Exception:
        pass
    kwargs = {
        "model_type": model_type,
        "model_index": model_index,
        "cfg_dir_override": dataset_configs_path,
        "model_config_dict": model_config_dict,
        "device": model_device,
    }
    if checkpoint_path is not None:
        kwargs["checkpoint_path"] = checkpoint_path
    return load_model(**kwargs)


def run_translation_chart(
    subject: str,
    date: str,
    dataset_configs_path: str,
    output_dir: Path,
    checkpoint_path: str | None,
    dataset_idx: int,
    grid_px: tuple[float, ...],
    min_samples: int,
    n_shuffle_matches: int,
    n_random_subspaces: int,
    jacobian_step_px: float,
    run_fem_sampling: bool,
    step2_image_windows_csv: str | None,
    model_type: str | None,
    model_index: int | None,
    model_device: str,
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

    model, _info = _load_model_for_session(
        dataset_configs_path=dataset_configs_path,
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        model_index=model_index,
        model_device=model_device,
    )
    model.model.eval()
    session_name = f"{subject}_{date}"
    if dataset_idx is not None and hasattr(model, "names"):
        try:
            auto_idx = model.names.index(session_name)
            dataset_idx = auto_idx
        except ValueError:
            pass

    grid_offsets = generate_translation_grid(grid_px)
    rng = np.random.default_rng(0)
    step2_metrics_by_window, step2_join_info = _load_step2_metrics(step2_image_windows_csv)
    window_state: list[dict] = []
    jacobians: list[np.ndarray] = []

    for win in windows:
        baseline_stim, baseline_eye_deg = _choose_baseline(win.stim, win.eyepos_deg)
        baseline_resp = _predict_responses(model, baseline_stim, dataset_idx)[0]
        eyepos_px = (win.eyepos_deg - baseline_eye_deg[None, :]) * pixels_per_degree
        sigma_eye = np.cov(eyepos_px.T).astype(np.float64)
        J = _compute_jacobian(model, baseline_stim, dataset_idx, jacobian_step_px)
        if not np.isfinite(J).all() or not np.isfinite(baseline_resp).all():
            continue
        jacobians.append(J)
        stim_stats = _stim_stats(baseline_stim.squeeze(0).cpu().numpy())
        window_state.append({
            "image_id": int(win.image_id),
            "window_id": f"image_{win.image_id}",
            "baseline_stim": baseline_stim,
            "baseline_resp": baseline_resp,
            "eyepos_px": eyepos_px,
            "jacobian": J,
            "mean_model_rate": float(np.mean(baseline_resp)),
            "eye_amplitude_px2": float(np.trace(sigma_eye)),
            "jacobian_fro_norm": float(np.linalg.norm(J, "fro")),
            **stim_stats,
        })

    global_basis = _build_global_basis(jacobians)
    image_rows: list[dict] = []
    radius_rows: list[dict] = []
    fem_rows: list[dict] = []
    example_payloads: list[dict] = []
    state_by_id = {row["window_id"]: row for row in window_state}
    n_windows_joined = sum(1 for row in window_state if row["window_id"] in step2_metrics_by_window)

    for state in window_state:
        baseline_stim = state["baseline_stim"]
        baseline_resp = state["baseline_resp"]
        grid_resp = compute_grid_responses(model, baseline_stim, grid_offsets, dataset_idx)
        delta_r = grid_resp - baseline_resp[None, :]
        cov_grid = np.cov(delta_r, rowvar=False).astype(np.float64)

        matched_ids = _matched_candidate_ids(window_state, state, n_matches=n_shuffle_matches)
        shuffled_bases = [_orthonormalize_basis(state_by_id[mid]["jacobian"]) for mid in matched_ids]
        pca_basis = _pca_basis(delta_r)
        jac_basis = _orthonormalize_basis(state["jacobian"])

        random_bases = []
        for _ in range(n_random_subspaces):
            random_bases.append(_orthonormalize_basis(rng.standard_normal((delta_r.shape[1], 2))))

        basis_map = {
            "jacobian": jac_basis,
            "pca": pca_basis,
            "global": global_basis,
        }
        grid_coord_maps: dict[str, np.ndarray] = {}

        def _basis_summary(name: str, basis: np.ndarray) -> tuple[dict, list[dict]]:
            z = delta_r @ basis
            grid_coord_maps[name] = _coordinate_map_from_grid(z, grid_offsets)
            metrics = coordinate_recovery_metrics(z, grid_offsets)
            smooth = chart_smoothness_metrics(z, grid_offsets)
            summary = {
                "image_id": state["image_id"],
                "window_id": state["window_id"],
                "chart_basis": name,
                "n_grid_points": int(grid_offsets.shape[0]),
                "baseline_model_rate": state["mean_model_rate"],
                "stim_rms": state["stim_rms"],
                "stim_grad_energy": state["stim_grad_energy"],
                "jacobian_fro_norm": state["jacobian_fro_norm"],
                "eye_amplitude_px2": state["eye_amplitude_px2"],
                "variance_capture": _capture_fraction(basis, cov_grid),
                **metrics,
                **smooth,
                **step2_metrics_by_window.get(state["window_id"], {}),
            }
            per_radius = []
            for row in _distance_breakdown(z, grid_offsets, DEFAULT_RADIUS_BINS):
                per_radius.append({
                    "image_id": state["image_id"],
                    "window_id": state["window_id"],
                    "chart_basis": name,
                    **row,
                })
            return summary, per_radius

        for basis_name, basis in basis_map.items():
            summary_row, per_radius = _basis_summary(basis_name, basis)
            image_rows.append(summary_row)
            radius_rows.extend(per_radius)
            if basis_name == "jacobian" and len(example_payloads) < 4:
                example_payloads.append({
                    "window_id": state["window_id"],
                    "image_id": state["image_id"],
                    "z": delta_r @ basis,
                    "offsets_px": grid_offsets.copy(),
                })

        if shuffled_bases:
            shuffled_metrics = []
            shuffled_radius = []
            for basis in shuffled_bases:
                summary_row, per_radius = _basis_summary("shuffled", basis)
                shuffled_metrics.append(summary_row)
                shuffled_radius.extend(per_radius)
            image_rows.append({
                **shuffled_metrics[0],
                "coord_R2_dx": float(np.nanmedian([row["coord_R2_dx"] for row in shuffled_metrics])),
                "coord_R2_dy": float(np.nanmedian([row["coord_R2_dy"] for row in shuffled_metrics])),
                "coord_R2_total": float(np.nanmedian([row["coord_R2_total"] for row in shuffled_metrics])),
                "angular_error_deg": float(np.nanmedian([row["angular_error_deg"] for row in shuffled_metrics])),
                "radial_error_px": float(np.nanmedian([row["radial_error_px"] for row in shuffled_metrics])),
                "smoothness_rho": float(np.nanmedian([row["smoothness_rho"] for row in shuffled_metrics])),
                "variance_capture": float(np.nanmedian([row["variance_capture"] for row in shuffled_metrics])),
            })
            radius_rows.extend(shuffled_radius)

        if random_bases:
            random_metrics = []
            for basis in random_bases:
                summary_row, _per_radius = _basis_summary("random", basis)
                random_metrics.append(summary_row)
            image_rows.append({
                **random_metrics[0],
                "coord_R2_dx": float(np.nanmedian([row["coord_R2_dx"] for row in random_metrics])),
                "coord_R2_dy": float(np.nanmedian([row["coord_R2_dy"] for row in random_metrics])),
                "coord_R2_total": float(np.nanmedian([row["coord_R2_total"] for row in random_metrics])),
                "angular_error_deg": float(np.nanmedian([row["angular_error_deg"] for row in random_metrics])),
                "radial_error_px": float(np.nanmedian([row["radial_error_px"] for row in random_metrics])),
                "smoothness_rho": float(np.nanmedian([row["smoothness_rho"] for row in random_metrics])),
                "variance_capture": float(np.nanmedian([row["variance_capture"] for row in random_metrics])),
            })

        if run_fem_sampling:
            fem_offsets = state["eyepos_px"]
            if fem_offsets.shape[0] >= 4:
                fem_resp = compute_grid_responses(model, baseline_stim, fem_offsets, dataset_idx)

                def _append_fem_row(name: str, basis: np.ndarray, coord_weights: np.ndarray) -> dict:
                    metrics = fem_sampling_metrics(
                        responses_fem=fem_resp,
                        offsets_fem=fem_offsets,
                        basis=basis,
                        coord_weights=coord_weights,
                        baseline_resp=baseline_resp,
                        grid_offsets=grid_offsets,
                    )
                    row = {
                        "image_id": state["image_id"],
                        "window_id": state["window_id"],
                        "chart_basis": name,
                        "baseline_model_rate": state["mean_model_rate"],
                        "stim_rms": state["stim_rms"],
                        "stim_grad_energy": state["stim_grad_energy"],
                        "jacobian_fro_norm": state["jacobian_fro_norm"],
                        "eye_amplitude_px2": state["eye_amplitude_px2"],
                        **metrics,
                        **step2_metrics_by_window.get(state["window_id"], {}),
                    }
                    fem_rows.append(row)
                    return row

                _append_fem_row("jacobian", jac_basis, grid_coord_maps["jacobian"])
                _append_fem_row("pca", pca_basis, grid_coord_maps["pca"])
                _append_fem_row("global", global_basis, grid_coord_maps["global"])

                if shuffled_bases:
                    shuffled_fem = []
                    for idx, basis in enumerate(shuffled_bases):
                        coord_w = _coordinate_map_from_grid(delta_r @ basis, grid_offsets)
                        shuffled_fem.append(_append_fem_row("shuffled_single", basis, coord_w))
                    fem_rows.append({
                        **shuffled_fem[0],
                        "chart_basis": "shuffled",
                        "fem_chart_capture": float(np.nanmedian([row["fem_chart_capture"] for row in shuffled_fem])),
                        "fem_coord_R2_dx": float(np.nanmedian([row["fem_coord_R2_dx"] for row in shuffled_fem])),
                        "fem_coord_R2_dy": float(np.nanmedian([row["fem_coord_R2_dy"] for row in shuffled_fem])),
                        "fem_coord_R2_total": float(np.nanmedian([row["fem_coord_R2_total"] for row in shuffled_fem])),
                        "fem_coord_angular_error": float(np.nanmedian([row["fem_coord_angular_error"] for row in shuffled_fem])),
                        "fem_coord_radial_error": float(np.nanmedian([row["fem_coord_radial_error"] for row in shuffled_fem])),
                    })

                if random_bases:
                    random_fem = []
                    for basis in random_bases:
                        coord_w = _coordinate_map_from_grid(delta_r @ basis, grid_offsets)
                        random_fem.append(_append_fem_row("random_single", basis, coord_w))
                    fem_rows.append({
                        **random_fem[0],
                        "chart_basis": "random",
                        "fem_chart_capture": float(np.nanmedian([row["fem_chart_capture"] for row in random_fem])),
                        "fem_coord_R2_dx": float(np.nanmedian([row["fem_coord_R2_dx"] for row in random_fem])),
                        "fem_coord_R2_dy": float(np.nanmedian([row["fem_coord_R2_dy"] for row in random_fem])),
                        "fem_coord_R2_total": float(np.nanmedian([row["fem_coord_R2_total"] for row in random_fem])),
                        "fem_coord_angular_error": float(np.nanmedian([row["fem_coord_angular_error"] for row in random_fem])),
                        "fem_coord_radial_error": float(np.nanmedian([row["fem_coord_radial_error"] for row in random_fem])),
                    })

    csv_path = output_dir / "translation_chart_image_windows.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_union_fieldnames(image_rows), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(image_rows)

    radius_csv = output_dir / "translation_chart_pairwise_grid.csv"
    if radius_rows:
        with radius_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_union_fieldnames(radius_rows), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(radius_rows)

    fem_csv = output_dir / "fem_chart_sampling.csv"
    if fem_rows:
        filtered_fem_rows = [row for row in fem_rows if not row["chart_basis"].endswith("_single")]
        with fem_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_union_fieldnames(filtered_fem_rows), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(filtered_fem_rows)

    summary_by_basis = {}
    for basis_name in sorted(set(row["chart_basis"] for row in image_rows)):
        subset = [row for row in image_rows if row["chart_basis"] == basis_name]
        summary_by_basis[basis_name] = {
            "n": len(subset),
            "median_coord_R2_total": float(np.nanmedian([row["coord_R2_total"] for row in subset])),
            "median_angular_error_deg": float(np.nanmedian([row["angular_error_deg"] for row in subset])),
            "median_variance_capture": float(np.nanmedian([row["variance_capture"] for row in subset])),
        }

    jac_rows = [row for row in image_rows if row["chart_basis"] == "jacobian"]
    shuf_rows = [row for row in image_rows if row["chart_basis"] == "shuffled"]
    paired = []
    shuf_by_window = {row["window_id"]: row for row in shuf_rows}
    for row in jac_rows:
        other = shuf_by_window.get(row["window_id"])
        if other is None:
            continue
        paired.append({
            "coord_R2_total_matched": row["coord_R2_total"],
            "coord_R2_total_shuffled": other["coord_R2_total"],
        })
    paired_summary = _paired_delta_summary(
        [{"matched": row["coord_R2_total_matched"], "shuffled": row["coord_R2_total_shuffled"]} for row in paired],
        "matched",
        "shuffled",
    ) if paired else {}

    summary = {
        "subject": subject,
        "date": date,
        "dataset_idx": dataset_idx,
        "n_windows": len(windows),
        "grid_px": list(grid_px),
        "summary_by_basis": summary_by_basis,
        "paired_matched_vs_shuffled_coord_R2_total": paired_summary,
        "step2_join_diagnostics": {
            **step2_join_info,
            "n_chart_windows": int(len(window_state)),
            "n_windows_joined_to_step2": int(n_windows_joined),
            "fraction_joined_to_step2": float(n_windows_joined / len(window_state)) if window_state else float("nan"),
            "n_joined_missing_trace_cov_model_fem": int(sum(1 for row in jac_rows if not np.isfinite(row.get("trace_cov_model_fem", float("nan"))))),
            "n_joined_missing_e_fem_cv_median": int(sum(1 for row in jac_rows if not np.isfinite(row.get("e_fem_cv_median", float("nan"))))),
        },
    }

    jac_metric_rows = [row for row in image_rows if row["chart_basis"] == "jacobian"]
    if jac_metric_rows:
        summary["jacobian_chart_metric_correlations"] = _correlation_summary(
            jac_metric_rows,
            target_field="coord_R2_total",
            predictor_fields=[
                "stim_grad_energy",
                "jacobian_fro_norm",
                "trace_cov_model_fem",
                "e_fem_cv_median",
            ],
        )
    paired_chart_rows = _paired_rows(image_rows, matched_basis="jacobian", shuffled_basis="shuffled", metric_field="coord_R2_total")
    if paired_chart_rows:
        summary["paired_jacobian_vs_shuffled_chart_correlations"] = _paired_correlation_summary(
            paired_chart_rows,
            metric_name="chart_coord_R2",
        )

    if fem_rows:
        filtered_fem_rows = [row for row in fem_rows if not row["chart_basis"].endswith("_single")]
        fem_summary_by_basis = {}
        for basis_name in sorted(set(row["chart_basis"] for row in filtered_fem_rows)):
            subset = [row for row in filtered_fem_rows if row["chart_basis"] == basis_name]
            fem_summary_by_basis[basis_name] = {
                "n": len(subset),
                "median_fem_chart_capture": float(np.nanmedian([row["fem_chart_capture"] for row in subset])),
                "median_fem_coord_R2_total": float(np.nanmedian([row["fem_coord_R2_total"] for row in subset])),
                "median_fem_coord_angular_error": float(np.nanmedian([row["fem_coord_angular_error"] for row in subset])),
            }
        jac_fem_rows = [row for row in filtered_fem_rows if row["chart_basis"] == "jacobian"]
        shuf_fem_rows = [row for row in filtered_fem_rows if row["chart_basis"] == "shuffled"]
        shuf_fem_by_window = {row["window_id"]: row for row in shuf_fem_rows}
        fem_paired = []
        for row in jac_fem_rows:
            other = shuf_fem_by_window.get(row["window_id"])
            if other is None:
                continue
            fem_paired.append({"matched": row["fem_chart_capture"], "shuffled": other["fem_chart_capture"]})
        summary["fem_sampling_summary_by_basis"] = fem_summary_by_basis
        summary["paired_matched_vs_shuffled_fem_capture"] = _paired_delta_summary(fem_paired, "matched", "shuffled") if fem_paired else {}
        jac_fem_metric_rows = [row for row in filtered_fem_rows if row["chart_basis"] == "jacobian"]
        if jac_fem_metric_rows:
            summary["jacobian_fem_metric_correlations"] = _correlation_summary(
                jac_fem_metric_rows,
                target_field="fem_chart_capture",
                predictor_fields=[
                    "stim_grad_energy",
                    "jacobian_fro_norm",
                    "trace_cov_model_fem",
                    "e_fem_cv_median",
                ],
            )
        paired_fem_rows = _paired_rows(filtered_fem_rows, matched_basis="jacobian", shuffled_basis="shuffled", metric_field="fem_chart_capture")
        if paired_fem_rows:
            summary["paired_jacobian_vs_shuffled_fem_correlations"] = _paired_correlation_summary(
                paired_fem_rows,
                metric_name="fem_chart_capture",
            )
        (output_dir / "fem_chart_sampling_summary.json").write_text(json.dumps({
            "subject": subject,
            "date": date,
            "dataset_idx": dataset_idx,
            "n_windows": len(filtered_fem_rows),
            "summary_by_basis": fem_summary_by_basis,
            "paired_matched_vs_shuffled_fem_capture": summary["paired_matched_vs_shuffled_fem_capture"],
            "jacobian_fem_metric_correlations": summary.get("jacobian_fem_metric_correlations", {}),
            "paired_jacobian_vs_shuffled_fem_correlations": summary.get("paired_jacobian_vs_shuffled_fem_correlations", {}),
        }, indent=2) + "\n")
    (output_dir / "translation_chart_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_summary_figure(image_rows, output_dir, date)
    if radius_rows:
        _write_distance_breakdown_figure(radius_rows, output_dir, date)
    if example_payloads:
        _write_chart_examples(example_payloads, output_dir, date)
    if fem_rows:
        _write_fem_summary_figure([row for row in fem_rows if not row["chart_basis"].endswith("_single")], output_dir, date)
    return summary


def _write_summary_figure(rows: list[dict], output_dir: Path, date: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    basis_order = ["jacobian", "shuffled", "random", "global", "pca"]
    values = [
        np.array([row["coord_R2_total"] for row in rows if row["chart_basis"] == basis], dtype=np.float64)
        for basis in basis_order
    ]
    values = [val for val in values if val.size]
    labels = [basis for basis in basis_order if any(row["chart_basis"] == basis for row in rows)]
    if not values:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(values, tick_labels=labels)
    ax.set_ylabel("Coordinate recovery $R^2_{total}$")
    ax.set_title(f"fixRSVP translation chart summary | {date}")
    fig.tight_layout()

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "chart_coordinate_recovery_summary.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    canonical = FIGURES_DIR / "jacobian_predictive_framework"
    canonical.mkdir(parents=True, exist_ok=True)
    shutil.copy(fig_path, canonical / f"chart_coordinate_recovery_summary_{date}.png")


def _write_fem_summary_figure(rows: list[dict], output_dir: Path, date: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    basis_order = ["jacobian", "shuffled", "random", "global", "pca"]
    values = [
        np.array([row["fem_chart_capture"] for row in rows if row["chart_basis"] == basis], dtype=np.float64)
        for basis in basis_order
    ]
    values = [val for val in values if val.size]
    labels = [basis for basis in basis_order if any(row["chart_basis"] == basis for row in rows)]
    if not values:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(values, tick_labels=labels)
    ax.set_ylabel("FEM chart capture")
    ax.set_title(f"fixRSVP FEM chart sampling | {date}")
    fig.tight_layout()

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "fem_chart_capture_summary.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    canonical = FIGURES_DIR / "jacobian_predictive_framework"
    canonical.mkdir(parents=True, exist_ok=True)
    shutil.copy(fig_path, canonical / f"fem_chart_capture_summary_{date}.png")


def _write_distance_breakdown_figure(rows: list[dict], output_dir: Path, date: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    basis_order = ["jacobian", "shuffled", "random", "global", "pca"]
    radius_bins = sorted(set(row["radius_bin"] for row in rows))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(radius_bins), dtype=np.float64)
    width = 0.14
    for idx, basis in enumerate(basis_order):
        medians = []
        for radius_bin in radius_bins:
            subset = [row["coord_R2_total"] for row in rows if row["chart_basis"] == basis and row["radius_bin"] == radius_bin]
            medians.append(float(np.nanmedian(subset)) if subset else float("nan"))
        if any(np.isfinite(medians)):
            ax.plot(x + (idx - 2) * width, medians, marker="o", linewidth=1.5, label=basis)
    ax.set_xticks(x)
    ax.set_xticklabels(radius_bins)
    ax.set_ylabel("Coordinate recovery $R^2_{total}$")
    ax.set_xlabel("Radius bin (px)")
    ax.set_title(f"Translation chart distance breakdown | {date}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "chart_distance_breakdown.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    canonical = FIGURES_DIR / "jacobian_predictive_framework"
    canonical.mkdir(parents=True, exist_ok=True)
    shutil.copy(fig_path, canonical / f"chart_distance_breakdown_{date}.png")


def _write_chart_examples(example_payloads: list[dict], output_dir: Path, date: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n = len(example_payloads)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False)
    for ax, payload in zip(axes[0], example_payloads):
        z = payload["z"]
        offsets = payload["offsets_px"]
        sc = ax.scatter(z[:, 0], z[:, 1], c=offsets[:, 0], cmap="coolwarm", s=35)
        for idx in range(z.shape[0]):
            ax.text(z[idx, 0], z[idx, 1], f"{offsets[idx, 0]:g},{offsets[idx, 1]:g}", fontsize=5)
        ax.set_title(f"img {payload['image_id']}")
        ax.set_xlabel("chart dim 1")
        ax.set_ylabel("chart dim 2")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="dx")
    fig.suptitle(f"Example Jacobian translation charts | {date}", fontsize=11)
    fig.tight_layout()

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "chart_example_images.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    canonical = FIGURES_DIR / "jacobian_predictive_framework"
    canonical.mkdir(parents=True, exist_ok=True)
    shutil.copy(fig_path, canonical / f"chart_example_images_{date}.png")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--dataset-configs-path", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--dataset-idx", type=int, required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--grid-px", default=",".join(str(x) for x in DEFAULT_GRID_PX))
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--n-shuffle-matches", type=int, default=10)
    parser.add_argument("--n-random-subspaces", type=int, default=100)
    parser.add_argument("--jacobian-step-px", type=float, default=DEFAULT_JACOBIAN_STEP_PX)
    parser.add_argument("--run-fem-sampling", action="store_true")
    parser.add_argument("--step2-image-windows-csv", default=None)
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--model-index", type=int, default=None)
    parser.add_argument("--model-device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_translation_chart(
        subject=args.subject,
        date=args.date,
        dataset_configs_path=args.dataset_configs_path,
        output_dir=Path(args.output_dir),
        checkpoint_path=args.checkpoint_path,
        dataset_idx=args.dataset_idx,
        grid_px=_parse_grid_px(args.grid_px),
        min_samples=args.min_samples,
        n_shuffle_matches=args.n_shuffle_matches,
        n_random_subspaces=args.n_random_subspaces,
        jacobian_step_px=args.jacobian_step_px,
        run_fem_sampling=args.run_fem_sampling,
        step2_image_windows_csv=args.step2_image_windows_csv,
        model_type=args.model_type,
        model_index=args.model_index,
        model_device=args.model_device,
    )
    print(json.dumps(result, indent=2))