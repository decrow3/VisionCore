#!/usr/bin/env python3
"""Phase 3 direct model-to-empirical geometry alignment for fixRSVP windows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval.fixrsvp import get_fixrsvp_data
from scripts.jacobian_predictive_framework.run_fixrsvp_empirical_geometry_ceiling import (
    _bundle_alignment_metrics,
    _capture_fraction,
    _matrix_stats,
    _subspace_alignment,
)
from scripts.jacobian_predictive_framework.run_fixrsvp_step2 import (
    DEFAULT_JACOBIAN_STEP_PX,
    REFERENCE_RATE_HZ,
    _choose_baseline,
    _collect_image_windows,
    _compute_jacobian,
    _predict_responses,
    _resolve_pixels_per_degree,
    _stim_stats,
)
from scripts.jacobian_predictive_framework.run_fixrsvp_steps01 import _paired_delta_summary
from scripts.jacobian_predictive_framework.run_fixrsvp_translation_chart import (
    _build_global_basis,
    _load_model_for_session,
    _orthonormalize_basis,
    compute_grid_responses,
)


ROOT_OUTPUT_DIR = Path("outputs/jacobian_predictive_framework")
DEFAULT_OUTPUT_DIR = ROOT_OUTPUT_DIR / "model_empirical_alignment"
DEFAULT_GRID_PX = (-4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0)


def _parse_grid_px(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _union_fieldnames(rows: list[dict]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    return sorted(keys)


def _load_split_bundles(path: str | None) -> dict[str, dict[str, np.ndarray]]:
    if not path:
        return {}
    bundle_path = Path(path)
    if not bundle_path.exists():
        return {}
    with np.load(bundle_path, allow_pickle=False) as data:
        window_ids = data["window_ids"]
        B_a = data["B_a"]
        B_b = data["B_b"]
    result: dict[str, dict[str, np.ndarray]] = {}
    for idx, window_id in enumerate(window_ids):
        if hasattr(window_id, "item"):
            window_id = window_id.item()
        if isinstance(window_id, bytes):
            key = window_id.decode("utf-8")
        else:
            key = str(window_id)
        result[key] = {
            "B_a": np.asarray(B_a[idx], dtype=np.float64),
            "B_b": np.asarray(B_b[idx], dtype=np.float64),
        }
    return result


def _nanmedian(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.nanmedian(np.asarray(values, dtype=np.float64)))


def _safe_ratio(numer: float, denom: float) -> float:
    if not np.isfinite(numer) or not np.isfinite(denom) or abs(denom) < 1e-12:
        return float("nan")
    return float(numer / denom)


def _median_metric_dict(metric_rows: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    return {
        key: _nanmedian([float(row[key]) for row in metric_rows if key in row])
        for key in keys
    }


def _matched_candidate_ids_local(rows: list[dict], target_row: dict, n_matches: int) -> list[str]:
    feature_names = [
        "jacobian_fro_norm",
        "stim_rms",
        "stim_grad_energy",
        "mean_model_rate",
        "eye_amplitude_px2",
    ]
    usable_features = [name for name in feature_names if all(name in row for row in rows) and name in target_row]
    if not usable_features:
        return [
            row["window_id"]
            for row in rows
            if row["window_id"] != target_row["window_id"] and row["image_id"] != target_row["image_id"]
        ][:n_matches]

    feature_matrix = np.array([[row[name] for name in usable_features] for row in rows], dtype=np.float64)
    center = np.nanmedian(feature_matrix, axis=0)
    scale = np.nanmedian(np.abs(feature_matrix - center[None, :]), axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    target_features = np.array([target_row[name] for name in usable_features], dtype=np.float64)

    distances = []
    for row in rows:
        if row["window_id"] == target_row["window_id"]:
            continue
        if row["image_id"] == target_row["image_id"]:
            continue
        row_features = np.array([row[name] for name in usable_features], dtype=np.float64)
        distances.append((float(np.linalg.norm((row_features - target_features) / scale)), row["window_id"]))
    distances.sort(key=lambda item: item[0])
    return [window_id for _distance, window_id in distances[:n_matches]]


def _basis_from_model_fem(
    model,
    baseline_stim: torch.Tensor,
    offsets_px: np.ndarray,
    dataset_idx: int,
    baseline_resp: np.ndarray,
) -> np.ndarray:
    if offsets_px.shape[0] < 4:
        return np.zeros((baseline_resp.shape[0], 2), dtype=np.float64)
    responses = compute_grid_responses(model, baseline_stim, offsets_px, dataset_idx)
    delta = responses - baseline_resp[None, :]
    eye_c = offsets_px - offsets_px.mean(axis=0, keepdims=True)
    if int(np.linalg.matrix_rank(eye_c)) < 1:
        return np.zeros((baseline_resp.shape[0], 2), dtype=np.float64)
    coeffs, _, _, _ = np.linalg.lstsq(eye_c, delta, rcond=None)
    return coeffs.T.astype(np.float64)


def _basis_from_fem_cov_pcs(
    model,
    baseline_stim: torch.Tensor,
    offsets_px: np.ndarray,
    dataset_idx: int,
    baseline_resp: np.ndarray,
) -> np.ndarray:
    if offsets_px.shape[0] < 4:
        return np.zeros((baseline_resp.shape[0], 2), dtype=np.float64)
    responses = compute_grid_responses(model, baseline_stim, offsets_px, dataset_idx)
    delta = responses - baseline_resp[None, :]
    if delta.shape[0] < 2:
        return np.zeros((baseline_resp.shape[0], 2), dtype=np.float64)
    centered = delta - delta.mean(axis=0, keepdims=True)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2].T if vh.shape[0] >= 2 else np.eye(delta.shape[1], 2, dtype=np.float64)
    return np.asarray(basis, dtype=np.float64)


def _basis_from_jacobian(J: np.ndarray) -> np.ndarray:
    return np.asarray(J, dtype=np.float64)


def _basis_summary(
    basis: np.ndarray,
    emp_bundle: dict[str, np.ndarray],
    rank_ratio_threshold: float,
) -> dict:
    B_a = emp_bundle["B_a"]
    B_b = emp_bundle["B_b"]
    if B_a.ndim != 3 or B_b.ndim != 3 or B_a.shape != B_b.shape or B_a.shape[0] == 0:
        return {
            "align_to_emp_2d": float("nan"),
            "align_to_emp_top1": float("nan"),
            "align_to_emp_rankadaptive": float("nan"),
            "capture_emp_2d": float("nan"),
            "capture_emp_top1": float("nan"),
            "capture_emp_rankadaptive": float("nan"),
            "emp_split_alignment_2d": float("nan"),
            "emp_split_alignment_top1": float("nan"),
            "emp_split_alignment_rankadaptive": float("nan"),
            "emp_split_capture_2d": float("nan"),
            "emp_split_capture_top1": float("nan"),
            "emp_split_capture_rankadaptive": float("nan"),
            "vec_corr_BA_BB": float("nan"),
            "min_B_rank_ratio": float("nan"),
            "alignment_fraction_of_ceiling_2d": float("nan"),
            "alignment_fraction_of_ceiling_top1": float("nan"),
            "alignment_fraction_of_ceiling_rankadaptive": float("nan"),
            "delta_vs_split_half_2d": float("nan"),
            "delta_vs_split_half_top1": float("nan"),
            "delta_vs_split_half_rankadaptive": float("nan"),
            "basis_rank_ratio": float("nan"),
        }
    align2_vals = []
    align1_vals = []
    align_rankadaptive_vals = []
    capture2_vals = []
    capture1_vals = []
    capture_rankadaptive_vals = []
    split_metrics = _bundle_alignment_metrics(B_a, B_b)
    min_rank_ratio_vals = []
    for idx in range(B_a.shape[0]):
        align2_a = _subspace_alignment(basis, B_a[idx], n_dim=2)
        align2_b = _subspace_alignment(basis, B_b[idx], n_dim=2)
        align1_a = _subspace_alignment(basis, B_a[idx], n_dim=1)
        align1_b = _subspace_alignment(basis, B_b[idx], n_dim=1)
        capture2_a = _capture_fraction(basis, B_a[idx], n_dim=2)
        capture2_b = _capture_fraction(basis, B_b[idx], n_dim=2)
        capture1_a = _capture_fraction(basis, B_a[idx], n_dim=1)
        capture1_b = _capture_fraction(basis, B_b[idx], n_dim=1)

        align2_vals.append(float(np.nanmean([align2_a, align2_b])))
        align1_vals.append(float(np.nanmean([align1_a, align1_b])))
        capture2_vals.append(float(np.nanmean([capture2_a, capture2_b])))
        capture1_vals.append(float(np.nanmean([capture1_a, capture1_b])))

        min_rank_ratio = float(np.nanmin([split_metrics["B_a_rank_ratio_values"][idx], split_metrics["B_b_rank_ratio_values"][idx]]))
        min_rank_ratio_vals.append(min_rank_ratio)
        if np.isfinite(min_rank_ratio) and min_rank_ratio < rank_ratio_threshold:
            align_rankadaptive_vals.append(align1_vals[-1])
            capture_rankadaptive_vals.append(capture1_vals[-1])
        else:
            align_rankadaptive_vals.append(align2_vals[-1])
            capture_rankadaptive_vals.append(capture2_vals[-1])
    stats = _matrix_stats(basis)
    align2_med = float(np.nanmedian(align2_vals))
    align1_med = float(np.nanmedian(align1_vals))
    align_rankadaptive_med = float(np.nanmedian(align_rankadaptive_vals))
    capture2_med = float(np.nanmedian(capture2_vals))
    capture1_med = float(np.nanmedian(capture1_vals))
    capture_rankadaptive_med = float(np.nanmedian(capture_rankadaptive_vals))
    split2_med = float(np.nanmedian(split_metrics["alignment2_values"]))
    split1_med = float(np.nanmedian(split_metrics["alignment1_values"]))
    split_capture2_med = float(np.nanmedian(split_metrics["capture2_values"]))
    split_capture1_med = float(np.nanmedian(split_metrics["capture1_values"]))
    min_rank_ratio_med = float(np.nanmedian(min_rank_ratio_vals))
    split_align_rankadaptive_vals = [
        float(split_metrics["alignment1_values"][idx])
        if np.isfinite(min_rank_ratio_vals[idx]) and min_rank_ratio_vals[idx] < rank_ratio_threshold
        else float(split_metrics["alignment2_values"][idx])
        for idx in range(len(min_rank_ratio_vals))
    ]
    split_capture_rankadaptive_vals = [
        float(split_metrics["capture1_values"][idx])
        if np.isfinite(min_rank_ratio_vals[idx]) and min_rank_ratio_vals[idx] < rank_ratio_threshold
        else float(split_metrics["capture2_values"][idx])
        for idx in range(len(min_rank_ratio_vals))
    ]
    split_rankadaptive_med = float(np.nanmedian(split_align_rankadaptive_vals))
    split_capture_rankadaptive_med = float(np.nanmedian(split_capture_rankadaptive_vals))
    return {
        "align_to_emp_2d": align2_med,
        "align_to_emp_top1": align1_med,
        "align_to_emp_rankadaptive": align_rankadaptive_med,
        "capture_emp_2d": capture2_med,
        "capture_emp_top1": capture1_med,
        "capture_emp_rankadaptive": capture_rankadaptive_med,
        "emp_split_alignment_2d": split2_med,
        "emp_split_alignment_top1": split1_med,
        "emp_split_alignment_rankadaptive": split_rankadaptive_med,
        "emp_split_capture_2d": split_capture2_med,
        "emp_split_capture_top1": split_capture1_med,
        "emp_split_capture_rankadaptive": split_capture_rankadaptive_med,
        "vec_corr_BA_BB": float(np.nanmedian(split_metrics["vec_corr_values"])),
        "min_B_rank_ratio": min_rank_ratio_med,
        "alignment_fraction_of_ceiling_2d": _safe_ratio(align2_med, split2_med),
        "alignment_fraction_of_ceiling_top1": _safe_ratio(align1_med, split1_med),
        "alignment_fraction_of_ceiling_rankadaptive": _safe_ratio(align_rankadaptive_med, split_rankadaptive_med),
        "delta_vs_split_half_2d": align2_med - split2_med if np.isfinite(split2_med) else float("nan"),
        "delta_vs_split_half_top1": align1_med - split1_med if np.isfinite(split1_med) else float("nan"),
        "delta_vs_split_half_rankadaptive": align_rankadaptive_med - split_rankadaptive_med if np.isfinite(split_rankadaptive_med) else float("nan"),
        "basis_rank_ratio": stats["rank_ratio"],
    }


def run_model_empirical_alignment(
    subject: str,
    date: str,
    dataset_configs_path: str,
    output_dir: Path,
    checkpoint_path: str | None,
    dataset_idx: int,
    split_bundle_path: str,
    grid_px: tuple[float, ...],
    min_samples: int,
    n_shuffle_matches: int,
    n_random_subspaces: int,
    jacobian_step_px: float,
    rank_ratio_threshold: float,
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
    split_bundles = _load_split_bundles(split_bundle_path)
    if not windows or not split_bundles:
        return {"subject": subject, "date": date, "n_windows": 0}

    model, _info = _load_model_for_session(
        dataset_configs_path=dataset_configs_path,
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        model_index=model_index,
        model_device=model_device,
    )
    model_any: Any = model
    model_any.model.eval()
    session_name = f"{subject}_{date}"
    if dataset_idx is not None and hasattr(model_any, "names"):
        try:
            dataset_idx = model_any.names.index(session_name)
        except ValueError:
            pass

    window_state: list[dict] = []

    for win in windows:
        window_id = f"image_{win.image_id}"
        emp_bundle = split_bundles.get(window_id)
        if emp_bundle is None:
            continue
        baseline_stim, baseline_eye_deg = _choose_baseline(win.stim, win.eyepos_deg)
        baseline_resp = _predict_responses(model, baseline_stim, dataset_idx)[0]
        eyepos_px = (win.eyepos_deg - baseline_eye_deg[None, :]) * pixels_per_degree
        sigma_eye = np.cov(eyepos_px.T).astype(np.float64)
        J = _compute_jacobian(model, baseline_stim, dataset_idx, jacobian_step_px)
        if not np.isfinite(J).all() or not np.isfinite(baseline_resp).all():
            continue
        stim_stats = _stim_stats(baseline_stim.squeeze(0).cpu().numpy())
        b_model_basis = _basis_from_model_fem(model, baseline_stim, eyepos_px, dataset_idx, baseline_resp)
        fem_pcs_basis = _basis_from_fem_cov_pcs(model, baseline_stim, eyepos_px, dataset_idx, baseline_resp)
        window_state.append({
            "image_id": int(win.image_id),
            "window_id": window_id,
            "baseline_stim": baseline_stim,
            "baseline_resp": baseline_resp,
            "eyepos_px": eyepos_px,
            "jacobian": J,
            "J_local_basis": _basis_from_jacobian(J),
            "B_model_basis": b_model_basis,
            "FEM_PCs_basis": fem_pcs_basis,
            "jacobian_fro_norm": float(np.linalg.norm(J, "fro")),
            "eye_amplitude_px2": float(np.trace(sigma_eye)),
            "mean_model_rate": float(np.mean(baseline_resp)),
            **stim_stats,
            "emp_bundle": emp_bundle,
        })

    if not window_state:
        return {"subject": subject, "date": date, "n_windows": 0}

    global_basis_map = {
        "B_model": _build_global_basis([row["B_model_basis"] for row in window_state]),
        "J_local": _build_global_basis([row["J_local_basis"] for row in window_state]),
        "FEM_PCs": _build_global_basis([row["FEM_PCs_basis"] for row in window_state]),
    }
    state_by_id = {row["window_id"]: row for row in window_state}
    rng = np.random.default_rng(0)

    rows: list[dict] = []
    paired_by_basis: dict[str, list[dict]] = {"B_model": [], "J_local": [], "FEM_PCs": []}
    summary_keys = [
        "align_to_emp_2d",
        "align_to_emp_top1",
        "align_to_emp_rankadaptive",
        "capture_emp_2d",
        "capture_emp_top1",
        "capture_emp_rankadaptive",
        "emp_split_alignment_2d",
        "emp_split_alignment_top1",
        "emp_split_alignment_rankadaptive",
        "emp_split_capture_2d",
        "emp_split_capture_top1",
        "emp_split_capture_rankadaptive",
        "vec_corr_BA_BB",
        "min_B_rank_ratio",
        "alignment_fraction_of_ceiling_2d",
        "alignment_fraction_of_ceiling_top1",
        "alignment_fraction_of_ceiling_rankadaptive",
        "delta_vs_split_half_2d",
        "delta_vs_split_half_top1",
        "delta_vs_split_half_rankadaptive",
        "basis_rank_ratio",
    ]

    for state in window_state:
        matched_ids = _matched_candidate_ids_local(window_state, state, n_matches=n_shuffle_matches)

        matched_basis_map = {
            "B_model": state["B_model_basis"],
            "J_local": state["J_local_basis"],
            "FEM_PCs": state["FEM_PCs_basis"],
        }
        shuffled_target_eye_basis_map: dict[str, list[np.ndarray]] = {"B_model": [], "J_local": [], "FEM_PCs": []}
        shuffled_ref_eye_basis_map: dict[str, list[np.ndarray]] = {"B_model": [], "J_local": [], "FEM_PCs": []}
        for matched_id in matched_ids:
            ref = state_by_id[matched_id]
            shuffled_target_eye_basis_map["B_model"].append(
                _basis_from_model_fem(model, ref["baseline_stim"], ref["eyepos_px"], dataset_idx, ref["baseline_resp"])
            )
            shuffled_ref_eye_basis_map["B_model"].append(
                _basis_from_model_fem(model, ref["baseline_stim"], ref["eyepos_px"], dataset_idx, ref["baseline_resp"])
            )
            shuffled_target_eye_basis_map["B_model"][-1] = _basis_from_model_fem(
                model,
                ref["baseline_stim"],
                state["eyepos_px"],
                dataset_idx,
                ref["baseline_resp"],
            )
            shuffled_target_eye_basis_map["J_local"].append(ref["J_local_basis"])
            shuffled_ref_eye_basis_map["J_local"].append(ref["J_local_basis"])
            shuffled_ref_eye_basis_map["FEM_PCs"].append(
                _basis_from_fem_cov_pcs(model, ref["baseline_stim"], ref["eyepos_px"], dataset_idx, ref["baseline_resp"])
            )
            shuffled_target_eye_basis_map["FEM_PCs"].append(
                _basis_from_fem_cov_pcs(model, ref["baseline_stim"], state["eyepos_px"], dataset_idx, ref["baseline_resp"])
            )

        n_neurons = int(state["baseline_resp"].shape[0])
        random_summaries = [
            _basis_summary(
                _orthonormalize_basis(rng.standard_normal((n_neurons, 2))),
                state["emp_bundle"],
                rank_ratio_threshold,
            )
            for _ in range(max(1, n_random_subspaces))
        ]
        random_summary = _median_metric_dict(random_summaries, summary_keys)

        for basis_name, basis in matched_basis_map.items():
            matched = _basis_summary(basis, state["emp_bundle"], rank_ratio_threshold)
            shuffled_target_eye_runs = [
                _basis_summary(shuf_basis, state["emp_bundle"], rank_ratio_threshold)
                for shuf_basis in shuffled_target_eye_basis_map[basis_name]
            ]
            shuffled_ref_eye_runs = [
                _basis_summary(shuf_basis, state["emp_bundle"], rank_ratio_threshold)
                for shuf_basis in shuffled_ref_eye_basis_map[basis_name]
            ]
            shuffled = _median_metric_dict(shuffled_target_eye_runs, summary_keys)
            shuffled_ref_eye = _median_metric_dict(shuffled_ref_eye_runs, summary_keys)
            global_summary = _basis_summary(global_basis_map[basis_name], state["emp_bundle"], rank_ratio_threshold)
            row = {
                "image_id": state["image_id"],
                "window_id": state["window_id"],
                "basis_name": basis_name,
                "global_basis_name": f"global_{basis_name}",
                "align_to_emp_2d_matched": matched["align_to_emp_2d"],
                "align_to_emp_2d_shuffled": shuffled["align_to_emp_2d"],
                "align_to_emp_2d_delta": matched["align_to_emp_2d"] - shuffled["align_to_emp_2d"] if np.isfinite(matched["align_to_emp_2d"]) and np.isfinite(shuffled["align_to_emp_2d"]) else float("nan"),
                "align_to_emp_2d_shuffled_ref_eye": shuffled_ref_eye["align_to_emp_2d"],
                "align_to_emp_top1_matched": matched["align_to_emp_top1"],
                "align_to_emp_top1_shuffled": shuffled["align_to_emp_top1"],
                "align_to_emp_top1_delta": matched["align_to_emp_top1"] - shuffled["align_to_emp_top1"] if np.isfinite(matched["align_to_emp_top1"]) and np.isfinite(shuffled["align_to_emp_top1"]) else float("nan"),
                "align_to_emp_top1_shuffled_ref_eye": shuffled_ref_eye["align_to_emp_top1"],
                "align_to_emp_rankadaptive_matched": matched["align_to_emp_rankadaptive"],
                "align_to_emp_rankadaptive_shuffled": shuffled["align_to_emp_rankadaptive"],
                "align_to_emp_rankadaptive_delta": matched["align_to_emp_rankadaptive"] - shuffled["align_to_emp_rankadaptive"] if np.isfinite(matched["align_to_emp_rankadaptive"]) and np.isfinite(shuffled["align_to_emp_rankadaptive"]) else float("nan"),
                "align_to_emp_rankadaptive_shuffled_ref_eye": shuffled_ref_eye["align_to_emp_rankadaptive"],
                "capture_emp_2d_matched": matched["capture_emp_2d"],
                "capture_emp_2d_shuffled": shuffled["capture_emp_2d"],
                "capture_emp_2d_delta": matched["capture_emp_2d"] - shuffled["capture_emp_2d"] if np.isfinite(matched["capture_emp_2d"]) and np.isfinite(shuffled["capture_emp_2d"]) else float("nan"),
                "capture_emp_2d_shuffled_ref_eye": shuffled_ref_eye["capture_emp_2d"],
                "capture_emp_top1_matched": matched["capture_emp_top1"],
                "capture_emp_top1_shuffled": shuffled["capture_emp_top1"],
                "capture_emp_top1_delta": matched["capture_emp_top1"] - shuffled["capture_emp_top1"] if np.isfinite(matched["capture_emp_top1"]) and np.isfinite(shuffled["capture_emp_top1"]) else float("nan"),
                "capture_emp_top1_shuffled_ref_eye": shuffled_ref_eye["capture_emp_top1"],
                "capture_emp_rankadaptive_matched": matched["capture_emp_rankadaptive"],
                "capture_emp_rankadaptive_shuffled": shuffled["capture_emp_rankadaptive"],
                "capture_emp_rankadaptive_delta": matched["capture_emp_rankadaptive"] - shuffled["capture_emp_rankadaptive"] if np.isfinite(matched["capture_emp_rankadaptive"]) and np.isfinite(shuffled["capture_emp_rankadaptive"]) else float("nan"),
                "capture_emp_rankadaptive_shuffled_ref_eye": shuffled_ref_eye["capture_emp_rankadaptive"],
                "align_to_emp_2d_global": global_summary["align_to_emp_2d"],
                "align_to_emp_2d_random": random_summary["align_to_emp_2d"],
                "align_to_emp_top1_global": global_summary["align_to_emp_top1"],
                "align_to_emp_top1_random": random_summary["align_to_emp_top1"],
                "align_to_emp_rankadaptive_global": global_summary["align_to_emp_rankadaptive"],
                "align_to_emp_rankadaptive_random": random_summary["align_to_emp_rankadaptive"],
                "capture_emp_2d_global": global_summary["capture_emp_2d"],
                "capture_emp_2d_random": random_summary["capture_emp_2d"],
                "capture_emp_top1_global": global_summary["capture_emp_top1"],
                "capture_emp_top1_random": random_summary["capture_emp_top1"],
                "capture_emp_rankadaptive_global": global_summary["capture_emp_rankadaptive"],
                "capture_emp_rankadaptive_random": random_summary["capture_emp_rankadaptive"],
                "emp_split_alignment_2d": matched["emp_split_alignment_2d"],
                "emp_split_alignment_top1": matched["emp_split_alignment_top1"],
                "emp_split_alignment_rankadaptive": matched["emp_split_alignment_rankadaptive"],
                "emp_split_capture_2d": matched["emp_split_capture_2d"],
                "emp_split_capture_top1": matched["emp_split_capture_top1"],
                "emp_split_capture_rankadaptive": matched["emp_split_capture_rankadaptive"],
                "vec_corr_BA_BB": matched["vec_corr_BA_BB"],
                "min_B_rank_ratio": matched["min_B_rank_ratio"],
                "alignment_fraction_of_ceiling_2d_matched": matched["alignment_fraction_of_ceiling_2d"],
                "alignment_fraction_of_ceiling_2d_shuffled": shuffled["alignment_fraction_of_ceiling_2d"],
                "alignment_fraction_of_ceiling_top1_matched": matched["alignment_fraction_of_ceiling_top1"],
                "alignment_fraction_of_ceiling_top1_shuffled": shuffled["alignment_fraction_of_ceiling_top1"],
                "alignment_fraction_of_ceiling_rankadaptive_matched": matched["alignment_fraction_of_ceiling_rankadaptive"],
                "alignment_fraction_of_ceiling_rankadaptive_shuffled": shuffled["alignment_fraction_of_ceiling_rankadaptive"],
                "basis_rank_ratio_matched": matched["basis_rank_ratio"],
                "jacobian_fro_norm": state["jacobian_fro_norm"],
                "eye_amplitude_px2": state["eye_amplitude_px2"],
                "mean_model_rate": state["mean_model_rate"],
            }
            rows.append(row)
            paired_by_basis[basis_name].append({
                "matched": row["align_to_emp_2d_matched"],
                "shuffled": row["align_to_emp_2d_shuffled"],
            })

    csv_path = output_dir / "model_empirical_alignment.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_union_fieldnames(rows), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary_by_basis = {}
    for basis_name in sorted({row["basis_name"] for row in rows}):
        subset = [row for row in rows if row["basis_name"] == basis_name]
        summary_by_basis[basis_name] = {
            "n": len(subset),
            "median_align_to_emp_2d_matched": _nanmedian([row["align_to_emp_2d_matched"] for row in subset]),
            "median_align_to_emp_2d_shuffled": _nanmedian([row["align_to_emp_2d_shuffled"] for row in subset]),
            "median_align_to_emp_2d_shuffled_ref_eye": _nanmedian([row["align_to_emp_2d_shuffled_ref_eye"] for row in subset]),
            "median_align_to_emp_2d_delta": _nanmedian([row["align_to_emp_2d_delta"] for row in subset]),
            "median_align_to_emp_top1_matched": _nanmedian([row["align_to_emp_top1_matched"] for row in subset]),
            "median_align_to_emp_top1_shuffled": _nanmedian([row["align_to_emp_top1_shuffled"] for row in subset]),
            "median_align_to_emp_top1_shuffled_ref_eye": _nanmedian([row["align_to_emp_top1_shuffled_ref_eye"] for row in subset]),
            "median_align_to_emp_top1_delta": _nanmedian([row["align_to_emp_top1_delta"] for row in subset]),
            "median_align_to_emp_rankadaptive_matched": _nanmedian([row["align_to_emp_rankadaptive_matched"] for row in subset]),
            "median_align_to_emp_rankadaptive_shuffled": _nanmedian([row["align_to_emp_rankadaptive_shuffled"] for row in subset]),
            "median_align_to_emp_rankadaptive_shuffled_ref_eye": _nanmedian([row["align_to_emp_rankadaptive_shuffled_ref_eye"] for row in subset]),
            "median_align_to_emp_rankadaptive_delta": _nanmedian([row["align_to_emp_rankadaptive_delta"] for row in subset]),
            "median_capture_emp_2d_matched": _nanmedian([row["capture_emp_2d_matched"] for row in subset]),
            "median_capture_emp_2d_shuffled": _nanmedian([row["capture_emp_2d_shuffled"] for row in subset]),
            "median_capture_emp_top1_matched": _nanmedian([row["capture_emp_top1_matched"] for row in subset]),
            "median_capture_emp_top1_shuffled": _nanmedian([row["capture_emp_top1_shuffled"] for row in subset]),
            "median_capture_emp_rankadaptive_matched": _nanmedian([row["capture_emp_rankadaptive_matched"] for row in subset]),
            "median_capture_emp_rankadaptive_shuffled": _nanmedian([row["capture_emp_rankadaptive_shuffled"] for row in subset]),
            "median_emp_split_alignment_2d": _nanmedian([row["emp_split_alignment_2d"] for row in subset]),
            "median_emp_split_alignment_top1": _nanmedian([row["emp_split_alignment_top1"] for row in subset]),
            "median_emp_split_alignment_rankadaptive": _nanmedian([row["emp_split_alignment_rankadaptive"] for row in subset]),
            "median_alignment_fraction_of_ceiling_2d_matched": _nanmedian([row["alignment_fraction_of_ceiling_2d_matched"] for row in subset]),
            "median_alignment_fraction_of_ceiling_2d_shuffled": _nanmedian([row["alignment_fraction_of_ceiling_2d_shuffled"] for row in subset]),
            "median_alignment_fraction_of_ceiling_rankadaptive_matched": _nanmedian([row["alignment_fraction_of_ceiling_rankadaptive_matched"] for row in subset]),
            "median_alignment_fraction_of_ceiling_rankadaptive_shuffled": _nanmedian([row["alignment_fraction_of_ceiling_rankadaptive_shuffled"] for row in subset]),
            "paired_align_to_emp_2d": _paired_delta_summary(paired_by_basis[basis_name], "matched", "shuffled"),
        }

    summary = {
        "subject": subject,
        "date": date,
        "dataset_idx": dataset_idx,
        "n_windows": len(window_state),
        "split_bundle_path": split_bundle_path,
        "summary_by_basis": summary_by_basis,
    }
    summary_path = output_dir / "model_empirical_alignment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--dataset-configs-path", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--dataset-idx", type=int, required=True)
    parser.add_argument("--split-bundle-path", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--grid-px", default=",".join(str(v) for v in DEFAULT_GRID_PX))
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--n-shuffle-matches", type=int, default=8)
    parser.add_argument("--n-random-subspaces", type=int, default=32)
    parser.add_argument("--jacobian-step-px", type=float, default=DEFAULT_JACOBIAN_STEP_PX)
    parser.add_argument("--rank-ratio-threshold", type=float, default=0.1)
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--model-index", type=int, default=None)
    parser.add_argument("--model-device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_model_empirical_alignment(
        subject=args.subject,
        date=args.date,
        dataset_configs_path=args.dataset_configs_path,
        output_dir=Path(args.output_dir),
        checkpoint_path=args.checkpoint_path,
        dataset_idx=args.dataset_idx,
        split_bundle_path=args.split_bundle_path,
        grid_px=_parse_grid_px(args.grid_px),
        min_samples=args.min_samples,
        n_shuffle_matches=args.n_shuffle_matches,
        n_random_subspaces=args.n_random_subspaces,
        jacobian_step_px=args.jacobian_step_px,
        rank_ratio_threshold=args.rank_ratio_threshold,
        model_type=args.model_type,
        model_index=args.model_index,
        model_device=args.model_device,
    )


if __name__ == "__main__":
    main()