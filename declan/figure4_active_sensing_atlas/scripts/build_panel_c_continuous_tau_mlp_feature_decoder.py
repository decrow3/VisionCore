"""Hybrid continuous-tau / nonlinear feature decoder for Figure 4C.

This diagnostic combines the two recent 4C directions:

1. the promoted continuous-eye-trace observer, which infers a continuous
   ``tau_hat`` path rather than selecting a trajectory candidate, and
2. the Tejas-style MLP feature decoder, which maps compact response features
   directly to a continuous image-feature embedding.

The endpoint is not image-candidate choice.  Each supervised row is a response
table with a true source image, and the target is a compact PCA embedding of
``phi(image)``.  The main hybrid input is:

    compact observed response movie + continuous tau_hat + response-by-tau terms

Cross-fitting is by source image.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.backimage_trajectory_observer.analyze_continuous_joint_trajectory import (
    _load_basis as _load_continuous_basis,
    _observed_trajectory_from_table_or_npz,
    _scale_value,
    _trajectory_from_table_or_npz,
    score_continuous_joint_score_vectors,
)
from declan.figure4_active_sensing_atlas.scripts.build_panel_c_continuous_feature_embedding_reconstruction import (
    COMPACT_BASIS,
    FEATURE_NPZ,
    MLPConfig,
    _assign_source_folds,
    _bootstrap_mean,
    _fit_feature_transform,
    _fit_predict_mlp,
    _json_ready,
    _load_basis,
    _load_feature_table,
    _load_npz,
    _metrics,
    _parse_scales,
    _parse_str_list,
    _source_row_from_candidate_id,
    _source_group_validation_mask,
    _stable_token_value,
    _transform_feature_sources,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PROMOTED_MANIFEST = (
    REPO_ROOT
    / "declan"
    / "figure4_active_sensing_atlas"
    / "figures"
    / "panel_C"
    / "diagnostics"
    / "continuous_joint"
    / "continuous_joint_promoted_observer_manifest.json"
)
OUT_DIR = (
    REPO_ROOT
    / "declan"
    / "figure4_active_sensing_atlas"
    / "figures"
    / "panel_C"
    / "diagnostics"
    / "continuous_tau_mlp_feature_decoder"
)

DEFAULT_INPUT_MODES = (
    "observed_compact",
    "augmented_observed_compact",
    "augmented_continuous_tau",
    "augmented_continuous_tau_residual",
    "augmented_continuous_tau_interactions",
    "augmented_true_tau",
    "augmented_true_tau_residual",
    "augmented_true_tau_interactions",
    "augmented_zero_static",
    "augmented_known_eye_model",
    "observed_plus_continuous_tau",
    "observed_plus_continuous_tau_interactions",
    "zero_static",
    "known_eye_model",
)
PRIMARY_MODE = "augmented_continuous_tau"
PRIOR_AUGMENTED_MODES = {
    "augmented_observed_compact",
    "augmented_continuous_tau",
    "augmented_continuous_tau_residual",
    "augmented_continuous_tau_interactions",
    "augmented_true_tau",
    "augmented_true_tau_residual",
    "augmented_true_tau_interactions",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float_key_map(value: Any) -> dict[float, Any]:
    if not value:
        return {}
    return {float(key): item for key, item in dict(value).items()}


def _scale_lookup(mapping: dict[float, Any], scale: float, default: Any) -> Any:
    return _scale_value(mapping, float(scale), default)


def _sidecar_path(sidecar_dir: Path | None, response_cache_path: str) -> Path | None:
    if sidecar_dir is None:
        return None
    return sidecar_dir / str(response_cache_path)


def _load_sidecar(path: Path | None) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory sidecar {path}")
    return _load_npz(path)


def _compact_matrix(response_counts: np.ndarray, basis: np.ndarray) -> np.ndarray:
    response = np.asarray(response_counts, dtype=np.float64)
    if response.ndim != 2:
        raise ValueError(f"response must be (time, unit), got {response.shape}")
    return (response @ basis).astype(np.float32, copy=False)


def _tau_features(tau: np.ndarray, n_time: int) -> np.ndarray:
    arr = np.asarray(tau, dtype=np.float64)
    if arr.shape != (int(n_time), 2):
        raise ValueError(f"tau must be ({n_time}, 2), got {arr.shape}")
    vel = np.diff(arr, axis=0, prepend=arr[:1])
    return np.concatenate([arr.reshape(-1), vel.reshape(-1), (arr * arr).reshape(-1)]).astype(np.float32)


def _response_tau_features(response_compact: np.ndarray, tau: np.ndarray) -> np.ndarray:
    response = np.asarray(response_compact, dtype=np.float64)
    arr = np.asarray(tau, dtype=np.float64)
    if response.ndim != 2 or arr.shape != (response.shape[0], 2):
        raise ValueError(f"response/tau shape mismatch: {response.shape}, {arr.shape}")
    interactions = np.concatenate(
        [
            (response * arr[:, 0:1]).reshape(-1),
            (response * arr[:, 1:2]).reshape(-1),
        ]
    )
    return interactions.astype(np.float32)


def _input_vector(
    mode: str,
    *,
    observed_compact: np.ndarray,
    zero_compact: np.ndarray,
    known_compact: np.ndarray,
    tau_hat: np.ndarray,
    true_tau: np.ndarray | None,
) -> np.ndarray:
    mode_name = str(mode)
    n_time = int(observed_compact.shape[0])
    obs_flat = observed_compact.reshape(-1).astype(np.float32, copy=False)
    if mode_name == "observed_compact":
        return obs_flat
    if mode_name == "zero_static":
        return zero_compact.reshape(-1).astype(np.float32, copy=False)
    if mode_name == "known_eye_model":
        return known_compact.reshape(-1).astype(np.float32, copy=False)
    if mode_name == "observed_plus_continuous_tau":
        return np.concatenate([obs_flat, _tau_features(tau_hat, n_time)])
    if mode_name == "observed_plus_continuous_tau_interactions":
        return np.concatenate(
            [
                obs_flat,
                _tau_features(tau_hat, n_time),
                _response_tau_features(observed_compact, tau_hat),
            ]
        )
    if mode_name == "observed_plus_true_tau":
        if true_tau is None:
            raise ValueError("observed_plus_true_tau requires observed trajectory coordinates")
        return np.concatenate([obs_flat, _tau_features(true_tau, n_time)])
    if mode_name == "observed_plus_true_tau_interactions":
        if true_tau is None:
            raise ValueError("observed_plus_true_tau_interactions requires observed trajectory coordinates")
        return np.concatenate(
            [
                obs_flat,
                _tau_features(true_tau, n_time),
                _response_tau_features(observed_compact, true_tau),
            ]
        )
    valid = ", ".join(DEFAULT_INPUT_MODES + ("observed_plus_true_tau",))
    raise ValueError(f"Unknown input mode {mode_name!r}; valid modes: {valid}")


def _base_mode(mode: str, *, for_test: bool) -> str:
    name = str(mode)
    if name == "augmented_observed_compact":
        return "observed_compact"
    if name == "augmented_continuous_tau":
        return "observed_plus_continuous_tau"
    if name == "augmented_continuous_tau_residual":
        return "observed_plus_continuous_tau"
    if name == "augmented_continuous_tau_interactions":
        return "observed_plus_continuous_tau_interactions"
    if name == "augmented_true_tau":
        return "observed_plus_true_tau" if for_test else "observed_plus_continuous_tau"
    if name == "augmented_true_tau_residual":
        return "observed_plus_true_tau" if for_test else "observed_plus_continuous_tau"
    if name == "augmented_true_tau_interactions":
        return "observed_plus_true_tau_interactions" if for_test else "observed_plus_continuous_tau_interactions"
    if name == "augmented_zero_static":
        return "zero_static"
    if name == "augmented_known_eye_model":
        return "known_eye_model"
    return name


def _is_augmented_mode(mode: str) -> bool:
    return str(mode).startswith("augmented_")


def _is_residual_mode(mode: str) -> bool:
    return str(mode) in {"augmented_continuous_tau_residual", "augmented_true_tau_residual"}


def _parse_float_list(text: str | None) -> list[float]:
    if text is None:
        return []
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _feature_cosine_array(z_hat: np.ndarray, z_true: np.ndarray) -> np.ndarray:
    pred = np.asarray(z_hat, dtype=np.float64)
    true = np.asarray(z_true, dtype=np.float64)
    denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1)
    out = np.full(pred.shape[0], np.nan, dtype=np.float64)
    valid = np.isfinite(denom) & (denom > 0.0)
    out[valid] = np.sum(pred[valid] * true[valid], axis=1) / denom[valid]
    return out


def _prefix_stats(prefix: str, stats: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def _fit_predict_residual_mlp(
    *,
    base_x_all: np.ndarray,
    residual_x_all: np.ndarray,
    z_all: np.ndarray,
    source_rows: np.ndarray,
    train_mask: np.ndarray,
    base_x_test: np.ndarray,
    residual_x_test: np.ndarray,
    config: MLPConfig,
    fold: int,
    spec_slug: str,
    feature_space_mode: str,
    alpha_grid: list[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit response-only base plus eye-trace residual correction.

    A source-held-out slice of the fold-training rows chooses the residual
    shrinkage.  Since the grid includes 0.0, this can fall back to the base
    decoder when the eye-trace correction is not useful on validation sources.
    """
    base_x = np.asarray(base_x_all, dtype=np.float32)
    residual_x = np.asarray(residual_x_all, dtype=np.float32)
    z = np.asarray(z_all, dtype=np.float32)
    sources = np.asarray(source_rows, dtype=int)
    train_mask_arr = np.asarray(train_mask, dtype=bool)
    if base_x.ndim != 2 or residual_x.ndim != 2 or z.ndim != 2:
        raise ValueError("residual MLP expects 2D base/residual/target matrices")
    if base_x.shape[0] != residual_x.shape[0] or base_x.shape[0] != z.shape[0]:
        raise ValueError(f"residual MLP row mismatch: {base_x.shape}, {residual_x.shape}, {z.shape}")
    train_indices_all = np.flatnonzero(train_mask_arr)
    if train_indices_all.size <= max(8, 2 * z.shape[1]):
        raise ValueError(f"Too few residual MLP training rows for {spec_slug} / {feature_space_mode}")

    alpha_val_rel = _source_group_validation_mask(
        sources[train_indices_all],
        validation_fraction=float(config.validation_fraction),
        seed=int(config.seed)
        + 911 * int(fold)
        + _stable_token_value(spec_slug)
        + _stable_token_value(feature_space_mode),
    )
    fit_indices = train_indices_all[~alpha_val_rel]
    alpha_val_indices = train_indices_all[alpha_val_rel]
    if fit_indices.size <= max(8, z.shape[1]) or alpha_val_indices.size == 0:
        fit_indices = train_indices_all
        alpha_val_indices = train_indices_all

    fit_mask = np.zeros(train_mask_arr.shape[0], dtype=bool)
    fit_mask[fit_indices] = True
    base_eval_x = np.concatenate(
        [
            base_x[fit_indices],
            base_x[alpha_val_indices],
            np.asarray(base_x_test, dtype=np.float32),
        ],
        axis=0,
    )
    base_eval_hat, base_stats = _fit_predict_mlp(
        x_all=base_x,
        z_all=z,
        source_rows=sources,
        train_mask=fit_mask,
        x_test=base_eval_x,
        config=config,
        fold=int(fold),
        spec_slug=f"{spec_slug}__base",
        feature_space_mode=feature_space_mode,
    )
    n_fit = int(fit_indices.size)
    n_alpha = int(alpha_val_indices.size)
    base_fit_hat = base_eval_hat[:n_fit]
    base_alpha_hat = base_eval_hat[n_fit : n_fit + n_alpha]
    base_test_hat = base_eval_hat[n_fit + n_alpha :]

    residual_targets = np.zeros_like(z, dtype=np.float32)
    residual_targets[fit_indices] = (z[fit_indices] - base_fit_hat).astype(np.float32)
    residual_eval_x = np.concatenate(
        [
            residual_x[alpha_val_indices],
            np.asarray(residual_x_test, dtype=np.float32),
        ],
        axis=0,
    )
    residual_eval_hat, residual_stats = _fit_predict_mlp(
        x_all=residual_x,
        z_all=residual_targets,
        source_rows=sources,
        train_mask=fit_mask,
        x_test=residual_eval_x,
        config=config,
        fold=int(fold),
        spec_slug=f"{spec_slug}__residual",
        feature_space_mode=feature_space_mode,
    )
    residual_alpha_hat = residual_eval_hat[:n_alpha]
    residual_test_hat = residual_eval_hat[n_alpha:]
    z_alpha = z[alpha_val_indices]

    grid = [float(value) for value in alpha_grid]
    if not grid:
        grid = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    if 0.0 not in grid:
        grid = [0.0, *grid]
    best_alpha = 0.0
    best_score = -float("inf")
    best_mse = float("inf")
    alpha_rows: list[dict[str, float]] = []
    for alpha in sorted(set(grid)):
        pred = base_alpha_hat + float(alpha) * residual_alpha_hat
        cos = _feature_cosine_array(pred, z_alpha)
        mse = float(np.mean((pred - z_alpha) ** 2))
        score = float(np.nanmean(cos))
        alpha_rows.append({"alpha": float(alpha), "mean_feature_cosine": score, "feature_mse": mse})
        if score > best_score + 1e-9 or (abs(score - best_score) <= 1e-9 and float(alpha) < best_alpha):
            best_alpha = float(alpha)
            best_score = score
            best_mse = mse

    z_hat = base_test_hat + best_alpha * residual_test_hat
    stats = {
        "residual_nested": True,
        "residual_base_input_dim": int(base_x.shape[1]),
        "residual_conditioned_input_dim": int(residual_x.shape[1]),
        "residual_alpha": float(best_alpha),
        "residual_alpha_grid": [float(row["alpha"]) for row in alpha_rows],
        "residual_alpha_val_mean_feature_cosine": float(best_score),
        "residual_alpha_val_feature_mse": float(best_mse),
        "residual_alpha_val_rows": int(n_alpha),
        "residual_fit_rows": int(n_fit),
        "residual_alpha_table": alpha_rows,
    }
    stats.update(_prefix_stats("base", base_stats))
    stats.update(_prefix_stats("residual", residual_stats))
    return z_hat.astype(np.float64, copy=False), stats


def _continuous_kwargs(metadata: dict[str, Any], *, scale: float) -> dict[str, Any]:
    ridge = _scale_lookup(_float_key_map(metadata.get("ridge_by_scale")), scale, float(metadata.get("ridge", 1e-6)))
    process_model = _scale_lookup(
        _float_key_map(metadata.get("trajectory_process_model_by_scale")),
        scale,
        str(metadata.get("trajectory_process_model", "ar1")),
    )
    brownian_scale = _scale_lookup(
        _float_key_map(metadata.get("brownian_cov_scale_by_scale")),
        scale,
        float(metadata.get("brownian_cov_scale", 1.0)),
    )
    return {
        "alpha": float(metadata.get("alpha", 0.92)),
        "process_var": float(metadata.get("process_var", 1e-3)),
        "observation_var": metadata.get("observation_var"),
        "observation_var_floor": float(metadata.get("observation_var_floor", 1e-6)),
        "ridge": float(ridge),
        "observation_model": str(metadata.get("observation_model", "time_constant")),
        "time_smoothing_sigma": float(metadata.get("time_smoothing_sigma", 0.0)),
        "time_shrinkage": float(metadata.get("time_shrinkage", 0.0)),
        "continuous_score_mode": str(metadata.get("continuous_score_mode", "quadratic_poisson_profile")),
        "trajectory_prior_mean": str(metadata.get("trajectory_prior_mean", "zero")),
        "trajectory_initial_position": str(metadata.get("trajectory_initial_position", "inferred")),
        "trajectory_initial_position_var": float(metadata.get("trajectory_initial_position_var", 1e-4)),
        "trajectory_process_model": str(process_model),
        "brownian_cov_floor": float(metadata.get("brownian_cov_floor", 1e-6)),
        "brownian_cov_scale": float(brownian_scale),
        "catalog_gaussian_smoothing_sigma": float(metadata.get("catalog_gaussian_smoothing_sigma", 0.0)),
        "catalog_gaussian_cov_floor": float(metadata.get("catalog_gaussian_cov_floor", 1e-6)),
        "catalog_gaussian_shrinkage": float(metadata.get("catalog_gaussian_shrinkage", 0.25)),
        "trajectory_basis_family": str(metadata.get("trajectory_basis_family", "dct")),
        "trajectory_basis_components": int(metadata.get("trajectory_basis_components", 4)),
        "trajectory_basis_smoothing_sigma": float(metadata.get("trajectory_basis_smoothing_sigma", 6.0)),
        "trajectory_basis_coeff_prior_var": float(metadata.get("trajectory_basis_coeff_prior_var", 1.0)),
        "catalog_residual_aggregation": str(metadata.get("catalog_residual_aggregation", "logmeanexp")),
        "catalog_residual_top_k": int(metadata.get("catalog_residual_top_k", 0)),
        "catalog_residual_all_anchor_shrinkage": float(metadata.get("catalog_residual_all_anchor_shrinkage", 0.0)),
        "catalog_residual_anchor_smoothing_sigma": float(metadata.get("catalog_residual_anchor_smoothing_sigma", 0.0)),
        "catalog_residual_anchor_smoothing_schedule": metadata.get("catalog_residual_anchor_smoothing_schedule") or None,
        "catalog_residual_refine_top_k": int(metadata.get("catalog_residual_refine_top_k", 0)),
        "quadratic_optimizer_max_iter": int(metadata.get("quadratic_optimizer_max_iter", 80)),
        "quadratic_continuation_scales": str(metadata.get("quadratic_continuation_scales", "1")),
        "quadratic_observation_scales": str(metadata.get("quadratic_observation_scales", "1")),
        "quadratic_intercept_ridge_multiplier": float(metadata.get("quadratic_intercept_ridge_multiplier", 1.0)),
        "quadratic_affine_intercept_scale": float(metadata.get("quadratic_affine_intercept_scale", 1.0)),
        "likelihood_scale": 1.0,
    }


def _selected_rows(trials: pd.DataFrame, *, scales: set[float], max_tables: int) -> pd.DataFrame:
    rows = trials.copy()
    rows = rows[rows["response_cache_path"].astype(str).str.len() > 0].copy()
    if scales:
        rows = rows[rows["prior_scale"].astype(float).isin(scales)].copy()
    rows = rows.reset_index(drop=True)
    if int(max_tables) > 0:
        rows = rows.iloc[: int(max_tables)].copy()
    if rows.empty:
        raise ValueError("No continuous-joint rows selected")
    return rows


def _build_dataset(
    *,
    rows: pd.DataFrame,
    response_run_dir: Path,
    metadata: dict[str, Any],
    projection_basis: np.ndarray,
    input_modes: list[str],
    progress_every: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    sidecar_dir = Path(metadata["trajectory_sidecar_dir"]) if metadata.get("trajectory_sidecar_dir") else None
    basis_path = Path(metadata["compact_basis_path"]) if metadata.get("compact_basis_path") else COMPACT_BASIS
    basis_key = str(metadata.get("basis_key", "basis"))
    basis_dim_by_scale = _float_key_map(metadata.get("basis_max_dim_by_scale"))
    default_basis_dim = int(metadata.get("basis_max_dim", 20))
    parts = {mode: [] for mode in input_modes}
    augmented_parts = {mode: [] for mode in input_modes if _is_augmented_mode(mode)}
    augmented_sources = {mode: [] for mode in input_modes if _is_augmented_mode(mode)}
    table_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []

    for row_index, row in rows.iterrows():
        if progress_every > 0 and (row_index + 1) % int(progress_every) == 0:
            print(f"built continuous-tau rows {row_index + 1} / {rows.shape[0]}", flush=True)
        response_cache_path = str(row["response_cache_path"])
        table_path = response_run_dir / response_cache_path
        table = _load_npz(table_path)
        sidecar = _load_sidecar(_sidecar_path(sidecar_dir, response_cache_path))
        prior = np.asarray(table["prior_lambda_counts"], dtype=np.float64)
        known = np.asarray(table["known_lambda_counts"], dtype=np.float64)
        zero = np.asarray(table["zero_lambda_counts"], dtype=np.float64)
        obs = np.asarray(table["y_obs_counts"], dtype=np.float64)
        n_units = int(obs.shape[1])
        candidate_ids = [str(value) for value in np.asarray(table["candidate_ids"]).tolist()]
        source_rows = [_source_row_from_candidate_id(candidate_id) for candidate_id in candidate_ids]
        true_idx = int(np.asarray(table["true_candidate_index"]).reshape(-1)[0])
        true_source = _source_row_from_candidate_id(candidate_ids[true_idx])
        prior_scale = float(row["prior_scale"])
        infer_basis_dim = int(_scale_lookup(basis_dim_by_scale, prior_scale, default_basis_dim))
        infer_basis, infer_meta = _load_continuous_basis(
            basis_path,
            n_units=n_units,
            basis_key=basis_key,
            basis_max_dim=infer_basis_dim,
        )
        trajectory_npz = sidecar
        trajectory_xy = _trajectory_from_table_or_npz(
            table=table,
            trajectory_npz=trajectory_npz,
            trajectory_key=str(metadata.get("trajectory_key", "prior_trajectory_xy")),
        )
        observed_xy = _observed_trajectory_from_table_or_npz(
            table=table,
            trajectory_npz=trajectory_npz,
            observed_trajectory_key=str(metadata.get("observed_trajectory_key", "observed_trajectory_xy")),
        )
        vectors = score_continuous_joint_score_vectors(
            y_obs_counts=obs,
            prior_lambda_counts=prior,
            known_lambda_counts=known,
            zero_lambda_counts=zero,
            trajectory_xy=trajectory_xy,
            true_candidate_index=true_idx,
            candidate_ids=candidate_ids,
            basis=infer_basis,
            true_trajectory_index=int(np.asarray(table.get("true_trajectory_index", [-1])).reshape(-1)[0]),
            observed_trajectory_xy=observed_xy,
            **_continuous_kwargs(metadata, scale=prior_scale),
        )
        tau_hat = np.asarray(vectors["filtered_state_means"], dtype=np.float64)[true_idx]
        observed_compact = _compact_matrix(obs, projection_basis)
        zero_compact = _compact_matrix(zero[true_idx], projection_basis)
        known_compact = _compact_matrix(known[true_idx], projection_basis)
        for mode in input_modes:
            parts[mode].append(
                _input_vector(
                    _base_mode(mode, for_test=True),
                    observed_compact=observed_compact,
                    zero_compact=zero_compact,
                    known_compact=known_compact,
                    tau_hat=tau_hat,
                    true_tau=observed_xy,
                )
            )
        for mode in augmented_parts:
            train_base_mode = _base_mode(mode, for_test=False)
            if mode in PRIOR_AUGMENTED_MODES:
                for candidate_index, source_row in enumerate(source_rows):
                    for trajectory_index in range(prior.shape[1]):
                        train_compact = _compact_matrix(prior[candidate_index, trajectory_index], projection_basis)
                        train_tau = np.asarray(trajectory_xy, dtype=np.float64)[candidate_index, trajectory_index]
                        augmented_parts[mode].append(
                            _input_vector(
                                train_base_mode,
                                observed_compact=train_compact,
                                zero_compact=train_compact,
                                known_compact=train_compact,
                                tau_hat=train_tau,
                                true_tau=train_tau,
                            )
                        )
                        augmented_sources[mode].append(int(source_row))
            elif mode in {"augmented_zero_static", "augmented_known_eye_model"}:
                bank = zero if mode == "augmented_zero_static" else known
                for candidate_index, source_row in enumerate(source_rows):
                    train_compact = _compact_matrix(bank[candidate_index], projection_basis)
                    train_tau = np.zeros((train_compact.shape[0], 2), dtype=np.float64)
                    augmented_parts[mode].append(
                        _input_vector(
                            train_base_mode,
                            observed_compact=train_compact,
                            zero_compact=train_compact,
                            known_compact=train_compact,
                            tau_hat=train_tau,
                            true_tau=train_tau,
                        )
                    )
                    augmented_sources[mode].append(int(source_row))
            else:
                raise ValueError(f"No augmented training bank defined for mode {mode!r}")
        recovery = vectors.get("trajectory_recovery", {})
        table_rows.append(
            {
                "row_index": int(row_index),
                "table_index": int(row.get("table_index", row_index)),
                "manifest_table_index": int(row.get("manifest_table_index", row_index)),
                "trial_id": int(row["trial_id"]),
                "response_cache_path": response_cache_path,
                "candidate_set_mode": str(row.get("candidate_set_mode", "")),
                "observation_family": str(row.get("observation_family", "")),
                "prior_family": str(row.get("prior_family", "")),
                "prior_scale": prior_scale,
                "axis_catalog_mode": str(row.get("axis_catalog_mode", "")),
                "true_candidate_index": int(true_idx),
                "true_candidate_id": candidate_ids[true_idx],
                "true_source_row": int(true_source),
                "n_candidates": int(prior.shape[0]),
                "n_trajectories": int(prior.shape[1]),
                "n_timebins": int(obs.shape[0]),
                "n_units": int(obs.shape[1]),
                "projection_basis_dim": int(projection_basis.shape[1]),
                "continuous_inference_basis_dim": int(infer_basis.shape[1]),
                "trajectory_rmse": float(recovery.get("trajectory_rmse", np.nan)),
                "trajectory_corr_mean": float(recovery.get("trajectory_corr_mean", np.nan)),
                "continuous_joint_true_rank": int(vectors["continuous_joint_true_rank"]),
                "continuous_joint_correct": bool(int(vectors["continuous_joint_pred_candidate_index"]) == int(true_idx)),
            }
        )
        qc_rows.append(
            {
                "row_index": int(row_index),
                "infer_basis_dim": int(infer_basis.shape[1]),
                "infer_basis_source": str(infer_meta.get("basis_source", "")),
                "continuous_score_corr_with_zero": float(vectors["continuous_joint_score_corr_with_zero"]),
                "continuous_minus_zero_true_score": float(vectors["continuous_joint_minus_zero_true_score"]),
            }
        )

    arrays = {mode: np.stack(values, axis=0).astype(np.float32) for mode, values in parts.items()}
    train_arrays = {mode: np.stack(values, axis=0).astype(np.float32) for mode, values in augmented_parts.items()}
    train_source_arrays = {
        mode: np.asarray(values, dtype=np.int64)
        for mode, values in augmented_sources.items()
    }
    meta = {
        "input_modes": input_modes,
        "n_rows": int(rows.shape[0]),
        "input_dims": {mode: int(arr.shape[1]) for mode, arr in arrays.items()},
        "augmented_train_rows": {mode: int(train_arrays[mode].shape[0]) for mode in train_arrays},
        "qc_rows": qc_rows,
    }
    return arrays, train_arrays, train_source_arrays, pd.DataFrame(table_rows), meta


def _fit_modes(
    *,
    arrays: dict[str, np.ndarray],
    train_arrays: dict[str, np.ndarray],
    train_source_rows_by_mode: dict[str, np.ndarray],
    table_rows: pd.DataFrame,
    feature_table: Any,
    feature_space_mode: str,
    feature_dim: int,
    n_folds: int,
    fold_seed: int,
    mlp_config: MLPConfig,
    residual_alpha_grid: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_rows = table_rows["true_source_row"].to_numpy(dtype=int)
    fold_by_source = _assign_source_folds(source_rows, n_folds=n_folds, seed=fold_seed)
    row_folds = np.asarray([fold_by_source[int(source)] for source in source_rows], dtype=int)
    if "observed_compact" in arrays:
        response_input_dim = int(arrays["observed_compact"].shape[1])
    else:
        response_input_dim = int(min(arr.shape[1] for arr in arrays.values()))
    trial_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []

    for fold in sorted(set(row_folds.tolist())):
        heldout_sources = {int(source) for source, source_fold in fold_by_source.items() if int(source_fold) == int(fold)}
        fit_sources = np.asarray(
            [int(source) for source in feature_table.source_rows.tolist() if int(source) not in heldout_sources],
            dtype=int,
        )
        transform = _fit_feature_transform(
            feature_table,
            fit_sources=fit_sources,
            feature_dim=int(feature_dim),
            feature_space_mode=str(feature_space_mode),
            feature_weights=None,
        )
        z_all = _transform_feature_sources(transform, feature_table, source_rows)
        test_mask = row_folds == int(fold)
        z_true = z_all[test_mask]
        test_meta = table_rows.loc[test_mask].reset_index(drop=True)
        for mode, x_all in arrays.items():
            if mode in train_arrays:
                fit_x_all = train_arrays[mode]
                fit_source_rows = train_source_rows_by_mode[mode]
                fit_z_all = _transform_feature_sources(transform, feature_table, fit_source_rows)
                train_mask = np.asarray([fold_by_source.get(int(source), -1) != int(fold) for source in fit_source_rows])
                if _is_residual_mode(mode):
                    train_source = "augmented_prior_trajectory_responses_nested_residual"
                elif mode == "augmented_zero_static":
                    train_source = "augmented_zero_static_responses"
                elif mode == "augmented_known_eye_model":
                    train_source = "augmented_known_eye_responses"
                else:
                    train_source = "augmented_prior_trajectory_responses"
            else:
                fit_x_all = x_all
                fit_source_rows = source_rows
                fit_z_all = z_all
                train_mask = ~test_mask
                train_source = "observed_true_rows"
            if _is_residual_mode(mode):
                z_hat, stats = _fit_predict_residual_mlp(
                    base_x_all=fit_x_all[:, :response_input_dim],
                    residual_x_all=fit_x_all,
                    z_all=fit_z_all,
                    source_rows=fit_source_rows,
                    train_mask=train_mask,
                    base_x_test=x_all[test_mask, :response_input_dim],
                    residual_x_test=x_all[test_mask],
                    config=mlp_config,
                    fold=int(fold),
                    spec_slug=mode,
                    feature_space_mode=transform.feature_space_mode,
                    alpha_grid=residual_alpha_grid,
                )
                decoder_mode = "mlp_residual"
            else:
                z_hat, stats = _fit_predict_mlp(
                    x_all=fit_x_all,
                    z_all=fit_z_all,
                    source_rows=fit_source_rows,
                    train_mask=train_mask,
                    x_test=x_all[test_mask],
                    config=mlp_config,
                    fold=int(fold),
                    spec_slug=mode,
                    feature_space_mode=transform.feature_space_mode,
                )
                decoder_mode = "mlp"
            model_row = {
                "input_mode": mode,
                "decoder_mode": decoder_mode,
                "fold": int(fold),
                "n_train_rows": int(np.sum(train_mask)),
                "n_test_rows": int(np.sum(test_mask)),
                "train_source": train_source,
                "input_dim": int(x_all.shape[1]),
                "latent": transform.latent,
                "feature_space_mode": transform.feature_space_mode,
                "feature_dim": int(transform.feature_dim),
                "raw_feature_dim": int(transform.raw_feature_dim),
                "feature_fit_scope": transform.fit_scope,
                "feature_preprocessing": transform.preprocessing,
                "feature_whitened": bool(transform.whitened),
                "feature_variance_fraction": float(transform.explained_variance_sum),
            }
            model_row.update(stats)
            model_rows.append(model_row)
            for local_index, meta in enumerate(test_meta.to_dict(orient="records")):
                out = dict(meta)
                out.update(
                    {
                        "input_mode": mode,
                        "decoder_mode": decoder_mode,
                        "fold": int(fold),
                        "latent": transform.latent,
                        "feature_space_mode": transform.feature_space_mode,
                        "feature_dim": int(transform.feature_dim),
                        "feature_variance_fraction": float(transform.explained_variance_sum),
                        "input_dim": int(x_all.shape[1]),
                        "n_train_rows": int(np.sum(train_mask)),
                        "train_source": train_source,
                    }
                )
                out.update(_metrics(z_hat[local_index], z_true[local_index]))
                trial_rows.append(out)

    return pd.DataFrame(trial_rows), pd.DataFrame(model_rows)


def _summarize(trials: pd.DataFrame) -> pd.DataFrame:
    scale = (
        trials.groupby(["input_mode", "prior_scale"], as_index=False)
        .agg(
            n=("feature_cosine", "size"),
            mean_feature_cosine=("feature_cosine", "mean"),
            median_feature_cosine=("feature_cosine", "median"),
            mean_feature_mse=("feature_mse", "mean"),
            median_feature_rmse=("feature_rmse", "median"),
            median_trajectory_rmse=("trajectory_rmse", "median"),
        )
        .sort_values(["prior_scale", "input_mode"])
    )
    overall = (
        trials.groupby(["input_mode"], as_index=False)
        .agg(
            n=("feature_cosine", "size"),
            mean_feature_cosine=("feature_cosine", "mean"),
            median_feature_cosine=("feature_cosine", "median"),
            mean_feature_mse=("feature_mse", "mean"),
            median_feature_rmse=("feature_rmse", "median"),
            median_trajectory_rmse=("trajectory_rmse", "median"),
        )
        .sort_values("input_mode")
    )
    overall["prior_scale"] = "all"
    return pd.concat([scale, overall[scale.columns]], ignore_index=True)


def _contrasts(trials: pd.DataFrame, *, primary_mode: str, n_bootstrap: int, seed: int) -> pd.DataFrame:
    pivot = trials.pivot_table(
        index=["table_index", "trial_id", "prior_scale", "true_source_row"],
        columns="input_mode",
        values="feature_cosine",
        aggfunc="first",
    )
    pairs = [
        (primary_mode, "augmented_zero_static", "continuous_tau_minus_augmented_zero_static"),
        (primary_mode, "zero_static", "continuous_tau_minus_zero_static"),
        (primary_mode, "observed_compact", "continuous_tau_minus_observed_compact"),
        (primary_mode, "augmented_observed_compact", "continuous_tau_minus_augmented_observed_compact"),
        (primary_mode, "augmented_known_eye_model", "continuous_tau_minus_augmented_known_eye_model"),
        (primary_mode, "known_eye_model", "continuous_tau_minus_known_eye_model"),
        ("augmented_continuous_tau_residual", "augmented_zero_static", "continuous_tau_residual_minus_augmented_zero_static"),
        ("augmented_continuous_tau_residual", primary_mode, "continuous_tau_residual_minus_raw_continuous_tau"),
        ("augmented_observed_compact", "augmented_zero_static", "augmented_observed_minus_augmented_zero_static"),
        (
            "augmented_continuous_tau_interactions",
            "augmented_zero_static",
            "continuous_tau_interactions_minus_augmented_zero_static",
        ),
        ("augmented_true_tau", "augmented_zero_static", "true_tau_no_interactions_minus_augmented_zero_static"),
        ("augmented_true_tau", primary_mode, "true_tau_no_interactions_minus_continuous_tau"),
        ("augmented_true_tau_residual", "augmented_zero_static", "true_tau_residual_minus_augmented_zero_static"),
        ("augmented_true_tau_residual", "augmented_observed_compact", "true_tau_residual_minus_augmented_observed"),
        ("augmented_true_tau_residual", "augmented_true_tau", "true_tau_residual_minus_raw_true_tau"),
        ("augmented_true_tau_interactions", "augmented_zero_static", "true_tau_minus_augmented_zero_static"),
        ("augmented_true_tau_interactions", "zero_static", "true_tau_minus_zero_static"),
        ("augmented_true_tau_interactions", primary_mode, "true_tau_minus_continuous_tau"),
    ]
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    for lhs, rhs, name in pairs:
        if lhs not in pivot.columns or rhs not in pivot.columns:
            continue
        vals = (pivot[lhs] - pivot[rhs]).rename("delta").reset_index()
        vals = vals[np.isfinite(vals["delta"].to_numpy(dtype=float))]
        for scale_value, scale_rows in vals.groupby("prior_scale", sort=True):
            values = scale_rows["delta"].to_numpy(dtype=float)
            mean, lo, hi = _bootstrap_mean(values, rng, int(n_bootstrap))
            rows.append(
                {
                    "contrast": name,
                    "lhs": lhs,
                    "rhs": rhs,
                    "prior_scale": float(scale_value),
                    "mean_feature_cosine_delta": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "fraction_positive": float(np.mean(values > 0.0)) if values.size else float("nan"),
                    "n": int(values.size),
                }
            )
        values = vals["delta"].to_numpy(dtype=float)
        mean, lo, hi = _bootstrap_mean(values, rng, int(n_bootstrap))
        rows.append(
            {
                "contrast": name,
                "lhs": lhs,
                "rhs": rhs,
                "prior_scale": "all",
                "mean_feature_cosine_delta": mean,
                "ci_low": lo,
                "ci_high": hi,
                "fraction_positive": float(np.mean(values > 0.0)) if values.size else float("nan"),
                "n": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, contrasts: pd.DataFrame, out_dir: Path, *, primary_mode: str) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.1), constrained_layout=True)
    colors = {
        "observed_compact": "#66717d",
        "augmented_observed_compact": "#4b5563",
        "augmented_continuous_tau": "#4c78a8",
        "augmented_continuous_tau_residual": "#2f6faa",
        "augmented_continuous_tau_interactions": "#235789",
        "augmented_true_tau": "#374151",
        "augmented_true_tau_residual": "#000000",
        "augmented_true_tau_interactions": "#111827",
        "augmented_zero_static": "#8a5ca8",
        "augmented_known_eye_model": "#2f8f6a",
        "observed_plus_continuous_tau": "#4c78a8",
        "observed_plus_continuous_tau_interactions": "#235789",
        "observed_plus_true_tau_interactions": "#111827",
        "zero_static": "#8a5ca8",
        "known_eye_model": "#2f8f6a",
    }
    scale_summary = summary[summary["prior_scale"].astype(str) != "all"].copy()
    ax = axes[0]
    x_lookup = {0.5: 0.0, 1.0: 1.0, 2.0: 2.0}
    for mode, block in scale_summary.groupby("input_mode", sort=True):
        block = block.sort_values("prior_scale")
        ax.plot(
            block["prior_scale"].astype(float).map(x_lookup),
            block["mean_feature_cosine"],
            marker="o",
            lw=2.1 if mode == primary_mode else 1.4,
            color=colors.get(str(mode), "#555555"),
            label=str(mode).replace("_", " "),
        )
    ax.set_title("A. direct feature decoding")
    ax.set_ylabel("feature cosine")
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.grid(axis="y", color="#d9dee5", lw=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, loc="best")

    ax = axes[1]
    plot_contrasts = contrasts[
        contrasts["contrast"].isin(
            ["continuous_tau_minus_augmented_zero_static", "continuous_tau_minus_augmented_observed_compact"]
        )
        & (contrasts["prior_scale"].astype(str) != "all")
    ].copy()
    for offset, (name, label, color) in zip(
        [-0.07, 0.07],
        [
            ("continuous_tau_minus_augmented_zero_static", "continuous tau - 0x", "#235789"),
            ("continuous_tau_minus_augmented_observed_compact", "continuous tau - hidden", "#4c78a8"),
        ],
        strict=True,
    ):
        block = plot_contrasts[plot_contrasts["contrast"] == name].sort_values("prior_scale")
        if block.empty:
            continue
        x = block["prior_scale"].astype(float).map(x_lookup).to_numpy(dtype=float) + offset
        y = block["mean_feature_cosine_delta"].to_numpy(dtype=float)
        yerr = np.vstack([y - block["ci_low"].to_numpy(dtype=float), block["ci_high"].to_numpy(dtype=float) - y])
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=2.5, lw=1.5, color=color, label=label)
    ax.axhline(0.0, color="#6b7280", lw=0.9)
    ax.set_title("B. paired contrasts")
    ax.set_ylabel("cosine difference")
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.grid(axis="y", color="#d9dee5", lw=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7.0, loc="best")

    png = out_dir / "continuous_tau_mlp_feature_decoder.png"
    pdf = out_dir / "continuous_tau_mlp_feature_decoder.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def _write_readme(out_dir: Path, summary: pd.DataFrame, contrasts: pd.DataFrame, manifest: dict[str, Any]) -> None:
    overall = summary[summary["prior_scale"].astype(str) == "all"].set_index("input_mode")
    def value(mode: str) -> float:
        if mode not in overall.index:
            return float("nan")
        return float(overall.loc[mode, "mean_feature_cosine"])

    contrast_all = contrasts[contrasts["prior_scale"].astype(str) == "all"].set_index("contrast")
    def contrast(name: str) -> tuple[float, float, float]:
        if name not in contrast_all.index:
            return float("nan"), float("nan"), float("nan")
        row = contrast_all.loc[name]
        return float(row["mean_feature_cosine_delta"]), float(row["ci_low"]), float(row["ci_high"])

    delta, lo, hi = contrast("continuous_tau_minus_augmented_zero_static")
    true_residual_delta, true_residual_lo, true_residual_hi = contrast(
        "true_tau_residual_minus_augmented_zero_static"
    )
    lines = [
        "# Continuous-Tau MLP Feature Decoder",
        "",
        "Hybrid diagnostic: continuous-eye-trace inference supplies `tau_hat`, then",
        "a Tejas-style MLP decodes directly to a compact image-feature embedding.",
        "The endpoint is continuous feature recovery, not image-candidate choice.",
        "Augmented modes train on the continuous response-bank rows",
        "`(prior response, trajectory) -> phi(source image)` and test on held-out",
        "observed responses with continuous `tau_hat`.",
        "Residual modes train a response-only MLP first, then allow an eye-trace",
        "correction whose shrinkage is selected on source-held-out validation rows;",
        "the correction grid includes zero, so the model can fall back to the",
        "response-only decoder.",
        "",
        "Primary input:",
        "",
        "```text",
        "compact observed response movie",
        "+ continuous tau_hat features",
        "-> z_hat ~= PCA(phi(image))",
        "```",
        "",
        "All-scale feature cosine:",
        "",
        "```text",
        f"observed compact only:                 {value('observed_compact'):.4f}",
        f"augmented compact only:                {value('augmented_observed_compact'):.4f}",
        f"augmented continuous tau:              {value('augmented_continuous_tau'):.4f}",
        f"augmented continuous tau residual:     {value('augmented_continuous_tau_residual'):.4f}",
        f"augmented continuous tau + interactions: {value('augmented_continuous_tau_interactions'):.4f}",
        f"augmented true tau:                    {value('augmented_true_tau'):.4f}",
        f"augmented true tau residual:           {value('augmented_true_tau_residual'):.4f}",
        f"augmented true tau + interactions:     {value('augmented_true_tau_interactions'):.4f}",
        f"augmented 0x stabilized response:      {value('augmented_zero_static'):.4f}",
        f"augmented known-eye model response:     {value('augmented_known_eye_model'):.4f}",
        f"true-row 0x stabilized response:       {value('zero_static'):.4f}",
        f"true-row known-eye model response:      {value('known_eye_model'):.4f}",
        "```",
        "",
        "Primary paired contrast:",
        "",
        "```text",
        f"continuous tau - augmented 0x stabilized: {delta:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]",
        f"true tau residual - augmented 0x stabilized: {true_residual_delta:+.4f}  CI [{true_residual_lo:+.4f}, {true_residual_hi:+.4f}]",
        "```",
        "",
        "Interpretation boundary: the continuous trajectory estimate is recovered",
        "using the true-image branch of the continuous-joint model for each held-out",
        "supervised row. This avoids trajectory-candidate selection and image",
        "candidate readout at the endpoint, but it is still a known-image",
        "trajectory-conditioning upper-bound diagnostic.",
        "",
        "Outputs:",
        "",
        "- `continuous_tau_mlp_feature_decoder_trials.csv`",
        "- `continuous_tau_mlp_feature_decoder_summary.csv`",
        "- `continuous_tau_mlp_feature_decoder_contrasts.csv`",
        "- `continuous_tau_mlp_feature_decoder_models.csv`",
        "- `continuous_tau_mlp_feature_decoder_dataset.npz`",
        "- `continuous_tau_mlp_feature_decoder_manifest.json`",
    ]
    (out_dir / "continuous_tau_mlp_feature_decoder_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promoted-manifest", type=Path, default=PROMOTED_MANIFEST)
    parser.add_argument("--feature-npz", type=Path, default=FEATURE_NPZ)
    parser.add_argument("--latent", default="pyramid_local_field")
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--feature-space-mode", default="fold_zscore_whitened_pca")
    parser.add_argument("--projection-basis-path", type=Path, default=COMPACT_BASIS)
    parser.add_argument("--projection-basis-key", default="basis")
    parser.add_argument("--projection-basis-dim", type=int, default=20)
    parser.add_argument("--input-modes", default=",".join(DEFAULT_INPUT_MODES))
    parser.add_argument("--scales", default="0.5,1.0,2.0")
    parser.add_argument("--max-tables", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260624)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--mlp-layers", type=int, default=4)
    parser.add_argument("--mlp-dropout", type=float, default=0.0)
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-5)
    parser.add_argument("--mlp-batch-size", type=int, default=256)
    parser.add_argument("--mlp-epochs", type=int, default=300)
    parser.add_argument("--mlp-patience", type=int, default=40)
    parser.add_argument("--mlp-validation-fraction", type=float, default=0.2)
    parser.add_argument("--mlp-max-train-samples", type=int, default=0)
    parser.add_argument("--mlp-device", default="auto")
    parser.add_argument("--residual-alpha-grid", default="0,0.05,0.1,0.2,0.35,0.5,0.75,1.0")
    parser.add_argument("--progress-every", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser


def build(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    promoted = _load_json(Path(args.promoted_manifest))
    artifact = dict(promoted["artifact"])
    continuous_run_dir = Path(artifact["run_dir"])
    metadata = _load_json(Path(artifact["metadata_json"]))
    response_run_dir = Path(metadata["run_dir"])
    continuous_trials = pd.read_csv(continuous_run_dir / "continuous_joint_trials.csv")
    selected = _selected_rows(continuous_trials, scales=_parse_scales(args.scales), max_tables=int(args.max_tables))
    first_table = _load_npz(response_run_dir / str(selected.iloc[0]["response_cache_path"]))
    n_units = int(np.asarray(first_table["y_obs_counts"]).shape[1])
    projection_basis, projection_meta = _load_basis(
        Path(args.projection_basis_path),
        n_units=n_units,
        basis_key=str(args.projection_basis_key),
        max_dim=int(args.projection_basis_dim),
    )
    input_modes = _parse_str_list(args.input_modes)
    if not input_modes:
        raise ValueError("--input-modes must list at least one mode")
    arrays, train_arrays, train_source_arrays, table_rows, dataset_meta = _build_dataset(
        rows=selected,
        response_run_dir=response_run_dir,
        metadata=metadata,
        projection_basis=projection_basis,
        input_modes=input_modes,
        progress_every=int(args.progress_every),
    )
    feature_table, feature_meta = _load_feature_table(Path(args.feature_npz), latent=str(args.latent))
    mlp_config = MLPConfig(
        hidden_dim=int(args.mlp_hidden_dim),
        layers=int(args.mlp_layers),
        dropout=float(args.mlp_dropout),
        learning_rate=float(args.mlp_learning_rate),
        weight_decay=float(args.mlp_weight_decay),
        batch_size=int(args.mlp_batch_size),
        epochs=int(args.mlp_epochs),
        patience=int(args.mlp_patience),
        validation_fraction=float(args.mlp_validation_fraction),
        max_train_samples=int(args.mlp_max_train_samples),
        device=str(args.mlp_device),
        seed=int(args.fold_seed),
    )
    trials, models = _fit_modes(
        arrays=arrays,
        train_arrays=train_arrays,
        train_source_rows_by_mode=train_source_arrays,
        table_rows=table_rows,
        feature_table=feature_table,
        feature_space_mode=str(args.feature_space_mode),
        feature_dim=int(args.feature_dim),
        n_folds=int(args.n_folds),
        fold_seed=int(args.fold_seed),
        mlp_config=mlp_config,
        residual_alpha_grid=_parse_float_list(args.residual_alpha_grid),
    )
    summary = _summarize(trials)
    contrasts = _contrasts(trials, primary_mode=PRIMARY_MODE, n_bootstrap=int(args.n_bootstrap), seed=int(args.fold_seed) + 17)

    trials_path = out_dir / "continuous_tau_mlp_feature_decoder_trials.csv"
    summary_path = out_dir / "continuous_tau_mlp_feature_decoder_summary.csv"
    contrasts_path = out_dir / "continuous_tau_mlp_feature_decoder_contrasts.csv"
    models_path = out_dir / "continuous_tau_mlp_feature_decoder_models.csv"
    dataset_path = out_dir / "continuous_tau_mlp_feature_decoder_dataset.npz"
    trials.to_csv(trials_path, index=False)
    summary.to_csv(summary_path, index=False)
    contrasts.to_csv(contrasts_path, index=False)
    models.to_csv(models_path, index=False)
    np.savez_compressed(
        dataset_path,
        **{f"X__{mode}": arr for mode, arr in arrays.items()},
        **{f"X_train__{mode}": arr for mode, arr in train_arrays.items()},
        **{f"source_row_train__{mode}": arr for mode, arr in train_source_arrays.items()},
        source_row=table_rows["true_source_row"].to_numpy(dtype=np.int64),
        table_index=table_rows["table_index"].to_numpy(dtype=np.int64),
        prior_scale=table_rows["prior_scale"].to_numpy(dtype=np.float32),
    )
    png, pdf = _plot(summary, contrasts, out_dir, primary_mode=PRIMARY_MODE)
    manifest = {
        "analysis": "continuous_tau_mlp_feature_decoder",
        "promoted_manifest": Path(args.promoted_manifest),
        "continuous_run_dir": continuous_run_dir,
        "response_run_dir": response_run_dir,
        "n_rows": int(table_rows.shape[0]),
        "feature": {**feature_meta, "feature_dim_requested": int(args.feature_dim), "feature_space_mode": str(args.feature_space_mode)},
        "projection_basis": projection_meta,
        "continuous_joint_metadata": {
            key: metadata.get(key)
            for key in [
                "continuous_score_mode",
                "basis_max_dim_by_scale",
                "ridge_by_scale",
                "trajectory_process_model_by_scale",
                "brownian_cov_scale_by_scale",
                "trajectory_initial_position",
                "trajectory_sidecar_dir",
            ]
        },
        "input_modes": input_modes,
        "primary_mode": PRIMARY_MODE,
        "dataset": dataset_meta,
        "mlp": mlp_config.__dict__,
        "outputs": {
            "trials": trials_path,
            "summary": summary_path,
            "contrasts": contrasts_path,
            "models": models_path,
            "dataset_npz": dataset_path,
            "figure_png": png,
            "figure_pdf": pdf,
        },
    }
    manifest_path = out_dir / "continuous_tau_mlp_feature_decoder_manifest.json"
    _write_json(manifest_path, manifest)
    _write_readme(out_dir, summary, contrasts, manifest)
    print(f"Wrote {out_dir}")
    return out_dir


def main() -> None:
    build(build_parser().parse_args())


if __name__ == "__main__":
    main()
