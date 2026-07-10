"""Build a linear synthetic-prior feature observer for Figure 4C.

This diagnostic is the candidate-free, no-MLP feature endpoint requested after
the continuous-joint observer experiments.  The default path first decodes a
response-only feature estimate, uses that estimate to condition a compact
response-vs-displacement map, recovers a continuous ``tau_hat`` path under an
empirically calibrated synthetic FEM prior (``synthetic_empirical_confined`` by
default), and then fits a source-disjoint linear-Gaussian feature observer on
observed response rows:

    compact_response_and_tau_features = A z + noise
    z ~ N(0, I)

The feature endpoint is not a finite image-candidate posterior, not finite
image-catalog search, and not catalog replay.  The forward-model modes invert
the fitted compact model ``F(z, tau)`` directly: fixed-tau modes solve for
``z`` by ridge/MAP regression, and the hidden-joint mode alternates tau MAP
updates under the empirically calibrated synthetic confined prior with fixed-tau
``z`` updates.  The current implementation still uses response-table samples to
calibrate compact response-vs-displacement geometry, but that is geometry
calibration rather than catalog search, and it does not train the feature
decoder by expanding cached prior trajectory rows.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.backimage_trajectory_observer.analyze_continuous_joint_trajectory import (
    _candidate_ids,
    _observed_trajectory_from_table_or_npz,
    _parse_scale_value_map,
    _scale_string_value,
    _scale_value,
    _scalar_int,
    _synthetic_empirical_confined_process_prior,
    _trajectory_from_table_or_npz,
    _trajectory_xy_by_candidate,
    ar1_profile_log_score,
    project_response_delta,
    quadratic_profile_log_score,
)
from declan.figure4_active_sensing_atlas.scripts.build_panel_c_continuous_feature_embedding_reconstruction import (
    COMPACT_BASIS,
    DEFAULT_FEATURE_SPACE_MODES,
    FEATURE_NPZ,
    PRIMARY_LATENT,
    SOURCE_ROOT,
    FeatureTable,
    FeatureTransform,
    ForwardPosteriorModel,
    _assign_source_folds,
    _bootstrap_mean,
    _clean_axis,
    _configure_matplotlib,
    _feature_space_config,
    _fit_feature_transform,
    _fit_forward_posterior,
    _json_ready,
    _load_basis,
    _load_feature_table,
    _load_feature_weights,
    _load_npz,
    _metrics,
    _parse_scales,
    _parse_str_list,
    _predict_z,
    _source_row_from_candidate_id,
    _transform_feature_sources,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = (
    REPO_ROOT
    / "declan"
    / "figure4_active_sensing_atlas"
    / "figures"
    / "panel_C"
    / "diagnostics"
    / "linear_synthetic_prior_feature_observer"
)

DEFAULT_OBSERVER_MODES = (
    "response_only",
    "pose_known_forward_model",
    "hidden_joint_forward_model",
    "estimated_tau_forward_model",
    "zero_tau_forward_model",
    "feature_conditioned_tau_linear",
    "feature_conditioned_tau_interactions",
    "pose_known_nested_tau_linear",
    "pose_known_nested_tau_interactions",
    "pose_known_tau_linear",
    "pose_known_tau_interactions",
    "synthetic_tau_linear",
    "synthetic_tau_interactions",
    "zero_static",
)
OBSERVER_ORDER = [
    "response_only",
    "pose_known_forward_model",
    "hidden_joint_forward_model",
    "estimated_tau_forward_model",
    "zero_tau_forward_model",
    "feature_conditioned_tau_linear",
    "feature_conditioned_tau_interactions",
    "pose_known_nested_tau_linear",
    "pose_known_nested_tau_interactions",
    "pose_known_tau_linear",
    "pose_known_tau_interactions",
    "synthetic_tau_linear",
    "synthetic_tau_interactions",
    "true_tau_linear",
    "true_tau_interactions",
    "zero_static",
    "known_eye_response_only",
]
OBSERVER_LABELS = {
    "response_only": "response only",
    "pose_known_forward_model": "recorded-tau compact forward diagnostic",
    "hidden_joint_forward_model": "hidden-joint forward model",
    "estimated_tau_forward_model": "estimated-tau forward model",
    "zero_tau_forward_model": "zero-tau forward model",
    "feature_conditioned_tau_linear": "z0-conditioned tau",
    "feature_conditioned_tau_interactions": "z0-conditioned tau x response",
    "pose_known_nested_tau_linear": "pose-known gated diagnostic",
    "pose_known_nested_tau_interactions": "pose-known gated diagnostic x response",
    "pose_known_tau_linear": "pose-known residual diagnostic",
    "pose_known_tau_interactions": "pose-known residual diagnostic x response",
    "synthetic_tau_linear": "synthetic tau",
    "synthetic_tau_interactions": "synthetic tau x response",
    "true_tau_linear": "recorded-tau linear diagnostic",
    "true_tau_interactions": "recorded-tau x response diagnostic",
    "zero_static": "0x stabilized",
    "known_eye_response_only": "known-trace response control",
}
OBSERVER_COLORS = {
    "response_only": "#4b5563",
    "pose_known_forward_model": "#047857",
    "hidden_joint_forward_model": "#0ea5e9",
    "estimated_tau_forward_model": "#a21caf",
    "zero_tau_forward_model": "#92400e",
    "feature_conditioned_tau_linear": "#0891b2",
    "feature_conditioned_tau_interactions": "#7c3aed",
    "pose_known_nested_tau_linear": "#0f172a",
    "pose_known_nested_tau_interactions": "#e11d48",
    "pose_known_tau_linear": "#111827",
    "pose_known_tau_interactions": "#db2777",
    "synthetic_tau_linear": "#0f766e",
    "synthetic_tau_interactions": "#2563eb",
    "true_tau_linear": "#374151",
    "true_tau_interactions": "#9333ea",
    "zero_static": "#b45309",
    "known_eye_response_only": "#be123c",
}


@dataclass(frozen=True)
class LinearFeatureObserverSpec:
    slug: str
    label: str
    train_bank: str
    test_input: str
    interpretation: str


@dataclass
class SampleBank:
    x: np.ndarray
    source_rows: np.ndarray
    table_indices: np.ndarray


@dataclass
class GeometryTable:
    source_rows: np.ndarray
    zero_compact: np.ndarray
    prior_delta_compact: np.ndarray
    trajectory_xy: np.ndarray
    source_samples: np.ndarray
    observation_scale: float


@dataclass
class TestSet:
    rows: pd.DataFrame
    x_by_mode: dict[str, np.ndarray]
    tau_hat: np.ndarray
    tau_true: np.ndarray
    observed_compact: np.ndarray
    geometry_tables: list[GeometryTable]


def _new_feature_prediction_parts() -> dict[str, list[Any]]:
    return {
        "rows": [],
        "z_hat": [],
        "z_true": [],
        "z_train_mean": [],
        "raw_hat_projected": [],
        "raw_true_projected": [],
        "raw_train_mean_projected": [],
    }


SPECS = [
    LinearFeatureObserverSpec(
        slug="response_only",
        label=OBSERVER_LABELS["response_only"],
        train_bank="observed_response_only",
        test_input="observed_response_only",
        interpretation="Linear feature posterior from observed compact response movies without explicit tau features.",
    ),
    LinearFeatureObserverSpec(
        slug="pose_known_forward_model",
        label=OBSERVER_LABELS["pose_known_forward_model"],
        train_bank="feature_conditioned_forward_model",
        test_input="pose_known_forward_model",
        interpretation=(
            "Compact-latent forward diagnostic: infer z by matching fitted F(z, recorded tau) "
            "to the observed compact response. This is not a known-pose ceiling unless the fitted "
            "F(z, tau) contract is separately validated."
        ),
    ),
    LinearFeatureObserverSpec(
        slug="hidden_joint_forward_model",
        label=OBSERVER_LABELS["hidden_joint_forward_model"],
        train_bank="feature_conditioned_forward_model",
        test_input="hidden_joint_forward_model",
        interpretation="Forward-model latent observer: alternate z and tau MAP updates under the empirically calibrated synthetic confined tau prior.",
    ),
    LinearFeatureObserverSpec(
        slug="estimated_tau_forward_model",
        label=OBSERVER_LABELS["estimated_tau_forward_model"],
        train_bank="feature_conditioned_forward_model",
        test_input="estimated_tau_forward_model",
        interpretation="Forward-model diagnostic: use the current feature-conditioned tau_hat, then infer z by matching F(z, tau_hat) to the observed compact response.",
    ),
    LinearFeatureObserverSpec(
        slug="zero_tau_forward_model",
        label=OBSERVER_LABELS["zero_tau_forward_model"],
        train_bank="feature_conditioned_forward_model",
        test_input="zero_tau_forward_model",
        interpretation="Forward-model diagnostic: force tau to zero, then infer z from the compact response through F(z, 0).",
    ),
    LinearFeatureObserverSpec(
        slug="feature_conditioned_tau_linear",
        label=OBSERVER_LABELS["feature_conditioned_tau_linear"],
        train_bank="feature_conditioned_response_tau_hat",
        test_input="feature_conditioned_response_tau_hat",
        interpretation="Two-pass linear observer: response-only z0_hat conditions the synthetic-prior tau MAP before the final feature posterior.",
    ),
    LinearFeatureObserverSpec(
        slug="feature_conditioned_tau_interactions",
        label=OBSERVER_LABELS["feature_conditioned_tau_interactions"],
        train_bank="feature_conditioned_response_tau_hat_interactions",
        test_input="feature_conditioned_response_tau_hat_interactions",
        interpretation="Two-pass linear observer with response, feature-conditioned synthetic-prior tau_hat, and response-by-tau constructed features.",
    ),
    LinearFeatureObserverSpec(
        slug="pose_known_nested_tau_linear",
        label=OBSERVER_LABELS["pose_known_nested_tau_linear"],
        train_bank="pose_known_nested_response_tau",
        test_input="pose_known_nested_response_tau",
        interpretation="Nested pose-known diagnostic: response-only z0_hat plus a recorded-eye residual update with source-disjoint validated shrinkage and explicit zero fallback.",
    ),
    LinearFeatureObserverSpec(
        slug="pose_known_nested_tau_interactions",
        label=OBSERVER_LABELS["pose_known_nested_tau_interactions"],
        train_bank="pose_known_nested_response_tau_interactions",
        test_input="pose_known_nested_response_tau_interactions",
        interpretation="Nested pose-known diagnostic with recorded-eye tau, response-by-tau residual features, validated shrinkage, and explicit response-only fallback.",
    ),
    LinearFeatureObserverSpec(
        slug="pose_known_tau_linear",
        label=OBSERVER_LABELS["pose_known_tau_linear"],
        train_bank="pose_known_response_tau",
        test_input="pose_known_response_tau",
        interpretation=(
            "Recorded-tau residual diagnostic: response-only z0_hat plus a supervised residual "
            "update from the recorded eye trace. This cheap correction is not a theoretical ceiling."
        ),
    ),
    LinearFeatureObserverSpec(
        slug="pose_known_tau_interactions",
        label=OBSERVER_LABELS["pose_known_tau_interactions"],
        train_bank="pose_known_response_tau_interactions",
        test_input="pose_known_response_tau_interactions",
        interpretation=(
            "Recorded-tau residual diagnostic with response-by-tau features. This tests whether a "
            "linear supervised correction can use tau, not the full image-conditioned pose-aware observer."
        ),
    ),
    LinearFeatureObserverSpec(
        slug="synthetic_tau_linear",
        label=OBSERVER_LABELS["synthetic_tau_linear"],
        train_bank="observed_response_tau_hat",
        test_input="observed_response_tau_hat",
        interpretation="Linear feature posterior from observed compact response movies plus candidate-free tau_hat recovered under the synthetic empirical confined prior.",
    ),
    LinearFeatureObserverSpec(
        slug="synthetic_tau_interactions",
        label=OBSERVER_LABELS["synthetic_tau_interactions"],
        train_bank="observed_response_tau_hat_interactions",
        test_input="observed_response_tau_hat_interactions",
        interpretation="Linear feature posterior from response, candidate-free tau_hat, and response-by-tau constructed features.",
    ),
    LinearFeatureObserverSpec(
        slug="true_tau_linear",
        label=OBSERVER_LABELS["true_tau_linear"],
        train_bank="observed_response_true_tau",
        test_input="observed_response_true_tau",
        interpretation=(
            "Same linear feature observer given the recorded eye trace. Treat as a recorded-tau "
            "linear diagnostic, not as a pose-aware ceiling."
        ),
    ),
    LinearFeatureObserverSpec(
        slug="true_tau_interactions",
        label=OBSERVER_LABELS["true_tau_interactions"],
        train_bank="observed_response_true_tau_interactions",
        test_input="observed_response_true_tau_interactions",
        interpretation=(
            "Response-by-recorded-tau interaction diagnostic for tau-estimation loss and linear "
            "conditioning limits."
        ),
    ),
    LinearFeatureObserverSpec(
        slug="zero_static",
        label=OBSERVER_LABELS["zero_static"],
        train_bank="zero_static_response_only",
        test_input="zero_static_response_only",
        interpretation="Stabilized 0x counterfactual response and zero-eye response model.",
    ),
    LinearFeatureObserverSpec(
        slug="known_eye_response_only",
        label=OBSERVER_LABELS["known_eye_response_only"],
        train_bank="known_eye_response_only",
        test_input="observed_response_only",
        interpretation="Known-eye response family applied to observed responses; retained as a response-table calibration reference.",
    ),
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_manifest(
    path: Path,
    *,
    scales: set[float],
    prior_family_filter: set[str],
    skip_tables: int,
    max_tables: int,
) -> pd.DataFrame:
    rows = pd.read_csv(path)
    rows = rows[rows["response_cache_path"].astype(str).str.len() > 0].copy()
    if scales:
        rows = rows[rows["scale"].astype(float).isin(scales)].copy()
    if prior_family_filter:
        rows = rows[rows["prior_family"].astype(str).isin(prior_family_filter)].copy()
    if int(skip_tables) > 0:
        rows = rows.iloc[int(skip_tables) :].copy()
    if int(max_tables) > 0:
        rows = rows.iloc[: int(max_tables)].copy()
    rows = rows.reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"No response tables selected from {path}")
    rows["table_index"] = np.arange(rows.shape[0], dtype=int)
    return rows


def _parse_csv_values(text: str | None) -> set[str]:
    return {part.strip() for part in str(text or "").split(",") if part.strip()}


def _selected_specs(observer_modes: list[str]) -> list[LinearFeatureObserverSpec]:
    requested = list(observer_modes) if observer_modes else list(DEFAULT_OBSERVER_MODES)
    by_slug = {spec.slug: spec for spec in SPECS}
    missing = sorted(set(requested).difference(by_slug))
    if missing:
        valid = ", ".join(spec.slug for spec in SPECS)
        raise ValueError(f"Unknown observer mode(s) {missing}; valid modes: {valid}")
    return [by_slug[slug] for slug in requested]


def _new_sample_parts() -> dict[str, dict[str, list[Any]]]:
    return {
        name: {"x": [], "source_rows": [], "table_indices": []}
        for name in [
            "observed_response_only",
            "observed_response_tau_hat",
            "observed_response_tau_hat_interactions",
            "observed_response_true_tau",
            "observed_response_true_tau_interactions",
            "zero_static_response_only",
            "known_eye_response_only",
        ]
    }


def _append_sample(
    parts: dict[str, list[Any]],
    *,
    x: np.ndarray,
    source_row: int,
    table_index: int,
) -> None:
    parts["x"].append(np.asarray(x, dtype=np.float32))
    parts["source_rows"].append(int(source_row))
    parts["table_indices"].append(int(table_index))


def _bank_from_parts(parts: dict[str, list[Any]]) -> SampleBank:
    if not parts["x"]:
        raise ValueError("empty sample bank")
    return SampleBank(
        x=np.stack(parts["x"], axis=0).astype(np.float32),
        source_rows=np.asarray(parts["source_rows"], dtype=int),
        table_indices=np.asarray(parts["table_indices"], dtype=int),
    )


def _compact_matrix(response_counts: np.ndarray, basis: np.ndarray) -> np.ndarray:
    response = np.asarray(response_counts, dtype=np.float64)
    if response.ndim != 2:
        raise ValueError(f"response must be (time, unit), got {response.shape}")
    if response.shape[1] != basis.shape[0]:
        raise ValueError(f"response unit count {response.shape[1]} does not match basis {basis.shape}")
    if not np.isfinite(response).all():
        raise ValueError("response contains non-finite values")
    return (response @ basis).astype(np.float32, copy=False)


def _response_vector(response_compact: np.ndarray) -> np.ndarray:
    arr = np.asarray(response_compact, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"compact response must be (time, basis), got {arr.shape}")
    return arr.reshape(-1).astype(np.float32, copy=False)


def _linear_tau_features(tau: np.ndarray, n_time: int) -> np.ndarray:
    arr = np.asarray(tau, dtype=np.float64)
    if arr.shape != (int(n_time), 2):
        raise ValueError(f"tau must be ({n_time}, 2), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("tau contains non-finite values")
    velocity = np.diff(arr, axis=0, prepend=arr[:1])
    return np.concatenate([arr.reshape(-1), velocity.reshape(-1)]).astype(np.float32)


def _response_tau_interactions(response_compact: np.ndarray, tau: np.ndarray) -> np.ndarray:
    response = np.asarray(response_compact, dtype=np.float64)
    arr = np.asarray(tau, dtype=np.float64)
    if response.ndim != 2 or arr.shape != (response.shape[0], 2):
        raise ValueError(f"response/tau shape mismatch: {response.shape}, {arr.shape}")
    if not np.isfinite(response).all() or not np.isfinite(arr).all():
        raise ValueError("response/tau contains non-finite values")
    interactions = np.concatenate(
        [
            (response * arr[:, 0:1]).reshape(-1),
            (response * arr[:, 1:2]).reshape(-1),
        ]
    )
    return interactions.astype(np.float32)


def _feature_vector(response_compact: np.ndarray, tau: np.ndarray | None, *, mode: str) -> np.ndarray:
    base = _response_vector(response_compact)
    if mode == "response_only":
        return base
    if tau is None:
        raise ValueError(f"{mode} requires tau")
    tau_values = _linear_tau_features(tau, response_compact.shape[0])
    if mode == "response_tau_linear":
        return np.concatenate([base, tau_values]).astype(np.float32)
    if mode == "response_tau_interactions":
        return np.concatenate([base, tau_values, _response_tau_interactions(response_compact, tau)]).astype(np.float32)
    raise ValueError(f"Unknown feature-vector mode {mode!r}")


def _nan_feature_vector(response_compact: np.ndarray, *, mode: str) -> np.ndarray:
    base = _response_vector(response_compact)
    n_time = int(np.asarray(response_compact).shape[0])
    tau_dim = 4 * n_time
    if mode == "response_tau_linear":
        return np.full(base.shape[0] + tau_dim, np.nan, dtype=np.float32)
    if mode == "response_tau_interactions":
        interaction_dim = 2 * base.shape[0]
        return np.full(base.shape[0] + tau_dim + interaction_dim, np.nan, dtype=np.float32)
    raise ValueError(f"Unknown NaN feature-vector mode {mode!r}")


def _quadratic_design_from_path(tau: np.ndarray, *, include_intercept: bool = False) -> np.ndarray:
    arr = np.asarray(tau, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"tau must be (time, 2), got {arr.shape}")
    x = arr[:, 0]
    y = arr[:, 1]
    terms = []
    if bool(include_intercept):
        terms.append(np.ones_like(x))
    terms.extend([x, y, x * x, x * y, y * y])
    return np.stack(terms, axis=1)


def _unique_trajectory_samples(trajectory_xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(trajectory_xy, dtype=np.float64)
    if xy.ndim == 4:
        samples = xy.reshape(-1, xy.shape[-2], 2)
    elif xy.ndim == 3:
        samples = xy
    else:
        raise ValueError(f"trajectory_xy must have shape (*, time, 2), got {xy.shape}")
    flat = np.round(samples.reshape(samples.shape[0], -1), decimals=10)
    _unique_flat, unique_indices = np.unique(flat, axis=0, return_index=True)
    return samples[np.sort(unique_indices)].astype(np.float64, copy=False)


def _fit_pooled_linear_observation_map(
    *,
    prior_lambda_counts: np.ndarray,
    zero_lambda_counts: np.ndarray,
    trajectory_xy: np.ndarray,
    basis: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    prior = np.asarray(prior_lambda_counts, dtype=np.float64)
    zero = np.asarray(zero_lambda_counts, dtype=np.float64)
    xy = np.asarray(trajectory_xy, dtype=np.float64)
    z = project_response_delta(prior - zero[:, None, :, :], basis)
    x = xy.reshape(-1, 2)
    y = z.reshape(-1, z.shape[-1])
    good = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    if int(np.sum(good)) < 2:
        raise ValueError("too few finite samples to fit pooled linear observation map")
    x = x[good]
    y = y[good]
    ridge_val = float(ridge)
    normal = x.T @ x + ridge_val * np.eye(2, dtype=np.float64)
    coef = np.linalg.solve(normal, x.T @ y).T
    pred = x @ coef.T
    residual = y - pred
    residual_var = float(np.mean(residual * residual))
    row = {
        "qc_type": "pooled_linear_observation_map",
        "observation_model": "pooled_linear_candidate_free",
        "n_fit_samples": int(x.shape[0]),
        "basis_dim": int(coef.shape[0]),
        "ridge": ridge_val,
        "residual_variance": residual_var,
        "B_fro_norm": float(np.linalg.norm(coef)),
    }
    return coef, residual_var, row


def _fit_pooled_quadratic_observation_map(
    *,
    prior_lambda_counts: np.ndarray,
    zero_lambda_counts: np.ndarray,
    trajectory_xy: np.ndarray,
    basis: np.ndarray,
    ridge: float,
    include_intercept: bool,
    intercept_ridge_multiplier: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    prior = np.asarray(prior_lambda_counts, dtype=np.float64)
    zero = np.asarray(zero_lambda_counts, dtype=np.float64)
    xy = np.asarray(trajectory_xy, dtype=np.float64)
    z = project_response_delta(prior - zero[:, None, :, :], basis)
    design_dim = 6 if bool(include_intercept) else 5
    design = np.stack(
        [
            _quadratic_design_from_path(xy.reshape(-1, xy.shape[-2], 2)[index], include_intercept=include_intercept)
            for index in range(xy.reshape(-1, xy.shape[-2], 2).shape[0])
        ],
        axis=0,
    )
    x = design.reshape(-1, design_dim)
    y = z.reshape(-1, z.shape[-1])
    good = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    if int(np.sum(good)) < design_dim:
        raise ValueError("too few finite samples to fit pooled quadratic observation map")
    x = x[good]
    y = y[good]
    ridge_diag = float(ridge) * np.ones(design_dim, dtype=np.float64)
    if include_intercept:
        ridge_diag[0] *= float(intercept_ridge_multiplier)
    coef = np.linalg.solve(x.T @ x + np.diag(ridge_diag), x.T @ y).T
    pred = x @ coef.T
    residual = y - pred
    residual_var = float(np.mean(residual * residual))
    row = {
        "qc_type": "pooled_quadratic_observation_map",
        "observation_model": "pooled_quadratic_candidate_free",
        "n_fit_samples": int(x.shape[0]),
        "basis_dim": int(coef.shape[0]),
        "quadratic_include_intercept": bool(include_intercept),
        "quadratic_intercept_ridge_multiplier": float(intercept_ridge_multiplier),
        "ridge": float(ridge),
        "residual_variance": residual_var,
        "B_fro_norm": float(np.linalg.norm(coef)),
    }
    return coef, residual_var, row


def _candidate_free_tau_hat(
    *,
    observed_counts: np.ndarray,
    prior_lambda_counts: np.ndarray,
    zero_lambda_counts: np.ndarray,
    trajectory_xy: np.ndarray,
    basis: np.ndarray,
    continuous_args: argparse.Namespace,
    table_index: int,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    prior = np.asarray(prior_lambda_counts, dtype=np.float64)
    zero = np.asarray(zero_lambda_counts, dtype=np.float64)
    observed = np.asarray(observed_counts, dtype=np.float64)
    n_candidates, _n_traj, n_time, _n_units = prior.shape
    pooled_zero = np.mean(zero, axis=0)
    z_obs = project_response_delta(observed - pooled_zero, basis)
    source_samples = _unique_trajectory_samples(trajectory_xy)
    process_model = str(continuous_args.trajectory_process_model)
    if process_model != "synthetic_empirical_confined":
        raise ValueError(
            "build_panel_c_linear_synthetic_prior_feature_observer currently expects "
            "--trajectory-process-model synthetic_empirical_confined for candidate-free tau_hat"
        )
    confined_prior, synthetic_samples = _synthetic_empirical_confined_process_prior(
        source_samples,
        n_traces=int(continuous_args.synthetic_prior_samples),
        n_frames=int(n_time),
        seed=int(continuous_args.synthetic_prior_seed) + 1_000_003 * int(table_index),
        kappa_weight_power=float(continuous_args.synthetic_prior_kappa_weight_power),
        covariance_floor=float(continuous_args.brownian_cov_floor),
        process_cov_scale=float(continuous_args.brownian_cov_scale),
    )
    include_intercept = str(continuous_args.continuous_score_mode) in {
        "quadratic_affine_poisson_profile",
        "quadratic_prior_mean_poisson_profile",
    }
    fit_rows: list[dict[str, Any]] = []
    if str(continuous_args.continuous_score_mode) in {
        "quadratic_poisson_profile",
        "quadratic_affine_poisson_profile",
        "quadratic_prior_mean_poisson_profile",
    }:
        coef, residual_var, fit_row = _fit_pooled_quadratic_observation_map(
            prior_lambda_counts=prior,
            zero_lambda_counts=zero,
            trajectory_xy=trajectory_xy,
            basis=basis,
            ridge=float(continuous_args.continuous_ridge),
            include_intercept=include_intercept,
            intercept_ridge_multiplier=float(continuous_args.quadratic_intercept_ridge_multiplier),
        )
        fit_rows.append(fit_row)
        observation_var = (
            float(continuous_args.observation_var)
            if continuous_args.observation_var is not None
            else float(residual_var)
        )
        starts = [
            np.zeros((n_time, 2), dtype=np.float64),
            np.mean(source_samples, axis=0),
            np.broadcast_to(confined_prior.init_mean[None, :], (n_time, 2)).copy(),
        ]
        out = quadratic_profile_log_score(
            z_obs,
            coef,
            starts=starts,
            observation_var=max(observation_var, float(continuous_args.observation_var_floor)),
            alpha=1.0,
            process_var=float(continuous_args.process_var),
            confined_step_prior=confined_prior,
            max_iter=int(continuous_args.quadratic_optimizer_max_iter),
            quadratic_scales=[float(value) for value in str(continuous_args.quadratic_continuation_scales).split(",") if value],
            observation_scales=[float(value) for value in str(continuous_args.quadratic_observation_scales).split(",") if value],
            intercept_scale=float(continuous_args.quadratic_affine_intercept_scale),
        )
        tau_hat = np.asarray(out["map_means"], dtype=np.float64)
        fit_rows.append(
            {
                "qc_type": "pooled_quadratic_profile_optimizer",
                "observation_model": "pooled_quadratic_candidate_free",
                "optimizer_success": bool(out.get("optimizer_success", False)),
                "optimizer_iterations": int(out.get("optimizer_iterations", -1)),
                "optimizer_start_index": int(out.get("optimizer_start_index", -1)),
                "profile_energy": float(out.get("profile_energy", np.nan)),
            }
        )
    else:
        coef, residual_var, fit_row = _fit_pooled_linear_observation_map(
            prior_lambda_counts=prior,
            zero_lambda_counts=zero,
            trajectory_xy=trajectory_xy,
            basis=basis,
            ridge=float(continuous_args.continuous_ridge),
        )
        fit_rows.append(fit_row)
        observation_var = (
            float(continuous_args.observation_var)
            if continuous_args.observation_var is not None
            else float(residual_var)
        )
        out = ar1_profile_log_score(
            z_obs,
            coef,
            alpha=1.0,
            process_var=float(continuous_args.process_var),
            observation_var=max(observation_var, float(continuous_args.observation_var_floor)),
            confined_step_prior=confined_prior,
        )
        tau_hat = np.asarray(out["map_means"], dtype=np.float64)
    meta = {
        "tau_hat_source": "candidate_free_pooled_response_geometry_full_path_map",
        "observation_geometry_source": "pooled_response_table_local_response_vs_displacement_map",
        "trajectory_process_model": "synthetic_empirical_confined",
        "synthetic_prior_samples": int(continuous_args.synthetic_prior_samples),
        "synthetic_prior_generated_trace_count": int(synthetic_samples.shape[0]),
        "synthetic_prior_source_model": str(confined_prior.metadata.get("synthetic_prior_source_model", "")),
        "synthetic_prior_empirical_param_count": int(
            confined_prior.metadata.get("synthetic_prior_empirical_param_count", -1)
        ),
        "synthetic_prior_sampled_beta_median": float(
            confined_prior.metadata.get("synthetic_prior_sampled_beta_median", np.nan)
        ),
        "synthetic_prior_sampled_kappa_median": float(
            confined_prior.metadata.get("synthetic_prior_sampled_kappa_median", np.nan)
        ),
        "synthetic_prior_fit_beta": float(confined_prior.metadata.get("fit_confined_beta", np.nan)),
        "synthetic_prior_fit_kappa": float(confined_prior.metadata.get("fit_confined_kappa", np.nan)),
        "synthetic_prior_fit_n_samples": int(confined_prior.metadata.get("fit_confined_n_samples", -1)),
        "pooled_observation_candidate_count": int(n_candidates),
        "pooled_observation_residual_variance": float(residual_var),
    }
    return tau_hat, meta, fit_rows


def _continuous_args_for_scale(continuous_args: argparse.Namespace, scale: float) -> argparse.Namespace:
    ridge_by_scale = _parse_scale_value_map(
        str(continuous_args.continuous_ridge_by_scale),
        value_type=float,
        name="continuous_ridge_by_scale",
    )
    process_model_by_scale = _parse_scale_value_map(
        str(continuous_args.trajectory_process_model_by_scale),
        value_type=str,
        name="trajectory_process_model_by_scale",
    )
    brownian_scale_by_scale = _parse_scale_value_map(
        str(continuous_args.brownian_cov_scale_by_scale),
        value_type=float,
        name="brownian_cov_scale_by_scale",
    )
    table_args = argparse.Namespace(**vars(continuous_args))
    table_args.continuous_ridge = float(_scale_value(ridge_by_scale, scale, float(continuous_args.continuous_ridge)))
    table_args.trajectory_process_model = _scale_string_value(
        process_model_by_scale,
        scale,
        str(continuous_args.trajectory_process_model),
    )
    table_args.brownian_cov_scale = float(
        _scale_value(brownian_scale_by_scale, scale, float(continuous_args.brownian_cov_scale))
    )
    return table_args


def _context_from_z(z: np.ndarray) -> np.ndarray:
    arr = np.asarray(z, dtype=np.float64).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError("feature-conditioning vector contains non-finite values")
    return np.concatenate([np.ones(1, dtype=np.float64), arr])


def _geometry_context_from_z(z: np.ndarray, observation_scale: float) -> np.ndarray:
    arr = np.asarray(z, dtype=np.float64).reshape(-1)
    scale = float(observation_scale)
    if not np.isfinite(arr).all() or not np.isfinite(scale):
        raise ValueError("feature/scale conditioning vector contains non-finite values")
    return np.concatenate(
        [
            np.ones(1, dtype=np.float64),
            np.asarray([scale], dtype=np.float64),
            arr,
            scale * arr,
        ]
    )


def _source_context_lookup(
    *,
    transform: FeatureTransform,
    feature_table: FeatureTable,
    sources: np.ndarray,
) -> dict[int, np.ndarray]:
    unique_sources = np.asarray(sorted(set(int(value) for value in np.asarray(sources).reshape(-1))), dtype=int)
    z = _transform_feature_sources(transform, feature_table, unique_sources)
    return {int(source): _context_from_z(z[index]) for index, source in enumerate(unique_sources.tolist())}


def _fit_feature_conditioned_baseline(
    *,
    geometry_tables: list[GeometryTable],
    transform: FeatureTransform,
    feature_table: FeatureTable,
    heldout_sources: set[int],
    ridge: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    train_sources = np.concatenate(
        [
            table.source_rows[
                np.asarray([int(source) not in heldout_sources for source in table.source_rows], dtype=bool)
            ]
            for table in geometry_tables
        ],
        axis=0,
    )
    contexts = _source_context_lookup(transform=transform, feature_table=feature_table, sources=train_sources)
    context_dim = int(transform.feature_dim) + 1
    response_shape = geometry_tables[0].zero_compact.shape[1:]
    response_dim = int(np.prod(response_shape))
    normal = np.zeros((context_dim, context_dim), dtype=np.float64)
    rhs = np.zeros((context_dim, response_dim), dtype=np.float64)
    n_fit = 0
    for table in geometry_tables:
        for candidate_index, source in enumerate(table.source_rows.tolist()):
            source_int = int(source)
            if source_int in heldout_sources:
                continue
            x = contexts[source_int]
            y = np.asarray(table.zero_compact[candidate_index], dtype=np.float64).reshape(-1)
            if not np.isfinite(y).all():
                continue
            normal += np.outer(x, x)
            rhs += np.outer(x, y)
            n_fit += 1
    if n_fit <= context_dim:
        raise ValueError("too few source-disjoint zero-response samples for feature-conditioned baseline")
    coef = np.linalg.solve(normal + float(ridge) * np.eye(context_dim, dtype=np.float64), rhs)

    sse = 0.0
    n_values = 0
    for table in geometry_tables:
        for candidate_index, source in enumerate(table.source_rows.tolist()):
            source_int = int(source)
            if source_int in heldout_sources:
                continue
            x = contexts[source_int]
            y = np.asarray(table.zero_compact[candidate_index], dtype=np.float64).reshape(-1)
            if not np.isfinite(y).all():
                continue
            resid = y - x @ coef
            sse += float(np.sum(resid * resid))
            n_values += int(resid.size)
    residual_var = float(sse / max(n_values, 1))
    row = {
        "qc_type": "feature_conditioned_zero_baseline",
        "observation_model": "linear_zero_compact_from_response_only_feature_z",
        "n_fit_samples": int(n_fit),
        "feature_dim": int(transform.feature_dim),
        "response_dim": int(response_dim),
        "ridge": float(ridge),
        "residual_variance": residual_var,
        "coef_fro_norm": float(np.linalg.norm(coef)),
    }
    return coef, residual_var, row


def _fit_feature_conditioned_quadratic_observation_map(
    *,
    geometry_tables: list[GeometryTable],
    transform: FeatureTransform,
    feature_table: FeatureTable,
    heldout_sources: set[int],
    ridge: float,
    include_intercept: bool,
    intercept_ridge_multiplier: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    train_sources = np.concatenate(
        [
            table.source_rows[
                np.asarray([int(source) not in heldout_sources for source in table.source_rows], dtype=bool)
            ]
            for table in geometry_tables
        ],
        axis=0,
    )
    contexts = _source_context_lookup(transform=transform, feature_table=feature_table, sources=train_sources)
    context_dim = 2 + 2 * int(transform.feature_dim)
    design_dim = 6 if bool(include_intercept) else 5
    conditioned_dim = context_dim * design_dim
    basis_dim = int(geometry_tables[0].prior_delta_compact.shape[-1])
    normal = np.zeros((conditioned_dim, conditioned_dim), dtype=np.float64)
    rhs = np.zeros((conditioned_dim, basis_dim), dtype=np.float64)
    n_fit = 0
    for table in geometry_tables:
        for candidate_index, source in enumerate(table.source_rows.tolist()):
            source_int = int(source)
            if source_int in heldout_sources:
                continue
            context = _geometry_context_from_z(contexts[source_int][1:], float(table.observation_scale))
            xy = np.asarray(table.trajectory_xy[candidate_index], dtype=np.float64)
            z = np.asarray(table.prior_delta_compact[candidate_index], dtype=np.float64)
            paths = xy.reshape(-1, xy.shape[-2], 2)
            design = np.stack(
                [
                    _quadratic_design_from_path(path, include_intercept=include_intercept)
                    for path in paths
                ],
                axis=0,
            ).reshape(-1, design_dim)
            y = z.reshape(-1, basis_dim)
            x = (design[:, :, None] * context[None, None, :]).reshape(-1, conditioned_dim)
            good = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
            if not np.any(good):
                continue
            x = x[good]
            y = y[good]
            normal += x.T @ x
            rhs += x.T @ y
            n_fit += int(x.shape[0])
    if n_fit <= conditioned_dim:
        raise ValueError("too few source-disjoint samples for feature-conditioned quadratic geometry")
    ridge_diag = float(ridge) * np.ones(conditioned_dim, dtype=np.float64)
    if include_intercept:
        ridge_diag[:context_dim] *= float(intercept_ridge_multiplier)
    coef = np.linalg.solve(normal + np.diag(ridge_diag), rhs).T

    sse = 0.0
    n_values = 0
    for table in geometry_tables:
        for candidate_index, source in enumerate(table.source_rows.tolist()):
            source_int = int(source)
            if source_int in heldout_sources:
                continue
            context = _geometry_context_from_z(contexts[source_int][1:], float(table.observation_scale))
            xy = np.asarray(table.trajectory_xy[candidate_index], dtype=np.float64)
            z = np.asarray(table.prior_delta_compact[candidate_index], dtype=np.float64)
            paths = xy.reshape(-1, xy.shape[-2], 2)
            design = np.stack(
                [
                    _quadratic_design_from_path(path, include_intercept=include_intercept)
                    for path in paths
                ],
                axis=0,
            ).reshape(-1, design_dim)
            y = z.reshape(-1, basis_dim)
            x = (design[:, :, None] * context[None, None, :]).reshape(-1, conditioned_dim)
            good = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
            if not np.any(good):
                continue
            resid = y[good] - x[good] @ coef.T
            sse += float(np.sum(resid * resid))
            n_values += int(resid.size)
    residual_var = float(sse / max(n_values, 1))
    row = {
        "qc_type": "feature_conditioned_quadratic_observation_map",
        "observation_model": "quadratic_compact_displacement_from_tau_and_response_only_feature_z",
        "n_fit_samples": int(n_fit),
        "basis_dim": int(basis_dim),
        "feature_dim": int(transform.feature_dim),
        "geometry_context": "intercept+scale+z+scale_times_z",
        "conditioned_design_dim": int(conditioned_dim),
        "quadratic_include_intercept": bool(include_intercept),
        "quadratic_intercept_ridge_multiplier": float(intercept_ridge_multiplier),
        "ridge": float(ridge),
        "residual_variance": residual_var,
        "B_fro_norm": float(np.linalg.norm(coef)),
    }
    return coef, residual_var, row


def _conditioned_quadratic_coefficients(
    coef: np.ndarray,
    z: np.ndarray,
    *,
    observation_scale: float,
    include_intercept: bool,
) -> np.ndarray:
    context = _geometry_context_from_z(z, float(observation_scale))
    design_dim = 6 if bool(include_intercept) else 5
    tensor = np.asarray(coef, dtype=np.float64).reshape(coef.shape[0], design_dim, context.size)
    return np.einsum("kdc,c->kd", tensor, context)


def _predict_feature_conditioned_baseline(
    baseline_coef: np.ndarray,
    z: np.ndarray,
    *,
    n_time: int,
    basis_dim: int,
) -> np.ndarray:
    context = _context_from_z(z)
    flat = context @ np.asarray(baseline_coef, dtype=np.float64)
    return flat.reshape(int(n_time), int(basis_dim))


def _feature_conditioned_tau_hat(
    *,
    observed_compact: np.ndarray,
    z0_hat: np.ndarray,
    geometry_table: GeometryTable,
    baseline_coef: np.ndarray,
    baseline_residual_var: float,
    observation_coef: np.ndarray,
    observation_residual_var: float,
    include_intercept: bool,
    continuous_args: argparse.Namespace,
    table_index: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    process_model = str(continuous_args.trajectory_process_model)
    if process_model != "synthetic_empirical_confined":
        raise ValueError("feature-conditioned tau currently expects synthetic_empirical_confined")
    if str(continuous_args.continuous_score_mode) not in {
        "quadratic_poisson_profile",
        "quadratic_affine_poisson_profile",
        "quadratic_prior_mean_poisson_profile",
    }:
        raise ValueError("feature-conditioned tau currently expects a quadratic continuous score mode")
    compact = np.asarray(observed_compact, dtype=np.float64)
    n_time, basis_dim = compact.shape
    baseline = _predict_feature_conditioned_baseline(
        baseline_coef,
        z0_hat,
        n_time=int(n_time),
        basis_dim=int(basis_dim),
    )
    z_obs = compact - baseline
    confined_prior, synthetic_samples = _synthetic_empirical_confined_process_prior(
        np.asarray(geometry_table.source_samples, dtype=np.float64),
        n_traces=int(continuous_args.synthetic_prior_samples),
        n_frames=int(n_time),
        seed=int(continuous_args.synthetic_prior_seed) + 1_000_003 * int(table_index) + 97_003,
        kappa_weight_power=float(continuous_args.synthetic_prior_kappa_weight_power),
        covariance_floor=float(continuous_args.brownian_cov_floor),
        process_cov_scale=float(continuous_args.brownian_cov_scale),
    )
    conditioned_coef = _conditioned_quadratic_coefficients(
        observation_coef,
        z0_hat,
        observation_scale=float(geometry_table.observation_scale),
        include_intercept=include_intercept,
    )
    observation_var = (
        float(continuous_args.observation_var)
        if continuous_args.observation_var is not None
        else float(observation_residual_var) + float(baseline_residual_var)
    )
    starts = [
        np.zeros((n_time, 2), dtype=np.float64),
        np.mean(np.asarray(geometry_table.source_samples, dtype=np.float64), axis=0),
        np.broadcast_to(confined_prior.init_mean[None, :], (n_time, 2)).copy(),
    ]
    out = quadratic_profile_log_score(
        z_obs,
        conditioned_coef,
        starts=starts,
        observation_var=max(observation_var, float(continuous_args.observation_var_floor)),
        alpha=1.0,
        process_var=float(continuous_args.process_var),
        confined_step_prior=confined_prior,
        max_iter=int(continuous_args.quadratic_optimizer_max_iter),
        quadratic_scales=[float(value) for value in str(continuous_args.quadratic_continuation_scales).split(",") if value],
        observation_scales=[float(value) for value in str(continuous_args.quadratic_observation_scales).split(",") if value],
        intercept_scale=float(continuous_args.quadratic_affine_intercept_scale),
    )
    tau_hat = np.asarray(out["map_means"], dtype=np.float64)
    meta = {
        "tau_hat_source": "response_only_feature_conditioned_synthetic_prior_full_path_map",
        "observation_geometry_source": (
            "source_disjoint_scale_and_feature_conditioned_response_table_local_response_vs_displacement_map"
        ),
        "trajectory_process_model": "synthetic_empirical_confined",
        "synthetic_prior_samples": int(continuous_args.synthetic_prior_samples),
        "synthetic_prior_generated_trace_count": int(synthetic_samples.shape[0]),
        "synthetic_prior_source_model": str(confined_prior.metadata.get("synthetic_prior_source_model", "")),
        "synthetic_prior_empirical_param_count": int(
            confined_prior.metadata.get("synthetic_prior_empirical_param_count", -1)
        ),
        "synthetic_prior_fit_beta": float(confined_prior.metadata.get("fit_confined_beta", np.nan)),
        "synthetic_prior_fit_kappa": float(confined_prior.metadata.get("fit_confined_kappa", np.nan)),
        "feature_conditioned_baseline_residual_variance": float(baseline_residual_var),
        "feature_conditioned_observation_residual_variance": float(observation_residual_var),
        "feature_conditioned_observation_variance": float(max(observation_var, float(continuous_args.observation_var_floor))),
        "optimizer_success": bool(out.get("optimizer_success", False)),
        "optimizer_iterations": int(out.get("optimizer_iterations", -1)),
        "optimizer_start_index": int(out.get("optimizer_start_index", -1)),
        "profile_energy": float(out.get("profile_energy", np.nan)),
    }
    return tau_hat, meta


def _forward_model_observation_var(
    *,
    baseline_residual_var: float,
    observation_residual_var: float,
    continuous_args: argparse.Namespace,
) -> float:
    if continuous_args.observation_var is not None:
        value = float(continuous_args.observation_var)
    else:
        value = float(baseline_residual_var) + float(observation_residual_var)
    return float(max(value, float(continuous_args.observation_var_floor)))


def _forward_model_z_design_for_tau(
    *,
    tau: np.ndarray,
    observation_scale: float,
    baseline_coef: np.ndarray,
    observation_coef: np.ndarray,
    include_intercept: bool,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    path = np.asarray(tau, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or not np.isfinite(path).all():
        raise ValueError(f"tau must be finite with shape (time, 2), got {path.shape}")
    baseline = np.asarray(baseline_coef, dtype=np.float64)
    if baseline.ndim != 2 or baseline.shape[0] < 2:
        raise ValueError(f"baseline_coef must be (feature_dim + 1, response_dim), got {baseline.shape}")
    z_dim = int(baseline.shape[0] - 1)
    n_time = int(path.shape[0])
    response_dim = int(baseline.shape[1])
    if response_dim % n_time != 0:
        raise ValueError(f"response_dim {response_dim} is not divisible by n_time {n_time}")
    basis_dim = int(response_dim // n_time)
    design_dim = 6 if bool(include_intercept) else 5
    context_dim = 2 + 2 * z_dim
    obs_coef = np.asarray(observation_coef, dtype=np.float64)
    if obs_coef.shape != (basis_dim, design_dim * context_dim):
        raise ValueError(
            "observation_coef shape mismatch for forward z solve: "
            f"expected {(basis_dim, design_dim * context_dim)}, got {obs_coef.shape}"
        )

    offset = baseline[0].copy()
    z_design = baseline[1:].T.copy()
    q = _quadratic_design_from_path(path, include_intercept=include_intercept)
    tensor = obs_coef.reshape(basis_dim, design_dim, context_dim)
    scale = float(observation_scale)
    motion_const_coef = tensor[:, :, 0] + scale * tensor[:, :, 1]
    motion_z_coef = tensor[:, :, 2 : 2 + z_dim] + scale * tensor[:, :, 2 + z_dim :]
    motion_const = np.einsum("td,kd->tk", q, motion_const_coef)
    motion_z = np.einsum("td,kdj->tkj", q, motion_z_coef)
    offset = offset + motion_const.reshape(-1)
    z_design = z_design + motion_z.reshape(-1, z_dim)
    return offset, z_design, (n_time, basis_dim)


def _predict_compact_from_z_tau(
    *,
    z: np.ndarray,
    tau: np.ndarray,
    observation_scale: float,
    baseline_coef: np.ndarray,
    observation_coef: np.ndarray,
    include_intercept: bool,
) -> np.ndarray:
    offset, z_design, shape = _forward_model_z_design_for_tau(
        tau=tau,
        observation_scale=float(observation_scale),
        baseline_coef=baseline_coef,
        observation_coef=observation_coef,
        include_intercept=include_intercept,
    )
    z_arr = np.asarray(z, dtype=np.float64).reshape(-1)
    if z_arr.shape[0] != z_design.shape[1] or not np.isfinite(z_arr).all():
        raise ValueError(f"z must be finite with shape ({z_design.shape[1]},), got {z_arr.shape}")
    return (offset + z_design @ z_arr).reshape(shape)


def _solve_z_given_tau(
    *,
    observed_compact: np.ndarray,
    tau: np.ndarray,
    geometry_table: GeometryTable,
    baseline_coef: np.ndarray,
    baseline_residual_var: float,
    observation_coef: np.ndarray,
    observation_residual_var: float,
    include_intercept: bool,
    continuous_args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    compact = np.asarray(observed_compact, dtype=np.float64)
    if compact.ndim != 2 or not np.isfinite(compact).all():
        raise ValueError(f"observed_compact must be finite with shape (time, basis), got {compact.shape}")
    offset, z_design, shape = _forward_model_z_design_for_tau(
        tau=tau,
        observation_scale=float(geometry_table.observation_scale),
        baseline_coef=baseline_coef,
        observation_coef=observation_coef,
        include_intercept=include_intercept,
    )
    if compact.shape != shape:
        raise ValueError(f"observed_compact shape {compact.shape} does not match forward model shape {shape}")
    y = compact.reshape(-1) - offset
    observation_var = _forward_model_observation_var(
        baseline_residual_var=float(baseline_residual_var),
        observation_residual_var=float(observation_residual_var),
        continuous_args=continuous_args,
    )
    prior_precision = float(getattr(continuous_args, "forward_model_z_prior_precision", 1.0))
    prior_precision = max(prior_precision, 0.0)
    normal = (z_design.T @ z_design) / observation_var
    normal += (prior_precision + 1e-10) * np.eye(z_design.shape[1], dtype=np.float64)
    rhs = (z_design.T @ y) / observation_var
    try:
        z_hat = np.linalg.solve(normal, rhs)
        solver = "solve"
        success = True
    except np.linalg.LinAlgError:
        z_hat = np.linalg.lstsq(normal, rhs, rcond=None)[0]
        solver = "lstsq"
        success = False
    pred_flat = offset + z_design @ z_hat
    resid = compact.reshape(-1) - pred_flat
    residual_mse = float(np.mean(resid * resid))
    energy = 0.5 * float(np.sum(resid * resid)) / observation_var
    energy += 0.5 * prior_precision * float(np.sum(z_hat * z_hat))
    meta = {
        "forward_z_solver": solver,
        "forward_z_solver_success": bool(success),
        "forward_z_prior_precision": float(prior_precision),
        "forward_observation_variance": float(observation_var),
        "forward_response_residual_mse": residual_mse,
        "forward_profile_energy": float(energy),
        "forward_prediction_norm": float(np.linalg.norm(pred_flat)),
        "forward_design_fro_norm": float(np.linalg.norm(z_design)),
        "feature_conditioned_baseline_residual_variance": float(baseline_residual_var),
        "feature_conditioned_observation_residual_variance": float(observation_residual_var),
        "feature_conditioned_observation_variance": float(observation_var),
    }
    return z_hat.astype(np.float64), pred_flat.reshape(shape), meta


def _joint_z_tau_map(
    *,
    observed_compact: np.ndarray,
    initial_z: np.ndarray,
    geometry_table: GeometryTable,
    baseline_coef: np.ndarray,
    baseline_residual_var: float,
    observation_coef: np.ndarray,
    observation_residual_var: float,
    include_intercept: bool,
    continuous_args: argparse.Namespace,
    table_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    z = np.asarray(initial_z, dtype=np.float64).reshape(-1)
    if not np.isfinite(z).all():
        raise ValueError("initial_z must be finite for hidden-joint forward model")
    n_iter = max(1, int(getattr(continuous_args, "forward_model_joint_iterations", 3)))
    previous_z = z.copy()
    tau_hat = np.full((np.asarray(observed_compact).shape[0], 2), np.nan, dtype=np.float64)
    pred = np.full_like(np.asarray(observed_compact, dtype=np.float64), np.nan)
    tau_meta: dict[str, Any] = {}
    z_meta: dict[str, Any] = {}
    z_delta_norm = float("nan")
    for iteration in range(n_iter):
        tau_hat, tau_meta = _feature_conditioned_tau_hat(
            observed_compact=observed_compact,
            z0_hat=z,
            geometry_table=geometry_table,
            baseline_coef=baseline_coef,
            baseline_residual_var=baseline_residual_var,
            observation_coef=observation_coef,
            observation_residual_var=observation_residual_var,
            include_intercept=include_intercept,
            continuous_args=continuous_args,
            table_index=int(table_index),
        )
        z, pred, z_meta = _solve_z_given_tau(
            observed_compact=observed_compact,
            tau=tau_hat,
            geometry_table=geometry_table,
            baseline_coef=baseline_coef,
            baseline_residual_var=baseline_residual_var,
            observation_coef=observation_coef,
            observation_residual_var=observation_residual_var,
            include_intercept=include_intercept,
            continuous_args=continuous_args,
        )
        z_delta_norm = float(np.linalg.norm(z - previous_z))
        previous_z = z.copy()
    meta = dict(tau_meta)
    meta.update(z_meta)
    meta.update(
        {
            "tau_hat_source": "hidden_joint_forward_model_synthetic_prior_alternating_map",
            "observation_geometry_source": (
                "source_disjoint_scale_and_feature_conditioned_forward_response_model"
            ),
            "forward_joint_iterations": int(n_iter),
            "forward_joint_final_z_delta_norm": float(z_delta_norm),
            "forward_joint_initialization": "source_disjoint_response_only_z0_hat",
        }
    )
    return z, tau_hat, pred, meta


def _load_table_with_sidecar(
    *,
    run_dir: Path,
    response_cache_path: str,
    trajectory_sidecar_dir: Path | None,
) -> dict[str, np.ndarray]:
    table = _load_npz(run_dir / response_cache_path)
    if trajectory_sidecar_dir is None or "prior_trajectory_xy" in table:
        return table
    sidecar_path = trajectory_sidecar_dir / response_cache_path
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Missing trajectory sidecar {sidecar_path}")
    sidecar = _load_npz(sidecar_path)
    return {**table, **sidecar}


def _correlation_by_axis(lhs: np.ndarray, rhs: np.ndarray, axis: int) -> float:
    a = np.asarray(lhs, dtype=np.float64)[:, axis]
    b = np.asarray(rhs, dtype=np.float64)[:, axis]
    if a.size < 2 or float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _tau_metrics(tau_hat: np.ndarray, tau_true: np.ndarray) -> dict[str, float]:
    pred = np.asarray(tau_hat, dtype=np.float64)
    true = np.asarray(tau_true, dtype=np.float64)
    if pred.shape != true.shape or not np.isfinite(pred).all() or not np.isfinite(true).all():
        return {
            "trajectory_rmse": float("nan"),
            "trajectory_corr_x": float("nan"),
            "trajectory_corr_y": float("nan"),
            "trajectory_corr_mean": float("nan"),
            "trajectory_r2": float("nan"),
        }
    diff = pred - true
    denom = float(np.sum((true - np.mean(true, axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - float(np.sum(diff * diff)) / denom if denom > 1e-12 else float("nan")
    corr_x = _correlation_by_axis(pred, true, 0)
    corr_y = _correlation_by_axis(pred, true, 1)
    finite_corr = [value for value in [corr_x, corr_y] if np.isfinite(value)]
    return {
        "trajectory_rmse": float(np.sqrt(np.mean(diff * diff))),
        "trajectory_corr_x": corr_x,
        "trajectory_corr_y": corr_y,
        "trajectory_corr_mean": float(np.mean(finite_corr)) if finite_corr else float("nan"),
        "trajectory_r2": r2,
    }


def _build_sample_banks(
    *,
    run_dir: Path,
    manifest: pd.DataFrame,
    basis: np.ndarray,
    feature_sources: set[int],
    trajectory_npz: dict[str, np.ndarray] | None,
    trajectory_key: str,
    observed_trajectory_key: str,
    trajectory_sidecar_dir: Path | None,
    continuous_args: argparse.Namespace,
    compute_pooled_tau_hat: bool,
    progress_every: int,
) -> tuple[dict[str, SampleBank], TestSet, pd.DataFrame]:
    parts = _new_sample_parts()
    test_vectors: dict[str, list[np.ndarray]] = {
        "observed_response_only": [],
        "observed_response_tau_hat": [],
        "observed_response_tau_hat_interactions": [],
        "observed_response_true_tau": [],
        "observed_response_true_tau_interactions": [],
        "zero_static_response_only": [],
    }
    tau_hat_rows: list[np.ndarray] = []
    tau_true_rows: list[np.ndarray] = []
    observed_compact_rows: list[np.ndarray] = []
    geometry_tables: list[GeometryTable] = []
    table_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []

    for local_index, man_row in manifest.iterrows():
        table_index = int(man_row["table_index"])
        if int(progress_every) > 0 and (local_index + 1) % int(progress_every) == 0:
            print(f"processed {local_index + 1} / {manifest.shape[0]} response tables")

        response_cache_path = str(man_row["response_cache_path"])
        table = _load_table_with_sidecar(
            run_dir=run_dir,
            response_cache_path=response_cache_path,
            trajectory_sidecar_dir=trajectory_sidecar_dir,
        )
        prior = np.asarray(table["prior_lambda_counts"], dtype=np.float64)
        known = np.asarray(table["known_lambda_counts"], dtype=np.float64)
        zero = np.asarray(table["zero_lambda_counts"], dtype=np.float64)
        observed = np.asarray(table["y_obs_counts"], dtype=np.float64)
        n_candidates, n_prior_trajectories, n_time, n_units = prior.shape
        if known.shape != (n_candidates, n_time, n_units) or zero.shape != (n_candidates, n_time, n_units):
            raise ValueError(f"Candidate response shapes mismatch in {response_cache_path}")
        if observed.shape != (n_time, n_units):
            raise ValueError(f"Observed response shape mismatch in {response_cache_path}")

        candidate_ids = _candidate_ids(table, n_candidates)
        source_rows = [_source_row_from_candidate_id(candidate_id) for candidate_id in candidate_ids]
        missing = [source for source in source_rows if source not in feature_sources]
        if missing:
            preview = ", ".join(str(value) for value in sorted(set(missing))[:8])
            raise KeyError(f"Missing feature embeddings for source rows {preview} in {response_cache_path}")

        true_idx = _scalar_int(table, "true_candidate_index", 0)
        if true_idx < 0 or true_idx >= n_candidates:
            raise ValueError(f"true_candidate_index {true_idx} outside candidate list in {response_cache_path}")
        true_source = int(source_rows[true_idx])
        true_tau_idx = _scalar_int(table, "true_trajectory_index", -1)
        nearest_tau_idx = _scalar_int(table, "nearest_trajectory_index", -1)

        trajectory_xy_raw = _trajectory_from_table_or_npz(
            table=table,
            trajectory_npz=trajectory_npz,
            trajectory_key=str(trajectory_key),
        )
        trajectory_xy = _trajectory_xy_by_candidate(
            trajectory_xy_raw,
            n_candidates=n_candidates,
            n_trajectories=n_prior_trajectories,
            n_time=n_time,
        )
        observed_xy = _observed_trajectory_from_table_or_npz(
            table=table,
            trajectory_npz=trajectory_npz,
            observed_trajectory_key=str(observed_trajectory_key),
        )
        true_tau: np.ndarray | None
        if observed_xy is not None:
            true_tau = np.asarray(observed_xy, dtype=np.float64).reshape(n_time, 2)
            tau_true_source = "observed_recorded_eye_trace"
        elif 0 <= true_tau_idx < n_prior_trajectories:
            true_tau = trajectory_xy[true_idx, true_tau_idx]
            tau_true_source = "catalog_true_trajectory_index"
        else:
            true_tau = None
            tau_true_source = "missing"

        scale = float(man_row.get("scale", np.nan))
        table_args = _continuous_args_for_scale(continuous_args, scale)
        brownian_scale_for_table = float(table_args.brownian_cov_scale)
        if bool(compute_pooled_tau_hat):
            tau_hat, tau_meta, tau_fit_rows = _candidate_free_tau_hat(
                observed_counts=observed,
                prior_lambda_counts=prior,
                zero_lambda_counts=zero,
                trajectory_xy=trajectory_xy,
                basis=basis,
                continuous_args=table_args,
                table_index=table_index,
            )
        else:
            tau_hat = np.full((n_time, 2), np.nan, dtype=np.float64)
            tau_fit_rows = []
            tau_meta = {
                "tau_hat_source": "candidate_free_pooled_tau_hat_not_requested",
                "observation_geometry_source": "pooled_response_table_geometry_not_requested",
                "trajectory_process_model": str(table_args.trajectory_process_model),
                "synthetic_prior_samples": int(table_args.synthetic_prior_samples),
                "synthetic_prior_generated_trace_count": 0,
                "synthetic_prior_source_model": "",
                "synthetic_prior_empirical_param_count": -1,
                "synthetic_prior_sampled_beta_median": float("nan"),
                "synthetic_prior_sampled_kappa_median": float("nan"),
                "synthetic_prior_fit_beta": float("nan"),
                "synthetic_prior_fit_kappa": float("nan"),
                "pooled_observation_candidate_count": int(n_candidates),
                "pooled_observation_residual_variance": float("nan"),
            }
        for fit_row in tau_fit_rows:
            row = dict(fit_row)
            row.update({"table_index": table_index, "response_cache_path": response_cache_path})
            fit_rows.append(row)

        observed_compact = _compact_matrix(observed, basis)
        known_true_compact = _compact_matrix(known[true_idx], basis)
        zero_true_compact = _compact_matrix(zero[true_idx], basis)
        zero_compact = project_response_delta(zero, basis).astype(np.float32)
        prior_delta_compact = project_response_delta(prior - zero[:, None, :, :], basis).astype(np.float32)
        geometry_tables.append(
            GeometryTable(
                source_rows=np.asarray(source_rows, dtype=int),
                zero_compact=zero_compact,
                prior_delta_compact=prior_delta_compact,
                trajectory_xy=np.asarray(trajectory_xy, dtype=np.float32),
                source_samples=_unique_trajectory_samples(trajectory_xy).astype(np.float32),
                observation_scale=float(scale),
            )
        )
        observed_compact_rows.append(observed_compact.astype(np.float32))
        observed_response_only = _feature_vector(observed_compact, None, mode="response_only")
        if np.isfinite(tau_hat).all():
            observed_response_tau_hat = _feature_vector(observed_compact, tau_hat, mode="response_tau_linear")
            observed_response_tau_hat_interactions = _feature_vector(
                observed_compact,
                tau_hat,
                mode="response_tau_interactions",
            )
        else:
            observed_response_tau_hat = _nan_feature_vector(observed_compact, mode="response_tau_linear")
            observed_response_tau_hat_interactions = _nan_feature_vector(
                observed_compact,
                mode="response_tau_interactions",
            )
        if true_tau is None:
            observed_response_true_tau = _nan_feature_vector(observed_compact, mode="response_tau_linear")
            observed_response_true_tau_interactions = _nan_feature_vector(
                observed_compact,
                mode="response_tau_interactions",
            )
            tau_true_for_metrics = np.full((n_time, 2), np.nan, dtype=np.float64)
        else:
            observed_response_true_tau = _feature_vector(observed_compact, true_tau, mode="response_tau_linear")
            observed_response_true_tau_interactions = _feature_vector(
                observed_compact,
                true_tau,
                mode="response_tau_interactions",
            )
            tau_true_for_metrics = true_tau
        zero_static_response_only = _feature_vector(zero_true_compact, None, mode="response_only")
        known_eye_response_only = _feature_vector(known_true_compact, None, mode="response_only")

        row_vectors = {
            "observed_response_only": observed_response_only,
            "observed_response_tau_hat": observed_response_tau_hat,
            "observed_response_tau_hat_interactions": observed_response_tau_hat_interactions,
            "observed_response_true_tau": observed_response_true_tau,
            "observed_response_true_tau_interactions": observed_response_true_tau_interactions,
            "zero_static_response_only": zero_static_response_only,
        }
        for name, x in row_vectors.items():
            test_vectors[name].append(x)
            _append_sample(parts[name], x=x, source_row=true_source, table_index=table_index)
        _append_sample(
            parts["known_eye_response_only"],
            x=known_eye_response_only,
            source_row=true_source,
            table_index=table_index,
        )
        tau_hat_rows.append(tau_hat.astype(np.float32))
        tau_true_rows.append(tau_true_for_metrics.astype(np.float32))

        recovery = _tau_metrics(tau_hat, tau_true_for_metrics)
        table_row = {
            "table_index": table_index,
            "trial_id": int(man_row.get("trial_id", table_index)),
            "response_cache_path": response_cache_path,
            "candidate_set_mode": str(man_row.get("candidate_set_mode", "")),
            "observation_family": str(man_row.get("observation_family", "")),
            "prior_family": str(man_row.get("prior_family", "")),
            "observation_scale": scale,
            "axis_catalog_mode": str(man_row.get("axis_catalog_mode", "")),
            "n_candidates": int(n_candidates),
            "n_prior_trajectories": int(n_prior_trajectories),
            "n_timebins": int(n_time),
            "n_units": int(n_units),
            "true_candidate_index": int(true_idx),
            "true_source_row": true_source,
            "true_candidate_id": candidate_ids[true_idx],
            "true_trajectory_index": int(true_tau_idx),
            "nearest_trajectory_index": int(nearest_tau_idx),
            "tau_true_source": tau_true_source,
            "continuous_joint_pred_candidate_index": -1,
            "continuous_joint_true_rank": -1,
            "continuous_joint_true_margin": float("nan"),
            "trajectory_rmse": float(recovery.get("trajectory_rmse", np.nan)),
            "trajectory_corr_x": float(recovery.get("trajectory_corr_x", np.nan)),
            "trajectory_corr_y": float(recovery.get("trajectory_corr_y", np.nan)),
            "trajectory_corr_mean": float(recovery.get("trajectory_corr_mean", np.nan)),
            "trajectory_r2": float(recovery.get("trajectory_r2", np.nan)),
            "tau_hat_source": str(tau_meta["tau_hat_source"]),
            "uses_candidate_posterior_endpoint": False,
            "uses_nonlinear_mlp": False,
            "uses_trajectory_catalog_replay_endpoint": False,
            "uses_true_image_conditioned_tau_hat": False,
            "uses_response_table_trajectory_training_rows": False,
            "observation_geometry_source": str(tau_meta["observation_geometry_source"]),
            "trajectory_prior_label": "synthetic_empirical_confined uses an empirically calibrated synthetic FEM prior",
            "continuous_score_mode": str(continuous_args.continuous_score_mode),
            "trajectory_process_model": str(tau_meta["trajectory_process_model"]),
            "brownian_cov_scale": brownian_scale_for_table,
            "synthetic_prior_samples": int(tau_meta["synthetic_prior_samples"]),
            "synthetic_prior_generated_trace_count": int(tau_meta["synthetic_prior_generated_trace_count"]),
            "synthetic_prior_source_model": str(tau_meta["synthetic_prior_source_model"]),
            "synthetic_prior_empirical_param_count": int(tau_meta["synthetic_prior_empirical_param_count"]),
            "synthetic_prior_sampled_beta_median": float(tau_meta["synthetic_prior_sampled_beta_median"]),
            "synthetic_prior_sampled_kappa_median": float(tau_meta["synthetic_prior_sampled_kappa_median"]),
            "synthetic_prior_fit_beta": float(tau_meta["synthetic_prior_fit_beta"]),
            "synthetic_prior_fit_kappa": float(tau_meta["synthetic_prior_fit_kappa"]),
            "pooled_observation_candidate_count": int(tau_meta["pooled_observation_candidate_count"]),
            "pooled_observation_residual_variance": float(tau_meta["pooled_observation_residual_variance"]),
        }
        table_rows.append(table_row)

    banks = {name: _bank_from_parts(value) for name, value in parts.items()}
    tests = TestSet(
        rows=pd.DataFrame(table_rows),
        x_by_mode={name: np.stack(values, axis=0).astype(np.float32) for name, values in test_vectors.items()},
        tau_hat=np.stack(tau_hat_rows, axis=0).astype(np.float32),
        tau_true=np.stack(tau_true_rows, axis=0).astype(np.float32),
        observed_compact=np.stack(observed_compact_rows, axis=0).astype(np.float32),
        geometry_tables=geometry_tables,
    )
    return banks, tests, pd.DataFrame(fit_rows)


def _canonical_feature_modes(feature_space_modes: list[str]) -> list[str]:
    return list(dict.fromkeys(str(_feature_space_config(mode)["canonical_mode"]) for mode in feature_space_modes))


def _fit_transform_for_fold(
    *,
    mode: str,
    fold: int,
    fold_by_source: dict[int, int],
    feature_table: FeatureTable,
    feature_dim: int,
    feature_weights: np.ndarray | None,
    global_transforms: dict[str, FeatureTransform],
) -> FeatureTransform:
    if mode in global_transforms:
        return global_transforms[mode]
    heldout_sources = {int(source) for source, source_fold in fold_by_source.items() if int(source_fold) == int(fold)}
    fold_train_sources = np.asarray(
        [int(source) for source in feature_table.source_rows.tolist() if int(source) not in heldout_sources],
        dtype=int,
    )
    return _fit_feature_transform(
        feature_table,
        fit_sources=fold_train_sources,
        feature_dim=int(feature_dim),
        feature_space_mode=mode,
        feature_weights=feature_weights,
    )


def _inverse_transform_scores(transform: FeatureTransform, scores: np.ndarray) -> np.ndarray:
    """Project locked feature-space scores back to the raw feature coordinates.

    For PCA-reduced spaces this is the raw-coordinate projection inside the
    fitted subspace, not a full reconstruction of discarded feature variance.
    """

    z = np.asarray(scores, dtype=np.float64)
    if z.ndim == 1:
        z = z[None, :]
    if z.ndim != 2 or z.shape[1] != int(transform.feature_dim):
        raise ValueError(f"scores must be (*, {transform.feature_dim}), got {z.shape}")
    projected = (z * transform.denom[None, :]) @ transform.components
    if transform.weights is not None:
        weights = np.asarray(transform.weights, dtype=np.float64)
        projected = projected / np.maximum(weights[None, :], 1e-12)
    raw = projected * transform.sd[None, :] + transform.mean[None, :]
    return raw.astype(np.float64, copy=False)


def _local_field_dim_metadata(
    *,
    latent: str,
    raw_feature_dim: int,
    local_grid: int = 8,
    pyramid_height: int = 4,
    pyramid_order: int = 3,
) -> pd.DataFrame:
    """Return raw feature-dimension metadata for local field targets."""

    latent_name = str(latent)
    channel_names = ("real", "imag", "magnitude")
    if latent_name == "pyramid_local_field":
        band_count = int(pyramid_height)
        orientation_count = int(pyramid_order) + 1
        feature_family = "complex_steerable_pyramid_like"
    elif latent_name == "gabor_local_field":
        band_count = 3
        orientation_count = 8
        feature_family = "gabor_like"
    else:
        return pd.DataFrame(
            {
                "raw_dim": np.arange(int(raw_feature_dim), dtype=int),
                "latent": latent_name,
                "feature_family": "unknown",
                "band": -1,
                "orientation": -1,
                "channel": "unknown",
                "block_index": -1,
                "block_row": -1,
                "block_col": -1,
            }
        )

    block_count = int(local_grid) * int(local_grid)
    rows: list[dict[str, Any]] = []
    raw_dim = 0
    for band in range(band_count):
        for orientation in range(orientation_count):
            for channel in channel_names:
                for block_index in range(block_count):
                    rows.append(
                        {
                            "raw_dim": int(raw_dim),
                            "latent": latent_name,
                            "feature_family": feature_family,
                            "band": int(band),
                            "orientation": int(orientation),
                            "channel": channel,
                            "block_index": int(block_index),
                            "block_row": int(block_index // int(local_grid)),
                            "block_col": int(block_index % int(local_grid)),
                        }
                    )
                    raw_dim += 1
    if raw_dim != int(raw_feature_dim):
        return pd.DataFrame(
            {
                "raw_dim": np.arange(int(raw_feature_dim), dtype=int),
                "latent": latent_name,
                "feature_family": "dimension_mismatch",
                "band": -1,
                "orientation": -1,
                "channel": "unknown",
                "block_index": -1,
                "block_row": -1,
                "block_col": -1,
                "expected_local_field_dim": raw_dim,
            }
        )
    return pd.DataFrame(rows)


def _append_feature_predictions(
    parts: dict[str, list[Any]] | None,
    *,
    tests: TestSet,
    global_indices: np.ndarray | list[int],
    z_hat: np.ndarray,
    z_true: np.ndarray,
    z_train_mean: np.ndarray,
    spec: LinearFeatureObserverSpec,
    transform: FeatureTransform,
    fold: int,
    n_train_samples: int,
    prediction_source: str,
) -> None:
    if parts is None:
        return
    indices = [int(index) for index in np.asarray(global_indices, dtype=int).reshape(-1).tolist()]
    pred = np.asarray(z_hat, dtype=np.float64)
    true = np.asarray(z_true, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred[None, :]
    if true.ndim == 1:
        true = true[None, :]
    if pred.shape != true.shape or pred.shape[0] != len(indices):
        raise ValueError(f"Prediction shape mismatch: pred={pred.shape}, true={true.shape}, indices={len(indices)}")
    train_mean = np.asarray(z_train_mean, dtype=np.float64).reshape(1, -1)
    train_mean_rows = np.repeat(train_mean, pred.shape[0], axis=0)
    raw_hat = _inverse_transform_scores(transform, pred)
    raw_true = _inverse_transform_scores(transform, true)
    raw_train_mean = _inverse_transform_scores(transform, train_mean_rows)
    start = len(parts["rows"])
    for local_index, global_index in enumerate(indices):
        row = dict(tests.rows.iloc[int(global_index)].to_dict())
        row.update(
            {
                "prediction_row": int(start + local_index),
                "decoder_mode": "linear_gaussian",
                "observer_mode": spec.slug,
                "observer_label": spec.label,
                "train_bank": spec.train_bank,
                "test_input": spec.test_input,
                "latent": transform.latent,
                "feature_space_mode": transform.feature_space_mode,
                "feature_fit_scope": transform.fit_scope,
                "feature_preprocessing": transform.preprocessing,
                "feature_whitened": bool(transform.whitened),
                "feature_weighted": bool(transform.weighted),
                "feature_variance_fraction": float(transform.explained_variance_sum),
                "fold": int(fold),
                "n_train_samples": int(n_train_samples),
                "n_fit_sources": int(transform.n_fit_sources),
                "feature_dim": int(transform.feature_dim),
                "raw_feature_dim": int(transform.raw_feature_dim),
                "prediction_source": str(prediction_source),
                "raw_projection_contract": (
                    "inverse_projection_of_locked_feature_scores_through_fold_feature_transform"
                ),
            }
        )
        parts["rows"].append(row)
    parts["z_hat"].extend(pred.astype(np.float32, copy=False))
    parts["z_true"].extend(true.astype(np.float32, copy=False))
    parts["z_train_mean"].extend(train_mean_rows.astype(np.float32, copy=False))
    parts["raw_hat_projected"].extend(raw_hat.astype(np.float32, copy=False))
    parts["raw_true_projected"].extend(raw_true.astype(np.float32, copy=False))
    parts["raw_train_mean_projected"].extend(raw_train_mean.astype(np.float32, copy=False))


def _finalize_feature_predictions(parts: dict[str, list[Any]] | None) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    if parts is None or not parts["rows"]:
        return pd.DataFrame(), {}
    rows = pd.DataFrame(parts["rows"])
    arrays = {
        key: np.stack(parts[key], axis=0).astype(np.float32, copy=False)
        for key in [
            "z_hat",
            "z_true",
            "z_train_mean",
            "raw_hat_projected",
            "raw_true_projected",
            "raw_train_mean_projected",
        ]
    }
    return rows, arrays


def _is_feature_conditioned_spec(spec: LinearFeatureObserverSpec) -> bool:
    return str(spec.slug).startswith("feature_conditioned_tau_")


def _is_forward_model_spec(spec: LinearFeatureObserverSpec) -> bool:
    return str(spec.slug) in {
        "pose_known_forward_model",
        "hidden_joint_forward_model",
        "estimated_tau_forward_model",
        "zero_tau_forward_model",
    }


def _needs_response_only_z0(spec: LinearFeatureObserverSpec) -> bool:
    return _is_residual_update_spec(spec) or str(spec.slug) in {
        "hidden_joint_forward_model",
        "estimated_tau_forward_model",
    }


def _needs_precomputed_feature_conditioned_tau(spec: LinearFeatureObserverSpec) -> bool:
    return _is_feature_conditioned_spec(spec) or str(spec.slug) == "estimated_tau_forward_model"


def _is_pose_known_spec(spec: LinearFeatureObserverSpec) -> bool:
    return str(spec.slug).startswith("pose_known_tau_") or str(spec.slug).startswith("pose_known_nested_tau_")


def _is_pose_known_nested_spec(spec: LinearFeatureObserverSpec) -> bool:
    return str(spec.slug).startswith("pose_known_nested_tau_")


def _is_residual_update_spec(spec: LinearFeatureObserverSpec) -> bool:
    return _is_feature_conditioned_spec(spec) or _is_pose_known_spec(spec)


@dataclass
class ResidualUpdateModel:
    x_mean: np.ndarray
    z_mean: np.ndarray
    coef: np.ndarray
    ridge: float
    n_train: int
    train_mse: float


def _fit_residual_update(
    *,
    residual_train: np.ndarray,
    x_train: np.ndarray,
    ridge: float,
) -> ResidualUpdateModel:
    z = np.asarray(residual_train, dtype=np.float64)
    x = np.asarray(x_train, dtype=np.float64)
    if z.ndim != 2 or x.ndim != 2 or z.shape[0] != x.shape[0]:
        raise ValueError(f"Expected residual/x train matrices with shared rows, got {z.shape} and {x.shape}")
    if z.shape[0] <= 1:
        raise ValueError("Need at least two residual-update training samples")
    x_mean = np.mean(x, axis=0)
    z_mean = np.mean(z, axis=0)
    xc = x - x_mean[None, :]
    zc = z - z_mean[None, :]
    ridge_val = max(float(ridge), 1e-12)
    if xc.shape[1] <= xc.shape[0]:
        normal = xc.T @ xc + ridge_val * np.eye(xc.shape[1], dtype=np.float64)
        coef = np.linalg.solve(normal, xc.T @ zc)
    else:
        gram = xc @ xc.T + ridge_val * np.eye(xc.shape[0], dtype=np.float64)
        alpha = np.linalg.solve(gram, zc)
        coef = xc.T @ alpha
    pred = (xc @ coef) + z_mean[None, :]
    train_mse = float(np.mean((z - pred) ** 2))
    return ResidualUpdateModel(
        x_mean=x_mean,
        z_mean=z_mean,
        coef=coef,
        ridge=ridge_val,
        n_train=int(x.shape[0]),
        train_mse=train_mse,
    )


def _predict_residual_update(model: ResidualUpdateModel, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return (arr - model.x_mean[None, :]) @ model.coef + model.z_mean[None, :]


def _mean_feature_cosine(pred: np.ndarray, true: np.ndarray) -> float:
    lhs = np.asarray(pred, dtype=np.float64)
    rhs = np.asarray(true, dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.ndim != 2:
        raise ValueError(f"Expected matching 2D arrays, got {lhs.shape} and {rhs.shape}")
    denom = np.linalg.norm(lhs, axis=1) * np.linalg.norm(rhs, axis=1)
    good = np.isfinite(lhs).all(axis=1) & np.isfinite(rhs).all(axis=1) & (denom > 1e-12)
    if not np.any(good):
        return float("nan")
    cosine = np.sum(lhs[good] * rhs[good], axis=1) / denom[good]
    return float(np.mean(cosine))


def _fit_scalar_gain(pred: np.ndarray, true: np.ndarray) -> float:
    lhs = np.asarray(pred, dtype=np.float64)
    rhs = np.asarray(true, dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.ndim != 2:
        raise ValueError(f"Expected matching 2D arrays, got {lhs.shape} and {rhs.shape}")
    good = np.isfinite(lhs).all(axis=1) & np.isfinite(rhs).all(axis=1)
    if not np.any(good):
        return float("nan")
    denom = float(np.sum(lhs[good] * lhs[good]))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(lhs[good] * rhs[good]) / denom)


def _gain_calibrated_metrics(
    z_hat: np.ndarray,
    z_true: np.ndarray,
    *,
    train_mean: np.ndarray,
    gain: float,
) -> dict[str, float | str]:
    if not np.isfinite(gain):
        return {
            "feature_scalar_gain_train": float("nan"),
            "feature_mse_gain_calibrated": float("nan"),
            "feature_sse_gain_calibrated": float("nan"),
            "feature_sst_gain_calibrated_train_baseline": float("nan"),
            "feature_r2_row_diagnostic_gain_calibrated": float("nan"),
            "feature_pred_norm_gain_calibrated": float("nan"),
            "feature_gain_calibration": "train_fold_scalar_gain",
        }
    metrics = _metrics(float(gain) * np.asarray(z_hat, dtype=np.float64), z_true, train_mean=train_mean)
    return {
        "feature_scalar_gain_train": float(gain),
        "feature_mse_gain_calibrated": float(metrics["feature_mse"]),
        "feature_sse_gain_calibrated": float(metrics["feature_sse"]),
        "feature_sst_gain_calibrated_train_baseline": float(metrics["feature_sst_train_baseline"]),
        "feature_r2_row_diagnostic_gain_calibrated": float(metrics["feature_r2_row_diagnostic"]),
        "feature_pred_norm_gain_calibrated": float(metrics["feature_pred_norm"]),
        "feature_gain_calibration": "train_fold_scalar_gain",
    }


def _prefixed_metric_fields(metrics: dict[str, float | str], prefix: str) -> dict[str, float | str]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _add_crossfold_scalar_gain_calibration(trials: pd.DataFrame) -> pd.DataFrame:
    required = {
        "decoder_mode",
        "latent",
        "feature_space_mode",
        "observer_mode",
        "observer_label",
        "fold",
        "feature_cosine",
        "feature_pred_norm",
        "feature_true_norm",
        "feature_sst_train_baseline",
    }
    if not required.issubset(trials.columns):
        return trials
    out = trials.copy()
    out["feature_cv_scalar_gain"] = np.nan
    out["feature_sse_cv_gain_calibrated"] = np.nan
    out["feature_sst_cv_gain_calibrated_train_baseline"] = out["feature_sst_train_baseline"]
    out["feature_r2_row_diagnostic_cv_gain_calibrated"] = np.nan
    out["feature_pred_norm_cv_gain_calibrated"] = np.nan
    out["feature_cv_gain_calibration"] = "outer_crossfold_scalar_gain_from_row_metrics"
    group_cols = ["decoder_mode", "latent", "feature_space_mode", "observer_mode", "observer_label"]
    for _group_key, group in out.groupby(group_cols, dropna=False, sort=False):
        idx = group.index.to_numpy()
        folds = group["fold"].to_numpy()
        pred_norm = group["feature_pred_norm"].to_numpy(dtype=np.float64)
        true_norm = group["feature_true_norm"].to_numpy(dtype=np.float64)
        cosine = group["feature_cosine"].to_numpy(dtype=np.float64)
        dot = cosine * pred_norm * true_norm
        for fold in sorted(set(folds.tolist())):
            test_local = folds == fold
            train_local = ~test_local
            train_good = train_local & np.isfinite(dot) & np.isfinite(pred_norm) & (pred_norm > 1e-12)
            denom = float(np.sum(pred_norm[train_good] * pred_norm[train_good]))
            gain = float(np.sum(dot[train_good]) / denom) if denom > 1e-12 else float("nan")
            test_good = test_local & np.isfinite(gain) & np.isfinite(dot) & np.isfinite(pred_norm) & np.isfinite(true_norm)
            if not np.any(test_good):
                continue
            sse = true_norm[test_good] ** 2 - 2.0 * gain * dot[test_good] + (gain**2) * pred_norm[test_good] ** 2
            sse = np.maximum(sse, 0.0)
            row_indices = idx[test_good]
            out.loc[row_indices, "feature_cv_scalar_gain"] = gain
            out.loc[row_indices, "feature_sse_cv_gain_calibrated"] = sse
            out.loc[row_indices, "feature_pred_norm_cv_gain_calibrated"] = gain * pred_norm[test_good]
            sst = out.loc[row_indices, "feature_sst_train_baseline"].to_numpy(dtype=np.float64)
            valid_sst = np.isfinite(sst) & (sst > 1e-12)
            r2 = np.full(row_indices.shape[0], np.nan, dtype=np.float64)
            r2[valid_sst] = 1.0 - sse[valid_sst] / sst[valid_sst]
            out.loc[row_indices, "feature_r2_row_diagnostic_cv_gain_calibrated"] = r2
    return out


def _validated_residual_shrinkage(
    *,
    z0_train: np.ndarray,
    z_true_train: np.ndarray,
    x_train: np.ndarray,
    source_rows: np.ndarray,
    ridge: float,
    seed: int,
) -> dict[str, Any]:
    z0 = np.asarray(z0_train, dtype=np.float64)
    z_true = np.asarray(z_true_train, dtype=np.float64)
    x = np.asarray(x_train, dtype=np.float64)
    sources = np.asarray(source_rows, dtype=int)
    if z0.shape != z_true.shape or z0.ndim != 2 or x.ndim != 2 or x.shape[0] != z0.shape[0]:
        raise ValueError(f"Shrinkage shape mismatch: z0={z0.shape}, z_true={z_true.shape}, x={x.shape}")
    lambda_grid = np.asarray([0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0], dtype=np.float64)
    residual_cv = np.full_like(z0, np.nan, dtype=np.float64)
    sst_cv = np.full(z0.shape[0], np.nan, dtype=np.float64)
    unique_sources = np.asarray(sorted(set(int(value) for value in sources.tolist())), dtype=int)
    if unique_sources.size >= 2:
        inner_fold_by_source = _assign_source_folds(
            unique_sources,
            n_folds=min(5, max(2, int(unique_sources.size))),
            seed=int(seed),
        )
        for inner_fold in sorted(set(inner_fold_by_source.values())):
            fit_sources = {
                int(source)
                for source, source_fold in inner_fold_by_source.items()
                if int(source_fold) != int(inner_fold)
            }
            predict_sources = {
                int(source)
                for source, source_fold in inner_fold_by_source.items()
                if int(source_fold) == int(inner_fold)
            }
            fit_mask = np.asarray([int(source) in fit_sources for source in sources], dtype=bool)
            predict_mask = np.asarray([int(source) in predict_sources for source in sources], dtype=bool)
            fit_mask &= np.isfinite(x).all(axis=1) & np.isfinite(z0).all(axis=1) & np.isfinite(z_true).all(axis=1)
            predict_mask &= np.isfinite(x).all(axis=1) & np.isfinite(z0).all(axis=1) & np.isfinite(z_true).all(axis=1)
            if int(np.sum(fit_mask)) <= 1 or int(np.sum(predict_mask)) == 0:
                continue
            train_mean = np.mean(z_true[fit_mask], axis=0)
            model = _fit_residual_update(
                residual_train=z_true[fit_mask] - z0[fit_mask],
                x_train=x[fit_mask],
                ridge=float(ridge),
            )
            residual_cv[predict_mask] = _predict_residual_update(model, x[predict_mask])
            baseline = z_true[predict_mask] - train_mean[None, :]
            sst_cv[predict_mask] = np.sum(baseline * baseline, axis=1)
    valid = (
        np.isfinite(residual_cv).all(axis=1)
        & np.isfinite(z0).all(axis=1)
        & np.isfinite(z_true).all(axis=1)
        & np.isfinite(sst_cv)
    )
    if not np.any(valid):
        finite = np.isfinite(z0).all(axis=1) & np.isfinite(z_true).all(axis=1)
        if np.any(finite):
            train_mean = np.mean(z_true[finite], axis=0)
            sse = float(np.sum((z_true[finite] - z0[finite]) ** 2))
            sst = float(np.sum((z_true[finite] - train_mean[None, :]) ** 2))
            zero_score = float(1.0 - sse / sst) if sst > 1e-12 else float("nan")
        else:
            zero_score = float("nan")
        return {
            "lambda": 0.0,
            "validation_score": zero_score,
            "validation_score_zero": zero_score,
            "validation_score_best": zero_score,
            "lambda_cosine_selected": 0.0,
            "validation_score_cosine_selected": _mean_feature_cosine(z0[finite], z_true[finite]) if np.any(finite) else float("nan"),
            "validation_score_cosine_zero": _mean_feature_cosine(z0[finite], z_true[finite]) if np.any(finite) else float("nan"),
            "validation_score_r2_at_cosine_lambda": zero_score,
            "validation_n": 0,
            "lambda_grid": ",".join(f"{value:g}" for value in lambda_grid.tolist()),
            "selection_reason": "no_valid_inner_residual_predictions",
        }
    sst_total = float(np.sum(sst_cv[valid]))
    scores = []
    cosine_scores = []
    for value in lambda_grid:
        pred = z0[valid] + float(value) * residual_cv[valid]
        sse_total = float(np.sum((z_true[valid] - pred) ** 2))
        score = float(1.0 - sse_total / sst_total) if sst_total > 1e-12 else float("nan")
        scores.append(score)
        cosine_scores.append(_mean_feature_cosine(pred, z_true[valid]))
    scores = np.asarray(scores, dtype=np.float64)
    cosine_scores = np.asarray(cosine_scores, dtype=np.float64)
    best_index = int(np.nanargmax(scores))
    cosine_best_index = int(np.nanargmax(cosine_scores))
    zero_index = int(np.where(np.isclose(lambda_grid, 0.0))[0][0])
    best_score = float(scores[best_index])
    zero_score = float(scores[zero_index])
    if zero_score >= best_score - 1e-6:
        best_index = zero_index
        best_score = zero_score
    return {
        "lambda": float(lambda_grid[best_index]),
        "validation_score": best_score,
        "validation_score_zero": zero_score,
        "validation_score_best": float(np.nanmax(scores)),
        "lambda_cosine_selected": float(lambda_grid[cosine_best_index]),
        "validation_score_cosine_selected": float(cosine_scores[cosine_best_index]),
        "validation_score_cosine_zero": float(cosine_scores[zero_index]),
        "validation_score_r2_at_cosine_lambda": float(scores[cosine_best_index]),
        "validation_n": int(np.sum(valid)),
        "lambda_grid": ",".join(f"{value:g}" for value in lambda_grid.tolist()),
        "selection_reason": "inner_source_disjoint_pooled_r2_cv_sse_sst",
    }


def _build_feature_conditioned_fold_state(
    *,
    banks: dict[str, SampleBank],
    tests: TestSet,
    feature_table: FeatureTable,
    transform: FeatureTransform,
    fold_by_source: dict[int, int],
    fold: int,
    ridge: float,
    noise_floor: float,
    continuous_args: argparse.Namespace,
    compute_z0: bool = True,
    precompute_feature_conditioned_tau: bool = True,
) -> dict[str, Any]:
    response_bank = banks["observed_response_only"]
    response_x_all = tests.x_by_mode["observed_response_only"]
    z0_hat = np.full((response_x_all.shape[0], transform.feature_dim), np.nan, dtype=np.float64)
    first_pass_n_train_samples = 0
    if bool(compute_z0):
        first_pass_train_mask = np.asarray(
            [fold_by_source.get(int(source), -1) != int(fold) for source in response_bank.source_rows],
            dtype=bool,
        )
        first_pass_train_mask &= np.isfinite(response_bank.x).all(axis=1)
        if int(np.sum(first_pass_train_mask)) <= transform.feature_dim:
            raise ValueError("too few first-pass response-only samples for feature-conditioned tau")
        response_bank_z = _transform_feature_sources(transform, feature_table, response_bank.source_rows)
        first_pass_model: ForwardPosteriorModel = _fit_forward_posterior(
            z_train=response_bank_z[first_pass_train_mask],
            x_train=response_bank.x[first_pass_train_mask],
            ridge=float(ridge),
            noise_floor=float(noise_floor),
        )
        first_pass_n_train_samples = int(first_pass_model.n_train)
        valid_response_rows = np.isfinite(response_x_all).all(axis=1)
        z0_hat[valid_response_rows] = _predict_z(first_pass_model, response_x_all[valid_response_rows])

        outer_train_sources = {
            int(source) for source, source_fold in fold_by_source.items() if int(source_fold) != int(fold)
        }
        outer_train_test_rows = np.asarray(
            [int(source) in outer_train_sources for source in tests.rows["true_source_row"].to_numpy(dtype=int)],
            dtype=bool,
        )
        inner_fold_by_source = _assign_source_folds(
            np.asarray(sorted(outer_train_sources), dtype=int),
            n_folds=min(5, max(2, len(outer_train_sources))),
            seed=7919 + 101 * int(fold),
        )
        for inner_fold in sorted(set(inner_fold_by_source.values())):
            inner_fit_sources = {
                int(source)
                for source, source_fold in inner_fold_by_source.items()
                if int(source_fold) != int(inner_fold)
            }
            inner_predict_sources = {
                int(source)
                for source, source_fold in inner_fold_by_source.items()
                if int(source_fold) == int(inner_fold)
            }
            inner_train_mask = np.asarray(
                [int(source) in inner_fit_sources for source in response_bank.source_rows],
                dtype=bool,
            )
            inner_train_mask &= np.isfinite(response_bank.x).all(axis=1)
            inner_predict_mask = (
                outer_train_test_rows
                & valid_response_rows
                & np.asarray(
                    [
                        int(source) in inner_predict_sources
                        for source in tests.rows["true_source_row"].to_numpy(dtype=int)
                    ],
                    dtype=bool,
                )
            )
            if int(np.sum(inner_train_mask)) <= transform.feature_dim or int(np.sum(inner_predict_mask)) == 0:
                continue
            inner_model: ForwardPosteriorModel = _fit_forward_posterior(
                z_train=response_bank_z[inner_train_mask],
                x_train=response_bank.x[inner_train_mask],
                ridge=float(ridge),
                noise_floor=float(noise_floor),
            )
            z0_hat[inner_predict_mask] = _predict_z(inner_model, response_x_all[inner_predict_mask])

    heldout_sources = {int(source) for source, source_fold in fold_by_source.items() if int(source_fold) == int(fold)}
    include_intercept = str(continuous_args.continuous_score_mode) in {
        "quadratic_affine_poisson_profile",
        "quadratic_prior_mean_poisson_profile",
    }
    baseline_coef, baseline_residual_var, baseline_fit_row = _fit_feature_conditioned_baseline(
        geometry_tables=tests.geometry_tables,
        transform=transform,
        feature_table=feature_table,
        heldout_sources=heldout_sources,
        ridge=float(continuous_args.continuous_ridge),
    )
    observation_coef, observation_residual_var, observation_fit_row = _fit_feature_conditioned_quadratic_observation_map(
        geometry_tables=tests.geometry_tables,
        transform=transform,
        feature_table=feature_table,
        heldout_sources=heldout_sources,
        ridge=float(continuous_args.continuous_ridge),
        include_intercept=include_intercept,
        intercept_ridge_multiplier=float(continuous_args.quadratic_intercept_ridge_multiplier),
    )

    x_linear: list[np.ndarray] = []
    x_interactions: list[np.ndarray] = []
    update_linear: list[np.ndarray] = []
    update_interactions: list[np.ndarray] = []
    pose_known_update_linear: list[np.ndarray] = []
    pose_known_update_interactions: list[np.ndarray] = []
    tau_hats: list[np.ndarray] = []
    tau_metrics_by_row: dict[int, dict[str, float]] = {}
    tau_meta_by_row: dict[int, dict[str, Any]] = {}
    pose_known_tau_metrics_by_row: dict[int, dict[str, float]] = {}
    pose_known_tau_meta_by_row: dict[int, dict[str, Any]] = {}
    for row_index, meta in tests.rows.reset_index(drop=True).iterrows():
        compact = np.asarray(tests.observed_compact[int(row_index)], dtype=np.float64)
        if bool(precompute_feature_conditioned_tau) and np.isfinite(z0_hat[int(row_index)]).all():
            table_args = _continuous_args_for_scale(continuous_args, float(meta["observation_scale"]))
            tau_hat, tau_meta = _feature_conditioned_tau_hat(
                observed_compact=compact,
                z0_hat=z0_hat[int(row_index)],
                geometry_table=tests.geometry_tables[int(row_index)],
                baseline_coef=baseline_coef,
                baseline_residual_var=baseline_residual_var,
                observation_coef=observation_coef,
                observation_residual_var=observation_residual_var,
                include_intercept=include_intercept,
                continuous_args=table_args,
                table_index=int(meta["table_index"]),
            )
        else:
            tau_hat = np.full((compact.shape[0], 2), np.nan, dtype=np.float64)
            tau_meta = {
                "tau_hat_source": "response_only_feature_conditioned_synthetic_prior_full_path_map",
                "observation_geometry_source": (
                    "source_disjoint_scale_and_feature_conditioned_response_table_local_response_vs_displacement_map"
                ),
                "feature_conditioned_tau_failure": (
                    "not_requested_by_observer_modes"
                    if not bool(precompute_feature_conditioned_tau)
                    else "non_finite_response_only_z0_hat"
                ),
            }
        tau_hats.append(tau_hat.astype(np.float32))
        tau_metrics_by_row[int(row_index)] = _tau_metrics(tau_hat, tests.tau_true[int(row_index)])
        tau_meta_by_row[int(row_index)] = tau_meta
        if np.isfinite(tau_hat).all():
            x_linear.append(_feature_vector(compact, tau_hat, mode="response_tau_linear"))
            x_interactions.append(_feature_vector(compact, tau_hat, mode="response_tau_interactions"))
            tau_values = _linear_tau_features(tau_hat, compact.shape[0])
            update_linear.append(tau_values)
            update_interactions.append(
                np.concatenate([tau_values, _response_tau_interactions(compact, tau_hat)]).astype(np.float32)
            )
        else:
            x_linear.append(_nan_feature_vector(compact, mode="response_tau_linear"))
            x_interactions.append(_nan_feature_vector(compact, mode="response_tau_interactions"))
            update_linear.append(np.full(4 * compact.shape[0], np.nan, dtype=np.float32))
            update_interactions.append(
                np.full(4 * compact.shape[0] + 2 * compact.size, np.nan, dtype=np.float32)
            )
        tau_true = np.asarray(tests.tau_true[int(row_index)], dtype=np.float64)
        tau_true_source = str(meta.get("tau_true_source", "unknown"))
        pose_known_tau_metrics_by_row[int(row_index)] = _tau_metrics(tau_true, tau_true)
        pose_known_tau_meta_by_row[int(row_index)] = {
            "tau_hat_source": f"{tau_true_source}_pose_known_upper_limit",
            "observation_geometry_source": "pose_known_no_tau_inference",
            "feature_conditioned_baseline_residual_variance": float("nan"),
            "feature_conditioned_observation_residual_variance": float("nan"),
            "feature_conditioned_observation_variance": float("nan"),
        }
        if np.isfinite(tau_true).all():
            tau_true_values = _linear_tau_features(tau_true, compact.shape[0])
            pose_known_update_linear.append(tau_true_values)
            pose_known_update_interactions.append(
                np.concatenate([tau_true_values, _response_tau_interactions(compact, tau_true)]).astype(np.float32)
            )
        else:
            pose_known_update_linear.append(np.full(4 * compact.shape[0], np.nan, dtype=np.float32))
            pose_known_update_interactions.append(
                np.full(4 * compact.shape[0] + 2 * compact.size, np.nan, dtype=np.float32)
            )

    fold_fit_rows = []
    for fit_row in [baseline_fit_row, observation_fit_row]:
        row = dict(fit_row)
        row.update(
            {
                "fold": int(fold),
                "latent": transform.latent,
                "feature_space_mode": transform.feature_space_mode,
                "feature_fit_scope": transform.fit_scope,
                "n_fit_sources": int(transform.n_fit_sources),
                "heldout_source_count": int(len(heldout_sources)),
                "tau_hat_source": "response_only_feature_conditioned_synthetic_prior_full_path_map",
            }
        )
        fold_fit_rows.append(row)
    return {
        "x_by_mode": {
            "feature_conditioned_response_tau_hat": np.stack(x_linear, axis=0).astype(np.float32),
            "feature_conditioned_response_tau_hat_interactions": np.stack(x_interactions, axis=0).astype(np.float32),
        },
        "update_x_by_mode": {
            "feature_conditioned_response_tau_hat": np.stack(update_linear, axis=0).astype(np.float32),
            "feature_conditioned_response_tau_hat_interactions": np.stack(update_interactions, axis=0).astype(np.float32),
            "pose_known_response_tau": np.stack(pose_known_update_linear, axis=0).astype(np.float32),
            "pose_known_response_tau_interactions": np.stack(pose_known_update_interactions, axis=0).astype(np.float32),
            "pose_known_nested_response_tau": np.stack(pose_known_update_linear, axis=0).astype(np.float32),
            "pose_known_nested_response_tau_interactions": np.stack(pose_known_update_interactions, axis=0).astype(np.float32),
        },
        "z0_hat": z0_hat.astype(np.float32),
        "tau_hat": np.stack(tau_hats, axis=0).astype(np.float32),
        "tau_metrics_by_row": tau_metrics_by_row,
        "tau_meta_by_row": tau_meta_by_row,
        "pose_known_tau_metrics_by_row": pose_known_tau_metrics_by_row,
        "pose_known_tau_meta_by_row": pose_known_tau_meta_by_row,
        "fit_rows": fold_fit_rows,
        "baseline_coef": baseline_coef.astype(np.float64),
        "observation_coef": observation_coef.astype(np.float64),
        "include_intercept": bool(include_intercept),
        "z0_computed": bool(compute_z0),
        "feature_conditioned_tau_precomputed": bool(precompute_feature_conditioned_tau),
        "first_pass_n_train_samples": int(first_pass_n_train_samples),
        "first_pass_response_dim": int(response_bank.x.shape[1]),
        "feature_conditioned_baseline_residual_variance": float(baseline_residual_var),
        "feature_conditioned_observation_residual_variance": float(observation_residual_var),
    }


def _run_crossfit(
    *,
    banks: dict[str, SampleBank],
    tests: TestSet,
    feature_table: FeatureTable,
    feature_weights: np.ndarray | None,
    feature_space_modes: list[str],
    specs: list[LinearFeatureObserverSpec],
    feature_dim: int,
    n_folds: int,
    fold_seed: int,
    ridge: float,
    noise_floor: float,
    continuous_args: argparse.Namespace,
    save_feature_predictions: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    fold_by_source = _assign_source_folds(
        tests.rows["true_source_row"].to_numpy(dtype=int),
        n_folds=int(n_folds),
        seed=int(fold_seed),
    )
    test_folds = tests.rows["true_source_row"].map(fold_by_source).to_numpy(dtype=int)
    canonical_modes = _canonical_feature_modes(feature_space_modes)
    global_transforms: dict[str, FeatureTransform] = {}
    for mode in canonical_modes:
        if _feature_space_config(mode)["fit_scope"] == "global":
            global_transforms[mode] = _fit_feature_transform(
                feature_table,
                fit_sources=feature_table.source_rows,
                feature_dim=int(feature_dim),
                feature_space_mode=mode,
                feature_weights=feature_weights,
            )

    trial_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    prediction_parts = _new_feature_prediction_parts() if bool(save_feature_predictions) else None
    compute_response_only_z0 = any(_needs_response_only_z0(spec) for spec in specs)
    precompute_feature_conditioned_tau = any(_needs_precomputed_feature_conditioned_tau(spec) for spec in specs)
    for mode in canonical_modes:
        for fold in sorted(set(test_folds.tolist())):
            base_test_mask = test_folds == int(fold)
            if int(np.sum(base_test_mask)) == 0:
                continue
            transform = _fit_transform_for_fold(
                mode=mode,
                fold=int(fold),
                fold_by_source=fold_by_source,
                feature_table=feature_table,
                feature_dim=int(feature_dim),
                feature_weights=feature_weights,
                global_transforms=global_transforms,
            )
            baseline_sources = np.asarray(
                [
                    int(source)
                    for source, source_fold in fold_by_source.items()
                    if int(source_fold) != int(fold)
                ],
                dtype=int,
            )
            z_train_baseline = _transform_feature_sources(transform, feature_table, baseline_sources)
            z_train_mean = np.mean(z_train_baseline, axis=0)
            feature_conditioned_state: dict[str, Any] | None = None
            for spec in specs:
                if _is_forward_model_spec(spec):
                    if feature_conditioned_state is None:
                        feature_conditioned_state = _build_feature_conditioned_fold_state(
                            banks=banks,
                            tests=tests,
                            feature_table=feature_table,
                            transform=transform,
                            fold_by_source=fold_by_source,
                            fold=int(fold),
                            ridge=float(ridge),
                            noise_floor=float(noise_floor),
                            continuous_args=continuous_args,
                            compute_z0=compute_response_only_z0,
                            precompute_feature_conditioned_tau=precompute_feature_conditioned_tau,
                        )
                        fit_rows.extend(feature_conditioned_state["fit_rows"])
                    z_true_all = _transform_feature_sources(
                        transform,
                        feature_table,
                        tests.rows["true_source_row"].to_numpy(dtype=int),
                    )
                    z_hats: list[np.ndarray] = []
                    z_trues: list[np.ndarray] = []
                    valid_indices: list[int] = []
                    tau_by_index: dict[int, np.ndarray] = {}
                    meta_by_index: dict[int, dict[str, Any]] = {}
                    for global_index in np.flatnonzero(base_test_mask):
                        compact = np.asarray(tests.observed_compact[int(global_index)], dtype=np.float64)
                        if not np.isfinite(compact).all():
                            continue
                        meta = tests.rows.iloc[int(global_index)]
                        table_args = _continuous_args_for_scale(continuous_args, float(meta["observation_scale"]))
                        common = {
                            "observed_compact": compact,
                            "geometry_table": tests.geometry_tables[int(global_index)],
                            "baseline_coef": np.asarray(feature_conditioned_state["baseline_coef"], dtype=np.float64),
                            "baseline_residual_var": float(
                                feature_conditioned_state["feature_conditioned_baseline_residual_variance"]
                            ),
                            "observation_coef": np.asarray(
                                feature_conditioned_state["observation_coef"],
                                dtype=np.float64,
                            ),
                            "observation_residual_var": float(
                                feature_conditioned_state["feature_conditioned_observation_residual_variance"]
                            ),
                            "include_intercept": bool(feature_conditioned_state["include_intercept"]),
                            "continuous_args": table_args,
                        }
                        if spec.slug == "pose_known_forward_model":
                            tau = np.asarray(tests.tau_true[int(global_index)], dtype=np.float64)
                            if not np.isfinite(tau).all():
                                continue
                            z_hat, _pred, solver_meta = _solve_z_given_tau(tau=tau, **common)
                            tau_true_source = str(meta.get("tau_true_source", "unknown"))
                            forward_meta = {
                                **solver_meta,
                                "tau_hat_source": f"{tau_true_source}_pose_known_forward_model",
                                "observation_geometry_source": (
                                    "source_disjoint_scale_and_feature_conditioned_forward_response_model"
                                ),
                                "feature_update_mode": "forward_model_z_map_given_recorded_tau",
                                "feature_conditioned_tau_optimizer_success": float("nan"),
                                "feature_conditioned_tau_optimizer_iterations": -1,
                            }
                        elif spec.slug == "estimated_tau_forward_model":
                            tau = np.asarray(feature_conditioned_state["tau_hat"][int(global_index)], dtype=np.float64)
                            if not np.isfinite(tau).all():
                                continue
                            z_hat, _pred, solver_meta = _solve_z_given_tau(tau=tau, **common)
                            tau_meta = dict(feature_conditioned_state["tau_meta_by_row"][int(global_index)])
                            forward_meta = {
                                **tau_meta,
                                **solver_meta,
                                "tau_hat_source": "response_only_feature_conditioned_tau_hat_used_by_forward_model",
                                "observation_geometry_source": (
                                    "source_disjoint_scale_and_feature_conditioned_forward_response_model"
                                ),
                                "feature_update_mode": "forward_model_z_map_given_feature_conditioned_tau_hat",
                                "feature_conditioned_tau_optimizer_success": bool(
                                    tau_meta.get("optimizer_success", False)
                                ),
                                "feature_conditioned_tau_optimizer_iterations": int(
                                    tau_meta.get("optimizer_iterations", -1)
                                ),
                            }
                        elif spec.slug == "zero_tau_forward_model":
                            tau = np.zeros((compact.shape[0], 2), dtype=np.float64)
                            z_hat, _pred, solver_meta = _solve_z_given_tau(tau=tau, **common)
                            forward_meta = {
                                **solver_meta,
                                "tau_hat_source": "zero_tau_forward_model",
                                "observation_geometry_source": (
                                    "source_disjoint_scale_and_feature_conditioned_forward_response_model"
                                ),
                                "feature_update_mode": "forward_model_z_map_given_zero_tau",
                                "feature_conditioned_tau_optimizer_success": float("nan"),
                                "feature_conditioned_tau_optimizer_iterations": -1,
                            }
                        elif spec.slug == "hidden_joint_forward_model":
                            initial_z = np.asarray(
                                feature_conditioned_state["z0_hat"][int(global_index)],
                                dtype=np.float64,
                            )
                            if not np.isfinite(initial_z).all():
                                continue
                            z_hat, tau, _pred, forward_meta = _joint_z_tau_map(
                                initial_z=initial_z,
                                table_index=int(meta["table_index"]),
                                **common,
                            )
                            forward_meta.update(
                                {
                                    "feature_update_mode": (
                                        "alternating_forward_model_z_tau_map_under_synthetic_empirical_confined_prior"
                                    ),
                                    "feature_conditioned_tau_optimizer_success": bool(
                                        forward_meta.get("optimizer_success", False)
                                    ),
                                    "feature_conditioned_tau_optimizer_iterations": int(
                                        forward_meta.get("optimizer_iterations", -1)
                                    ),
                                }
                            )
                        else:
                            raise ValueError(f"Unhandled forward-model observer mode {spec.slug!r}")
                        if not np.isfinite(z_hat).all():
                            continue
                        z_hats.append(z_hat.astype(np.float64))
                        z_trues.append(z_true_all[int(global_index)].astype(np.float64))
                        valid_indices.append(int(global_index))
                        tau_by_index[int(global_index)] = tau.astype(np.float64)
                        meta_by_index[int(global_index)] = forward_meta

                    if not valid_indices:
                        model_rows.append(
                            {
                                "decoder_mode": "linear_gaussian",
                                "observer_mode": spec.slug,
                                "observer_label": spec.label,
                                "train_bank": spec.train_bank,
                                "test_input": spec.test_input,
                                "latent": transform.latent,
                                "feature_space_mode": transform.feature_space_mode,
                                "fold": int(fold),
                                "n_fit_sources": int(transform.n_fit_sources),
                                "n_test_rows": int(np.sum(base_test_mask)),
                                "n_train_samples": int(feature_conditioned_state["first_pass_n_train_samples"]),
                                "feature_dim": int(transform.feature_dim),
                                "raw_feature_dim": int(transform.raw_feature_dim),
                                "response_dim": int(np.prod(tests.observed_compact.shape[1:])),
                                "skipped": True,
                                "skip_reason": "no_valid_forward_model_rows",
                                "interpretation": spec.interpretation,
                            }
                        )
                        continue

                    z_hat_arr = np.stack(z_hats, axis=0)
                    z_true_arr = np.stack(z_trues, axis=0)
                    residual_mse_values = np.asarray(
                        [
                            float(meta_by_index[index].get("forward_response_residual_mse", np.nan))
                            for index in valid_indices
                        ],
                        dtype=np.float64,
                    )
                    model_rows.append(
                        {
                            "decoder_mode": "linear_gaussian",
                            "observer_mode": spec.slug,
                            "observer_label": spec.label,
                            "train_bank": spec.train_bank,
                            "test_input": spec.test_input,
                            "latent": transform.latent,
                            "feature_space_mode": transform.feature_space_mode,
                            "fold": int(fold),
                            "n_fit_sources": int(transform.n_fit_sources),
                            "n_test_rows": int(len(valid_indices)),
                            "n_train_samples": int(feature_conditioned_state["first_pass_n_train_samples"]),
                            "feature_dim": int(transform.feature_dim),
                            "raw_feature_dim": int(transform.raw_feature_dim),
                            "response_dim": int(np.prod(tests.observed_compact.shape[1:])),
                            "feature_fit_scope": transform.fit_scope,
                            "feature_preprocessing": transform.preprocessing,
                            "feature_whitened": bool(transform.whitened),
                            "feature_weighted": bool(transform.weighted),
                            "feature_variance_fraction": float(transform.explained_variance_sum),
                            "r2_cv_train_baseline": "source_fold_train_feature_mean",
                            "ridge": float(getattr(continuous_args, "forward_model_z_prior_precision", 1.0)),
                            "noise_variance": float(
                                feature_conditioned_state["feature_conditioned_baseline_residual_variance"]
                                + feature_conditioned_state["feature_conditioned_observation_residual_variance"]
                            ),
                            "response_map_fro_norm": float(
                                np.linalg.norm(feature_conditioned_state["observation_coef"])
                            ),
                            "posterior_gain_fro_norm": float("nan"),
                            "feature_update_mode": "forward_model_latent_map",
                            "first_pass_observer_mode": (
                                "response_only" if feature_conditioned_state["z0_computed"] else "none"
                            ),
                            "first_pass_n_train_samples": int(
                                feature_conditioned_state["first_pass_n_train_samples"]
                            ),
                            "first_pass_response_dim": int(feature_conditioned_state["first_pass_response_dim"]),
                            "tau_hat_source": (
                                "varies_by_forward_model_mode"
                                if spec.slug != "hidden_joint_forward_model"
                                else "hidden_joint_forward_model_synthetic_prior_alternating_map"
                            ),
                            "observation_geometry_source": (
                                "source_disjoint_scale_and_feature_conditioned_forward_response_model"
                            ),
                            "feature_conditioned_baseline_residual_variance": float(
                                feature_conditioned_state["feature_conditioned_baseline_residual_variance"]
                            ),
                            "feature_conditioned_observation_residual_variance": float(
                                feature_conditioned_state["feature_conditioned_observation_residual_variance"]
                            ),
                            "mean_forward_response_residual_mse": float(np.nanmean(residual_mse_values)),
                            "uses_response_table_trajectory_training_rows": True,
                            "uses_response_table_geometry_calibration_rows": True,
                            "uses_response_table_trajectory_rows_for_feature_endpoint": False,
                            "uses_response_table_trajectory_rows_for_geometry_calibration": True,
                            "interpretation": spec.interpretation,
                        }
                    )
                    _append_feature_predictions(
                        prediction_parts,
                        tests=tests,
                        global_indices=valid_indices,
                        z_hat=z_hat_arr,
                        z_true=z_true_arr,
                        z_train_mean=z_train_mean,
                        spec=spec,
                        transform=transform,
                        fold=int(fold),
                        n_train_samples=int(feature_conditioned_state["first_pass_n_train_samples"]),
                        prediction_source="forward_model_out_of_fold",
                    )
                    for local_index, global_index in enumerate(valid_indices):
                        trial_meta = meta_by_index[int(global_index)]
                        row = dict(tests.rows.iloc[int(global_index)].to_dict())
                        row.update(_tau_metrics(tau_by_index[int(global_index)], tests.tau_true[int(global_index)]))
                        row.update(
                            {
                                "decoder_mode": "linear_gaussian",
                                "observer_mode": spec.slug,
                                "observer_label": spec.label,
                                "train_bank": spec.train_bank,
                                "test_input": spec.test_input,
                                "latent": transform.latent,
                                "feature_space_mode": transform.feature_space_mode,
                                "feature_fit_scope": transform.fit_scope,
                                "feature_preprocessing": transform.preprocessing,
                                "feature_whitened": bool(transform.whitened),
                                "feature_weighted": bool(transform.weighted),
                                "feature_variance_fraction": float(transform.explained_variance_sum),
                                "r2_cv_train_baseline": "source_fold_train_feature_mean",
                                "fold": int(fold),
                                "n_train_samples": int(feature_conditioned_state["first_pass_n_train_samples"]),
                                "n_fit_sources": int(transform.n_fit_sources),
                                "feature_update_mode": str(trial_meta.get("feature_update_mode", "")),
                                "tau_hat_source": str(trial_meta.get("tau_hat_source", "")),
                                "observation_geometry_source": str(
                                    trial_meta.get("observation_geometry_source", "")
                                ),
                                "first_pass_observer_mode": (
                                    "response_only" if feature_conditioned_state["z0_computed"] else "none"
                                ),
                                "feature_conditioned_baseline_residual_variance": float(
                                    trial_meta.get("feature_conditioned_baseline_residual_variance", np.nan)
                                ),
                                "feature_conditioned_observation_residual_variance": float(
                                    trial_meta.get("feature_conditioned_observation_residual_variance", np.nan)
                                ),
                                "feature_conditioned_observation_variance": float(
                                    trial_meta.get("feature_conditioned_observation_variance", np.nan)
                                ),
                                "feature_conditioned_tau_optimizer_success": trial_meta.get(
                                    "feature_conditioned_tau_optimizer_success",
                                    np.nan,
                                ),
                                "feature_conditioned_tau_optimizer_iterations": int(
                                    trial_meta.get("feature_conditioned_tau_optimizer_iterations", -1)
                                ),
                                "forward_z_solver": str(trial_meta.get("forward_z_solver", "")),
                                "forward_z_solver_success": bool(
                                    trial_meta.get("forward_z_solver_success", False)
                                ),
                                "forward_z_prior_precision": float(
                                    trial_meta.get("forward_z_prior_precision", np.nan)
                                ),
                                "forward_response_residual_mse": float(
                                    trial_meta.get("forward_response_residual_mse", np.nan)
                                ),
                                "forward_profile_energy": float(
                                    trial_meta.get("forward_profile_energy", np.nan)
                                ),
                                "forward_prediction_norm": float(
                                    trial_meta.get("forward_prediction_norm", np.nan)
                                ),
                                "forward_design_fro_norm": float(
                                    trial_meta.get("forward_design_fro_norm", np.nan)
                                ),
                                "forward_joint_iterations": int(
                                    trial_meta.get("forward_joint_iterations", 0)
                                ),
                                "forward_joint_final_z_delta_norm": float(
                                    trial_meta.get("forward_joint_final_z_delta_norm", np.nan)
                                ),
                                "uses_response_table_trajectory_training_rows": True,
                                "uses_response_table_geometry_calibration_rows": True,
                                "uses_response_table_trajectory_rows_for_feature_endpoint": False,
                                "uses_response_table_trajectory_rows_for_geometry_calibration": True,
                            }
                        )
                        row.update(_metrics(z_hat_arr[local_index], z_true_arr[local_index], train_mean=z_train_mean))
                        trial_rows.append(row)
                    continue

                if _is_residual_update_spec(spec):
                    if feature_conditioned_state is None:
                        feature_conditioned_state = _build_feature_conditioned_fold_state(
                            banks=banks,
                            tests=tests,
                            feature_table=feature_table,
                            transform=transform,
                            fold_by_source=fold_by_source,
                            fold=int(fold),
                            ridge=float(ridge),
                            noise_floor=float(noise_floor),
                            continuous_args=continuous_args,
                            compute_z0=compute_response_only_z0,
                            precompute_feature_conditioned_tau=precompute_feature_conditioned_tau,
                        )
                        fit_rows.extend(feature_conditioned_state["fit_rows"])
                    x_all = feature_conditioned_state["update_x_by_mode"][spec.test_input]
                    bank = SampleBank(
                        x=x_all,
                        source_rows=tests.rows["true_source_row"].to_numpy(dtype=int),
                        table_indices=tests.rows["table_index"].to_numpy(dtype=int),
                    )
                else:
                    bank = banks[spec.train_bank]
                    x_all = tests.x_by_mode[spec.test_input]
                valid_test = np.isfinite(x_all).all(axis=1)
                if _is_residual_update_spec(spec) and feature_conditioned_state is not None:
                    valid_test &= np.isfinite(feature_conditioned_state["z0_hat"]).all(axis=1)
                test_mask = base_test_mask & valid_test
                if int(np.sum(test_mask)) == 0:
                    continue
                train_mask = np.asarray(
                    [fold_by_source.get(int(source), -1) != int(fold) for source in bank.source_rows],
                    dtype=bool,
                )
                train_mask &= np.isfinite(bank.x).all(axis=1)
                if _is_residual_update_spec(spec) and feature_conditioned_state is not None:
                    train_mask &= np.isfinite(feature_conditioned_state["z0_hat"]).all(axis=1)
                if int(np.sum(train_mask)) <= transform.feature_dim:
                    model_rows.append(
                        {
                            "decoder_mode": "linear_gaussian",
                            "observer_mode": spec.slug,
                            "observer_label": spec.label,
                            "train_bank": spec.train_bank,
                            "test_input": spec.test_input,
                            "latent": transform.latent,
                            "feature_space_mode": transform.feature_space_mode,
                            "fold": int(fold),
                            "n_fit_sources": int(transform.n_fit_sources),
                            "n_test_rows": int(np.sum(test_mask)),
                            "n_train_samples": int(np.sum(train_mask)),
                            "feature_dim": int(transform.feature_dim),
                            "raw_feature_dim": int(transform.raw_feature_dim),
                            "response_dim": int(bank.x.shape[1]),
                            "skipped": True,
                            "skip_reason": "too_few_finite_source_disjoint_training_samples",
                            "interpretation": spec.interpretation,
                        }
                    )
                    continue

                if _is_residual_update_spec(spec) and feature_conditioned_state is not None:
                    z0_all = np.asarray(feature_conditioned_state["z0_hat"], dtype=np.float64)
                    z_true_all = _transform_feature_sources(
                        transform,
                        feature_table,
                        tests.rows["true_source_row"].to_numpy(dtype=int),
                    )
                    residual_model = _fit_residual_update(
                        residual_train=z_true_all[train_mask] - z0_all[train_mask],
                        x_train=bank.x[train_mask],
                        ridge=float(ridge),
                    )
                    is_pose_known = _is_pose_known_spec(spec)
                    is_nested_pose_known = _is_pose_known_nested_spec(spec)
                    if is_nested_pose_known:
                        shrinkage = _validated_residual_shrinkage(
                            z0_train=z0_all[train_mask],
                            z_true_train=z_true_all[train_mask],
                            x_train=bank.x[train_mask],
                            source_rows=bank.source_rows[train_mask],
                            ridge=float(ridge),
                            seed=104729 + 1009 * int(fold),
                        )
                    else:
                        shrinkage = {
                            "lambda": 1.0,
                            "validation_score": float("nan"),
                            "validation_score_zero": float("nan"),
                            "validation_score_best": float("nan"),
                            "validation_n": 0,
                            "lambda_grid": "",
                            "selection_reason": "fixed_full_residual",
                        }
                    residual_shrinkage_lambda = float(shrinkage["lambda"])
                    z_delta = residual_shrinkage_lambda * _predict_residual_update(residual_model, x_all[test_mask])
                    z0_test = z0_all[test_mask]
                    z_hat = z0_test + z_delta
                    z_true = z_true_all[test_mask]
                    first_pass_sse = float(np.sum((z_true - z0_test) ** 2))
                    final_sse = float(np.sum((z_true - z_hat) ** 2))
                    first_pass_sst = float(np.sum((z_true - z_train_mean[None, :]) ** 2))
                    first_pass_r2 = (
                        float(1.0 - first_pass_sse / first_pass_sst)
                        if first_pass_sst > 1e-12
                        else float("nan")
                    )
                    z_hat_train_uncalibrated = z0_all[train_mask] + residual_shrinkage_lambda * _predict_residual_update(
                        residual_model,
                        bank.x[train_mask],
                    )
                    scalar_gain = _fit_scalar_gain(z_hat_train_uncalibrated, z_true_all[train_mask])
                    if is_nested_pose_known:
                        feature_update_mode = "response_only_z0_plus_pose_known_tau_validated_shrinkage"
                        tau_hat_source = "observed_recorded_eye_trace_pose_known_nested_upper_limit"
                        observation_geometry_source = "pose_known_no_tau_inference_validated_zero_fallback"
                    elif is_pose_known:
                        feature_update_mode = "response_only_z0_plus_pose_known_tau_residual_ridge"
                        tau_hat_source = "observed_recorded_eye_trace_pose_known_upper_limit"
                        observation_geometry_source = "pose_known_no_tau_inference"
                    else:
                        feature_update_mode = "response_only_z0_plus_tau_residual_ridge"
                        tau_hat_source = "response_only_feature_conditioned_synthetic_prior_full_path_map"
                        observation_geometry_source = (
                            "source_disjoint_scale_and_feature_conditioned_response_table_local_response_vs_displacement_map"
                        )
                    model_row = {
                        "decoder_mode": "linear_gaussian",
                        "observer_mode": spec.slug,
                        "observer_label": spec.label,
                        "train_bank": spec.train_bank,
                        "test_input": spec.test_input,
                        "latent": transform.latent,
                        "feature_space_mode": transform.feature_space_mode,
                        "fold": int(fold),
                        "n_fit_sources": int(transform.n_fit_sources),
                        "n_test_rows": int(np.sum(test_mask)),
                        "n_train_samples": int(residual_model.n_train),
                        "feature_dim": int(transform.feature_dim),
                        "raw_feature_dim": int(transform.raw_feature_dim),
                        "response_dim": int(bank.x.shape[1]),
                        "feature_fit_scope": transform.fit_scope,
                        "feature_preprocessing": transform.preprocessing,
                        "feature_whitened": bool(transform.whitened),
                        "feature_weighted": bool(transform.weighted),
                        "feature_variance_fraction": float(transform.explained_variance_sum),
                        "r2_cv_train_baseline": "source_fold_train_feature_mean",
                        "ridge": float(residual_model.ridge),
                        "noise_variance": float(residual_model.train_mse),
                        "response_map_fro_norm": float(np.linalg.norm(residual_model.coef)),
                        "posterior_gain_fro_norm": float(np.linalg.norm(residual_model.coef)),
                        "feature_update_mode": feature_update_mode,
                        "first_pass_base_observer": "internal_response_only_z0_hat",
                        "first_pass_feature_sse": first_pass_sse,
                        "first_pass_feature_sst_train_baseline": first_pass_sst,
                        "first_pass_R2_cv": first_pass_r2,
                        "known_nested_sse_minus_first_pass": final_sse - first_pass_sse,
                        "known_nested_sse_le_first_pass": bool(final_sse <= first_pass_sse + 1e-9),
                        "residual_shrinkage_lambda": residual_shrinkage_lambda,
                        "residual_shrinkage_validation_score": float(shrinkage["validation_score"]),
                        "residual_shrinkage_validation_score_zero": float(shrinkage["validation_score_zero"]),
                        "residual_shrinkage_validation_score_best": float(shrinkage["validation_score_best"]),
                        "residual_shrinkage_lambda_cosine_selected": float(
                            shrinkage.get("lambda_cosine_selected", np.nan)
                        ),
                        "residual_shrinkage_validation_score_cosine_selected": float(
                            shrinkage.get("validation_score_cosine_selected", np.nan)
                        ),
                        "residual_shrinkage_validation_score_cosine_zero": float(
                            shrinkage.get("validation_score_cosine_zero", np.nan)
                        ),
                        "residual_shrinkage_validation_score_r2_at_cosine_lambda": float(
                            shrinkage.get("validation_score_r2_at_cosine_lambda", np.nan)
                        ),
                        "residual_shrinkage_validation_n": int(shrinkage["validation_n"]),
                        "residual_shrinkage_grid": str(shrinkage["lambda_grid"]),
                        "residual_shrinkage_selection_reason": str(shrinkage["selection_reason"]),
                        "feature_scalar_gain_train": float(scalar_gain),
                        "first_pass_observer_mode": "response_only",
                        "first_pass_n_train_samples": int(
                            feature_conditioned_state["first_pass_n_train_samples"]
                        ),
                        "first_pass_response_dim": int(feature_conditioned_state["first_pass_response_dim"]),
                        "tau_hat_source": tau_hat_source,
                        "observation_geometry_source": observation_geometry_source,
                        "feature_conditioned_baseline_residual_variance": (
                            float("nan")
                            if is_pose_known
                            else float(feature_conditioned_state["feature_conditioned_baseline_residual_variance"])
                        ),
                        "feature_conditioned_observation_residual_variance": (
                            float("nan")
                            if is_pose_known
                            else float(feature_conditioned_state["feature_conditioned_observation_residual_variance"])
                        ),
                        "interpretation": spec.interpretation,
                    }
                    model_rows.append(model_row)
                    test_meta = tests.rows.loc[test_mask].reset_index(drop=True)
                    test_indices = np.flatnonzero(test_mask)
                    _append_feature_predictions(
                        prediction_parts,
                        tests=tests,
                        global_indices=test_indices,
                        z_hat=z_hat,
                        z_true=z_true,
                        z_train_mean=z_train_mean,
                        spec=spec,
                        transform=transform,
                        fold=int(fold),
                        n_train_samples=int(residual_model.n_train),
                        prediction_source="residual_update_out_of_fold",
                    )
                    for row_index, meta in enumerate(test_meta.to_dict(orient="records")):
                        global_index = int(test_indices[row_index])
                        row = dict(meta)
                        if _is_pose_known_spec(spec):
                            row.update(feature_conditioned_state["pose_known_tau_metrics_by_row"][global_index])
                            tau_meta = feature_conditioned_state["pose_known_tau_meta_by_row"][global_index]
                        else:
                            row.update(feature_conditioned_state["tau_metrics_by_row"][global_index])
                            tau_meta = feature_conditioned_state["tau_meta_by_row"][global_index]
                        row.update(
                            {
                                "decoder_mode": "linear_gaussian",
                                "observer_mode": spec.slug,
                                "observer_label": spec.label,
                                "train_bank": spec.train_bank,
                                "test_input": spec.test_input,
                                "latent": transform.latent,
                                "feature_space_mode": transform.feature_space_mode,
                                "feature_fit_scope": transform.fit_scope,
                                "feature_preprocessing": transform.preprocessing,
                                "feature_whitened": bool(transform.whitened),
                                "feature_weighted": bool(transform.weighted),
                                "feature_variance_fraction": float(transform.explained_variance_sum),
                                "r2_cv_train_baseline": "source_fold_train_feature_mean",
                                "fold": int(fold),
                                "n_train_samples": int(residual_model.n_train),
                                "n_fit_sources": int(transform.n_fit_sources),
                                "feature_update_mode": feature_update_mode,
                                "first_pass_base_observer": "internal_response_only_z0_hat",
                                "residual_shrinkage_lambda": residual_shrinkage_lambda,
                                "residual_shrinkage_validation_score": float(shrinkage["validation_score"]),
                                "residual_shrinkage_validation_score_zero": float(shrinkage["validation_score_zero"]),
                                "residual_shrinkage_validation_score_best": float(shrinkage["validation_score_best"]),
                                "residual_shrinkage_lambda_cosine_selected": float(
                                    shrinkage.get("lambda_cosine_selected", np.nan)
                                ),
                                "residual_shrinkage_validation_score_cosine_selected": float(
                                    shrinkage.get("validation_score_cosine_selected", np.nan)
                                ),
                                "residual_shrinkage_validation_score_cosine_zero": float(
                                    shrinkage.get("validation_score_cosine_zero", np.nan)
                                ),
                                "residual_shrinkage_validation_score_r2_at_cosine_lambda": float(
                                    shrinkage.get("validation_score_r2_at_cosine_lambda", np.nan)
                                ),
                                "residual_shrinkage_validation_n": int(shrinkage["validation_n"]),
                                "residual_shrinkage_grid": str(shrinkage["lambda_grid"]),
                                "residual_shrinkage_selection_reason": str(shrinkage["selection_reason"]),
                                "tau_hat_source": str(tau_meta.get("tau_hat_source", "")),
                                "observation_geometry_source": str(tau_meta.get("observation_geometry_source", "")),
                                "first_pass_observer_mode": "response_only",
                                "feature_conditioned_baseline_residual_variance": float(
                                    tau_meta.get("feature_conditioned_baseline_residual_variance", np.nan)
                                ),
                                "feature_conditioned_observation_residual_variance": float(
                                    tau_meta.get("feature_conditioned_observation_residual_variance", np.nan)
                                ),
                                "feature_conditioned_observation_variance": float(
                                    tau_meta.get("feature_conditioned_observation_variance", np.nan)
                                ),
                                "feature_conditioned_tau_optimizer_success": (
                                    float("nan") if is_pose_known else bool(tau_meta.get("optimizer_success", False))
                                ),
                                "feature_conditioned_tau_optimizer_iterations": int(
                                    -1 if is_pose_known else tau_meta.get("optimizer_iterations", -1)
                                ),
                                "uses_response_table_trajectory_training_rows": False,
                            }
                        )
                        first_pass_metrics = _metrics(z0_test[row_index], z_true[row_index], train_mean=z_train_mean)
                        final_metrics = _metrics(z_hat[row_index], z_true[row_index], train_mean=z_train_mean)
                        row.update(_prefixed_metric_fields(first_pass_metrics, "first_pass"))
                        row.update(final_metrics)
                        row.update(
                            {
                                "known_nested_sse_minus_first_pass": float(
                                    final_metrics["feature_sse"] - first_pass_metrics["feature_sse"]
                                ),
                                "known_nested_sse_le_first_pass": bool(
                                    final_metrics["feature_sse"] <= first_pass_metrics["feature_sse"] + 1e-9
                                ),
                            }
                        )
                        row.update(
                            _gain_calibrated_metrics(
                                z_hat[row_index],
                                z_true[row_index],
                                train_mean=z_train_mean,
                                gain=scalar_gain,
                            )
                        )
                        trial_rows.append(row)
                    continue

                bank_z = _transform_feature_sources(transform, feature_table, bank.source_rows)
                z_true = _transform_feature_sources(
                    transform,
                    feature_table,
                    tests.rows.loc[test_mask, "true_source_row"].to_numpy(dtype=int),
                )
                model: ForwardPosteriorModel = _fit_forward_posterior(
                    z_train=bank_z[train_mask],
                    x_train=bank.x[train_mask],
                    ridge=float(ridge),
                    noise_floor=float(noise_floor),
                )
                z_hat = _predict_z(model, x_all[test_mask])
                z_hat_train_uncalibrated = _predict_z(model, bank.x[train_mask])
                scalar_gain = _fit_scalar_gain(z_hat_train_uncalibrated, bank_z[train_mask])
                model_row = {
                    "decoder_mode": "linear_gaussian",
                    "observer_mode": spec.slug,
                    "observer_label": spec.label,
                    "train_bank": spec.train_bank,
                    "test_input": spec.test_input,
                    "latent": transform.latent,
                    "feature_space_mode": transform.feature_space_mode,
                    "fold": int(fold),
                    "n_fit_sources": int(transform.n_fit_sources),
                    "n_test_rows": int(np.sum(test_mask)),
                    "n_train_samples": int(model.n_train),
                    "feature_dim": int(transform.feature_dim),
                    "raw_feature_dim": int(transform.raw_feature_dim),
                    "response_dim": int(bank.x.shape[1]),
                    "feature_fit_scope": transform.fit_scope,
                    "feature_preprocessing": transform.preprocessing,
                    "feature_whitened": bool(transform.whitened),
                    "feature_weighted": bool(transform.weighted),
                    "feature_variance_fraction": float(transform.explained_variance_sum),
                    "r2_cv_train_baseline": "source_fold_train_feature_mean",
                    "ridge": float(model.ridge),
                    "noise_variance": float(model.noise_variance),
                    "response_map_fro_norm": float(np.linalg.norm(model.response_map)),
                    "posterior_gain_fro_norm": float(np.linalg.norm(model.posterior_gain)),
                    "feature_scalar_gain_train": float(scalar_gain),
                    "interpretation": spec.interpretation,
                }
                if _is_feature_conditioned_spec(spec) and feature_conditioned_state is not None:
                    model_row.update(
                        {
                            "first_pass_observer_mode": "response_only",
                            "first_pass_n_train_samples": int(
                                feature_conditioned_state["first_pass_n_train_samples"]
                            ),
                            "first_pass_response_dim": int(feature_conditioned_state["first_pass_response_dim"]),
                            "tau_hat_source": "response_only_feature_conditioned_synthetic_prior_full_path_map",
                            "observation_geometry_source": (
                                "source_disjoint_scale_and_feature_conditioned_response_table_local_response_vs_displacement_map"
                            ),
                            "feature_conditioned_baseline_residual_variance": float(
                                feature_conditioned_state["feature_conditioned_baseline_residual_variance"]
                            ),
                            "feature_conditioned_observation_residual_variance": float(
                                feature_conditioned_state["feature_conditioned_observation_residual_variance"]
                            ),
                        }
                    )
                model_rows.append(model_row)
                test_meta = tests.rows.loc[test_mask].reset_index(drop=True)
                test_indices = np.flatnonzero(test_mask)
                _append_feature_predictions(
                    prediction_parts,
                    tests=tests,
                    global_indices=test_indices,
                    z_hat=z_hat,
                    z_true=z_true,
                    z_train_mean=z_train_mean,
                    spec=spec,
                    transform=transform,
                    fold=int(fold),
                    n_train_samples=int(model.n_train),
                    prediction_source="linear_observer_out_of_fold",
                )
                for row_index, meta in enumerate(test_meta.to_dict(orient="records")):
                    row = dict(meta)
                    if _is_feature_conditioned_spec(spec) and feature_conditioned_state is not None:
                        global_index = int(test_indices[row_index])
                        row.update(feature_conditioned_state["tau_metrics_by_row"][global_index])
                        tau_meta = feature_conditioned_state["tau_meta_by_row"][global_index]
                        row.update(
                            {
                                "tau_hat_source": str(tau_meta.get("tau_hat_source", "")),
                                "observation_geometry_source": str(tau_meta.get("observation_geometry_source", "")),
                                "first_pass_observer_mode": "response_only",
                                "feature_conditioned_baseline_residual_variance": float(
                                    tau_meta.get(
                                        "feature_conditioned_baseline_residual_variance",
                                        np.nan,
                                    )
                                ),
                                "feature_conditioned_observation_residual_variance": float(
                                    tau_meta.get(
                                        "feature_conditioned_observation_residual_variance",
                                        np.nan,
                                    )
                                ),
                                "feature_conditioned_observation_variance": float(
                                    tau_meta.get("feature_conditioned_observation_variance", np.nan)
                                ),
                                "feature_conditioned_tau_optimizer_success": bool(
                                    tau_meta.get("optimizer_success", False)
                                ),
                                "feature_conditioned_tau_optimizer_iterations": int(
                                    tau_meta.get("optimizer_iterations", -1)
                                ),
                                "uses_response_table_trajectory_training_rows": False,
                            }
                        )
                    elif str(spec.slug) in {"response_only", "zero_static", "known_eye_response_only"}:
                        row.update(
                            {
                                "trajectory_rmse": float("nan"),
                                "trajectory_corr_x": float("nan"),
                                "trajectory_corr_y": float("nan"),
                                "trajectory_corr_mean": float("nan"),
                                "trajectory_r2": float("nan"),
                                "tau_hat_source": "none",
                                "observation_geometry_source": "none",
                            }
                        )
                    elif str(spec.slug).startswith("true_tau_"):
                        row.update(
                            {
                                "trajectory_rmse": 0.0,
                                "trajectory_corr_x": 1.0,
                                "trajectory_corr_y": 1.0,
                                "trajectory_corr_mean": 1.0,
                                "trajectory_r2": 1.0,
                                "tau_hat_source": "observed_recorded_eye_trace",
                                "observation_geometry_source": "none",
                            }
                        )
                    row.update(
                        {
                            "decoder_mode": "linear_gaussian",
                            "observer_mode": spec.slug,
                            "observer_label": spec.label,
                            "train_bank": spec.train_bank,
                            "test_input": spec.test_input,
                            "latent": transform.latent,
                            "feature_space_mode": transform.feature_space_mode,
                            "feature_fit_scope": transform.fit_scope,
                            "feature_preprocessing": transform.preprocessing,
                            "feature_whitened": bool(transform.whitened),
                            "feature_weighted": bool(transform.weighted),
                            "feature_variance_fraction": float(transform.explained_variance_sum),
                            "r2_cv_train_baseline": "source_fold_train_feature_mean",
                            "fold": int(fold),
                            "n_train_samples": int(model.n_train),
                            "n_fit_sources": int(transform.n_fit_sources),
                        }
                    )
                    row.update(_metrics(z_hat[row_index], z_true[row_index], train_mean=z_train_mean))
                    row.update(
                        _gain_calibrated_metrics(
                            z_hat[row_index],
                            z_true[row_index],
                            train_mean=z_train_mean,
                            gain=scalar_gain,
                        )
                    )
                    trial_rows.append(row)
    if not trial_rows:
        raise ValueError("No valid linear feature observer trials were produced")
    trials = _add_crossfold_scalar_gain_calibration(pd.DataFrame(trial_rows))
    prediction_rows, prediction_arrays = _finalize_feature_predictions(prediction_parts)
    return trials, pd.DataFrame(model_rows), pd.DataFrame(fit_rows), prediction_rows, prediction_arrays


def _summarize(trials: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "decoder_mode",
        "latent",
        "feature_space_mode",
        "observer_mode",
        "observer_label",
        "observation_scale",
        "prior_family",
    ]
    summary = (
        trials.groupby(group_cols, as_index=False)
        .agg(
            n=("feature_cosine", "size"),
            mean_feature_cosine=("feature_cosine", "mean"),
            median_feature_cosine=("feature_cosine", "median"),
            mean_feature_mse=("feature_mse", "mean"),
            median_feature_mse=("feature_mse", "median"),
            feature_sse=("feature_sse", "sum"),
            feature_sst_train_baseline=("feature_sst_train_baseline", "sum"),
            mean_feature_rmse=("feature_rmse", "mean"),
            median_feature_pred_norm=("feature_pred_norm", "median"),
            median_feature_true_norm=("feature_true_norm", "median"),
            median_trajectory_rmse=("trajectory_rmse", "median"),
            median_trajectory_corr_mean=("trajectory_corr_mean", "median"),
        )
        .sort_values(["observation_scale", "prior_family", "observer_mode"])
    )
    overall = (
        trials.groupby(["decoder_mode", "latent", "feature_space_mode", "observer_mode", "observer_label"], as_index=False)
        .agg(
            n=("feature_cosine", "size"),
            mean_feature_cosine=("feature_cosine", "mean"),
            median_feature_cosine=("feature_cosine", "median"),
            mean_feature_mse=("feature_mse", "mean"),
            median_feature_mse=("feature_mse", "median"),
            feature_sse=("feature_sse", "sum"),
            feature_sst_train_baseline=("feature_sst_train_baseline", "sum"),
            mean_feature_rmse=("feature_rmse", "mean"),
            median_feature_pred_norm=("feature_pred_norm", "median"),
            median_feature_true_norm=("feature_true_norm", "median"),
            median_trajectory_rmse=("trajectory_rmse", "median"),
            median_trajectory_corr_mean=("trajectory_corr_mean", "median"),
        )
        .sort_values("observer_mode")
    )
    summary["R2_cv"] = 1.0 - summary["feature_sse"] / summary["feature_sst_train_baseline"]
    summary.loc[summary["feature_sst_train_baseline"] <= 1e-12, "R2_cv"] = np.nan
    overall["R2_cv"] = 1.0 - overall["feature_sse"] / overall["feature_sst_train_baseline"]
    overall.loc[overall["feature_sst_train_baseline"] <= 1e-12, "R2_cv"] = np.nan
    overall["observation_scale"] = "all"
    overall["prior_family"] = "all"
    out = pd.concat([summary, overall[summary.columns]], ignore_index=True)
    calibrated_cols = {"feature_sse_gain_calibrated", "feature_sst_gain_calibrated_train_baseline"}
    if calibrated_cols.issubset(trials.columns):
        cal_trials = trials[
            np.isfinite(trials["feature_sse_gain_calibrated"])
            & np.isfinite(trials["feature_sst_gain_calibrated_train_baseline"])
        ].copy()
        if not cal_trials.empty:
            cal_summary = (
                cal_trials.groupby(group_cols, as_index=False)
                .agg(
                    n_gain_calibrated=("feature_sse_gain_calibrated", "size"),
                    feature_sse_gain_calibrated=("feature_sse_gain_calibrated", "sum"),
                    feature_sst_gain_calibrated_train_baseline=(
                        "feature_sst_gain_calibrated_train_baseline",
                        "sum",
                    ),
                    median_feature_pred_norm_gain_calibrated=("feature_pred_norm_gain_calibrated", "median"),
                    median_feature_scalar_gain_train=("feature_scalar_gain_train", "median"),
                )
            )
            cal_overall = (
                cal_trials.groupby(
                    ["decoder_mode", "latent", "feature_space_mode", "observer_mode", "observer_label"],
                    as_index=False,
                )
                .agg(
                    n_gain_calibrated=("feature_sse_gain_calibrated", "size"),
                    feature_sse_gain_calibrated=("feature_sse_gain_calibrated", "sum"),
                    feature_sst_gain_calibrated_train_baseline=(
                        "feature_sst_gain_calibrated_train_baseline",
                        "sum",
                    ),
                    median_feature_pred_norm_gain_calibrated=("feature_pred_norm_gain_calibrated", "median"),
                    median_feature_scalar_gain_train=("feature_scalar_gain_train", "median"),
                )
            )
            cal_overall["observation_scale"] = "all"
            cal_overall["prior_family"] = "all"
            cal = pd.concat([cal_summary, cal_overall[cal_summary.columns]], ignore_index=True)
            cal["R2_cv_gain_calibrated"] = (
                1.0
                - cal["feature_sse_gain_calibrated"]
                / cal["feature_sst_gain_calibrated_train_baseline"]
            )
            cal.loc[
                cal["feature_sst_gain_calibrated_train_baseline"] <= 1e-12,
                "R2_cv_gain_calibrated",
            ] = np.nan
            out = out.merge(cal, on=group_cols, how="left")
    cv_calibrated_cols = {
        "feature_sse_cv_gain_calibrated",
        "feature_sst_cv_gain_calibrated_train_baseline",
    }
    if cv_calibrated_cols.issubset(trials.columns):
        cv_cal_trials = trials[
            np.isfinite(trials["feature_sse_cv_gain_calibrated"])
            & np.isfinite(trials["feature_sst_cv_gain_calibrated_train_baseline"])
        ].copy()
        if not cv_cal_trials.empty:
            cv_cal_summary = (
                cv_cal_trials.groupby(group_cols, as_index=False)
                .agg(
                    n_cv_gain_calibrated=("feature_sse_cv_gain_calibrated", "size"),
                    feature_sse_cv_gain_calibrated=("feature_sse_cv_gain_calibrated", "sum"),
                    feature_sst_cv_gain_calibrated_train_baseline=(
                        "feature_sst_cv_gain_calibrated_train_baseline",
                        "sum",
                    ),
                    median_feature_pred_norm_cv_gain_calibrated=(
                        "feature_pred_norm_cv_gain_calibrated",
                        "median",
                    ),
                    median_feature_cv_scalar_gain=("feature_cv_scalar_gain", "median"),
                )
            )
            cv_cal_overall = (
                cv_cal_trials.groupby(
                    ["decoder_mode", "latent", "feature_space_mode", "observer_mode", "observer_label"],
                    as_index=False,
                )
                .agg(
                    n_cv_gain_calibrated=("feature_sse_cv_gain_calibrated", "size"),
                    feature_sse_cv_gain_calibrated=("feature_sse_cv_gain_calibrated", "sum"),
                    feature_sst_cv_gain_calibrated_train_baseline=(
                        "feature_sst_cv_gain_calibrated_train_baseline",
                        "sum",
                    ),
                    median_feature_pred_norm_cv_gain_calibrated=(
                        "feature_pred_norm_cv_gain_calibrated",
                        "median",
                    ),
                    median_feature_cv_scalar_gain=("feature_cv_scalar_gain", "median"),
                )
            )
            cv_cal_overall["observation_scale"] = "all"
            cv_cal_overall["prior_family"] = "all"
            cv_cal = pd.concat(
                [cv_cal_summary, cv_cal_overall[cv_cal_summary.columns]],
                ignore_index=True,
            )
            cv_cal["R2_cv_cv_gain_calibrated"] = (
                1.0
                - cv_cal["feature_sse_cv_gain_calibrated"]
                / cv_cal["feature_sst_cv_gain_calibrated_train_baseline"]
            )
            cv_cal.loc[
                cv_cal["feature_sst_cv_gain_calibrated_train_baseline"] <= 1e-12,
                "R2_cv_cv_gain_calibrated",
            ] = np.nan
            out = out.merge(cv_cal, on=group_cols, how="left")
    return out


def _contrasts(trials: pd.DataFrame, *, n_boot: int, seed: int) -> pd.DataFrame:
    key_cols = [
        "decoder_mode",
        "latent",
        "feature_space_mode",
        "table_index",
        "trial_id",
        "observation_scale",
        "prior_family",
        "true_source_row",
    ]
    pivot = trials.pivot_table(index=key_cols, columns="observer_mode", values="feature_cosine", aggfunc="first")
    pairs = [
        ("pose_known_forward_model", "response_only", "pose_known_forward_model_minus_response_only"),
        ("hidden_joint_forward_model", "response_only", "hidden_joint_forward_model_minus_response_only"),
        (
            "estimated_tau_forward_model",
            "response_only",
            "estimated_tau_forward_model_minus_response_only",
        ),
        ("zero_tau_forward_model", "response_only", "zero_tau_forward_model_minus_response_only"),
        (
            "pose_known_forward_model",
            "hidden_joint_forward_model",
            "pose_known_forward_model_minus_hidden_joint_forward_model",
        ),
        (
            "hidden_joint_forward_model",
            "estimated_tau_forward_model",
            "hidden_joint_forward_model_minus_estimated_tau_forward_model",
        ),
        (
            "pose_known_nested_tau_interactions",
            "response_only",
            "pose_known_nested_tau_interactions_minus_response_only",
        ),
        (
            "pose_known_nested_tau_interactions",
            "feature_conditioned_tau_interactions",
            "pose_known_nested_limit_minus_estimated_tau",
        ),
        ("pose_known_tau_linear", "response_only", "pose_known_tau_minus_response_only"),
        (
            "pose_known_tau_interactions",
            "response_only",
            "pose_known_tau_interactions_minus_response_only",
        ),
        (
            "pose_known_tau_interactions",
            "feature_conditioned_tau_interactions",
            "pose_known_upper_limit_minus_estimated_tau",
        ),
        ("feature_conditioned_tau_linear", "response_only", "feature_conditioned_tau_minus_response_only"),
        (
            "feature_conditioned_tau_interactions",
            "response_only",
            "feature_conditioned_tau_interactions_minus_response_only",
        ),
        (
            "feature_conditioned_tau_linear",
            "synthetic_tau_linear",
            "feature_conditioned_tau_minus_pooled_synthetic_tau",
        ),
        ("synthetic_tau_linear", "response_only", "synthetic_tau_minus_response_only"),
        ("synthetic_tau_interactions", "response_only", "synthetic_tau_interactions_minus_response_only"),
        ("synthetic_tau_interactions", "synthetic_tau_linear", "interactions_minus_linear_tau"),
        ("true_tau_linear", "synthetic_tau_linear", "true_tau_minus_synthetic_tau"),
        ("synthetic_tau_linear", "zero_static", "synthetic_tau_minus_zero_static"),
    ]
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    for lhs, rhs, contrast in pairs:
        if lhs not in pivot.columns or rhs not in pivot.columns:
            continue
        vals = (pivot[lhs] - pivot[rhs]).rename("delta").reset_index()
        vals = vals[np.isfinite(vals["delta"].to_numpy(dtype=float))]
        for (decoder_mode, latent, feature_space_mode), mode_rows in vals.groupby(
            ["decoder_mode", "latent", "feature_space_mode"],
            sort=True,
        ):
            for scale_value, scale_rows in mode_rows.groupby("observation_scale", sort=True):
                values = scale_rows["delta"].to_numpy(dtype=float)
                mean, lo, hi = _bootstrap_mean(values, rng, int(n_boot))
                rows.append(
                    {
                        "decoder_mode": str(decoder_mode),
                        "latent": str(latent),
                        "feature_space_mode": str(feature_space_mode),
                        "contrast": contrast,
                        "lhs": lhs,
                        "rhs": rhs,
                        "observation_scale": float(scale_value),
                        "prior_family": "all",
                        "mean_feature_cosine_delta": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "fraction_positive": float(np.mean(values > 0.0)) if values.size else float("nan"),
                        "n": int(values.size),
                    }
                )
            values = mode_rows["delta"].to_numpy(dtype=float)
            mean, lo, hi = _bootstrap_mean(values, rng, int(n_boot))
            rows.append(
                {
                    "decoder_mode": str(decoder_mode),
                    "latent": str(latent),
                    "feature_space_mode": str(feature_space_mode),
                    "contrast": contrast,
                    "lhs": lhs,
                    "rhs": rhs,
                    "observation_scale": "all",
                    "prior_family": "all",
                    "mean_feature_cosine_delta": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "fraction_positive": float(np.mean(values > 0.0)) if values.size else float("nan"),
                    "n": int(values.size),
                }
            )
    return pd.DataFrame(rows)


def _observer_sort_key(name: str) -> tuple[int, str]:
    return (OBSERVER_ORDER.index(name) if name in OBSERVER_ORDER else len(OBSERVER_ORDER), name)


def _filter_plot_summary(summary: pd.DataFrame, *, latent: str, feature_space_mode: str) -> tuple[pd.DataFrame, str]:
    frame = summary[summary["latent"].astype(str) == str(latent)].copy()
    if frame.empty:
        return summary.copy(), str(feature_space_mode)
    if str(feature_space_mode) in set(frame["feature_space_mode"].astype(str)):
        return frame[frame["feature_space_mode"].astype(str) == str(feature_space_mode)].copy(), str(feature_space_mode)
    chosen = str(frame["feature_space_mode"].iloc[0])
    return frame[frame["feature_space_mode"].astype(str) == chosen].copy(), chosen


def _plot(
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    out_dir: Path,
    *,
    latent: str,
    feature_space_mode: str,
) -> tuple[Path, Path, str]:
    _configure_matplotlib()
    plot_summary, plotted_mode = _filter_plot_summary(summary, latent=latent, feature_space_mode=feature_space_mode)
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.0), constrained_layout=True)

    overall = plot_summary[plot_summary["observation_scale"].astype(str) == "all"].copy()
    overall["observer_sort"] = overall["observer_mode"].map(_observer_sort_key)
    overall = overall.sort_values("observer_sort")
    ax = axes[0]
    x = np.arange(overall.shape[0], dtype=float)
    colors = [OBSERVER_COLORS.get(str(mode), "#64748b") for mode in overall["observer_mode"]]
    ax.bar(x, overall["mean_feature_cosine"].to_numpy(dtype=float), color=colors, width=0.7)
    ax.set_xticks(x, [str(value) for value in overall["observer_label"]], rotation=35, ha="right")
    ax.set_ylabel("feature cosine")
    ax.set_title("A. all scales")
    _clean_axis(ax)

    scale_rows = plot_summary[plot_summary["observation_scale"].astype(str) != "all"].copy()
    ax = axes[1]
    for observer in sorted(set(scale_rows["observer_mode"].astype(str)), key=_observer_sort_key):
        block = scale_rows[scale_rows["observer_mode"].astype(str) == observer].copy()
        if block.empty:
            continue
        block = block.sort_values("observation_scale")
        ax.plot(
            block["observation_scale"].to_numpy(dtype=float),
            block["mean_feature_cosine"].to_numpy(dtype=float),
            marker="o",
            lw=1.5,
            color=OBSERVER_COLORS.get(observer, "#64748b"),
            label=OBSERVER_LABELS.get(observer, observer),
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.5, 1.0, 2.0], ["0.5x", "1x", "2x"])
    ax.set_title("B. by scale")
    ax.set_ylabel("feature cosine")
    ax.legend(frameon=False, loc="best")
    _clean_axis(ax)

    png = out_dir / "linear_synthetic_prior_feature_observer.png"
    pdf = out_dir / "linear_synthetic_prior_feature_observer.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf, plotted_mode


def _readme_value(summary: pd.DataFrame, observer: str, scale: str = "all") -> float:
    frame = summary[
        (summary["observer_mode"].astype(str) == observer)
        & (summary["observation_scale"].astype(str) == str(scale))
        & (summary["prior_family"].astype(str) == "all")
    ]
    if frame.empty:
        frame = summary[
            (summary["observer_mode"].astype(str) == observer)
            & (summary["observation_scale"].astype(str) == str(scale))
        ]
    if frame.empty:
        return float("nan")
    return float(frame["mean_feature_cosine"].mean())


def _write_readme(
    *,
    out_dir: Path,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    lines = [
        "# Linear Synthetic-Prior Feature Observer",
        "",
        "This diagnostic decodes a continuous feature embedding with a linear-Gaussian",
        "observer. Its default two-pass modes first fit response-only `z0_hat`,",
        "use `z0_hat` plus observation scale to condition the compact displacement",
        "geometry, infer `tau_hat` with the `synthetic_empirical_confined` transition",
        "prior, then apply a tau-driven residual update to `z0_hat`.",
        "",
        "Key interpretation boundary:",
        "",
        "- Feature endpoint: candidate-free, no image posterior averaging.",
        "- Catalog policy: finite image-catalog search is avoided for the main endpoint; response-table rows are used only for source-disjoint geometry calibration and diagnostics.",
        "- Default tau endpoint: candidate-free response-only-feature-and-scale-conditioned full-path MAP, not the true-image branch.",
        "- Default feature endpoint: response-only `z0_hat` plus a linear tau residual update.",
        "- Recorded-tau nested diagnostic: same residual-update decoder, replacing inferred `tau_hat` with the recorded eye trace, with source-disjoint validated shrinkage and explicit response-only fallback.",
        "- Raw recorded-tau residual modes are retained as non-nested diagnostics.",
        "- Pooled `synthetic_tau_*` modes are retained as candidate-free baselines.",
        "- Decoder: linear-Gaussian only; no nonlinear MLP.",
        "- Trajectory prior: `synthetic_empirical_confined` uses an empirically calibrated synthetic FEM prior.",
        "- Calibration: response-table samples still fit response-vs-displacement geometry, source-disjoint by fold for the feature-conditioned modes.",
        "- Feature training rows: observed response-table rows only, source-row disjoint.",
        "",
        "All-scale mean feature cosine:",
        "",
        "```text",
        f"response only:                 {_readme_value(summary, 'response_only'):.4f}",
        f"z0-conditioned tau linear:     {_readme_value(summary, 'feature_conditioned_tau_linear'):.4f}",
        f"z0-conditioned tau x response: {_readme_value(summary, 'feature_conditioned_tau_interactions'):.4f}",
        f"recorded-tau gated linear:     {_readme_value(summary, 'pose_known_nested_tau_linear'):.4f}",
        f"recorded-tau gated x resp:     {_readme_value(summary, 'pose_known_nested_tau_interactions'):.4f}",
        f"recorded-tau residual linear:  {_readme_value(summary, 'pose_known_tau_linear'):.4f}",
        f"recorded-tau residual x resp:  {_readme_value(summary, 'pose_known_tau_interactions'):.4f}",
        f"synthetic tau linear:          {_readme_value(summary, 'synthetic_tau_linear'):.4f}",
        f"synthetic tau interactions:    {_readme_value(summary, 'synthetic_tau_interactions'):.4f}",
        f"recorded-tau linear diag:      {_readme_value(summary, 'true_tau_linear'):.4f}",
        f"0x stabilized:                 {_readme_value(summary, 'zero_static'):.4f}",
        "```",
        "",
        "Outputs:",
        "",
        "- `linear_synthetic_prior_feature_observer_trials.csv`",
        "- `linear_synthetic_prior_feature_observer_summary.csv`",
        "- `linear_synthetic_prior_feature_observer_contrasts.csv`",
        "- `linear_synthetic_prior_feature_observer_models.csv`",
        "- `linear_synthetic_prior_feature_observer_fit_rows.csv`",
        "- `linear_synthetic_prior_feature_observer_manifest.json`",
        "- `linear_synthetic_prior_feature_observer.png`",
    ]
    prediction_outputs = manifest.get("feature_prediction_outputs", {})
    if isinstance(prediction_outputs, dict) and prediction_outputs.get("prediction_rows"):
        lines.extend(
            [
                "- `linear_synthetic_prior_feature_observer_prediction_rows.csv`",
                "- `linear_synthetic_prior_feature_observer_prediction_arrays.npz`",
                "- `linear_synthetic_prior_feature_observer_raw_feature_dim_metadata.csv`",
            ]
        )
    (out_dir / "linear_synthetic_prior_feature_observer_README.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--response-manifest", type=Path, default=None)
    parser.add_argument("--trajectory-sidecar-dir", type=Path, default=None)
    parser.add_argument("--trajectory-npz", type=Path, default=None)
    parser.add_argument("--trajectory-key", default="trajectory_xy")
    parser.add_argument("--observed-trajectory-key", default="observed_trajectory_xy")
    parser.add_argument("--feature-npz", type=Path, default=FEATURE_NPZ)
    parser.add_argument("--feature-weights-npz", type=Path, default=None)
    parser.add_argument("--latent", default=PRIMARY_LATENT)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--feature-space-modes", default="fold_zscore_whitened_pca")
    parser.add_argument("--observer-modes", default=",".join(DEFAULT_OBSERVER_MODES))
    parser.add_argument("--compact-basis-path", type=Path, default=COMPACT_BASIS)
    parser.add_argument("--basis-key", default="basis")
    parser.add_argument("--basis-max-dim", type=int, default=20)
    parser.add_argument("--linear-ridge", type=float, default=1e-2)
    parser.add_argument("--noise-floor", type=float, default=1e-8)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260624)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--scales", default="0.5,1.0,2.0")
    parser.add_argument("--prior-family-filter", default="")
    parser.add_argument("--skip-tables", type=int, default=0)
    parser.add_argument("--max-tables", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--save-feature-predictions",
        action="store_true",
        help=(
            "Save out-of-fold z_true/z_hat arrays and raw-coordinate PCA projections "
            "for feature-group calibration diagnostics."
        ),
    )

    parser.add_argument("--alpha", type=float, default=0.92)
    parser.add_argument("--process-var", type=float, default=1e-3)
    parser.add_argument("--observation-var", type=float, default=None)
    parser.add_argument("--observation-var-floor", type=float, default=1e-6)
    parser.add_argument("--forward-model-z-prior-precision", type=float, default=1.0)
    parser.add_argument("--forward-model-joint-iterations", type=int, default=3)
    parser.add_argument("--continuous-ridge", type=float, default=1e-6)
    parser.add_argument("--continuous-ridge-by-scale", default="")
    parser.add_argument("--observation-model", default="time_constant")
    parser.add_argument("--time-smoothing-sigma", type=float, default=0.0)
    parser.add_argument("--time-shrinkage", type=float, default=0.0)
    parser.add_argument("--continuous-score-mode", default="quadratic_affine_poisson_profile")
    parser.add_argument("--trajectory-prior-mean", default="zero")
    parser.add_argument("--trajectory-initial-position", default="inferred")
    parser.add_argument("--trajectory-initial-position-var", type=float, default=1e-4)
    parser.add_argument("--trajectory-process-model", default="synthetic_empirical_confined")
    parser.add_argument("--trajectory-process-model-by-scale", default="")
    parser.add_argument("--brownian-cov-floor", type=float, default=1e-6)
    parser.add_argument("--brownian-cov-scale", type=float, default=1.0)
    parser.add_argument("--brownian-cov-scale-by-scale", default="")
    parser.add_argument("--synthetic-prior-samples", type=int, default=512)
    parser.add_argument("--synthetic-prior-kappa-weight-power", type=float, default=0.5)
    parser.add_argument("--synthetic-prior-seed", type=int, default=20260627)
    parser.add_argument("--catalog-gaussian-smoothing-sigma", type=float, default=0.0)
    parser.add_argument("--catalog-gaussian-cov-floor", type=float, default=1e-6)
    parser.add_argument("--catalog-gaussian-shrinkage", type=float, default=0.25)
    parser.add_argument("--trajectory-basis-family", default="dct")
    parser.add_argument("--trajectory-basis-components", type=int, default=4)
    parser.add_argument("--trajectory-basis-smoothing-sigma", type=float, default=6.0)
    parser.add_argument("--trajectory-basis-coeff-prior-var", type=float, default=1.0)
    parser.add_argument("--catalog-residual-aggregation", default="logmeanexp")
    parser.add_argument("--catalog-residual-top-k", type=int, default=0)
    parser.add_argument("--catalog-residual-all-anchor-shrinkage", type=float, default=0.0)
    parser.add_argument("--catalog-residual-anchor-smoothing-sigma", type=float, default=0.0)
    parser.add_argument("--catalog-residual-anchor-smoothing-schedule", default="0")
    parser.add_argument("--catalog-residual-refine-top-k", type=int, default=0)
    parser.add_argument("--quadratic-optimizer-max-iter", type=int, default=80)
    parser.add_argument("--quadratic-continuation-scales", default="1")
    parser.add_argument("--quadratic-observation-scales", default="1")
    parser.add_argument("--quadratic-intercept-ridge-multiplier", type=float, default=1.0)
    parser.add_argument("--quadratic-affine-intercept-scale", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--likelihood-scale", type=float, default=1.0)
    return parser


def build(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir)
    manifest_path = Path(args.response_manifest) if args.response_manifest else run_dir / "response_cache_manifest.csv"
    manifest = _read_manifest(
        manifest_path,
        scales=_parse_scales(str(args.scales)),
        prior_family_filter=_parse_csv_values(str(args.prior_family_filter)),
        skip_tables=int(args.skip_tables),
        max_tables=int(args.max_tables),
    )

    sidecar_dir = Path(args.trajectory_sidecar_dir) if args.trajectory_sidecar_dir is not None else run_dir / "continuous_joint_trajectory_sidecars"
    if not sidecar_dir.exists():
        sidecar_dir = None
    trajectory_npz = _load_npz(Path(args.trajectory_npz)) if args.trajectory_npz is not None else None

    first_table = _load_table_with_sidecar(
        run_dir=run_dir,
        response_cache_path=str(manifest.iloc[0]["response_cache_path"]),
        trajectory_sidecar_dir=sidecar_dir,
    )
    n_units = int(np.asarray(first_table["y_obs_counts"]).shape[1])
    basis, basis_meta = _load_basis(
        Path(args.compact_basis_path),
        n_units=n_units,
        basis_key=str(args.basis_key),
        max_dim=int(args.basis_max_dim),
    )
    feature_table, feature_meta = _load_feature_table(Path(args.feature_npz), latent=str(args.latent))
    feature_weights, feature_weight_meta = _load_feature_weights(
        Path(args.feature_weights_npz) if args.feature_weights_npz is not None else None,
        latent=str(args.latent),
        raw_feature_dim=int(feature_table.features.shape[1]),
    )
    feature_space_modes = _parse_str_list(args.feature_space_modes)
    if not feature_space_modes:
        feature_space_modes = [DEFAULT_FEATURE_SPACE_MODES[0]]
    specs = _selected_specs(_parse_str_list(args.observer_modes))
    compute_pooled_tau_hat = any(
        spec.slug in {"synthetic_tau_linear", "synthetic_tau_interactions"} for spec in specs
    )

    banks, tests, fit_rows = _build_sample_banks(
        run_dir=run_dir,
        manifest=manifest,
        basis=basis,
        feature_sources=set(int(value) for value in feature_table.source_rows.tolist()),
        trajectory_npz=trajectory_npz,
        trajectory_key=str(args.trajectory_key),
        observed_trajectory_key=str(args.observed_trajectory_key),
        trajectory_sidecar_dir=sidecar_dir,
        continuous_args=args,
        compute_pooled_tau_hat=compute_pooled_tau_hat,
        progress_every=int(args.progress_every),
    )
    trials, models, feature_conditioned_fit_rows, prediction_rows, prediction_arrays = _run_crossfit(
        banks=banks,
        tests=tests,
        feature_table=feature_table,
        feature_weights=feature_weights,
        feature_space_modes=feature_space_modes,
        specs=specs,
        feature_dim=int(args.feature_dim),
        n_folds=int(args.n_folds),
        fold_seed=int(args.fold_seed),
        ridge=float(args.linear_ridge),
        noise_floor=float(args.noise_floor),
        continuous_args=args,
        save_feature_predictions=bool(args.save_feature_predictions),
    )
    if not feature_conditioned_fit_rows.empty:
        fit_rows = pd.concat([fit_rows, feature_conditioned_fit_rows], ignore_index=True, sort=False)
    summary = _summarize(trials)
    contrasts = _contrasts(trials, n_boot=int(args.n_bootstrap), seed=int(args.fold_seed) + 17)

    trials_path = out_dir / "linear_synthetic_prior_feature_observer_trials.csv"
    summary_path = out_dir / "linear_synthetic_prior_feature_observer_summary.csv"
    contrasts_path = out_dir / "linear_synthetic_prior_feature_observer_contrasts.csv"
    models_path = out_dir / "linear_synthetic_prior_feature_observer_models.csv"
    fit_rows_path = out_dir / "linear_synthetic_prior_feature_observer_fit_rows.csv"
    prediction_rows_path = out_dir / "linear_synthetic_prior_feature_observer_prediction_rows.csv"
    prediction_arrays_path = out_dir / "linear_synthetic_prior_feature_observer_prediction_arrays.npz"
    raw_feature_metadata_path = out_dir / "linear_synthetic_prior_feature_observer_raw_feature_dim_metadata.csv"
    trials.to_csv(trials_path, index=False)
    summary.to_csv(summary_path, index=False)
    contrasts.to_csv(contrasts_path, index=False)
    models.to_csv(models_path, index=False)
    fit_rows.to_csv(fit_rows_path, index=False)
    prediction_outputs: dict[str, str | None] = {
        "prediction_rows": None,
        "prediction_arrays": None,
        "raw_feature_dim_metadata": None,
    }
    if bool(args.save_feature_predictions):
        if prediction_rows.empty or not prediction_arrays:
            raise ValueError("--save-feature-predictions was requested but no prediction rows were collected")
        prediction_rows.to_csv(prediction_rows_path, index=False)
        np.savez_compressed(prediction_arrays_path, **prediction_arrays)
        raw_metadata = _local_field_dim_metadata(
            latent=str(args.latent),
            raw_feature_dim=int(feature_table.features.shape[1]),
        )
        raw_metadata.to_csv(raw_feature_metadata_path, index=False)
        prediction_outputs = {
            "prediction_rows": str(prediction_rows_path),
            "prediction_arrays": str(prediction_arrays_path),
            "raw_feature_dim_metadata": str(raw_feature_metadata_path),
        }
    png, pdf, plotted_mode = _plot(
        summary,
        contrasts,
        out_dir,
        latent=str(args.latent),
        feature_space_mode=str(feature_space_modes[0]),
    )

    manifest_payload = {
        "analysis": "linear_synthetic_prior_feature_observer",
        "run_dir": run_dir,
        "response_manifest": manifest_path,
        "trajectory_sidecar_dir": sidecar_dir,
        "trajectory_npz": args.trajectory_npz,
        "n_response_tables": int(manifest.shape[0]),
        "feature": {
            **feature_meta,
            "feature_dim_requested": int(args.feature_dim),
            "feature_space_modes_requested": feature_space_modes,
            "feature_space_modes_canonical": sorted(set(models["feature_space_mode"].astype(str).tolist())),
            "feature_weights": feature_weight_meta,
        },
        "observer_modes": [spec.slug for spec in specs],
        "compute_pooled_tau_hat": bool(compute_pooled_tau_hat),
        "feature_prediction_outputs": prediction_outputs,
        "decoder": {
            "mode": "linear_gaussian",
            "ridge": float(args.linear_ridge),
            "noise_floor": float(args.noise_floor),
            "uses_nonlinear_mlp": False,
        },
        "trajectory_prior": {
            "trajectory_process_model": str(args.trajectory_process_model),
            "trajectory_process_model_by_scale": str(args.trajectory_process_model_by_scale),
            "synthetic_prior_samples": int(args.synthetic_prior_samples),
            "synthetic_prior_kappa_weight_power": float(args.synthetic_prior_kappa_weight_power),
            "synthetic_prior_seed": int(args.synthetic_prior_seed),
            "wording": "synthetic_empirical_confined uses an empirically calibrated synthetic FEM prior",
        },
        "continuous_scorer": {
            "continuous_score_mode": str(args.continuous_score_mode),
            "alpha": float(args.alpha),
            "process_var": float(args.process_var),
            "observation_var": args.observation_var,
            "observation_var_floor": float(args.observation_var_floor),
            "continuous_ridge": float(args.continuous_ridge),
            "continuous_ridge_by_scale": str(args.continuous_ridge_by_scale),
            "observation_model": str(args.observation_model),
            "trajectory_prior_mean": str(args.trajectory_prior_mean),
            "trajectory_initial_position": str(args.trajectory_initial_position),
            "brownian_cov_scale": float(args.brownian_cov_scale),
            "brownian_cov_scale_by_scale": str(args.brownian_cov_scale_by_scale),
            "quadratic_optimizer_max_iter": int(args.quadratic_optimizer_max_iter),
            "quadratic_continuation_scales": str(args.quadratic_continuation_scales),
            "quadratic_observation_scales": str(args.quadratic_observation_scales),
            "forward_model_z_prior_precision": float(args.forward_model_z_prior_precision),
            "forward_model_joint_iterations": int(args.forward_model_joint_iterations),
        },
        "basis": basis_meta,
        "crossfit": {"n_folds": int(args.n_folds), "fold_seed": int(args.fold_seed)},
        "sample_banks": {
            name: {
                "n_samples": int(bank.x.shape[0]),
                "response_dim": int(bank.x.shape[1]),
                "n_source_rows": int(len(set(bank.source_rows.tolist()))),
            }
            for name, bank in banks.items()
        },
        "test_inputs": {name: list(value.shape) for name, value in tests.x_by_mode.items()},
        "dynamic_test_inputs": {
            "pose_known_forward_model": (
                "fixed recorded-tau compact-forward diagnostic: infer z by matching fitted F(z, tau_true) to observed compact response"
            ),
            "hidden_joint_forward_model": (
                "alternating forward-model z/tau MAP under synthetic_empirical_confined tau prior"
            ),
            "estimated_tau_forward_model": (
                "fixed estimated-tau forward-model latent MAP: infer z by matching F(z, tau_hat) to observed compact response"
            ),
            "zero_tau_forward_model": (
                "fixed zero-tau forward-model latent MAP: infer z by matching F(z, 0) to observed compact response"
            ),
            "feature_conditioned_response_tau_hat": (
                "tau residual-update features computed inside each source-disjoint fold from response-only z0_hat"
            ),
            "feature_conditioned_response_tau_hat_interactions": (
                "tau and response-by-tau residual-update features computed inside each source-disjoint fold from response-only z0_hat"
            ),
            "pose_known_response_tau": (
                "recorded-eye tau residual-update features computed inside each source-disjoint fold from response-only z0_hat"
            ),
            "pose_known_response_tau_interactions": (
                "recorded-eye tau and response-by-tau residual-update features computed inside each source-disjoint fold from response-only z0_hat"
            ),
            "pose_known_nested_response_tau": (
                "recorded-eye tau residual-update features with source-disjoint validated shrinkage and zero fallback"
            ),
            "pose_known_nested_response_tau_interactions": (
                "recorded-eye tau and response-by-tau residual-update features with source-disjoint validated shrinkage and zero fallback"
            ),
        },
        "interpretation_boundary": {
            "uses_candidate_posterior_endpoint": False,
            "uses_image_candidate_feature_readout": False,
            "uses_nonlinear_mlp": False,
            "uses_trajectory_catalog_replay_endpoint": False,
            "uses_true_image_conditioned_tau_hat": False,
            "uses_response_table_trajectory_training_rows": True,
            "uses_response_table_trajectory_rows_for_feature_endpoint": False,
            "uses_response_table_trajectory_rows_for_geometry_calibration": True,
            "default_tau_hat_source": "response_only_feature_conditioned_synthetic_prior_full_path_map",
            "pooled_baseline_tau_hat_source": "candidate_free_pooled_response_geometry_full_path_map",
            "default_observation_geometry_source": (
                "source_disjoint_scale_and_feature_conditioned_response_table_local_response_vs_displacement_map"
            ),
            "forward_model_observation_geometry_source": (
                "source_disjoint_scale_and_feature_conditioned_forward_response_model"
            ),
            "pooled_baseline_observation_geometry_source": "pooled_response_table_local_response_vs_displacement_map",
            "first_pass_feature_source": "cross-fit response-only linear-Gaussian z0_hat",
            "default_feature_update": "response_only_z0_hat_plus_linear_tau_residual_update",
            "forward_model_feature_update": "direct latent z MAP by matching compact F(z, tau) to observed response",
            "pose_known_forward_model": (
                "direct z MAP with tau fixed to the recorded trajectory; diagnostic only until the compact F(z, tau) contract is validated"
            ),
            "hidden_joint_forward_model": "alternating z/tau MAP with tau governed by the synthetic_empirical_confined prior",
            "pose_known_nested_limit": (
                "response_only_z0_hat_plus_recorded_eye_trace_tau_residual_update_with_validated_shrinkage_and_zero_fallback"
            ),
            "raw_pose_known_diagnostic": "response_only_z0_hat_plus_recorded_eye_trace_tau_residual_update_without_gate",
            "known_pose_ceiling_status": (
                "not_provided_by_this_runner; residual and compact-forward recorded-tau modes are diagnostics, "
                "not a physical image-conditioned known-pose ceiling"
            ),
            "supervised_feature_rows_source": "observed response-table rows, cross-fit by held-out true source",
        },
        "outputs": {
            "trials": trials_path,
            "summary": summary_path,
            "contrasts": contrasts_path,
            "models": models_path,
            "fit_rows": fit_rows_path,
            "figure_png": png,
            "figure_pdf": pdf,
        },
    }
    manifest_json = out_dir / "linear_synthetic_prior_feature_observer_manifest.json"
    _write_json(manifest_json, manifest_payload)
    _write_readme(out_dir=out_dir, summary=summary, contrasts=contrasts, manifest=manifest_payload)
    print(f"Wrote {out_dir}")
    return out_dir


def main() -> None:
    build(build_parser().parse_args())


if __name__ == "__main__":
    main()
