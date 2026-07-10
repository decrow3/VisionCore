from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from declan.figure4_active_sensing_atlas.scripts.build_panel_c_linear_synthetic_prior_feature_observer import (
    GeometryTable,
    _context_from_z,
    _fit_feature_conditioned_baseline,
    _fit_feature_conditioned_quadratic_observation_map,
    _inverse_transform_scores,
    _local_field_dim_metadata,
    _predict_compact_from_z_tau,
    _quadratic_design_from_path,
    _solve_z_given_tau,
    _validated_residual_shrinkage,
)
from declan.figure4_active_sensing_atlas.scripts.build_panel_c_continuous_feature_embedding_reconstruction import (
    FeatureTable,
    FeatureTransform,
)


def test_fixed_tau_forward_model_recovers_synthetic_feature_latent() -> None:
    rng = np.random.default_rng(123)
    n_time = 6
    basis_dim = 5
    z_dim = 4
    include_intercept = False
    design_dim = 5
    context_dim = 2 + 2 * z_dim
    observation_scale = 0.75
    baseline_coef = rng.normal(scale=0.2, size=(z_dim + 1, n_time * basis_dim))
    observation_coef = rng.normal(scale=0.05, size=(basis_dim, design_dim * context_dim))
    z_true = rng.normal(size=z_dim)
    tau = rng.normal(scale=0.1, size=(n_time, 2))
    geometry_table = GeometryTable(
        source_rows=np.asarray([], dtype=int),
        zero_compact=np.empty((0, n_time, basis_dim), dtype=np.float64),
        prior_delta_compact=np.empty((0, 0, n_time, basis_dim), dtype=np.float64),
        trajectory_xy=np.empty((0, 0, n_time, 2), dtype=np.float64),
        source_samples=np.zeros((1, n_time, 2), dtype=np.float64),
        observation_scale=observation_scale,
    )
    compact = _predict_compact_from_z_tau(
        z=z_true,
        tau=tau,
        observation_scale=observation_scale,
        baseline_coef=baseline_coef,
        observation_coef=observation_coef,
        include_intercept=include_intercept,
    )

    z_hat, compact_hat, meta = _solve_z_given_tau(
        observed_compact=compact,
        tau=tau,
        geometry_table=geometry_table,
        baseline_coef=baseline_coef,
        baseline_residual_var=1e-9,
        observation_coef=observation_coef,
        observation_residual_var=1e-9,
        include_intercept=include_intercept,
        continuous_args=argparse.Namespace(
            observation_var=1e-9,
            observation_var_floor=1e-12,
            forward_model_z_prior_precision=1e-12,
        ),
    )

    assert bool(meta["forward_z_solver_success"])
    np.testing.assert_allclose(z_hat, z_true, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(compact_hat, compact, atol=1e-6, rtol=1e-6)


def test_fitted_feature_conditioned_forward_model_recovers_heldout_latent() -> None:
    rng = np.random.default_rng(456)
    n_sources = 12
    n_train_trajectories = 8
    n_time = 5
    basis_dim = 4
    z_dim = 3
    source_rows = np.arange(100, 100 + n_sources, dtype=int)
    features = rng.normal(size=(n_sources, z_dim))
    scale = 1.2
    include_intercept = False
    design_dim = 5
    context_dim = 2 + 2 * z_dim
    baseline_coef_true = rng.normal(scale=0.15, size=(z_dim + 1, n_time * basis_dim))
    observation_coef_true = rng.normal(scale=0.04, size=(basis_dim, design_dim * context_dim))
    tensor_true = observation_coef_true.reshape(basis_dim, design_dim, context_dim)

    zero_compact = np.empty((n_sources, n_time, basis_dim), dtype=np.float64)
    prior_delta_compact = np.empty((n_sources, n_train_trajectories, n_time, basis_dim), dtype=np.float64)
    trajectory_xy = rng.normal(scale=0.2, size=(n_sources, n_train_trajectories, n_time, 2))
    for source_index, z in enumerate(features):
        zero_compact[source_index] = (_context_from_z(z) @ baseline_coef_true).reshape(n_time, basis_dim)
        geometry_context = np.concatenate(
            [
                np.ones(1, dtype=np.float64),
                np.asarray([scale], dtype=np.float64),
                z,
                scale * z,
            ]
        )
        conditioned = np.einsum("kdc,c->kd", tensor_true, geometry_context)
        for trajectory_index in range(n_train_trajectories):
            design = _quadratic_design_from_path(
                trajectory_xy[source_index, trajectory_index],
                include_intercept=include_intercept,
            )
            prior_delta_compact[source_index, trajectory_index] = design @ conditioned.T

    geometry_table = GeometryTable(
        source_rows=source_rows,
        zero_compact=zero_compact,
        prior_delta_compact=prior_delta_compact,
        trajectory_xy=trajectory_xy,
        source_samples=trajectory_xy.reshape(-1, n_time, 2),
        observation_scale=scale,
    )
    feature_table = FeatureTable(
        feature_npz=Path("synthetic_features.npz"),
        latent="synthetic",
        source_rows=source_rows,
        features=features,
        source_to_index={int(source): index for index, source in enumerate(source_rows.tolist())},
    )
    transform = FeatureTransform(
        latent="synthetic",
        feature_space_mode="identity",
        feature_dim=z_dim,
        mean=np.zeros(z_dim),
        sd=np.ones(z_dim),
        components=np.eye(z_dim),
        denom=np.ones(z_dim),
        weights=None,
        fit_scope="synthetic",
        preprocessing="identity",
        whitened=False,
        weighted=False,
        n_fit_sources=n_sources - 1,
        raw_feature_dim=z_dim,
        explained_variance_sum=1.0,
        explained_variance_first5=[1.0],
    )
    heldout_source = int(source_rows[-1])
    baseline_coef, _baseline_var, _baseline_row = _fit_feature_conditioned_baseline(
        geometry_tables=[geometry_table],
        transform=transform,
        feature_table=feature_table,
        heldout_sources={heldout_source},
        ridge=1e-10,
    )
    observation_coef, _observation_var, _observation_row = _fit_feature_conditioned_quadratic_observation_map(
        geometry_tables=[geometry_table],
        transform=transform,
        feature_table=feature_table,
        heldout_sources={heldout_source},
        ridge=1e-10,
        include_intercept=include_intercept,
        intercept_ridge_multiplier=1.0,
    )

    z_true = features[-1]
    tau = rng.normal(scale=0.2, size=(n_time, 2))
    geometry_context = np.concatenate(
        [np.ones(1, dtype=np.float64), np.asarray([scale], dtype=np.float64), z_true, scale * z_true]
    )
    conditioned = np.einsum("kdc,c->kd", tensor_true, geometry_context)
    compact = (_context_from_z(z_true) @ baseline_coef_true).reshape(n_time, basis_dim)
    compact = compact + _quadratic_design_from_path(tau, include_intercept=include_intercept) @ conditioned.T

    z_hat, compact_hat, meta = _solve_z_given_tau(
        observed_compact=compact,
        tau=tau,
        geometry_table=geometry_table,
        baseline_coef=baseline_coef,
        baseline_residual_var=1e-10,
        observation_coef=observation_coef,
        observation_residual_var=1e-10,
        include_intercept=include_intercept,
        continuous_args=argparse.Namespace(
            observation_var=1e-10,
            observation_var_floor=1e-12,
            forward_model_z_prior_precision=1e-10,
        ),
    )

    assert bool(meta["forward_z_solver_success"])
    np.testing.assert_allclose(z_hat, z_true, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(compact_hat, compact, atol=1e-5, rtol=1e-5)


def test_validated_residual_shrinkage_uses_pooled_r2_cv_contract() -> None:
    rng = np.random.default_rng(789)
    z_true = rng.normal(size=(12, 4))
    z0 = z_true.copy()
    x = rng.normal(size=(12, 6))
    sources = np.arange(1000, 1012, dtype=int)

    shrinkage = _validated_residual_shrinkage(
        z0_train=z0,
        z_true_train=z_true,
        x_train=x,
        source_rows=sources,
        ridge=1.0,
        seed=17,
    )

    assert shrinkage["lambda"] == 0.0
    assert shrinkage["selection_reason"] == "inner_source_disjoint_pooled_r2_cv_sse_sst"
    assert shrinkage["validation_n"] > 0


def test_inverse_transform_scores_projects_locked_scores_to_raw_space() -> None:
    transform = FeatureTransform(
        latent="synthetic",
        feature_space_mode="synthetic_pca",
        feature_dim=2,
        mean=np.asarray([10.0, -2.0, 0.5]),
        sd=np.asarray([2.0, 4.0, 1.0]),
        components=np.asarray([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]]),
        denom=np.asarray([2.0, 4.0]),
        weights=None,
        fit_scope="synthetic",
        preprocessing="zscore",
        whitened=True,
        weighted=False,
        n_fit_sources=5,
        raw_feature_dim=3,
        explained_variance_sum=1.0,
        explained_variance_first5=[0.8, 0.2],
    )

    scores = np.asarray([[1.0, -0.5]])
    raw = _inverse_transform_scores(transform, scores)

    expected_projected = np.asarray([[2.0, -1.0, -1.0]])
    expected_raw = expected_projected * transform.sd[None, :] + transform.mean[None, :]
    np.testing.assert_allclose(raw, expected_raw)


def test_pyramid_local_field_metadata_matches_feature_contract() -> None:
    meta = _local_field_dim_metadata(latent="pyramid_local_field", raw_feature_dim=3072)

    assert meta.shape[0] == 3072
    assert set(meta["channel"]) == {"real", "imag", "magnitude"}
    assert meta["band"].nunique() == 4
    assert meta["orientation"].nunique() == 4
    assert meta["block_index"].nunique() == 64
    first = meta.iloc[0]
    assert first["channel"] == "real"
    assert int(first["band"]) == 0
    assert int(first["orientation"]) == 0
