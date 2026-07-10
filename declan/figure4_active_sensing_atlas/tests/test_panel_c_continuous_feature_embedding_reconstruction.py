from __future__ import annotations

import numpy as np
import pandas as pd

from declan.figure4_active_sensing_atlas.scripts.build_panel_c_continuous_feature_embedding_reconstruction import (
    _filter_plot_tables,
    _apply_response_population,
    _project_response,
    _source_balanced_weights,
)
from declan.figure4_active_sensing_atlas.scripts.build_panel_c_endpoint_history_feature_readout import (
    STATIC_CONDITION,
    EndpointDataset,
    _concat_response_tau,
    _condition_for_family,
    _endpoint_aligned_trace,
    _endpoint_bank,
    _fit_history_generative_model,
    _fit_correlated_history_generative_model,
    _history_vector,
    _history_coordinate_matrices,
    _predict_history_joint,
    _predict_history_known,
    _predict_history_correlated_joint,
    _predict_history_correlated_known,
    _response_tau_interaction_features,
    _adjust_response_for_known_history,
    _fit_source_centered_history_map,
    _terminal_response_counts,
)
from declan.redundancy_resolved_v1_population import PopulationView


def test_response_population_reduces_last_unit_axis_for_observed_and_prior_arrays() -> None:
    membership = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    population = PopulationView(
        name="synthetic_rr3",
        membership=membership,
        input_channels=4,
        n_units=3,
        meta={"pooling_mode": "synthetic"},
    )

    observed = np.arange(2 * 4, dtype=np.float64).reshape(2, 4)
    reduced_observed = _apply_response_population(observed, population)

    expected_observed = np.stack(
        [
            observed[:, 0],
            0.5 * observed[:, 1] + 0.5 * observed[:, 2],
            observed[:, 3],
        ],
        axis=1,
    )
    np.testing.assert_allclose(reduced_observed, expected_observed)

    prior = np.arange(2 * 3 * 5 * 4, dtype=np.float64).reshape(2, 3, 5, 4)
    reduced_prior = _apply_response_population(prior, population)

    assert reduced_prior.shape == (2, 3, 5, 3)
    np.testing.assert_allclose(reduced_prior[..., 0], prior[..., 0])
    np.testing.assert_allclose(reduced_prior[..., 1], 0.5 * prior[..., 1] + 0.5 * prior[..., 2])
    np.testing.assert_allclose(reduced_prior[..., 2], prior[..., 3])


def test_project_response_applies_population_before_basis_projection() -> None:
    membership = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    population = PopulationView(
        name="synthetic_rr2",
        membership=membership,
        input_channels=4,
        n_units=2,
        meta={"pooling_mode": "medoid"},
    )
    response = np.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float64)
    basis = np.asarray([[1.0], [2.0]], dtype=np.float64)

    projected = _project_response(response, basis, population=population)

    expected = np.asarray([[1.0 * 1.0 + 3.0 * 2.0], [5.0 * 1.0 + 7.0 * 2.0]], dtype=np.float32)
    np.testing.assert_allclose(projected, expected.reshape(-1))


def test_source_balanced_weights_equalize_total_weight_per_source() -> None:
    sources = np.asarray([10, 10, 10, 20, 30, 30], dtype=int)

    weights = _source_balanced_weights(sources)

    assert np.isclose(weights.mean(), 1.0)
    totals = {
        int(source): float(weights[sources == source].sum())
        for source in np.unique(sources)
    }
    assert totals == {10: 2.0, 20: 2.0, 30: 2.0}


def test_filter_plot_tables_raises_when_requested_mode_absent() -> None:
    summary = pd.DataFrame(
        {
            "decoder_mode": ["linear_gaussian"],
            "latent": ["synthetic"],
            "feature_space_mode": ["fold_zscore_whitened_pca"],
            "observer_mode": ["known_eye"],
        }
    )
    contrasts = pd.DataFrame(
        {
            "decoder_mode": ["linear_gaussian"],
            "latent": ["synthetic"],
            "feature_space_mode": ["fold_zscore_whitened_pca"],
            "contrast": ["known_minus_hidden"],
        }
    )

    try:
        _filter_plot_tables(
            summary,
            contrasts,
            decoder_mode="linear_gaussian",
            latent="synthetic",
            feature_space_mode="missing_mode",
        )
    except ValueError as exc:
        assert "Requested plot feature-space mode" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("Expected missing plot feature-space mode to raise")


def test_endpoint_aligned_trace_sets_final_position_to_zero_and_preserves_steps() -> None:
    trace = np.asarray([[1.0, 2.0], [1.5, 1.0], [2.0, 4.0]], dtype=np.float32)

    aligned = _endpoint_aligned_trace(trace)

    np.testing.assert_allclose(aligned[-1], np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(np.diff(aligned, axis=0), np.diff(trace, axis=0))


def test_terminal_response_counts_uses_last_aligned_response_frame_only() -> None:
    response = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)

    counts = _terminal_response_counts(
        response,
        n_timepoints=4,
        terminal_frames=1,
        bin_seconds=0.5,
    )

    np.testing.assert_allclose(counts, response[-1:] * 0.5)


def test_endpoint_history_banks_match_known_joint_and_static_contracts() -> None:
    rows = pd.DataFrame({"true_source_row": [10, 20], "observation_scale": [1.0, 1.0]})
    conditions = [_condition_for_family("static"), _condition_for_family("empirical"), _condition_for_family("ou")]
    x_static = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x_empirical = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    x_ou = np.asarray([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32)
    tau_empirical = np.asarray([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32)
    tau_ou = tau_empirical + 1.0
    dataset = EndpointDataset(
        rows=rows,
        x_by_condition={
            STATIC_CONDITION: x_static,
            "empirical_endpoint_history": x_empirical,
            "ou_endpoint_history": x_ou,
        },
        tau_by_condition={
            STATIC_CONDITION: np.zeros_like(tau_empirical),
            "empirical_endpoint_history": tau_empirical,
            "ou_endpoint_history": tau_ou,
        },
        conditions=conditions,
        trace_metrics=pd.DataFrame(),
    )

    known = _endpoint_bank(
        dataset,
        bank_name="primary_response_plus_tau",
        primary_condition="empirical_endpoint_history",
        joint_conditions=["empirical_endpoint_history", "ou_endpoint_history"],
    )
    joint = _endpoint_bank(
        dataset,
        bank_name="joint_prior_response",
        primary_condition="empirical_endpoint_history",
        joint_conditions=["empirical_endpoint_history", "ou_endpoint_history"],
    )
    joint_with_tau = _endpoint_bank(
        dataset,
        bank_name="joint_prior_response_plus_tau",
        primary_condition="empirical_endpoint_history",
        joint_conditions=["empirical_endpoint_history", "ou_endpoint_history"],
    )
    static = _endpoint_bank(
        dataset,
        bank_name="static_response",
        primary_condition="empirical_endpoint_history",
        joint_conditions=["empirical_endpoint_history", "ou_endpoint_history"],
    )

    np.testing.assert_allclose(known.x, _concat_response_tau(x_empirical, tau_empirical))
    np.testing.assert_array_equal(known.source_rows, np.asarray([10, 20]))
    np.testing.assert_allclose(joint.x, np.concatenate([x_empirical, x_ou], axis=0))
    np.testing.assert_array_equal(joint.source_rows, np.asarray([10, 20, 10, 20]))
    np.testing.assert_allclose(
        joint_with_tau.x,
        np.concatenate(
            [
                _concat_response_tau(x_empirical, tau_empirical),
                _concat_response_tau(x_ou, tau_ou),
            ],
            axis=0,
        ),
    )
    np.testing.assert_array_equal(joint_with_tau.source_rows, np.asarray([10, 20, 10, 20]))
    np.testing.assert_allclose(static.x, x_static)
    np.testing.assert_allclose(_history_vector(np.asarray([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32)), [0.0, 0.0])


def test_fold_pca_history_coordinates_project_static_zero_through_train_basis() -> None:
    tau = np.asarray(
        [
            [1.0, 0.0, 0.5],
            [2.0, 0.0, 1.0],
            [3.0, 0.0, 1.5],
            [4.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    train_mask = np.asarray([True, True, True, False])

    coords, zero_coords, meta = _history_coordinate_matrices(
        tau=tau,
        train_mask=train_mask,
        mode="fold_pca",
        n_components=1,
    )

    assert coords.shape == (4, 1)
    assert zero_coords.shape == (4, 1)
    assert meta["history_coordinate_mode"] == "fold_pca"
    assert meta["history_dim_used"] == 1
    assert np.isfinite(meta["history_coordinate_variance_fraction"])
    assert not np.allclose(zero_coords, 0.0)


def test_response_tau_interaction_features_include_main_and_product_terms() -> None:
    response = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tau = np.asarray([[0.5, 1.0], [2.0, 3.0]], dtype=np.float32)

    features = _response_tau_interaction_features(response, tau)

    expected = np.asarray(
        [
            [1.0, 2.0, 0.5, 1.0, 0.5, 1.0, 1.0, 2.0],
            [3.0, 4.0, 2.0, 3.0, 6.0, 9.0, 8.0, 12.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(features, expected)


def test_source_centered_history_map_recovers_repeated_history_component() -> None:
    sources = np.asarray([1, 1, 1, 2, 2, 2], dtype=int)
    tau = np.asarray([[-1.0], [0.0], [1.0], [-1.0], [0.0], [1.0]], dtype=np.float64)
    source_signal = np.asarray([[10.0, -2.0], [10.0, -2.0], [10.0, -2.0], [4.0, 3.0], [4.0, 3.0], [4.0, 3.0]])
    history_map_true = np.asarray([[2.0, -0.5]], dtype=np.float64)
    response = source_signal + tau @ history_map_true

    history_map = _fit_source_centered_history_map(
        x_train=response,
        tau_train=tau,
        source_rows=sources,
        ridge=1e-8,
        sample_weight=np.ones(sources.shape[0], dtype=np.float64),
    )
    adjusted = _adjust_response_for_known_history(
        x=response,
        tau=tau,
        tau_reference=np.zeros_like(tau),
        history_map=history_map,
        gamma=1.0,
    )

    np.testing.assert_allclose(history_map, history_map_true, atol=1e-6)
    np.testing.assert_allclose(adjusted, source_signal, atol=1e-6)


def test_history_generative_known_tau_beats_forced_zero_tau_on_synthetic_data() -> None:
    z = np.asarray([[-1.0], [-0.4], [0.2], [0.8], [1.4], [2.0]], dtype=np.float64)
    tau = np.asarray([[0.6], [-0.3], [0.4], [-0.5], [0.2], [-0.4]], dtype=np.float64)
    feature_map = np.asarray([[1.2, -0.7]], dtype=np.float64)
    tau_map = np.asarray([[0.9, 0.5]], dtype=np.float64)
    x = z @ feature_map + tau @ tau_map

    model = _fit_history_generative_model(
        z_train=z,
        x_train=x,
        tau_train=tau,
        ridge=1e-8,
        noise_floor=1e-8,
        sample_weight=np.ones(z.shape[0], dtype=np.float64),
    )
    known = _predict_history_known(model, x, tau)
    zero = _predict_history_known(model, x, np.zeros_like(tau))
    joint = _predict_history_joint(model, x)

    assert known.shape == z.shape
    assert joint.shape == z.shape
    assert np.mean((known - z) ** 2) < np.mean((zero - z) ** 2)
    assert np.isfinite(joint).all()


def test_correlated_history_generative_known_tau_uses_feature_history_prior() -> None:
    tau = np.asarray([[-1.5], [-1.0], [-0.5], [0.0], [0.5], [1.0], [1.5], [2.0]], dtype=np.float64)
    z = 0.8 * tau + np.asarray([[-0.1], [0.05], [-0.05], [0.1], [-0.1], [0.05], [-0.05], [0.1]], dtype=np.float64)
    response_map = np.asarray([[0.4, -0.2]], dtype=np.float64)
    tau_map = np.asarray([[1.1, 0.7]], dtype=np.float64)
    x = z @ response_map + tau @ tau_map

    model = _fit_correlated_history_generative_model(
        z_train=z,
        x_train=x,
        tau_train=tau,
        ridge=1e-8,
        noise_floor=1e-8,
        sample_weight=np.ones(z.shape[0], dtype=np.float64),
    )
    known = _predict_history_correlated_known(model, x, tau)
    joint = _predict_history_correlated_joint(model, x)
    zero = _predict_history_correlated_known(model, x, np.zeros_like(tau))

    assert known.shape == z.shape
    assert joint.shape == z.shape
    assert np.mean((known - z) ** 2) < np.mean((joint - z) ** 2)
    assert np.mean((known - z) ** 2) < np.mean((zero - z) ** 2)
