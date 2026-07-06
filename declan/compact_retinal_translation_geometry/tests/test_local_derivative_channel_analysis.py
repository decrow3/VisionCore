"""Focused tests for local derivative channel analysis helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from declan.compact_retinal_translation_geometry.run_local_derivative_channel_analysis import (
    DerivativeObject,
    DirectionSpec,
    FeatureExtractor,
    _build_feature_samples,
    _decision_table,
    _decode_train_test,
    _orth_residual_basis,
    _raw_pixel_jacobian_descriptor,
    _shift_stack_subpixel,
    parse_directions,
)


def test_parse_direction_groups_are_unit_vectors() -> None:
    specs = parse_directions("cardinal,diagonal")
    by_name = {spec.name: spec for spec in specs}
    expected = {"+x", "-x", "+y", "-y", "+x+y", "+x-y", "-x+y", "-x-y"}

    assert set(by_name) == expected
    for spec in by_name.values():
        assert np.isclose(np.hypot(spec.ux, spec.uy), 1.0)


def test_raw_pixel_jacobian_matches_direct_shift_for_ramp() -> None:
    width = 41
    frame = np.tile(np.linspace(-1.0, 1.0, width), (41, 1))
    direction = DirectionSpec("+x", 1.0, 0.0)
    delta = 0.2
    expected_slope = -2.0 / float(width - 1)

    direct = (
        _shift_stack_subpixel(frame, dx_px=delta, dy_px=0.0)
        - _shift_stack_subpixel(frame, dx_px=-delta, dy_px=0.0)
    ) / (2.0 * delta)
    jac = _raw_pixel_jacobian_descriptor(frame, direction, grid_size=3)

    assert np.isclose(float(np.mean(direct[:, 4:-4])), expected_slope, atol=1e-6)
    assert np.allclose(jac, np.full(9, expected_slope), atol=0.02)


def test_feature_derivatives_are_antisymmetric_for_opposite_directions() -> None:
    yy, xx = np.mgrid[-1:1:31j, -1:1:31j]
    frame = np.sin(3.0 * xx) + 0.5 * np.cos(2.0 * yy)
    history = np.stack([frame, frame], axis=0)
    obj = DerivativeObject(
        object_id="obj0",
        group_id="img0",
        image_id="img0",
        trial_index="0",
        time_index="0",
        delta_arcmin=0.25,
        delta_model_px=0.25,
        r0=np.zeros(4),
        bx=np.ones(4),
        by=np.arange(4, dtype=float),
        history=history,
    )
    extractor = FeatureExtractor(
        target_families=["raw_pixel_grid"],
        gabor_wavelengths=[4.0],
        gabor_orientations_deg=[0.0],
        grid_size=5,
    )

    sample_df, dr, _, targets, _, _ = _build_feature_samples(
        objects_by_delta={0.25: [obj]},
        directions=parse_directions("+x,-x"),
        feature_frame_mode="current",
        extractor=extractor,
        target_families=["raw_pixel_grid"],
    )

    plus_idx = int(sample_df.loc[sample_df["direction"] == "+x", "sample_index"].iloc[0])
    minus_idx = int(sample_df.loc[sample_df["direction"] == "-x", "sample_index"].iloc[0])
    assert np.allclose(dr[plus_idx], obj.bx)
    assert np.allclose(dr[minus_idx], -obj.bx)
    assert np.allclose(dr[plus_idx] + dr[minus_idx], 0.0)
    assert np.linalg.norm(targets["raw_pixel_grid"][plus_idx] + targets["raw_pixel_grid"][minus_idx]) < 1e-8


def test_residual_basis_is_orthogonal_to_nuisance() -> None:
    primary = np.eye(5)[:, :3]
    nuisance = np.eye(5)[:, 1:3]

    residual = _orth_residual_basis(primary, nuisance)

    assert residual.shape == (5, 1)
    assert np.allclose(residual.T @ residual, np.eye(1))
    assert np.linalg.norm(nuisance.T @ residual) < 1e-10
    assert np.isclose(abs(float(residual[:, 0] @ np.eye(5)[:, 0])), 1.0)


def test_ridge_decode_recovers_linear_target_on_heldout_groups() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 4))
    w = rng.normal(size=(4, 3))
    y = x @ w + 0.01 * rng.normal(size=(80, 3))
    groups = np.asarray([f"g{i // 8}" for i in range(80)])
    train = np.asarray([not g.endswith(("8", "9")) for g in groups])
    test = ~train

    pred, true, lam, k_eff = _decode_train_test(x, y, groups, train, test, seed=0)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true, axis=0, keepdims=True)) ** 2)

    assert k_eff == 4
    assert np.isfinite(lam)
    assert 1.0 - ss_res / ss_tot > 0.95


def test_decision_table_requires_positive_full_response_signal() -> None:
    compact_minus_static = pd.DataFrame(
        [
            {
                "target_family": "phase_vector",
                "k": 10,
                "metric": "R2_mean",
                "mean_lhs_minus_rhs": -0.001,
                "ci_low": -0.003,
            }
        ]
    )
    bootstrap = pd.DataFrame(
        [
            {
                "target_family": "phase_vector",
                "basis_type": "compact",
                "metric": "R2_mean",
                "mean": -0.01,
                "ci_low": -0.02,
            },
            {
                "target_family": "phase_vector",
                "basis_type": "full_response",
                "metric": "R2_mean",
                "mean": -0.001,
                "ci_low": -0.004,
            },
        ]
    )
    linearity = pd.DataFrame(
        [
            {
                "target_family": "phase_vector",
                "basis_type": "true_target",
                "linearity_test": "antisymmetry",
                "median_residual": 0.0,
            },
            {
                "target_family": "phase_vector",
                "basis_type": "compact",
                "linearity_test": "antisymmetry",
                "median_residual": 0.0,
            },
        ]
    )

    decision = _decision_table(
        compact_minus_static=compact_minus_static,
        bootstrap=bootstrap,
        linearity=linearity,
        primary_k=10,
        primary_targets=["phase_vector"],
    )

    assert decision["outcome"] == "decoder_sanity_failed_no_reliable_heldout_derivative_readout"
    assert decision["full_response_R2_mean_mean"] == -0.001
    assert decision["full_response_reliably_positive"] is False
