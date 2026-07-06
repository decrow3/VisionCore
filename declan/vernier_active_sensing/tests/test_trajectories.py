"""Tests for Vernier motion-condition trace builders."""

from __future__ import annotations

import numpy as np

from declan.vernier_active_sensing.trajectories import condition_trace


def _toy_trace() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ],
        dtype=np.float32,
    )


def test_anisotropic_real_trace_scales_across_x_and_along_y_about_trial_mean() -> None:
    trace = _toy_trace()
    mean = np.mean(trace, axis=0, keepdims=True)

    out, meta = condition_trace(trace, condition="real_aniso_across_0p5_along_2")

    expected = mean + (trace - mean) * np.asarray([[0.5, 2.0]], dtype=np.float32)
    assert np.allclose(out, expected)
    assert meta["condition_family"] == "anisotropic_scaled"
    assert meta["across_scale"] == 0.5
    assert meta["along_scale"] == 2.0
    assert meta["axis_convention"] == "vertical_vernier_across_x_along_y"


def test_anisotropic_phase_cloud_uses_same_scaled_positions_in_shuffled_order() -> None:
    trace = _toy_trace()
    mean = np.mean(trace, axis=0, keepdims=True)
    expected_positions = mean + (trace - mean) * np.asarray([[0.25, 1.0]], dtype=np.float32)
    rng = np.random.default_rng(0)

    out, meta = condition_trace(
        trace,
        condition="static_phase_cloud_matched_aniso_across_0p25_along_1",
        rng=rng,
    )

    assert {tuple(row) for row in out} == {tuple(row) for row in expected_positions}
    assert meta["condition_family"] == "cloud"
    assert meta["paired_phase_set"] is True
    assert meta["scale_matched_to"] == "real_aniso"


def test_anisotropic_order_shuffle_alias_uses_same_scaled_positions() -> None:
    trace = _toy_trace()
    mean = np.mean(trace, axis=0, keepdims=True)
    expected_positions = mean + (trace - mean) * np.asarray([[1.0, 0.0]], dtype=np.float32)
    rng = np.random.default_rng(1)

    out, meta = condition_trace(trace, condition="order_shuffled_aniso_across_1_along_0", rng=rng)

    assert {tuple(row) for row in out} == {tuple(row) for row in expected_positions}
    assert meta["condition_family"] == "trajectory_control"
    assert meta["scale_matched_to"] == "real_aniso"
