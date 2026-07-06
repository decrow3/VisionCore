"""Tests for reusable artificial FEM trajectory priors."""

from __future__ import annotations

import numpy as np

from declan.vernier_active_sensing.synthetic_trajectory_priors import (
    SyntheticTrajectoryPriorConfig,
    generate_synthetic_trajectory_prior,
    recommended_empirical_confined_config,
    trace_features,
)


def _source_traces(n_traces: int = 12, n_frames: int = 18) -> np.ndarray:
    rng = np.random.default_rng(42)
    traces = np.zeros((n_traces, n_frames, 2), dtype=np.float64)
    for trace_idx in range(n_traces):
        beta = -0.25 - 0.02 * (trace_idx % 3)
        kappa = 0.65 + 0.08 * (trace_idx % 4)
        step = rng.normal(scale=0.02, size=2)
        for frame_idx in range(1, n_frames):
            eps = rng.normal(scale=0.015 + 0.001 * trace_idx, size=2)
            step = beta * step - kappa * traces[trace_idx, frame_idx - 1] + eps
            traces[trace_idx, frame_idx] = traces[trace_idx, frame_idx - 1] + step
    return traces


def test_brownian_prior_is_reproducible_and_centered() -> None:
    cfg = SyntheticTrajectoryPriorConfig(
        process_model="brownian",
        covariance_mode="full_empirical",
        center_mode="zero_mean",
    )
    first = generate_synthetic_trajectory_prior(_source_traces(), n_traces=8, n_frames=10, seed=7, config=cfg)
    second = generate_synthetic_trajectory_prior(_source_traces(), n_traces=8, n_frames=10, seed=7, config=cfg)

    assert first.traces_deg.shape == (8, 10, 2)
    assert np.allclose(first.traces_deg, second.traces_deg)
    assert np.allclose(np.mean(first.traces_deg, axis=1), 0.0, atol=1e-7)
    assert "step_cov_deg2_per_frame" in first.metadata


def test_recommended_empirical_confined_prior_exposes_reusable_metadata() -> None:
    cfg = recommended_empirical_confined_config(kappa_weight_power=0.5)
    result = generate_synthetic_trajectory_prior(_source_traces(), n_traces=16, n_frames=12, seed=11, config=cfg)

    assert result.traces_deg.shape == (16, 12, 2)
    assert result.base_traces_deg is not None
    assert result.chosen_confined_params is not None
    assert result.chosen_confined_params.shape == (16, 2)
    assert result.metadata["process_model"] == "scale_mixture_empirical_confined_step_ar1"
    assert result.metadata["confined_param_kappa_weight_power"] == 0.5
    assert result.metadata["scale_mixture_feature"] == "step_rms_arcmin"
    assert result.metadata["empirical_confined_param_count"] > 0
    assert np.all(np.isfinite(result.traces_deg))


def test_fixed_confined_prior_keeps_step_reversal_available() -> None:
    cfg = SyntheticTrajectoryPriorConfig(
        process_model="scale_mixture_confined_step_ar1",
        covariance_mode="full_empirical",
        center_mode="zero_mean",
        anti_step_beta=-0.30,
        position_spring_kappa=0.90,
        scale_mixture_feature="step_rms_arcmin",
        scale_mixture_distribution="empirical",
    )
    result = generate_synthetic_trajectory_prior(_source_traces(), n_traces=32, n_frames=14, seed=5, config=cfg)
    features = trace_features(result.traces_deg)

    assert result.metadata["step_transition_beta"] == -0.30
    assert result.metadata["position_spring_kappa"] == 0.90
    assert features["step_lag1_cosine"].median() < -0.25
    assert features["step_rms_arcmin"].median() > 0.0
