"""Focused tests for lag-geometry diagnostic guardrails."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from declan.vernier_active_sensing.run_lag_geometry_diagnostic import build_lag_kernel_for_trace, diagnostic_rows_for_trace
from declan.vernier_active_sensing.stimulus import RenderGeometry


def test_lag_geometry_rejects_invalid_lag_indices_before_model_work() -> None:
    args = SimpleNamespace(
        inference_mode="framewise",
        lag_translation_eps_arcmin=0.25,
        device="cpu",
        batch_size=1,
        spatial_collapse="max",
    )
    try:
        build_lag_kernel_for_trace(
            model=None,
            readout=None,
            args=args,
            out_dir=None,
            geometry=RenderGeometry(),
            condition="synthetic",
            trace_idx=0,
            fd_step_arcmin=0.25,
            reference_trace_deg=np.zeros((2, 2), dtype=np.float32),
            lag_indices=[0, 32],
        )
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("Expected invalid lag indices to raise before model execution")


def test_lag_geometry_reports_likelihood_and_decision_fidelity() -> None:
    plus = np.asarray([[2.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    minus = np.asarray([[0.0, 2.0], [0.0, 2.0]], dtype=np.float64)
    mu0 = np.zeros((2, 2, 2), dtype=np.float64)
    lag_payload = {
        "path": "synthetic.npz",
        "lag_indices": np.asarray([0], dtype=np.int32),
        "reference_trace_deg": np.zeros((2, 2), dtype=np.float64),
        "mu0_rates": mu0,
        "lag_kernel_rates_per_arcmin": np.asarray(
            [
                [[[[2.0, 0.0], [0.0, 0.0]]], [[[2.0, 0.0], [0.0, 0.0]]]],
                [[[[0.0, 0.0], [2.0, 0.0]]], [[[0.0, 0.0], [2.0, 0.0]]]],
            ],
            dtype=np.float64,
        ),
    }
    instant_cache = {
        "jacobian_rates_per_arcmin": np.asarray(
            [
                [[[1.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]],
            ],
            dtype=np.float64,
        )
    }
    rows = diagnostic_rows_for_trace(
        condition="synthetic",
        fd_step_arcmin=0.25,
        trace_idx=0,
        plus_rates=plus,
        minus_rates=minus,
        pose_trace_deg=np.asarray([[1.0 / 60.0, 0.0], [1.0 / 60.0, 0.0]], dtype=np.float64),
        lag_payload=lag_payload,
        instant_cache=instant_cache,
        bin_seconds=1.0,
        phi=1.0,
    )
    types = {row["diagnostic_type"] for row in rows}
    assert {"rate_fidelity", "likelihood_fidelity", "decision_fidelity"} <= types
    lag_decisions = [
        row
        for row in rows
        if row["diagnostic_type"] == "decision_fidelity"
        and row["geometry_mode"] == "lag_kernel"
        and row["theta_label"] == "plus"
    ]
    assert lag_decisions
    assert lag_decisions[0]["model_matches_exact_decision"] is True
