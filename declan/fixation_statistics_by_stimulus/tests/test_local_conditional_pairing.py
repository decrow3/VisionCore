from __future__ import annotations

import numpy as np
import pandas as pd

from declan.fixation_statistics_by_stimulus.model_backimage_local_conditional_pairing import (
    _edge_drift_alignment,
    _fit_session_centered,
    _session_demean,
)


def test_edge_drift_alignment_uses_axis_periodicity() -> None:
    edge = np.asarray([0.0, 0.0, 0.0, 180.0])
    drift = np.asarray([0.0, 45.0, 90.0, 0.0])

    alignment = _edge_drift_alignment(edge, drift)

    assert np.allclose(alignment, [1.0, 0.0, -1.0, 1.0], atol=1e-6)


def test_session_demean_zeroes_each_session_mean() -> None:
    values = np.asarray([1.0, 3.0, 10.0, 14.0])
    sessions = np.asarray(["a", "a", "b", "b"])

    centered = _session_demean(values, sessions)

    assert np.allclose(centered, [-1.0, 1.0, -2.0, 2.0])
    for session in np.unique(sessions):
        assert np.isclose(np.mean(centered[sessions == session]), 0.0)


def test_fit_session_centered_recovers_within_session_slope() -> None:
    df = pd.DataFrame(
        {
            "session": ["a", "a", "a", "b", "b", "b"],
            "x": [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
            "y": [3.0, 5.0, 7.0, -12.0, -10.0, -8.0],
        }
    )

    fit = _fit_session_centered(df, ["x"], "y")

    assert np.isclose(fit["beta"][0], 2.0)
    assert fit["r2_within_session"] > 0.99
