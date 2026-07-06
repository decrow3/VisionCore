from __future__ import annotations

import numpy as np
import pandas as pd

from declan.figure4_active_sensing_atlas.scripts.audit_panel_c_continuous_joint_feature_calibration import (
    _best_rows,
)
from declan.figure4_active_sensing_atlas.scripts.audit_panel_c_joint_decoder_inherited_contracts import (
    _audit_ci_file,
)
from declan.figure4_active_sensing_atlas.scripts.build_panel_c_continuous_joint_feature_recovery import (
    PRIMARY_LATENT,
    _compute_temperature_cv,
    _summarize_temperature_cv,
    _vectorized_mode_rows,
)


def test_vectorized_feature_recovery_rows_parse_source_identity() -> None:
    rows = pd.DataFrame(
        {
            "run_slug": ["demo"] * 4,
            "run_label": ["demo"] * 4,
            "table_index": [0, 0, 1, 1],
            "observer_mode": ["continuous_joint"] * 4,
            "n_candidates": [2] * 4,
            "candidate_index": [0, 1, 0, 1],
            "candidate_id": ["source_row:10", "source_row:20", "source_row:30", "source_row:40"],
            "is_true_candidate": [True, False, False, True],
            "candidate_score": [2.0, 1.0, 0.5, 3.0],
            "posterior_temperature": [1.0] * 4,
        }
    )
    feature_table = {
        10: np.asarray([1.0, 0.0]),
        20: np.asarray([0.0, 1.0]),
        30: np.asarray([0.5, 0.5]),
        40: np.asarray([1.0, 1.0]),
    }

    scored = _vectorized_mode_rows(rows=rows, latent=PRIMARY_LATENT, feature_table=feature_table)

    assert scored["true_source_row"].tolist() == [10, 40]
    assert scored["pred_source_row"].tolist() == [10, 40]
    assert scored["true_candidate_id"].tolist() == ["source_row:10", "source_row:40"]


def test_temperature_cv_prefers_source_row_split_for_best_rows() -> None:
    sweep_rows = []
    for table_index, source_row in enumerate([10, 20, 30, 40]):
        for temp, feature_cosine in [(1.0, 0.20 + 0.01 * table_index), (0.5, 0.80 + 0.01 * table_index)]:
            sweep_rows.append(
                {
                    "run_slug": "demo",
                    "run_label": "demo",
                    "latent": PRIMARY_LATENT,
                    "observer_mode": "continuous_joint",
                    "table_index": table_index,
                    "trial_id": table_index,
                    "true_source_row": source_row,
                    "prior_scale": 1.0,
                    "prior_family": "all",
                    "posterior_temperature": temp,
                    "feature_cosine": feature_cosine,
                    "image_correct": True,
                    "candidate_posterior_true_mass": 0.5 + 0.05 * table_index,
                    "candidate_posterior_N_eff_fraction": 0.4,
                }
            )
    cv = _compute_temperature_cv(pd.DataFrame(sweep_rows))
    summary = _summarize_temperature_cv(cv)
    best = _best_rows(summary, calibration_mode="scale_specific")

    assert {"table_index", "trial_id", "source_row"}.issubset(set(cv["split_key"]))
    assert best["split_key"].tolist() == ["source_row"]
    assert best["selected_temperature_by_split"].iloc[0] == "1.0:0.5;1.0:0.5"


def test_ci_integrity_audit_accepts_ordered_intervals(tmp_path) -> None:
    path = tmp_path / "contrasts.csv"
    pd.DataFrame(
        {
            "mean_feature_cosine_delta": [0.1, -0.2],
            "ci_low": [0.0, -0.3],
            "ci_high": [0.2, -0.1],
        }
    ).to_csv(path, index=False)
    rows = []

    _audit_ci_file(rows, path)

    assert [row.status for row in rows] == ["PASS"]
    assert "All 2 finite point estimates lie inside their CIs" in rows[0].detail
