"""Focused tests for BackImage trajectory-table observer utilities."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from declan.backimage_trajectory_observer.candidate_sets import build_candidate_set
from declan.backimage_trajectory_observer.likelihood import poisson_expected_count_loglik
from declan.backimage_trajectory_observer.observer import score_image_identity_table, summarize_observer_rows
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _trajectory_spec


def test_poisson_expected_count_loglik_clips_predictions() -> None:
    y = np.asarray([[1.0, 0.0]], dtype=np.float64)
    pred = np.asarray([[[0.0, 0.0]], [[1.0, 1.0]]], dtype=np.float64)
    score = poisson_expected_count_loglik(y, pred, eps=1e-6)
    assert score.shape == (2,)
    assert np.isfinite(score).all()
    assert score[1] > score[0]


def test_poisson_expected_count_loglik_rejects_negative_inputs() -> None:
    y = np.asarray([[1.0]], dtype=np.float64)
    try:
        poisson_expected_count_loglik(y, np.asarray([[[-1.0]]], dtype=np.float64))
    except ValueError as exc:
        assert "negative" in str(exc)
    else:
        raise AssertionError("Expected negative predictions to fail")
    try:
        poisson_expected_count_loglik(np.asarray([[-1.0]], dtype=np.float64), np.asarray([[[1.0]]], dtype=np.float64))
    except ValueError as exc:
        assert "negative" in str(exc)
    else:
        raise AssertionError("Expected negative observations to fail")


def test_image_identity_observer_keeps_known_eye_separate_from_loo_prior() -> None:
    y = np.asarray([[8.0, 1.0], [7.0, 1.0]], dtype=np.float64)
    known = np.asarray(
        [
            [[8.0, 1.0], [7.0, 1.0]],
            [[1.0, 8.0], [1.0, 7.0]],
        ],
        dtype=np.float64,
    )
    zero = np.asarray(
        [
            [[3.0, 3.0], [3.0, 3.0]],
            [[3.0, 3.0], [3.0, 3.0]],
        ],
        dtype=np.float64,
    )
    # Prior catalog intentionally lacks the exact known-eye true trajectory.
    prior = np.asarray(
        [
            [
                [[5.0, 2.0], [5.0, 2.0]],
                [[4.0, 2.0], [4.0, 2.0]],
            ],
            [
                [[2.0, 5.0], [2.0, 5.0]],
                [[2.0, 4.0], [2.0, 4.0]],
            ],
        ],
        dtype=np.float64,
    )
    result = score_image_identity_table(
        y_obs_counts=y,
        prior_lambda_counts=prior,
        known_lambda_counts=known,
        zero_lambda_counts=zero,
        true_candidate_index=0,
        candidate_ids=["true", "other"],
        true_trajectory_index=-1,
        nearest_trajectory_index=0,
        nearest_trajectory_distance=0.25,
    )
    assert result["known_correct"] is True
    assert result["joint_correct"] is True
    assert math.isnan(result["true_tau_rank"])
    assert result["nearest_tau_rank"] == 1.0
    assert 1.0 <= result["N_eff_true_image"] <= 2.0
    assert result["joint_vs_best_single_tau_gap"] >= 0.0
    assert result["joint_vs_best_dilution_gap"] >= 0.0


def test_candidate_set_contains_true_once_and_reports_distances() -> None:
    df = pd.DataFrame(
        {
            "source_row": [10, 11, 12, 13, 14],
            "session": ["s"] * 5,
            "image_patch_center_x_px": [0, 1, 2, 3, 4],
            "image_patch_center_y_px": [0, 1, 2, 3, 4],
            "image_patch_rms_contrast": [1.0, 1.1, 3.0, 4.0, 5.0],
            "image_gradient_energy": [1.0, 1.2, 3.0, 4.0, 5.0],
            "image_edge_density": [1.0, 1.1, 3.0, 4.0, 5.0],
            "image_orientation_coherence": [0.5, 0.52, 0.2, 0.3, 0.4],
        }
    )
    result = build_candidate_set(
        df,
        0,
        mode="hard_negative_structure",
        n_candidates=3,
        rng=np.random.default_rng(0),
    )
    assert result["true_candidate_index"] == 0
    assert result["candidate_ids"][0] == "source_row:10"
    assert result["candidate_ids"].count("source_row:10") == 1
    assert result["n_candidates"] == 3
    assert result["candidate_duplicate_flag"] is False
    assert result["n_random_fallback_distractors"] == 0
    assert math.isfinite(result["contrast_distance_to_nearest_distractor"])


def test_matched_candidate_set_fails_without_random_fallback() -> None:
    df = pd.DataFrame(
        {
            "source_row": [1, 2, 3, 4],
            "image_patch_rms_contrast": [1.0, 1.1, 10.0, 11.0],
            "image_gradient_energy": [1.0, 1.1, 10.0, 11.0],
            "image_edge_density": [1.0, 1.1, 10.0, 11.0],
            "image_orientation_coherence": [0.1, 0.1, 0.9, 0.9],
        }
    )
    try:
        build_candidate_set(
            df,
            0,
            mode="matched_structure_bins",
            n_candidates=4,
            rng=np.random.default_rng(0),
        )
    except ValueError as exc:
        assert "allow_random_fallback" in str(exc) or "lost candidates" in str(exc)
    else:
        raise AssertionError("Expected strict matched candidate set to fail without enough matches")


def test_same_session_region_empty_pool_fails_without_global_bypass() -> None:
    df = pd.DataFrame(
        {
            "source_row": [1, 2, 3],
            "session": ["a", "b", "b"],
            "image_patch_center_x_px": [0.0, 1.0, 2.0],
            "image_patch_center_y_px": [0.0, 1.0, 2.0],
            "image_patch_rms_contrast": [1.0, 2.0, 3.0],
        }
    )
    try:
        build_candidate_set(
            df,
            0,
            mode="same_session_region",
            n_candidates=2,
            rng=np.random.default_rng(0),
        )
    except ValueError as exc:
        assert "allow_random_fallback" in str(exc)
    else:
        raise AssertionError("Expected empty same-session pool to fail without global matching")


def test_trajectory_spec_has_role_independent_identity() -> None:
    trace = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    item = {"source_row": 7, "session": "session"}
    meta = {"requested_rms_deg": 1.0, "effective_rms_deg": 1.0}
    obs = _trajectory_spec(
        role="observation",
        family="empirical",
        scale=0.5,
        trace=trace,
        item=item,
        meta=meta,
        sample_index=0,
        is_true=True,
    )
    prior = _trajectory_spec(
        role="prior",
        family="empirical",
        scale=0.5,
        trace=trace,
        item=item,
        meta=meta,
        sample_index=0,
        is_true=True,
    )
    assert obs["trajectory_id"] != prior["trajectory_id"]
    assert obs["trajectory_identity_id"] == prior["trajectory_identity_id"]


def test_matched_static_response_requires_static_features() -> None:
    df = pd.DataFrame({"source_row": [1, 2], "image_patch_rms_contrast": [0.1, 0.2]})
    try:
        build_candidate_set(
            df,
            0,
            mode="matched_static_response",
            n_candidates=2,
            rng=np.random.default_rng(0),
        )
    except ValueError as exc:
        assert "static-response" in str(exc)
    else:
        raise AssertionError("Expected matched_static_response to require static-response feature columns")


def test_matched_static_response_uses_static_feature_neighbors() -> None:
    df = pd.DataFrame(
        {
            "source_row": [1, 2, 3, 4],
            "static_response_unit_0000": [0.0, 0.1, 10.0, 20.0],
            "static_response_unit_0001": [0.0, 0.1, 10.0, 20.0],
            "image_patch_rms_contrast": [0.0, 5.0, 0.1, 0.2],
        }
    )
    result = build_candidate_set(
        df,
        0,
        mode="matched_static_response",
        n_candidates=2,
        rng=np.random.default_rng(0),
    )
    assert result["candidate_ids"] == ["source_row:1", "source_row:2"]
    assert result["structure_feature_columns"] == "static_response_unit_0000,static_response_unit_0001"
    assert result["n_random_fallback_distractors"] == 0


def test_summary_aggregates_observer_rows() -> None:
    y = np.asarray([[4.0, 1.0]], dtype=np.float64)
    known = np.asarray([[[4.0, 1.0]], [[1.0, 4.0]]], dtype=np.float64)
    prior = known[:, None, :, :]
    rows = []
    for scale in (0.5, 1.0):
        result = score_image_identity_table(
            y_obs_counts=y,
            prior_lambda_counts=prior,
            known_lambda_counts=known,
            zero_lambda_counts=known[::-1],
            true_candidate_index=0,
            candidate_ids=["a", "b"],
        )
        rows.append(
            {
                "candidate_set_mode": "random_global",
                "observation_condition": "empirical",
                "observation_family": "empirical",
                "observation_scale": scale,
                "prior_condition": "empirical",
                "prior_family": "empirical",
                "prior_scale": scale,
                "trajectory_prior_mode": "include_self",
                **result,
            }
        )
    summary = summarize_observer_rows(rows)
    assert len(summary) == 2
    assert all(row["known_eye_accuracy"] == 1.0 for row in summary)
