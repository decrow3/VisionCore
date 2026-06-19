"""Synthetic cache tests for feature-posterior BackImage decoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from declan.backimage_trajectory_observer.analyze_feature_posterior import (
    _axis_contrasts,
    _wide_trial_metrics,
    analyze,
    build_parser,
)


def _write_response_table(path: Path, *, prior: np.ndarray) -> None:
    y = np.asarray([[8.0, 1.0]], dtype=np.float32)
    known = np.asarray([[[8.0, 1.0]], [[1.0, 8.0]]], dtype=np.float32)
    zero = np.asarray([[[3.0, 3.0]], [[3.0, 3.0]]], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        y_obs_counts=y,
        prior_lambda_counts=prior.astype(np.float32),
        known_lambda_counts=known,
        zero_lambda_counts=zero,
        candidate_ids=np.asarray(["source_row:1", "source_row:2"]),
        true_candidate_index=np.asarray([0], dtype=np.int64),
        true_trajectory_index=np.asarray([-1], dtype=np.int64),
        nearest_trajectory_index=np.asarray([0], dtype=np.int64),
        nearest_trajectory_distance=np.asarray([0.0], dtype=np.float32),
    )


def test_feature_posterior_analyzer_scores_synthetic_cache(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "feature_posthoc"
    response_dir = run_dir / "response_tables"
    run_dir.mkdir()
    pd.DataFrame({"source_row": [1, 2, 3], "image_index": [0, 1, 2]}).to_csv(
        run_dir / "selected_windows.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "trial_id": 0,
                "candidate_set_mode": "hard_negative_structure",
                "candidate_ids": "source_row:1;source_row:2",
                "candidate_indices": "0;1",
            }
        ]
    ).to_csv(run_dir / "candidate_sets.csv", index=False)
    parallel_prior = np.asarray(
        [
            [[[8.0, 1.0]], [[7.0, 1.0]]],
            [[[1.0, 8.0]], [[1.0, 7.0]]],
        ],
        dtype=np.float32,
    )
    orthogonal_prior = np.asarray(
        [
            [[[3.0, 3.0]], [[2.0, 3.0]]],
            [[[3.0, 3.0]], [[3.0, 2.0]]],
        ],
        dtype=np.float32,
    )
    _write_response_table(response_dir / "parallel.npz", prior=parallel_prior)
    _write_response_table(response_dir / "orthogonal.npz", prior=orthogonal_prior)
    pd.DataFrame(
        [
            {
                "trial_id": 0,
                "candidate_set_mode": "hard_negative_structure",
                "observation_family": "empirical",
                "prior_family": "axis_edge_parallel",
                "scale": 1.0,
                "axis_catalog_mode": "per_candidate",
                "axis_shared_source_catalog": True,
                "response_cache_path": "response_tables/parallel.npz",
            },
            {
                "trial_id": 0,
                "candidate_set_mode": "hard_negative_structure",
                "observation_family": "empirical",
                "prior_family": "axis_edge_orthogonal",
                "scale": 1.0,
                "axis_catalog_mode": "per_candidate",
                "axis_shared_source_catalog": True,
                "response_cache_path": "response_tables/orthogonal.npz",
            },
        ]
    ).to_csv(run_dir / "response_cache_manifest.csv", index=False)
    feature_npz = tmp_path / "features.npz"
    np.savez_compressed(
        feature_npz,
        gabor_local_field=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32),
        source_row=np.asarray([1, 2, 3], dtype=np.int64),
        image_index=np.asarray([0, 1, 2], dtype=np.int64),
    )
    args = build_parser().parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(out_dir),
            "--feature-npz",
            str(feature_npz),
            "--latent-names",
            "gabor_local_field",
            "--pca-k-list",
            "1",
            "--likelihood-scales",
            "1.0",
            "--n-bootstrap",
            "25",
            "--n-permutations",
            "25",
            "--uncertainty-seed",
            "3",
        ]
    )
    result_dir = analyze(args)
    assert result_dir == out_dir
    trials = pd.read_csv(out_dir / "feature_posterior_trials.csv")
    summary = pd.read_csv(out_dir / "feature_posterior_summary.csv")
    axis = pd.read_csv(out_dir / "feature_axis_contrasts.csv")
    uncertainty = pd.read_csv(out_dir / "feature_posterior_uncertainty.csv")
    qc = pd.read_csv(out_dir / "feature_posterior_qc.csv")
    assert set(trials["observer_mode"]) == {"known", "zero", "joint", "best_single_tau", "motion_delta"}
    assert len(trials) == 10
    assert not summary.empty
    assert "joint_minus_zero_feature_gain" in summary.columns
    assert "joint_minus_zero_feature_gain_ci_low" in summary.columns
    assert "joint_minus_zero_feature_gain_permutation_p_two_sided" in summary.columns
    assert not axis.empty
    assert "mean_joint_parallel_minus_orthogonal" in axis.columns
    assert "mean_joint_parallel_minus_orthogonal_ci_low" in axis.columns
    assert "mean_joint_parallel_minus_orthogonal_permutation_p_two_sided" in axis.columns
    assert not uncertainty.empty
    assert {
        "within_prior",
        "pairwise_prior_lhs_minus_rhs",
        "axis_parallel_minus_orthogonal",
    }.issubset(set(uncertainty["contrast_scope"]))
    assert "candidate_alignment" in set(qc["qc_type"])
    assert set(trials.loc[trials["observer_mode"].eq("motion_delta"), "score_interpretation"]) == {
        "candidate_log_likelihood_ratio_joint_minus_zero"
    }


def test_feature_posterior_contrast_key_keeps_observation_families_separate() -> None:
    rows = []
    for observation_family in ("empirical", "rotated"):
        for prior_family in ("axis_edge_parallel", "axis_edge_orthogonal"):
            for observer_mode, value in {
                "known": -1.0,
                "zero": -2.0,
                "joint": -3.0 if observation_family == "empirical" else -4.0,
                "best_single_tau": -1.5,
                "motion_delta": -2.5,
            }.items():
                rows.append(
                    {
                        "trial_id": 0,
                        "candidate_set_mode": "hard_negative_structure",
                        "observation_condition": observation_family,
                        "observation_family": observation_family,
                        "observation_scale": 1.0,
                        "prior_scale": 1.0,
                        "axis_catalog_mode": "per_candidate",
                        "axis_shared_source_catalog": True,
                        "trajectory_prior_mode": "leave_one_out",
                        "zero_reference_mode": "patch_center_static_tau_zero",
                        "bin_seconds": 1.0 / 120.0,
                        "likelihood_scale": 1.0,
                        "latent": "gabor_local_field",
                        "requested_k": 1,
                        "k_eff": 1,
                        "prior_family": prior_family,
                        "observer_mode": observer_mode,
                        "feature_neg_mse": value,
                    }
                )
    wide = _wide_trial_metrics(rows)
    assert len(wide) == 4
    assert set(wide["observation_family"]) == {"empirical", "rotated"}


def test_feature_posterior_validates_all_shared_feature_identities(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "feature_posthoc"
    response_dir = run_dir / "response_tables"
    run_dir.mkdir()
    pd.DataFrame({"source_row": [1, 2, 3], "image_index": [0, 1, 2]}).to_csv(
        run_dir / "selected_windows.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "trial_id": 0,
                "candidate_set_mode": "hard_negative_structure",
                "candidate_ids": "source_row:1;source_row:2",
                "candidate_indices": "0;1",
            }
        ]
    ).to_csv(run_dir / "candidate_sets.csv", index=False)
    prior = np.asarray(
        [
            [[[8.0, 1.0]], [[7.0, 1.0]]],
            [[[1.0, 8.0]], [[1.0, 7.0]]],
        ],
        dtype=np.float32,
    )
    _write_response_table(response_dir / "table.npz", prior=prior)
    pd.DataFrame(
        [
            {
                "trial_id": 0,
                "candidate_set_mode": "hard_negative_structure",
                "observation_family": "empirical",
                "prior_family": "empirical",
                "scale": 1.0,
                "response_cache_path": "response_tables/table.npz",
            }
        ]
    ).to_csv(run_dir / "response_cache_manifest.csv", index=False)
    feature_npz = tmp_path / "features_bad_index.npz"
    np.savez_compressed(
        feature_npz,
        gabor_local_field=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32),
        source_row=np.asarray([1, 2, 3], dtype=np.int64),
        image_index=np.asarray([2, 1, 0], dtype=np.int64),
    )
    args = build_parser().parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(out_dir),
            "--feature-npz",
            str(feature_npz),
            "--latent-names",
            "gabor_local_field",
            "--pca-k-list",
            "1",
            "--likelihood-scales",
            "1.0",
        ]
    )
    try:
        analyze(args)
    except ValueError as exc:
        assert "image_index" in str(exc)
    else:
        raise AssertionError("Expected mismatched image_index identity to fail")


def test_feature_posterior_dryrun_manifest_fails_clearly(tmp_path: Path) -> None:
    run_dir = tmp_path / "dryrun"
    run_dir.mkdir()
    pd.DataFrame({"source_row": [1], "image_index": [0]}).to_csv(run_dir / "selected_windows.csv", index=False)
    pd.DataFrame(
        [
            {
                "trial_id": 0,
                "candidate_set_mode": "hard_negative_structure",
                "observation_family": "empirical",
                "prior_family": "empirical",
                "scale": 1.0,
                "response_cache_path": np.nan,
            }
        ]
    ).to_csv(run_dir / "response_cache_manifest.csv", index=False)
    args = build_parser().parse_args(["--run-dir", str(run_dir), "--out-dir", str(tmp_path / "out")])
    try:
        analyze(args)
    except ValueError as exc:
        assert "No response cache tables available" in str(exc)
    else:
        raise AssertionError("Expected dry-run manifest without cache paths to fail clearly")


def test_feature_axis_contrasts_require_shared_source_catalog() -> None:
    rows = []
    for shared in (False, True):
        for prior_family in ("axis_edge_parallel", "axis_edge_orthogonal"):
            for observer_mode, value in {
                "known": -1.0,
                "zero": -2.0,
                "joint": -3.0,
                "best_single_tau": -1.5,
                "motion_delta": -2.5,
            }.items():
                rows.append(
                    {
                        "trial_id": int(shared),
                        "candidate_set_mode": "hard_negative_structure",
                        "observation_condition": "empirical",
                        "observation_family": "empirical",
                        "observation_scale": 1.0,
                        "prior_scale": 1.0,
                        "axis_catalog_mode": "per_candidate",
                        "axis_shared_source_catalog": shared,
                        "trajectory_prior_mode": "leave_one_out",
                        "zero_reference_mode": "patch_center_static_tau_zero",
                        "bin_seconds": 1.0 / 120.0,
                        "likelihood_scale": 1.0,
                        "latent": "gabor_local_field",
                        "requested_k": 1,
                        "k_eff": 1,
                        "prior_family": prior_family,
                        "observer_mode": observer_mode,
                        "feature_neg_mse": value,
                    }
                )
    axis = _axis_contrasts(rows)
    assert len(axis) == 1
    assert axis[0]["axis_shared_source_catalog"] is True
