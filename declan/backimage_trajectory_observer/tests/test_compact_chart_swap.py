"""Focused tests for compact chart-swap scoring."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from declan.backimage_trajectory_observer.analyze_compact_chart_swap import (
    _score_chart_family,
    analyze,
    build_parser,
)


def _write_tiny_chart_fixture(root: Path, *, basis_image_disjoint: bool = False) -> tuple[Path, Path]:
    base = root / "base"
    response_dir = base / "response_tables"
    response_dir.mkdir(parents=True)

    zero = np.zeros((2, 2, 2), dtype=np.float32)
    prior = np.asarray(
        [
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[0.75, 0.0], [0.75, 0.0]],
            ],
            [
                [[0.0, 1.0], [0.0, 1.0]],
                [[0.0, 0.75], [0.0, 0.75]],
            ],
        ],
        dtype=np.float32,
    )
    known = prior[:, 0]
    table_rel = "response_tables/tiny.npz"
    np.savez_compressed(
        base / table_rel,
        prior_lambda_counts=prior,
        known_lambda_counts=known,
        zero_lambda_counts=zero,
        y_obs_counts=known[0],
        candidate_ids=np.asarray(["x_chart", "y_chart"]),
        prior_trajectory_ids=np.asarray(["tau0", "tau1"]),
        true_candidate_index=np.asarray([0], dtype=np.int32),
        true_trajectory_index=np.asarray([0], dtype=np.int32),
        nearest_trajectory_index=np.asarray([0], dtype=np.int32),
        nearest_trajectory_distance=np.asarray([0.0], dtype=np.float32),
    )
    pd.DataFrame(
        [
            {
                "trial_id": 0,
                "candidate_set_mode": "tiny",
                "observation_family": "synthetic",
                "prior_family": "synthetic",
                "scale": 1.0,
                "axis_catalog_mode": "toy",
                "axis_shared_source_catalog": True,
                "response_cache_path": table_rel,
                "n_candidates": 2,
                "n_prior_trajectories": 2,
                "n_timebins": 2,
                "n_units": 2,
                "true_trajectory_index": 0,
                "nearest_trajectory_index": 0,
                "nearest_trajectory_distance": 0.0,
                "dry_run": False,
            }
        ]
    ).to_csv(base / "response_cache_manifest.csv", index=False)
    basis_path = root / "basis.npz"
    if basis_image_disjoint:
        np.savez(basis_path, basis=np.eye(2, dtype=np.float64), image_disjoint=np.asarray([True]))
    else:
        np.savez(basis_path, basis=np.eye(2, dtype=np.float64))
    return base, basis_path


def test_correct_chart_beats_rolled_wrong_chart_on_toy_table() -> None:
    zero = np.zeros((2, 2, 2), dtype=float)
    prior = np.asarray(
        [
            [
                [[1.0, 0.0], [1.0, 0.0]],
            ],
            [
                [[0.0, 1.0], [0.0, 1.0]],
            ],
        ],
        dtype=float,
    )
    y_obs = prior[0, 0]
    basis = np.eye(2)

    correct = _score_chart_family(
        y_obs_counts=y_obs,
        prior_lambda_counts=prior,
        zero_lambda_counts=zero,
        basis=basis,
        chart_family="correct_chart",
        true_candidate_index=0,
        candidate_ids=["x_chart", "y_chart"],
    )
    wrong = _score_chart_family(
        y_obs_counts=y_obs,
        prior_lambda_counts=prior,
        zero_lambda_counts=zero,
        basis=basis,
        chart_family="wrong_chart_roll",
        true_candidate_index=0,
        candidate_ids=["x_chart", "y_chart"],
    )

    assert correct["chart_correct"] is True
    assert wrong["chart_correct"] is False
    assert correct["chart_true_score"] > wrong["chart_true_score"]


def test_chart_swap_analyze_writes_summary_and_contrasts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base, basis_path = _write_tiny_chart_fixture(Path(tmp), basis_image_disjoint=True)
        manifest_path = base / "response_cache_manifest.csv"
        manifest = pd.read_csv(manifest_path)
        manifest["nearest_trajectory_index"] = np.nan
        manifest.to_csv(manifest_path, index=False)
        out_dir = Path(tmp) / "out"
        args = build_parser().parse_args(
            [
                "--base-run-dir",
                str(base),
                "--compact-basis-path",
                str(basis_path),
                "--output-dir",
                str(out_dir),
                "--basis-mode",
                "image_disjoint",
                "--k-dims",
                "2",
                "--basis-types",
                "compact",
                "--chart-families",
                "correct_chart,wrong_chart_roll,global_chart,zero_chart",
                "--n-random",
                "0",
            ]
        )
        analyze(args)
        trials = pd.read_csv(out_dir / "compact_chart_swap_trials.csv")
        summary = pd.read_csv(out_dir / "compact_chart_swap_summary.csv")
        contrasts = pd.read_csv(out_dir / "compact_chart_swap_contrasts.csv")

        assert set(trials["chart_family"]) == {
            "correct_chart",
            "wrong_chart_roll",
            "global_chart",
            "static_no_motion_chart",
        }
        assert not summary.empty
        assert not contrasts.empty
        assert set(trials["basis_fit_scope"]) == {"provided_compact_basis"}
        correct_vs_wrong = contrasts[
            contrasts["rhs_chart_family"].astype(str).eq("wrong_chart_roll")
        ].iloc[0]
        assert correct_vs_wrong["mean_chart_true_score_lhs_minus_rhs"] > 0.0


def test_slot_aligned_controls_reject_unaligned_candidate_trajectory_slots() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base, basis_path = _write_tiny_chart_fixture(Path(tmp), basis_image_disjoint=True)
        table_path = base / "response_tables" / "tiny.npz"
        with np.load(table_path, allow_pickle=True) as data:
            payload = {key: data[key] for key in data.files}
        payload["prior_trajectory_ids"] = np.asarray(
            [
                ["prior:toy:src0:s0:hash_a", "prior:toy:src1:s1:hash_b"],
                ["prior:toy:src999:s0:hash_c", "prior:toy:src1:s1:hash_d"],
            ]
        )
        np.savez_compressed(table_path, **payload)
        args = build_parser().parse_args(
            [
                "--base-run-dir",
                str(base),
                "--compact-basis-path",
                str(basis_path),
                "--basis-mode",
                "image_disjoint",
                "--k-dims",
                "2",
                "--basis-types",
                "compact",
                "--chart-families",
                "correct_chart,global_chart",
                "--n-random",
                "0",
            ]
        )
        try:
            analyze(args)
        except ValueError as exc:
            assert "Slot-aligned chart families require candidate-aligned prior_trajectory_ids" in str(exc)
        else:
            raise AssertionError("Expected unaligned trajectory slots to be rejected")


def test_static_pc_default_is_candidate_set_fold_disjoint() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base, basis_path = _write_tiny_chart_fixture(Path(tmp), basis_image_disjoint=True)
        out_dir = Path(tmp) / "out"
        args = build_parser().parse_args(
            [
                "--base-run-dir",
                str(base),
                "--compact-basis-path",
                str(basis_path),
                "--output-dir",
                str(out_dir),
                "--basis-mode",
                "image_disjoint",
                "--k-dims",
                "1",
                "--basis-types",
                "static_pc_k",
                "--chart-families",
                "correct_chart,static_no_motion_chart",
                "--n-random",
                "0",
                "--static-pc-folds",
                "2",
            ]
        )
        analyze(args)
        trials = pd.read_csv(out_dir / "compact_chart_swap_trials.csv")
        assert set(trials["basis_type"]) == {"static_pc_k"}
        assert set(trials["basis_fit_scope"]) == {"candidate_set_fold_disjoint_1fold"}
        assert set(trials["static_pc_fold"]) == {0}
