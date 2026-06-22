"""Tests for feature-space compact-subspace posterior controls."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from declan.backimage_trajectory_observer.analyze_feature_posterior import (
    analyze as analyze_feature_posterior,
)
from declan.backimage_trajectory_observer.analyze_feature_posterior import (
    build_parser as build_feature_posterior_parser,
)
from declan.backimage_trajectory_observer.analyze_feature_posterior_compact_mechanism import (
    _variant_tables,
)
from declan.backimage_trajectory_observer.analyze_feature_posterior_compact_mechanism import (
    analyze as analyze_feature_compact,
)
from declan.backimage_trajectory_observer.analyze_feature_posterior_compact_mechanism import (
    build_parser as build_feature_compact_parser,
)


def _write_tiny_feature_fixture(root: Path) -> tuple[Path, Path, Path]:
    run_dir = root / "run"
    response_dir = run_dir / "response_tables"
    response_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "source_row": [1, 2, 3],
            "image_index": [0, 1, 2],
        }
    ).to_csv(run_dir / "selected_windows.csv", index=False)
    table_rel = "response_tables/tiny.npz"
    known = np.asarray(
        [
            [[8.0, 1.0]],
            [[1.0, 8.0]],
        ],
        dtype=np.float32,
    )
    zero = np.asarray(
        [
            [[3.2, 2.2]],
            [[2.2, 3.2]],
        ],
        dtype=np.float32,
    )
    prior = known[:, None, :, :]
    np.savez_compressed(
        run_dir / table_rel,
        prior_lambda_counts=prior,
        known_lambda_counts=known,
        zero_lambda_counts=zero,
        y_obs_counts=known[0],
        candidate_ids=np.asarray(["source_row:1", "source_row:2"]),
        true_candidate_index=np.asarray([0], dtype=np.int64),
        true_trajectory_index=np.asarray([0], dtype=np.int64),
        nearest_trajectory_index=np.asarray([0], dtype=np.int64),
        nearest_trajectory_distance=np.asarray([0.0], dtype=np.float32),
    )
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
                "response_cache_path": table_rel,
            }
        ]
    ).to_csv(run_dir / "response_cache_manifest.csv", index=False)
    feature_npz = root / "features.npz"
    np.savez_compressed(
        feature_npz,
        pyramid_local_field=np.asarray(
            [
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        source_row=np.asarray([1, 2, 3], dtype=np.int64),
        image_index=np.asarray([0, 1, 2], dtype=np.int64),
    )
    basis_path = root / "basis.npz"
    np.savez_compressed(
        basis_path,
        basis=np.asarray([[1.0], [0.0]], dtype=np.float64),
        image_disjoint=np.asarray([True]),
        basis_mode=np.asarray(["image_disjoint"]),
    )
    return run_dir, feature_npz, basis_path


def test_compact_variant_addback_reconstructs_full_response() -> None:
    rng = np.random.default_rng(3)
    zero = rng.uniform(0.2, 1.0, size=(3, 4, 5))
    prior = zero[:, None, :, :] + rng.normal(scale=0.05, size=(3, 2, 4, 5))
    known = zero + rng.normal(scale=0.05, size=(3, 4, 5))
    u, _ = np.linalg.qr(rng.normal(size=(5, 2)))

    add_prior, add_known, add_zero = _variant_tables(
        "compact_addback",
        prior_full=prior,
        known_full=known,
        zero=zero,
        u=u,
    )

    assert np.allclose(add_prior, prior)
    assert np.allclose(add_known, known)
    assert np.allclose(add_zero, zero)


def test_feature_compact_removed_uses_residual_not_projection() -> None:
    zero = np.ones((2, 1, 3), dtype=float)
    prior = zero[:, None, :, :] + np.asarray(
        [
            [[[2.0, 3.0, 0.0]]],
            [[[4.0, 5.0, 0.0]]],
        ],
        dtype=float,
    )
    known = prior[:, 0]
    u = np.asarray([[1.0], [0.0], [0.0]], dtype=float)

    removed_prior, removed_known, _ = _variant_tables(
        "compact_removed",
        prior_full=prior,
        known_full=known,
        zero=zero,
        u=u,
    )
    compact_prior, compact_known, _ = _variant_tables(
        "compact_only",
        prior_full=prior,
        known_full=known,
        zero=zero,
        u=u,
    )

    assert np.allclose(removed_prior - zero[:, None, :, :], prior - compact_prior)
    assert np.allclose(removed_known - zero, known - compact_known)
    assert not np.allclose(removed_prior, compact_prior)


def test_feature_compact_full_zero_and_known_match_existing_observer_modes(tmp_path: Path) -> None:
    run_dir, feature_npz, basis_path = _write_tiny_feature_fixture(tmp_path)
    old_out = tmp_path / "old_feature"
    old_args = build_feature_posterior_parser().parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(old_out),
            "--feature-npz",
            str(feature_npz),
            "--latent-names",
            "pyramid_local_field",
            "--pca-k-list",
            "1",
            "--likelihood-scales",
            "1.0",
            "--n-bootstrap",
            "5",
            "--n-permutations",
            "5",
        ]
    )
    analyze_feature_posterior(old_args)

    new_out = tmp_path / "feature_compact"
    new_args = build_feature_compact_parser().parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(new_out),
            "--feature-npz",
            str(feature_npz),
            "--compact-basis-path",
            str(basis_path),
            "--basis-mode",
            "image_disjoint",
            "--latent-names",
            "pyramid_local_field",
            "--pca-k-list",
            "1",
            "--k-dims",
            "1",
            "--likelihood-scales",
            "1.0",
            "--reference-feature-summary",
            str(old_out / "feature_posterior_summary.csv"),
            "--n-bootstrap",
            "5",
            "--n-permutations",
            "5",
        ]
    )
    analyze_feature_compact(new_args)

    old = pd.read_csv(old_out / "feature_posterior_summary.csv").iloc[0]
    new = pd.read_csv(new_out / "feature_compact_mechanism_summary.csv")
    value_by_variant = {
        row["response_variant"]: float(row["mean_feature_neg_mse"])
        for _, row in new.iterrows()
    }
    assert np.isclose(value_by_variant["full_exact"], float(old["joint_mean_neg_mse"]))
    assert np.isclose(value_by_variant["zero_static"], float(old["zero_mean_neg_mse"]))
    assert np.isclose(value_by_variant["known_eye"], float(old["known_mean_neg_mse"]))

    qc = pd.read_csv(new_out / "feature_compact_mechanism_qc.csv")
    ref = qc[qc["qc_type"].eq("reference_feature_posterior_match")]
    assert not ref.empty
    assert pd.to_numeric(ref["abs_delta"], errors="coerce").max() < 1e-12


def test_feature_compact_summary_has_required_contrasts(tmp_path: Path) -> None:
    run_dir, feature_npz, basis_path = _write_tiny_feature_fixture(tmp_path)
    out_dir = tmp_path / "feature_compact"
    args = build_feature_compact_parser().parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(out_dir),
            "--feature-npz",
            str(feature_npz),
            "--compact-basis-path",
            str(basis_path),
            "--basis-mode",
            "image_disjoint",
            "--latent-names",
            "pyramid_local_field",
            "--pca-k-list",
            "1",
            "--k-dims",
            "1",
            "--likelihood-scales",
            "1.0",
            "--n-bootstrap",
            "5",
            "--n-permutations",
            "5",
        ]
    )
    analyze_feature_compact(args)

    uncertainty = pd.read_csv(out_dir / "feature_compact_mechanism_uncertainty.csv")
    feature_cosine = uncertainty[uncertainty["metric"].eq("feature_cosine")]
    expected = {
        "full_exact_minus_zero_static",
        "compact_only_minus_zero_static",
        "compact_removed_minus_zero_static",
        "compact_only_minus_compact_removed",
        "full_exact_minus_compact_removed",
        "known_eye_minus_full_exact",
    }
    assert expected.issubset(set(feature_cosine["contrast"]))
