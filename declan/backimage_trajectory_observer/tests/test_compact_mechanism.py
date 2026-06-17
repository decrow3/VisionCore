"""Focused tests for BackImage compact-mechanism projection analysis."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from declan.backimage_trajectory_observer.analyze_compact_mechanism import (
    _project_delta,
    _random_basis,
    _rate_audit,
    _score_variant,
    _unit_shuffle_basis,
    _variant_tables,
    analyze,
    build_parser,
)


def test_projection_decomposition_reconstructs_delta() -> None:
    rng = np.random.default_rng(0)
    delta = rng.normal(size=(3, 4, 5, 6))
    u = _random_basis(6, 2, rng)
    projected = _project_delta(delta, u)
    residual = delta - projected
    assert np.allclose(projected + residual, delta)


def test_projection_applies_per_timebin_unitspace() -> None:
    rng = np.random.default_rng(1)
    delta = rng.normal(size=(2, 3, 4, 6))
    u = _random_basis(6, 3, rng)
    projected = _project_delta(delta, u)
    manual = delta[1, 2, 3] @ u @ u.T
    assert projected.shape == delta.shape
    assert np.allclose(projected[1, 2, 3], manual)


def test_compact_only_and_removed_shapes_match_full() -> None:
    rng = np.random.default_rng(2)
    zero = rng.uniform(0.1, 1.0, size=(3, 5, 6))
    prior = zero[:, None, :, :] + rng.normal(scale=0.05, size=(3, 4, 5, 6))
    known = zero + rng.normal(scale=0.05, size=(3, 5, 6))
    u = _random_basis(6, 2, rng)
    compact_prior, compact_known, compact_zero = _variant_tables(
        "compact_only",
        prior_full=prior,
        known_full=known,
        zero=zero,
        u=u,
    )
    removed_prior, removed_known, removed_zero = _variant_tables(
        "compact_removed",
        prior_full=prior,
        known_full=known,
        zero=zero,
        u=u,
    )
    assert compact_prior.shape == prior.shape
    assert compact_known.shape == known.shape
    assert compact_zero.shape == zero.shape
    assert removed_prior.shape == prior.shape
    assert removed_known.shape == known.shape
    assert removed_zero.shape == zero.shape
    assert np.allclose((compact_prior - zero[:, None]) + (removed_prior - zero[:, None]), prior - zero[:, None])


def test_full_exact_reproduces_original_scores_on_toy_table() -> None:
    y = np.asarray([[3.0, 1.0], [2.0, 1.0]])
    zero = np.asarray(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ],
        dtype=float,
    )
    known = np.asarray(
        [
            [[3.0, 1.0], [2.0, 1.0]],
            [[1.0, 3.0], [1.0, 2.0]],
            [[1.0, 1.0], [3.0, 3.0]],
        ],
        dtype=float,
    )
    prior = np.stack([known, zero], axis=1)
    result = _score_variant(
        y_obs_counts=y,
        prior_lambda_counts=prior,
        known_lambda_counts=known,
        zero_lambda_counts=zero,
        true_candidate_index=0,
        candidate_ids=["a", "b", "c"],
        true_trajectory_index=0,
        nearest_trajectory_index=0,
        nearest_trajectory_distance=0.0,
        eps=1e-8,
        likelihood_scale=1.0,
    )
    assert result["known_correct"] is True
    assert result["joint_correct"] is True


def test_negative_rate_clipping_is_reported() -> None:
    arr = np.asarray([1.0, -0.5, 0.0, 1e-10])
    audit = _rate_audit(arr, eps=1e-8)
    assert audit["negative_rate_fraction_before_clamp"] == 0.25
    assert audit["negative_rate_mass"] == 0.5
    assert audit["clipped_rate_fraction"] == 0.75


def test_random_subspace_seed_determinism() -> None:
    a = _random_basis(6, 2, np.random.default_rng(10))
    b = _random_basis(6, 2, np.random.default_rng(10))
    c = _random_basis(6, 2, np.random.default_rng(11))
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_unit_shuffle_changes_basis_but_preserves_shape() -> None:
    u = np.eye(6, 3)
    shuffled, perm = _unit_shuffle_basis(u, np.random.default_rng(4))
    assert shuffled.shape == u.shape
    assert perm.shape == (6,)
    assert not np.array_equal(perm, np.arange(6))


def _write_tiny_compact_fixture(root: Path, *, basis_image_disjoint: bool = False, observer_distance: float = 1.25) -> tuple[Path, Path]:
    base = root / "base"
    response_dir = base / "response_tables"
    response_dir.mkdir(parents=True)
    prior = np.asarray(
        [
            [
                [[3.0, 1.0, 0.5], [2.0, 1.0, 0.5]],
            ],
            [
                [[1.0, 3.0, 0.5], [1.0, 2.0, 0.5]],
            ],
        ],
        dtype=np.float32,
    )
    known = prior[:, 0]
    zero = np.ones_like(known)
    table_rel = "response_tables/tiny.npz"
    np.savez_compressed(
        base / table_rel,
        prior_lambda_counts=prior,
        known_lambda_counts=known,
        zero_lambda_counts=zero,
        y_obs_counts=known[0],
        candidate_ids=np.asarray(["a", "b"]),
        prior_trajectory_ids=np.asarray(["tau0"]),
        true_candidate_index=np.asarray([0], dtype=np.int32),
        true_trajectory_index=np.asarray([0], dtype=np.int32),
        nearest_trajectory_index=np.asarray([0], dtype=np.int32),
        zero_reference_mode=np.asarray(["patch_center_static_tau_zero"]),
    )
    pd.DataFrame(
        [
            {
                "trial_id": 0,
                "candidate_set_mode": "tiny",
                "observation_family": "empirical",
                "prior_family": "empirical",
                "scale": 1.0,
                "response_cache_path": table_rel,
                "n_candidates": 2,
                "n_prior_trajectories": 1,
                "n_timebins": 2,
                "n_units": 3,
                "true_trajectory_index": 0,
                "nearest_trajectory_index": 0,
                "prior_duplicate_trajectory_count": 0,
                "dry_run": False,
            }
        ]
    ).to_csv(base / "response_cache_manifest.csv", index=False)
    pd.DataFrame(
        [
            {
                "response_cache_path": table_rel,
                "nearest_tau_distance": observer_distance,
            }
        ]
    ).to_csv(base / "observer_trials.csv", index=False)
    basis_path = root / "basis.npz"
    basis = np.eye(3, dtype=np.float64)
    if basis_image_disjoint:
        np.savez(basis_path, basis=basis, image_disjoint=np.asarray([True]))
    else:
        np.savez(basis_path, basis=basis)
    return base, basis_path


def test_random_k_with_zero_randoms_does_not_consume_missing_basis() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base, basis_path = _write_tiny_compact_fixture(Path(tmp))
        out_dir = Path(tmp) / "out"
        args = build_parser().parse_args(
            [
                "--base-run-dir",
                str(base),
                "--compact-basis-path",
                str(basis_path),
                "--output-dir",
                str(out_dir),
                "--variants",
                "full_exact,random_k",
                "--n-random",
                "0",
                "--k-dims",
                "2",
                "--likelihood-scales",
                "1.0",
            ]
        )
        analyze(args)
        trials = pd.read_csv(out_dir / "compact_mechanism_trials.csv")
        assert set(trials["response_variant"]) == {"full_exact"}


def test_nearest_distance_is_recovered_from_observer_trials_for_old_caches() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base, basis_path = _write_tiny_compact_fixture(Path(tmp), observer_distance=2.5)
        out_dir = Path(tmp) / "out"
        args = build_parser().parse_args(
            [
                "--base-run-dir",
                str(base),
                "--compact-basis-path",
                str(basis_path),
                "--output-dir",
                str(out_dir),
                "--variants",
                "full_exact",
                "--k-dims",
                "2",
                "--likelihood-scales",
                "1.0",
            ]
        )
        analyze(args)
        trials = pd.read_csv(out_dir / "compact_mechanism_trials.csv")
        assert np.allclose(trials["nearest_tau_distance"], 2.5)


def test_image_disjoint_basis_mode_requires_declared_provenance() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base, basis_path = _write_tiny_compact_fixture(Path(tmp), basis_image_disjoint=False)
        args = build_parser().parse_args(
            [
                "--base-run-dir",
                str(base),
                "--compact-basis-path",
                str(basis_path),
                "--basis-mode",
                "image_disjoint",
                "--variants",
                "full_exact",
                "--k-dims",
                "2",
            ]
        )
        try:
            analyze(args)
        except ValueError as exc:
            assert "requires basis-file provenance" in str(exc)
        else:
            raise AssertionError("Expected image_disjoint provenance validation to fail")


def test_image_disjoint_basis_mode_accepts_declared_provenance() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base, basis_path = _write_tiny_compact_fixture(Path(tmp), basis_image_disjoint=True)
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
                "--variants",
                "full_exact",
                "--k-dims",
                "2",
            ]
        )
        analyze(args)
        assert (out_dir / "compact_mechanism_run_metadata.json").exists()


def test_compact_score_validation_rejects_bad_inputs() -> None:
    y = np.asarray([[1.0, -1.0]])
    prior = np.ones((1, 1, 1, 2), dtype=float)
    known = np.ones((1, 1, 2), dtype=float)
    zero = np.ones((1, 1, 2), dtype=float)
    try:
        _score_variant(
            y_obs_counts=y,
            prior_lambda_counts=prior,
            known_lambda_counts=known,
            zero_lambda_counts=zero,
            true_candidate_index=0,
            candidate_ids=["a"],
            true_trajectory_index=0,
            nearest_trajectory_index=0,
            nearest_trajectory_distance=0.0,
            eps=1e-8,
            likelihood_scale=1.0,
        )
    except ValueError as exc:
        assert "negative counts" in str(exc)
    else:
        raise AssertionError("Expected negative observations to be rejected")
