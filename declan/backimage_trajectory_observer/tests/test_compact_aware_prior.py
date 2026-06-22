"""Tests for compact-aware trajectory prior analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from declan.backimage_trajectory_observer.analyze_compact_aware_prior import (
    _beta_to_match_neff,
    _compact_leakage,
    _mean_neff_fraction_for_beta,
    analyze,
    build_parser,
)
from declan.backimage_trajectory_observer.diagnose_compact_aware_prior_weights import _trajectory_catalog_lookup


def _write_tiny_prior_fixture(root: Path) -> tuple[Path, Path, Path]:
    run_dir = root / "run"
    response_dir = run_dir / "response_tables"
    response_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "source_row": [1, 2],
            "image_index": [0, 1],
        }
    ).to_csv(run_dir / "selected_windows.csv", index=False)

    zero = np.asarray(
        [
            [[1.0, 1.0]],
            [[1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    prior = zero[:, None, :, :] + np.asarray(
        [
            [
                [[4.0, 0.0]],
                [[0.0, 4.0]],
            ],
            [
                [[0.0, 4.0]],
                [[4.0, 0.0]],
            ],
        ],
        dtype=np.float32,
    )
    known = prior[:, 0]
    manifest_rows = []
    for trial_id in (0, 1):
        table_rel = f"response_tables/tiny_{trial_id}.npz"
        suffix = "ab" if trial_id == 0 else "cd"
        np.savez_compressed(
            run_dir / table_rel,
            prior_lambda_counts=prior,
            known_lambda_counts=known,
            zero_lambda_counts=zero,
            y_obs_counts=prior[0, 0],
            candidate_ids=np.asarray(["source_row:1", "source_row:2"]),
            prior_trajectory_ids=np.asarray(
                [
                    [f"prior:test:trace0:{suffix[0]}", f"prior:test:trace1:{suffix[0]}"],
                    [f"prior:test:trace0:{suffix[1]}", f"prior:test:trace1:{suffix[1]}"],
                ]
            ),
            true_candidate_index=np.asarray([0], dtype=np.int64),
            true_trajectory_index=np.asarray([0], dtype=np.int64),
            nearest_trajectory_index=np.asarray([0], dtype=np.int64),
            nearest_trajectory_distance=np.asarray([0.0], dtype=np.float32),
        )
        manifest_rows.append(
            {
                "trial_id": trial_id,
                "candidate_set_mode": "hard_negative_structure",
                "observation_family": "empirical",
                "prior_family": "axis_edge_parallel",
                "scale": 1.0,
                "axis_catalog_mode": "per_candidate",
                "axis_shared_source_catalog": True,
                "response_cache_path": table_rel,
            }
        )
    pd.DataFrame(
        manifest_rows
    ).to_csv(run_dir / "response_cache_manifest.csv", index=False)

    feature_npz = root / "features.npz"
    np.savez_compressed(
        feature_npz,
        pyramid_local_field=np.asarray(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        ),
        source_row=np.asarray([1, 2], dtype=np.int64),
        image_index=np.asarray([0, 1], dtype=np.int64),
    )
    basis_path = root / "basis.npz"
    np.savez_compressed(
        basis_path,
        basis=np.asarray([[1.0], [0.0]], dtype=np.float64),
        image_disjoint=np.asarray([True]),
        basis_mode=np.asarray(["image_disjoint"]),
    )
    return run_dir, feature_npz, basis_path


def test_compact_leakage_is_zero_for_compact_delta() -> None:
    zero = np.ones((1, 1, 2), dtype=np.float64)
    prior = zero[:, None, :, :] + np.asarray(
        [
            [
                [[2.0, 0.0]],
                [[0.0, 2.0]],
            ]
        ],
        dtype=np.float64,
    )
    basis = np.asarray([[1.0], [0.0]], dtype=np.float64)
    leakage = _compact_leakage(prior, zero, basis, eps=1e-12)
    assert leakage.shape == (1, 2)
    assert leakage[0, 0] < 1e-12
    assert leakage[0, 1] > 1.0 - 1e-9


def test_beta_matching_hits_requested_neff_fraction() -> None:
    raw = np.asarray([1.0, 0.0, -1.0], dtype=np.float64)
    target = 0.75
    beta = _beta_to_match_neff(raw, target, n_candidates=1, n_trajectories=3, beta_max=12.0)
    observed = _mean_neff_fraction_for_beta(raw, beta, n_candidates=1, n_trajectories=3)
    assert abs(observed - target) < 1e-6


def test_compact_aware_prior_analyzer_outputs_distinct_prior_shapes(tmp_path: Path) -> None:
    run_dir, feature_npz, basis_path = _write_tiny_prior_fixture(tmp_path)
    out_dir = tmp_path / "compact_prior"
    args = build_parser().parse_args(
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
            "--prior-families",
            "uniform_base,image_independent_compact_prior,candidate_conditioned_compact_weight,gain_axis_aware,static_pc_aware",
            "--prior-beta",
            "1.0",
            "--n-random",
            "1",
            "--progress-every",
            "1",
        ]
    )
    analyze(args)

    summary = pd.read_csv(out_dir / "compact_aware_prior_summary.csv")
    families = set(summary["trajectory_weight_family"].astype(str))
    assert "uniform_base" in families
    assert "image_independent_compact_prior" in families
    assert "candidate_conditioned_compact_weight" in families

    qc = pd.read_csv(out_dir / "compact_aware_prior_qc.csv")
    weight_qc = qc[qc["qc_type"].eq("trajectory_weight")]
    shape_by_family = {
        str(row["trajectory_weight_family"]): str(row["log_prior_shape"])
        for _, row in weight_qc.iterrows()
    }
    assert shape_by_family["image_independent_compact_prior"] == "(2,)"
    assert shape_by_family["candidate_conditioned_compact_weight"] == "(2,2)"
    assert set(weight_qc["uses_y_obs_counts"].astype(str).str.lower()) == {"false"}
    source_by_family = {
        str(row["trajectory_weight_family"]): str(row["raw_weight_source"])
        for _, row in weight_qc.iterrows()
    }
    assert source_by_family["image_independent_compact_prior"] == "selected_manifest_stable_trajectory_leave_one_table_out"
    assert source_by_family["candidate_conditioned_compact_weight"] == "current_candidate_table"
    compact_qc = weight_qc[weight_qc["trajectory_weight_family"].eq("image_independent_compact_prior")]
    assert "shared_prior_self_fallback_slots" not in compact_qc.columns
    assert set(compact_qc["shared_prior_key_scope"].astype(str)) == {"hash_stripped_stable_trajectory_key"}
    assert pd.to_numeric(compact_qc["shared_prior_matched_stable_keys"], errors="coerce").min() > 0
    assert pd.to_numeric(compact_qc["shared_prior_total_stable_keys"], errors="coerce").min() > 0
    assert pd.to_numeric(compact_qc["shared_prior_stable_key_fallback_fraction"], errors="coerce").max() == 0.0
    assert pd.to_numeric(compact_qc["shared_prior_matched_slots"], errors="coerce").min() > 0
    assert pd.to_numeric(compact_qc["shared_prior_total_slots"], errors="coerce").min() > 0
    assert pd.to_numeric(compact_qc["shared_prior_fallback_fraction"], errors="coerce").max() == 0.0
    assert pd.to_numeric(compact_qc["shared_prior_nonmatching_fallback_slots"], errors="coerce").max() == 0

    contrasts = pd.read_csv(out_dir / "compact_aware_prior_contrasts.csv")
    assert not contrasts.empty
    assert "mean_feature_neg_mse_lhs_minus_uniform" in contrasts.columns


def test_trajectory_catalog_lookup_keeps_candidate_specific_stable_metadata(tmp_path: Path) -> None:
    catalog_path = tmp_path / "axis_trajectory_catalog.csv"
    pd.DataFrame(
        [
            {
                "trial_id": 7,
                "candidate_index": 0,
                "candidate_id": "source_row:1",
                "trajectory_id": "prior:axis:rel_1x:src10:s0:hash_a",
                "trajectory_identity_id": "axis:rel_1x:src10:s0:hash_a",
                "effective_rms_deg": 0.5,
                "path_length_deg": 1.0,
                "speed_mean_deg_s": 2.0,
                "output_axis_deg": 10.0,
            },
            {
                "trial_id": 7,
                "candidate_index": 1,
                "candidate_id": "source_row:2",
                "trajectory_id": "prior:axis:rel_1x:src10:s0:hash_b",
                "trajectory_identity_id": "axis:rel_1x:src10:s0:hash_b",
                "effective_rms_deg": 0.5,
                "path_length_deg": 3.0,
                "speed_mean_deg_s": 5.0,
                "output_axis_deg": 90.0,
            },
        ]
    ).to_csv(catalog_path, index=False)

    lookup, ambiguity = _trajectory_catalog_lookup(catalog_path)
    stable = "prior:axis:rel_1x:src10:s0"

    assert lookup[(7, 0, stable)]["path_length_deg"] == 1.0
    assert lookup[(7, 1, stable)]["path_length_deg"] == 3.0
    path_row = next(row for row in ambiguity if row["covariate"] == "path_length_deg")
    assert path_row["n_stable_key_groups"] == 1
    assert path_row["n_groups_vary_across_candidates"] == 1
    assert path_row["max_within_stable_key_range"] == 2.0


def test_compact_aware_prior_rejects_uniform_entropy_target(tmp_path: Path) -> None:
    run_dir, feature_npz, basis_path = _write_tiny_prior_fixture(tmp_path)
    out_dir = tmp_path / "compact_prior_bad_target"
    args = build_parser().parse_args(
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
            "--prior-families",
            "uniform_base,image_independent_compact_prior",
            "--entropy-match-target",
            "uniform_base",
        ]
    )
    try:
        analyze(args)
    except ValueError as exc:
        assert "entropy_match_target='uniform_base'" in str(exc)
    else:
        raise AssertionError("Expected uniform entropy target to fail")
