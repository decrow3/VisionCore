"""Focused tests for RR100 compact-geometry comparison helpers."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from declan.compact_retinal_translation_geometry.run_rr100_compact_geometry_comparison import (
    _collect_paired_objects,
    _orthonormalize,
    _restrict_basis_to_rr100,
    _selected_channels_from_membership,
)


def test_collect_paired_objects_applies_population_membership(tmp_path) -> None:
    tangent_dir = tmp_path / "tangent_maps"
    tangent_dir.mkdir()
    payload = {
        "delta_arcmins": [0.25],
        "object_payload": {
            0.25: {
                "img/trial/time": {
                    "image_id": 7,
                    "trial_index": 3,
                    "time_index": 5,
                    "r0": np.asarray([1.0, 2.0, 3.0]),
                    "bx": np.asarray([4.0, 5.0, 6.0]),
                    "by": np.asarray([7.0, 8.0, 9.0]),
                }
            }
        },
    }
    with (tangent_dir / "twin_tangent_maps.pkl").open("wb") as handle:
        pickle.dump(payload, handle)

    membership = np.asarray([[0.0, 1.0, 0.0], [0.5, 0.0, 0.5]])
    delta, objects, n_units, skipped = _collect_paired_objects(
        tfts_root=tmp_path,
        requested_delta=0.25,
        group_by="image_id",
        membership=membership,
    )

    assert delta == 0.25
    assert n_units == 3
    assert skipped == []
    assert len(objects) == 1
    assert objects[0].group_id == "7"
    np.testing.assert_allclose(objects[0].rr100.r0, [2.0, 2.0])
    np.testing.assert_allclose(objects[0].rr100.bx, [5.0, 5.0])
    np.testing.assert_allclose(objects[0].rr100.by, [8.0, 8.0])


def test_orthonormalize_drops_dependent_columns() -> None:
    basis = _orthonormalize(np.asarray([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]]))

    assert basis.shape == (3, 1)
    np.testing.assert_allclose(basis.T @ basis, np.eye(1), atol=1e-12)


def test_selected_channels_from_one_hot_membership() -> None:
    membership = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    selected = _selected_channels_from_membership(membership)

    np.testing.assert_array_equal(selected, [1, 0])


def test_selected_channels_rejects_non_one_hot_rows() -> None:
    membership = np.asarray([[1.0, 0.5, -0.5], [0.0, 1.0, 0.0]])

    with pytest.raises(ValueError, match="one-hot"):
        _selected_channels_from_membership(membership)


def test_restrict_basis_to_rr100_uses_top_k_columns_only() -> None:
    membership = np.eye(3, dtype=float)
    full_basis = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )

    restricted = _restrict_basis_to_rr100(membership, full_basis, k=1)

    assert restricted.shape == (3, 1)
    np.testing.assert_allclose(np.abs(restricted[:, 0]), [1.0, 0.0, 0.0])
