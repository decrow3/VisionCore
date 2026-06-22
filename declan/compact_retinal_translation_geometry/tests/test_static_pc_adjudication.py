"""Focused tests for compact-vs-static-PC adjudication helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from declan.compact_retinal_translation_geometry.run_static_pc_adjudication import (
    TangentObject,
    _assign_group_folds,
    _capture_tangent_fraction,
    _cluster_bootstrap_mean_ci,
)


def _object(object_id: str, group_id: str) -> TangentObject:
    zero = np.zeros(3, dtype=float)
    return TangentObject(
        object_id=object_id,
        group_id=group_id,
        image_id=group_id,
        trial_index="0",
        time_index="0",
        r0=zero,
        bx=zero,
        by=zero,
    )


def test_capture_tangent_fraction_uses_combined_x_y_energy() -> None:
    basis = np.eye(3, dtype=float)[:, :2]
    bx = np.asarray([1.0, 0.0, 0.0])
    by = np.asarray([0.0, 2.0, 0.0])

    capture, capture_bx, capture_by, k_eff = _capture_tangent_fraction(bx, by, basis, k=2)
    assert k_eff == 2
    assert capture == 1.0
    assert capture_bx == 1.0
    assert capture_by == 1.0

    capture_one, _, _, k_eff_one = _capture_tangent_fraction(bx, by, basis, k=1)
    assert k_eff_one == 1
    assert np.isclose(capture_one, 0.2)


def test_group_folds_are_group_disjoint_and_balanced() -> None:
    objects = (
        [_object(f"a{i}", "a") for i in range(4)]
        + [_object(f"b{i}", "b") for i in range(3)]
        + [_object(f"c{i}", "c") for i in range(2)]
        + [_object("d0", "d")]
    )
    group_to_fold, fold_rows = _assign_group_folds(objects, n_folds=2, seed=3)

    assert set(group_to_fold) == {"a", "b", "c", "d"}
    assert all(row["n_groups"] > 0 for row in fold_rows)
    for group in group_to_fold:
        object_folds = {group_to_fold[obj.group_id] for obj in objects if obj.group_id == group}
        assert len(object_folds) == 1
    loads = sorted(row["n_objects"] for row in fold_rows)
    assert loads == [5, 5]


def test_cluster_bootstrap_mean_ci_resamples_groups() -> None:
    frame = pd.DataFrame(
        {
            "group_id": ["a", "a", "b", "b", "c", "c"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    mean, lo, hi, boot = _cluster_bootstrap_mean_ci(
        frame,
        value_col="value",
        group_col="group_id",
        n_bootstrap=200,
        seed=0,
    )
    assert np.isclose(mean, 3.5)
    assert boot.shape == (200,)
    assert lo <= mean <= hi
