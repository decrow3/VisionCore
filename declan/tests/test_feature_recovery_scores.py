from __future__ import annotations

import numpy as np

from declan.feature_recovery_scores import (
    R2_CV_METHOD,
    per_sample_sse_sst,
    pooled_multioutput_r2,
    pooled_multioutput_r2_from_sse_sst,
)


def test_pooled_multioutput_r2_uses_train_mean_baseline() -> None:
    y_true = np.asarray([[1.0, 2.0], [3.0, 6.0]], dtype=np.float64)
    y_pred = np.asarray([[1.0, 1.0], [4.0, 6.0]], dtype=np.float64)
    train_mean = np.asarray([1.0, 1.0], dtype=np.float64)

    result = pooled_multioutput_r2(y_true, y_pred, train_mean=train_mean)

    expected_sse = 2.0
    expected_sst = 0.0**2 + 1.0**2 + 2.0**2 + 5.0**2
    assert np.isclose(result.sse, expected_sse)
    assert np.isclose(result.sst, expected_sst)
    assert np.isclose(result.r2, 1.0 - expected_sse / expected_sst)
    assert result.method == R2_CV_METHOD


def test_pooled_multioutput_cv_r2_pools_sse_sst_not_fold_r2_mean() -> None:
    fold_sse = np.asarray([1.0, 10.0], dtype=np.float64)
    fold_sst = np.asarray([2.0, 100.0], dtype=np.float64)

    pooled = pooled_multioutput_r2_from_sse_sst(fold_sse, fold_sst)
    unweighted_mean_fold_r2 = np.mean(1.0 - fold_sse / fold_sst)

    assert np.isclose(pooled.r2, 1.0 - 11.0 / 102.0)
    assert not np.isclose(pooled.r2, unweighted_mean_fold_r2)


def test_per_sample_sse_sst_marks_nonfinite_rows_invalid() -> None:
    y_true = np.asarray([[1.0, 2.0], [np.nan, 1.0], [3.0, 4.0]], dtype=np.float64)
    y_pred = np.asarray([[1.5, 2.0], [1.0, 1.0], [4.0, 4.0]], dtype=np.float64)

    sse, sst, valid = per_sample_sse_sst(y_true, y_pred, train_mean=np.asarray([1.0, 1.0]))

    assert valid.tolist() == [True, False, True]
    assert np.isclose(sse[0], 0.25)
    assert np.isnan(sse[1])
    assert np.isclose(sst[2], 2.0**2 + 3.0**2)
