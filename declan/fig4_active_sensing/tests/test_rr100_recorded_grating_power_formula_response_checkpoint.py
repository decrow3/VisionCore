import numpy as np
import pandas as pd

from declan.fig4_active_sensing.analyze_rr100_recorded_grating_power_formula_response_checkpoint import (
    NESTED_FORMULA,
    formula_differences,
    grouped_cv_radial_plus_signed_delta,
)


def test_grouped_cv_radial_plus_signed_delta_holds_out_complete_folds() -> None:
    x = np.column_stack([np.arange(12, dtype=float), np.tile([-1.0, 1.0], 6)])
    y = 3.0 + 2.0 * x[:, 0] - 0.5 * x[:, 1]
    folds = np.repeat(np.arange(3), 4)
    prediction, baseline, fits = grouped_cv_radial_plus_signed_delta(x, y, folds)
    np.testing.assert_allclose(prediction, y, atol=1e-10)
    assert np.isfinite(baseline).all()
    assert set(fits.fold) == {0, 1, 2}
    assert (fits.n_test == 4).all()
    assert not fits.radial_nonnegative_constraint_active.any()


def test_nested_fit_enforces_nonnegative_radial_coefficient() -> None:
    radial = np.arange(18, dtype=float)
    delta = np.tile([-1.0, 0.0, 1.0], 6)
    x = np.column_stack([radial, delta])
    y = 5.0 - 3.0 * radial + 2.0 * delta
    folds = np.repeat(np.arange(3), 6)
    _, _, fits = grouped_cv_radial_plus_signed_delta(x, y, folds)
    assert fits.radial_nonnegative_constraint_active.all()
    np.testing.assert_allclose(fits.standardized_coefficient_0, 0.0)


def test_formula_differences_uses_matched_unit_target_rows() -> None:
    rows = []
    values = {
        "rf_local_radial_direct_f0": 0.10,
        "rf_local_oriented_direct_f0": 0.13,
        NESTED_FORMULA: 0.16,
        "rf_local_sf_tf_h2": 0.08,
    }
    for unit in (2, 7):
        for target in ("recorded", "full_twin"):
            for formula, value in values.items():
                rows.append(
                    {
                        "rr100_index": unit,
                        "target": target,
                        "formula": formula,
                        "heldout_cv_r2": value + unit / 1000.0,
                    }
                )
    differences = formula_differences(pd.DataFrame(rows))
    np.testing.assert_allclose(
        differences.oriented_minus_radial_direct_f0_cv_r2, 0.03
    )
    np.testing.assert_allclose(
        differences.nested_minus_radial_direct_f0_cv_r2, 0.06
    )
    np.testing.assert_allclose(
        differences.h2_minus_radial_direct_f0_cv_r2, -0.02
    )
