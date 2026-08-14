import numpy as np
import pandas as pd

from declan.active_sensing_movie_information.run_backimage_rr100_dense_sf_tf_grating_probe import (
    aggregate_rows,
    validate_pair_shard,
)


def synthetic_complete_pair() -> pd.DataFrame:
    rows = []
    for orientation in (0.0, 45.0, 90.0, 135.0):
        for direction in (-1, 1):
            for phase in range(4):
                for unit in range(100):
                    signed_f0 = 0.1 * unit + 0.01 * phase
                    rows.append(
                        {
                            "unit_index": unit,
                            "unit_label": f"u{unit:03d}",
                            "pair_id": 0,
                            "speed_family": "cycle_valid",
                            "speed_dps": 32.0,
                            "log2_speed_dps": 5.0,
                            "spatial_cpd": 1.0,
                            "temporal_hz": 32.0,
                            "temporal_direction_sign": direction,
                            "signed_temporal_hz": direction * 32.0,
                            "log2_spatial_cpd": 0.0,
                            "log2_temporal_hz": 5.0,
                            "cycles_across_window": 2.69,
                            "is_cycle_valid_sf": True,
                            "is_extended_tf_core": False,
                            "is_nyquist_edge_control": False,
                            "n_spatial_cpds_for_family": 8,
                            "n_temporal_hz_for_family": 14,
                            "n_temporal_hz_for_spatial_cpd": 14,
                            "n_spatial_cpds_for_temporal_hz": 8,
                            "probe_orientation_deg": orientation,
                            "prior_preferred_orientation_deg": orientation,
                            "prior_orientation_selectivity_index": 0.5,
                            "phase_index": phase,
                            "phase_policy": "dynamic_uniform_grid",
                            "scalar_readout": "center_pixel",
                            "mean_rate": signed_f0 + 1.0,
                            "blank_mean_rate": 1.0,
                            "signed_f0_hz": signed_f0,
                            "positive_f0_hz": max(signed_f0, 0.0),
                            "response_amp_sq": 0.25,
                            "response_amp": 0.5,
                            "contrast": 0.8,
                            "probe_contract": "synthetic complete atomic pair",
                        }
                    )
    return pd.DataFrame(rows)


def test_complete_pair_is_immediately_analyzable() -> None:
    frame = synthetic_complete_pair()
    validate_pair_shard(frame, pair_id=0, n_units=100, n_conditions=32)
    grouped, surface = aggregate_rows(frame.to_dict("records"))
    assert len(frame) == 3200
    assert len(grouped) == 800
    assert len(surface) == 100
    assert grouped["n_phases"].eq(4).all()
    assert surface["n_orientations"].eq(4).all()
    assert surface["n_directions"].eq(2).all()
    assert np.isfinite(surface["positive_f0_hz_mean"]).all()

