from __future__ import annotations

import numpy as np

from declan.fixation_statistics_by_stimulus.classifier import classify_stimulus_from_windows
from declan.fixation_statistics_by_stimulus.extraction import _speed_threshold_mad_valid_pairs
from declan.fixation_statistics_by_stimulus.features import event_feature_rows, fixation_window_features
from declan.fixation_statistics_by_stimulus.image_features import _radial_log_power_slope, image_axis_rad_to_gaze_axis_rad
from declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information import (
    _add_ssi_incremental_summaries,
    _summarize_ssi_features,
)
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _normalize_scores,
    _pixel_isophote_cost,
    _trace_xy_to_twin_helper_order,
)
from declan.fixation_statistics_by_stimulus.summaries import paired_metric_contrasts, summarize_events, summarize_windows


def test_fixation_window_features_reports_dispersion_and_steps() -> None:
    t = np.arange(64, dtype=np.float64)
    trace = np.column_stack([0.001 * t, 0.002 * np.sin(t / 5.0)])

    row = fixation_window_features(trace, dt=1.0 / 120.0)

    assert row["n_samples"] == 64
    assert row["rms_radius_deg"] > 0
    assert row["step_mean_deg"] > 0
    assert np.isfinite(row["anisotropy"])
    assert "msd_lag4_deg2" in row
    assert "position_psd_slope_1_30hz" in row


def test_event_feature_rows_computes_amplitude_direction_and_timing() -> None:
    trace = np.zeros((10, 2), dtype=np.float64)
    trace[5:, 0] = 0.3
    events = [{"onset": 4, "offset": 5, "peak_speed_deg_s": 40.0}]

    rows = event_feature_rows(trace, events, dt=0.01)

    assert len(rows) == 1
    assert np.isclose(rows[0]["event_amplitude_deg"], 0.3)
    assert np.isclose(rows[0]["event_direction_deg"], 0.0)
    assert np.isclose(rows[0]["event_onset_s"], 0.04)


def test_summaries_use_inventory_duration_for_event_rate() -> None:
    inventory = [
        {"session": "s1", "stimulus": "fixrsvp", "regime": "fixation_task", "n_trials": 2, "valid_duration_s": 10.0},
    ]
    events = [
        {"session": "s1", "stimulus": "fixrsvp", "regime": "fixation_task", "trial_idx": 0, "event_amplitude_deg": 0.1},
        {"session": "s1", "stimulus": "fixrsvp", "regime": "fixation_task", "trial_idx": 0, "event_amplitude_deg": 0.2},
    ]

    rows = summarize_events(events, inventory)

    assert len(rows) == 1
    assert rows[0]["n_events"] == 2
    assert np.isclose(rows[0]["event_rate_hz"], 0.2)


def test_paired_metric_contrasts_pairs_by_session_and_phase() -> None:
    rows = []
    for session, base, comp in [("s1", 1.0, 1.5), ("s2", 2.0, 3.0)]:
        for stim, value in [("fixrsvp", base), ("backimage", comp)]:
            rows.append({
                "session": session,
                "stimulus": stim,
                "regime": stim,
                "phase": "late_fixation",
                "trial_idx": 0,
                "rms_radius_deg": value,
            })

    summary = summarize_windows(rows, metrics=("rms_radius_deg",))
    contrasts = paired_metric_contrasts(summary, baseline="fixrsvp", metrics=("rms_radius_deg",), n_bootstrap=0)

    assert len(contrasts) == 1
    assert contrasts[0]["n_sessions"] == 2
    assert np.isclose(contrasts[0]["mean_diff"], 0.75)


def test_speed_threshold_ignores_jumps_across_invalid_gaps() -> None:
    trace = np.zeros((12, 2), dtype=np.float64)
    trace[:6, 0] = np.arange(6) * 0.01
    trace[6:, 0] = 100.0 + np.arange(6) * 0.01
    valid = np.ones(12, dtype=bool)
    valid[5:7] = False

    threshold = _speed_threshold_mad_valid_pairs(trace, valid, dt=0.01, z=6.0)

    assert threshold < 2.0


def test_classifier_skips_loso_fold_with_unseen_test_class() -> None:
    rows = []
    for session, stimuli in {
        "s1": ("a", "b"),
        "s2": ("a", "b"),
        "s3": ("c",),
    }.items():
        for stim in stimuli:
            center = {"a": 0.0, "b": 1.0, "c": 2.0}[stim]
            for i in range(12):
                rows.append({
                    "session": session,
                    "stimulus": stim,
                    "phase": "all",
                    "rms_radius_deg": center + i * 0.001,
                })

    summary = classify_stimulus_from_windows(rows, features=("rms_radius_deg",), seed=0)

    assert len(summary) >= 1
    assert summary[0]["cv"] == "leave_one_session_out"
    assert summary[0]["n_windows"] == 48


def test_twin_trace_preflip_offsets_ryan_internal_flip() -> None:
    trace_xy = np.asarray([[1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)

    helper_order = _trace_xy_to_twin_helper_order(trace_xy)

    assert np.allclose(helper_order, trace_xy[:, [1, 0]])
    assert np.allclose(np.fliplr(helper_order), trace_xy)


def test_pixel_isophote_cost_prefers_edge_parallel_motion() -> None:
    patch = np.zeros((201, 201), dtype=np.float32)
    patch[:, 101:] = 255.0

    horizontal_cost = _pixel_isophote_cost(patch, axis_deg=0.0, scale_deg=0.1, ppd=40.0)
    vertical_cost = _pixel_isophote_cost(patch, axis_deg=90.0, scale_deg=0.1, ppd=40.0)

    assert horizontal_cost > vertical_cost * 100.0


def test_image_axis_angles_convert_from_array_to_gaze_coordinates() -> None:
    assert np.isclose(np.degrees(image_axis_rad_to_gaze_axis_rad(np.radians(45.0))), -45.0)


def test_radial_log_power_slope_recovers_power_law() -> None:
    yy, xx = np.indices((129, 129), dtype=np.float64)
    fx = xx - 64.0
    fy = yy - 64.0
    rr = np.hypot(fx, fy)
    power = np.where(rr > 0, rr ** -2.0, 0.0)

    slope = _radial_log_power_slope(power, rr, min_cpd=1.0, max_cpd=32.0)

    assert np.isclose(slope, -2.0, atol=0.08)


def test_pixel_isophote_cost_uses_gaze_y_up_axis_convention() -> None:
    yy, xx = np.indices((201, 201), dtype=np.float32)
    patch = xx + yy

    edge_parallel_cost = _pixel_isophote_cost(patch, axis_deg=45.0, scale_deg=0.1, ppd=40.0)
    gradient_axis_cost = _pixel_isophote_cost(patch, axis_deg=135.0, scale_deg=0.1, ppd=40.0)

    assert gradient_axis_cost > edge_parallel_cost * 100.0


def test_normalize_scores_suppresses_near_constant_motor_noise() -> None:
    rows = [{"motor_cost": 0.48044570}, {"motor_cost": 0.48044572}, {"motor_cost": 0.48044574}]

    _normalize_scores(rows, ("motor_cost",))

    assert [row["motor_cost_z"] for row in rows] == [0.0, 0.0, 0.0]


def test_ssi_summaries_and_incremental_features_are_flat_and_named() -> None:
    ssi = {
        "ssi_itn": np.arange(6, dtype=np.float32).reshape(2, 3),
        "ssi_rbar_tn": np.ones((2, 3), dtype=np.float32) * 2.0,
        "ssi_ispike_t": np.asarray([0.5, 0.75], dtype=np.float32),
        "ssi_irate_t": np.asarray([4.0, 5.0], dtype=np.float32),
    }

    out = _summarize_ssi_features(ssi, ["ssi_itn", "ssi_unit_mean", "ssi_itn_plus_rbar", "ssi_population_time"])

    assert out["ssi_itn"].shape == (6,)
    assert out["ssi_unit_mean"].shape == (3,)
    assert out["ssi_itn_plus_rbar"].shape == (12,)
    assert out["ssi_population_time"].shape == (4,)
    assert np.allclose(out["ssi_unit_mean"], [1.5, 2.5, 3.5])

    summaries = {"delta_mean": np.asarray([10.0, 11.0], dtype=np.float32), **out}
    _add_ssi_incremental_summaries(
        summaries,
        base_summaries=["delta_mean"],
        ssi_summary_names=["ssi_unit_mean"],
    )

    assert "delta_mean_plus_ssi_unit_mean" in summaries
    assert np.allclose(summaries["delta_mean_plus_ssi_unit_mean"], [10.0, 11.0, 1.5, 2.5, 3.5])
