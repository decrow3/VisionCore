import numpy as np
from numpy.testing import assert_allclose

from declan.active_sensing_movie_information.temporal_remapping import (
    geometric_path_from_trace,
    invariance_report,
    retime_trace,
    scale_trace_about_center,
    trajectory_metrics,
)


def l_path(n: int = 32) -> np.ndarray:
    first = np.stack([np.linspace(0.0, 1.0, n // 2), np.zeros(n // 2)], axis=1)
    second = np.stack([np.ones(n - n // 2), np.linspace(0.0, 1.0, n - n // 2)], axis=1)
    return np.concatenate([first, second], axis=0).astype(np.float32)


def test_retime_terminal_holds_start_and_hits_endpoint():
    source = l_path()
    out = retime_trace(source, traversal_frames=8, total_frames=32, placement="terminal", profile="uniform")

    assert out.shape == (32, 2)
    assert_allclose(out[:24], np.repeat(source[0][None, :], 24, axis=0), atol=1e-7)
    assert_allclose(out[24], source[0], atol=1e-7)
    assert_allclose(out[-1], source[-1], atol=1e-7)


def test_retime_endpoint_hold_arrives_early_and_holds_endpoint():
    source = l_path()
    out = retime_trace(source, traversal_frames=8, total_frames=32, placement="endpoint_hold", profile="uniform")

    assert_allclose(out[0], source[0], atol=1e-7)
    assert_allclose(out[7:], np.repeat(source[-1][None, :], 25, axis=0), atol=1e-7)


def test_retime_centered_places_motion_in_middle():
    source = l_path()
    out = retime_trace(source, traversal_frames=8, total_frames=32, placement="centered", profile="uniform")

    assert_allclose(out[:12], np.repeat(source[0][None, :], 12, axis=0), atol=1e-7)
    assert_allclose(out[12], source[0], atol=1e-7)
    assert_allclose(out[19:], np.repeat(source[-1][None, :], 13, axis=0), atol=1e-7)


def test_continuous_geometry_invariant_across_durations_and_placements():
    source = l_path()
    rows = []
    for frames in (8, 16, 32):
        for placement in ("terminal", "endpoint_hold", "centered"):
            out = retime_trace(source, traversal_frames=frames, total_frames=32, placement=placement, profile="uniform")
            rows.append(
                trajectory_metrics(
                    out,
                    source_trace=source,
                    traversal_frames=frames,
                    total_frames=32,
                    timing_placement=placement,
                    retiming_profile="uniform",
                )
            )

    report = invariance_report(rows)
    assert report["ok"], report
    assert_allclose(rows[0]["path_length_deg"], geometric_path_from_trace(source).path_length_deg)


def test_natural_speed_profile_preserves_source_progress_nonuniformity():
    source = np.asarray([[0.0, 0.0], [0.0, 0.0], [0.8, 0.0], [1.0, 0.0]], dtype=np.float32)
    uniform = retime_trace(source, traversal_frames=4, total_frames=4, placement="terminal", profile="uniform")
    natural = retime_trace(source, traversal_frames=4, total_frames=4, placement="terminal", profile="natural_speed_profile")

    assert_allclose(uniform[:, 0], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], atol=1e-6)
    assert_allclose(natural[:, 0], [0.0, 0.0, 0.8, 1.0], atol=1e-6)


def test_scale_trace_beta_zero_is_stationary_at_centroid():
    source = l_path(10)
    scaled = scale_trace_about_center(source, 0.0, center="centroid")
    centroid = np.mean(source, axis=0)

    assert_allclose(scaled, np.repeat(centroid[None, :], source.shape[0], axis=0), atol=1e-7)


def test_trajectory_metrics_flags_super_nyquist_characteristic_tf():
    trace = np.stack([np.zeros(32), np.linspace(0.0, 32.0, 32)], axis=1).astype(np.float32)
    metrics = trajectory_metrics(
        trace,
        source_trace=trace,
        traversal_frames=32,
        total_frames=32,
        contour_orientation_deg=0.0,
        preferred_sf_cpd=1.0,
    )

    assert metrics["rms_across_velocity_deg_s"] > 60.0
    assert metrics["characteristic_motion_tf_hz"] > 60.0
    assert metrics["exceeds_model_nyquist"]
    assert metrics["nyquist_margin_hz"] < 0.0
