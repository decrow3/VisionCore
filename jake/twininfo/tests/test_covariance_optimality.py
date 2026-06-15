from __future__ import annotations

import numpy as np

from jake.twininfo.covariance_optimality import (
    alignment_rows,
    coding_covariance_from_j,
    covariance_residual_after_subspace,
    covariance_residual_noise_side,
    covariance_fisher_by_time,
    geometry_covariance_rows,
    independent_fisher_by_time,
    movement_covariance_pooled_residual,
    parse_scaled_condition,
    scale_trace,
    scaled_condition_name,
    sensitivity_metric_rows,
    signal_covariance_from_pair_means,
    trajectories_for_scaled_family,
    trajectory_for_scaled_family,
)
from jake.twininfo.run_covariance_optimality import _center_cache_condition, _population_metadata_subset
from declan.active_sensing_movie_information.summarize_covariance_optimality import paired_contrasts


def _trace_stats(trace: np.ndarray) -> tuple[float, float]:
    centered = trace - np.mean(trace, axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    cov_trace = float(np.trace(np.cov(centered.T)))
    return rms, cov_trace


def test_scaled_condition_roundtrip():
    name = scaled_condition_name("scaled_real", 0.5)
    parsed = parse_scaled_condition(name)
    assert parsed.family == "scaled_real"
    assert parsed.scale == 0.5
    assert parsed.condition == "scaled_real_D0p5"


def test_scale_trace_invariants_and_quadratic_covariance():
    t = np.arange(80, dtype=np.float32)
    trace = np.column_stack([0.01 * np.sin(t / 9.0), 0.02 * np.cos(t / 7.0)]).astype(np.float32)
    center = np.mean(trace, axis=0, keepdims=True)

    stable = scale_trace(trace, 0.0)
    empirical = scale_trace(trace, 1.0)
    doubled = scale_trace(trace, 2.0)

    assert np.allclose(stable, np.repeat(center, trace.shape[0], axis=0))
    assert np.allclose(empirical, trace)
    rms1, cov1 = _trace_stats(empirical)
    rms2, cov2 = _trace_stats(doubled)
    assert np.isclose(rms2, 2.0 * rms1, rtol=1e-6)
    assert np.isclose(cov2, 4.0 * cov1, rtol=1e-6)


def test_scaled_random_controls_are_deterministic_and_centered():
    t = np.arange(64, dtype=np.float32)
    trace = np.column_stack([0.002 * np.sin(t / 5.0), 0.003 * np.cos(t / 11.0)]).astype(np.float32)
    center = np.mean(trace, axis=0)
    a, desc_a = trajectory_for_scaled_family(trace, "random_amp_scaled", 0.25, t_max=64, seed=4)
    b, desc_b = trajectory_for_scaled_family(trace, "random_amp_scaled", 0.25, t_max=64, seed=4)

    assert desc_a == desc_b
    assert np.allclose(a, b)
    assert np.allclose(np.mean(a, axis=0), center, atol=1e-7)
    assert _trace_stats(a)[0] < _trace_stats(trace)[0]


def test_batched_scaled_control_matches_single_scaled_control():
    t = np.arange(64, dtype=np.float32)
    trace = np.column_stack([0.002 * np.sin(t / 5.0), 0.003 * np.cos(t / 11.0)]).astype(np.float32)
    batched = trajectories_for_scaled_family(trace, "random_amp_scaled", (0.25, 1.0), t_max=64, seed=4)
    single, single_desc = trajectory_for_scaled_family(trace, "random_amp_scaled", 0.25, t_max=64, seed=4)

    assert np.allclose(batched[0.25][0], single)
    assert batched[0.25][1] == single_desc
    assert batched[1.0][0].shape == trace.shape


def test_covariance_fisher_matches_independent_when_no_extra_covariance():
    rng = np.random.default_rng(1)
    mu = rng.uniform(0.2, 1.0, size=(6, 5))
    jac = rng.normal(scale=0.1, size=(6, 5, 2))
    independent = covariance_fisher_by_time(mu, jac, None, ridge_frac=0.0)
    manual = np.zeros_like(independent)
    for t in range(mu.shape[0]):
        manual[t] = jac[t].T @ (jac[t] / mu[t, None].T)
    assert np.allclose(independent, manual, rtol=1e-6, atol=1e-8)


def test_zero_extra_covariance_uses_matched_covariance_path():
    rng = np.random.default_rng(11)
    mu = rng.uniform(0.2, 1.0, size=(6, 5))
    jac = rng.normal(scale=0.1, size=(6, 5, 2))
    ridge_frac = 1e-4

    pose_aware = covariance_fisher_by_time(mu, jac, None, ridge_frac=ridge_frac)
    pose_blind_zero = covariance_fisher_by_time(mu, jac, np.zeros((5, 5)), ridge_frac=ridge_frac)
    independent = independent_fisher_by_time(mu, jac)

    assert np.allclose(pose_aware, pose_blind_zero, rtol=1e-10, atol=1e-12)
    assert not np.allclose(pose_aware, independent, rtol=1e-12, atol=1e-14)


def test_psd_extra_covariance_does_not_increase_fisher():
    rng = np.random.default_rng(2)
    mu = rng.uniform(0.5, 1.5, size=(8, 6))
    jac = rng.normal(scale=0.2, size=(8, 6, 2))
    v = rng.normal(size=(6, 2))
    sigma = v @ v.T
    aware = np.sum(covariance_fisher_by_time(mu, jac, None, ridge_frac=0.0), axis=0)
    blind = np.sum(covariance_fisher_by_time(mu, jac, sigma, ridge_frac=0.0), axis=0)
    assert np.trace(blind) <= np.trace(aware) + 1e-8


def test_pose_blind_penalty_is_larger_when_covariance_aligns_with_j():
    mu = np.ones((5, 4), dtype=np.float64)
    jac = np.zeros((5, 4, 2), dtype=np.float64)
    jac[:, 0, 0] = 1.0
    aligned = np.diag([10.0, 0.0, 0.0, 0.0])
    orthogonal = np.diag([0.0, 10.0, 0.0, 0.0])

    f_aligned = np.trace(np.sum(covariance_fisher_by_time(mu, jac, aligned, ridge_frac=0.0), axis=0))
    f_orth = np.trace(np.sum(covariance_fisher_by_time(mu, jac, orthogonal, ridge_frac=0.0), axis=0))
    assert f_aligned < f_orth


def test_movement_covariance_pooled_residual_scales_like_d_squared():
    rng = np.random.default_rng(3)
    base = [rng.normal(size=(20, 5)) for _ in range(4)]
    cov1, _, _ = movement_covariance_pooled_residual(base)
    cov2, _, _ = movement_covariance_pooled_residual([2.0 * arr for arr in base])
    assert np.isclose(np.trace(cov2), 4.0 * np.trace(cov1), rtol=1e-6)


def test_sensitivity_rows_record_gain_noise_grid():
    mu = np.ones((4, 3), dtype=np.float64) * 0.5
    jac = np.ones((4, 3, 2), dtype=np.float64) * 0.1
    sigma = np.eye(3)
    rows = sensitivity_metric_rows(
        row_id=0,
        record={"example_id": "a", "kind": "fixation", "image_index": 1, "crop_rank": 0},
        family="scaled_real",
        scale=1.0,
        mu_tn=mu,
        j_tnd=jac,
        sigma_extra=sigma,
        rate_gains=(0.5, 1.0),
        noise_floor_multipliers=(1.0, 2.0),
        ridge_frac=0.0,
    )
    assert len(rows) == 4
    assert {(row["rate_gain"], row["noise_floor_multiplier"]) for row in rows} == {
        (0.5, 1.0),
        (0.5, 2.0),
        (1.0, 1.0),
        (1.0, 2.0),
    }


def test_gain_sensitivity_scales_extra_covariance_quadratically():
    mu = np.ones((4, 3), dtype=np.float64) * 0.5
    jac = np.zeros((4, 3, 2), dtype=np.float64)
    jac[:, 0, 0] = 1.0
    sigma = np.diag([1.0, 0.0, 0.0])
    rows = sensitivity_metric_rows(
        row_id=0,
        record={"example_id": "a", "kind": "fixation", "image_index": 1, "crop_rank": 0},
        family="scaled_real",
        scale=1.0,
        mu_tn=mu,
        j_tnd=jac,
        sigma_extra=sigma,
        rate_gains=(0.5, 2.0),
        noise_floor_multipliers=(1.0,),
        ridge_frac=0.0,
    )
    low = [row for row in rows if row["rate_gain"] == 0.5][0]["final_fisher_trace"]
    high = [row for row in rows if row["rate_gain"] == 2.0][0]["final_fisher_trace"]
    assert high < 4.0 * low


def test_alignment_separates_coding_and_signal_subspaces():
    j = np.zeros((5, 4, 2), dtype=np.float64)
    j[:, 0, 0] = 1.0
    coding = coding_covariance_from_j([j])
    sigma = np.diag([5.0, 0.0, 0.0, 0.0])
    signal = np.diag([0.0, 0.0, 5.0, 0.0])
    rows = alignment_rows(
        family="scaled_real",
        scale=1.0,
        kind="fixation",
        sigma_fem=sigma,
        coding_cov=coding,
        signal_cov=signal,
        k_list=(1,),
    )
    assert rows[0]["coding_variance_fem"] > 0.99
    assert rows[0]["signal_variance_fem"] < 1e-9


def test_covariance_residual_after_subspace_removes_top_mode():
    sigma = np.diag([4.0, 1.0, 0.25])
    basis = np.eye(3, 1)
    compact, residual = covariance_residual_after_subspace(sigma, basis)

    assert np.isclose(np.trace(compact), 4.0)
    assert np.isclose(np.trace(residual), 1.25)
    assert np.all(np.linalg.eigvalsh(residual) >= -1e-10)


def test_covariance_residual_noise_side_uses_R_sigma_R():
    sigma = np.array(
        [
            [2.0, 0.7, 0.0],
            [0.7, 1.0, 0.2],
            [0.0, 0.2, 0.5],
        ],
        dtype=np.float64,
    )
    basis = np.array([[1.0], [0.0], [0.0]], dtype=np.float64)
    _compact, residual = covariance_residual_noise_side(sigma, basis)
    r = np.diag([0.0, 1.0, 1.0])

    assert np.allclose(residual, r @ sigma @ r)
    assert np.all(np.linalg.eigvalsh(residual) >= -1e-10)


def test_noise_side_residual_closes_synthetic_J_covariance():
    rng = np.random.default_rng(0)
    j_cols = rng.normal(size=(6, 2))
    eye_cov = np.array([[1.0, 0.25], [0.25, 0.5]], dtype=np.float64)
    sigma_fem = j_cols @ eye_cov @ j_cols.T
    _compact, residual = covariance_residual_noise_side(sigma_fem, j_cols)

    mu = np.full((4, 6), 2.0, dtype=np.float64)
    jac = np.broadcast_to(j_cols[None, :, :], (4, 6, 2)).copy()
    aware = np.trace(np.sum(covariance_fisher_by_time(mu, jac, None, ridge_frac=0.0), axis=0))
    blind = np.trace(np.sum(covariance_fisher_by_time(mu, jac, sigma_fem, ridge_frac=0.0), axis=0))
    corrected = np.trace(np.sum(covariance_fisher_by_time(mu, jac, residual, ridge_frac=0.0), axis=0))
    closure = (corrected - blind) / (aware - blind)

    assert np.trace(residual) / np.trace(sigma_fem) < 1e-10
    assert np.isclose(closure, 1.0, atol=1e-10)


def test_noise_side_residual_D0_null_keeps_fisher_equal():
    mu = np.full((3, 4), 1.5, dtype=np.float64)
    jac = np.zeros((3, 4, 2), dtype=np.float64)
    jac[:, 0, 0] = 1.0
    basis = np.eye(4, 2)
    sigma_zero = np.zeros((4, 4), dtype=np.float64)
    _compact, residual = covariance_residual_noise_side(sigma_zero, basis)

    aware = covariance_fisher_by_time(mu, jac, None, ridge_frac=0.0)
    blind = covariance_fisher_by_time(mu, jac, sigma_zero, ridge_frac=0.0)
    corrected = covariance_fisher_by_time(mu, jac, residual, ridge_frac=0.0)

    assert np.isclose(np.trace(sigma_zero), 0.0)
    assert np.allclose(aware, blind)
    assert np.allclose(aware, corrected)


def test_geometry_covariance_rows_track_removed_nuisance_and_signal_overlap():
    sigma = np.diag([4.0, 1.0, 0.0])
    coding = np.diag([0.0, 3.0, 0.0])
    signal = np.diag([2.0, 0.0, 0.0])

    rows = geometry_covariance_rows(
        family="scaled_real",
        scale=1.0,
        kind="fixation",
        sigma_fem=sigma,
        coding_cov=coding,
        signal_cov=signal,
        k_list=(1,),
    )

    row = rows[0]
    assert np.isclose(row["nuisance_variance_removed_fraction"], 0.8)
    assert np.isclose(row["nuisance_variance_remaining_fraction"], 0.2)
    assert row["signal_variance_geometry"] > 0.99
    assert row["coding_variance_geometry"] < 1e-9


def test_signal_covariance_handles_single_pair():
    signal = signal_covariance_from_pair_means([np.ones((5, 3), dtype=np.float64)])
    assert signal.shape == (3, 3)
    assert np.allclose(signal, 0.0)


def test_population_metadata_subset_picks_center_channel_per_biological_unit():
    rows = []
    sim = 0
    for global_unit in [0, 1]:
        for grid_row, grid_col in [(0, 0), (2, 2), (4, 4)]:
            rows.append({
                "global_unit_idx": str(global_unit),
                "simulated_unit_idx": str(sim),
                "grid_row": str(grid_row),
                "grid_col": str(grid_col),
                "grid_shape_h": "5",
                "grid_shape_w": "5",
            })
            sim += 1
    subset = _population_metadata_subset(rows, "sampled_units")
    assert len(subset) == 2
    assert [int(row["global_unit_idx"]) for row in subset] == [0, 1]
    assert all(int(row["grid_row"]) == 2 and int(row["grid_col"]) == 2 for row in subset)
    assert len(_population_metadata_subset(rows, "metadata_all")) == len(rows)


def test_center_rate_cache_condition_mapping():
    assert _center_cache_condition("scaled_real", 0.0) == "stabilized"
    assert _center_cache_condition("random_amp_scaled", 0.0) == "stabilized"
    assert _center_cache_condition("scaled_real", 1.0) == "real"
    assert _center_cache_condition("random_amp_scaled", 1.0) == "random_amp"
    assert _center_cache_condition("random_amp_cloud_matched_scaled", 1.0) == "random_amp_cloud_matched"
    assert _center_cache_condition("trajectory_order_shuffle_scaled", 1.0) == "trajectory_order_shuffle"
    assert _center_cache_condition("scaled_real", 0.5) is None


def test_covopt_summary_contrasts_are_explicit_and_d0_corrected():
    rows = []
    for scale, aware, geometry, blind, ind in [
        (0.0, 10.0, 9.8, 9.5, 10.2),
        (1.0, 12.0, 11.0, 10.0, 12.4),
    ]:
        for regime, value in [
            ("cov_pose_aware", aware),
            ("cov_geometry_aware_k1", geometry),
            ("cov_pose_blind", blind),
            ("independent_pose_aware", ind),
        ]:
            rows.append(
                {
                    "example_id": "a",
                    "kind": "fixation",
                    "image_index": "1",
                    "crop_rank": "0",
                    "family": "scaled_real",
                    "scale_D": str(scale),
                    "regime": regime,
                    "rate_gain": "1.0",
                    "noise_floor_multiplier": "1.0",
                    "final_fisher_trace_per_spike": str(value),
                }
            )

    out = {
        (row["scale_D"], row["contrast"]): row
        for row in paired_contrasts(rows)
    }

    assert np.isclose(out[(1.0, "pose_gap")]["mean"], 2.0)
    assert np.isclose(out[(1.0, "pose_gap_minus_D0")]["mean"], 1.5)
    assert np.isclose(out[(1.0, "independent_minus_cov_pose_aware")]["mean"], 0.4)
    assert np.isclose(out[(1.0, "independent_minus_cov_pose_blind")]["mean"], 2.4)
    assert np.isclose(out[(1.0, "pose_to_geometry_gap_k1")]["mean"], 1.0)
    assert np.isclose(out[(1.0, "geometry_to_blind_gap_k1")]["mean"], 1.0)
    assert np.isclose(out[(1.0, "geometry_fraction_of_pose_gap_k1")]["mean"], 0.5)
