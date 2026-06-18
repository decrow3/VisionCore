"""Tests for axis-conditioned trace construction."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

from declan.axis_conditioned_backimage_trajectory_observer.axis_conditioned_traces import (
    axis_conditioned_trace,
    axis_perp,
    axis_unit,
    matched_axis_trace_pair,
    trace_metrics,
)
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import (
    _axis_candidate_meta_rows,
    _axis_per_candidate_prior_trajectories,
    _axis_shared_sampled_source_indices,
    _nested_duplicate_trace_count,
    _trace_from_item,
    _trajectory_spec,
)


def _source_trace() -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    return np.column_stack(
        [
            0.08 * np.sin(t) + 0.01 * np.sin(3.0 * t),
            0.03 * np.cos(t + 0.2),
        ]
    ).astype(np.float32)


def test_axis_unit_and_perp_are_orthonormal() -> None:
    u = axis_unit(30.0)
    v = axis_perp(30.0)
    assert np.isclose(np.linalg.norm(u), 1.0)
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert abs(float(u @ v)) < 1e-12


def test_matched_pair_has_same_rms_path_duration_and_clipping() -> None:
    pair = matched_axis_trace_pair(
        _source_trace(),
        edge_axis_deg=25.0,
        template_mode="same_dominant_projection",
        scale=0.75,
        max_rms_deg=0.2,
        source_id="source-row-7",
    )
    par = pair["parallel"]
    orth = pair["orthogonal"]
    assert par["trace"].shape == orth["trace"].shape == (40, 2)
    assert par["meta"]["axis_match_status"] == "matched"
    assert orth["meta"]["axis_match_status"] == "matched"
    assert math.isclose(
        par["meta"]["rendered_rms_displacement_deg"],
        orth["meta"]["rendered_rms_displacement_deg"],
        abs_tol=1e-7,
    )
    assert math.isclose(
        par["meta"]["rendered_path_length_deg"],
        orth["meta"]["rendered_path_length_deg"],
        abs_tol=1e-7,
    )
    assert par["meta"]["rendered_duration_s"] == orth["meta"]["rendered_duration_s"]
    assert par["meta"]["clipping_fraction"] == orth["meta"]["clipping_fraction"]
    assert par["meta"]["path_length_deg"] == par["meta"]["rendered_path_length_deg"]
    assert math.isfinite(par["meta"]["generated_lag1_autocorr"])
    assert math.isfinite(par["meta"]["speed_mean_deg_s"])
    assert math.isfinite(par["meta"]["speed_p95_deg_s"])
    assert "source-row-7" in par["meta"]["axis_pair_id"]
    assert "scale-0.75" in par["meta"]["axis_pair_id"]
    assert par["meta"]["axis_pair_id"] == orth["meta"]["axis_pair_id"]


def test_parallel_and_orthogonal_traces_lie_on_requested_axes() -> None:
    edge_axis = 40.0
    pair = matched_axis_trace_pair(_source_trace(), edge_axis_deg=edge_axis, template_mode="arclength_signed")
    u = axis_unit(edge_axis)
    v = axis_perp(edge_axis)
    par_trace = pair["parallel"]["trace"].astype(np.float64)
    orth_trace = pair["orthogonal"]["trace"].astype(np.float64)
    assert np.max(np.abs(par_trace @ v)) < 1e-7
    assert np.max(np.abs(orth_trace @ u)) < 1e-7


def test_axis_conditioned_trace_reports_rms_clipping() -> None:
    trace, meta = axis_conditioned_trace(
        _source_trace(),
        axis_deg=0.0,
        relation="parallel",
        template_mode="same_parallel_projection",
        scale=10.0,
        max_rms_deg=0.05,
    )
    metrics = trace_metrics(trace)
    assert meta["rms_clipped_high"] is True
    assert 0.0 < meta["clipping_fraction"] < 1.0
    assert math.isclose(metrics["rms_displacement_deg"], 0.05, rel_tol=1e-6, abs_tol=1e-7)
    assert math.isclose(meta["effective_rms_deg"], 0.05, rel_tol=1e-6, abs_tol=1e-7)


def test_zero_source_trace_returns_finite_zero_axis_trace() -> None:
    source = np.zeros((8, 2), dtype=np.float32)
    pair = matched_axis_trace_pair(source, edge_axis_deg=90.0)
    for item in pair.values():
        assert np.isfinite(item["trace"]).all()
        assert np.allclose(item["trace"], 0.0)
        assert item["meta"]["rendered_rms_displacement_deg"] == 0.0
        assert item["meta"]["axis_match_status"] == "matched"


def test_degenerate_requested_motion_is_not_marked_matched() -> None:
    source = np.zeros((8, 2), dtype=np.float32)
    pair = matched_axis_trace_pair(source, edge_axis_deg=90.0, target_rms_deg=0.1)
    for item in pair.values():
        assert np.isfinite(item["trace"]).all()
        assert np.allclose(item["trace"], 0.0)
        assert item["meta"]["degenerate_requested_motion"] is True
        assert item["meta"]["axis_match_status"] == "invalid_degenerate"
        assert item["meta"]["axis_match_degenerate"] is True


def test_negative_scale_target_and_cap_fail_loudly() -> None:
    for kwargs in (
        {"scale": -1.0},
        {"target_rms_deg": -0.1},
        {"max_rms_deg": -0.1},
    ):
        try:
            axis_conditioned_trace(_source_trace(), axis_deg=0.0, relation="parallel", **kwargs)
        except ValueError as exc:
            assert "non-negative" in str(exc)
        else:
            raise AssertionError(f"Expected {kwargs} to fail")


def test_axis_pair_id_changes_with_source_scale_and_cap() -> None:
    base = matched_axis_trace_pair(_source_trace(), edge_axis_deg=25.0, scale=0.5, max_rms_deg=0.2, source_id=1)
    other_source = matched_axis_trace_pair(_source_trace(), edge_axis_deg=25.0, scale=0.5, max_rms_deg=0.2, source_id=2)
    other_scale = matched_axis_trace_pair(_source_trace(), edge_axis_deg=25.0, scale=1.0, max_rms_deg=0.2, source_id=1)
    other_cap = matched_axis_trace_pair(_source_trace(), edge_axis_deg=25.0, scale=0.5, max_rms_deg=0.1, source_id=1)
    ids = {
        base["parallel"]["meta"]["axis_pair_id"],
        other_source["parallel"]["meta"]["axis_pair_id"],
        other_scale["parallel"]["meta"]["axis_pair_id"],
        other_cap["parallel"]["meta"]["axis_pair_id"],
    }
    assert len(ids) == 4


def test_axis_trace_from_item_feeds_trajectory_spec_metadata() -> None:
    item = {
        "source_row": 17,
        "session": "session",
        "trace": _source_trace(),
        "observed_rms_deg": 0.08,
        "path_length_deg": 0.5,
        "lag1_autocorr": 0.2,
        "image_edge_axis_deg": 35.0,
    }
    trace, meta = _trace_from_item(
        family="axis_edge_parallel",
        item=item,
        scale=0.5,
        rng=np.random.default_rng(0),
        max_rms_deg=0.2,
        axis_source_column="image_edge_axis_deg",
        axis_template_mode="same_dominant_projection",
        axis_match_policy="strict",
    )
    spec = _trajectory_spec(
        role="prior",
        family="axis_edge_parallel",
        scale=0.5,
        trace=trace,
        item=item,
        meta=meta,
        sample_index=0,
        is_true=False,
    )
    assert spec["path_length_deg"] > 0.0
    assert math.isfinite(spec["generated_lag1_autocorr"])
    assert math.isfinite(spec["speed_mean_deg_s"])
    assert math.isfinite(spec["speed_p95_deg_s"])
    assert spec["axis_conditioned"] is True
    assert spec["axis_relation"] == "parallel"
    assert spec["axis_match_status"] == "matched"
    assert "src-17" in spec["axis_pair_id"]


def test_axis_trace_from_item_requires_axis_column() -> None:
    item = {
        "source_row": 17,
        "session": "session",
        "trace": _source_trace(),
        "observed_rms_deg": 0.08,
        "path_length_deg": 0.5,
        "lag1_autocorr": 0.2,
    }
    try:
        _trace_from_item(
            family="axis_edge_parallel",
            item=item,
            scale=0.5,
            rng=np.random.default_rng(0),
            max_rms_deg=0.2,
            axis_source_column="image_edge_axis_deg",
        )
    except ValueError as exc:
        assert "image_edge_axis_deg" in str(exc)
    else:
        raise AssertionError("Expected axis-conditioned family to require an axis column")


def test_per_candidate_axis_catalog_uses_candidate_local_axes() -> None:
    trace = _source_trace()
    trace_bank = [
        {
            "source_row": 1,
            "session": "s",
            "trace": trace,
            "observed_rms_deg": 0.08,
            "path_length_deg": 0.5,
            "lag1_autocorr": 0.2,
            "source_max_radius_deg": 0.1,
            "source_path_length_deg": 0.5,
            "source_speed_p95_deg_s": 1.0,
            "n_microsaccade_events": 0,
        },
        {
            "source_row": 2,
            "session": "s",
            "trace": trace * 0.75,
            "observed_rms_deg": 0.06,
            "path_length_deg": 0.4,
            "lag1_autocorr": 0.1,
            "source_max_radius_deg": 0.1,
            "source_path_length_deg": 0.4,
            "source_speed_p95_deg_s": 1.0,
            "n_microsaccade_events": 0,
        },
        {
            "source_row": 3,
            "session": "s",
            "trace": trace * 0.5,
            "observed_rms_deg": 0.04,
            "path_length_deg": 0.3,
            "lag1_autocorr": 0.05,
            "source_max_radius_deg": 0.1,
            "source_path_length_deg": 0.3,
            "source_speed_p95_deg_s": 1.0,
            "n_microsaccade_events": 0,
        },
    ]
    work = pd.DataFrame(
        {
            "source_row": [1, 3],
            "image_edge_axis_deg": [0.0, 90.0],
        }
    )
    args = SimpleNamespace(
        trajectory_prior_mode="leave_one_out",
        loo_exclude_trace_rmse_deg=1e-9,
        max_trace_source_rms_deg=None,
        max_trace_source_radius_deg=None,
        max_trace_source_path_length_deg=None,
        max_rendered_trace_path_length_deg=None,
        max_source_trace_path_length_deg=None,
        max_trace_source_speed_p95_deg_s=None,
        max_trace_source_microsaccade_events=None,
        axis_source_column="image_edge_axis_deg",
        axis_template_mode="same_dominant_projection",
        axis_match_policy="strict",
        max_rms_deg=0.2,
        seed=3,
    )
    traces, specs, true_idx, meta = _axis_per_candidate_prior_trajectories(
        current_source_row=1,
        observation_trace=trace,
        prior_family="axis_edge_parallel",
        prior_scale=0.5,
        n_prior_trajectories=1,
        trace_bank=trace_bank,
        candidate_indices=[0, 1],
        candidate_ids=["true", "other"],
        work=work,
        args=args,
        rng=np.random.default_rng(0),
    )
    assert true_idx == -1
    assert meta["axis_catalog_mode"] == "per_candidate"
    assert meta["excluded_candidate_source_rows"] == "1,3"
    assert meta["excluded_candidate_source_row_count"] == 2
    assert len(traces) == len(specs) == 2
    assert len(traces[0]) == len(traces[1]) == 1
    assert specs[0][0]["source_row"] == 2
    assert specs[1][0]["source_row"] == 2
    assert specs[0][0]["axis_candidate_id"] == "true"
    assert specs[1][0]["axis_candidate_id"] == "other"
    assert specs[0][0]["axis_candidate_axis_deg"] == 0.0
    assert specs[1][0]["axis_candidate_axis_deg"] == 90.0
    assert specs[0][0]["axis_pair_id"] != specs[1][0]["axis_pair_id"]
    assert "candidate-true" in specs[0][0]["axis_pair_id"]
    assert "candidate-other" in specs[1][0]["axis_pair_id"]
    assert specs[0][0]["axis_source_column"] == "image_edge_axis_deg"
    assert specs[0][0]["axis_catalog_mode"] == "per_candidate"
    assert specs[0][0]["axis_match_status"] == "matched"
    assert specs[1][0]["axis_match_status"] == "matched"


def test_per_candidate_axis_catalog_rejects_rendered_near_duplicate_traces() -> None:
    trace = _source_trace()
    trace_bank = [
        {
            "source_row": 1,
            "session": "s",
            "trace": trace,
            "observed_rms_deg": 0.08,
            "path_length_deg": 0.5,
            "lag1_autocorr": 0.2,
            "source_max_radius_deg": 0.1,
            "source_path_length_deg": 0.5,
            "source_speed_p95_deg_s": 1.0,
            "n_microsaccade_events": 0,
        },
        {
            "source_row": 2,
            "session": "s",
            "trace": trace * 0.9,
            "observed_rms_deg": 0.07,
            "path_length_deg": 0.45,
            "lag1_autocorr": 0.2,
            "source_max_radius_deg": 0.1,
            "source_path_length_deg": 0.45,
            "source_speed_p95_deg_s": 1.0,
            "n_microsaccade_events": 0,
        },
        {
            "source_row": 4,
            "session": "s",
            "trace": np.column_stack([trace[:, 1], -trace[:, 0]]).astype(np.float32),
            "observed_rms_deg": 0.05,
            "path_length_deg": 0.35,
            "lag1_autocorr": 0.1,
            "source_max_radius_deg": 0.1,
            "source_path_length_deg": 0.35,
            "source_speed_p95_deg_s": 1.0,
            "n_microsaccade_events": 0,
        },
    ]
    work = pd.DataFrame(
        {
            "source_row": [1, 3],
            "image_edge_axis_deg": [0.0, 0.0],
        }
    )
    args = SimpleNamespace(
        trajectory_prior_mode="leave_one_out",
        loo_exclude_trace_rmse_deg=1e-6,
        max_trace_source_rms_deg=None,
        max_trace_source_radius_deg=None,
        max_trace_source_path_length_deg=None,
        max_rendered_trace_path_length_deg=None,
        max_source_trace_path_length_deg=None,
        max_trace_source_speed_p95_deg_s=None,
        max_trace_source_microsaccade_events=None,
        axis_source_column="image_edge_axis_deg",
        axis_template_mode="same_dominant_projection",
        axis_match_policy="strict",
        max_rms_deg=0.2,
        seed=3,
    )
    duplicate_observation, _meta = _trace_from_item(
        family="axis_edge_parallel",
        item={**trace_bank[1], "image_edge_axis_deg": 0.0},
        scale=0.5,
        rng=np.random.default_rng(0),
        max_rms_deg=0.2,
        axis_source_column="image_edge_axis_deg",
        axis_template_mode="same_dominant_projection",
        axis_match_policy="strict",
    )
    traces, specs, true_idx, meta = _axis_per_candidate_prior_trajectories(
        current_source_row=1,
        observation_trace=duplicate_observation + np.float32(1e-8),
        prior_family="axis_edge_parallel",
        prior_scale=0.5,
        n_prior_trajectories=1,
        trace_bank=trace_bank,
        candidate_indices=[0, 1],
        candidate_ids=["true", "other"],
        work=work,
        args=args,
        rng=np.random.default_rng(0),
    )
    assert true_idx == -1
    assert meta["excluded_exact_trace_hash"] == 0
    assert meta["excluded_near_duplicate_rmse"] == 1
    assert len(traces) == len(specs) == 2
    assert specs[0][0]["source_row"] == 4
    assert specs[1][0]["source_row"] == 4


def test_per_candidate_axis_families_can_share_source_samples() -> None:
    trace = _source_trace()
    trace_bank = [
        {
            "source_row": idx,
            "session": "s",
            "trace": trace * scale,
            "observed_rms_deg": 0.08 * scale,
            "path_length_deg": 0.5 * scale,
            "lag1_autocorr": 0.1,
            "source_max_radius_deg": 0.1,
            "source_path_length_deg": 0.5 * scale,
            "source_speed_p95_deg_s": 1.0,
            "n_microsaccade_events": 0,
        }
        for idx, scale in [(1, 1.0), (2, 0.9), (3, 0.8), (4, 0.7), (5, 0.6), (6, 0.5)]
    ]
    work = pd.DataFrame(
        {
            "source_row": [1, 3],
            "image_edge_axis_deg": [0.0, 90.0],
        }
    )
    args = SimpleNamespace(
        trajectory_prior_mode="leave_one_out",
        loo_exclude_trace_rmse_deg=1e-9,
        max_trace_source_rms_deg=None,
        max_trace_source_radius_deg=None,
        max_trace_source_path_length_deg=None,
        max_rendered_trace_path_length_deg=None,
        max_source_trace_path_length_deg=None,
        max_trace_source_speed_p95_deg_s=None,
        max_trace_source_microsaccade_events=None,
        axis_source_column="image_edge_axis_deg",
        axis_template_mode="same_dominant_projection",
        axis_match_policy="strict",
        max_rms_deg=0.2,
        seed=3,
    )
    candidate_meta_rows = _axis_candidate_meta_rows(
        candidate_indices=[0, 1],
        candidate_ids=["true", "other"],
        work=work,
        args=args,
    )
    sampled = _axis_shared_sampled_source_indices(
        current_source_row=1,
        observation_trace=trace,
        prior_families=["axis_edge_parallel", "axis_edge_orthogonal"],
        prior_scale=0.5,
        n_prior_trajectories=2,
        trace_bank=trace_bank,
        candidate_meta_rows=candidate_meta_rows,
        args=args,
        rng=np.random.default_rng(4),
    )
    _par_traces, par_specs, _true_idx, par_meta = _axis_per_candidate_prior_trajectories(
        current_source_row=1,
        observation_trace=trace,
        prior_family="axis_edge_parallel",
        prior_scale=0.5,
        n_prior_trajectories=2,
        trace_bank=trace_bank,
        candidate_indices=[0, 1],
        candidate_ids=["true", "other"],
        work=work,
        args=args,
        rng=np.random.default_rng(0),
        sampled_bank_indices=sampled,
    )
    _orth_traces, orth_specs, _true_idx, orth_meta = _axis_per_candidate_prior_trajectories(
        current_source_row=1,
        observation_trace=trace,
        prior_family="axis_edge_orthogonal",
        prior_scale=0.5,
        n_prior_trajectories=2,
        trace_bank=trace_bank,
        candidate_indices=[0, 1],
        candidate_ids=["true", "other"],
        work=work,
        args=args,
        rng=np.random.default_rng(99),
        sampled_bank_indices=sampled,
    )
    assert par_meta["axis_shared_source_catalog"] is True
    assert orth_meta["axis_shared_source_catalog"] is True
    for candidate_index in range(2):
        par_source_rows = [spec["source_row"] for spec in par_specs[candidate_index]]
        orth_source_rows = [spec["source_row"] for spec in orth_specs[candidate_index]]
        assert par_source_rows == orth_source_rows
        assert [spec["sample_index"] for spec in par_specs[candidate_index]] == [0, 1]
        assert [spec["sample_index"] for spec in orth_specs[candidate_index]] == [0, 1]
        assert {spec["axis_relation"] for spec in par_specs[candidate_index]} == {"parallel"}
        assert {spec["axis_relation"] for spec in orth_specs[candidate_index]} == {"orthogonal"}


def test_per_candidate_duplicate_count_is_within_candidate_only() -> None:
    specs_by_candidate = [
        [{"trajectory_identity_id": "shared"}, {"trajectory_identity_id": "a"}],
        [{"trajectory_identity_id": "shared"}, {"trajectory_identity_id": "b"}],
    ]
    assert _nested_duplicate_trace_count(specs_by_candidate) == 0

    specs_by_candidate[1].append({"trajectory_identity_id": "b"})
    assert _nested_duplicate_trace_count(specs_by_candidate) == 1


def test_invalid_relation_and_template_mode_fail_loudly() -> None:
    try:
        axis_conditioned_trace(_source_trace(), axis_deg=0.0, relation="diagonal")
    except ValueError as exc:
        assert "relation" in str(exc)
    else:
        raise AssertionError("Expected invalid relation to fail")

    try:
        axis_conditioned_trace(_source_trace(), axis_deg=0.0, relation="parallel", template_mode="unknown")
    except ValueError as exc:
        assert "template_mode" in str(exc)
    else:
        raise AssertionError("Expected invalid template mode to fail")
