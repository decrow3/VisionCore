from __future__ import annotations

import numpy as np
import pandas as pd

from declan.fixation_statistics_by_stimulus.assemble_backimage_response_cache_bank import _assemble_summary_arrays
from declan.fixation_statistics_by_stimulus.assemble_backimage_response_cache_bank import _load_latent_shards, _load_shards
from declan.fixation_statistics_by_stimulus.backimage_cache import (
    array_hash,
    atomic_write_csv,
    atomic_savez,
    atomic_write_json,
    done_marker_path,
    load_trace_catalog,
    load_trace_npz,
    make_source_shard,
    source_row_shard,
    stable_hash,
    validate_trace_catalog,
    write_trace_catalog,
)
from declan.fixation_statistics_by_stimulus.run_backimage_response_cache_bank import (
    _check_trace_batch_equivalence,
    _completion_marker_outputs_exist,
    _score_trace_response_map,
    _summarize,
    _trace_cache_key,
)


class _FakeScorer:
    def __init__(self, *, batch_offset: float = 0.0):
        self.calls: list[tuple[int, int]] = []
        self.batch_offset = float(batch_offset)

    def responses(self, patch, traces, *, trace_batch_size: int = 1):
        self.calls.append((len(traces), int(trace_batch_size)))
        out = []
        for trace in traces:
            value = float(np.sum(trace)) + (self.batch_offset if int(trace_batch_size) > 1 else 0.0)
            out.append(np.full((trace.shape[0], 2), value, dtype=np.float32))
        return out


def _write_shard_marker(tmp_path, shard_token: str, *, request_hash: str = "request-a") -> None:
    atomic_write_json(
        tmp_path / f"response_cache_bank_{shard_token}.done.json",
        {"status": "complete", "request_hash": request_hash},
    )


def test_stable_hash_and_array_hash_are_content_based() -> None:
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})

    arr = np.arange(6, dtype=np.float32).reshape(3, 2)
    same = np.arange(6, dtype=np.float32).reshape(3, 2)
    changed = arr.copy()
    changed[0, 0] = -1.0

    assert array_hash(arr) == array_hash(same)
    assert array_hash(arr) != array_hash(changed)


def test_source_row_shards_are_deterministic_and_complete() -> None:
    rows = list(range(50))
    shards = [make_source_shard(rows, shard_index=i, n_shards=7) for i in range(7)]
    covered = sorted(row for shard in shards for row in shard.source_rows)

    assert covered == rows
    for row in rows:
        assert sum(row in shard.source_rows for shard in shards) == 1
        assert source_row_shard(row, 7) == source_row_shard(row, 7)


def test_trace_catalog_round_trip_allows_static_without_trace_array(tmp_path) -> None:
    trace = np.zeros((4, 2), dtype=np.float32)
    rows = [
        {
            "source_row": 10,
            "trace_id": "static:10",
            "trace_key": "",
            "family": "static",
            "scale_id": "static",
        },
        {
            "source_row": 10,
            "trace_id": "trace-a",
            "trace_key": "trace-a",
            "family": "empirical",
            "scale_id": "rel_1x",
        },
    ]

    csv_path = tmp_path / "catalog.csv"
    write_trace_catalog(csv_path, rows, {"trace-a": trace})
    catalog = load_trace_catalog(csv_path)
    arrays = load_trace_npz(tmp_path / "catalog.npz")

    validate_trace_catalog(catalog, arrays)
    assert list(catalog["trace_id"]) == ["static:10", "trace-a"]
    assert np.array_equal(arrays["trace-a"], trace)


def test_atomic_savez_replaces_file(tmp_path) -> None:
    path = tmp_path / "arrays.npz"
    atomic_savez(path, {"x": np.ones((2, 2), dtype=np.float32)})
    atomic_savez(path, {"x": np.zeros((1, 3), dtype=np.float32)})

    with np.load(path) as loaded:
        assert loaded["x"].shape == (1, 3)
        assert float(np.sum(loaded["x"])) == 0.0


def test_atomic_write_csv_can_write_empty_schema(tmp_path) -> None:
    path = tmp_path / "empty.csv"
    atomic_write_csv(path, [], fieldnames=["source_row", "family"])

    frame = pd.read_csv(path)

    assert list(frame.columns) == ["source_row", "family"]
    assert frame.empty


def test_summarize_writes_mean_delta_and_dct_features() -> None:
    response = np.asarray([[1.0, 3.0], [3.0, 7.0], [5.0, 11.0], [7.0, 15.0]], dtype=np.float32)
    static = np.ones_like(response)
    dct_basis = np.eye(4, 2, dtype=np.float32)

    out = _summarize(
        response,
        static,
        summaries={"mean", "delta_mean", "temporal_dct", "temporal_dct_delta"},
        dct_basis=dct_basis,
        temporal_basis=None,
    )

    assert set(out) == {"mean", "delta_mean", "temporal_dct", "temporal_dct_delta"}
    assert np.allclose(out["mean"], [4.0, 9.0])
    assert np.allclose(out["delta_mean"], [3.0, 8.0])
    assert out["temporal_dct"].shape == (4,)
    assert out["temporal_dct_delta"].shape == (4,)


def test_trace_response_map_scores_unique_trace_contents_once() -> None:
    trace_a = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    trace_b = np.asarray([[0.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    duplicate_a = trace_a.copy()
    trace_by_key = {
        _trace_cache_key(trace_a): trace_a,
        _trace_cache_key(trace_b): trace_b,
    }

    assert _trace_cache_key(trace_a) == _trace_cache_key(duplicate_a)

    scorer = _FakeScorer()
    out = _score_trace_response_map(
        scorer,
        np.zeros((4, 4), dtype=np.float32),
        trace_by_key,
        trace_batch_size=8,
        n_timepoints=2,
        check_equivalence=False,
        equivalence_atol=1e-5,
    )

    assert scorer.calls == [(2, 8)]
    assert set(out) == set(trace_by_key)
    assert np.allclose(out[_trace_cache_key(trace_a)], 1.0)
    assert np.allclose(out[_trace_cache_key(trace_b)], 2.0)


def test_trace_batch_equivalence_rejects_batch_dependent_responses() -> None:
    traces = [np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)]
    scorer = _FakeScorer(batch_offset=1.0)

    try:
        _check_trace_batch_equivalence(
            scorer,
            np.zeros((4, 4), dtype=np.float32),
            traces,
            trace_batch_size=2,
            n_timepoints=2,
            atol=1e-5,
        )
    except ValueError as exc:
        assert "Trace-batch equivalence failed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("_check_trace_batch_equivalence should reject batch-dependent responses")


def test_validate_trace_catalog_rejects_missing_trace_arrays() -> None:
    catalog = pd.DataFrame(
        [
            {
                "source_row": 1,
                "trace_id": "trace-a",
                "trace_key": "trace-a",
                "family": "empirical",
                "scale_id": "rel_1x",
            }
        ]
    )

    try:
        validate_trace_catalog(catalog, {})
    except ValueError as exc:
        assert "missing trace arrays" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("validate_trace_catalog should reject missing arrays")


def test_assemble_summary_arrays_averages_replicates_by_source_and_condition() -> None:
    rows = pd.DataFrame(
        [
            {"_bank_row": 0, "source_row": 1, "family": "empirical", "scale_id": "rel_1x"},
            {"_bank_row": 1, "source_row": 1, "family": "empirical", "scale_id": "rel_1x"},
            {"_bank_row": 2, "source_row": 2, "family": "empirical", "scale_id": "rel_1x"},
            {"_bank_row": 3, "source_row": 2, "family": "static", "scale_id": "static"},
            {"_bank_row": 4, "source_row": 1, "family": "static", "scale_id": "static"},
        ]
    )
    summaries = {
        "mean": np.asarray(
            [
                [1.0, 3.0],
                [3.0, 5.0],
                [10.0, 20.0],
                [7.0, 9.0],
                [5.0, 6.0],
            ],
            dtype=np.float32,
        )
    }

    arrays = _assemble_summary_arrays(
        rows,
        summaries,
        np.asarray([1, 2], dtype=np.int64),
        condition_cols=["family", "scale_id"],
        families=None,
        scale_ids=None,
        sample_families=set(),
        sample_condition_col="sample_index",
        allow_missing=True,
    )

    assert np.allclose(arrays["mean__empirical__rel_1x"], [[2.0, 4.0], [10.0, 20.0]])
    assert np.allclose(arrays["mean__static__static"], [[5.0, 6.0], [7.0, 9.0]])


def test_assemble_summary_arrays_can_emit_sample_specific_conditions() -> None:
    rows = pd.DataFrame(
        [
            {"_bank_row": 0, "source_row": 1, "family": "matched_unpaired_empirical", "scale_id": "rel_1x", "sample_index": 0},
            {"_bank_row": 1, "source_row": 2, "family": "matched_unpaired_empirical", "scale_id": "rel_1x", "sample_index": 0},
            {"_bank_row": 2, "source_row": 1, "family": "matched_unpaired_empirical", "scale_id": "rel_1x", "sample_index": 1},
            {"_bank_row": 3, "source_row": 2, "family": "matched_unpaired_empirical", "scale_id": "rel_1x", "sample_index": 1},
        ]
    )
    summaries = {
        "mean": np.asarray(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
            ],
            dtype=np.float32,
        )
    }

    arrays = _assemble_summary_arrays(
        rows,
        summaries,
        np.asarray([1, 2], dtype=np.int64),
        condition_cols=["family", "scale_id"],
        families=None,
        scale_ids=None,
        sample_families={"matched_unpaired_empirical"},
        sample_condition_col="sample_index",
        allow_missing=False,
    )

    assert np.allclose(arrays["mean__matched_unpaired_empirical__rel_1x"], [[2.0, 20.0], [3.0, 30.0]])
    assert np.allclose(arrays["mean__matched_unpaired_empirical_sample0__rel_1x"], [[1.0, 10.0], [2.0, 20.0]])
    assert np.allclose(arrays["mean__matched_unpaired_empirical_sample1__rel_1x"], [[3.0, 30.0], [4.0, 40.0]])


def test_completion_marker_requires_real_outputs(tmp_path) -> None:
    shard = make_source_shard([1], shard_index=0, n_shards=1)
    marker = done_marker_path(tmp_path, "response_cache_bank", shard)
    row_path = tmp_path / "rows.csv"
    summary_path = tmp_path / "summaries.npz"
    latent_path = tmp_path / "latents.npz"
    atomic_write_json(marker, {"status": "dry_run_complete", "rows": str(row_path), "summaries": str(summary_path)})

    assert not _completion_marker_outputs_exist(
        marker,
        row_path=row_path,
        summary_path=summary_path,
        latent_path=latent_path,
    )

    atomic_write_csv(row_path, [{"response_row": 0}])
    atomic_savez(summary_path, {"mean": np.zeros((1, 2), dtype=np.float32)})
    atomic_write_json(marker, {"status": "complete", "rows": str(row_path), "summaries": str(summary_path)})

    assert _completion_marker_outputs_exist(
        marker,
        row_path=row_path,
        summary_path=summary_path,
        latent_path=latent_path,
    )


def test_load_shards_rejects_mixed_summary_keys(tmp_path) -> None:
    atomic_write_csv(tmp_path / "response_cache_bank_shard00000of00002_rows.csv", [{"response_row": 0, "source_row": 1}])
    atomic_write_csv(tmp_path / "response_cache_bank_shard00001of00002_rows.csv", [{"response_row": 0, "source_row": 2}])
    _write_shard_marker(tmp_path, "shard00000of00002")
    _write_shard_marker(tmp_path, "shard00001of00002")
    atomic_savez(tmp_path / "response_cache_bank_shard00000of00002_summaries.npz", {"mean": np.zeros((1, 2), dtype=np.float32)})
    atomic_savez(
        tmp_path / "response_cache_bank_shard00001of00002_summaries.npz",
        {"delta_mean": np.zeros((1, 2), dtype=np.float32)},
    )

    try:
        _load_shards(tmp_path, "response_cache_bank_shard*_rows.csv")
    except ValueError as exc:
        assert "summary keys do not match" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("_load_shards should reject mixed summary keys")


def test_load_shards_rejects_missing_completion_marker(tmp_path) -> None:
    atomic_write_csv(tmp_path / "response_cache_bank_shard00000of00001_rows.csv", [{"response_row": 0, "source_row": 1}])
    atomic_savez(tmp_path / "response_cache_bank_shard00000of00001_summaries.npz", {"mean": np.zeros((1, 2), dtype=np.float32)})

    try:
        _load_shards(tmp_path, "response_cache_bank_shard*_rows.csv")
    except FileNotFoundError as exc:
        assert "Missing completion marker" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("_load_shards should reject row shards without completion markers")


def test_load_shards_rejects_mixed_request_hashes(tmp_path) -> None:
    atomic_write_csv(tmp_path / "response_cache_bank_shard00000of00002_rows.csv", [{"response_row": 0, "source_row": 1}])
    atomic_write_csv(tmp_path / "response_cache_bank_shard00001of00002_rows.csv", [{"response_row": 0, "source_row": 2}])
    _write_shard_marker(tmp_path, "shard00000of00002", request_hash="request-a")
    _write_shard_marker(tmp_path, "shard00001of00002", request_hash="request-b")
    atomic_savez(tmp_path / "response_cache_bank_shard00000of00002_summaries.npz", {"mean": np.zeros((1, 2), dtype=np.float32)})
    atomic_savez(tmp_path / "response_cache_bank_shard00001of00002_summaries.npz", {"mean": np.zeros((1, 2), dtype=np.float32)})

    try:
        _load_shards(tmp_path, "response_cache_bank_shard*_rows.csv")
    except ValueError as exc:
        assert "does not match earlier shards" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("_load_shards should reject mixed request hashes")


def test_load_latent_shards_rejects_duplicate_sources(tmp_path) -> None:
    atomic_savez(
        tmp_path / "response_cache_bank_shard00000of00002_latents.npz",
        {
            "source_row": np.asarray([1], dtype=np.int64),
            "gabor_local_field": np.zeros((1, 3), dtype=np.float32),
        },
    )
    atomic_savez(
        tmp_path / "response_cache_bank_shard00001of00002_latents.npz",
        {
            "source_row": np.asarray([1], dtype=np.int64),
            "gabor_local_field": np.ones((1, 3), dtype=np.float32),
        },
    )

    try:
        _load_latent_shards(tmp_path, np.asarray([1], dtype=np.int64), "response_cache_bank_shard*_latents.npz")
    except ValueError as exc:
        assert "duplicate source_row" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("_load_latent_shards should reject duplicate source rows")
