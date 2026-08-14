#!/usr/bin/env python3
"""Regression tests for the corrected production-cache interruption contract."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from declan.fig4_active_sensing.run_rr100_corrected_production_cache import (
    N_SCORE,
    N_UNITS,
    SUMMARY_ARRAYS,
    atomic_npz,
    make_balanced_schedule,
    moving_valid,
)


def connected(schedule, image_ids: np.ndarray, trace_ids: np.ndarray) -> bool:
    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = {}
    for row in schedule.itertuples(index=False):
        image = ("i", int(row.image_index))
        trace = ("t", int(row.trace_index))
        adjacency.setdefault(image, set()).add(trace)
        adjacency.setdefault(trace, set()).add(image)
    expected = {("i", int(value)) for value in image_ids} | {("t", int(value)) for value in trace_ids}
    seen: set[tuple[str, int]] = set()
    stack = [next(iter(expected))]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency.get(node, set()).difference(seen))
    return seen == expected


class CorrectedProductionCacheContractTest(unittest.TestCase):
    def test_balanced_rounds_cover_cartesian_grid_and_halves_are_connected(self) -> None:
        images = np.arange(4, dtype=int)
        traces = np.arange(100, 112, dtype=int)
        schedule = make_balanced_schedule(images, traces, block_size=3)
        self.assertEqual(len(schedule), 48)
        self.assertFalse(schedule[["image_index", "trace_index"]].duplicated().any())
        self.assertEqual(schedule[["image_index", "trace_index"]].drop_duplicates().shape[0], 48)
        for round_index, round_rows in schedule.groupby("round_index"):
            self.assertEqual(round_rows.image_index.value_counts().to_dict(), {0: 3, 1: 3, 2: 3, 3: 3})
            self.assertEqual(set(round_rows.trace_index), set(traces))
        for half_index in (0, 1):
            half = schedule[schedule.half_index.eq(half_index)]
            self.assertTrue(connected(half, images, traces))
            self.assertTrue(np.all(half.image_index.value_counts().sort_index().to_numpy() == 6))
            self.assertTrue(np.all(half.trace_index.value_counts().sort_index().to_numpy() == 2))

    def test_atomic_moving_shard_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "moving.npz"
            trace_ids = np.asarray([3, 7], dtype=np.int64)
            arrays = {
                name: np.zeros((len(trace_ids), N_UNITS), dtype=np.float32)
                for name in SUMMARY_ARRAYS
            }
            atomic_npz(
                path,
                request_sha256=np.asarray("identity"),
                round_index=np.asarray(2, dtype=np.int64),
                half_index=np.asarray(0, dtype=np.int64),
                image_index=np.asarray(5, dtype=np.int64),
                trace_index=trace_ids,
                rate_timecourse_hz=np.zeros((len(trace_ids), N_SCORE, N_UNITS), dtype=np.float32),
                instantaneous_ssi_bits_per_spike=np.zeros(
                    (len(trace_ids), N_SCORE, N_UNITS), dtype=np.float32
                ),
                **arrays,
            )
            self.assertTrue(
                moving_valid(
                    path,
                    image_index=5,
                    round_index=2,
                    trace_indices=trace_ids,
                    request_sha256="identity",
                )
            )
            self.assertFalse(
                moving_valid(
                    path,
                    image_index=5,
                    round_index=2,
                    trace_indices=trace_ids[::-1],
                    request_sha256="identity",
                )
            )


if __name__ == "__main__":
    unittest.main()
