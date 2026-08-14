#!/usr/bin/env python3
"""Assemble complete rounds from the provisional 100-image x 577-trace cache."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from declan.fig4_active_sensing.run_rr100_clean_history_provisional_cache import clean_valid
from declan.fig4_active_sensing.run_rr100_corrected_production_cache import (
    N_UNITS,
    SUMMARY_ARRAYS,
    baseline_path,
    baseline_valid,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rounds", required=True, help="Inclusive range such as 0-22")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def open_memmap_atomic(path: Path, shape: tuple[int, ...]) -> tuple[np.memmap, Path]:
    temporary = path.with_name(path.name + ".partial.npy")
    temporary.unlink(missing_ok=True)
    return np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape), temporary


def parse_range(text: str) -> list[int]:
    first, last = map(int, text.split("-", 1))
    return list(range(first, last + 1))


def main() -> None:
    args = parse_args()
    cache = args.cache_dir.resolve()
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    rounds = parse_range(args.rounds)
    warning = json.loads((cache / "CACHE_WARNING.json").read_text())
    policy = str(warning["policy_sha256"])
    bad = set(map(int, warning["must_exclude_trace_indices"]))
    identity = json.loads((cache / "request_identity.json").read_text())
    request_sha = str(identity["request_sha256"])
    schedule = pd.read_csv(cache / "balanced_round_schedule.csv")
    images = np.sort(schedule.image_index.unique().astype(int))

    expected_by_block: dict[tuple[int, int], np.ndarray] = {}
    for round_index in rounds:
        rows = schedule[schedule.round_index.eq(round_index)]
        if len(rows) != 1000:
            raise ValueError(f"Round {round_index} missing from canonical schedule")
        for image_index, block in rows.groupby("image_index", sort=True):
            expected = block.loc[
                ~block.trace_index.astype(int).isin(bad)
            ].sort_values("within_block").trace_index.to_numpy(np.int64)
            path = cache / "moving" / f"round_{round_index:03d}" / f"image_{int(image_index):03d}.npz"
            if not path.exists() or not clean_valid(path, expected, policy, request_sha):
                raise RuntimeError(f"Round {round_index} is not clean-history complete: {path}")
            expected_by_block[(round_index, int(image_index))] = expected

    n_conditions = len(rounds) * 577
    out.mkdir(parents=True)
    maps = {
        name: open_memmap_atomic(out / f"moving_{name}.npy", (n_conditions, N_UNITS))
        for name in SUMMARY_ARRAYS
    }
    condition_rows: list[dict[str, int]] = []
    offset = 0
    for round_index in rounds:
        seen: list[int] = []
        for image_index in images:
            expected = expected_by_block[(round_index, int(image_index))]
            path = cache / "moving" / f"round_{round_index:03d}" / f"image_{int(image_index):03d}.npz"
            with np.load(path, allow_pickle=False) as data:
                n = len(expected)
                for name, (array, _) in maps.items():
                    array[offset : offset + n] = data[name]
                for local, trace_index in enumerate(expected):
                    condition_rows.append(
                        {
                            "matrix_row_index": offset + local,
                            "round_index": round_index,
                            "half_index": int(data["half_index"].item()),
                            "image_index": int(image_index),
                            "trace_index": int(trace_index),
                        }
                    )
                seen.extend(map(int, expected))
                offset += n
        if len(seen) != 577 or len(set(seen)) != 577:
            raise AssertionError(f"Round {round_index} is not balanced over the 577 clean traces")
    if offset != n_conditions:
        raise AssertionError((offset, n_conditions))
    for array, temporary in maps.values():
        array.flush()
        del array
        os.replace(temporary, Path(str(temporary).removesuffix(".partial.npy")))
    conditions = pd.DataFrame(condition_rows)
    conditions.to_csv(out / "condition_index.csv", index=False)

    baseline = {name: np.empty((len(images), N_UNITS), np.float32) for name in SUMMARY_ARRAYS[:5]}
    for ordinal, image_index in enumerate(images):
        path = baseline_path(cache, int(image_index))
        if not baseline_valid(path, image_index=int(image_index), request_sha256=request_sha):
            raise RuntimeError(f"Invalid baseline: {path}")
        with np.load(path, allow_pickle=False) as data:
            for name in baseline:
                baseline[name][ordinal] = data[name]
    np.savez_compressed(out / "stabilized_by_image_sufficient_statistics.npz", image_index=images, **baseline)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "provisional_clean_history_balanced_snapshot_complete",
        "warning": "100x577 clean-history subset; not final 100x1000 production cache",
        "history_qc_policy_sha256": policy,
        "complete_rounds_assembled": rounds,
        "n_complete_rounds": len(rounds),
        "n_conditions": n_conditions,
        "n_images": len(images),
        "n_unique_traces": 577,
        "conditions_per_round": 577,
        "source_cache": str(cache),
        "source_cache_warning": str((cache / "CACHE_WARNING.json").resolve()),
        "outputs": {
            "condition_index": str((out / "condition_index.csv").resolve()),
            "stabilized": str((out / "stabilized_by_image_sufficient_statistics.npz").resolve()),
        },
    }
    atomic_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
