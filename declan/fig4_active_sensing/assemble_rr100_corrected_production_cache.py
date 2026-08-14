#!/usr/bin/env python3
"""Assemble balanced partial or complete corrected RR100 response-cache rounds.

Only fully completed balanced rounds are promoted into analysis arrays by
default.  In production, each such round contains 1,000 movies, every image
ten times, and every trace once.  Raw per-frame timecourses remain in atomic
shards; the assembler materializes compact movie x unit sufficient statistics
and an immutable condition-index table.
"""
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

from declan.fig4_active_sensing.run_rr100_corrected_production_cache import (
    N_SCORE,
    N_UNITS,
    SUMMARY_ARRAYS,
    baseline_path,
    baseline_valid,
    moving_path,
    moving_valid,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--rounds", default="", help="Optional comma/range list, e.g. 0-4,8")
    parser.add_argument("--require-complete-half", action="store_true")
    parser.add_argument("--include-timecourses", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def parse_rounds(text: str) -> set[int] | None:
    if not str(text).strip():
        return None
    out: set[int] = set()
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, stop_text = token.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            out.update(range(start, stop + 1))
        else:
            out.add(int(token))
    return out


def open_memmap_atomic(path: Path, shape: tuple[int, ...], dtype: np.dtype) -> tuple[np.memmap, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial.npy")
    if temporary.exists():
        temporary.unlink()
    return np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=shape), temporary


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir.resolve()
    manifest_path = cache_dir / "manifest.json"
    identity_path = cache_dir / "request_identity.json"
    schedule_path = cache_dir / "balanced_round_schedule.csv"
    for path in (manifest_path, identity_path, schedule_path):
        if not path.exists():
            raise FileNotFoundError(path)
    cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    request_sha256 = str(identity["request_sha256"])
    request = identity["request"]
    schedule = pd.read_csv(schedule_path)
    image_indices = np.sort(schedule.image_index.unique().astype(int))
    all_rounds = np.sort(schedule.round_index.unique().astype(int))
    block_size = int(request["trace_block_size"])
    requested = parse_rounds(args.rounds)

    complete_rounds: list[int] = []
    invalid_shards: list[str] = []
    for round_index in all_rounds:
        if requested is not None and int(round_index) not in requested:
            continue
        valid = True
        round_rows = schedule[schedule.round_index.eq(round_index)]
        for image_index in image_indices:
            expected_traces = (
                round_rows[round_rows.image_index.eq(image_index)]
                .sort_values("within_block")
                .trace_index.to_numpy(np.int64)
            )
            path = moving_path(cache_dir, int(round_index), int(image_index))
            if not path.exists() or not moving_valid(
                path,
                image_index=int(image_index),
                round_index=int(round_index),
                trace_indices=expected_traces,
                request_sha256=request_sha256,
            ):
                valid = False
                if path.exists():
                    invalid_shards.append(str(path))
        if valid:
            complete_rounds.append(int(round_index))
    if not complete_rounds:
        raise RuntimeError("No fully complete balanced round is available for analysis")

    n_rounds_total = len(all_rounds)
    half_size = n_rounds_total // 2
    complete_half0 = set(range(half_size)).issubset(complete_rounds)
    complete_half1 = set(range(half_size, n_rounds_total)).issubset(complete_rounds)
    if args.require_complete_half and not (complete_half0 or complete_half1):
        raise RuntimeError("Neither independently analyzable 50-round half is complete")

    n_conditions = len(complete_rounds) * len(image_indices) * block_size
    suffix = f"rounds_{complete_rounds[0]:03d}_{complete_rounds[-1]:03d}_n{len(complete_rounds):03d}"
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else cache_dir / "assembled" / suffix
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "manifest.json").exists():
        raise FileExistsError(f"Assembly output already exists: {out_dir}")

    metric_maps: dict[str, tuple[np.memmap, Path]] = {
        name: open_memmap_atomic(out_dir / f"moving_{name}.npy", (n_conditions, N_UNITS), np.float32)
        for name in SUMMARY_ARRAYS
    }
    timecourse_maps: dict[str, tuple[np.memmap, Path]] = {}
    if args.include_timecourses:
        timecourse_maps = {
            "rate_timecourse_hz": open_memmap_atomic(
                out_dir / "moving_rate_timecourse_hz.npy", (n_conditions, N_SCORE, N_UNITS), np.float32
            ),
            "instantaneous_ssi_bits_per_spike": open_memmap_atomic(
                out_dir / "moving_instantaneous_ssi_bits_per_spike.npy",
                (n_conditions, N_SCORE, N_UNITS),
                np.float32,
            ),
        }

    condition_rows: list[dict[str, int]] = []
    offset = 0
    for round_index in complete_rounds:
        round_rows = schedule[schedule.round_index.eq(round_index)]
        for image_index in image_indices:
            expected = round_rows[round_rows.image_index.eq(image_index)].sort_values("within_block")
            path = moving_path(cache_dir, round_index, int(image_index))
            with np.load(path, allow_pickle=False) as data:
                stop = offset + block_size
                for name, (array, _) in metric_maps.items():
                    array[offset:stop] = data[name]
                for name, (array, _) in timecourse_maps.items():
                    array[offset:stop] = data[name]
                for local, trace_index in enumerate(data["trace_index"].astype(int)):
                    condition_rows.append(
                        {
                            "matrix_row_index": int(offset + local),
                            "round_index": int(round_index),
                            "half_index": int(data["half_index"].item()),
                            "image_index": int(image_index),
                            "trace_index": int(trace_index),
                        }
                    )
                offset = stop
    if offset != n_conditions:
        raise AssertionError(f"Assembled {offset} conditions, expected {n_conditions}")

    for _, (array, temporary) in {**metric_maps, **timecourse_maps}.items():
        array.flush()
        del array
        destination = Path(str(temporary).removesuffix(".partial.npy"))
        os.replace(temporary, destination)
    pd.DataFrame(condition_rows).to_csv(out_dir / "condition_index.csv", index=False)

    baseline_metric = {
        name: np.empty((len(image_indices), N_UNITS), dtype=np.float32)
        for name in SUMMARY_ARRAYS[:5]
    }
    baseline_rate = (
        np.empty((len(image_indices), N_SCORE, N_UNITS), dtype=np.float32)
        if args.include_timecourses
        else None
    )
    baseline_ssi = np.empty_like(baseline_rate) if baseline_rate is not None else None
    for ordinal, image_index in enumerate(image_indices):
        path = baseline_path(cache_dir, int(image_index))
        if not baseline_valid(path, image_index=int(image_index), request_sha256=request_sha256):
            raise RuntimeError(f"Missing or invalid baseline for image {image_index}: {path}")
        with np.load(path, allow_pickle=False) as data:
            for name in baseline_metric:
                baseline_metric[name][ordinal] = data[name]
            if baseline_rate is not None and baseline_ssi is not None:
                baseline_rate[ordinal] = data["rate_timecourse_hz"]
                baseline_ssi[ordinal] = data["instantaneous_ssi_bits_per_spike"]
    np.savez_compressed(
        out_dir / "stabilized_by_image_sufficient_statistics.npz",
        image_index=image_indices,
        **baseline_metric,
        **(
            {
                "rate_timecourse_hz": baseline_rate,
                "instantaneous_ssi_bits_per_spike": baseline_ssi,
            }
            if baseline_rate is not None
            else {}
        ),
    )

    output_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "balanced_partial_assembly_complete",
        "source_cache": str(cache_dir),
        "request_sha256": request_sha256,
        "complete_rounds_assembled": complete_rounds,
        "n_complete_rounds": len(complete_rounds),
        "n_conditions": n_conditions,
        "n_images": len(image_indices),
        "n_unique_traces": int(pd.DataFrame(condition_rows).trace_index.nunique()),
        "complete_half0": bool(complete_half0),
        "complete_half1": bool(complete_half1),
        "analysis_scope": (
            "Every assembled round is balanced across all images and traces. "
            "A complete half supports the predeclared independently analyzable half-bank result; fewer rounds are interim convergence evidence."
        ),
        "timecourses_consolidated": bool(args.include_timecourses),
        "invalid_existing_shards_excluded": invalid_shards,
        "source_progress_manifest": cache_manifest,
        "outputs": {
            "condition_index": str((out_dir / "condition_index.csv").resolve()),
            "stabilized": str((out_dir / "stabilized_by_image_sufficient_statistics.npz").resolve()),
        },
    }
    atomic_json(out_dir / "manifest.json", output_manifest)
    print(json.dumps(output_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
