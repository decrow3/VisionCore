"""Utilities for sharded BackImage response caches.

The existing BackImage runners each write analysis-specific caches.  This
module provides a small shared substrate for larger cache banks: stable hashes,
source-row sharding, atomic file writes, and a simple trace-catalog format.
It is deliberately free of GPU/model imports so it can be used by tests and
posthoc tooling without initializing the twin.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


TRACE_CATALOG_REQUIRED_COLUMNS = ("source_row", "trace_id", "family", "scale_id")
TRACE_CATALOG_OPTIONAL_COLUMNS = (
    "trace_key",
    "scale",
    "seed",
    "sample_index",
    "trace_source_row",
    "trace_source_session",
    "pairing_mode",
    "axis_relation",
    "axis_deg",
    "requested_rms_deg",
    "effective_rms_deg",
    "path_length_deg",
    "trace_hash",
)


@dataclass(frozen=True)
class CacheShard:
    """A deterministic shard of source rows."""

    shard_index: int
    n_shards: int
    source_rows: tuple[int, ...]


def jsonable(value: Any) -> Any:
    """Convert common numpy/path objects into stable JSON primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=True)


def stable_hash(payload: Any, *, n_hex: int = 16) -> str:
    text = stable_json_dumps(payload).encode("utf-8")
    return hashlib.sha1(text).hexdigest()[: int(n_hex)]


def array_hash(array: np.ndarray, *, n_hex: int = 16) -> str:
    arr = np.asarray(array)
    digest = hashlib.sha1()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(np.ascontiguousarray(arr).view(np.uint8))
    return digest.hexdigest()[: int(n_hex)]


def trace_catalog_id(row: dict[str, Any]) -> str:
    """Build a stable trace id from the catalog-defining fields in a row."""
    keep = {
        key: row.get(key)
        for key in (
            "source_row",
            "family",
            "scale_id",
            "scale",
            "seed",
            "sample_index",
            "trace_source_row",
            "axis_relation",
            "axis_deg",
            "trace_hash",
        )
        if key in row
    }
    return stable_hash(keep, n_hex=20)


def source_row_shard(source_row: int, n_shards: int) -> int:
    if int(n_shards) <= 0:
        raise ValueError("n_shards must be positive")
    digest = hashlib.sha1(str(int(source_row)).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % int(n_shards)


def make_source_shard(source_rows: Iterable[int], *, shard_index: int, n_shards: int) -> CacheShard:
    if int(shard_index) < 0 or int(shard_index) >= int(n_shards):
        raise ValueError(f"shard_index={shard_index} must be in [0, {int(n_shards)})")
    selected = tuple(
        int(row)
        for row in sorted({int(v) for v in source_rows})
        if source_row_shard(int(row), int(n_shards)) == int(shard_index)
    )
    return CacheShard(shard_index=int(shard_index), n_shards=int(n_shards), source_rows=selected)


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: Iterable[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_fieldnames = list(fieldnames) if fieldnames is not None else []
    for row in rows:
        for key in row:
            if key not in ordered_fieldnames:
                ordered_fieldnames.append(key)
    if not ordered_fieldnames:
        atomic_write_text(path, "")
        return
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=ordered_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def atomic_savez(path: Path, arrays: dict[str, np.ndarray], *, compressed: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".npz"
    with tempfile.NamedTemporaryFile(suffix=suffix, dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        if compressed:
            np.savez_compressed(tmp_path, **arrays)
        else:
            np.savez(tmp_path, **arrays)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def done_marker_path(out_dir: Path, prefix: str, shard: CacheShard) -> Path:
    return Path(out_dir) / f"{prefix}_shard{shard.shard_index:05d}of{shard.n_shards:05d}.done.json"


def shard_stem(prefix: str, shard: CacheShard) -> str:
    return f"{prefix}_shard{shard.shard_index:05d}of{shard.n_shards:05d}"


def load_trace_catalog(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(TRACE_CATALOG_REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Trace catalog is missing required columns: {missing}")
    out = frame.copy()
    out["source_row"] = out["source_row"].astype(int)
    if "trace_key" not in out.columns:
        out["trace_key"] = out["trace_id"].astype(str)
    if "sample_index" not in out.columns:
        out["sample_index"] = 0
    if "seed" not in out.columns:
        out["seed"] = 0
    return out


def validate_trace_catalog(frame: pd.DataFrame, trace_arrays: dict[str, np.ndarray] | None = None) -> None:
    missing = sorted(set(TRACE_CATALOG_REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Trace catalog is missing required columns: {missing}")
    if frame["trace_id"].astype(str).duplicated().any():
        dup = frame.loc[frame["trace_id"].astype(str).duplicated(), "trace_id"].astype(str).head(5).to_list()
        raise ValueError(f"Trace catalog has duplicate trace_id values: {dup}")
    if trace_arrays is not None:
        key_series = frame.get("trace_key", frame["trace_id"]).astype(str)
        if "family" in frame.columns:
            needs_array = frame["family"].astype(str) != "static"
        else:
            needs_array = np.ones(frame.shape[0], dtype=bool)
        keys = {str(key) for key in key_series[needs_array] if str(key)}
        missing_keys = sorted(keys.difference(trace_arrays))
        if missing_keys:
            preview = ", ".join(missing_keys[:8])
            suffix = "..." if len(missing_keys) > 8 else ""
            raise ValueError(f"Trace NPZ is missing trace arrays referenced by catalog: {preview}{suffix}")


def load_trace_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key], dtype=np.float32) for key in loaded.files}


def write_trace_catalog(path: Path, rows: list[dict[str, Any]], trace_arrays: dict[str, np.ndarray], *, trace_npz_path: Path | None = None) -> None:
    """Write paired CSV/NPZ trace catalog files atomically."""
    trace_npz_path = Path(trace_npz_path) if trace_npz_path is not None else Path(path).with_suffix(".npz")
    atomic_savez(trace_npz_path, trace_arrays)
    atomic_write_csv(Path(path), rows)
