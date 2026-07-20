#!/usr/bin/env python3
"""Merge image-sharded BackImage real-trace SSI matrix outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd


MATRIX_FILES = (
    "ssi_matrix.npy",
    "expected_spikes_matrix.npy",
    "mean_rate_matrix.npy",
    "population_ssi.npy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("shard_dirs", type=Path, nargs="+")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_same_table(first: pd.DataFrame, other: pd.DataFrame, *, name: str) -> None:
    left = first.reset_index(drop=True)
    right = other.reset_index(drop=True)
    if left.shape != right.shape or list(left.columns) != list(right.columns):
        raise ValueError(f"{name} differs across shards: shape/columns mismatch.")
    if not left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str)):
        raise ValueError(f"{name} differs across shards.")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not bool(args.force):
        raise FileExistsError(f"{out_dir} already exists and is not empty. Pass --force.")
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_dirs = [Path(path) for path in args.shard_dirs]
    summaries = [load_json(path / "summary.json") for path in shard_dirs]
    image_table = pd.read_csv(shard_dirs[0] / "image_feature_table.csv")
    trace_table = pd.read_csv(shard_dirs[0] / "trace_feature_table.csv")
    unit_table = pd.read_csv(shard_dirs[0] / "unit_feature_table.csv")
    trace_xy = np.load(shard_dirs[0] / "trace_xy.npy")
    for shard_dir in shard_dirs[1:]:
        assert_same_table(image_table, pd.read_csv(shard_dir / "image_feature_table.csv"), name="image_feature_table.csv")
        assert_same_table(trace_table, pd.read_csv(shard_dir / "trace_feature_table.csv"), name="trace_feature_table.csv")
        assert_same_table(unit_table, pd.read_csv(shard_dir / "unit_feature_table.csv"), name="unit_feature_table.csv")
        other_xy = np.load(shard_dir / "trace_xy.npy")
        if trace_xy.shape != other_xy.shape or not np.array_equal(trace_xy, other_xy):
            raise ValueError("trace_xy.npy differs across shards.")

    n_images = int(image_table.shape[0])
    n_traces = int(trace_table.shape[0])
    n_units = int(unit_table.shape[0])
    n_movies = n_images * n_traces

    movie_parts: list[pd.DataFrame] = []
    arrays: dict[str, np.ndarray] = {}
    for name in MATRIX_FILES:
        sample = np.load(shard_dirs[0] / name)
        if sample.ndim == 1:
            arrays[name] = np.full((n_movies,), np.nan, dtype=sample.dtype)
        else:
            arrays[name] = np.full((n_movies, n_units), np.nan, dtype=sample.dtype)

    for shard_dir in shard_dirs:
        movies = pd.read_csv(shard_dir / "movie_feature_table.csv")
        movie_index = movies["movie_index"].astype(int).to_numpy()
        if movie_index.min(initial=0) < 0 or movie_index.max(initial=-1) >= n_movies:
            raise ValueError(f"{shard_dir} has movie_index outside expected 0-{n_movies - 1}.")
        if movie_index.size != np.unique(movie_index).size:
            raise ValueError(f"{shard_dir} has duplicate movie_index values.")
        matrix_row = movies["matrix_row_index"].astype(int).to_numpy() if "matrix_row_index" in movies.columns else np.arange(movies.shape[0])
        for name in MATRIX_FILES:
            values = np.load(shard_dir / name)
            if values.shape[0] != movies.shape[0]:
                raise ValueError(f"{name} rows do not match movie table in {shard_dir}.")
            arrays[name][movie_index] = values[matrix_row]
        movie_parts.append(movies)

    merged_movie = pd.concat(movie_parts, ignore_index=True).sort_values("movie_index", kind="mergesort").reset_index(drop=True)
    if merged_movie.shape[0] != n_movies:
        raise ValueError(f"Merged movie table has {merged_movie.shape[0]} rows; expected {n_movies}.")
    if merged_movie["movie_index"].astype(int).nunique() != n_movies:
        raise ValueError("Merged movie table does not cover every movie_index exactly once.")
    for name, values in arrays.items():
        if np.isnan(values).any():
            raise ValueError(f"{name} has unfilled rows after merge.")
        np.save(out_dir / name, values)

    image_table.to_csv(out_dir / "image_feature_table.csv", index=False)
    trace_table.to_csv(out_dir / "trace_feature_table.csv", index=False)
    unit_table.to_csv(out_dir / "unit_feature_table.csv", index=False)
    merged_movie.to_csv(out_dir / "movie_feature_table.csv", index=False)
    np.save(out_dir / "trace_xy.npy", trace_xy)
    if (shard_dirs[0] / "trace_bank_metric_summary.csv").exists():
        pd.read_csv(shard_dirs[0] / "trace_bank_metric_summary.csv").to_csv(out_dir / "trace_bank_metric_summary.csv", index=False)

    summary = {
        "analysis": "backimage_real_trace_ssi_matrix_merged_shards",
        "shard_dirs": shard_dirs,
        "n_shards": len(shard_dirs),
        "n_images": n_images,
        "n_traces": n_traces,
        "n_units": n_units,
        "n_movies": n_movies,
        "shard_summaries": summaries,
        "outputs": {name.removesuffix(".npy"): out_dir / name for name in MATRIX_FILES}
        | {
            "movie_feature_table": out_dir / "movie_feature_table.csv",
            "image_feature_table": out_dir / "image_feature_table.csv",
            "trace_feature_table": out_dir / "trace_feature_table.csv",
            "unit_feature_table": out_dir / "unit_feature_table.csv",
            "trace_xy": out_dir / "trace_xy.npy",
        },
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
