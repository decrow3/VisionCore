"""Assemble sharded BackImage response cache banks into condition arrays."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd

from declan.fixation_statistics_by_stimulus.backimage_cache import atomic_savez, atomic_write_csv, atomic_write_json


DEFAULT_CACHE_DIR = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_response_cache_bank"
)


def _parse_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _progress(message: str) -> None:
    print(f"[assemble-backimage-response-cache-bank] {message}", flush=True)


def _summary_path_for_rows(row_path: Path) -> Path:
    name = row_path.name
    if not name.endswith("_rows.csv"):
        raise ValueError(f"Unexpected row filename: {row_path}")
    return row_path.with_name(name[: -len("_rows.csv")] + "_summaries.npz")


def _marker_path_for_rows(row_path: Path) -> Path:
    name = row_path.name
    if not name.endswith("_rows.csv"):
        raise ValueError(f"Unexpected row filename: {row_path}")
    return row_path.with_name(name[: -len("_rows.csv")] + ".done.json")


def _load_complete_marker_for_rows(row_path: Path) -> dict[str, Any]:
    marker_path = _marker_path_for_rows(row_path)
    if not marker_path.exists():
        raise FileNotFoundError(f"Missing completion marker for {row_path}: {marker_path}")
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Completion marker is not valid JSON: {marker_path}") from exc
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise ValueError(f"Completion marker is not complete for {row_path}: {marker_path}")
    request_hash = payload.get("request_hash")
    if not request_hash:
        raise ValueError(f"Completion marker lacks request_hash for {row_path}: {marker_path}")
    return payload


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _load_shards(cache_dir: Path, row_glob: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    row_files = sorted(Path(cache_dir).glob(row_glob))
    if not row_files:
        raise ValueError(f"No shard row files matched {row_glob!r} in {cache_dir}")
    frames: list[pd.DataFrame] = []
    summary_parts: dict[str, list[np.ndarray]] = {}
    expected_summary_keys: set[str] | None = None
    expected_request_hash: str | None = None
    offset = 0
    for row_file in row_files:
        rows = _read_csv_or_empty(row_file)
        if rows.empty:
            continue
        marker_payload = _load_complete_marker_for_rows(row_file)
        request_hash = str(marker_payload["request_hash"])
        if expected_request_hash is None:
            expected_request_hash = request_hash
        elif request_hash != expected_request_hash:
            raise ValueError(
                f"{row_file} request_hash={request_hash} does not match earlier shards "
                f"request_hash={expected_request_hash}"
            )
        summary_path = _summary_path_for_rows(row_file)
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary NPZ for {row_file}: {summary_path}")
        with np.load(summary_path, allow_pickle=False) as loaded:
            summary_keys = set(loaded.files)
            if expected_summary_keys is None:
                expected_summary_keys = summary_keys
            elif summary_keys != expected_summary_keys:
                missing = sorted(expected_summary_keys.difference(summary_keys))
                extra = sorted(summary_keys.difference(expected_summary_keys))
                raise ValueError(
                    f"{summary_path} summary keys do not match earlier shards; "
                    f"missing={missing}, extra={extra}"
                )
            for key in loaded.files:
                arr = np.asarray(loaded[key], dtype=np.float32)
                if arr.shape[0] != rows.shape[0]:
                    raise ValueError(
                        f"{summary_path}:{key} has {arr.shape[0]} rows, but {row_file} has {rows.shape[0]}"
                    )
                summary_parts.setdefault(key, []).append(arr)
        rows = rows.copy()
        rows["_bank_row"] = np.arange(offset, offset + rows.shape[0], dtype=int)
        rows["_shard_rows_file"] = str(row_file)
        offset += int(rows.shape[0])
        frames.append(rows)
    if not frames:
        raise ValueError("Shard row files were present but empty")
    summaries = {key: np.vstack(parts).astype(np.float32, copy=False) for key, parts in summary_parts.items()}
    return pd.concat(frames, ignore_index=True), summaries


def _condition_key(row: pd.Series, condition_cols: list[str]) -> tuple[str, str]:
    values = [str(row[col]) for col in condition_cols]
    if len(values) == 1:
        return values[0], str(row.get("scale_id", ""))
    if condition_cols == ["family", "scale_id"]:
        return values[0], values[1]
    return "__".join(values), str(row.get("scale_id", ""))


def _format_condition_token(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return str(value)


def _mean_condition_by_source(
    block: pd.DataFrame,
    values: np.ndarray,
    source_order: np.ndarray,
    source_to_pos: dict[int, int],
) -> np.ndarray:
    first_idx = int(block["_bank_row"].iloc[0])
    feature_dim = int(values[first_idx].shape[0])
    out = np.full((source_order.size, feature_dim), np.nan, dtype=np.float32)
    for source_row, source_block in block.groupby("source_row", sort=False):
        pos = source_to_pos.get(int(source_row))
        if pos is None:
            continue
        idx = source_block["_bank_row"].to_numpy(dtype=int)
        out[pos] = np.nanmean(values[idx], axis=0).astype(np.float32)
    return out


def _check_condition_missing(
    out: np.ndarray,
    source_order: np.ndarray,
    *,
    condition_label: str,
    allow_missing: bool,
) -> None:
    missing = np.flatnonzero(~np.isfinite(out).any(axis=1))
    if missing.size and not allow_missing:
        preview = ", ".join(str(int(source_order[i])) for i in missing[:8])
        suffix = "..." if missing.size > 8 else ""
        raise ValueError(f"Condition {condition_label} missing source rows: {preview}{suffix}")


def _assemble_summary_arrays(
    rows: pd.DataFrame,
    summaries: dict[str, np.ndarray],
    source_order: np.ndarray,
    *,
    condition_cols: list[str],
    families: set[str] | None,
    scale_ids: set[str] | None,
    sample_families: set[str],
    sample_condition_col: str,
    allow_missing: bool,
) -> dict[str, np.ndarray]:
    source_to_pos = {int(source_row): i for i, source_row in enumerate(source_order)}
    arrays: dict[str, np.ndarray] = {}
    working = rows.copy()
    working["source_row"] = working["source_row"].astype(int)
    if families is not None:
        working = working[working["family"].astype(str).isin(families)]
    if scale_ids is not None:
        working = working[working["scale_id"].astype(str).isin(scale_ids)]
    if working.empty:
        raise ValueError("No response rows survived family/scale filters")

    for summary_name, values in summaries.items():
        for condition_values, block in working.groupby(condition_cols, dropna=False, sort=True):
            condition_values_tuple = condition_values if isinstance(condition_values, tuple) else (condition_values,)
            condition = dict(zip(condition_cols, condition_values_tuple, strict=True))
            family, scale_id = _condition_key(pd.Series(condition), condition_cols)
            out = _mean_condition_by_source(block, values, source_order, source_to_pos)
            _check_condition_missing(
                out,
                source_order,
                condition_label=f"{family}/{scale_id}/{summary_name}",
                allow_missing=allow_missing,
            )
            arrays[f"{summary_name}__{family}__{scale_id}"] = out
        if sample_families and {"family", "scale_id", sample_condition_col}.issubset(working.columns):
            sample_work = working[working["family"].astype(str).isin(sample_families)].copy()
            if sample_work.empty:
                continue
            group_cols = ["family", "scale_id", sample_condition_col]
            include_seed = "seed" in sample_work.columns and sample_work["seed"].astype(str).nunique(dropna=False) > 1
            if include_seed:
                group_cols.append("seed")
            for sample_values, block in sample_work.groupby(group_cols, dropna=False, sort=True):
                sample_tuple = sample_values if isinstance(sample_values, tuple) else (sample_values,)
                parts = dict(zip(group_cols, sample_tuple, strict=True))
                family = str(parts["family"])
                scale_id = str(parts["scale_id"])
                sample_token = _format_condition_token(parts[sample_condition_col])
                sample_family = f"{family}_sample{sample_token}"
                if include_seed:
                    sample_family = f"{sample_family}_seed{_format_condition_token(parts['seed'])}"
                out = _mean_condition_by_source(block, values, source_order, source_to_pos)
                _check_condition_missing(
                    out,
                    source_order,
                    condition_label=f"{sample_family}/{scale_id}/{summary_name}",
                    allow_missing=allow_missing,
                )
                arrays[f"{summary_name}__{sample_family}__{scale_id}"] = out
    return arrays


def _load_latent_shards(cache_dir: Path, source_order: np.ndarray, latent_glob: str) -> dict[str, np.ndarray]:
    files = sorted(Path(cache_dir).glob(latent_glob))
    if not files:
        return {}
    source_parts: list[np.ndarray] = []
    arrays_by_name: dict[str, list[np.ndarray]] = {}
    expected_latent_names: set[str] | None = None
    for path in files:
        with np.load(path, allow_pickle=False) as loaded:
            if "source_row" not in loaded.files:
                continue
            source_rows = np.asarray(loaded["source_row"], dtype=np.int64)
            if source_rows.size != np.unique(source_rows).size:
                raise ValueError(f"{path} contains duplicate source_row entries")
            latent_names = set(loaded.files).difference({"source_row", "image_index"})
            if expected_latent_names is None:
                expected_latent_names = latent_names
            elif latent_names != expected_latent_names:
                missing = sorted(expected_latent_names.difference(latent_names))
                extra = sorted(latent_names.difference(expected_latent_names))
                raise ValueError(f"{path} latent keys do not match earlier shards; missing={missing}, extra={extra}")
            source_parts.append(source_rows)
            for key in sorted(latent_names):
                arr = np.asarray(loaded[key], dtype=np.float32)
                if arr.shape[0] != source_rows.size:
                    raise ValueError(f"{path}:{key} has {arr.shape[0]} rows, but source_row has {source_rows.size}")
                arrays_by_name.setdefault(key, []).append(np.asarray(loaded[key], dtype=np.float32))
    if not source_parts:
        return {}
    all_source_rows = np.concatenate(source_parts).astype(np.int64, copy=False)
    duplicates = pd.Series(all_source_rows).duplicated(keep=False)
    if bool(duplicates.any()):
        dup = sorted({int(v) for v in all_source_rows[duplicates.to_numpy()][:8]})
        raise ValueError(f"Latent shard arrays contain duplicate source_row entries: {dup}")
    source_to_pos = {int(source_row): i for i, source_row in enumerate(all_source_rows)}
    out: dict[str, np.ndarray] = {}
    for name, parts in arrays_by_name.items():
        stacked = np.vstack(parts).astype(np.float32, copy=False)
        ordered = []
        for source_row in source_order:
            pos = source_to_pos.get(int(source_row))
            if pos is None:
                raise ValueError(f"Latent shard arrays are missing source_row={int(source_row)}")
            ordered.append(stacked[pos])
        out[name] = np.vstack(ordered).astype(np.float32, copy=False)
    out["source_row"] = source_order.astype(np.int64)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--analysis-windows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--row-glob", default="response_cache_bank_shard*_rows.csv")
    parser.add_argument("--latent-glob", default="response_cache_bank_shard*_latents.npz")
    parser.add_argument("--condition-cols", default="family,scale_id")
    parser.add_argument("--families", default="all")
    parser.add_argument("--scale-ids", default="all")
    parser.add_argument("--sample-families", default="matched_unpaired_empirical")
    parser.add_argument("--sample-condition-col", default="sample_index")
    parser.add_argument("--allow-missing", action="store_true")
    return parser


def run(args: argparse.Namespace) -> Path:
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis = pd.read_csv(args.analysis_windows)
    if "source_row" not in analysis.columns:
        raise ValueError("--analysis-windows must include source_row")
    source_order = analysis["source_row"].astype(int).to_numpy()
    if source_order.size != np.unique(source_order).size:
        raise ValueError("--analysis-windows must contain unique source_row values")
    rows, summaries = _load_shards(cache_dir, str(args.row_glob))
    condition_cols = _parse_list(args.condition_cols)
    missing_cols = sorted(set(condition_cols).difference(rows.columns))
    if missing_cols:
        raise ValueError(f"Response rows are missing condition columns: {missing_cols}")
    family_list = _parse_list(args.families)
    scale_list = _parse_list(args.scale_ids)
    families = None if not family_list or "all" in family_list else set(family_list)
    scale_ids = None if not scale_list or "all" in scale_list else set(scale_list)
    response_arrays = _assemble_summary_arrays(
        rows,
        summaries,
        source_order,
        condition_cols=condition_cols,
        families=families,
        scale_ids=scale_ids,
        sample_families=set(_parse_list(args.sample_families)),
        sample_condition_col=str(args.sample_condition_col),
        allow_missing=bool(args.allow_missing),
    )
    latent_arrays = _load_latent_shards(cache_dir, source_order, str(args.latent_glob))
    atomic_savez(out_dir / "response_summary_arrays.npz", response_arrays)
    if latent_arrays:
        atomic_savez(out_dir / "latent_feature_arrays.npz", latent_arrays)
    atomic_write_csv(out_dir / "analysis_images.csv", analysis.to_dict(orient="records"))
    atomic_write_json(
        out_dir / "run_metadata.json",
        {
            "source_cache_dir": str(cache_dir),
            "analysis_windows": str(args.analysis_windows),
            "row_glob": str(args.row_glob),
            "latent_glob": str(args.latent_glob),
            "condition_cols": condition_cols,
            "families": sorted(families) if families is not None else "all",
            "scale_ids": sorted(scale_ids) if scale_ids is not None else "all",
            "sample_families": _parse_list(args.sample_families),
            "sample_condition_col": str(args.sample_condition_col),
            "n_response_rows": int(rows.shape[0]),
            "n_analysis_rows": int(analysis.shape[0]),
            "response_arrays": {key: list(value.shape) for key, value in response_arrays.items()},
            "latent_arrays": {key: list(value.shape) for key, value in latent_arrays.items()},
        },
    )
    _progress(f"wrote {len(response_arrays)} response arrays to {out_dir}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
