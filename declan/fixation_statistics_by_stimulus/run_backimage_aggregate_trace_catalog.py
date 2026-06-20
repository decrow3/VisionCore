"""Materialize aggregate-style BackImage trace catalogs.

The aggregate runner currently samples and renders traces inside the same loop.
This script separates the sampling step: it writes a CSV/NPZ catalog of
per-image trace work units that can be rendered by
``run_backimage_response_cache_bank`` in resumable shards.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .backimage_cache import array_hash, atomic_write_csv, atomic_write_json, stable_hash, trace_catalog_id, write_trace_catalog
    from .run_backimage_aggregate_fem_information import (
        DEFAULT_INPUT,
        _build_trace_bank,
        _eligible_trace_bank_indices,
        _family_raw_trace,
        _parse_float_list,
        _parse_str_list,
        _prepare_windows,
        _scale_family_raw_trace,
        _scale_token,
        _session_dataset_cache,
        _trace_filter_kwargs,
    )
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.fixation_statistics_by_stimulus.backimage_cache import (
        array_hash,
        atomic_write_csv,
        atomic_write_json,
        stable_hash,
        trace_catalog_id,
        write_trace_catalog,
    )
    from declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information import (
        DEFAULT_INPUT,
        _build_trace_bank,
        _eligible_trace_bank_indices,
        _family_raw_trace,
        _parse_float_list,
        _parse_str_list,
        _prepare_windows,
        _scale_family_raw_trace,
        _scale_token,
        _session_dataset_cache,
        _trace_filter_kwargs,
    )


DEFAULT_OUT_DIR = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_aggregate_trace_catalog"
)


def _progress(message: str) -> None:
    print(f"[backimage-aggregate-trace-catalog] {message}", flush=True)


def _parse_int_list(text: str) -> list[int]:
    return [int(part) for part in _parse_str_list(text)]


def _trace_row(base: dict[str, Any], trace: np.ndarray) -> tuple[dict[str, Any], str, np.ndarray]:
    trace_hash = array_hash(trace, n_hex=20)
    row = {**base, "trace_hash": trace_hash}
    trace_id = trace_catalog_id(row)
    trace_key = f"trace_{trace_id}"
    row["trace_id"] = trace_id
    row["trace_key"] = trace_key
    return row, trace_key, np.asarray(trace, dtype=np.float32)


def _prepare_trace_pool_windows(args: argparse.Namespace, analysis_work: pd.DataFrame) -> pd.DataFrame:
    if str(args.trace_pool_scope) == "analysis":
        return analysis_work.copy().reset_index(drop=True)
    pool_args = argparse.Namespace(**vars(args))
    pool_args.window_manifest = args.trace_pool_manifest
    pool_args.max_images = int(args.trace_pool_max_images)
    pool_args.seed = int(args.trace_pool_seed) if args.trace_pool_seed is not None else int(args.seed)
    pool = _prepare_windows(pool_args)
    if pool.empty:
        raise ValueError("No BackImage windows survived filters for the trace pool.")
    return pool.reset_index(drop=True)


def _source_rows_hash(frame: pd.DataFrame) -> str:
    return stable_hash(frame["source_row"].astype(int).to_list(), n_hex=20)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--window-manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--catalog-name", default="aggregate_trace_catalog")
    parser.add_argument("--max-images", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0, help="Seed for window-manifest sampling.")
    parser.add_argument("--trace-pool-scope", choices=("full_filtered", "analysis"), default="full_filtered")
    parser.add_argument("--trace-pool-manifest", type=Path, default=None)
    parser.add_argument("--trace-pool-max-images", type=int, default=0)
    parser.add_argument("--trace-pool-seed", type=int, default=None)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--trace-samples-per-condition", type=int, default=4)
    parser.add_argument("--motion-families", default="empirical,ou,brownian,rotated")
    parser.add_argument("--observed-rms-scales", default="0.25,0.5,1.0")
    parser.add_argument("--n-timepoints", type=int, default=40)
    parser.add_argument("--reuse-trace-sources-across-scales", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-rms-deg", type=float, default=0.12)

    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--min-patch-image-margin-px", type=float, default=None)
    parser.add_argument("--reliable-image-coherence-min", type=float, default=0.20)
    parser.add_argument("--reliable-drift-anisotropy-min", type=float, default=0.20)
    parser.add_argument("--min-duration-s", type=float, default=0.10)

    parser.add_argument("--max-trace-source-rms-deg", type=float, default=0.06)
    parser.add_argument("--max-trace-source-radius-deg", type=float, default=0.2)
    parser.add_argument("--max-trace-source-path-length-deg", type=float, default=None)
    parser.add_argument("--max-rendered-trace-path-length-deg", type=float, default=1.5)
    parser.add_argument("--max-source-trace-path-length-deg", type=float, default=None)
    parser.add_argument("--max-trace-source-speed-p95-deg-s", type=float, default=20.0)
    parser.add_argument("--max-trace-source-microsaccade-events", type=int, default=0)
    parser.add_argument("--microsaccade-speed-threshold-dps", type=float, default=None)
    parser.add_argument("--microsaccade-threshold-z", type=float, default=6.0)
    parser.add_argument("--microsaccade-pad-frames", type=int, default=1)
    return parser


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = _prepare_windows(args)
    if work.empty:
        raise ValueError("No BackImage windows survived filters.")
    seeds = _parse_int_list(args.seeds)
    families = _parse_str_list(args.motion_families)
    scales = _parse_float_list(args.observed_rms_scales)
    _progress(
        f"windows={work.shape[0]}, seeds={seeds}, families={families}, "
        f"scales={scales}, K={int(args.trace_samples_per_condition)}"
    )

    trace_pool_work = _prepare_trace_pool_windows(args, work)
    _progress(
        f"trace_pool_scope={args.trace_pool_scope}; trace_pool_windows={trace_pool_work.shape[0]}"
    )
    eyepos_by_session = _session_dataset_cache(trace_pool_work["session"].astype(str).to_list())
    trace_bank = _build_trace_bank(
        trace_pool_work,
        eyepos_by_session,
        int(args.n_timepoints),
        microsaccade_speed_threshold_dps=(
            float(args.microsaccade_speed_threshold_dps) if args.microsaccade_speed_threshold_dps is not None else None
        ),
        microsaccade_threshold_z=float(args.microsaccade_threshold_z),
        microsaccade_pad_frames=int(args.microsaccade_pad_frames),
    )
    rows: list[dict[str, Any]] = []
    trace_arrays: dict[str, np.ndarray] = {}
    total = int(work.shape[0]) * len(seeds)
    for done, (seed, (_, row)) in enumerate(
        ((seed, item) for seed in seeds for item in work.iterrows()),
        start=1,
    ):
        rng = np.random.default_rng(int(seed) + int(row["source_row"]) * 1009)
        source_row = int(row["source_row"])
        reusable_sources: dict[tuple[str, int], int] = {}
        reusable_raw: dict[tuple[str, int], np.ndarray] = {}
        eligible = _eligible_trace_bank_indices(trace_bank, current_source_row=source_row, **_trace_filter_kwargs(args))
        if not eligible:
            raise ValueError(f"No eligible trace-bank entries for source_row={source_row}")
        if bool(args.reuse_trace_sources_across_scales):
            for family in families:
                for sample_index in range(int(args.trace_samples_per_condition)):
                    key = (family, int(sample_index))
                    bank_index = int(eligible[int(rng.integers(0, len(eligible)))])
                    item = trace_bank[bank_index]
                    reusable_sources[key] = bank_index
                    reusable_raw[key] = _family_raw_trace(
                        family,
                        item["trace"],
                        float(item["lag1_autocorr"]),
                        rng=rng,
                        max_rms_deg=float(args.max_rms_deg),
                        source_shape=item.get("covariance_shape"),
                        selection_rms=float(item["observed_rms_deg"]),
                        target_path_length=float(item["path_length_deg"]),
                    )
        for scale in scales:
            scale_id = f"rel_{_scale_token(scale)}x"
            for family in families:
                for sample_index in range(int(args.trace_samples_per_condition)):
                    key = (family, int(sample_index))
                    if bool(args.reuse_trace_sources_across_scales):
                        bank_index = reusable_sources[key]
                        item = trace_bank[bank_index]
                        raw = reusable_raw[key]
                    else:
                        bank_index = int(eligible[int(rng.integers(0, len(eligible)))])
                        item = trace_bank[bank_index]
                        raw = _family_raw_trace(
                            family,
                            item["trace"],
                            float(item["lag1_autocorr"]),
                            rng=rng,
                            max_rms_deg=float(args.max_rms_deg),
                            source_shape=item.get("covariance_shape"),
                            selection_rms=float(item["observed_rms_deg"]),
                            target_path_length=float(item["path_length_deg"]),
                        )
                    target_rms = float(scale) * float(item["observed_rms_deg"])
                    trace, meta = _scale_family_raw_trace(raw, target_rms, max_rms_deg=float(args.max_rms_deg))
                    base = {
                        "source_row": source_row,
                        "family": family,
                        "scale_id": scale_id,
                        "scale": float(scale),
                        "seed": int(seed),
                        "sample_index": int(sample_index),
                        "trace_bank_index": int(bank_index),
                        "trace_source_row": int(item["source_row"]),
                        "trace_source_session": str(item["session"]),
                        "pairing_mode": "unpaired_ensemble",
                        "raw_trace_reused_across_scales": bool(args.reuse_trace_sources_across_scales),
                        "source_trace_rms_deg": float(item["observed_rms_deg"]),
                        "source_trace_path_length_deg": float(item["path_length_deg"]),
                        "source_trace_lag1": float(item["lag1_autocorr"]),
                        "requested_rms_deg": float(meta["requested_rms_deg"]),
                        "effective_rms_deg": float(meta["effective_rms_deg"]),
                        "effective_to_requested_rms": (
                            float(meta["effective_rms_deg"]) / float(meta["requested_rms_deg"])
                            if float(meta["requested_rms_deg"]) > 0.0
                            else np.nan
                        ),
                        "rms_clipped_high": bool(meta["rms_clipped_high"]),
                        "generated_lag1_autocorr": float(meta["generated_lag1_autocorr"]),
                        "path_length_deg": float(meta["path_length_deg"]),
                        "speed_mean_deg_s": float(meta["speed_mean_deg_s"]),
                        "speed_median_deg_s": float(meta["speed_median_deg_s"]),
                        "speed_p95_deg_s": float(meta["speed_p95_deg_s"]),
                    }
                    trace_row, trace_key, trace_array = _trace_row(base, trace)
                    rows.append(trace_row)
                    trace_arrays[trace_key] = trace_array
        if done == 1 or done == total or done % 16 == 0:
            _progress(f"catalog rows for seed/window {done}/{total}: rows={len(rows)}")

    catalog_path = out_dir / f"{args.catalog_name}.csv"
    trace_npz_path = out_dir / f"{args.catalog_name}.npz"
    write_trace_catalog(catalog_path, rows, trace_arrays, trace_npz_path=trace_npz_path)
    analysis_path = out_dir / f"{args.catalog_name}_analysis_windows.csv"
    trace_pool_path = out_dir / f"{args.catalog_name}_trace_pool_windows.csv"
    metadata_path = out_dir / f"{args.catalog_name}_run_metadata.json"
    atomic_write_csv(analysis_path, work.to_dict(orient="records"))
    atomic_write_csv(trace_pool_path, trace_pool_work.to_dict(orient="records"))
    atomic_write_json(
        metadata_path,
        {
            "config": vars(args),
            "analysis_windows": str(analysis_path),
            "trace_pool_windows": str(trace_pool_path),
            "catalog": str(catalog_path),
            "trace_npz": str(trace_npz_path),
            "n_analysis_windows": int(work.shape[0]),
            "n_trace_pool_windows": int(trace_pool_work.shape[0]),
            "n_trace_catalog_rows": int(len(rows)),
            "n_trace_arrays": int(len(trace_arrays)),
            "analysis_source_rows_hash": _source_rows_hash(work),
            "trace_pool_source_rows_hash": _source_rows_hash(trace_pool_work),
            "families": families,
            "scales": scales,
            "seeds": seeds,
            "trace_samples_per_condition": int(args.trace_samples_per_condition),
        },
    )
    _progress(f"wrote {len(rows)} trace rows to {catalog_path}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
