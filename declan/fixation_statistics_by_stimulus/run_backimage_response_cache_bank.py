"""Generate sharded BackImage response-summary cache banks.

This runner is intentionally lower-level than the aggregate/local/joint
analyses.  It consumes an explicit trace catalog and renders compact
per-(image, trace) summaries in resumable shards.  Downstream analyses can then
assemble condition arrays from the shard row metadata without rerunning the
canonical twin for changes to k, fold policy, alphas, contrasts, or bootstrap
counts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .backimage_cache import (
        atomic_savez,
        atomic_write_csv,
        atomic_write_json,
        done_marker_path,
        load_trace_catalog,
        load_trace_npz,
        make_source_shard,
        shard_stem,
        stable_hash,
        validate_trace_catalog,
    )
    from .image_features import _backimage_canvas, gaze_deg_to_screen_px
    from .run_backimage_aggregate_fem_information import (
        DEFAULT_INPUT,
        CanonicalTwinScorer,
        _add_temporal_basis_summaries,
        _align_response_to_trace,
        _clip_patch,
        _extract_requested_latents,
        _fixed_dct_basis,
        _parse_str_list,
        _prepare_windows,
        _static_trace,
    )
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.fixation_statistics_by_stimulus.backimage_cache import (
        atomic_savez,
        atomic_write_csv,
        atomic_write_json,
        done_marker_path,
        load_trace_catalog,
        load_trace_npz,
        make_source_shard,
        shard_stem,
        stable_hash,
        validate_trace_catalog,
    )
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px
    from declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information import (
        DEFAULT_INPUT,
        CanonicalTwinScorer,
        _add_temporal_basis_summaries,
        _align_response_to_trace,
        _clip_patch,
        _extract_requested_latents,
        _fixed_dct_basis,
        _parse_str_list,
        _prepare_windows,
        _static_trace,
    )


DEFAULT_OUT_DIR = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_response_cache_bank"
)


def _progress(message: str) -> None:
    print(f"[backimage-response-cache-bank] {message}", flush=True)


def _load_temporal_basis(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as loaded:
        if "temporal_basis" in loaded.files:
            return np.asarray(loaded["temporal_basis"], dtype=np.float32)
        if "basis" in loaded.files:
            return np.asarray(loaded["basis"], dtype=np.float32)
        raise ValueError(f"{path} does not contain 'temporal_basis' or 'basis'")


def _summarize(
    response: np.ndarray,
    static: np.ndarray,
    *,
    summaries: set[str],
    dct_basis: np.ndarray,
    temporal_basis: np.ndarray | None,
) -> dict[str, np.ndarray]:
    response = np.asarray(response, dtype=np.float32)
    static = np.asarray(static, dtype=np.float32)
    if response.shape != static.shape:
        raise ValueError(f"Response shape {response.shape} does not match static shape {static.shape}")
    delta = response - static
    out: dict[str, np.ndarray] = {}
    if "mean" in summaries:
        out["mean"] = np.mean(response, axis=0).astype(np.float32)
    if "delta_mean" in summaries:
        out["delta_mean"] = np.mean(delta, axis=0).astype(np.float32)
    if "temporal_dct" in summaries or "temporal_dct_delta" in summaries:
        temp: dict[str, np.ndarray] = {}
        _add_temporal_basis_summaries(temp, response, static, dct_basis, prefix="temporal_dct")
        if "temporal_dct" in summaries:
            out["temporal_dct"] = temp["temporal_dct"]
        if "temporal_dct_delta" in summaries:
            out["temporal_dct_delta"] = temp["temporal_dct_delta"]
    if "temporal_pca" in summaries or "temporal_delta_pca" in summaries:
        if temporal_basis is None:
            raise ValueError("temporal_pca summaries require --temporal-basis-npz")
        if "temporal_pca" in summaries:
            out["temporal_pca"] = (temporal_basis.T @ response).reshape(-1).astype(np.float32)
        if "temporal_delta_pca" in summaries:
            out["temporal_delta_pca"] = (temporal_basis.T @ delta).reshape(-1).astype(np.float32)
    return out


def _extract_patch(row: pd.Series, canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]], patch_size_px: int) -> np.ndarray:
    key = (str(row["session"]), int(row["trial_idx"]))
    if key not in canvas_cache:
        canvas_cache[key] = _backimage_canvas(str(row["session"]), int(row["trial_idx"]))
    canvas, ppd, screen_shape = canvas_cache[key]
    center_px = gaze_deg_to_screen_px(
        np.asarray([float(row["mean_x_deg"]), float(row["mean_y_deg"])]),
        ppd=ppd,
        screen_shape=screen_shape,
    )
    return _clip_patch(canvas, (float(center_px[0]), float(center_px[1])), int(patch_size_px))


def _row_dict(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.to_dict().items():
        if isinstance(value, np.generic):
            out[str(key)] = value.item()
        else:
            out[str(key)] = value
    return out


def _ordered_unique(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _trace_cache_key(trace: np.ndarray) -> tuple[tuple[int, ...], bytes]:
    arr = np.ascontiguousarray(np.asarray(trace, dtype=np.float32))
    return tuple(int(v) for v in arr.shape), arr.tobytes()


def _check_trace_batch_equivalence(
    scorer: Any,
    patch: np.ndarray,
    traces: list[np.ndarray],
    *,
    trace_batch_size: int,
    n_timepoints: int,
    atol: float,
) -> None:
    if int(trace_batch_size) <= 1 or not traces:
        return
    sample = traces[: min(4, len(traces))]
    single = [
        _align_response_to_trace(resp, int(n_timepoints))
        for resp in scorer.responses(patch, sample, trace_batch_size=1)
    ]
    batched = [
        _align_response_to_trace(resp, int(n_timepoints))
        for resp in scorer.responses(patch, sample, trace_batch_size=int(trace_batch_size))
    ]
    max_abs = 0.0
    for one, many in zip(single, batched, strict=True):
        if one.shape != many.shape:
            raise ValueError(f"Trace-batch equivalence failed: response shape {one.shape} != {many.shape}")
        max_abs = max(max_abs, float(np.nanmax(np.abs(one - many))))
    if max_abs > float(atol):
        raise ValueError(f"Trace-batch equivalence failed: max_abs_diff={max_abs:.6g} > {float(atol):.6g}")
    _progress(f"trace-batch equivalence passed for {len(sample)} traces; max_abs_diff={max_abs:.3g}")


def _score_trace_response_map(
    scorer: Any,
    patch: np.ndarray,
    trace_by_key: dict[tuple[tuple[int, ...], bytes], np.ndarray],
    *,
    trace_batch_size: int,
    n_timepoints: int,
    check_equivalence: bool,
    equivalence_atol: float,
) -> dict[tuple[tuple[int, ...], bytes], np.ndarray]:
    keys = list(trace_by_key)
    traces = [trace_by_key[key] for key in keys]
    if bool(check_equivalence):
        _check_trace_batch_equivalence(
            scorer,
            patch,
            traces,
            trace_batch_size=int(trace_batch_size),
            n_timepoints=int(n_timepoints),
            atol=float(equivalence_atol),
        )
    responses = [
        _align_response_to_trace(resp, int(n_timepoints))
        for resp in scorer.responses(patch, traces, trace_batch_size=int(trace_batch_size))
    ]
    return dict(zip(keys, responses, strict=True))


def _response_row_fieldnames(catalog: pd.DataFrame) -> list[str]:
    return _ordered_unique(
        [
            "response_row",
            "image_index",
            "source_row",
            "session",
            "trial_idx",
            *[str(col) for col in catalog.columns],
            "response_frames",
            "response_units",
            "summary_names",
        ]
    )


def _marker_payload(marker: Path) -> dict[str, Any]:
    if not marker.exists():
        return {}
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _payload_path(payload: dict[str, Any], key: str, fallback: Path) -> Path:
    value = payload.get(key)
    return Path(value) if value else Path(fallback)


def _path_signature(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _frame_signature(frame: pd.DataFrame) -> str:
    if frame.empty:
        return stable_hash([], n_hex=24)
    normalized = frame.copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    return stable_hash(normalized.to_dict(orient="records"), n_hex=24)


def _request_signature(
    args: argparse.Namespace,
    *,
    summaries: set[str],
    trace_catalog_path: Path,
    trace_npz_path: Path,
    catalog: pd.DataFrame,
    shard_work: pd.DataFrame,
    shard_source_rows: tuple[int, ...],
) -> tuple[str, dict[str, Any]]:
    source_set = {int(row) for row in shard_source_rows}
    catalog_block = catalog[catalog["source_row"].astype(int).isin(source_set)].copy()
    if {"source_row", "trace_id"}.issubset(catalog_block.columns):
        catalog_block = catalog_block.sort_values(["source_row", "trace_id"]).reset_index(drop=True)
    payload = {
        "version": "response_cache_bank_request_v3",
        "trace_catalog": _path_signature(trace_catalog_path),
        "trace_npz": _path_signature(trace_npz_path),
        "temporal_basis_npz": _path_signature(args.temporal_basis_npz),
        "catalog_rows_hash": _frame_signature(catalog_block),
        "analysis_windows_hash": _frame_signature(shard_work),
        "shard_source_rows": list(shard_source_rows),
        "summaries": sorted(summaries),
        "latent_names": sorted(_parse_str_list(args.latent_names)),
        "write_latents": bool(args.write_latents),
        "write_static_output": bool(args.write_static_output),
        "patch_size_px": int(args.patch_size_px),
        "latent_crop_px": int(args.latent_crop_px),
        "center_crop_px": int(args.center_crop_px),
        "local_field_grid": int(args.local_field_grid),
        "n_timepoints": int(args.n_timepoints),
        "temporal_components": int(args.temporal_components),
        "twin_batch_size": int(args.twin_batch_size),
        "twin_trace_batch_size": int(args.twin_trace_batch_size),
    }
    return stable_hash(payload, n_hex=24), payload


def _completion_marker_outputs_exist(
    marker: Path,
    *,
    row_path: Path,
    summary_path: Path,
    latent_path: Path,
    expected_request_hash: str | None = None,
    expected_summary_names: set[str] | None = None,
    require_latents: bool = False,
) -> bool:
    payload = _marker_payload(marker)
    status = str(payload.get("status", ""))
    if expected_request_hash is not None and payload.get("request_hash") != expected_request_hash:
        return False
    if status == "complete_empty":
        return _payload_path(payload, "rows", row_path).exists()
    if status != "complete":
        return False
    if not _payload_path(payload, "rows", row_path).exists():
        return False
    if not _payload_path(payload, "summaries", summary_path).exists():
        return False
    if expected_summary_names is not None:
        marker_summaries = payload.get("summary_names")
        if marker_summaries is None and isinstance(payload.get("summary_arrays"), dict):
            marker_summaries = sorted(payload["summary_arrays"])
        if set(marker_summaries or []) != set(expected_summary_names):
            return False
    latents = str(payload.get("latents", ""))
    if require_latents and not latents:
        return False
    if latents and not Path(latents).exists():
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--window-manifest", type=Path, required=True)
    parser.add_argument("--trace-catalog", type=Path, required=True)
    parser.add_argument("--trace-npz", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--only-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--min-patch-image-margin-px", type=float, default=None)
    parser.add_argument("--latent-crop-px", type=int, default=151)
    parser.add_argument("--center-crop-px", type=int, default=41)
    parser.add_argument("--local-field-grid", type=int, default=8)
    parser.add_argument("--latent-names", default="gabor_local_field,pyramid_local_field")
    parser.add_argument("--write-latents", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-static-output", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--n-timepoints", type=int, default=40)
    parser.add_argument("--temporal-components", type=int, default=4)
    parser.add_argument("--temporal-basis-npz", type=Path, default=None)
    parser.add_argument("--summaries", default="mean,delta_mean,temporal_dct,temporal_dct_delta")

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--twin-batch-size", type=int, default=128)
    parser.add_argument("--twin-trace-batch-size", type=int, default=4)
    parser.add_argument(
        "--check-trace-batch-equivalence",
        action="store_true",
        help="On the first non-empty image, verify responses match for trace_batch_size=1 and --twin-trace-batch-size.",
    )
    parser.add_argument("--trace-batch-equivalence-atol", type=float, default=1e-5)
    parser.add_argument("--progress-every", type=int, default=8)

    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--reliable-image-coherence-min", type=float, default=0.20)
    parser.add_argument("--reliable-drift-anisotropy-min", type=float, default=0.20)
    parser.add_argument("--min-duration-s", type=float, default=0.10)
    return parser


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = set(_parse_str_list(args.summaries))
    allowed = {"mean", "delta_mean", "temporal_dct", "temporal_dct_delta", "temporal_pca", "temporal_delta_pca"}
    unknown = sorted(summaries.difference(allowed))
    if unknown:
        raise ValueError(f"Unknown summaries requested: {unknown}")

    trace_catalog_path = Path(args.trace_catalog)
    trace_npz_path = Path(args.trace_npz) if args.trace_npz is not None else trace_catalog_path.with_suffix(".npz")
    catalog = load_trace_catalog(trace_catalog_path)
    trace_arrays = {} if bool(args.dry_run) else load_trace_npz(trace_npz_path)
    validate_trace_catalog(catalog, None if bool(args.dry_run) else trace_arrays)

    work = _prepare_windows(args)
    catalog_source_rows = set(catalog["source_row"].astype(int).to_list())
    work = work[work["source_row"].astype(int).isin(catalog_source_rows)].copy().reset_index(drop=True)
    if work.empty:
        raise ValueError("No prepared windows match source_row values in the trace catalog.")
    shard = make_source_shard(work["source_row"].astype(int).to_list(), shard_index=int(args.shard_index), n_shards=int(args.n_shards))
    marker = done_marker_path(out_dir, "response_cache_bank", shard)
    stem = shard_stem("response_cache_bank", shard)
    row_path = out_dir / f"{stem}_rows.csv"
    summary_path = out_dir / f"{stem}_summaries.npz"
    latent_path = out_dir / f"{stem}_latents.npz"
    analysis_path = out_dir / f"{stem}_analysis_windows.csv"
    dry_run_marker = out_dir / f"{stem}.dry_run.json"
    shard_work = work[work["source_row"].astype(int).isin(set(shard.source_rows))].copy().reset_index(drop=True)
    request_hash, request_payload = _request_signature(
        args,
        summaries=summaries,
        trace_catalog_path=trace_catalog_path,
        trace_npz_path=trace_npz_path,
        catalog=catalog,
        shard_work=shard_work,
        shard_source_rows=shard.source_rows,
    )
    dry_run_payload = _marker_payload(dry_run_marker)
    stale_dry_run_marker = str(dry_run_payload.get("status", "")) == "dry_run_complete"
    if (
        bool(args.only_missing)
        and not bool(args.overwrite)
        and _completion_marker_outputs_exist(
            marker,
            row_path=row_path,
            summary_path=summary_path,
            latent_path=latent_path,
            expected_request_hash=request_hash,
            expected_summary_names=summaries,
            require_latents=bool(args.write_latents),
        )
    ):
        _progress(f"shard already complete: {marker}")
        return out_dir
    if not bool(args.overwrite):
        existing = [path for path in (row_path, summary_path, latent_path) if path.exists()]
        if existing and not bool(args.dry_run) and not stale_dry_run_marker:
            raise FileExistsError(
                "Shard outputs already exist without completion marker. "
                f"Pass --overwrite to replace: {', '.join(str(p) for p in existing)}"
            )

    if shard_work.empty:
        _progress(f"shard {shard.shard_index}/{shard.n_shards} has no rows")
        atomic_write_csv(analysis_path, [], fieldnames=[str(col) for col in work.columns])
        atomic_write_csv(row_path, [], fieldnames=_response_row_fieldnames(catalog))
        atomic_write_json(
            marker,
            {
                "status": "complete_empty",
                "shard_index": shard.shard_index,
                "n_shards": shard.n_shards,
                "rows": str(row_path),
                "analysis_windows": str(analysis_path),
                "request_hash": request_hash,
                "request": request_payload,
                "execution": {
                    "device": str(args.device),
                    "twin_batch_size": int(args.twin_batch_size),
                    "twin_trace_batch_size": int(args.twin_trace_batch_size),
                    "check_trace_batch_equivalence": bool(args.check_trace_batch_equivalence),
                    "trace_batch_equivalence_atol": float(args.trace_batch_equivalence_atol),
                },
            },
        )
        return out_dir
    atomic_write_csv(analysis_path, [_row_dict(row) for _, row in shard_work.iterrows()])
    _progress(
        f"prepared shard {shard.shard_index}/{shard.n_shards}: "
        f"windows={shard_work.shape[0]}, catalog_rows={catalog.shape[0]}, dry_run={bool(args.dry_run)}"
    )
    if bool(args.dry_run):
        atomic_write_json(
            dry_run_marker,
            {
                "status": "dry_run_complete",
                "shard_index": shard.shard_index,
                "n_shards": shard.n_shards,
                "n_windows": int(shard_work.shape[0]),
                "trace_catalog": str(trace_catalog_path),
                "analysis_windows": str(analysis_path),
                "request_hash": request_hash,
                "request": request_payload,
                "execution": {
                    "device": str(args.device),
                    "twin_batch_size": int(args.twin_batch_size),
                    "twin_trace_batch_size": int(args.twin_trace_batch_size),
                    "check_trace_batch_equivalence": bool(args.check_trace_batch_equivalence),
                    "trace_batch_equivalence_atol": float(args.trace_batch_equivalence_atol),
                },
            },
        )
        return out_dir

    latent_filter = set(_parse_str_list(args.latent_names))
    if "all" in latent_filter:
        latent_filter = set()
    dct_basis = _fixed_dct_basis(int(args.n_timepoints), int(args.temporal_components))
    temporal_basis = _load_temporal_basis(args.temporal_basis_npz)
    scorer = CanonicalTwinScorer(device=str(args.device), batch_size=int(args.twin_batch_size))
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}

    rows: list[dict[str, Any]] = []
    summary_values: dict[str, list[np.ndarray]] = {name: [] for name in summaries}
    latent_values: dict[str, list[np.ndarray]] = {}

    by_source = {int(source_row): block.copy() for source_row, block in catalog.groupby("source_row", sort=False)}
    response_row = 0
    progress_every = max(1, int(args.progress_every))
    trace_batch_checked = False
    duplicate_trace_rows_skipped = 0
    requested_trace_response_count = 0
    unique_trace_response_count = 0
    for image_i, (_, row) in enumerate(tqdm(list(shard_work.iterrows()), desc="response cache shard"), start=1):
        source_row = int(row["source_row"])
        patch = _extract_patch(row, canvas_cache, int(args.patch_size_px))
        if bool(args.write_latents):
            latents = _extract_requested_latents(
                patch,
                latent_crop_px=int(args.latent_crop_px),
                center_crop_px=int(args.center_crop_px),
                local_field_grid=int(args.local_field_grid),
                requested=latent_filter,
            )
            if not latents:
                raise ValueError(f"No requested latent features were available for source_row={source_row}.")
            for name, value in latents.items():
                latent_values.setdefault(name, []).append(np.asarray(value, dtype=np.float32))

        trace_block = by_source.get(source_row)
        if trace_block is None or trace_block.empty:
            continue

        static_rows = []
        trace_rows: list[tuple[pd.Series, tuple[tuple[int, ...], bytes]]] = []
        static_trace = _static_trace(int(args.n_timepoints))
        static_key = _trace_cache_key(static_trace)
        trace_by_key: dict[tuple[tuple[int, ...], bytes], np.ndarray] = {static_key: static_trace}
        for _, trace_row in trace_block.iterrows():
            family = str(trace_row["family"])
            if family == "static":
                static_rows.append(trace_row)
                continue
            trace_key = str(trace_row.get("trace_key", trace_row["trace_id"]))
            trace = np.asarray(trace_arrays[trace_key], dtype=np.float32)
            if trace.shape != (int(args.n_timepoints), 2):
                raise ValueError(
                    f"Trace {trace_key!r} has shape {trace.shape}; expected {(int(args.n_timepoints), 2)}"
                )
            key = _trace_cache_key(trace)
            if key in trace_by_key:
                duplicate_trace_rows_skipped += 1
            else:
                trace_by_key[key] = trace
            trace_rows.append((trace_row, key))
        responses_by_key = _score_trace_response_map(
            scorer,
            patch,
            trace_by_key,
            trace_batch_size=int(args.twin_trace_batch_size),
            n_timepoints=int(args.n_timepoints),
            check_equivalence=bool(args.check_trace_batch_equivalence) and not trace_batch_checked,
            equivalence_atol=float(args.trace_batch_equivalence_atol),
        )
        if bool(args.check_trace_batch_equivalence) and not trace_batch_checked:
            trace_batch_checked = True
        static = responses_by_key[static_key]
        requested_trace_response_count += 1 + len(trace_rows)
        unique_trace_response_count += len(trace_by_key)
        output_pairs: list[tuple[pd.Series, np.ndarray]] = []
        if bool(args.write_static_output):
            if static_rows:
                output_pairs.extend((static_row, static) for static_row in static_rows)
            else:
                static_meta = pd.Series(
                    {
                        "source_row": source_row,
                        "trace_id": f"static:{source_row}",
                        "trace_key": "",
                        "family": "static",
                        "scale_id": "static",
                        "scale": 0.0,
                        "seed": "",
                        "sample_index": 0,
                        "pairing_mode": "static",
                    }
                )
                output_pairs.append((static_meta, static))
        output_pairs.extend((trace_row, responses_by_key[key]) for trace_row, key in trace_rows)

        for trace_row, response in output_pairs:
            summary = _summarize(
                response,
                static,
                summaries=summaries,
                dct_basis=dct_basis,
                temporal_basis=temporal_basis,
            )
            for name in summaries:
                summary_values[name].append(summary[name])
            trace_meta = _row_dict(trace_row)
            rows.append(
                {
                    "response_row": int(response_row),
                    "image_index": int(row["image_index"]),
                    "source_row": source_row,
                    "session": str(row["session"]),
                    "trial_idx": int(row["trial_idx"]),
                    **trace_meta,
                    "response_frames": int(response.shape[0]),
                    "response_units": int(response.shape[1]),
                    "summary_names": ",".join(sorted(summaries)),
                }
            )
            response_row += 1
        if image_i == 1 or image_i == shard_work.shape[0] or image_i % progress_every == 0:
            _progress(f"windows {image_i}/{shard_work.shape[0]}; response rows={response_row}")

    if not rows:
        raise ValueError("No response rows were generated for this shard.")
    arrays = {name: np.vstack(values).astype(np.float32, copy=False) for name, values in summary_values.items()}
    atomic_write_csv(row_path, rows)
    atomic_savez(summary_path, arrays)
    if bool(args.write_latents) and latent_values:
        latent_arrays = {name: np.vstack(values).astype(np.float32, copy=False) for name, values in latent_values.items()}
        latent_arrays["source_row"] = shard_work["source_row"].to_numpy(dtype=np.int64)
        latent_arrays["image_index"] = shard_work["image_index"].to_numpy(dtype=np.int64)
        atomic_savez(latent_path, latent_arrays)
    atomic_write_json(
        marker,
        {
            "status": "complete",
            "shard_index": shard.shard_index,
            "n_shards": shard.n_shards,
            "n_windows": int(shard_work.shape[0]),
            "n_response_rows": int(len(rows)),
            "n_requested_trace_responses": int(requested_trace_response_count),
            "n_unique_trace_responses_scored": int(unique_trace_response_count),
            "n_duplicate_trace_rows_reused": int(duplicate_trace_rows_skipped),
            "summary_arrays": {name: list(arr.shape) for name, arr in arrays.items()},
            "summary_names": sorted(summaries),
            "trace_catalog": str(trace_catalog_path),
            "trace_npz": str(trace_npz_path),
            "analysis_windows": str(analysis_path),
            "rows": str(row_path),
            "summaries": str(summary_path),
            "latents": str(latent_path) if bool(args.write_latents) and latent_values else "",
            "request_hash": request_hash,
            "request": request_payload,
            "execution": {
                "device": str(args.device),
                "twin_batch_size": int(args.twin_batch_size),
                "twin_trace_batch_size": int(args.twin_trace_batch_size),
                "check_trace_batch_equivalence": bool(args.check_trace_batch_equivalence),
                "trace_batch_equivalence_atol": float(args.trace_batch_equivalence_atol),
                "trace_batch_equivalence_checked": bool(trace_batch_checked),
            },
        },
    )
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
