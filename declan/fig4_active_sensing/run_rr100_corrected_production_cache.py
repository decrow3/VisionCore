#!/usr/bin/env python3
"""Build the corrected 100-image x 1,000-trace RR100 response cache.

The production grid is scheduled as balanced rounds.  With the canonical
100 x 1,000 cohort and a trace block size of 10, each round contains every
image ten times and every trace exactly once.  Fifty rounds are therefore a
balanced, connected half of the Cartesian grid.

Each image-within-round block (ten movies in production) is written atomically.
The runner validates and skips completed blocks on restart.  A stopped run can
therefore be resumed without recomputing any completed neural condition, and a
completed round is already a balanced partial-analysis cohort.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd

from declan.fig4_active_sensing.audit_rr100_eye_trace_conditioning_and_nyquist_power import (
    corrected_crop_xy_deg,
    load_dset,
)
from declan.fig4_active_sensing.run_rr100_corrected_ssi_map_first_smoke import (
    MAPPING,
)
from declan.fig4_active_sensing.run_rr100_direct_fem_orientation_checkpoint import (
    RR100_MOVIE_MEDOID_VERSION,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
    CanonicalTwinScorer,
)
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _standardize_uint_like,
    _trace_xy_to_twin_helper_order,
)
from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COHORT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1"
DEFAULT_OUT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1"
DEFAULT_IMAGE_AUDIT = ROOT / "outputs/fig4_active_sensing/rr100_legacy100_corrected_image_audit_checkpoint_24_v1"

N_HISTORY = 32
N_SCORE = 40
FRAME_RATE_HZ = 120.0
DT = 1.0 / FRAME_RATE_HZ
EPS = 1e-10
N_UNITS = 100
PRODUCTION_IMAGES = 100
PRODUCTION_TRACES = 1000
PRODUCTION_BLOCK_SIZE = 10
PRODUCTION_STATUS = "corrected_100x1000_production_cohort_frozen"

SUMMARY_ARRAYS = (
    "information_numerator_bits_spikes",
    "expected_spikes",
    "mean_rate_hz",
    "movie_ssi_bits_per_spike",
    "temporal_sd_rate_hz",
    "temporal_rms_delta_from_stabilized_hz",
    "temporal_mean_abs_delta_from_stabilized_hz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--image-audit-dir", type=Path, default=DEFAULT_IMAGE_AUDIT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=16)
    parser.add_argument("--trace-block-size", type=int, default=PRODUCTION_BLOCK_SIZE)
    parser.add_argument("--half-index", choices=("0", "1", "all"), default="0")
    parser.add_argument("--round-start", type=int, default=0)
    parser.add_argument("--round-stop", type=int, default=0)
    parser.add_argument("--image-shard-count", type=int, default=1)
    parser.add_argument("--image-shard-index", type=int, default=0)
    parser.add_argument("--max-new-blocks", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-nonproduction-cohort", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.resolve().stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "sha256": sha256_file(path),
    }


def identity_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def cohort_paths(cohort_dir: Path) -> tuple[Path, Path, Path]:
    return (
        cohort_dir / "corrected100_images.csv",
        cohort_dir / "corrected1000_traces.csv",
        cohort_dir / "manifest.json",
    )


def load_and_validate_cohort(
    cohort_dir: Path,
    *,
    block_size: int,
    allow_nonproduction: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Path, Path, Path]:
    image_path, trace_path, manifest_path = cohort_paths(cohort_dir)
    missing = [path for path in (image_path, trace_path, manifest_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The immutable corrected production cohort is not frozen yet; missing: "
            + ", ".join(str(path) for path in missing)
        )
    images = pd.read_csv(image_path).sort_values("image_index").reset_index(drop=True)
    traces = pd.read_csv(trace_path).sort_values("trace_index").reset_index(drop=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_image = {"image_index", "session"}
    required_trace = {
        "trace_index",
        "session",
        "corrected_history_global_start",
        "corrected_history_global_stop_exclusive",
        "corrected_scored_global_start",
        "corrected_scored_global_stop_exclusive",
        "explicit_history_valid",
    }
    if missing_columns := required_image.difference(images.columns):
        raise ValueError(f"Image table is missing columns: {sorted(missing_columns)}")
    if missing_columns := required_trace.difference(traces.columns):
        raise ValueError(f"Trace table is missing columns: {sorted(missing_columns)}")
    if images.image_index.duplicated().any() or traces.trace_index.duplicated().any():
        raise ValueError("Image and trace identities must be unique")
    if not traces.explicit_history_valid.astype(bool).all():
        raise ValueError("Every production trace must pass the explicit recorded-history gate")
    if len(traces) % len(images) != 0:
        raise ValueError("Balanced-round schedule requires n_traces divisible by n_images")
    if len(traces) % int(block_size) != 0 or int(block_size) * len(images) != len(traces):
        raise ValueError(
            "One round must cover every trace exactly once: require trace_block_size * n_images == n_traces"
        )
    is_production = (
        len(images) == PRODUCTION_IMAGES
        and len(traces) == PRODUCTION_TRACES
        and int(block_size) == PRODUCTION_BLOCK_SIZE
        and manifest.get("status") == PRODUCTION_STATUS
    )
    if not is_production and not allow_nonproduction:
        raise RuntimeError(
            "Cohort is not the frozen corrected 100x1,000 production cohort. "
            "Use --allow-nonproduction-cohort only for an explicitly labeled smoke test."
        )
    return images, traces, manifest, image_path, trace_path, manifest_path


def make_balanced_schedule(
    image_indices: np.ndarray,
    trace_indices: np.ndarray,
    *,
    block_size: int,
) -> pd.DataFrame:
    """Return a complete Cartesian schedule grouped into balanced rounds."""
    image_ids = np.asarray(image_indices, dtype=int)
    trace_ids = np.asarray(trace_indices, dtype=int)
    n_images = len(image_ids)
    n_traces = len(trace_ids)
    if n_traces != n_images * int(block_size):
        raise ValueError("Expected n_traces == n_images * block_size")
    rows: list[dict[str, int]] = []
    n_rounds = n_traces // int(block_size)
    for round_index in range(n_rounds):
        for image_ordinal, image_index in enumerate(image_ids):
            block_start = (int(block_size) * (image_ordinal + round_index)) % n_traces
            for within_block in range(int(block_size)):
                trace_ordinal = (block_start + within_block) % n_traces
                rows.append(
                    {
                        "round_index": int(round_index),
                        "half_index": int(round_index >= n_rounds // 2),
                        "image_ordinal": int(image_ordinal),
                        "image_index": int(image_index),
                        "within_block": int(within_block),
                        "trace_ordinal": int(trace_ordinal),
                        "trace_index": int(trace_ids[trace_ordinal]),
                    }
                )
    schedule = pd.DataFrame(rows)
    if len(schedule) != n_images * n_traces:
        raise AssertionError("Schedule does not contain the complete Cartesian grid")
    if schedule[["image_index", "trace_index"]].duplicated().any():
        raise AssertionError("Schedule contains duplicate image x trace pairs")
    for _, group in schedule.groupby("round_index"):
        if group.image_index.value_counts().nunique() != 1:
            raise AssertionError("Round image degrees are not balanced")
        if not np.all(group.trace_index.value_counts().to_numpy() == 1):
            raise AssertionError("A round must contain every trace exactly once")
    return schedule


def selected_rounds(args: argparse.Namespace, n_rounds: int) -> list[int]:
    half = n_rounds // 2
    if n_rounds % 2:
        raise ValueError("Production schedule requires an even number of rounds")
    if args.half_index == "0":
        rounds = list(range(0, half))
    elif args.half_index == "1":
        rounds = list(range(half, n_rounds))
    else:
        rounds = list(range(n_rounds))
    lower = max(0, int(args.round_start))
    upper = int(args.round_stop) if int(args.round_stop) > 0 else n_rounds
    return [round_index for round_index in rounds if lower <= round_index < min(upper, n_rounds)]


def trace_sample_indices(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    history = np.arange(
        int(row.corrected_history_global_start),
        int(row.corrected_history_global_stop_exclusive),
        2,
        dtype=np.int64,
    )
    score = np.arange(
        int(row.corrected_scored_global_start),
        int(row.corrected_scored_global_stop_exclusive),
        2,
        dtype=np.int64,
    )
    if history.shape != (N_HISTORY,) or score.shape != (N_SCORE,):
        raise ValueError(
            f"Trace {row.trace_index} has invalid history/score lengths {history.shape}/{score.shape}"
        )
    return history, score


def materialize_trace_segments(traces: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    history_out = np.empty((len(traces), N_HISTORY, 2), dtype=np.float32)
    score_out = np.empty((len(traces), N_SCORE, 2), dtype=np.float32)
    for session, rows in traces.groupby("session", sort=True):
        cache: dict[Any, Any] = {}
        dset = load_dset(str(session), cache)
        crop = corrected_crop_xy_deg(dset)
        for ordinal, row in rows.iterrows():
            history_indices, score_indices = trace_sample_indices(row)
            history = crop[history_indices]
            score = crop[score_indices]
            center = score.mean(axis=0, keepdims=True)
            history_out[int(ordinal)] = (history - center).astype(np.float32)
            score_out[int(ordinal)] = (score - center).astype(np.float32)
        del crop, dset
        cache.clear()
        gc.collect()
    if not np.isfinite(history_out).all() or not np.isfinite(score_out).all():
        raise ValueError("Materialized trace segments contain non-finite values")
    return history_out, score_out


def load_or_prepare_trace_segments(
    out_dir: Path,
    traces: pd.DataFrame,
    *,
    request_sha256: str,
) -> tuple[np.ndarray, np.ndarray]:
    path = out_dir / "input_cache" / "corrected_trace_segments.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if str(data["request_sha256"].item()) != request_sha256:
                raise RuntimeError("Trace-segment cache belongs to a different request identity")
            if not np.array_equal(data["trace_index"], traces.trace_index.to_numpy(int)):
                raise RuntimeError("Trace-segment identity order changed")
            history = np.asarray(data["history_xy_deg"], dtype=np.float32)
            score = np.asarray(data["score_xy_deg"], dtype=np.float32)
            if str(data["history_sha256"].item()) != sha256_array(history):
                raise RuntimeError("Frozen history trace checksum mismatch")
            if str(data["score_sha256"].item()) != sha256_array(score):
                raise RuntimeError("Frozen scored trace checksum mismatch")
        if history.shape != (len(traces), N_HISTORY, 2) or score.shape != (len(traces), N_SCORE, 2):
            raise RuntimeError("Trace-segment cache has invalid shapes")
        return history, score
    history, score = materialize_trace_segments(traces)
    atomic_npz(
        path,
        request_sha256=np.asarray(request_sha256),
        trace_index=traces.trace_index.to_numpy(np.int64),
        history_xy_deg=history,
        score_xy_deg=score,
        history_sha256=np.asarray(sha256_array(history)),
        score_sha256=np.asarray(sha256_array(score)),
    )
    return history, score


def resolve_image_patch_path(row: pd.Series, image_audit_dir: Path) -> Path:
    for column in ("corrected_patch_npz", "patch_npz_path", "image_npz_path"):
        if column in row.index and pd.notna(row[column]) and str(row[column]).strip():
            path = Path(str(row[column]))
            return path if path.is_absolute() else ROOT / path
    return image_audit_dir / "partials" / f"image_{int(row.image_index):03d}.npz"


def load_image_patches(
    images: pd.DataFrame,
    image_audit_dir: Path,
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    ppd_lookup: dict[str, float] = {}
    sessions_needing_ppd = (
        sorted(images.session.astype(str).unique())
        if "patch_ppd" not in images.columns
        else sorted(images.loc[images.patch_ppd.isna(), "session"].astype(str).unique())
    )
    if sessions_needing_ppd:
        for session in sessions_needing_ppd:
            cache: dict[Any, Any] = {}
            dset = load_dset(session, cache)
            ppd_lookup[session] = float(dset.metadata["ppd"])
            del dset
            cache.clear()
            gc.collect()
    patches: list[np.ndarray] = []
    ppds: list[float] = []
    paths: list[str] = []
    for _, row in images.iterrows():
        path = resolve_image_patch_path(row, image_audit_dir)
        if not path.exists():
            raise FileNotFoundError(f"Corrected source patch missing for image {row.image_index}: {path}")
        key = str(row.get("corrected_patch_key", "corrected_patch"))
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                raise KeyError(f"Patch key {key!r} missing from {path}")
            patch = np.asarray(data[key], dtype=np.float32)
        if patch.ndim != 2 or min(patch.shape) < 51 or not np.isfinite(patch).all():
            raise ValueError(f"Invalid corrected patch for image {row.image_index}: {patch.shape}")
        patches.append(patch)
        ppds.append(
            float(row.patch_ppd)
            if "patch_ppd" in row.index and pd.notna(row.patch_ppd)
            else ppd_lookup[str(row.session)]
        )
        paths.append(str(path.resolve()))
    return patches, np.asarray(ppds, dtype=np.float64), paths


def load_or_prepare_image_patches(
    out_dir: Path,
    images: pd.DataFrame,
    image_audit_dir: Path,
    *,
    request_sha256: str,
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Freeze corrected source patches inside the response-cache directory."""
    cache_dir = out_dir / "input_cache" / "images"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = [cache_dir / f"image_{int(row.image_index):03d}.npz" for _, row in images.iterrows()]
    if all(path.exists() for path in cache_paths):
        patches: list[np.ndarray] = []
        ppds: list[float] = []
        frozen_paths: list[str] = []
        for (_, row), path in zip(images.iterrows(), cache_paths, strict=True):
            with np.load(path, allow_pickle=False) as data:
                if str(data["request_sha256"].item()) != request_sha256:
                    raise RuntimeError(f"Frozen patch belongs to a different request: {path}")
                if int(data["image_index"].item()) != int(row.image_index):
                    raise RuntimeError(f"Frozen patch identity mismatch: {path}")
                patch = np.asarray(data["corrected_patch"], dtype=np.float32)
                if str(data["patch_sha256"].item()) != sha256_array(patch):
                    raise RuntimeError(f"Frozen patch checksum mismatch: {path}")
                patches.append(patch)
                ppds.append(float(data["patch_ppd"].item()))
                frozen_paths.append(str(path.resolve()))
        return patches, np.asarray(ppds, dtype=np.float64), frozen_paths
    live_patches, live_ppds, source_paths = load_image_patches(images, image_audit_dir)
    patches: list[np.ndarray] = []
    ppds: list[float] = []
    for (_, row), live_patch, live_ppd, source_path, path in zip(
        images.iterrows(), live_patches, live_ppds, source_paths, cache_paths, strict=True
    ):
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                if str(data["request_sha256"].item()) != request_sha256:
                    raise RuntimeError(f"Frozen patch belongs to a different request: {path}")
                patch = np.asarray(data["corrected_patch"], dtype=np.float32)
                ppd = float(data["patch_ppd"].item())
                if str(data["patch_sha256"].item()) != sha256_array(patch):
                    raise RuntimeError(f"Frozen patch checksum mismatch: {path}")
                if sha256_array(live_patch) != sha256_array(patch) or not np.isclose(live_ppd, ppd):
                    raise RuntimeError(f"Live corrected patch changed after partial input freeze: {path}")
        else:
            patch = live_patch
            ppd = float(live_ppd)
            atomic_npz(
                path,
                request_sha256=np.asarray(request_sha256),
                image_index=np.asarray(int(row.image_index), dtype=np.int64),
                corrected_patch=patch,
                patch_ppd=np.asarray(ppd, dtype=np.float64),
                patch_sha256=np.asarray(sha256_array(patch)),
                source_patch_path=np.asarray(source_path),
            )
        patches.append(patch)
        ppds.append(ppd)
    ppds_array = np.asarray(ppds, dtype=np.float64)
    pd.DataFrame(
        {
            "image_index": images.image_index.to_numpy(int),
            "source_patch_path": source_paths,
            "frozen_patch_path": [str(path.resolve()) for path in cache_paths],
            "patch_ppd": ppds_array,
            "patch_sha256": [sha256_array(patch) for patch in patches],
        }
    ).to_csv(out_dir / "input_cache" / "frozen_image_inputs.csv", index=False)
    return patches, ppds_array, [str(path.resolve()) for path in cache_paths]


def render_scored_embedding(common: Any, torch: Any, patch: np.ndarray, trace72: np.ndarray, ppd: float):
    trace = np.asarray(trace72, dtype=np.float32)
    if trace.shape != (N_HISTORY + N_SCORE, 2):
        raise ValueError(f"Expected a 72 x 2 explicit-history trace, got {trace.shape}")
    image = _standardize_uint_like(patch)
    full_stack = np.broadcast_to(
        image[None], (trace.shape[0] + int(common.N_LAGS) + 1, *image.shape)
    ).copy()
    eye = torch.from_numpy(_trace_xy_to_twin_helper_order(-trace))
    embedded = common.make_counterfactual_stim(
        full_stack,
        eye,
        ppd=float(ppd),
        scale_factor=1.0,
        n_lags=int(common.N_LAGS),
        out_size=common.OUT_SIZE,
    )
    if int(embedded.shape[0]) != trace.shape[0] + 1:
        raise ValueError(f"Expected native T+1 embedding, got {tuple(embedded.shape)}")
    aligned = embedded[1 : 1 + trace.shape[0]]
    scored = aligned[N_HISTORY:]
    if int(scored.shape[0]) != N_SCORE:
        raise AssertionError(f"Expected exactly {N_SCORE} scored inputs, got {scored.shape[0]}")
    return (scored - 127.0) / 255.0


def response_timecourses(scorer: CanonicalTwinScorer, view: Any, scored_stim: Any) -> tuple[np.ndarray, np.ndarray]:
    full = scorer._compute_rate_map_batched(scored_stim)
    rr100 = apply_population_view(full, view).clamp_min(0.0).to(scorer.torch.float64)
    flat = rr100.reshape(N_SCORE, N_UNITS, -1)
    rate = flat.mean(dim=2)
    gain = flat / (rate[..., None] + EPS)
    instantaneous_ssi = (gain * scorer.torch.log2(gain + EPS)).mean(dim=2)
    rate_np = rate.detach().cpu().numpy().astype(np.float32)
    ssi_np = instantaneous_ssi.detach().cpu().numpy().astype(np.float32)
    del full, rr100, flat, rate, gain, instantaneous_ssi
    return rate_np, ssi_np


def summarize_timecourses(
    rate: np.ndarray,
    instantaneous_ssi: np.ndarray,
    baseline_rate: np.ndarray,
) -> dict[str, np.ndarray]:
    expected_t = np.asarray(rate, dtype=np.float64) * DT
    numerator = (np.asarray(instantaneous_ssi, dtype=np.float64) * expected_t).sum(axis=0)
    expected = expected_t.sum(axis=0)
    delta = np.asarray(rate, dtype=np.float64) - np.asarray(baseline_rate, dtype=np.float64)
    return {
        "information_numerator_bits_spikes": numerator.astype(np.float32),
        "expected_spikes": expected.astype(np.float32),
        "mean_rate_hz": np.asarray(rate, dtype=np.float64).mean(axis=0).astype(np.float32),
        "movie_ssi_bits_per_spike": (numerator / np.maximum(expected, EPS)).astype(np.float32),
        "temporal_sd_rate_hz": np.asarray(rate, dtype=np.float64).std(axis=0).astype(np.float32),
        "temporal_rms_delta_from_stabilized_hz": np.sqrt(np.mean(delta**2, axis=0)).astype(np.float32),
        "temporal_mean_abs_delta_from_stabilized_hz": np.mean(np.abs(delta), axis=0).astype(np.float32),
    }


def baseline_path(out_dir: Path, image_index: int) -> Path:
    return out_dir / "baselines" / f"image_{int(image_index):03d}.npz"


def moving_path(out_dir: Path, round_index: int, image_index: int) -> Path:
    return out_dir / "moving" / f"round_{int(round_index):03d}" / f"image_{int(image_index):03d}.npz"


def baseline_valid(path: Path, *, image_index: int, request_sha256: str) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                int(data["image_index"].item()) == int(image_index)
                and str(data["request_sha256"].item()) == request_sha256
                and data["rate_timecourse_hz"].shape == (N_SCORE, N_UNITS)
                and data["instantaneous_ssi_bits_per_spike"].shape == (N_SCORE, N_UNITS)
                and all(data[name].shape == (N_UNITS,) for name in SUMMARY_ARRAYS[:5])
            )
    except Exception:
        return False


def moving_valid(
    path: Path,
    *,
    image_index: int,
    round_index: int,
    trace_indices: np.ndarray,
    request_sha256: str,
) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                int(data["image_index"].item()) == int(image_index)
                and int(data["round_index"].item()) == int(round_index)
                and str(data["request_sha256"].item()) == request_sha256
                and np.array_equal(data["trace_index"], np.asarray(trace_indices, dtype=np.int64))
                and data["rate_timecourse_hz"].shape == (len(trace_indices), N_SCORE, N_UNITS)
                and data["instantaneous_ssi_bits_per_spike"].shape == (len(trace_indices), N_SCORE, N_UNITS)
                and all(data[name].shape == (len(trace_indices), N_UNITS) for name in SUMMARY_ARRAYS)
            )
    except Exception:
        return False


def progress_manifest(
    *,
    out_dir: Path,
    request: dict[str, Any],
    request_sha256: str,
    schedule: pd.DataFrame,
    selected: list[int],
    image_indices: np.ndarray,
    status: str,
    started_utc: str,
) -> dict[str, Any]:
    all_rounds = sorted(schedule.round_index.unique().astype(int).tolist())
    completed_by_round: dict[str, int] = {}
    for round_index in all_rounds:
        count = sum(
            moving_path(out_dir, round_index, int(image_index)).exists()
            for image_index in image_indices
        )
        completed_by_round[str(round_index)] = int(count)
    complete_rounds = [
        int(round_index)
        for round_index, count in completed_by_round.items()
        if count == len(image_indices)
    ]
    completed_blocks = int(sum(completed_by_round.values()))
    block_size = int(request["trace_block_size"])
    selected_complete = set(selected).issubset(complete_rounds)
    if status == "selected_rounds_complete" and not selected_complete:
        status = "process_image_shard_complete_waiting_for_other_shards"
    payload = {
        "created_utc": utc_now(),
        "started_utc": started_utc,
        "status": status,
        "request_sha256": request_sha256,
        "active_selected_rounds": selected,
        "complete_balanced_rounds": complete_rounds,
        "completed_atomic_blocks": completed_blocks,
        "total_atomic_blocks": int(len(all_rounds) * len(image_indices)),
        "active_selected_atomic_blocks": int(len(selected) * len(image_indices)),
        "completed_movies": int(completed_blocks * block_size),
        "balanced_analyzable_movies": int(len(complete_rounds) * len(image_indices) * block_size),
        "completed_blocks_by_round": completed_by_round,
        "resume_contract": "atomic image-within-round NPZ; validated complete blocks are skipped",
        "partial_analysis_contract": (
            "Use complete_balanced_rounds only. Every complete round spans all images and every trace exactly once."
        ),
        "request": request,
        "outputs": {
            "schedule": str((out_dir / "balanced_round_schedule.csv").resolve()),
            "request_identity": str((out_dir / "request_identity.json").resolve()),
            "baselines": str((out_dir / "baselines").resolve()),
            "moving": str((out_dir / "moving").resolve()),
        },
    }
    atomic_json(out_dir / "manifest.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    if int(args.image_shard_count) < 1 or not 0 <= int(args.image_shard_index) < int(args.image_shard_count):
        raise ValueError("Invalid image shard count/index")
    images, traces, cohort_manifest, image_path, trace_path, cohort_manifest_path = load_and_validate_cohort(
        args.cohort_dir,
        block_size=int(args.trace_block_size),
        allow_nonproduction=bool(args.allow_nonproduction_cohort),
    )
    schedule = make_balanced_schedule(
        images.image_index.to_numpy(int),
        traces.trace_index.to_numpy(int),
        block_size=int(args.trace_block_size),
    )
    rounds = selected_rounds(args, int(schedule.round_index.max()) + 1)
    if not rounds:
        raise ValueError("No rounds selected")
    runner_path = Path(__file__)
    request = {
        "analysis": "rr100_corrected_explicit_history_production_cache",
        "cohort_status": cohort_manifest.get("status"),
        "cohort": {
            "images": file_identity(image_path),
            "traces": file_identity(trace_path),
            "manifest": file_identity(cohort_manifest_path),
        },
        "runner": file_identity(runner_path),
        "rr100_version": RR100_MOVIE_MEDOID_VERSION,
        "n_images": int(len(images)),
        "n_traces": int(len(traces)),
        "n_units": N_UNITS,
        "history_frames": N_HISTORY,
        "scored_frames": N_SCORE,
        "frame_rate_hz": FRAME_RATE_HZ,
        "trace_block_size": int(args.trace_block_size),
        "n_rounds": int(schedule.round_index.max()) + 1,
        "conditions": ["moving_explicit_recorded_history", "stabilized_zero_relative_translation_explicit_history"],
        "response_alignment": "native T+1 helper output: drop first, retain aligned frames 32:72 (exactly 40)",
        "retinal_motion_sign": "negative corrected dpi_pix crop trajectory",
        "saved_metrics": list(SUMMARY_ARRAYS),
        "saved_timecourses": ["mean spatial rate", "instantaneous spatial SSI"],
    }
    request_sha256 = identity_digest(request)
    if args.dry_run:
        print(
            json.dumps(
                json_ready(
                    {
                        "status": "dry_run_no_model",
                        "request_sha256": request_sha256,
                        "selected_rounds": rounds,
                        "selected_images_for_this_process": images.iloc[
                            int(args.image_shard_index) :: int(args.image_shard_count)
                        ].image_index.astype(int).tolist(),
                        "atomic_blocks": len(rounds)
                        * len(images.iloc[int(args.image_shard_index) :: int(args.image_shard_count)]),
                        "movies": len(rounds)
                        * len(images.iloc[int(args.image_shard_index) :: int(args.image_shard_count)])
                        * int(args.trace_block_size),
                        "request": request,
                    }
                ),
                indent=2,
            )
        )
        return

    out_dir = args.out_dir.resolve()
    if out_dir.exists() and not bool(args.resume):
        raise FileExistsError(f"Output exists and --no-resume was requested: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    identity_path = out_dir / "request_identity.json"
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing.get("request_sha256") != request_sha256:
            raise RuntimeError("Output directory contains a different immutable request identity")
    else:
        atomic_json(identity_path, {"request_sha256": request_sha256, "request": request})
        schedule.to_csv(out_dir / "balanced_round_schedule.csv", index=False)
    started_utc = utc_now()
    process_images = images.iloc[int(args.image_shard_index) :: int(args.image_shard_count)].copy()
    progress_manifest(
        out_dir=out_dir,
        request=request,
        request_sha256=request_sha256,
        schedule=schedule,
        selected=rounds,
        image_indices=images.image_index.to_numpy(int),
        status="prepared_not_scored" if args.prepare_only else "scoring_in_progress",
        started_utc=started_utc,
    )
    history, score = load_or_prepare_trace_segments(out_dir, traces, request_sha256=request_sha256)
    patches, ppds, patch_paths = load_or_prepare_image_patches(
        out_dir,
        images,
        args.image_audit_dir,
        request_sha256=request_sha256,
    )
    patch_by_image = {int(row.image_index): patches[ordinal] for ordinal, row in images.iterrows()}
    ppd_by_image = {int(row.image_index): float(ppds[ordinal]) for ordinal, row in images.iterrows()}
    patch_path_by_image = {int(row.image_index): patch_paths[ordinal] for ordinal, row in images.iterrows()}
    if args.prepare_only:
        print(json.dumps(json_ready({"status": "prepared_not_scored", "out_dir": out_dir}), indent=2))
        return

    view = load_population_view(version_name=RR100_MOVIE_MEDOID_VERSION)
    if int(view.n_units) != N_UNITS:
        raise ValueError(f"Expected RR100 view, got {view.n_units} units")
    mapping = pd.read_csv(MAPPING).sort_values("rr100_index")
    if not np.array_equal(np.argmax(view.membership, axis=1), mapping.canonical_channel.to_numpy(int)):
        raise ValueError("RR100 mapping mismatch")
    scorer = CanonicalTwinScorer(
        device=str(args.device),
        batch_size=int(args.frame_batch_size),
        empty_cache_every_batch=True,
    )
    new_blocks = 0
    start_time = time.perf_counter()
    trace_ordinal = {int(value): ordinal for ordinal, value in enumerate(traces.trace_index.to_numpy(int))}

    for round_index in rounds:
        round_rows = schedule[schedule.round_index.eq(round_index)]
        for _, image_row in process_images.iterrows():
            image_index = int(image_row.image_index)
            block = round_rows[round_rows.image_index.eq(image_index)].sort_values("within_block")
            trace_ids = block.trace_index.to_numpy(np.int64)
            destination = moving_path(out_dir, round_index, image_index)
            if destination.exists():
                if moving_valid(
                    destination,
                    image_index=image_index,
                    round_index=round_index,
                    trace_indices=trace_ids,
                    request_sha256=request_sha256,
                ):
                    continue
                raise RuntimeError(f"Existing moving shard is incomplete or incompatible: {destination}")

            base_file = baseline_path(out_dir, image_index)
            if not base_file.exists():
                zero = np.zeros((N_HISTORY + N_SCORE, 2), dtype=np.float32)
                stim = render_scored_embedding(
                    scorer.common, scorer.torch, patch_by_image[image_index], zero, ppd_by_image[image_index]
                )
                baseline_rate, baseline_ssi = response_timecourses(scorer, view, stim)
                base_summary = summarize_timecourses(baseline_rate, baseline_ssi, baseline_rate)
                atomic_npz(
                    base_file,
                    request_sha256=np.asarray(request_sha256),
                    image_index=np.asarray(image_index, dtype=np.int64),
                    patch_path=np.asarray(patch_path_by_image[image_index]),
                    patch_ppd=np.asarray(ppd_by_image[image_index], dtype=np.float64),
                    rate_timecourse_hz=baseline_rate,
                    instantaneous_ssi_bits_per_spike=baseline_ssi,
                    **{name: base_summary[name] for name in SUMMARY_ARRAYS[:5]},
                )
            elif not baseline_valid(base_file, image_index=image_index, request_sha256=request_sha256):
                raise RuntimeError(f"Existing baseline shard is incomplete or incompatible: {base_file}")
            with np.load(base_file, allow_pickle=False) as data:
                baseline_rate = np.asarray(data["rate_timecourse_hz"], dtype=np.float32)

            block_rates = np.empty((len(trace_ids), N_SCORE, N_UNITS), dtype=np.float32)
            block_ssi = np.empty_like(block_rates)
            block_summary = {
                name: np.empty((len(trace_ids), N_UNITS), dtype=np.float32) for name in SUMMARY_ARRAYS
            }
            for local, trace_index in enumerate(trace_ids):
                ordinal = trace_ordinal[int(trace_index)]
                trace72 = np.concatenate([history[ordinal], score[ordinal]], axis=0)
                stim = render_scored_embedding(
                    scorer.common,
                    scorer.torch,
                    patch_by_image[image_index],
                    trace72,
                    ppd_by_image[image_index],
                )
                rate, instantaneous_ssi = response_timecourses(scorer, view, stim)
                summary = summarize_timecourses(rate, instantaneous_ssi, baseline_rate)
                block_rates[local] = rate
                block_ssi[local] = instantaneous_ssi
                for name in SUMMARY_ARRAYS:
                    block_summary[name][local] = summary[name]
            atomic_npz(
                destination,
                request_sha256=np.asarray(request_sha256),
                round_index=np.asarray(round_index, dtype=np.int64),
                half_index=np.asarray(int(round_index >= request["n_rounds"] // 2), dtype=np.int64),
                image_index=np.asarray(image_index, dtype=np.int64),
                trace_index=trace_ids,
                rate_timecourse_hz=block_rates,
                instantaneous_ssi_bits_per_spike=block_ssi,
                **block_summary,
            )
            new_blocks += 1
            elapsed = time.perf_counter() - start_time
            print(
                f"round {round_index:03d} image {image_index:03d}: wrote {len(trace_ids)} movies; "
                f"new_blocks={new_blocks}; elapsed={elapsed / 60.0:.1f} min",
                flush=True,
            )
            progress_manifest(
                out_dir=out_dir,
                request=request,
                request_sha256=request_sha256,
                schedule=schedule,
                selected=rounds,
                image_indices=images.image_index.to_numpy(int),
                status="scoring_in_progress",
                started_utc=started_utc,
            )
            if int(args.max_new_blocks) > 0 and new_blocks >= int(args.max_new_blocks):
                progress_manifest(
                    out_dir=out_dir,
                    request=request,
                    request_sha256=request_sha256,
                    schedule=schedule,
                    selected=rounds,
                    image_indices=images.image_index.to_numpy(int),
                    status="stopped_at_requested_new_block_limit",
                    started_utc=started_utc,
                )
                return

    final = progress_manifest(
        out_dir=out_dir,
        request=request,
        request_sha256=request_sha256,
        schedule=schedule,
        selected=rounds,
        image_indices=images.image_index.to_numpy(int),
        status="selected_rounds_complete",
        started_utc=started_utc,
    )
    print(json.dumps(json_ready(final), indent=2), flush=True)


if __name__ == "__main__":
    main()
