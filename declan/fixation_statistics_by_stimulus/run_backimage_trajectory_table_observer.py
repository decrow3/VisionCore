#!/usr/bin/env python3
"""Run a BackImage exact trajectory-table image-identity observer."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    from declan.backimage_trajectory_observer.candidate_sets import (
        SUPPORTED_CANDIDATE_SET_MODES,
        build_candidate_set,
    )
    from declan.backimage_trajectory_observer.observer import (
        score_image_identity_table,
        summarize_observer_rows,
    )
    from declan.axis_conditioned_backimage_trajectory_observer.axis_conditioned_traces import (
        SUPPORTED_TEMPLATE_MODES,
        matched_axis_trace_pair,
    )
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px
    from declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information import (
        _build_trace_bank,
        _eligible_trace_bank_indices,
        _family_trace,
        _parse_float_list,
        _parse_str_list,
        _prepare_windows,
        _scale_token,
        _session_dataset_cache,
        _trace_filter_kwargs,
    )
    from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
        CanonicalTwinScorer,
        _align_response_to_trace,
        _static_trace,
    )
    from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import _clip_patch
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.backimage_trajectory_observer.candidate_sets import (
        SUPPORTED_CANDIDATE_SET_MODES,
        build_candidate_set,
    )
    from declan.backimage_trajectory_observer.observer import (
        score_image_identity_table,
        summarize_observer_rows,
    )
    from declan.axis_conditioned_backimage_trajectory_observer.axis_conditioned_traces import (
        SUPPORTED_TEMPLATE_MODES,
        matched_axis_trace_pair,
    )
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px
    from declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information import (
        _build_trace_bank,
        _eligible_trace_bank_indices,
        _family_trace,
        _parse_float_list,
        _parse_str_list,
        _prepare_windows,
        _scale_token,
        _session_dataset_cache,
        _trace_filter_kwargs,
    )
    from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
        CanonicalTwinScorer,
        _align_response_to_trace,
        _static_trace,
    )
    from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import _clip_patch


DEFAULT_INPUT = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
)
DEFAULT_OUT_DIR = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_trajectory_table_observer_smoke"
)
AXIS_CONDITIONED_FAMILIES = {
    "axis_edge_parallel": "parallel",
    "axis_edge_orthogonal": "orthogonal",
}
AXIS_TRAJECTORY_SPEC_KEYS = (
    "axis_conditioned",
    "axis_deg",
    "axis_relation",
    "output_axis_deg",
    "axis_template_mode",
    "template_source",
    "source_rms_displacement_deg",
    "source_path_length_deg",
    "source_max_radius_deg",
    "source_duration_s",
    "rendered_rms_displacement_deg",
    "rendered_path_length_deg",
    "rendered_max_radius_deg",
    "rendered_duration_s",
    "clipping_fraction",
    "degenerate_requested_motion",
    "axis_pair_id",
    "axis_source_id",
    "axis_match_status",
    "axis_match_rms_delta_deg",
    "axis_match_path_delta_deg",
    "axis_match_duration_delta_s",
    "axis_match_clipping_fraction_delta",
    "axis_match_tolerance",
    "axis_match_degenerate",
    "axis_source_column",
    "axis_match_policy",
    "source_table_path_length_deg",
    "target_path_length_deg",
    "axis_catalog_mode",
    "axis_candidate_index",
    "axis_candidate_id",
    "axis_candidate_source_row",
    "axis_candidate_axis_deg",
)


def _progress(message: str) -> None:
    print(f"[backimage-trajectory-table] {message}", flush=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _trace_hash(trace: np.ndarray) -> str:
    arr = np.asarray(trace, dtype=np.float32)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def _child_rng(seed: int, *parts: Any) -> np.random.Generator:
    text = "|".join([str(int(seed)), *(str(part) for part in parts)])
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    child_seed = int.from_bytes(digest[:8], "little", signed=False) % (2**32)
    return np.random.default_rng(child_seed)


def _trace_rmse(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    t = min(aa.shape[0], bb.shape[0])
    if t <= 0:
        return float("nan")
    diff = aa[:t] - bb[:t]
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _counts_from_rates(rates: np.ndarray, bin_seconds: float) -> np.ndarray:
    arr = np.asarray(rates, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("rate table contains non-finite values")
    min_rate = float(np.min(arr)) if arr.size else 0.0
    if min_rate < -1e-7:
        raise ValueError(f"rate table contains negative values; min={min_rate:.6g}")
    arr = np.maximum(arr, 0.0)
    return (arr * float(bin_seconds)).astype(np.float32, copy=False)


def _extract_patch(
    row: pd.Series,
    *,
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]],
    patch_size_px: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    key = (str(row["session"]), int(row["trial_idx"]))
    if key not in canvas_cache:
        canvas_cache[key] = _backimage_canvas(str(row["session"]), int(row["trial_idx"]))
    canvas, ppd, screen_shape = canvas_cache[key]
    center_px = gaze_deg_to_screen_px(
        np.asarray([float(row["mean_x_deg"]), float(row["mean_y_deg"])]),
        ppd=ppd,
        screen_shape=screen_shape,
    )
    patch = _clip_patch(canvas, (float(center_px[0]), float(center_px[1])), int(patch_size_px))
    return patch, {
        "patch_center_x_px": float(center_px[0]),
        "patch_center_y_px": float(center_px[1]),
        "patch_ppd": float(ppd),
    }


def _add_static_response_features(
    work: pd.DataFrame,
    *,
    scorer: CanonicalTwinScorer,
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]],
    patch_size_px: int,
    n_timepoints: int,
    trace_batch_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Append stabilized twin-response features for matched-static candidates."""
    if work.empty:
        return work.copy(), {"n_static_response_feature_windows": 0, "n_static_response_units": 0}
    static_trace = _static_trace(int(n_timepoints))
    rows = []
    for _, row in tqdm(list(work.iterrows()), desc="static response features"):
        patch, _meta = _extract_patch(row, canvas_cache=canvas_cache, patch_size_px=int(patch_size_px))
        response = scorer.responses(patch, [static_trace], trace_batch_size=int(trace_batch_size))[0]
        aligned = _align_response_to_trace(response, int(n_timepoints))
        rows.append(np.mean(aligned, axis=0, dtype=np.float64))
    features = np.stack(rows, axis=0).astype(np.float32, copy=False)
    feature_cols = [f"static_response_unit_{unit_idx:04d}" for unit_idx in range(features.shape[1])]
    feature_frame = pd.DataFrame(features, columns=feature_cols, index=work.index)
    feature_frame["mean_static_response_rate"] = np.mean(features, axis=1)
    out = pd.concat([work.copy(), feature_frame], axis=1)
    return out, {
        "n_static_response_feature_windows": int(features.shape[0]),
        "n_static_response_units": int(features.shape[1]),
        "static_response_feature_definition": "mean over aligned patch_center_static_tau_zero rates per canonical unit",
    }


def _trace_from_item(
    *,
    family: str,
    item: dict[str, Any],
    scale: float,
    rng: np.random.Generator,
    max_rms_deg: float,
    axis_source_column: str = "image_edge_axis_deg",
    axis_template_mode: str = "same_dominant_projection",
    axis_match_policy: str = "strict",
) -> tuple[np.ndarray, dict[str, Any]]:
    if str(family) == "static":
        trace = _static_trace(int(item["trace"].shape[0]))
        meta = {
            "requested_rms_deg": 0.0,
            "effective_rms_deg": 0.0,
            "rms_clipped_high": False,
            "path_length_deg": 0.0,
            "generated_lag1_autocorr": float("nan"),
            "speed_mean_deg_s": 0.0,
            "speed_median_deg_s": 0.0,
            "speed_p95_deg_s": 0.0,
        }
        return trace, meta
    target_rms = float(scale) * float(item["observed_rms_deg"])
    target_path = float(scale) * float(item["path_length_deg"])
    if str(family) in AXIS_CONDITIONED_FAMILIES:
        if str(axis_source_column) not in item:
            raise ValueError(
                f"Axis-conditioned family={family!r} requires {axis_source_column!r} in the trace-bank item"
            )
        edge_axis = float(item[str(axis_source_column)])
        if not np.isfinite(edge_axis):
            raise ValueError(
                f"Axis-conditioned family={family!r} found non-finite {axis_source_column} "
                f"for source_row={item.get('source_row', 'unknown')}"
            )
        pair = matched_axis_trace_pair(
            np.asarray(item["trace"], dtype=np.float32),
            edge_axis_deg=edge_axis,
            template_mode=str(axis_template_mode),
            scale=float(scale),
            target_rms_deg=target_rms,
            max_rms_deg=float(max_rms_deg),
            source_id=item.get("source_row", None),
        )
        relation = AXIS_CONDITIONED_FAMILIES[str(family)]
        trace = np.asarray(pair[relation]["trace"], dtype=np.float32)
        meta = dict(pair[relation]["meta"])
        meta["axis_source_column"] = str(axis_source_column)
        meta["axis_match_policy"] = str(axis_match_policy)
        meta["source_table_path_length_deg"] = float(item.get("path_length_deg", np.nan))
        meta["target_path_length_deg"] = float(target_path)
        if str(axis_match_policy) == "strict" and str(meta.get("axis_match_status", "")) != "matched":
            raise ValueError(
                f"Axis-conditioned family={family!r} produced axis_match_status="
                f"{meta.get('axis_match_status')!r} for source_row={item.get('source_row', 'unknown')}"
            )
        return trace, meta
    return _family_trace(
        str(family),
        np.asarray(item["trace"], dtype=np.float32),
        float(item["lag1_autocorr"]),
        target_rms,
        rng=rng,
        max_rms_deg=float(max_rms_deg),
        source_shape=item.get("covariance_shape"),
        target_path_length=target_path,
    )


def _trajectory_spec(
    *,
    role: str,
    family: str,
    scale: float,
    trace: np.ndarray,
    item: dict[str, Any] | None,
    meta: dict[str, Any],
    sample_index: int,
    is_true: bool,
) -> dict[str, Any]:
    source_row = -1 if item is None else int(item["source_row"])
    identity_id = f"{family}:rel_{_scale_token(scale)}x:src{source_row}:s{int(sample_index)}:{_trace_hash(trace)}"
    trace_id = f"{role}:{identity_id}"
    out: dict[str, Any] = {
        "trajectory_id": trace_id,
        "trajectory_identity_id": identity_id,
        "role": str(role),
        "family": str(family),
        "scale": float(scale),
        "scale_id": f"rel_{_scale_token(scale)}x" if str(family) != "static" else "static",
        "source_row": source_row,
        "source_session": "" if item is None else str(item["session"]),
        "sample_index": int(sample_index),
        "is_true_trajectory": bool(is_true),
        "trace_hash": _trace_hash(trace),
        "requested_rms_deg": float(meta.get("requested_rms_deg", np.nan)),
        "effective_rms_deg": float(meta.get("effective_rms_deg", np.nan)),
        "path_length_deg": float(meta.get("path_length_deg", np.nan)),
        "generated_lag1_autocorr": float(meta.get("generated_lag1_autocorr", np.nan)),
        "speed_mean_deg_s": float(meta.get("speed_mean_deg_s", np.nan)),
        "speed_p95_deg_s": float(meta.get("speed_p95_deg_s", np.nan)),
        "rms_clipped_high": bool(meta.get("rms_clipped_high", False)),
    }
    for key in AXIS_TRAJECTORY_SPEC_KEYS:
        if key not in meta:
            continue
        val = meta[key]
        if isinstance(val, (bool, np.bool_)):
            out[key] = bool(val)
        elif isinstance(val, (int, np.integer)):
            out[key] = int(val)
        elif isinstance(val, (float, np.floating)):
            out[key] = float(val)
        else:
            out[key] = str(val)
    return out


def _prior_trajectories(
    *,
    current_source_row: int,
    observation_family: str,
    observation_scale: float,
    observation_trace: np.ndarray,
    observation_item: dict[str, Any],
    observation_meta: dict[str, Any],
    prior_family: str,
    prior_scale: float,
    trajectory_prior_mode: str,
    n_prior_trajectories: int,
    trace_bank: list[dict[str, Any]],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[dict[str, Any]], int, dict[str, Any]]:
    rejection_meta = {
        "excluded_current_source_row": 0,
        "excluded_exact_trace_hash": 0,
        "excluded_near_duplicate_rmse": 0,
        "near_duplicate_rmse_threshold": float(args.loo_exclude_trace_rmse_deg),
    }
    if str(prior_family) == "static":
        trace = _static_trace(int(args.n_timepoints))
        meta = {
            "requested_rms_deg": 0.0,
            "effective_rms_deg": 0.0,
            "path_length_deg": 0.0,
            "generated_lag1_autocorr": float("nan"),
            "speed_mean_deg_s": 0.0,
            "speed_p95_deg_s": 0.0,
            "rms_clipped_high": False,
        }
        return [trace], [_trajectory_spec(role="prior", family="static", scale=0.0, trace=trace, item=None, meta=meta, sample_index=0, is_true=False)], -1, rejection_meta

    traces: list[np.ndarray] = []
    specs: list[dict[str, Any]] = []
    true_index = -1
    can_include_self = (
        str(trajectory_prior_mode) == "include_self"
        and str(prior_family) == str(observation_family)
        and abs(float(prior_scale) - float(observation_scale)) < 1e-12
    )
    if can_include_self:
        traces.append(np.asarray(observation_trace, dtype=np.float32))
        specs.append(
            _trajectory_spec(
                role="prior",
                family=str(prior_family),
                scale=float(prior_scale),
                trace=observation_trace,
                item=observation_item,
                meta=observation_meta,
                sample_index=0,
                is_true=True,
            )
        )
        true_index = 0

    eligible = _eligible_trace_bank_indices(
        trace_bank,
        current_source_row=int(current_source_row),
        **_trace_filter_kwargs(args),
    )
    rejection_meta["excluded_current_source_row"] = int(
        sum(int(item["source_row"]) == int(current_source_row) for item in trace_bank)
    )
    obs_hash = _trace_hash(observation_trace)
    candidate_pool: list[tuple[int, np.ndarray, dict[str, Any]]] = []
    for idx in eligible:
        item = trace_bank[int(idx)]
        candidate_raw, candidate_meta = _trace_from_item(
            family=str(prior_family),
            item=item,
            scale=float(prior_scale),
            rng=_child_rng(int(args.seed), "prior-candidate", current_source_row, prior_family, prior_scale, int(idx)),
            max_rms_deg=float(args.max_rms_deg),
            axis_source_column=str(args.axis_source_column),
            axis_template_mode=str(args.axis_template_mode),
            axis_match_policy=str(args.axis_match_policy),
        )
        if str(trajectory_prior_mode) == "leave_one_out" and str(prior_family) == str(observation_family):
            if _trace_hash(candidate_raw) == obs_hash:
                rejection_meta["excluded_exact_trace_hash"] += 1
                continue
            if float(args.loo_exclude_trace_rmse_deg) >= 0.0 and _trace_rmse(observation_trace, candidate_raw) <= float(args.loo_exclude_trace_rmse_deg):
                rejection_meta["excluded_near_duplicate_rmse"] += 1
                continue
        candidate_pool.append((int(idx), candidate_raw, candidate_meta))
    if not candidate_pool and len(traces) < int(n_prior_trajectories):
        raise ValueError("No eligible prior trajectories after filtering")
    sample_index = len(traces)
    n_needed = int(n_prior_trajectories) - len(traces)
    if n_needed > len(candidate_pool):
        raise ValueError(
            f"Need {n_needed} retained prior trajectories for family={prior_family!r}, "
            f"but only {len(candidate_pool)} eligible unique trace sources are available"
        )
    sampled_pool_indices = rng.choice(np.arange(len(candidate_pool)), size=n_needed, replace=False) if n_needed > 0 else []
    for pool_index_raw in sampled_pool_indices:
        bank_index, trace, meta = candidate_pool[int(pool_index_raw)]
        item = trace_bank[bank_index]
        traces.append(trace)
        specs.append(
            _trajectory_spec(
                role="prior",
                family=str(prior_family),
                scale=float(prior_scale),
                trace=trace,
                item=item,
                meta=meta,
                sample_index=sample_index,
                is_true=False,
            )
        )
        sample_index += 1
    return traces, specs, true_index, rejection_meta


def _axis_candidate_meta_rows(
    *,
    candidate_indices: list[int],
    candidate_ids: list[str],
    work: pd.DataFrame,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if len(candidate_indices) != len(candidate_ids):
        raise ValueError("candidate_indices and candidate_ids must have the same length")

    rows: list[dict[str, Any]] = []
    for candidate_index, candidate_pos in enumerate(candidate_indices):
        candidate_row = work.iloc[int(candidate_pos)]
        candidate_axis = float(candidate_row[str(args.axis_source_column)])
        if not np.isfinite(candidate_axis):
            raise ValueError(
                f"Non-finite candidate axis for candidate_index={candidate_index}, "
                f"source_row={candidate_row.get('source_row', 'unknown')}"
            )
        rows.append(
            {
                "candidate_index": int(candidate_index),
                "candidate_id": str(candidate_ids[int(candidate_index)]),
                "source_row": int(candidate_row["source_row"]),
                "axis_deg": float(candidate_axis),
            }
        )
    return rows


def _axis_per_candidate_retained_source_indices(
    *,
    current_source_row: int,
    observation_trace: np.ndarray,
    prior_family: str,
    prior_scale: float,
    trace_bank: list[dict[str, Any]],
    candidate_meta_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[int], dict[str, Any]]:
    if str(prior_family) not in AXIS_CONDITIONED_FAMILIES:
        raise ValueError(f"per-candidate axis catalog requires an axis-conditioned family, got {prior_family!r}")
    candidate_source_rows = sorted({int(row["source_row"]) for row in candidate_meta_rows})
    candidate_source_row_set = set(candidate_source_rows)
    rejection_meta = {
        "excluded_current_source_row": int(
            sum(int(item["source_row"]) == int(current_source_row) for item in trace_bank)
        ),
        "excluded_candidate_source_rows": ",".join(str(row) for row in candidate_source_rows),
        "excluded_candidate_source_row_count": int(len(candidate_source_rows)),
        "excluded_candidate_source_row_hits": int(
            sum(int(item["source_row"]) in candidate_source_row_set for item in trace_bank)
        ),
        "excluded_exact_trace_hash": 0,
        "excluded_near_duplicate_rmse": 0,
        "near_duplicate_rmse_threshold": float(args.loo_exclude_trace_rmse_deg),
        "axis_catalog_mode": "per_candidate",
    }
    eligible = _eligible_trace_bank_indices(
        trace_bank,
        current_source_row=int(current_source_row),
        **_trace_filter_kwargs(args),
    )
    obs_hash = _trace_hash(observation_trace)
    retained: list[int] = []
    for bank_index in eligible:
        source_item_base = trace_bank[int(bank_index)]
        if int(source_item_base["source_row"]) in candidate_source_row_set:
            continue
        reject_source = False
        for candidate_meta in candidate_meta_rows:
            source_item = dict(source_item_base)
            source_item[str(args.axis_source_column)] = float(candidate_meta["axis_deg"])
            trace, _meta = _trace_from_item(
                family=str(prior_family),
                item=source_item,
                scale=float(prior_scale),
                rng=_child_rng(
                    int(args.seed),
                    "axis-per-candidate-screen",
                    current_source_row,
                    prior_family,
                    prior_scale,
                    int(candidate_meta["candidate_index"]),
                    int(bank_index),
                ),
                max_rms_deg=float(args.max_rms_deg),
                axis_source_column=str(args.axis_source_column),
                axis_template_mode=str(args.axis_template_mode),
                axis_match_policy=str(args.axis_match_policy),
            )
            if _trace_hash(trace) == obs_hash:
                rejection_meta["excluded_exact_trace_hash"] += 1
                reject_source = True
                break
            if (
                float(args.loo_exclude_trace_rmse_deg) >= 0.0
                and _trace_rmse(observation_trace, trace) <= float(args.loo_exclude_trace_rmse_deg)
            ):
                rejection_meta["excluded_near_duplicate_rmse"] += 1
                reject_source = True
                break
        if not reject_source:
            retained.append(int(bank_index))
    return retained, rejection_meta


def _axis_shared_sampled_source_indices(
    *,
    current_source_row: int,
    observation_trace: np.ndarray,
    prior_families: list[str],
    prior_scale: float,
    n_prior_trajectories: int,
    trace_bank: list[dict[str, Any]],
    candidate_meta_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[int]:
    retained_sets = []
    retained_sizes: dict[str, int] = {}
    for prior_family in prior_families:
        retained, _meta = _axis_per_candidate_retained_source_indices(
            current_source_row=int(current_source_row),
            observation_trace=observation_trace,
            prior_family=str(prior_family),
            prior_scale=float(prior_scale),
            trace_bank=trace_bank,
            candidate_meta_rows=candidate_meta_rows,
            args=args,
        )
        retained_sets.append(set(int(v) for v in retained))
        retained_sizes[str(prior_family)] = int(len(retained))
    shared = sorted(set.intersection(*retained_sets)) if retained_sets else []
    if len(shared) < int(n_prior_trajectories):
        raise ValueError(
            f"Need {int(n_prior_trajectories)} shared retained source traces for per-candidate "
            f"axis families={','.join(str(family) for family in prior_families)}, but only "
            f"{len(shared)} are retained by every family; per-family retained sizes={retained_sizes}"
        )
    sampled = rng.choice(np.asarray(shared, dtype=int), size=int(n_prior_trajectories), replace=False)
    return [int(v) for v in sampled]


def _axis_per_candidate_prior_trajectories(
    *,
    current_source_row: int,
    observation_trace: np.ndarray,
    prior_family: str,
    prior_scale: float,
    n_prior_trajectories: int,
    trace_bank: list[dict[str, Any]],
    candidate_indices: list[int],
    candidate_ids: list[str],
    work: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
    sampled_bank_indices: list[int] | None = None,
) -> tuple[list[list[np.ndarray]], list[list[dict[str, Any]]], int, dict[str, Any]]:
    """Build axis-conditioned catalogs using each candidate patch's own axis."""
    if str(prior_family) not in AXIS_CONDITIONED_FAMILIES:
        raise ValueError(f"per-candidate axis catalog requires an axis-conditioned family, got {prior_family!r}")
    if str(args.trajectory_prior_mode) != "leave_one_out":
        raise ValueError("axis_catalog_mode='per_candidate' currently requires trajectory_prior_mode='leave_one_out'")
    if int(n_prior_trajectories) < 1:
        raise ValueError("n_prior_trajectories must be positive for per-candidate axis catalogs")
    candidate_meta_rows = _axis_candidate_meta_rows(
        candidate_indices=candidate_indices,
        candidate_ids=candidate_ids,
        work=work,
        args=args,
    )
    retained, rejection_meta = _axis_per_candidate_retained_source_indices(
        current_source_row=int(current_source_row),
        observation_trace=observation_trace,
        prior_family=str(prior_family),
        prior_scale=float(prior_scale),
        trace_bank=trace_bank,
        candidate_meta_rows=candidate_meta_rows,
        args=args,
    )

    if sampled_bank_indices is None and len(retained) < int(n_prior_trajectories):
        raise ValueError(
            f"Need {int(n_prior_trajectories)} retained prior trace sources for per-candidate "
            f"family={prior_family!r}, but only {len(retained)} eligible sources are available "
            "after candidate-source and rendered-duplicate exclusion"
        )
    if sampled_bank_indices is None:
        sampled = rng.choice(np.asarray(retained, dtype=int), size=int(n_prior_trajectories), replace=False)
        rejection_meta["axis_shared_source_catalog"] = False
    else:
        sampled = np.asarray([int(v) for v in sampled_bank_indices], dtype=int)
        if len(sampled) != int(n_prior_trajectories):
            raise ValueError(
                f"Expected {int(n_prior_trajectories)} sampled source indices, got {len(sampled)}"
            )
        invalid = sorted(set(int(v) for v in sampled).difference(retained))
        if invalid:
            raise ValueError(
                f"Shared sampled source indices are not retained for family={prior_family!r}: {invalid[:8]}"
            )
        rejection_meta["axis_shared_source_catalog"] = True
        rejection_meta["axis_shared_sampled_source_rows"] = ";".join(
            str(int(trace_bank[int(bank_index)]["source_row"])) for bank_index in sampled
        )
    all_traces: list[list[np.ndarray]] = []
    all_specs: list[list[dict[str, Any]]] = []
    for candidate_meta in candidate_meta_rows:
        candidate_index = int(candidate_meta["candidate_index"])
        candidate_id = str(candidate_meta["candidate_id"])
        candidate_source_row = int(candidate_meta["source_row"])
        candidate_axis = float(candidate_meta["axis_deg"])
        candidate_traces: list[np.ndarray] = []
        candidate_specs: list[dict[str, Any]] = []
        for sample_index, bank_index_raw in enumerate(sampled):
            source_item = dict(trace_bank[int(bank_index_raw)])
            source_item[str(args.axis_source_column)] = candidate_axis
            trace, meta = _trace_from_item(
                family=str(prior_family),
                item=source_item,
                scale=float(prior_scale),
                rng=_child_rng(
                    int(args.seed),
                    "axis-per-candidate",
                    current_source_row,
                    prior_family,
                    prior_scale,
                    int(candidate_index),
                    int(bank_index_raw),
                ),
                max_rms_deg=float(args.max_rms_deg),
                axis_source_column=str(args.axis_source_column),
                axis_template_mode=str(args.axis_template_mode),
                axis_match_policy=str(args.axis_match_policy),
            )
            meta = dict(meta)
            meta.update(
                {
                    "axis_catalog_mode": "per_candidate",
                    "axis_candidate_index": int(candidate_index),
                    "axis_candidate_id": candidate_id,
                    "axis_candidate_source_row": candidate_source_row,
                    "axis_candidate_axis_deg": candidate_axis,
                }
            )
            if "axis_pair_id" in meta:
                meta["axis_pair_id"] = (
                    f"{meta['axis_pair_id']}:candidate-{candidate_id}:"
                    f"candidate-src-{candidate_source_row}:candidate-edge-{candidate_axis:.6f}"
                )
            candidate_traces.append(trace)
            candidate_specs.append(
                _trajectory_spec(
                    role="prior",
                    family=str(prior_family),
                    scale=float(prior_scale),
                    trace=trace,
                    item=source_item,
                    meta=meta,
                    sample_index=int(sample_index),
                    is_true=False,
                )
            )
        all_traces.append(candidate_traces)
        all_specs.append(candidate_specs)
    return all_traces, all_specs, -1, rejection_meta


def _duplicate_trace_count(specs: list[dict[str, Any]]) -> int:
    identities = [str(spec.get("trajectory_identity_id", spec.get("trajectory_id", ""))) for spec in specs]
    return int(len(identities) - len(set(identities)))


def _nested_duplicate_trace_count(specs_by_candidate: list[list[dict[str, Any]]]) -> int:
    duplicate_count = 0
    for specs in specs_by_candidate:
        identities = [str(spec.get("trajectory_identity_id", spec.get("trajectory_id", ""))) for spec in specs]
        duplicate_count += int(len(identities) - len(set(identities)))
    return int(duplicate_count)


def _spec_id_array(specs_by_candidate: list[list[dict[str, Any]]]) -> np.ndarray:
    return np.asarray([[str(spec["trajectory_id"]) for spec in specs] for specs in specs_by_candidate])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--window-manifest", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--n-candidates", type=int, default=4)
    parser.add_argument("--candidate-set-modes", default="hard_negative_structure")
    parser.add_argument("--observation-family", default="empirical")
    parser.add_argument("--prior-families", default="empirical,ou")
    parser.add_argument("--observed-rms-scales", default="0.5")
    parser.add_argument("--trajectory-prior-mode", choices=("include_self", "leave_one_out"), default="leave_one_out")
    parser.add_argument("--n-prior-trajectories", type=int, default=4)
    parser.add_argument("--axis-source-column", default="image_edge_axis_deg")
    parser.add_argument("--axis-template-mode", choices=SUPPORTED_TEMPLATE_MODES, default="same_dominant_projection")
    parser.add_argument("--axis-match-policy", choices=("strict", "allow_invalid"), default="strict")
    parser.add_argument("--axis-catalog-mode", choices=("shared", "per_candidate"), default="shared")
    parser.add_argument("--likelihood-scales", default="1.0")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--bin-seconds", type=float, default=1.0 / 120.0)
    parser.add_argument(
        "--loo-exclude-trace-rmse-deg",
        type=float,
        default=1e-9,
        help="In leave-one-out mode, exclude retained prior traces with RMSE to the observed trace at or below this threshold.",
    )
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--min-patch-image-margin-px", type=float, default=None)
    parser.add_argument("--n-timepoints", type=int, default=40)
    parser.add_argument("--reliable-image-coherence-min", type=float, default=0.20)
    parser.add_argument("--reliable-drift-anisotropy-min", type=float, default=0.20)
    parser.add_argument("--min-duration-s", type=float, default=0.10)
    parser.add_argument("--max-rms-deg", type=float, default=0.12)
    parser.add_argument("--max-trace-source-rms-deg", type=float, default=None)
    parser.add_argument("--max-trace-source-radius-deg", type=float, default=None)
    parser.add_argument("--max-trace-source-path-length-deg", type=float, default=None)
    parser.add_argument("--max-rendered-trace-path-length-deg", type=float, default=None)
    parser.add_argument("--max-source-trace-path-length-deg", type=float, default=None)
    parser.add_argument("--max-trace-source-speed-p95-deg-s", type=float, default=None)
    parser.add_argument("--max-trace-source-microsaccade-events", type=int, default=None)
    parser.add_argument("--microsaccade-speed-threshold-dps", type=float, default=None)
    parser.add_argument("--microsaccade-threshold-z", type=float, default=6.0)
    parser.add_argument("--microsaccade-pad-frames", type=int, default=1)
    parser.add_argument("--twin-batch-size", type=int, default=24)
    parser.add_argument("--twin-trace-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-candidate-random-fallback",
        action="store_true",
        help="Allow matched candidate modes to top up with random distractors. Use only for smoke/debug runs.",
    )
    parser.add_argument("--skip-response-cache", action="store_true")
    return parser


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "response_tables"
    if not bool(args.skip_response_cache):
        cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    work = _prepare_windows(args)
    if work.empty:
        raise ValueError("No BackImage windows survived filtering")
    if int(args.n_candidates) > work.shape[0]:
        raise ValueError(f"--n-candidates={args.n_candidates} exceeds selected windows={work.shape[0]}")

    candidate_modes = _parse_str_list(args.candidate_set_modes)
    invalid_modes = sorted(set(candidate_modes).difference(SUPPORTED_CANDIDATE_SET_MODES))
    if invalid_modes:
        raise ValueError(f"Unknown candidate-set modes: {invalid_modes}")
    prior_families = _parse_str_list(args.prior_families)
    valid_families = {"empirical", "ou", "brownian", "rotated", "static", *AXIS_CONDITIONED_FAMILIES.keys()}
    invalid_families = sorted(({str(args.observation_family)} | set(prior_families)).difference(valid_families))
    if invalid_families:
        raise ValueError(f"Unknown motion families: {invalid_families}")
    requested_families = {str(args.observation_family), *prior_families}
    axis_families_requested = sorted(requested_families.intersection(AXIS_CONDITIONED_FAMILIES))
    if axis_families_requested:
        if str(args.axis_source_column) not in work.columns:
            raise ValueError(
                f"Axis-conditioned families {axis_families_requested} require column "
                f"{args.axis_source_column!r} in the selected BackImage window table"
            )
        axis_vals = pd.to_numeric(work[str(args.axis_source_column)], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(axis_vals).all():
            n_bad = int(np.count_nonzero(~np.isfinite(axis_vals)))
            raise ValueError(
                f"Axis-conditioned families {axis_families_requested} require finite "
                f"{args.axis_source_column!r}; found {n_bad} non-finite selected windows"
            )
    if str(args.axis_catalog_mode) == "per_candidate":
        if any(str(family) not in AXIS_CONDITIONED_FAMILIES for family in prior_families):
            raise ValueError("axis_catalog_mode='per_candidate' requires all prior_families to be axis-conditioned")
        if str(args.trajectory_prior_mode) != "leave_one_out":
            raise ValueError("axis_catalog_mode='per_candidate' requires trajectory_prior_mode='leave_one_out'")
    scales = _parse_float_list(args.observed_rms_scales)
    likelihood_scales = _parse_float_list(args.likelihood_scales)

    metadata = {
        "config": {k: _json_ready(v) for k, v in vars(args).items()},
        "rate_to_count_conversion": "expected_counts = rates * bin_seconds",
        "primary_zero_reference": "patch_center_static_tau_zero",
        "response_cache_schema": "separate prior_lambda_counts, known_lambda_counts, zero_lambda_counts",
    }
    _write_json(out_dir / "run_metadata.json", metadata)
    _progress(
        f"selected {work.shape[0]} windows; candidates={args.n_candidates}; modes={candidate_modes}; "
        f"obs={args.observation_family}; priors={prior_families}; scales={scales}; dry_run={args.dry_run}"
    )

    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    scorer = None if bool(args.dry_run) else CanonicalTwinScorer(device=str(args.device), batch_size=int(args.twin_batch_size))
    if "matched_static_response" in candidate_modes:
        if scorer is None:
            raise ValueError("candidate_set_mode='matched_static_response' requires response scoring; dry-run is not supported")
        _progress("precomputing stabilized static-response features for matched_static_response")
        work, static_feature_meta = _add_static_response_features(
            work,
            scorer=scorer,
            canvas_cache=canvas_cache,
            patch_size_px=int(args.patch_size_px),
            n_timepoints=int(args.n_timepoints),
            trace_batch_size=int(args.twin_trace_batch_size),
        )
        metadata["static_response_candidate_features"] = static_feature_meta
        _write_json(out_dir / "run_metadata.json", metadata)
    work.to_csv(out_dir / "selected_windows.csv", index=False)

    eyepos_by_session = _session_dataset_cache(work["session"].astype(str).to_list())
    trace_bank = _build_trace_bank(
        work,
        eyepos_by_session,
        int(args.n_timepoints),
        microsaccade_speed_threshold_dps=(
            float(args.microsaccade_speed_threshold_dps) if args.microsaccade_speed_threshold_dps is not None else None
        ),
        microsaccade_threshold_z=float(args.microsaccade_threshold_z),
        microsaccade_pad_frames=int(args.microsaccade_pad_frames),
    )
    if str(args.axis_source_column) in work.columns:
        axis_by_source = (
            work[["source_row", str(args.axis_source_column)]]
            .drop_duplicates("source_row")
            .set_index("source_row")[str(args.axis_source_column)]
        )
        for item in trace_bank:
            source_row = int(item["source_row"])
            if source_row in axis_by_source.index:
                item[str(args.axis_source_column)] = float(axis_by_source.loc[source_row])
    source_to_item = {int(item["source_row"]): item for item in trace_bank}
    trace_bank_rows = [
        {
            "trace_bank_index": int(j),
            "source_row": int(item["source_row"]),
            "session": str(item["session"]),
            "observed_rms_deg": float(item["observed_rms_deg"]),
            "path_length_deg": float(item["path_length_deg"]),
            "lag1_autocorr": float(item["lag1_autocorr"]),
            "n_microsaccade_events": int(item["n_microsaccade_events"]),
            str(args.axis_source_column): float(item.get(str(args.axis_source_column), np.nan)),
        }
        for j, item in enumerate(trace_bank)
    ]
    _write_csv(out_dir / "trace_bank.csv", trace_bank_rows)

    candidate_rows: list[dict[str, Any]] = []
    motion_rows: list[dict[str, Any]] = []
    axis_catalog_rows: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    observer_rows: list[dict[str, Any]] = []

    trial_counter = 0
    for obs_pos, row in tqdm(list(work.iterrows()), desc="backimage trajectory observer"):
        true_source_row = int(row["source_row"])
        if true_source_row not in source_to_item:
            raise ValueError(f"Missing trace-bank item for source_row={true_source_row}")
        obs_item = source_to_item[true_source_row]
        for scale in scales:
            obs_rng = _child_rng(int(args.seed), "observation", true_source_row, args.observation_family, float(scale))
            obs_trace, obs_meta = _trace_from_item(
                family=str(args.observation_family),
                item=obs_item,
                scale=float(scale),
                rng=obs_rng,
                max_rms_deg=float(args.max_rms_deg),
                axis_source_column=str(args.axis_source_column),
                axis_template_mode=str(args.axis_template_mode),
                axis_match_policy=str(args.axis_match_policy),
            )
            obs_spec = _trajectory_spec(
                role="observation",
                family=str(args.observation_family),
                scale=float(scale),
                trace=obs_trace,
                item=obs_item,
                meta=obs_meta,
                sample_index=0,
                is_true=True,
            )
            for candidate_mode in candidate_modes:
                current_trial_id = int(trial_counter)
                motion_rows.append({"trial_id": current_trial_id, **obs_spec})
                candidate_rng = _child_rng(int(args.seed), "candidates", true_source_row, float(scale), candidate_mode)
                cand = build_candidate_set(
                    work,
                    int(obs_pos),
                    mode=str(candidate_mode),
                    n_candidates=int(args.n_candidates),
                    rng=candidate_rng,
                    allow_random_fallback=bool(args.allow_candidate_random_fallback),
                )
                static_response_distance = (
                    float(cand["structure_distance_to_nearest_distractor"])
                    if str(candidate_mode) == "matched_static_response"
                    else float("nan")
                )
                candidate_rows.append(
                    {
                        "trial_id": current_trial_id,
                        "observation_source_row": true_source_row,
                        "candidate_set_mode": str(candidate_mode),
                        "candidate_ids": ";".join(cand["candidate_ids"]),
                        "candidate_indices": ";".join(str(v) for v in cand["candidate_indices"]),
                        "n_candidates": int(cand["n_candidates"]),
                        "candidate_duplicate_flag": bool(cand["candidate_duplicate_flag"]),
                        "near_duplicate_flag": bool(cand["near_duplicate_flag"]),
                        "n_matched_distractors": int(cand["n_matched_distractors"]),
                        "n_random_fallback_distractors": int(cand["n_random_fallback_distractors"]),
                        "random_fallback_used": bool(cand["random_fallback_used"]),
                        "contrast_distance_to_nearest_distractor": float(cand["contrast_distance_to_nearest_distractor"]),
                        "structure_distance_to_nearest_distractor": float(cand["structure_distance_to_nearest_distractor"]),
                        "static_response_distance_to_nearest_distractor": static_response_distance,
                        "mean_rate_distance_to_nearest_distractor": float("nan"),
                        "structure_feature_columns": str(cand.get("structure_feature_columns", "")),
                    }
                )
                patches = []
                candidate_patch_meta: list[dict[str, Any]] = []
                if not bool(args.dry_run):
                    for candidate_pos in cand["candidate_indices"]:
                        patch, _meta = _extract_patch(
                            work.iloc[int(candidate_pos)],
                            canvas_cache=canvas_cache,
                            patch_size_px=int(args.patch_size_px),
                        )
                        patches.append(patch)
                        candidate_patch_meta.append(_meta)

                axis_shared_sampled_bank_indices: list[int] | None = None
                axis_per_candidate_families = [
                    str(family)
                    for family in prior_families
                    if str(args.axis_catalog_mode) == "per_candidate"
                    and str(family) in AXIS_CONDITIONED_FAMILIES
                ]
                if len(axis_per_candidate_families) > 1:
                    axis_candidate_meta_rows = _axis_candidate_meta_rows(
                        candidate_indices=[int(v) for v in cand["candidate_indices"]],
                        candidate_ids=[str(v) for v in cand["candidate_ids"]],
                        work=work,
                        args=args,
                    )
                    axis_shared_sampled_bank_indices = _axis_shared_sampled_source_indices(
                        current_source_row=true_source_row,
                        observation_trace=obs_trace,
                        prior_families=axis_per_candidate_families,
                        prior_scale=float(scale),
                        n_prior_trajectories=int(args.n_prior_trajectories),
                        trace_bank=trace_bank,
                        candidate_meta_rows=axis_candidate_meta_rows,
                        args=args,
                        rng=_child_rng(
                            int(args.seed),
                            "axis-shared-prior-source-set",
                            true_source_row,
                            float(scale),
                            candidate_mode,
                            args.trajectory_prior_mode,
                        ),
                    )

                for prior_family in prior_families:
                    prior_rng = _child_rng(
                        int(args.seed),
                        "prior",
                        true_source_row,
                        float(scale),
                        candidate_mode,
                        prior_family,
                        args.trajectory_prior_mode,
                    )
                    axis_per_candidate = (
                        str(args.axis_catalog_mode) == "per_candidate"
                        and str(prior_family) in AXIS_CONDITIONED_FAMILIES
                    )
                    if axis_per_candidate:
                        prior_traces_by_candidate, prior_specs_by_candidate, true_prior_index, prior_rejection_meta = (
                            _axis_per_candidate_prior_trajectories(
                                current_source_row=true_source_row,
                                observation_trace=obs_trace,
                                prior_family=str(prior_family),
                                prior_scale=float(scale),
                                n_prior_trajectories=int(args.n_prior_trajectories),
                                trace_bank=trace_bank,
                                candidate_indices=[int(v) for v in cand["candidate_indices"]],
                                candidate_ids=[str(v) for v in cand["candidate_ids"]],
                                work=work,
                                args=args,
                                rng=prior_rng,
                                sampled_bank_indices=axis_shared_sampled_bank_indices,
                            )
                        )
                        for candidate_index, specs in enumerate(prior_specs_by_candidate):
                            for spec in specs:
                                row_payload = {
                                    "trial_id": current_trial_id,
                                    "candidate_index": int(candidate_index),
                                    "candidate_id": str(cand["candidate_ids"][int(candidate_index)]),
                                    **spec,
                                }
                                motion_rows.append(row_payload)
                                axis_catalog_rows.append(row_payload)
                        true_candidate_index = int(cand["true_candidate_index"])
                        distances = np.asarray(
                            [_trace_rmse(obs_trace, tr) for tr in prior_traces_by_candidate[true_candidate_index]],
                            dtype=np.float64,
                        )
                        prior_duplicate_count = _nested_duplicate_trace_count(prior_specs_by_candidate)
                        n_prior_catalog = int(len(prior_specs_by_candidate[0])) if prior_specs_by_candidate else 0
                    else:
                        prior_traces, prior_specs, true_prior_index, prior_rejection_meta = _prior_trajectories(
                            current_source_row=true_source_row,
                            observation_family=str(args.observation_family),
                            observation_scale=float(scale),
                            observation_trace=obs_trace,
                            observation_item=obs_item,
                            observation_meta=obs_meta,
                            prior_family=str(prior_family),
                            prior_scale=float(scale),
                            trajectory_prior_mode=str(args.trajectory_prior_mode),
                            n_prior_trajectories=int(args.n_prior_trajectories),
                            trace_bank=trace_bank,
                            args=args,
                            rng=prior_rng,
                        )
                        for spec in prior_specs:
                            motion_rows.append({"trial_id": current_trial_id, **spec})
                        distances = np.asarray([_trace_rmse(obs_trace, tr) for tr in prior_traces], dtype=np.float64)
                        prior_duplicate_count = _duplicate_trace_count(prior_specs)
                        n_prior_catalog = int(len(prior_specs))
                    nearest_idx = int(np.nanargmin(distances)) if np.isfinite(distances).any() else -1
                    nearest_dist = float(distances[nearest_idx]) if nearest_idx >= 0 else float("nan")
                    nearest_prior_spec = None
                    if nearest_idx >= 0:
                        if axis_per_candidate:
                            nearest_prior_spec = prior_specs_by_candidate[int(cand["true_candidate_index"])][nearest_idx]
                        else:
                            nearest_prior_spec = prior_specs[nearest_idx]
                    if bool(args.dry_run):
                        cache_rows.append(
                            {
                                "trial_id": current_trial_id,
                                "candidate_set_mode": str(candidate_mode),
                                "observation_family": str(args.observation_family),
                                "prior_family": str(prior_family),
                                "scale": float(scale),
                                "axis_catalog_mode": str(args.axis_catalog_mode) if axis_per_candidate else "shared",
                                "response_cache_path": "",
                                "n_candidates": int(cand["n_candidates"]),
                                "n_prior_trajectories": n_prior_catalog,
                                "n_timebins": int(args.n_timepoints),
                                "n_units": 0,
                                "true_trajectory_index": int(true_prior_index),
                                "nearest_trajectory_index": int(nearest_idx),
                                "nearest_trajectory_distance": float(nearest_dist),
                                "has_prior_trajectory_xy": False,
                                "has_observed_trajectory_xy": False,
                                "prior_duplicate_trajectory_count": int(prior_duplicate_count),
                                **prior_rejection_meta,
                                "dry_run": True,
                            }
                        )
                        continue

                    known_rates = []
                    zero_rates = []
                    prior_rates = []
                    response_frames_before_alignment: list[int] = []
                    static_trace = _static_trace(int(args.n_timepoints))
                    for candidate_index, patch in enumerate(patches):
                        candidate_prior_traces = (
                            prior_traces_by_candidate[int(candidate_index)] if axis_per_candidate else prior_traces
                        )
                        traces = [obs_trace, static_trace, *candidate_prior_traces]
                        responses = scorer.responses(patch, traces, trace_batch_size=int(args.twin_trace_batch_size))
                        response_frames_before_alignment.extend(int(resp.shape[0]) for resp in responses)
                        aligned = [_align_response_to_trace(resp, int(args.n_timepoints)) for resp in responses]
                        known_rates.append(aligned[0])
                        zero_rates.append(aligned[1])
                        prior_rates.append(np.stack(aligned[2:], axis=0))
                    known_counts = _counts_from_rates(np.stack(known_rates, axis=0), float(args.bin_seconds))
                    zero_counts = _counts_from_rates(np.stack(zero_rates, axis=0), float(args.bin_seconds))
                    prior_counts = _counts_from_rates(np.stack(prior_rates, axis=0), float(args.bin_seconds))
                    # The observation remains the true moved response. The
                    # zero-eye baseline only changes the response table used to
                    # explain that same moved observation.
                    y_obs_counts = known_counts[int(cand["true_candidate_index"])]
                    prior_trajectory_xy = (
                        np.asarray(prior_traces_by_candidate, dtype=np.float32)
                        if axis_per_candidate
                        else np.asarray(prior_traces, dtype=np.float32)
                    )

                    cache_rel = ""
                    if not bool(args.skip_response_cache):
                        cache_path = cache_dir / (
                            f"trial_{current_trial_id:05d}_{candidate_mode}_obs-{args.observation_family}_"
                            f"prior-{prior_family}_rel{_scale_token(scale)}x.npz"
                        )
                        np.savez_compressed(
                            cache_path,
                            prior_lambda_counts=prior_counts.astype(np.float32, copy=False),
                            known_lambda_counts=known_counts.astype(np.float32, copy=False),
                            zero_lambda_counts=zero_counts.astype(np.float32, copy=False),
                            y_obs_counts=y_obs_counts.astype(np.float32, copy=False),
                            prior_trajectory_xy=prior_trajectory_xy.astype(np.float32, copy=False),
                            observed_trajectory_xy=np.asarray(obs_trace, dtype=np.float32),
                            candidate_ids=np.asarray(cand["candidate_ids"]),
                            prior_trajectory_ids=(
                                _spec_id_array(prior_specs_by_candidate)
                                if axis_per_candidate
                                else np.asarray([spec["trajectory_id"] for spec in prior_specs])
                            ),
                            true_candidate_index=np.asarray([int(cand["true_candidate_index"])], dtype=np.int32),
                            true_trajectory_index=np.asarray([int(true_prior_index)], dtype=np.int32),
                            nearest_trajectory_index=np.asarray([int(nearest_idx)], dtype=np.int32),
                            nearest_trajectory_distance=np.asarray([float(nearest_dist)], dtype=np.float32),
                            observation_reference_mode=np.asarray(["observed_trace_moved"]),
                            zero_reference_mode=np.asarray(["patch_center_static_tau_zero"]),
                            trajectory_coordinate_schema=np.asarray(
                                ["prior_trajectory_xy is (candidate,trajectory,time,xy) for per_candidate catalogs, else (trajectory,time,xy); observed_trajectory_xy is (time,xy)"]
                            ),
                        )
                        cache_rel = str(cache_path.relative_to(out_dir))
                        cache_rows.append(
                            {
                                "trial_id": current_trial_id,
                                "candidate_set_mode": str(candidate_mode),
                                "observation_family": str(args.observation_family),
                                "prior_family": str(prior_family),
                                "scale": float(scale),
                                "axis_catalog_mode": str(args.axis_catalog_mode) if axis_per_candidate else "shared",
                                "response_cache_path": cache_rel,
                                "n_candidates": int(prior_counts.shape[0]),
                                "n_prior_trajectories": int(prior_counts.shape[1]),
                                "n_timebins": int(prior_counts.shape[2]),
                                "n_units": int(prior_counts.shape[3]),
                                "true_trajectory_index": int(true_prior_index),
                                "nearest_trajectory_index": int(nearest_idx),
                                "nearest_trajectory_distance": float(nearest_dist),
                                "has_prior_trajectory_xy": True,
                                "has_observed_trajectory_xy": True,
                                "prior_duplicate_trajectory_count": int(prior_duplicate_count),
                                "response_frames_before_alignment_min": int(min(response_frames_before_alignment)),
                                "response_frames_before_alignment_max": int(max(response_frames_before_alignment)),
                                "alignment_rule": "identity_if_T_drop_first_if_T_plus_1",
                                "true_patch_center_x_px": float(candidate_patch_meta[int(cand["true_candidate_index"])]["patch_center_x_px"]),
                                "true_patch_center_y_px": float(candidate_patch_meta[int(cand["true_candidate_index"])]["patch_center_y_px"]),
                                "patch_ppd": float(candidate_patch_meta[int(cand["true_candidate_index"])]["patch_ppd"]),
                                **prior_rejection_meta,
                                "dry_run": False,
                            }
                        )

                    for likelihood_scale in likelihood_scales:
                        result = score_image_identity_table(
                            y_obs_counts=y_obs_counts,
                            prior_lambda_counts=prior_counts,
                            known_lambda_counts=known_counts,
                            zero_lambda_counts=zero_counts,
                            true_candidate_index=int(cand["true_candidate_index"]),
                            candidate_ids=cand["candidate_ids"],
                            true_trajectory_index=true_prior_index,
                            nearest_trajectory_index=nearest_idx,
                            nearest_trajectory_distance=nearest_dist,
                            eps=float(args.eps),
                            likelihood_scale=float(likelihood_scale),
                        )
                        observer_rows.append(
                            {
                                "trial_id": current_trial_id,
                                "observation_source_row": true_source_row,
                                "candidate_set_mode": str(candidate_mode),
                                "observation_condition": str(args.observation_family),
                                "observation_family": str(args.observation_family),
                                "observation_scale": float(scale),
                                "prior_condition": str(prior_family),
                                "prior_family": str(prior_family),
                                "prior_scale": float(scale),
                                "axis_catalog_mode": str(args.axis_catalog_mode) if axis_per_candidate else "shared",
                                "trajectory_prior_mode": str(args.trajectory_prior_mode),
                                "zero_reference_mode": "patch_center_static_tau_zero",
                                "bin_seconds": float(args.bin_seconds),
                                "response_cache_path": cache_rel,
                                "true_trajectory_id": str(obs_spec["trajectory_id"]),
                                "nearest_prior_trajectory_id": (
                                    str(nearest_prior_spec["trajectory_id"]) if nearest_prior_spec is not None else ""
                                ),
                                "nearest_prior_trajectory_identity_id": (
                                    str(nearest_prior_spec["trajectory_identity_id"]) if nearest_prior_spec is not None else ""
                                ),
                                "true_trajectory_identity_id": str(obs_spec["trajectory_identity_id"]),
                                "prior_duplicate_trajectory_count": int(prior_duplicate_count),
                                "response_frames_before_alignment_min": int(min(response_frames_before_alignment)),
                                "response_frames_before_alignment_max": int(max(response_frames_before_alignment)),
                                "alignment_rule": "identity_if_T_drop_first_if_T_plus_1",
                                "true_patch_center_x_px": float(candidate_patch_meta[int(cand["true_candidate_index"])]["patch_center_x_px"]),
                                "true_patch_center_y_px": float(candidate_patch_meta[int(cand["true_candidate_index"])]["patch_center_y_px"]),
                                "patch_ppd": float(candidate_patch_meta[int(cand["true_candidate_index"])]["patch_ppd"]),
                                **prior_rejection_meta,
                                "static_response_distance_to_nearest_distractor": static_response_distance,
                                "mean_rate_distance_to_nearest_distractor": float("nan"),
                                "contrast_distance_to_nearest_distractor": float(cand["contrast_distance_to_nearest_distractor"]),
                                "candidate_duplicate_flag": bool(cand["candidate_duplicate_flag"]),
                                "near_duplicate_flag": bool(cand["near_duplicate_flag"]),
                                "n_matched_distractors": int(cand["n_matched_distractors"]),
                                "n_random_fallback_distractors": int(cand["n_random_fallback_distractors"]),
                                "random_fallback_used": bool(cand["random_fallback_used"]),
                                **result,
                            }
                        )
                trial_counter += 1
        done = int(obs_pos) + 1
        if done == 1 or done == work.shape[0] or (int(args.progress_every) > 0 and done % int(args.progress_every) == 0):
            _progress(f"windows {done}/{work.shape[0]}; observer rows={len(observer_rows)}")

    _write_csv(out_dir / "candidate_sets.csv", candidate_rows)
    _write_csv(out_dir / "motion_catalog.csv", motion_rows)
    _write_csv(out_dir / "axis_trajectory_catalog.csv", axis_catalog_rows)
    _write_csv(out_dir / "response_cache_manifest.csv", cache_rows)
    _write_csv(out_dir / "observer_trials.csv", observer_rows)
    _write_csv(out_dir / "observer_summary.csv", summarize_observer_rows(observer_rows))
    report = [
        "# BackImage Trajectory-Table Observer",
        "",
        f"- Selected windows: {work.shape[0]}",
        f"- Candidate modes: {', '.join(candidate_modes)}",
        f"- Observation family: {args.observation_family}",
        f"- Prior families: {', '.join(prior_families)}",
        f"- Axis catalog mode: {args.axis_catalog_mode}",
        f"- Scales: {', '.join(str(v) for v in scales)}",
        f"- Dry run: {bool(args.dry_run)}",
        "",
        "Primary files:",
        "- `selected_windows.csv`",
        "- `candidate_sets.csv`",
        "- `motion_catalog.csv`",
        "- `axis_trajectory_catalog.csv`",
        "- `response_cache_manifest.csv`",
        "- `observer_trials.csv`",
        "- `observer_summary.csv`",
    ]
    (out_dir / "analysis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _progress(f"wrote BackImage trajectory-table observer outputs to {out_dir}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
