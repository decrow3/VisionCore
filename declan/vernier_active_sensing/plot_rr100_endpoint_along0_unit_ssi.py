#!/usr/bin/env python3
"""Diagnose RR100 unit SSI along the endpoint-history along=0 scale line."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.redundancy_resolved_v1_population import load_population_view
from declan.vernier_active_sensing.forward import PKL_PATH, STIMULUS_NORMALIZATION, load_model_and_readout
from declan.vernier_active_sensing.run_endpoint_history_last_frame_readout import (
    ENDPOINT_ALIGNMENT,
    HISTORY_WINDOW,
    endpoint_aligned_trace,
    terminal_history_window,
)
from declan.vernier_active_sensing.run_rr100_endpoint_history_scale_grid import (
    DEFAULT_OUT_DIR as DEFAULT_SCALE_GRID_DIR,
    _terminal_rr100_map,
)
from declan.vernier_active_sensing.run_rr100_real_trace_scale_grid import (
    DEFAULT_SCALES,
    RR100_VERSION,
    canonical_vernier_spec,
    condition_name,
    parse_scales,
    scale_token,
)
from declan.vernier_active_sensing.trajectories import (
    DEFAULT_EYE_TRACES_PATH,
    condition_trace,
    load_eye_traces,
    subsample_traces,
    valid_trace,
)
from scripts.temporal_decoding.stimulus_hires import N_LAGS as MODEL_HISTORY_FRAMES


DEFAULT_OUT_DIR = DEFAULT_SCALE_GRID_DIR / "unit_ssi_along0_diagnostics"
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SCALE_GRID_DIR / "rr100_endpoint_history_scale_grid_summary.csv")
    parser.add_argument("--eye-traces-path", type=Path, default=ROOT / DEFAULT_EYE_TRACES_PATH)
    parser.add_argument("--across-scales", type=str, default=DEFAULT_SCALES)
    parser.add_argument("--along-scale", type=float, default=0.0)
    parser.add_argument("--n-traces", type=int, default=16)
    parser.add_argument("--history-frames", type=int, default=int(MODEL_HISTORY_FRAMES))
    parser.add_argument("--fd-step-arcmin", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, cuda:1, ...")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-units", type=int, default=12)
    parser.add_argument(
        "--skip-highlighted-unit-maps",
        action="store_true",
        help="Only write unit SSI line diagnostics; skip highlighted-unit activation-map sheets.",
    )
    parser.add_argument(
        "--skip-individual-unit-maps",
        action="store_true",
        help="Skip one-PNG-per-highlighted-unit-per-scale activation maps.",
    )
    parser.add_argument("--map-vmin-percentile", type=float, default=0.5)
    parser.add_argument("--map-vmax-percentile", type=float, default=99.5)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--force", action="store_true", help="Recompute cached unit SSI stats.")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


CACHE_SCHEMA_VERSION = 4


def stats_cache_path(out_dir: Path, condition: str, fd_step_arcmin: float) -> Path:
    return out_dir / "cache" / f"rr100_endpoint_along0_unit_ssi_{condition}_fd{float(fd_step_arcmin):.4f}arcmin.npz"


def _cache_identity(args: argparse.Namespace, *, condition: str) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "stimulus_normalization": STIMULUS_NORMALIZATION,
        "condition": str(condition),
        "eye_traces_path": str(Path(args.eye_traces_path).expanduser().resolve()),
        "across_scales": parse_scales(args.across_scales),
        "along_scale": float(args.along_scale),
        "n_traces": int(args.n_traces),
        "history_frames": int(args.history_frames),
        "model_history_frames": int(MODEL_HISTORY_FRAMES),
        "fd_step_arcmin": float(args.fd_step_arcmin),
        "seed": int(args.seed),
        "history_window": HISTORY_WINDOW,
        "endpoint_alignment": ENDPOINT_ALIGNMENT,
        "readout_time_contract": "terminal_response_only",
        "population_version": RR100_VERSION,
        "readout_source_pkl": str(PKL_PATH.expanduser().resolve()),
        "plus_spec": asdict(canonical_vernier_spec(+float(args.fd_step_arcmin))),
        "minus_spec": asdict(canonical_vernier_spec(-float(args.fd_step_arcmin))),
        "map_contract": "trace-mean terminal finite-difference midpoint RR100 spatial maps",
    }


def _identity_text(identity: dict[str, Any]) -> str:
    return json.dumps(json_ready(identity), sort_keys=True, separators=(",", ":"))


def cache_has_required_fields(path: Path, *, require_maps: bool, expected_identity: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    if not require_maps:
        required = {"cache_identity_json"}
    else:
        required = {"cache_identity_json", "mean_rate_map"}
    try:
        with np.load(path) as data:
            if not required.issubset(set(data.files)):
                return False
            return str(np.asarray(data["cache_identity_json"]).ravel()[0]) == _identity_text(expected_identity)
    except Exception:
        return False


def unit_ssi_single_frame(rate_maps: np.ndarray, eps: float = EPS) -> dict[str, np.ndarray | float]:
    """Return unit SSI bits and rates from one ``(unit, H, W)`` map."""
    y = np.asarray(rate_maps, dtype=np.float64)
    if y.ndim != 3:
        raise ValueError(f"Expected (unit, H, W), got {y.shape}")
    flat = y.reshape(y.shape[0], -1)
    unit_mean_rate = flat.mean(axis=1)
    gain = flat / (unit_mean_rate[:, None] + float(eps))
    unit_bits = np.mean(gain * np.log2(gain + float(eps)), axis=1)
    weights = unit_mean_rate / max(float(unit_mean_rate.sum()), float(eps))
    return {
        "unit_bits_per_spike": unit_bits.astype(np.float32),
        "unit_mean_rate": unit_mean_rate.astype(np.float32),
        "population_bits_per_spike": float(np.sum(weights * unit_bits)),
    }


def compute_condition_stats(
    args: argparse.Namespace,
    *,
    condition: str,
    trace_set: Any,
    model: Any,
    readout: Any,
    view: Any,
    device: str,
) -> dict[str, Any]:
    plus_spec = canonical_vernier_spec(+float(args.fd_step_arcmin))
    minus_spec = canonical_vernier_spec(-float(args.fd_step_arcmin))
    unit_bits: list[np.ndarray] = []
    unit_rates: list[np.ndarray] = []
    population_bits: list[float] = []
    map_sum: np.ndarray | None = None
    map_sq_sum: np.ndarray | None = None

    for trace_idx in range(trace_set.traces.shape[0]):
        base_trace = valid_trace(trace_set, trace_idx)
        rng = np.random.default_rng(int(args.seed) + 1009 * trace_idx)
        effective_trace, _trace_meta = condition_trace(
            base_trace,
            condition=condition,
            trace_set=trace_set,
            rng=rng,
        )
        endpoint_trace = endpoint_aligned_trace(terminal_history_window(effective_trace, int(args.history_frames)))
        if endpoint_trace.shape[0] != int(args.history_frames):
            raise RuntimeError(f"Expected {args.history_frames} endpoint frames, got {endpoint_trace.shape[0]}")

        plus_map = _terminal_rr100_map(
            model,
            readout,
            view,
            plus_spec,
            endpoint_trace,
            device=device,
            batch_size=int(args.batch_size),
            history_frames=int(args.history_frames),
        )
        minus_map = _terminal_rr100_map(
            model,
            readout,
            view,
            minus_spec,
            endpoint_trace,
            device=device,
            batch_size=int(args.batch_size),
            history_frames=int(args.history_frames),
        )
        ssi = unit_ssi_single_frame(0.5 * (plus_map[0] + minus_map[0]))
        midpoint_map = np.asarray(0.5 * (plus_map[0] + minus_map[0]), dtype=np.float32)
        if map_sum is None:
            map_sum = np.zeros_like(midpoint_map, dtype=np.float64)
            map_sq_sum = np.zeros_like(midpoint_map, dtype=np.float64)
        map_sum += midpoint_map
        map_sq_sum += np.square(midpoint_map, dtype=np.float64)
        unit_bits.append(np.asarray(ssi["unit_bits_per_spike"], dtype=np.float32))
        unit_rates.append(np.asarray(ssi["unit_mean_rate"], dtype=np.float32))
        population_bits.append(float(ssi["population_bits_per_spike"]))
        print(f"  {condition} trace {trace_idx}: pop SSI={population_bits[-1]:.6g}", flush=True)
        del plus_map, minus_map
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    bits = np.asarray(unit_bits, dtype=np.float32)
    rates = np.asarray(unit_rates, dtype=np.float32)
    pop = np.asarray(population_bits, dtype=np.float32)
    n_traces = max(int(bits.shape[0]), 1)
    if map_sum is None or map_sq_sum is None:
        raise RuntimeError(f"No maps were computed for {condition}.")
    mean_map = (map_sum / float(n_traces)).astype(np.float32)
    second_moment = map_sq_sum / float(n_traces)
    std_map = np.sqrt(np.maximum(second_moment - np.square(mean_map, dtype=np.float64), 0.0)).astype(np.float32)
    return {
        "unit_bits_per_trace": bits,
        "unit_mean_rate_per_trace": rates,
        "population_bits_per_trace": pop,
        "mean_rate_map": mean_map,
        "std_rate_map": std_map,
    }


def load_or_compute_condition_stats(
    args: argparse.Namespace,
    *,
    condition: str,
    trace_set: Any | None,
    model: Any | None,
    readout: Any | None,
    view: Any | None,
    device: str | None,
    require_maps: bool,
) -> dict[str, Any]:
    path = stats_cache_path(Path(args.out_dir), condition, float(args.fd_step_arcmin))
    expected_identity = _cache_identity(args, condition=condition)
    if path.exists() and not bool(args.force):
        with np.load(path) as data:
            has_required_maps = (not require_maps) or "mean_rate_map" in data
            has_matching_identity = (
                "cache_identity_json" in data
                and str(np.asarray(data["cache_identity_json"]).ravel()[0]) == _identity_text(expected_identity)
            )
            if has_required_maps and has_matching_identity:
                print(f"Loaded unit SSI cache: {path}", flush=True)
                out = {
                    "unit_bits_per_trace": np.asarray(data["unit_bits_per_trace"], dtype=np.float32),
                    "unit_mean_rate_per_trace": np.asarray(data["unit_mean_rate_per_trace"], dtype=np.float32),
                    "population_bits_per_trace": np.asarray(data["population_bits_per_trace"], dtype=np.float32),
                }
                if "mean_rate_map" in data:
                    out["mean_rate_map"] = np.asarray(data["mean_rate_map"], dtype=np.float32)
                if "std_rate_map" in data:
                    out["std_rate_map"] = np.asarray(data["std_rate_map"], dtype=np.float32)
                return out
            print(f"Unit SSI cache metadata mismatch or missing maps; recomputing: {path}", flush=True)
    if trace_set is None or model is None or readout is None or view is None or device is None:
        raise RuntimeError("Model/readout/trace resources are required to compute missing caches.")
    print(f"Computing unit SSI cache: {condition}", flush=True)
    stats = compute_condition_stats(
        args,
        condition=condition,
        trace_set=trace_set,
        model=model,
        readout=readout,
        view=view,
        device=device,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        unit_bits_per_trace=stats["unit_bits_per_trace"],
        unit_mean_rate_per_trace=stats["unit_mean_rate_per_trace"],
        population_bits_per_trace=stats["population_bits_per_trace"],
        mean_rate_map=stats["mean_rate_map"],
        std_rate_map=stats["std_rate_map"],
        condition=np.asarray([condition]),
        fd_step_arcmin=np.asarray([float(args.fd_step_arcmin)], dtype=np.float32),
        endpoint_alignment=np.asarray([ENDPOINT_ALIGNMENT]),
        readout_time_contract=np.asarray(["terminal_response_only"]),
        stimulus_normalization=np.asarray([STIMULUS_NORMALIZATION]),
        cache_identity_json=np.asarray([_identity_text(expected_identity)]),
    )
    print(f"Saved unit SSI cache: {path}", flush=True)
    return stats


def _mean_sem(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(arr, axis=axis)
    n = arr.shape[axis]
    sem = np.nanstd(arr, axis=axis, ddof=1) / max(math.sqrt(float(n)), 1.0) if n > 1 else np.zeros_like(mean)
    return mean, sem


def condition_sequence(across_scales: list[float], along_scale: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "condition": "static_center",
            "label": "static center",
            "across_scale": float("nan"),
            "is_static_baseline": True,
        }
    ]
    for across in across_scales:
        rows.append(
            {
                "condition": condition_name(float(across), float(along_scale)),
                "label": f"across {float(across):g}; along {float(along_scale):g}",
                "across_scale": float(across),
                "is_static_baseline": False,
            }
        )
    return rows


def summarize_units(
    stats_by_condition: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    static = stats_by_condition["static_center"]
    static_bits_mean, static_bits_sem = _mean_sem(static["unit_bits_per_trace"], axis=0)
    static_rates_mean, _static_rates_sem = _mean_sem(static["unit_mean_rate_per_trace"], axis=0)
    static_pop_mean = float(np.nanmean(static["population_bits_per_trace"]))
    n_units = int(static_bits_mean.size)

    unit_rows: list[dict[str, Any]] = []
    population_x: list[float] = []
    population_ratio: list[float] = []
    unit_log2_ratio = np.zeros((n_units, sum(not bool(r["is_static_baseline"]) for r in rows)), dtype=np.float64)
    unit_ratio = np.zeros_like(unit_log2_ratio)
    conditions: list[str] = []
    across_values: list[float] = []

    for row in rows:
        condition = str(row["condition"])
        stats = stats_by_condition[condition]
        bits_mean, bits_sem = _mean_sem(stats["unit_bits_per_trace"], axis=0)
        rates_mean, rates_sem = _mean_sem(stats["unit_mean_rate_per_trace"], axis=0)
        pop_mean, pop_sem = _mean_sem(stats["population_bits_per_trace"], axis=0)
        is_static = bool(row["is_static_baseline"])
        if not is_static:
            idx = len(conditions)
            conditions.append(condition)
            across_values.append(float(row["across_scale"]))
            unit_ratio[:, idx] = (bits_mean + EPS) / (static_bits_mean + EPS)
            unit_log2_ratio[:, idx] = np.log2(unit_ratio[:, idx])
            population_x.append(float(row["across_scale"]))
            population_ratio.append(float(pop_mean) / max(static_pop_mean, EPS))

        rates = np.asarray(stats["unit_mean_rate_per_trace"], dtype=np.float64)
        bits = np.asarray(stats["unit_bits_per_trace"], dtype=np.float64)
        total_rate = np.maximum(np.sum(rates, axis=1, keepdims=True), EPS)
        weighted_contrib = (rates / total_rate) * bits
        contrib_mean, contrib_sem = _mean_sem(weighted_contrib, axis=0)

        for unit_index in range(n_units):
            unit_rows.append(
                {
                    "condition": condition,
                    "label": row["label"],
                    "across_scale": row["across_scale"],
                    "along_scale": 0.0,
                    "is_static_baseline": is_static,
                    "unit_index": int(unit_index),
                    "unit_ssi_bits_per_spike_mean": float(bits_mean[unit_index]),
                    "unit_ssi_bits_per_spike_sem": float(bits_sem[unit_index]),
                    "unit_mean_rate_mean": float(rates_mean[unit_index]),
                    "unit_mean_rate_sem": float(rates_sem[unit_index]),
                    "unit_weighted_population_contribution_mean": float(contrib_mean[unit_index]),
                    "unit_weighted_population_contribution_sem": float(contrib_sem[unit_index]),
                    "unit_ssi_vs_static": float((bits_mean[unit_index] + EPS) / (static_bits_mean[unit_index] + EPS)),
                    "unit_log2_ssi_vs_static": float(
                        np.log2((bits_mean[unit_index] + EPS) / (static_bits_mean[unit_index] + EPS))
                    ),
                    "static_unit_ssi_bits_per_spike_mean": float(static_bits_mean[unit_index]),
                    "static_unit_mean_rate_mean": float(static_rates_mean[unit_index]),
                    "population_ssi_bits_per_spike_mean": float(pop_mean),
                    "population_ssi_bits_per_spike_sem": float(pop_sem),
                    "population_ssi_vs_static": float(float(pop_mean) / max(static_pop_mean, EPS)),
                    "static_population_ssi_bits_per_spike_mean": static_pop_mean,
                }
            )

    unit_df = pd.DataFrame(unit_rows)
    unit_max_abs_log2 = np.nanmax(np.abs(unit_log2_ratio), axis=1)
    unit_df["unit_max_abs_log2_ssi_vs_static_along0"] = unit_df["unit_index"].map(
        {int(u): float(unit_max_abs_log2[int(u)]) for u in range(n_units)}
    )

    loo_ratio, loo_delta = leave_one_unit_out_ratios(stats_by_condition, conditions, np.asarray(population_ratio))
    top_rows: list[dict[str, Any]] = []
    for unit_index in range(n_units):
        top_rows.append(
            {
                "unit_index": int(unit_index),
                "max_abs_log2_unit_ssi_vs_static": float(unit_max_abs_log2[unit_index]),
                "max_abs_leave_one_out_population_ratio_delta": float(np.nanmax(np.abs(loo_delta[unit_index]))),
                "static_unit_ssi_bits_per_spike_mean": float(static_bits_mean[unit_index]),
                "static_unit_mean_rate_mean": float(static_rates_mean[unit_index]),
            }
        )
    top_df = pd.DataFrame(top_rows).sort_values(
        ["max_abs_leave_one_out_population_ratio_delta", "max_abs_log2_unit_ssi_vs_static"],
        ascending=[False, False],
        kind="mergesort",
    )
    diagnostics = {
        "n_units": n_units,
        "conditions": conditions,
        "across_values": np.asarray(across_values, dtype=np.float64),
        "population_ratio": np.asarray(population_ratio, dtype=np.float64),
        "unit_ratio": unit_ratio,
        "unit_log2_ratio": unit_log2_ratio,
        "unit_max_abs_log2": unit_max_abs_log2,
        "leave_one_out_ratio": loo_ratio,
        "leave_one_out_delta": loo_delta,
        "static_population_ssi_bits_per_spike_mean": static_pop_mean,
    }
    return unit_df, top_df, diagnostics


def leave_one_unit_out_ratios(
    stats_by_condition: dict[str, dict[str, Any]],
    conditions: list[str],
    full_population_ratio: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    static_loo = leave_one_unit_out_population(stats_by_condition["static_center"])
    n_units = int(static_loo.size)
    loo_ratio = np.zeros((n_units, len(conditions)), dtype=np.float64)
    for idx, condition in enumerate(conditions):
        cond_loo = leave_one_unit_out_population(stats_by_condition[condition])
        loo_ratio[:, idx] = (cond_loo + EPS) / (static_loo + EPS)
    return loo_ratio, loo_ratio - np.asarray(full_population_ratio, dtype=np.float64)[None, :]


def leave_one_unit_out_population(stats: dict[str, Any]) -> np.ndarray:
    rates = np.asarray(stats["unit_mean_rate_per_trace"], dtype=np.float64)
    bits = np.asarray(stats["unit_bits_per_trace"], dtype=np.float64)
    total_rate = np.sum(rates, axis=1, keepdims=True)
    total_numer = np.sum(rates * bits, axis=1, keepdims=True)
    loo_rate = np.maximum(total_rate - rates, EPS)
    loo_bits = (total_numer - rates * bits) / loo_rate
    return np.nanmean(loo_bits, axis=0)


def compare_to_summary(summary_csv: Path, diagnostics: dict[str, Any]) -> dict[str, Any] | None:
    if not summary_csv.exists():
        return None
    summary = pd.read_csv(summary_csv)
    pop = []
    for condition in diagnostics["conditions"]:
        row = summary[summary["condition"].eq(condition)]
        pop.append(float(row.iloc[0]["ssi_bits_per_spike_vs_static"]) if not row.empty else np.nan)
    expected = np.asarray(pop, dtype=np.float64)
    observed = np.asarray(diagnostics["population_ratio"], dtype=np.float64)
    return {
        "summary_csv": str(summary_csv),
        "max_abs_population_ratio_delta": float(np.nanmax(np.abs(observed - expected))),
        "summary_population_ratio": expected,
        "recomputed_population_ratio": observed,
    }


def highlighted_unit_color_map(highlighted_units: list[int]) -> dict[int, Any]:
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(len(highlighted_units), 1)))
    return {int(unit_index): colors[pos] for pos, unit_index in enumerate(highlighted_units)}


def order_units_by_y_at_x(
    diagnostics: dict[str, Any],
    highlighted_units: list[int],
    *,
    x_value: float = 1.0,
) -> list[int]:
    x = np.asarray(diagnostics["across_values"], dtype=np.float64)
    if x.size == 0:
        return [int(unit_index) for unit_index in highlighted_units]
    idx = int(np.nanargmin(np.abs(x - float(x_value))))
    unit_log2 = np.asarray(diagnostics["unit_log2_ratio"], dtype=np.float64)
    return sorted(
        [int(unit_index) for unit_index in highlighted_units],
        key=lambda unit_index: (-float(unit_log2[int(unit_index), idx]), int(unit_index)),
    )


def draw_unit_lines(
    diagnostics: dict[str, Any],
    highlighted_units: list[int],
    path: Path,
    dpi: int,
    *,
    highlight_note: str,
    figure_title: str = "RR100 endpoint-history unit SSI along the along=0 scale line",
) -> None:
    x = np.asarray(diagnostics["across_values"], dtype=np.float64)
    unit_log2 = np.asarray(diagnostics["unit_log2_ratio"], dtype=np.float64)
    population_log2 = np.log2(np.asarray(diagnostics["population_ratio"], dtype=np.float64))
    colors = highlighted_unit_color_map(highlighted_units)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=int(dpi), constrained_layout=True)
    ax = axes[0]
    for unit_index in range(unit_log2.shape[0]):
        ax.plot(x, unit_log2[unit_index], color="#a8a8a8", linewidth=0.65, alpha=0.35, zorder=1)
    for unit_index in highlighted_units:
        ax.plot(
            x,
            unit_log2[int(unit_index)],
            marker="o",
            linewidth=1.35,
            markersize=3.2,
            color=colors[int(unit_index)],
            label=f"u{int(unit_index):03d}",
            zorder=3,
        )
    ax.plot(x, population_log2, color="black", marker="o", linewidth=2.0, markersize=4.0, label="population", zorder=4)
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
    ax.axvline(1.0, color="#999999", linestyle=":", linewidth=0.9)
    ax.set_xlabel("across-contour motion scale, along=0")
    ax.set_ylabel("log2 SSI / own static SSI")
    ax.set_title(f"All units; {highlight_note}")
    ax.grid(True, axis="y", color="#e4e4e4", linewidth=0.7)
    ax.legend(frameon=False, fontsize=6.0, ncols=2, loc="best")

    ax = axes[1]
    for unit_index in highlighted_units:
        ax.plot(
            x,
            unit_log2[int(unit_index)],
            marker="o",
            linewidth=1.5,
            markersize=3.4,
            color=colors[int(unit_index)],
            label=f"u{int(unit_index):03d}",
        )
    ax.plot(x, population_log2, color="black", marker="o", linewidth=2.1, markersize=4.0, label="population")
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
    ax.axvline(1.0, color="#999999", linestyle=":", linewidth=0.9)
    ax.set_xlabel("across-contour motion scale, along=0")
    ax.set_ylabel("log2 SSI / own static SSI")
    ax.set_title("Highlighted units only")
    ax.grid(True, axis="y", color="#e4e4e4", linewidth=0.7)
    ax.legend(frameon=False, fontsize=6.2, ncols=2, loc="best")
    fig.suptitle(figure_title, fontsize=12.0)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_unit_lines_on_axes(
    *,
    axes: np.ndarray,
    diagnostics: dict[str, Any],
    highlighted_units: list[int],
    color_by_unit: dict[int, Any],
    highlight_note: str,
) -> None:
    x = np.asarray(diagnostics["across_values"], dtype=np.float64)
    unit_log2 = np.asarray(diagnostics["unit_log2_ratio"], dtype=np.float64)
    population_log2 = np.log2(np.asarray(diagnostics["population_ratio"], dtype=np.float64))

    ax = axes[0]
    for unit_index in range(unit_log2.shape[0]):
        ax.plot(x, unit_log2[unit_index], color="#a8a8a8", linewidth=0.65, alpha=0.35, zorder=1)
    for unit_index in highlighted_units:
        ax.plot(
            x,
            unit_log2[int(unit_index)],
            marker="o",
            linewidth=1.35,
            markersize=3.2,
            color=color_by_unit[int(unit_index)],
            label=f"u{int(unit_index):03d}",
            zorder=3,
        )
    ax.plot(x, population_log2, color="black", marker="o", linewidth=2.0, markersize=4.0, label="population", zorder=4)
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
    ax.axvline(1.0, color="#999999", linestyle=":", linewidth=0.9)
    ax.set_xlabel("across-contour motion scale, along=0")
    ax.set_ylabel("log2 SSI / own static SSI")
    ax.set_title(f"All units; {highlight_note}")
    ax.grid(True, axis="y", color="#e4e4e4", linewidth=0.7)
    ax.legend(frameon=False, fontsize=6.0, ncols=2, loc="best")

    ax = axes[1]
    for unit_index in highlighted_units:
        ax.plot(
            x,
            unit_log2[int(unit_index)],
            marker="o",
            linewidth=1.5,
            markersize=3.4,
            color=color_by_unit[int(unit_index)],
            label=f"u{int(unit_index):03d}",
        )
    ax.plot(x, population_log2, color="black", marker="o", linewidth=2.1, markersize=4.0, label="population")
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
    ax.axvline(1.0, color="#999999", linestyle=":", linewidth=0.9)
    ax.set_xlabel("across-contour motion scale, along=0")
    ax.set_ylabel("log2 SSI / own static SSI")
    ax.set_title("Highlighted units only")
    ax.grid(True, axis="y", color="#e4e4e4", linewidth=0.7)
    ax.legend(frameon=False, fontsize=6.2, ncols=2, loc="best")


def draw_unit_lines_with_activation_rows(
    *,
    diagnostics: dict[str, Any],
    highlighted_units: list[int],
    rows: list[dict[str, Any]],
    stats_by_condition: dict[str, dict[str, Any]],
    path: Path,
    dpi: int,
    highlight_note: str,
    map_vmin_percentile: float,
    map_vmax_percentile: float,
    figure_title: str = "RR100 endpoint-history unit SSI along the along=0 scale line",
    figure_subtitle: str = "activation maps below are row-normalized and use the same unit colors as the legend",
) -> None:
    plot_rows = [row for row in rows if not bool(row["is_static_baseline"])]
    if len(plot_rows) != len(np.asarray(diagnostics["across_values"]).ravel()):
        raise RuntimeError("Activation-map columns no longer match plotted along=0 x-points.")
    color_by_unit = highlighted_unit_color_map(highlighted_units)
    n_units = len(highlighted_units)
    n_cols = len(plot_rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig_width = max(12.4, 1.13 * (n_cols + 1))
    fig_height = 4.7 + 0.72 * max(n_units, 1)
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=int(dpi))
    outer = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[3.8, max(0.62 * max(n_units, 1), 1.4)],
        hspace=0.16,
    )

    top_grid = outer[0].subgridspec(1, 2, wspace=0.12)
    line_axes = np.asarray([fig.add_subplot(top_grid[0, 0]), fig.add_subplot(top_grid[0, 1])])
    _draw_unit_lines_on_axes(
        axes=line_axes,
        diagnostics=diagnostics,
        highlighted_units=highlighted_units,
        color_by_unit=color_by_unit,
        highlight_note=highlight_note,
    )

    map_grid = outer[1].subgridspec(
        nrows=n_units + 1,
        ncols=n_cols + 1,
        height_ratios=[0.34, *([1.0] * n_units)],
        width_ratios=[0.82, *([1.0] * n_cols)],
        hspace=0.045,
        wspace=0.035,
    )
    label_header_ax = fig.add_subplot(map_grid[0, 0])
    label_header_ax.axis("off")
    label_header_ax.text(
        0.98,
        0.42,
        "unit",
        ha="right",
        va="center",
        fontsize=6.7,
        color="#555555",
    )
    for col_idx, row in enumerate(plot_rows):
        ax = fig.add_subplot(map_grid[0, col_idx + 1])
        ax.axis("off")
        label = _scale_label(row)
        is_oracle_scale = np.isclose(float(row["across_scale"]), 1.0)
        ax.text(
            0.5,
            0.42,
            label,
            ha="center",
            va="center",
            fontsize=6.7,
            fontweight="bold" if is_oracle_scale else "normal",
            color="#333333",
        )

    for unit_pos, unit_index in enumerate(highlighted_units):
        unit_color = color_by_unit[int(unit_index)]
        images = [
            np.asarray(stats_by_condition[str(row["condition"])]["mean_rate_map"], dtype=np.float32)[int(unit_index)]
            for row in plot_rows
        ]
        vmin, vmax = image_scale(images, float(map_vmin_percentile), float(map_vmax_percentile))
        label_ax = fig.add_subplot(map_grid[unit_pos + 1, 0])
        label_ax.axis("off")
        label_ax.text(
            0.98,
            0.5,
            f"u{int(unit_index):03d}",
            ha="right",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color=unit_color,
            transform=label_ax.transAxes,
        )
        label_ax.plot(
            [0.18, 0.92],
            [0.2, 0.2],
            color=unit_color,
            linewidth=2.4,
            solid_capstyle="round",
            transform=label_ax.transAxes,
        )
        for col_idx, image in enumerate(images):
            ax = fig.add_subplot(map_grid[unit_pos + 1, col_idx + 1])
            ax.imshow(image, origin="lower", cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine_name, spine in ax.spines.items():
                spine.set_linewidth(0.55 if spine_name != "left" else 1.15)
                spine.set_color(unit_color if spine_name == "left" else "#686868")

    fig.suptitle(
        f"{figure_title}\n{figure_subtitle}",
        fontsize=11.4,
        y=0.99,
    )
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.035, top=0.925)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_leave_one_out(
    diagnostics: dict[str, Any],
    top_units_by_influence: list[int],
    top_df: pd.DataFrame,
    path: Path,
    dpi: int,
    *,
    figure_title: str = "Does any single RR100 unit drive the endpoint along=0 SSI line?",
) -> None:
    x = np.asarray(diagnostics["across_values"], dtype=np.float64)
    full_ratio = np.asarray(diagnostics["population_ratio"], dtype=np.float64)
    loo_ratio = np.asarray(diagnostics["leave_one_out_ratio"], dtype=np.float64)
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(len(top_units_by_influence), 1)))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=int(dpi), constrained_layout=True)
    ax = axes[0]
    for color, unit_index in zip(colors, top_units_by_influence):
        ax.plot(
            x,
            loo_ratio[int(unit_index)],
            marker="o",
            linewidth=1.35,
            markersize=3.2,
            color=color,
            label=f"remove u{int(unit_index):03d}",
        )
    ax.plot(x, full_ratio, color="black", marker="o", linewidth=2.2, markersize=4.2, label="full RR100")
    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=0.8)
    ax.axvline(1.0, color="#999999", linestyle=":", linewidth=0.9)
    ax.set_xlabel("across-contour motion scale, along=0")
    ax.set_ylabel("population SSI / static after unit removal")
    ax.set_title("Leave-one-unit-out population curves")
    ax.grid(True, axis="y", color="#e4e4e4", linewidth=0.7)
    ax.legend(frameon=False, fontsize=6.2, ncols=2, loc="best")

    bar = top_df.head(len(top_units_by_influence)).copy()
    bar = bar.iloc[::-1]
    ax = axes[1]
    ax.barh(
        [f"u{int(u):03d}" for u in bar["unit_index"]],
        bar["max_abs_leave_one_out_population_ratio_delta"],
        color="#2f6f8f",
        alpha=0.9,
    )
    ax.set_xlabel("max |leave-one-out ratio - full ratio|")
    ax.set_title("Largest population-curve influence")
    ax.grid(True, axis="x", color="#e4e4e4", linewidth=0.7)
    fig.suptitle(figure_title, fontsize=12.0)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def image_scale(images: list[np.ndarray], vmin_percentile: float, vmax_percentile: float) -> tuple[float, float]:
    finite = np.concatenate([np.asarray(img, dtype=np.float32).ravel() for img in images])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, float(vmin_percentile)))
    vmax = float(np.nanpercentile(finite, float(vmax_percentile)))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _scale_label(row: dict[str, Any]) -> str:
    if bool(row["is_static_baseline"]):
        return "static"
    return f"{float(row['across_scale']):g}x"


def draw_unit_map_sheet(
    *,
    unit_index: int,
    rows: list[dict[str, Any]],
    stats_by_condition: dict[str, dict[str, Any]],
    unit_df: pd.DataFrame,
    vmin: float,
    vmax: float,
    path: Path,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_cols = len(rows)
    fig, axes = plt.subplots(1, n_cols, figsize=(1.65 * n_cols, 2.2), dpi=int(dpi), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    last_im = None
    for pos, row in enumerate(rows):
        condition = str(row["condition"])
        unit_map = np.asarray(stats_by_condition[condition]["mean_rate_map"], dtype=np.float32)[int(unit_index)]
        meta = unit_df[(unit_df["unit_index"].eq(int(unit_index))) & (unit_df["condition"].eq(condition))]
        if meta.empty:
            ssi_text = ""
        else:
            record = meta.iloc[0]
            ssi_text = f"SSI {float(record['unit_ssi_bits_per_spike_mean']):.4f}\nratio {float(record['unit_ssi_vs_static']):.2f}x"
        ax = axes_arr[pos]
        last_im = ax.imshow(unit_map, origin="lower", cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(f"{_scale_label(row)}\n{ssi_text}", fontsize=6.0, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.45)
            spine.set_color("#777777")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes_arr.tolist(), fraction=0.018, pad=0.01)
        cbar.ax.tick_params(labelsize=6.0, length=2)
        cbar.set_label("terminal activation", fontsize=6.5)
    fig.suptitle(
        f"RR100 u{int(unit_index):03d}: endpoint along=0 finite-difference midpoint maps",
        fontsize=10.0,
        y=1.08,
    )
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_individual_unit_map(
    *,
    unit_index: int,
    row: dict[str, Any],
    stats_by_condition: dict[str, dict[str, Any]],
    unit_df: pd.DataFrame,
    vmin: float,
    vmax: float,
    path: Path,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    condition = str(row["condition"])
    unit_map = np.asarray(stats_by_condition[condition]["mean_rate_map"], dtype=np.float32)[int(unit_index)]
    meta = unit_df[(unit_df["unit_index"].eq(int(unit_index))) & (unit_df["condition"].eq(condition))]
    title = f"RR100 u{int(unit_index):03d}  {_scale_label(row)}"
    subtitle = ""
    if not meta.empty:
        record = meta.iloc[0]
        subtitle = (
            f"SSI {float(record['unit_ssi_bits_per_spike_mean']):.5f} bits/spike  "
            f"ratio {float(record['unit_ssi_vs_static']):.3f}x"
        )
    fig, ax = plt.subplots(figsize=(2.35, 2.65), dpi=int(dpi), constrained_layout=True)
    im = ax.imshow(unit_map, origin="lower", cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(f"{title}\n{subtitle}", fontsize=7.2, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.025)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_highlighted_unit_maps(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    stats_by_condition: dict[str, dict[str, Any]],
    unit_df: pd.DataFrame,
    highlighted_units: list[int],
) -> tuple[Path, list[dict[str, Any]]]:
    map_root = Path(args.out_dir) / "highlighted_unit_activation_maps"
    sheet_dir = map_root / "unit_sheets"
    individual_dir = map_root / "individual"
    manifest_rows: list[dict[str, Any]] = []

    for unit_index in highlighted_units:
        images = [
            np.asarray(stats_by_condition[str(row["condition"])]["mean_rate_map"], dtype=np.float32)[int(unit_index)]
            for row in rows
        ]
        vmin, vmax = image_scale(images, float(args.map_vmin_percentile), float(args.map_vmax_percentile))
        sheet_path = sheet_dir / f"rr100_endpoint_along0_unit_{int(unit_index):03d}_activation_maps.png"
        draw_unit_map_sheet(
            unit_index=int(unit_index),
            rows=rows,
            stats_by_condition=stats_by_condition,
            unit_df=unit_df,
            vmin=vmin,
            vmax=vmax,
            path=sheet_path,
            dpi=int(args.dpi),
        )
        for row in rows:
            condition = str(row["condition"])
            meta = unit_df[(unit_df["unit_index"].eq(int(unit_index))) & (unit_df["condition"].eq(condition))]
            individual_path = ""
            if not bool(args.skip_individual_unit_maps):
                scale_slug = "static" if bool(row["is_static_baseline"]) else f"across_{scale_token(float(row['across_scale']))}"
                individual_png = (
                    individual_dir
                    / f"unit_{int(unit_index):03d}"
                    / f"rr100_endpoint_along0_unit_{int(unit_index):03d}_{scale_slug}_activation_map.png"
                )
                draw_individual_unit_map(
                    unit_index=int(unit_index),
                    row=row,
                    stats_by_condition=stats_by_condition,
                    unit_df=unit_df,
                    vmin=vmin,
                    vmax=vmax,
                    path=individual_png,
                    dpi=int(args.dpi),
                )
                individual_path = str(individual_png)
            record = meta.iloc[0].to_dict() if not meta.empty else {}
            manifest_rows.append(
                {
                    "unit_index": int(unit_index),
                    "condition": condition,
                    "across_scale": row["across_scale"],
                    "along_scale": float(args.along_scale),
                    "is_static_baseline": bool(row["is_static_baseline"]),
                    "map_source": "trace_mean_terminal_fd_midpoint_map",
                    "n_traces": int(np.asarray(stats_by_condition[condition]["unit_bits_per_trace"]).shape[0]),
                    "unit_ssi_bits_per_spike_mean": record.get("unit_ssi_bits_per_spike_mean", np.nan),
                    "unit_ssi_vs_static": record.get("unit_ssi_vs_static", np.nan),
                    "unit_log2_ssi_vs_static": record.get("unit_log2_ssi_vs_static", np.nan),
                    "unit_mean_rate_mean": record.get("unit_mean_rate_mean", np.nan),
                    "unit_sheet_png": str(sheet_path),
                    "individual_png": individual_path,
                    "unit_sheet_colormap": "gray_monotonic",
                    "unit_sheet_vmin": float(vmin),
                    "unit_sheet_vmax": float(vmax),
                }
            )

    manifest_csv = map_root / "rr100_endpoint_along0_highlighted_unit_map_manifest.csv"
    write_csv_rows(manifest_csv, manifest_rows)
    return manifest_csv, manifest_rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    across_scales = parse_scales(args.across_scales)
    rows = condition_sequence(across_scales, float(args.along_scale))
    conditions = [str(row["condition"]) for row in rows]
    require_maps = not bool(args.skip_highlighted_unit_maps)

    missing = [
        condition
        for condition in conditions
        if bool(args.force)
        or not cache_has_required_fields(
            stats_cache_path(Path(args.out_dir), condition, float(args.fd_step_arcmin)),
            require_maps=require_maps,
            expected_identity=_cache_identity(args, condition=condition),
        )
    ]
    trace_set = model = readout = view = None
    device: str | None = None
    if missing:
        print(f"Need to compute {len(missing)} unit SSI caches.", flush=True)
        trace_set = subsample_traces(load_eye_traces(Path(args.eye_traces_path)), int(args.n_traces), int(args.seed))
        device_arg = None if str(args.device).lower() == "auto" else str(args.device)
        print("Loading model/readout...", flush=True)
        model, readout = load_model_and_readout(device=device_arg)
        device = str(next(model.model.parameters()).device)
        print(f"Model device: {device}", flush=True)
        view = load_population_view(version_name=RR100_VERSION)
        print(f"Population view: {view.name}; n_units={int(view.n_units)}", flush=True)

    stats_by_condition = {
        condition: load_or_compute_condition_stats(
            args,
            condition=condition,
            trace_set=trace_set,
            model=model,
            readout=readout,
            view=view,
            device=device,
            require_maps=require_maps,
        )
        for condition in conditions
    }

    unit_df, top_df, diagnostics = summarize_units(stats_by_condition, rows)
    comparison = compare_to_summary(Path(args.summary_csv), diagnostics)
    if comparison is not None:
        print(
            "Summary cross-check max abs SSI ratio delta: "
            f"{comparison['max_abs_population_ratio_delta']:.6g}",
            flush=True,
        )

    unit_csv = Path(args.out_dir) / "rr100_endpoint_along0_unit_ssi_table.csv"
    top_csv = Path(args.out_dir) / "rr100_endpoint_along0_unit_ssi_top_units.csv"
    unit_df.to_csv(unit_csv, index=False)
    top_df.to_csv(top_csv, index=False)

    top_n = max(1, int(args.top_units))
    top_by_unit = (
        unit_df[["unit_index", "unit_max_abs_log2_ssi_vs_static_along0"]]
        .drop_duplicates()
        .sort_values("unit_max_abs_log2_ssi_vs_static_along0", ascending=False, kind="mergesort")
        .head(top_n)["unit_index"]
        .astype(int)
        .tolist()
    )
    top_by_influence = top_df.head(top_n)["unit_index"].astype(int).tolist()
    top_by_unit_for_plot = order_units_by_y_at_x(diagnostics, top_by_unit, x_value=1.0)
    top_by_influence_for_plot = order_units_by_y_at_x(diagnostics, top_by_influence, x_value=1.0)
    highlighted_units = sorted(set(top_by_unit).union(top_by_influence))

    unit_lines_png = Path(args.out_dir) / "rr100_endpoint_along0_unit_ssi_lines.png"
    influence_unit_lines_png = Path(args.out_dir) / "rr100_endpoint_along0_unit_ssi_lines_top_influence.png"
    influence_unit_lines_with_maps_png = (
        Path(args.out_dir) / "rr100_endpoint_along0_unit_ssi_lines_top_influence_with_activation_rows.png"
    )
    loo_png = Path(args.out_dir) / "rr100_endpoint_along0_unit_ssi_leave_one_out.png"
    draw_unit_lines(
        diagnostics,
        top_by_unit_for_plot,
        unit_lines_png,
        int(args.dpi),
        highlight_note="largest unit SSI changes highlighted",
    )
    draw_unit_lines(
        diagnostics,
        top_by_influence_for_plot,
        influence_unit_lines_png,
        int(args.dpi),
        highlight_note="largest leave-one-out influences highlighted",
    )
    draw_leave_one_out(diagnostics, top_by_influence, top_df, loo_png, int(args.dpi))
    map_manifest_csv = None
    map_manifest_rows: list[dict[str, Any]] = []
    if not bool(args.skip_highlighted_unit_maps):
        draw_unit_lines_with_activation_rows(
            diagnostics=diagnostics,
            highlighted_units=top_by_influence_for_plot,
            rows=rows,
            stats_by_condition=stats_by_condition,
            path=influence_unit_lines_with_maps_png,
            dpi=int(args.dpi),
            highlight_note="largest leave-one-out influences highlighted",
            map_vmin_percentile=float(args.map_vmin_percentile),
            map_vmax_percentile=float(args.map_vmax_percentile),
            figure_subtitle=(
                "activation maps below use monotonic grayscale per unit row; low is black and high is white; "
                "rows and legend are ordered by y at across=1"
            ),
        )
        map_manifest_csv, map_manifest_rows = write_highlighted_unit_maps(
            args=args,
            rows=rows,
            stats_by_condition=stats_by_condition,
            unit_df=unit_df,
            highlighted_units=highlighted_units,
        )

    npz_path = Path(args.out_dir) / "rr100_endpoint_along0_unit_ssi_diagnostics_arrays.npz"
    np.savez_compressed(
        npz_path,
        across_values=np.asarray(diagnostics["across_values"], dtype=np.float32),
        population_ratio=np.asarray(diagnostics["population_ratio"], dtype=np.float32),
        unit_ratio=np.asarray(diagnostics["unit_ratio"], dtype=np.float32),
        unit_log2_ratio=np.asarray(diagnostics["unit_log2_ratio"], dtype=np.float32),
        leave_one_out_ratio=np.asarray(diagnostics["leave_one_out_ratio"], dtype=np.float32),
        leave_one_out_delta=np.asarray(diagnostics["leave_one_out_delta"], dtype=np.float32),
        stimulus_normalization=np.asarray([STIMULUS_NORMALIZATION]),
    )
    manifest_path = Path(args.out_dir) / "rr100_endpoint_along0_unit_ssi_manifest.json"
    write_json(
        manifest_path,
        {
            "analysis": "rr100_endpoint_along0_unit_ssi_diagnostic",
            "out_dir": Path(args.out_dir),
            "conditions": conditions,
            "across_scales": across_scales,
            "along_scale": float(args.along_scale),
            "n_traces": int(args.n_traces),
            "history_frames": int(args.history_frames),
            "stimulus_normalization": STIMULUS_NORMALIZATION,
            "fd_step_arcmin": float(args.fd_step_arcmin),
            "seed": int(args.seed),
            "device_arg": str(args.device),
            "actual_device": device,
            "batch_size": int(args.batch_size),
            "population_version": RR100_VERSION,
            "endpoint_alignment": "tau_endpoint[t] = tau[t] - tau[-1]",
            "readout_time_contract": "terminal response frame only",
            "unit_table_csv": unit_csv,
            "top_units_csv": top_csv,
            "unit_lines_png": unit_lines_png,
            "influence_unit_lines_png": influence_unit_lines_png,
            "influence_unit_lines_with_activation_rows_png": influence_unit_lines_with_maps_png
            if not bool(args.skip_highlighted_unit_maps)
            else None,
            "leave_one_out_png": loo_png,
            "arrays_npz": npz_path,
            "summary_cross_check": comparison,
            "top_units_by_unit_log2_deviation": top_by_unit,
            "top_units_by_leave_one_out_influence": top_by_influence,
            "top_units_by_unit_log2_deviation_plot_order_at_across1": top_by_unit_for_plot,
            "top_units_by_leave_one_out_influence_plot_order_at_across1": top_by_influence_for_plot,
            "highlighted_units_union": highlighted_units,
            "highlighted_unit_map_manifest_csv": map_manifest_csv,
            "highlighted_unit_map_count": len(map_manifest_rows),
        },
    )

    print(f"Wrote unit SSI table: {unit_csv}", flush=True)
    print(f"Wrote top-unit table: {top_csv}", flush=True)
    print(f"Wrote unit line plot: {unit_lines_png}", flush=True)
    print(f"Wrote influence unit line plot: {influence_unit_lines_png}", flush=True)
    if not bool(args.skip_highlighted_unit_maps):
        print(f"Wrote influence unit line plot with activation rows: {influence_unit_lines_with_maps_png}", flush=True)
    print(f"Wrote leave-one-out plot: {loo_png}", flush=True)
    if map_manifest_csv is not None:
        print(f"Wrote highlighted-unit activation-map manifest: {map_manifest_csv}", flush=True)
    print(f"Wrote manifest: {manifest_path}", flush=True)
    print(
        top_df.head(top_n)[
            [
                "unit_index",
                "max_abs_leave_one_out_population_ratio_delta",
                "max_abs_log2_unit_ssi_vs_static",
                "static_unit_ssi_bits_per_spike_mean",
                "static_unit_mean_rate_mean",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6g}"),
        flush=True,
    )


if __name__ == "__main__":
    main()
