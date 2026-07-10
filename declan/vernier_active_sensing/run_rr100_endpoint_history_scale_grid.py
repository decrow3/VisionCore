"""Run an RR100 Vernier endpoint-history anisotropic scale grid.

This is the endpoint-aligned, terminal-frame counterpart of
``run_rr100_real_trace_scale_grid.py``. Each anisotropically scaled real FEM
history is shifted so the final eye position is zero:

    endpoint_trace[t] = trace[t] - trace[-1]

The twin sees the full 32-frame history in the final lag cube, but Fisher and
SSI are computed from only the terminal model response frame.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view
from declan.vernier_active_sensing.forward import (
    PKL_PATH,
    STIMULUS_NORMALIZATION,
    build_vernier_movie,
    load_model_and_readout,
)
from declan.vernier_active_sensing.metrics import expected_counts, poisson_fisher_counts, pose_blind_diagonal_fisher
from declan.vernier_active_sensing.plot_rr100_real_trace_scale_grid_rows import write_two_baseline_row_figure
from declan.vernier_active_sensing.run_endpoint_history_last_frame_readout import (
    ENDPOINT_ALIGNMENT,
    HISTORY_WINDOW,
    endpoint_aligned_trace,
    terminal_history_window,
)
from declan.vernier_active_sensing.run_rr100_real_trace_scale_grid import (
    DEFAULT_SCALES,
    RR100_VERSION,
    canonical_vernier_spec,
    collapse_max,
    condition_specs,
    parse_scales,
    ssi_single_frame,
)
from declan.vernier_active_sensing.trajectories import (
    DEFAULT_EYE_TRACES_PATH,
    condition_trace,
    load_eye_traces,
    subsample_traces,
    valid_trace,
)
from scripts.temporal_decoding.rate_computation import compute_trial_rates
from scripts.temporal_decoding.stimulus_hires import N_LAGS as MODEL_HISTORY_FRAMES


DEFAULT_OUT_DIR = ROOT / "outputs/vernier_endpoint_history_last_frame_tutorial/rr100_endpoint_history_scale_grid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--eye-traces-path", type=Path, default=ROOT / DEFAULT_EYE_TRACES_PATH)
    parser.add_argument("--scales", type=str, default=DEFAULT_SCALES)
    parser.add_argument("--across-scales", type=str, default="")
    parser.add_argument("--along-scales", type=str, default="")
    parser.add_argument("--plot-along-scales", type=str, default="0,0.5,1,2")
    parser.add_argument("--n-traces", type=int, default=16)
    parser.add_argument("--history-frames", type=int, default=int(MODEL_HISTORY_FRAMES))
    parser.add_argument("--fd-step-arcmin", type=float, default=0.25)
    parser.add_argument("--bin-seconds", type=float, default=1.0 / 120.0)
    parser.add_argument("--phi", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, cuda:1, ...")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true", help="Recompute per-condition RR100 terminal caches.")
    return parser.parse_args()


def cache_path(args: argparse.Namespace, condition: str) -> Path:
    return (
        Path(args.out_dir)
        / "cache"
        / f"rr100_endpoint_last_frame_{condition}_fd{float(args.fd_step_arcmin):.4f}arcmin.npz"
    )


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
        return [json_ready(val) for val in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _cache_identity(args: argparse.Namespace, *, condition: str) -> dict[str, Any]:
    return {
        "schema_version": 4,
        "stimulus_normalization": STIMULUS_NORMALIZATION,
        "condition": str(condition),
        "eye_traces_path": str(Path(args.eye_traces_path).expanduser().resolve()),
        "n_traces": int(args.n_traces),
        "history_frames": int(args.history_frames),
        "model_history_frames": int(MODEL_HISTORY_FRAMES),
        "fd_step_arcmin": float(args.fd_step_arcmin),
        "bin_seconds": float(args.bin_seconds),
        "phi": float(args.phi),
        "seed": int(args.seed),
        "history_window": HISTORY_WINDOW,
        "endpoint_alignment": ENDPOINT_ALIGNMENT,
        "readout_time_contract": "terminal_response_only",
        "population_version": RR100_VERSION,
        "readout_source_pkl": str(PKL_PATH.expanduser().resolve()),
        "plus_spec": asdict(canonical_vernier_spec(+float(args.fd_step_arcmin))),
        "minus_spec": asdict(canonical_vernier_spec(-float(args.fd_step_arcmin))),
        "map_contract": "RR100 terminal spatial maps collapsed to per-unit max rates plus population SSI",
    }


def _identity_text(identity: dict[str, Any]) -> str:
    return json.dumps(json_ready(identity), sort_keys=True, separators=(",", ":"))


def _trace_metrics(trace: np.ndarray) -> dict[str, float]:
    arr = np.asarray(trace, dtype=np.float64)
    steps = np.diff(arr, axis=0) if arr.shape[0] > 1 else np.zeros((0, 2), dtype=np.float64)
    radius = np.linalg.norm(arr, axis=1)
    return {
        "endpoint_x_deg": float(arr[-1, 0]),
        "endpoint_y_deg": float(arr[-1, 1]),
        "endpoint_norm_deg": float(radius[-1]),
        "history_rms_deg": float(np.sqrt(np.mean(np.sum(arr * arr, axis=1)))),
        "history_path_length_deg": float(np.sum(np.linalg.norm(steps, axis=1))),
        "history_max_radius_deg": float(np.max(radius)),
    }


def _terminal_rr100_map(
    model: Any,
    readout: Any,
    view: Any,
    spec: Any,
    endpoint_trace: np.ndarray,
    *,
    device: str,
    batch_size: int,
    history_frames: int,
) -> np.ndarray:
    stim = build_vernier_movie(spec, endpoint_trace, n_lags=int(MODEL_HISTORY_FRAMES), device=device)
    if int(stim.shape[0]) != int(history_frames):
        raise RuntimeError(f"Expected {history_frames} lag windows, got {int(stim.shape[0])}")
    full_terminal = compute_trial_rates(
        model,
        readout,
        stim[-1:],
        batch_size=int(batch_size),
        return_spatial=True,
    ).astype(np.float32)
    return apply_population_view(full_terminal, view).astype(np.float32)


def compute_condition_cache(
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
    plus_rates: list[np.ndarray] = []
    minus_rates: list[np.ndarray] = []
    ssi_values: list[float] = []
    pose_traces: list[np.ndarray] = []
    inventory_rows: list[dict[str, Any]] = []

    for trace_idx in range(trace_set.traces.shape[0]):
        base_trace = valid_trace(trace_set, trace_idx)
        rng = np.random.default_rng(int(args.seed) + 1009 * trace_idx)
        effective_trace, trace_meta = condition_trace(
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
        plus_rates.append(collapse_max(plus_map))
        minus_rates.append(collapse_max(minus_map))
        ssi_values.append(ssi_single_frame(0.5 * (plus_map[0] + minus_map[0])))
        pose_traces.append(endpoint_trace.astype(np.float32))
        metrics = _trace_metrics(endpoint_trace)
        inventory_rows.append(
            {
                "condition": condition,
                "trace_index": int(trace_idx),
                "n_input_frames": int(base_trace.shape[0]),
                "n_output_timebins": 1,
                "history_window": HISTORY_WINDOW,
                "endpoint_alignment": ENDPOINT_ALIGNMENT,
                "readout_time_contract": "terminal_response_only",
                "terminal_frames": 1,
                "trace_x_mean_deg": float(np.mean(endpoint_trace[:, 0])),
                "trace_y_mean_deg": float(np.mean(endpoint_trace[:, 1])),
                "trace_x_std_deg": float(np.std(endpoint_trace[:, 0])),
                "trace_y_std_deg": float(np.std(endpoint_trace[:, 1])),
                **metrics,
                **trace_meta,
            }
        )
        print(f"  trace {trace_idx}: endpoint_norm={metrics['endpoint_norm_deg']:.3g}", flush=True)
        del plus_map, minus_map
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "plus_rates": np.asarray(plus_rates, dtype=np.float32),
        "minus_rates": np.asarray(minus_rates, dtype=np.float32),
        "ssi_values": np.asarray(ssi_values, dtype=np.float32),
        "pose_traces": np.asarray(pose_traces, dtype=np.float32),
        "inventory_rows": inventory_rows,
    }


def load_or_compute_condition(
    args: argparse.Namespace,
    *,
    condition: str,
    trace_set: Any,
    model: Any,
    readout: Any,
    view: Any,
    device: str,
) -> dict[str, Any]:
    path = cache_path(args, condition)
    expected_identity = _cache_identity(args, condition=condition)
    if path.exists() and not bool(args.force):
        with np.load(path, allow_pickle=True) as data:
            matches_identity = (
                "cache_identity_json" in data
                and str(np.asarray(data["cache_identity_json"]).ravel()[0]) == _identity_text(expected_identity)
            )
            if matches_identity:
                print(f"Loaded cache: {path}", flush=True)
                return {
                    "plus_rates": np.asarray(data["plus_rates"], dtype=np.float32),
                    "minus_rates": np.asarray(data["minus_rates"], dtype=np.float32),
                    "ssi_values": np.asarray(data["ssi_values"], dtype=np.float32),
                    "pose_traces": np.asarray(data["pose_traces"], dtype=np.float32),
                    "inventory_rows": list(data["inventory_rows"].tolist()),
                }
        print(f"RR100 endpoint scale-grid cache metadata mismatch; recomputing {path}", flush=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    cache = compute_condition_cache(
        args,
        condition=condition,
        trace_set=trace_set,
        model=model,
        readout=readout,
        view=view,
        device=device,
    )
    np.savez_compressed(
        path,
        plus_rates=cache["plus_rates"],
        minus_rates=cache["minus_rates"],
        ssi_values=cache["ssi_values"],
        pose_traces=cache["pose_traces"],
        inventory_rows=np.asarray(cache["inventory_rows"], dtype=object),
        condition=np.asarray([condition]),
        fd_step_arcmin=np.asarray([float(args.fd_step_arcmin)], dtype=np.float32),
        bin_seconds=np.asarray([float(args.bin_seconds)], dtype=np.float32),
        history_frames=np.asarray([int(args.history_frames)], dtype=np.int32),
        model_history_frames=np.asarray([int(MODEL_HISTORY_FRAMES)], dtype=np.int32),
        terminal_frames=np.asarray([1], dtype=np.int32),
        history_window=np.asarray([HISTORY_WINDOW]),
        endpoint_alignment=np.asarray([ENDPOINT_ALIGNMENT]),
        readout_time_contract=np.asarray(["terminal_response_only"]),
        stimulus_normalization=np.asarray([STIMULUS_NORMALIZATION]),
        seed=np.asarray([int(args.seed)], dtype=np.int64),
        eye_traces_path=np.asarray([str(Path(args.eye_traces_path).expanduser().resolve())]),
        cache_identity_json=np.asarray([_identity_text(expected_identity)]),
    )
    print(f"Saved cache: {path}", flush=True)
    return cache


def summarize_condition(
    args: argparse.Namespace,
    row: dict[str, Any],
    cache: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    condition = str(row["condition"])
    plus = np.asarray(cache["plus_rates"], dtype=np.float32)
    minus = np.asarray(cache["minus_rates"], dtype=np.float32)
    ssi_values = np.asarray(cache["ssi_values"], dtype=np.float32)
    trace_rows: list[dict[str, Any]] = []
    fisher_finals: list[float] = []
    dprime2_finals: list[float] = []
    for trace_idx in range(plus.shape[0]):
        info = poisson_fisher_counts(
            expected_counts(plus[trace_idx], float(args.bin_seconds)),
            expected_counts(minus[trace_idx], float(args.bin_seconds)),
            step_arcmin=float(args.fd_step_arcmin),
            phi=float(args.phi),
        )
        fisher_final = float(info.cumulative_fisher[-1])
        dprime2_final = float(info.cumulative_dprime2[-1])
        fisher_finals.append(fisher_final)
        dprime2_finals.append(dprime2_final)
        trace_rows.append(
            {
                "condition": condition,
                "trace_index": int(trace_idx),
                "across_scale": row["across_scale"],
                "along_scale": row["along_scale"],
                "pose_aware_fisher": fisher_final,
                "pose_aware_dprime2": dprime2_final,
                "ssi_bits_per_spike": float(ssi_values[trace_idx]),
            }
        )

    blind = pose_blind_diagonal_fisher(
        [arr for arr in plus],
        [arr for arr in minus],
        step_arcmin=float(args.fd_step_arcmin),
        bin_seconds=float(args.bin_seconds),
        phi=float(args.phi),
    )
    fisher_arr = np.asarray(fisher_finals, dtype=np.float64)
    dprime_arr = np.asarray(dprime2_finals, dtype=np.float64)
    ssi_arr = np.asarray(ssi_values, dtype=np.float64)
    return (
        {
            "condition": condition,
            "label": row["label"],
            "across_scale": row["across_scale"],
            "along_scale": row["along_scale"],
            "is_static_baseline": bool(row["is_static_baseline"]),
            "n_traces": int(plus.shape[0]),
            "n_frames": int(plus.shape[1]),
            "n_units": int(plus.shape[2]),
            "fd_step_arcmin": float(args.fd_step_arcmin),
            "history_frames": int(args.history_frames),
            "terminal_frames": 1,
            "history_window": HISTORY_WINDOW,
            "endpoint_alignment": ENDPOINT_ALIGNMENT,
            "readout_time_contract": "terminal_response_only",
            "pose_aware_fisher_mean": float(np.nanmean(fisher_arr)),
            "pose_aware_fisher_sem": float(np.nanstd(fisher_arr, ddof=1) / max(math.sqrt(fisher_arr.size), 1.0)),
            "pose_aware_dprime2_mean": float(np.nanmean(dprime_arr)),
            "pose_aware_dprime2_sem": float(np.nanstd(dprime_arr, ddof=1) / max(math.sqrt(dprime_arr.size), 1.0)),
            "pose_hidden_fisher": float(blind["cumulative_fisher"][-1]),
            "pose_hidden_dprime2": float(blind["cumulative_dprime2"][-1]),
            "ssi_bits_per_spike_mean": float(np.nanmean(ssi_arr)),
            "ssi_bits_per_spike_sem": float(np.nanstd(ssi_arr, ddof=1) / max(math.sqrt(ssi_arr.size), 1.0)),
        },
        trace_rows,
    )


def add_static_ratios(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    static = out[out["condition"].eq("static_center")]
    if static.empty:
        return out
    ref = static.iloc[0]
    for col, ref_col in [
        ("pose_aware_fisher_mean", "pose_aware_fisher_vs_static"),
        ("pose_hidden_fisher", "pose_hidden_fisher_vs_static"),
        ("ssi_bits_per_spike_mean", "ssi_bits_per_spike_vs_static"),
    ]:
        denom = float(ref[col])
        out[ref_col] = out[col] / denom if denom > 0 else np.nan
    return out


def _plot_along_scales(args: argparse.Namespace) -> list[float] | None:
    text = str(args.plot_along_scales).strip()
    return parse_scales(text) if text else None


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scales = parse_scales(args.scales)
    across_scales = parse_scales(args.across_scales) if str(args.across_scales).strip() else scales
    along_scales = parse_scales(args.along_scales) if str(args.along_scales).strip() else scales
    specs = condition_specs(across_scales, along_scales)
    trace_set = subsample_traces(load_eye_traces(Path(args.eye_traces_path)), int(args.n_traces), int(args.seed))

    device_arg = None if str(args.device).lower() == "auto" else str(args.device)
    print("Loading model/readout...", flush=True)
    model, readout = load_model_and_readout(device=device_arg)
    device = str(next(model.model.parameters()).device)
    print(f"Model device: {device}", flush=True)
    view = load_population_view(version_name=RR100_VERSION)
    print(f"Population view: {view.name}; n_units={int(view.n_units)}", flush=True)

    manifest = {
        "analysis": "rr100_endpoint_history_last_frame_scale_grid",
        "args": {
            "out_dir": str(args.out_dir),
            "eye_traces_path": str(args.eye_traces_path),
            "scales": scales,
            "across_scales": across_scales,
            "along_scales": along_scales,
            "n_traces": int(args.n_traces),
            "history_frames": int(args.history_frames),
            "fd_step_arcmin": float(args.fd_step_arcmin),
            "bin_seconds": float(args.bin_seconds),
            "phi": float(args.phi),
            "seed": int(args.seed),
            "device_arg": str(args.device),
            "actual_device": device,
            "batch_size": int(args.batch_size),
        },
        "population_version": RR100_VERSION,
        "axis_convention": "vertical_vernier_across_x_along_y",
        "history_window": HISTORY_WINDOW,
        "endpoint_alignment": "tau_endpoint[t] = tau_tail[t] - tau_tail[-1]",
        "readout_time_contract": "terminal response frame only",
        "stimulus_normalization": STIMULUS_NORMALIZATION,
        "condition_count": len(specs),
        "note": "grid_0_0 and static_center both collapse to zero endpoint history after endpoint alignment",
    }
    (Path(args.out_dir) / "rr100_endpoint_history_scale_grid_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(specs, start=1):
        condition = str(spec["condition"])
        print(f"[{idx}/{len(specs)}] condition={condition}", flush=True)
        cache = load_or_compute_condition(
            args,
            condition=condition,
            trace_set=trace_set,
            model=model,
            readout=readout,
            view=view,
            device=device,
        )
        summary, traces = summarize_condition(args, spec, cache)
        summary_rows.append(summary)
        trace_rows.extend(traces)
        inventory_rows.extend(cache["inventory_rows"])
        partial_summary = add_static_ratios(pd.DataFrame(summary_rows))
        partial_summary.to_csv(Path(args.out_dir) / "rr100_endpoint_history_scale_grid_summary_partial.csv", index=False)

    summary_df = add_static_ratios(pd.DataFrame(summary_rows))
    trace_df = pd.DataFrame(trace_rows)
    inventory_df = pd.DataFrame(inventory_rows)
    summary_path = Path(args.out_dir) / "rr100_endpoint_history_scale_grid_summary.csv"
    trace_path = Path(args.out_dir) / "rr100_endpoint_history_scale_grid_trace_table.csv"
    inventory_path = Path(args.out_dir) / "rr100_endpoint_history_scale_grid_motion_inventory.csv"
    summary_df.to_csv(summary_path, index=False)
    trace_df.to_csv(trace_path, index=False)
    inventory_df.to_csv(inventory_path, index=False)
    saved = write_two_baseline_row_figure(
        summary_df,
        Path(args.out_dir),
        along_scales=_plot_along_scales(args),
        figure_title="RR100 Vernier endpoint-history last-frame scale grid: absolute loss and incremental motion gain",
        file_prefix="rr100_endpoint_history_last_frame_scale_grid",
        row_titles=(
            "A. Absolute information scale at shared endpoint",
            "B. Same baseline after endpoint alignment (grid 0x0 = static)",
        ),
    )
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6g}"), flush=True)
    print(f"Saved summary: {summary_path}", flush=True)
    print(f"Saved trace table: {trace_path}", flush=True)
    print(f"Saved inventory: {inventory_path}", flush=True)
    for path in saved:
        print(f"Saved plot: {path}", flush=True)


if __name__ == "__main__":
    main()
