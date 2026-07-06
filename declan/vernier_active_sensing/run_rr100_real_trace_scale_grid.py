"""Run an RR100 Vernier grid for anisotropically scaled real FEM traces.

Each grid cell uses the same recorded eye traces but scales the trace around
its trial mean independently on the canonical vertical-Vernier axes:

    across-contour = horizontal/x eye component
    along-contour  = vertical/y eye component

For each scale pair, the script computes RR100 movie-medoid responses for
``+delta`` and ``-delta`` Vernier offsets, then reports:

- pose-aware diagonal Poisson Fisher, averaged across traces;
- pose-hidden diagonal Fisher with count noise plus trajectory marginal variance;
- general SSI from the spatial RR100 response maps.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view
from declan.vernier_active_sensing.forward import build_vernier_movie, load_model_and_readout
from declan.vernier_active_sensing.metrics import (
    expected_counts,
    poisson_fisher_counts,
    pose_blind_diagonal_fisher,
)
from declan.vernier_active_sensing.stimulus import VernierSpec
from declan.vernier_active_sensing.trajectories import (
    DEFAULT_EYE_TRACES_PATH,
    condition_trace,
    load_eye_traces,
    subsample_traces,
    valid_trace,
)
from scripts.temporal_decoding.rate_computation import compute_trial_rates


RR100_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75"
    "_medoidPosthocminRepcomplete0p45_movieMedoid"
)
DEFAULT_SCALES = "0,0.125,0.25,0.5,0.75,1,1.5,2,3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/notebook_vernier_walkthrough/rr100_real_trace_scale_grid",
    )
    parser.add_argument("--eye-traces-path", type=Path, default=ROOT / DEFAULT_EYE_TRACES_PATH)
    parser.add_argument("--scales", type=str, default=DEFAULT_SCALES)
    parser.add_argument("--n-traces", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--fd-step-arcmin", type=float, default=0.25)
    parser.add_argument("--bin-seconds", type=float, default=1.0 / 120.0)
    parser.add_argument("--phi", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, cuda:1, ...")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true", help="Recompute per-condition RR100 rate caches.")
    return parser.parse_args()


def canonical_vernier_spec(offset_arcmin: float = 0.0) -> VernierSpec:
    return VernierSpec(
        offset_arcmin=float(offset_arcmin),
        bar_width_arcmin=2.0,
        gap_arcmin=4.0,
        bar_length_arcmin=12.0,
        contrast=0.5,
        polarity="bright",
    )


def parse_scales(text: str) -> list[float]:
    scales = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not scales:
        raise ValueError("At least one scale is required.")
    return scales


def scale_token(scale: float) -> str:
    text = f"{float(scale):g}"
    return text.replace(".", "p")


def condition_name(across_scale: float, along_scale: float) -> str:
    return f"real_aniso_across_{scale_token(across_scale)}_along_{scale_token(along_scale)}"


def condition_specs(scales: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "condition": "static_center",
            "label": "static center",
            "across_scale": float("nan"),
            "along_scale": float("nan"),
            "is_static_baseline": True,
        }
    ]
    for along in scales:
        for across in scales:
            rows.append(
                {
                    "condition": condition_name(across, along),
                    "label": f"across {across:g}; along {along:g}",
                    "across_scale": float(across),
                    "along_scale": float(along),
                    "is_static_baseline": False,
                }
            )
    return rows


def ssi_single_frame(rate_maps: np.ndarray, eps: float = 1e-8) -> float:
    y = np.asarray(rate_maps, dtype=np.float64)
    if y.ndim != 3:
        raise ValueError(f"Expected (unit, H, W), got {y.shape}")
    flat = y.reshape(y.shape[0], -1)
    rbar = flat.mean(axis=1)
    gain = flat / (rbar[:, None] + eps)
    unit_bits = np.mean(gain * np.log2(gain + eps), axis=1)
    weights = rbar / max(float(rbar.sum()), eps)
    return float(np.sum(weights * unit_bits))


def ssi_timecourse(rate_movie: np.ndarray) -> np.ndarray:
    return np.asarray([ssi_single_frame(rate_movie[t]) for t in range(rate_movie.shape[0])], dtype=np.float32)


def collapse_max(rate_movie: np.ndarray) -> np.ndarray:
    return np.asarray(rate_movie, dtype=np.float32).max(axis=(2, 3))


def cache_path(args: argparse.Namespace, condition: str) -> Path:
    return (
        Path(args.out_dir)
        / "cache"
        / f"rr100_rates_{condition}_fd{float(args.fd_step_arcmin):.4f}arcmin.npz"
    )


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
    ssi_curves: list[np.ndarray] = []
    pose_traces: list[np.ndarray] = []
    inventory_rows: list[dict[str, Any]] = []

    for trace_idx in range(trace_set.traces.shape[0]):
        base_trace = valid_trace(trace_set, trace_idx, max_frames=int(args.max_frames))
        rng = np.random.default_rng(int(args.seed) + 1009 * trace_idx)
        effective_trace, trace_meta = condition_trace(
            base_trace,
            condition=condition,
            trace_set=trace_set,
            rng=rng,
        )
        per_sign_maps: dict[str, np.ndarray] = {}
        for sign, spec in (("plus", plus_spec), ("minus", minus_spec)):
            stim = build_vernier_movie(spec, effective_trace, device=device)
            full_spatial = compute_trial_rates(
                model,
                readout,
                stim,
                batch_size=int(args.batch_size),
                return_spatial=True,
            ).astype(np.float32)
            per_sign_maps[sign] = apply_population_view(full_spatial, view).astype(np.float32)
            del stim, full_spatial
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        t = min(per_sign_maps["plus"].shape[0], per_sign_maps["minus"].shape[0])
        plus_map = per_sign_maps["plus"][:t]
        minus_map = per_sign_maps["minus"][:t]
        plus_rates.append(collapse_max(plus_map))
        minus_rates.append(collapse_max(minus_map))
        ssi_curves.append(ssi_timecourse(0.5 * (plus_map + minus_map)))
        pose_traces.append(np.asarray(effective_trace[:t], dtype=np.float32))
        inventory_rows.append(
            {
                "condition": condition,
                "trace_index": int(trace_idx),
                "n_input_frames": int(base_trace.shape[0]),
                "n_output_timebins": int(t),
                "trace_x_mean_deg": float(np.mean(effective_trace[:, 0])),
                "trace_y_mean_deg": float(np.mean(effective_trace[:, 1])),
                "trace_x_std_deg": float(np.std(effective_trace[:, 0])),
                "trace_y_std_deg": float(np.std(effective_trace[:, 1])),
                **trace_meta,
            }
        )
        print(f"  trace {trace_idx}: T={t}", flush=True)
        del per_sign_maps, plus_map, minus_map
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "plus_rates": np.asarray(plus_rates, dtype=np.float32),
        "minus_rates": np.asarray(minus_rates, dtype=np.float32),
        "ssi_curves": np.asarray(ssi_curves, dtype=np.float32),
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
    if path.exists() and not bool(args.force):
        with np.load(path, allow_pickle=True) as data:
            print(f"Loaded cache: {path}", flush=True)
            return {
                "plus_rates": np.asarray(data["plus_rates"], dtype=np.float32),
                "minus_rates": np.asarray(data["minus_rates"], dtype=np.float32),
                "ssi_curves": np.asarray(data["ssi_curves"], dtype=np.float32),
                "pose_traces": np.asarray(data["pose_traces"], dtype=np.float32),
                "inventory_rows": list(data["inventory_rows"].tolist()),
            }

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
        ssi_curves=cache["ssi_curves"],
        pose_traces=cache["pose_traces"],
        inventory_rows=np.asarray(cache["inventory_rows"], dtype=object),
        condition=np.asarray([condition]),
        fd_step_arcmin=np.asarray([float(args.fd_step_arcmin)], dtype=np.float32),
        bin_seconds=np.asarray([float(args.bin_seconds)], dtype=np.float32),
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
    ssi_curves = np.asarray(cache["ssi_curves"], dtype=np.float32)
    trace_rows: list[dict[str, Any]] = []
    fisher_finals: list[float] = []
    dprime2_finals: list[float] = []
    ssi_means: list[float] = []
    for trace_idx in range(plus.shape[0]):
        t = min(plus.shape[1], minus.shape[1], ssi_curves.shape[1])
        info = poisson_fisher_counts(
            expected_counts(plus[trace_idx, :t], float(args.bin_seconds)),
            expected_counts(minus[trace_idx, :t], float(args.bin_seconds)),
            step_arcmin=float(args.fd_step_arcmin),
            phi=float(args.phi),
        )
        fisher_final = float(info.cumulative_fisher[-1])
        dprime2_final = float(info.cumulative_dprime2[-1])
        ssi_mean = float(np.nanmean(ssi_curves[trace_idx, :t]))
        fisher_finals.append(fisher_final)
        dprime2_finals.append(dprime2_final)
        ssi_means.append(ssi_mean)
        trace_rows.append(
            {
                "condition": condition,
                "trace_index": int(trace_idx),
                "across_scale": row["across_scale"],
                "along_scale": row["along_scale"],
                "pose_aware_fisher": fisher_final,
                "pose_aware_dprime2": dprime2_final,
                "ssi_bits_per_spike": ssi_mean,
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
    ssi_arr = np.asarray(ssi_means, dtype=np.float64)
    summary = {
        "condition": condition,
        "label": row["label"],
        "across_scale": row["across_scale"],
        "along_scale": row["along_scale"],
        "is_static_baseline": bool(row["is_static_baseline"]),
        "n_traces": int(plus.shape[0]),
        "n_frames": int(plus.shape[1]),
        "n_units": int(plus.shape[2]),
        "fd_step_arcmin": float(args.fd_step_arcmin),
        "pose_aware_fisher_mean": float(np.nanmean(fisher_arr)),
        "pose_aware_fisher_sem": float(np.nanstd(fisher_arr, ddof=1) / max(math.sqrt(fisher_arr.size), 1.0)),
        "pose_aware_dprime2_mean": float(np.nanmean(dprime_arr)),
        "pose_aware_dprime2_sem": float(np.nanstd(dprime_arr, ddof=1) / max(math.sqrt(dprime_arr.size), 1.0)),
        "pose_hidden_fisher": float(blind["cumulative_fisher"][-1]),
        "pose_hidden_dprime2": float(blind["cumulative_dprime2"][-1]),
        "ssi_bits_per_spike_mean": float(np.nanmean(ssi_arr)),
        "ssi_bits_per_spike_sem": float(np.nanstd(ssi_arr, ddof=1) / max(math.sqrt(ssi_arr.size), 1.0)),
    }
    return summary, trace_rows


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


def _grid(summary: pd.DataFrame, scales: list[float], value_col: str) -> np.ndarray:
    values = np.full((len(scales), len(scales)), np.nan, dtype=np.float64)
    grid_rows = summary[~summary["is_static_baseline"].astype(bool)]
    for y, along in enumerate(scales):
        for x, across in enumerate(scales):
            row = grid_rows[
                np.isclose(pd.to_numeric(grid_rows["across_scale"], errors="coerce"), across)
                & np.isclose(pd.to_numeric(grid_rows["along_scale"], errors="coerce"), along)
            ]
            if not row.empty:
                values[y, x] = float(row.iloc[0][value_col])
    return values


def _annotate_heatmap(ax: Any, values: np.ndarray) -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    span = float(np.nanmax(finite) - np.nanmin(finite))
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            if not np.isfinite(value):
                continue
            text_color = "white" if span > 0 and value < np.nanmin(finite) + 0.38 * span else "black"
            ax.text(x, y, f"{value:.2g}", ha="center", va="center", fontsize=5.6, color=text_color)


def write_heatmaps(args: argparse.Namespace, summary: pd.DataFrame, scales: list[float]) -> None:
    out_dir = Path(args.out_dir)
    metric_specs = [
        ("pose_aware_fisher_vs_static", "Known-trace Fisher / static"),
        ("pose_hidden_fisher_vs_static", "Unknown-trace Fisher / static"),
        ("ssi_bits_per_spike_vs_static", "SSI / static"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), dpi=220, constrained_layout=True)
    for ax, (value_col, title) in zip(axes, metric_specs, strict=True):
        values = _grid(summary, scales, value_col)
        im = ax.imshow(values, origin="lower", interpolation="nearest", cmap="viridis")
        _annotate_heatmap(ax, values)
        if 1.0 in scales:
            one_idx = scales.index(1.0)
            ax.scatter([one_idx], [one_idx], marker="x", s=36, color="white", linewidths=1.4)
        ax.set_xticks(np.arange(len(scales)))
        ax.set_yticks(np.arange(len(scales)))
        ax.set_xticklabels([f"{s:g}" for s in scales], rotation=45, ha="right")
        ax.set_yticklabels([f"{s:g}" for s in scales])
        ax.set_xlabel("across-contour scale")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0].set_ylabel("along-contour scale")
    fig.suptitle(
        "RR100 real-FEM anisotropic scale grid, metrics relative to static center\n"
        f"fd={float(args.fd_step_arcmin):g} arcmin; {int(args.n_traces)} traces x {int(args.max_frames)} frames",
        y=1.06,
        fontsize=11,
    )
    fig.savefig(out_dir / "rr100_real_trace_scale_grid_vs_static_heatmaps.png", bbox_inches="tight")
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, 3, figsize=(12.6, 4.2), dpi=220, constrained_layout=True)
    for ax, (value_col, title) in zip(axes2, metric_specs, strict=True):
        values = np.log2(np.maximum(_grid(summary, scales, value_col), 1e-12))
        finite = values[np.isfinite(values)]
        vmax = max(float(np.nanmax(np.abs(finite))), 1.0) if finite.size else 1.0
        im = ax.imshow(values, origin="lower", interpolation="nearest", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        _annotate_heatmap(ax, values)
        if 1.0 in scales:
            one_idx = scales.index(1.0)
            ax.scatter([one_idx], [one_idx], marker="x", s=36, color="black", linewidths=1.4)
        ax.set_xticks(np.arange(len(scales)))
        ax.set_yticks(np.arange(len(scales)))
        ax.set_xticklabels([f"{s:g}" for s in scales], rotation=45, ha="right")
        ax.set_yticklabels([f"{s:g}" for s in scales])
        ax.set_xlabel("across-contour scale")
        ax.set_title(f"log2 {title}", fontsize=10)
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes2[0].set_ylabel("along-contour scale")
    fig2.suptitle("RR100 real-FEM anisotropic scale grid, log2 relative to static center", y=1.04, fontsize=11)
    fig2.savefig(out_dir / "rr100_real_trace_scale_grid_log2_vs_static_heatmaps.png", bbox_inches="tight")
    plt.close(fig2)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scales = parse_scales(args.scales)
    specs = condition_specs(scales)
    trace_set = subsample_traces(load_eye_traces(Path(args.eye_traces_path)), int(args.n_traces), int(args.seed))

    device_arg = None if str(args.device).lower() == "auto" else str(args.device)
    print("Loading model/readout...", flush=True)
    model, readout = load_model_and_readout(device=device_arg)
    device = str(next(model.model.parameters()).device)
    print(f"Model device: {device}", flush=True)
    view = load_population_view(version_name=RR100_VERSION)
    print(f"Population view: {view.name}; n_units={int(view.n_units)}", flush=True)

    manifest = {
        "args": {
            "out_dir": str(args.out_dir),
            "eye_traces_path": str(args.eye_traces_path),
            "scales": scales,
            "n_traces": int(args.n_traces),
            "max_frames": int(args.max_frames),
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
        "condition_count": len(specs),
    }
    (Path(args.out_dir) / "rr100_real_trace_scale_grid_manifest.json").write_text(
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
        partial_summary.to_csv(Path(args.out_dir) / "rr100_real_trace_scale_grid_summary_partial.csv", index=False)

    summary_df = add_static_ratios(pd.DataFrame(summary_rows))
    trace_df = pd.DataFrame(trace_rows)
    inventory_df = pd.DataFrame(inventory_rows)
    summary_path = Path(args.out_dir) / "rr100_real_trace_scale_grid_summary.csv"
    trace_path = Path(args.out_dir) / "rr100_real_trace_scale_grid_trace_table.csv"
    inventory_path = Path(args.out_dir) / "rr100_real_trace_scale_grid_motion_inventory.csv"
    summary_df.to_csv(summary_path, index=False)
    trace_df.to_csv(trace_path, index=False)
    inventory_df.to_csv(inventory_path, index=False)
    write_heatmaps(args, summary_df, scales)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6g}"), flush=True)
    print(f"Saved summary: {summary_path}", flush=True)
    print(f"Saved trace table: {trace_path}", flush=True)
    print(f"Saved inventory: {inventory_path}", flush=True)
    print(f"Saved heatmaps: {Path(args.out_dir) / 'rr100_real_trace_scale_grid_vs_static_heatmaps.png'}", flush=True)


if __name__ == "__main__":
    main()
