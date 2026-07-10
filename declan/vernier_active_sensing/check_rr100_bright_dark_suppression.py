#!/usr/bin/env python3
"""Check whether negative-looking RR100 Vernier maps suppress bright and dark bars."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, replace
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

from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view
from declan.vernier_active_sensing.forward import (
    STIMULUS_NORMALIZATION,
    build_vernier_movie,
    load_model_and_readout,
)
from declan.vernier_active_sensing.run_rr100_real_trace_scale_grid import (
    RR100_VERSION,
    canonical_vernier_spec,
)
from scripts.temporal_decoding.rate_computation import compute_trial_rates
from scripts.temporal_decoding.stimulus_hires import N_LAGS


DEFAULT_POLARITY_CSV = (
    ROOT
    / "outputs/notebook_vernier_walkthrough/rr100_real_trace_scale_grid/unit_ssi_along0_diagnostics"
    / "rr100_real_trace_along0_polarity_unit_table.csv"
)
DEFAULT_OUT_DIR = ROOT / "outputs/notebook_vernier_walkthrough/rr100_bright_dark_suppression_check"
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polarity-csv", type=Path, default=DEFAULT_POLARITY_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rr100-version", type=str, default=RR100_VERSION)
    parser.add_argument("--fd-step-arcmin", type=float, default=0.25)
    parser.add_argument("--n-frames", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, cuda:1, ...")
    parser.add_argument("--low-percentile", type=float, default=5.0)
    parser.add_argument("--high-percentile", type=float, default=95.0)
    parser.add_argument("--top-units", type=int, default=16)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def choose_device(device_arg: str) -> str:
    if str(device_arg).lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device_arg)


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def rr100_spatial_movie(
    *,
    model: Any,
    readout: Any,
    view: Any,
    spec: Any,
    eye_trace: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    stim = build_vernier_movie(spec, eye_trace, n_lags=N_LAGS, device=device)
    full_spatial = compute_trial_rates(
        model,
        readout,
        stim,
        batch_size=int(batch_size),
        return_spatial=True,
    )
    return apply_population_view(full_spatial, view).astype(np.float32)


def midpoint_spatial_movie(
    *,
    model: Any,
    readout: Any,
    view: Any,
    base_spec: Any,
    fd_step_arcmin: float,
    eye_trace: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    plus = replace(base_spec, offset_arcmin=+float(fd_step_arcmin))
    minus = replace(base_spec, offset_arcmin=-float(fd_step_arcmin))
    plus_movie = rr100_spatial_movie(
        model=model,
        readout=readout,
        view=view,
        spec=plus,
        eye_trace=eye_trace,
        device=device,
        batch_size=batch_size,
    )
    minus_movie = rr100_spatial_movie(
        model=model,
        readout=readout,
        view=view,
        spec=minus,
        eye_trace=eye_trace,
        device=device,
        batch_size=batch_size,
    )
    t = min(int(plus_movie.shape[0]), int(minus_movie.shape[0]))
    return (0.5 * (plus_movie[:t] + minus_movie[:t])).astype(np.float32)


def summarize_delta(
    delta_map: np.ndarray,
    *,
    low_percentile: float,
    high_percentile: float,
) -> dict[str, float | str | bool]:
    values = np.asarray(delta_map, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean_delta": math.nan,
            "median_delta": math.nan,
            "low_delta": math.nan,
            "high_delta": math.nan,
            "negative_tail_strength": math.nan,
            "positive_tail_strength": math.nan,
            "negative_tail_mean": math.nan,
            "positive_tail_mean": math.nan,
            "fraction_negative": math.nan,
            "dominant_signed_map": "unknown",
            "dominant_negative_map": False,
        }
    median = float(np.nanmedian(values))
    low = float(np.nanpercentile(values, float(low_percentile)))
    high = float(np.nanpercentile(values, float(high_percentile)))
    negative_tail_strength = median - low
    positive_tail_strength = high - median
    low_mask = values <= low
    high_mask = values >= high
    negative_tail_mean = float(np.nanmean(values[low_mask])) if np.any(low_mask) else math.nan
    positive_tail_mean = float(np.nanmean(values[high_mask])) if np.any(high_mask) else math.nan
    dominant_negative = bool(negative_tail_strength > positive_tail_strength and low < -EPS)
    dominant_positive = bool(positive_tail_strength >= negative_tail_strength and high > EPS)
    if dominant_negative:
        dominant = "negative"
    elif dominant_positive:
        dominant = "positive"
    else:
        dominant = "flat"
    return {
        "mean_delta": float(np.nanmean(values)),
        "median_delta": median,
        "low_delta": low,
        "high_delta": high,
        "negative_tail_strength": float(negative_tail_strength),
        "positive_tail_strength": float(positive_tail_strength),
        "negative_tail_mean": negative_tail_mean,
        "positive_tail_mean": positive_tail_mean,
        "fraction_negative": float(np.mean(values < 0.0)),
        "dominant_signed_map": dominant,
        "dominant_negative_map": dominant == "negative",
    }


def image_limits(images: list[np.ndarray], *, center: float | None = None) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(image, dtype=np.float64).ravel() for image in images])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    if center is None:
        vmin = float(np.nanpercentile(vals, 1.0))
        vmax = float(np.nanpercentile(vals, 99.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return float(np.nanmin(vals)), float(np.nanmax(vals) + EPS)
        return vmin, vmax
    span = float(np.nanpercentile(np.abs(vals - center), 99.0))
    if not np.isfinite(span) or span <= 0.0:
        span = float(np.nanmax(np.abs(vals - center))) if vals.size else 1.0
    span = max(span, EPS)
    return float(center - span), float(center + span)


def plot_negative_unit_sheet(
    *,
    out_path: Path,
    unit_rows: pd.DataFrame,
    maps: dict[str, np.ndarray],
    bright_delta: np.ndarray,
    dark_delta: np.ndarray,
    top_units: int,
    dpi: int,
) -> None:
    if unit_rows.empty:
        return
    ranked = unit_rows.sort_values(
        ["negative_strength", "static_unit_ssi_bits_per_spike_mean"],
        ascending=[False, False],
    ).head(int(top_units))
    unit_indices = [int(v) for v in ranked["unit_index"].to_list()]
    columns = [
        ("blank", maps["blank_mean"], "gray", False),
        ("bright", maps["bright_mean"], "gray", False),
        ("dark", maps["dark_mean"], "gray", False),
        ("bright - blank", bright_delta, "coolwarm", True),
        ("dark - blank", dark_delta, "coolwarm", True),
    ]
    n_rows = len(unit_indices)
    fig, axes = plt.subplots(
        n_rows,
        len(columns),
        figsize=(2.0 * len(columns), 1.8 * max(n_rows, 1)),
        squeeze=False,
        constrained_layout=True,
    )
    rate_limits = image_limits([maps["blank_mean"][u] for u in unit_indices] + [maps["bright_mean"][u] for u in unit_indices] + [maps["dark_mean"][u] for u in unit_indices])
    delta_limits = image_limits([bright_delta[u] for u in unit_indices] + [dark_delta[u] for u in unit_indices], center=0.0)
    for row_i, unit_index in enumerate(unit_indices):
        for col_i, (title, image_stack, cmap, is_delta) in enumerate(columns):
            ax = axes[row_i, col_i]
            vmin, vmax = delta_limits if is_delta else rate_limits
            ax.imshow(image_stack[unit_index], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_i == 0:
                ax.set_title(title, fontsize=8)
            if col_i == 0:
                ax.set_ylabel(f"u{unit_index:03d}", fontsize=8, rotation=0, labelpad=18, va="center")
    fig.suptitle(
        "RR100 negative-classified units: static blank/bright/dark maps and signed deltas",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(str(args.device))

    polarity_df = pd.read_csv(args.polarity_csv)
    polarity_df["unit_index"] = polarity_df["unit_index"].astype(int)
    negative_df = polarity_df[polarity_df["polarity"].astype(str) == "negative"].copy()

    print(f"Loaded polarity table: {args.polarity_csv}", flush=True)
    print(
        "Polarity counts: "
        + ", ".join(
            f"{key}={int(value)}" for key, value in polarity_df["polarity"].value_counts().sort_index().items()
        ),
        flush=True,
    )
    print(f"Computing RR100 static bright/dark suppression check on {device}", flush=True)

    model, readout = load_model_and_readout(device=device)
    view = load_population_view(version_name=str(args.rr100_version))
    zero_trace = np.zeros((int(args.n_frames), 2), dtype=np.float32)
    base = canonical_vernier_spec(0.0)
    blank_spec = replace(base, contrast=0.0, polarity="bright")
    bright_spec = replace(base, contrast=0.5, polarity="bright")
    dark_spec = replace(base, contrast=0.5, polarity="dark")

    blank_movie = midpoint_spatial_movie(
        model=model,
        readout=readout,
        view=view,
        base_spec=blank_spec,
        fd_step_arcmin=float(args.fd_step_arcmin),
        eye_trace=zero_trace,
        device=device,
        batch_size=int(args.batch_size),
    )
    bright_movie = midpoint_spatial_movie(
        model=model,
        readout=readout,
        view=view,
        base_spec=bright_spec,
        fd_step_arcmin=float(args.fd_step_arcmin),
        eye_trace=zero_trace,
        device=device,
        batch_size=int(args.batch_size),
    )
    dark_movie = midpoint_spatial_movie(
        model=model,
        readout=readout,
        view=view,
        base_spec=dark_spec,
        fd_step_arcmin=float(args.fd_step_arcmin),
        eye_trace=zero_trace,
        device=device,
        batch_size=int(args.batch_size),
    )

    maps = {
        "blank_mean": blank_movie.mean(axis=0),
        "bright_mean": bright_movie.mean(axis=0),
        "dark_mean": dark_movie.mean(axis=0),
        "blank_terminal": blank_movie[-1],
        "bright_terminal": bright_movie[-1],
        "dark_terminal": dark_movie[-1],
    }
    map_npz = out_dir / "rr100_static_bright_dark_maps.npz"
    np.savez_compressed(
        map_npz,
        **maps,
        cache_identity_json=json.dumps(
            {
                "stimulus_normalization": STIMULUS_NORMALIZATION,
                "rr100_version": str(args.rr100_version),
                "n_frames": int(args.n_frames),
                "fd_step_arcmin": float(args.fd_step_arcmin),
                "blank_spec": asdict(blank_spec),
                "bright_spec": asdict(bright_spec),
                "dark_spec": asdict(dark_spec),
                "map_contract": "+/- fd midpoint maps; mean uses all frames, terminal uses final frame",
            },
            sort_keys=True,
        ),
    )

    rows: list[dict[str, Any]] = []
    for contract in ("mean", "terminal"):
        bright_delta = maps[f"bright_{contract}"] - maps[f"blank_{contract}"]
        dark_delta = maps[f"dark_{contract}"] - maps[f"blank_{contract}"]
        for _, unit_row in polarity_df.iterrows():
            unit_index = int(unit_row["unit_index"])
            bright_summary = summarize_delta(
                bright_delta[unit_index],
                low_percentile=float(args.low_percentile),
                high_percentile=float(args.high_percentile),
            )
            dark_summary = summarize_delta(
                dark_delta[unit_index],
                low_percentile=float(args.low_percentile),
                high_percentile=float(args.high_percentile),
            )
            rows.append(
                {
                    "map_contract": contract,
                    "unit_index": unit_index,
                    "polarity_table_label": str(unit_row["polarity"]),
                    "static_unit_ssi_bits_per_spike_mean": float(
                        unit_row.get("static_unit_ssi_bits_per_spike_mean", math.nan)
                    ),
                    "static_unit_mean_rate_mean": float(unit_row.get("static_unit_mean_rate_mean", math.nan)),
                    "static_map_negative_strength": float(unit_row.get("negative_strength", math.nan)),
                    "static_map_positive_strength": float(unit_row.get("positive_strength", math.nan)),
                    "bright_dominant_signed_map": bright_summary["dominant_signed_map"],
                    "dark_dominant_signed_map": dark_summary["dominant_signed_map"],
                    "both_bright_dark_dominant_negative": bool(
                        bright_summary["dominant_negative_map"] and dark_summary["dominant_negative_map"]
                    ),
                    "both_mean_delta_negative": bool(
                        float(bright_summary["mean_delta"]) < 0.0 and float(dark_summary["mean_delta"]) < 0.0
                    ),
                    **{f"bright_{key}": value for key, value in bright_summary.items() if key != "dominant_signed_map"},
                    **{f"dark_{key}": value for key, value in dark_summary.items() if key != "dominant_signed_map"},
                }
            )

    csv_path = out_dir / "rr100_bright_dark_suppression_by_unit.csv"
    write_csv(csv_path, rows)
    result_df = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "polarity_csv": Path(args.polarity_csv),
        "out_dir": out_dir,
        "stimulus_normalization": STIMULUS_NORMALIZATION,
        "rr100_version": str(args.rr100_version),
        "n_frames": int(args.n_frames),
        "fd_step_arcmin": float(args.fd_step_arcmin),
        "maps_npz": map_npz,
        "unit_csv": csv_path,
        "counts": {},
    }
    for contract, sub in result_df.groupby("map_contract"):
        negative_sub = sub[sub["polarity_table_label"] == "negative"]
        positive_sub = sub[sub["polarity_table_label"] == "positive"]
        summary["counts"][str(contract)] = {
            "all_units": int(len(sub)),
            "polarity_negative_units": int(len(negative_sub)),
            "polarity_positive_units": int(len(positive_sub)),
            "negative_units_both_bright_dark_dominant_negative": int(
                negative_sub["both_bright_dark_dominant_negative"].sum()
            ),
            "negative_units_bright_dominant_negative": int(
                (negative_sub["bright_dominant_signed_map"] == "negative").sum()
            ),
            "negative_units_dark_dominant_negative": int(
                (negative_sub["dark_dominant_signed_map"] == "negative").sum()
            ),
            "negative_units_both_mean_delta_negative": int(negative_sub["both_mean_delta_negative"].sum()),
        }

    figure_path = out_dir / "rr100_negative_units_bright_dark_suppression_maps.png"
    plot_negative_unit_sheet(
        out_path=figure_path,
        unit_rows=negative_df,
        maps=maps,
        bright_delta=maps["bright_mean"] - maps["blank_mean"],
        dark_delta=maps["dark_mean"] - maps["blank_mean"],
        top_units=int(args.top_units),
        dpi=int(args.dpi),
    )
    summary["figure"] = figure_path

    summary_path = out_dir / "rr100_bright_dark_suppression_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")

    print(f"Saved unit CSV: {csv_path}", flush=True)
    print(f"Saved maps: {map_npz}", flush=True)
    print(f"Saved figure: {figure_path}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)
    for contract in ("mean", "terminal"):
        counts = summary["counts"][contract]
        print(
            f"{contract}: negative-labelled units with dominant negative bright+dark maps = "
            f"{counts['negative_units_both_bright_dark_dominant_negative']}/"
            f"{counts['polarity_negative_units']}; "
            f"bright negative = {counts['negative_units_bright_dominant_negative']}/"
            f"{counts['polarity_negative_units']}; "
            f"dark negative = {counts['negative_units_dark_dominant_negative']}/"
            f"{counts['polarity_negative_units']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
