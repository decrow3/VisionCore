#!/usr/bin/env python3
"""All-unit space-time SSI decomposition for cached BackImage RR100 maps.

The decomposition treats each unit's activation movie as a rate map over
``time x space`` and separates the joint single-spike information into:

    joint space-time SSI = temporal-only SSI + conditional spatial SSI

The conditional spatial term is the original displayed-movie instantaneous-map
metric: framewise spatial SSI accumulated over time with expected-spike weights.
The time-averaged-map SSI is reported as a blur/stability diagnostic, not as a
term in the decomposition identity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_instantaneous_unit_maps_latest_v1"
)
DEFAULT_CACHE = DEFAULT_RUN_DIR / "cache" / "backimage_rr100_instantaneous_unit_maps.npz"
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "population_ssi_summary" / "space_time_ssi_decomposition"
EPS = 1e-12

PALETTE = {
    "conditional_spatial_bits_per_spike": "#0072B2",  # blue
    "temporal_only_bits_per_spike": "#E69F00",  # orange
    "joint_spacetime_bits_per_spike": "#CC79A7",  # purple
    "time_averaged_map_bits_per_spike": "#666666",  # gray
}
MARKERS = {
    "conditional_spatial_bits_per_spike": "o",
    "temporal_only_bits_per_spike": "s",
    "joint_spacetime_bits_per_spike": "^",
    "time_averaged_map_bits_per_spike": "D",
}
LINESTYLES = {
    "conditional_spatial_bits_per_spike": "-",
    "temporal_only_bits_per_spike": "--",
    "joint_spacetime_bits_per_spike": "-.",
    "time_averaged_map_bits_per_spike": ":",
}
METRIC_LABELS = {
    "conditional_spatial_bits_per_spike": "conditional spatial",
    "temporal_only_bits_per_spike": "temporal-only",
    "joint_spacetime_bits_per_spike": "joint space-time",
    "time_averaged_map_bits_per_spike": "SSI of time-averaged map",
}
METRICS = tuple(METRIC_LABELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--reference-scale", type=float, default=1.0)
    parser.add_argument("--endpoint-scale", type=float, default=3.0)
    parser.add_argument("--dpi", type=int, default=220)
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
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def xlogx_ratio(ratio: np.ndarray) -> np.ndarray:
    arr = np.asarray(ratio, dtype=np.float64)
    out = np.zeros_like(arr, dtype=np.float64)
    positive = arr > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out[positive] = arr[positive] * np.log2(arr[positive])
    return out


def condition_axis_rows(along: np.ndarray, across: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition_index, (along_scale, across_scale) in enumerate(zip(along, across, strict=True)):
        if np.isclose(across_scale, 1.0):
            rows.append(
                {
                    "axis_mode": "along_sweep",
                    "display_scale": float(along_scale),
                    "condition_index": int(condition_index),
                }
            )
        if np.isclose(along_scale, 1.0):
            rows.append(
                {
                    "axis_mode": "across_sweep",
                    "display_scale": float(across_scale),
                    "condition_index": int(condition_index),
                }
            )
    return rows


def decompose_condition(condition_maps: np.ndarray) -> dict[str, np.ndarray]:
    """Return unit-level decomposition for one condition.

    Parameters
    ----------
    condition_maps:
        Activation movie with shape ``(T, N, H, W)``.
    """
    rates = np.clip(np.asarray(condition_maps, dtype=np.float64), 0.0, None)
    if rates.ndim != 4:
        raise ValueError(f"Expected condition_maps with shape (T,N,H,W), got {rates.shape}")
    t_max, n_units, height, width = rates.shape
    flat = rates.reshape(t_max, n_units, height * width)
    frame_mean = np.mean(flat, axis=2)
    global_mean = np.mean(frame_mean, axis=0)
    expected_spikes = np.sum(frame_mean, axis=0)

    frame_gain = np.divide(
        flat,
        frame_mean[:, :, None],
        out=np.zeros_like(flat, dtype=np.float64),
        where=frame_mean[:, :, None] > EPS,
    )
    frame_ssi = np.mean(xlogx_ratio(frame_gain), axis=2)
    conditional_spatial = np.divide(
        np.sum(frame_mean * frame_ssi, axis=0),
        expected_spikes,
        out=np.full(n_units, np.nan, dtype=np.float64),
        where=expected_spikes > EPS,
    )

    temporal_gain = np.divide(
        frame_mean,
        global_mean[None, :],
        out=np.zeros_like(frame_mean, dtype=np.float64),
        where=global_mean[None, :] > EPS,
    )
    temporal_only = np.mean(xlogx_ratio(temporal_gain), axis=0)

    joint_gain = np.divide(
        flat,
        global_mean[None, :, None],
        out=np.zeros_like(flat, dtype=np.float64),
        where=global_mean[None, :, None] > EPS,
    )
    joint_spacetime = np.mean(xlogx_ratio(joint_gain), axis=(0, 2))

    time_averaged_map = np.mean(flat, axis=0)
    time_avg_gain = np.divide(
        time_averaged_map,
        global_mean[:, None],
        out=np.zeros_like(time_averaged_map, dtype=np.float64),
        where=global_mean[:, None] > EPS,
    )
    time_averaged_map_ssi = np.mean(xlogx_ratio(time_avg_gain), axis=1)

    cv_time_mean_rate = np.divide(
        np.std(frame_mean, axis=0),
        global_mean,
        out=np.full(n_units, np.nan, dtype=np.float64),
        where=global_mean > EPS,
    )
    identity_residual = joint_spacetime - temporal_only - conditional_spatial

    return {
        "conditional_spatial_bits_per_spike": conditional_spatial,
        "temporal_only_bits_per_spike": temporal_only,
        "joint_spacetime_bits_per_spike": joint_spacetime,
        "time_averaged_map_bits_per_spike": time_averaged_map_ssi,
        "identity_residual_bits_per_spike": identity_residual,
        "mean_rate": global_mean,
        "expected_spikes_arbitrary_dt": expected_spikes,
        "cv_time_mean_rate": cv_time_mean_rate,
        "mean_frame_spatial_ssi_unweighted": np.mean(frame_ssi, axis=0),
        "max_frame_spatial_ssi": np.max(frame_ssi, axis=0),
    }


def build_unit_table(cache_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    z = np.load(cache_path, allow_pickle=True)
    maps = np.asarray(z["maps"], dtype=np.float32)
    condition_id = z["condition_id"].astype(str)
    along = z["condition_along_scale"].astype(float)
    across = z["condition_across_scale"].astype(float)
    n_conditions, _n_time, n_units, _height, _width = maps.shape

    axis_rows = condition_axis_rows(along, across)
    condition_metrics = [decompose_condition(maps[condition_index]) for condition_index in range(n_conditions)]
    rows: list[dict[str, Any]] = []
    for axis_row in axis_rows:
        condition_index = int(axis_row["condition_index"])
        metrics = condition_metrics[condition_index]
        for unit_index in range(n_units):
            row = {
                "unit_index": int(unit_index),
                "unit_label": f"u{unit_index:03d}",
                "axis_mode": str(axis_row["axis_mode"]),
                "display_scale": float(axis_row["display_scale"]),
                "condition_index": condition_index,
                "condition_id": str(condition_id[condition_index]),
                "along_scale": float(along[condition_index]),
                "across_scale": float(across[condition_index]),
            }
            for key, values in metrics.items():
                row[key] = float(values[unit_index])
            rows.append(row)
    meta = {
        "cache_path": cache_path,
        "maps_shape": maps.shape,
        "n_conditions": int(n_conditions),
        "n_units": int(n_units),
    }
    return pd.DataFrame(rows), meta


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    if not bool(valid.any()):
        return float("nan")
    return float(np.sum(values[valid] * weights[valid]) / max(float(np.sum(weights[valid])), EPS))


def finite_quantile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.nanquantile(arr, q)) if arr.size else float("nan")


def build_population_summary(unit_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (axis_mode, display_scale), sub in unit_df.groupby(["axis_mode", "display_scale"], sort=True):
        weights = sub["expected_spikes_arbitrary_dt"].to_numpy(dtype=np.float64)
        row: dict[str, Any] = {
            "axis_mode": str(axis_mode),
            "display_scale": float(display_scale),
            "condition_index": int(sub["condition_index"].iloc[0]),
            "condition_id": str(sub["condition_id"].iloc[0]),
            "along_scale": float(sub["along_scale"].iloc[0]),
            "across_scale": float(sub["across_scale"].iloc[0]),
            "n_units": int(sub["unit_index"].nunique()),
            "population_expected_spikes_arbitrary_dt": float(np.nansum(weights)),
        }
        for metric in METRICS:
            values = sub[metric].to_numpy(dtype=np.float64)
            row[f"population_{metric}"] = weighted_mean(values, weights)
            row[f"equal_unit_mean_{metric}"] = float(np.nanmean(values))
            row[f"median_unit_{metric}"] = float(np.nanmedian(values))
            row[f"q25_unit_{metric}"] = finite_quantile(values, 0.25)
            row[f"q75_unit_{metric}"] = finite_quantile(values, 0.75)
        row["max_abs_identity_residual_bits_per_spike"] = float(
            np.nanmax(np.abs(sub["identity_residual_bits_per_spike"].to_numpy(dtype=np.float64)))
        )
        row["population_mean_rate"] = weighted_mean(
            sub["mean_rate"].to_numpy(dtype=np.float64),
            np.ones(len(sub), dtype=np.float64),
        )
        row["mean_cv_time_mean_rate"] = float(np.nanmean(sub["cv_time_mean_rate"].to_numpy(dtype=np.float64)))
        rows.append(row)
    return pd.DataFrame(rows)


def build_endpoint_summary(unit_df: pd.DataFrame, population: pd.DataFrame, reference_scale: float, endpoint_scale: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for axis_mode in sorted(unit_df["axis_mode"].unique()):
        unit_ref = unit_df[(unit_df["axis_mode"] == axis_mode) & np.isclose(unit_df["display_scale"].astype(float), reference_scale)]
        unit_end = unit_df[(unit_df["axis_mode"] == axis_mode) & np.isclose(unit_df["display_scale"].astype(float), endpoint_scale)]
        pair = unit_end.merge(
            unit_ref[["unit_index", *METRICS]],
            on="unit_index",
            suffixes=("_endpoint", "_reference"),
            validate="one_to_one",
        )
        pop_ref = population[
            (population["axis_mode"] == axis_mode)
            & np.isclose(population["display_scale"].astype(float), reference_scale)
        ].iloc[0]
        pop_end = population[
            (population["axis_mode"] == axis_mode)
            & np.isclose(population["display_scale"].astype(float), endpoint_scale)
        ].iloc[0]
        row: dict[str, Any] = {
            "axis_mode": str(axis_mode),
            "reference_scale": float(reference_scale),
            "endpoint_scale": float(endpoint_scale),
            "n_units": int(pair["unit_index"].nunique()),
        }
        for metric in METRICS:
            delta = pair[f"{metric}_endpoint"].to_numpy(dtype=np.float64) - pair[f"{metric}_reference"].to_numpy(
                dtype=np.float64
            )
            row[f"population_delta_{metric}"] = float(
                pop_end[f"population_{metric}"] - pop_ref[f"population_{metric}"]
            )
            row[f"equal_unit_mean_delta_{metric}"] = float(np.nanmean(delta))
            row[f"median_unit_delta_{metric}"] = float(np.nanmedian(delta))
            row[f"q25_unit_delta_{metric}"] = finite_quantile(delta, 0.25)
            row[f"q75_unit_delta_{metric}"] = finite_quantile(delta, 0.75)
            row[f"fraction_units_positive_delta_{metric}"] = float(np.nanmean(delta[np.isfinite(delta)] > 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_population_curves(population: pd.DataFrame, out_dir: Path, *, dpi: int) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, constrained_layout=True)
    for ax, axis_mode in zip(axes, ["across_sweep", "along_sweep"], strict=True):
        sub = population[population["axis_mode"] == axis_mode].sort_values("display_scale")
        for metric in METRICS:
            ax.plot(
                sub["display_scale"].to_numpy(dtype=float),
                sub[f"population_{metric}"].to_numpy(dtype=float),
                color=PALETTE[metric],
                marker=MARKERS[metric],
                linestyle=LINESTYLES[metric],
                linewidth=2.0,
                label=METRIC_LABELS[metric],
            )
        ax.axvline(1.0, color="0.55", linestyle=":", linewidth=1.0)
        ax.grid(True, alpha=0.25)
        ax.set_title(axis_mode.replace("_", " "))
        ax.set_xlabel("display scale")
        ax.set_ylabel("population bits/spike")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    png = out_dir / "backimage_rr100_space_time_ssi_population_curves.png"
    pdf = out_dir / "backimage_rr100_space_time_ssi_population_curves.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_population_delta_curves(population: pd.DataFrame, out_dir: Path, *, reference_scale: float, dpi: int) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, constrained_layout=True)
    for ax, axis_mode in zip(axes, ["across_sweep", "along_sweep"], strict=True):
        sub = population[population["axis_mode"] == axis_mode].sort_values("display_scale")
        for metric in METRICS:
            ref = sub[np.isclose(sub["display_scale"].astype(float), reference_scale)]
            if ref.empty:
                continue
            baseline = float(ref[f"population_{metric}"].iloc[0])
            ax.plot(
                sub["display_scale"].to_numpy(dtype=float),
                sub[f"population_{metric}"].to_numpy(dtype=float) - baseline,
                color=PALETTE[metric],
                marker=MARKERS[metric],
                linestyle=LINESTYLES[metric],
                linewidth=2.0,
                label=METRIC_LABELS[metric],
            )
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        ax.axvline(reference_scale, color="0.55", linestyle=":", linewidth=1.0)
        ax.grid(True, alpha=0.25)
        ax.set_title(axis_mode.replace("_", " "))
        ax.set_xlabel("display scale")
        ax.set_ylabel(f"delta from {reference_scale:g}x (bits/spike)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    png = out_dir / "backimage_rr100_space_time_ssi_population_delta_curves.png"
    pdf = out_dir / "backimage_rr100_space_time_ssi_population_delta_curves.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_endpoint_unit_deltas(unit_df: pd.DataFrame, out_dir: Path, *, reference_scale: float, endpoint_scale: float, dpi: int) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), sharey=True, constrained_layout=True)
    metric_order = list(METRICS)
    for ax, axis_mode in zip(axes, ["across_sweep", "along_sweep"], strict=True):
        ref = unit_df[(unit_df["axis_mode"] == axis_mode) & np.isclose(unit_df["display_scale"].astype(float), reference_scale)]
        end = unit_df[(unit_df["axis_mode"] == axis_mode) & np.isclose(unit_df["display_scale"].astype(float), endpoint_scale)]
        pair = end.merge(
            ref[["unit_index", *metric_order]],
            on="unit_index",
            suffixes=("_endpoint", "_reference"),
            validate="one_to_one",
        )
        positions = np.arange(len(metric_order))
        for idx, metric in enumerate(metric_order):
            delta = pair[f"{metric}_endpoint"].to_numpy(dtype=np.float64) - pair[f"{metric}_reference"].to_numpy(
                dtype=np.float64
            )
            jitter = np.linspace(-0.14, 0.14, delta.size)
            ax.scatter(
                np.full(delta.size, positions[idx]) + jitter,
                delta,
                s=16,
                color=PALETTE[metric],
                alpha=0.36,
                edgecolors="none",
            )
            q25, med, q75 = np.nanquantile(delta, [0.25, 0.5, 0.75])
            ax.plot([positions[idx] - 0.22, positions[idx] + 0.22], [med, med], color="black", linewidth=2.0)
            ax.plot([positions[idx], positions[idx]], [q25, q75], color="black", linewidth=1.4)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels([METRIC_LABELS[m] for m in metric_order], rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_title(axis_mode.replace("_", " "))
        ax.set_ylabel(f"unit delta {endpoint_scale:g}x - {reference_scale:g}x (bits/spike)")
    png = out_dir / "backimage_rr100_space_time_ssi_unit_delta_distributions.png"
    pdf = out_dir / "backimage_rr100_space_time_ssi_unit_delta_distributions.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_spatial_temporal_delta_scatter(
    unit_df: pd.DataFrame,
    out_dir: Path,
    *,
    reference_scale: float,
    endpoint_scale: float,
    dpi: int,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharex=True, sharey=True, constrained_layout=True)
    for ax, axis_mode in zip(axes, ["across_sweep", "along_sweep"], strict=True):
        ref = unit_df[(unit_df["axis_mode"] == axis_mode) & np.isclose(unit_df["display_scale"].astype(float), reference_scale)]
        end = unit_df[(unit_df["axis_mode"] == axis_mode) & np.isclose(unit_df["display_scale"].astype(float), endpoint_scale)]
        pair = end.merge(
            ref[["unit_index", "conditional_spatial_bits_per_spike", "temporal_only_bits_per_spike"]],
            on="unit_index",
            suffixes=("_endpoint", "_reference"),
            validate="one_to_one",
        )
        spatial_delta = pair["conditional_spatial_bits_per_spike_endpoint"].to_numpy(dtype=np.float64) - pair[
            "conditional_spatial_bits_per_spike_reference"
        ].to_numpy(dtype=np.float64)
        temporal_delta = pair["temporal_only_bits_per_spike_endpoint"].to_numpy(dtype=np.float64) - pair[
            "temporal_only_bits_per_spike_reference"
        ].to_numpy(dtype=np.float64)
        ax.scatter(
            spatial_delta,
            temporal_delta,
            s=34,
            color="#0072B2",
            alpha=0.58,
            edgecolor="white",
            linewidth=0.35,
        )
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.7)
        ax.grid(True, alpha=0.25)
        ax.set_title(axis_mode.replace("_", " "))
        ax.set_xlabel("conditional spatial delta (bits/spike)")
        ax.set_ylabel("temporal-only delta (bits/spike)")
    png = out_dir / "backimage_rr100_space_time_ssi_spatial_vs_temporal_unit_deltas.png"
    pdf = out_dir / "backimage_rr100_space_time_ssi_spatial_vs_temporal_unit_deltas.pdf"
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unit_df, meta = build_unit_table(Path(args.cache))
    population = build_population_summary(unit_df)
    endpoint = build_endpoint_summary(
        unit_df,
        population,
        reference_scale=float(args.reference_scale),
        endpoint_scale=float(args.endpoint_scale),
    )

    unit_csv = out_dir / "space_time_ssi_decomposition_all_units.csv"
    population_csv = out_dir / "space_time_ssi_decomposition_population_summary.csv"
    endpoint_csv = out_dir / "space_time_ssi_decomposition_endpoint_summary.csv"
    unit_df.to_csv(unit_csv, index=False)
    population.to_csv(population_csv, index=False)
    endpoint.to_csv(endpoint_csv, index=False)

    curves_png, curves_pdf = plot_population_curves(population, out_dir, dpi=int(args.dpi))
    delta_png, delta_pdf = plot_population_delta_curves(
        population,
        out_dir,
        reference_scale=float(args.reference_scale),
        dpi=int(args.dpi),
    )
    unit_delta_png, unit_delta_pdf = plot_endpoint_unit_deltas(
        unit_df,
        out_dir,
        reference_scale=float(args.reference_scale),
        endpoint_scale=float(args.endpoint_scale),
        dpi=int(args.dpi),
    )
    scatter_png, scatter_pdf = plot_spatial_temporal_delta_scatter(
        unit_df,
        out_dir,
        reference_scale=float(args.reference_scale),
        endpoint_scale=float(args.endpoint_scale),
        dpi=int(args.dpi),
    )

    summary_json = out_dir / "summary.json"
    write_json(
        summary_json,
        {
            "analysis": "backimage_rr100_space_time_ssi_decomposition",
            **meta,
            "reference_scale": float(args.reference_scale),
            "endpoint_scale": float(args.endpoint_scale),
            "definitions": {
                "joint_spacetime_bits_per_spike": "E_t,x [(r(t,x)/mean_t,x r) log2(r(t,x)/mean_t,x r)]",
                "temporal_only_bits_per_spike": "E_t [(mean_x r(t,x)/mean_t,x r) log2(mean_x r(t,x)/mean_t,x r)]",
                "conditional_spatial_bits_per_spike": (
                    "sum_t mean_x r(t,x) * SSI_space(t) / sum_t mean_x r(t,x); "
                    "the original displayed-movie instantaneous-map metric"
                ),
                "time_averaged_map_bits_per_spike": "SSI of mean_t r(t,x); blur/stability diagnostic, not identity term",
                "identity": "joint_spacetime = temporal_only + conditional_spatial per unit, up to numerical precision",
                "population_aggregation": "spike-weighted across units using total expected spikes over the movie",
            },
            "outputs": {
                "unit_csv": unit_csv,
                "population_csv": population_csv,
                "endpoint_csv": endpoint_csv,
                "curves_png": curves_png,
                "curves_pdf": curves_pdf,
                "delta_curves_png": delta_png,
                "delta_curves_pdf": delta_pdf,
                "unit_delta_distribution_png": unit_delta_png,
                "unit_delta_distribution_pdf": unit_delta_pdf,
                "spatial_temporal_delta_scatter_png": scatter_png,
                "spatial_temporal_delta_scatter_pdf": scatter_pdf,
            },
        },
    )

    print(f"Wrote unit table: {unit_csv}")
    print(f"Wrote population summary: {population_csv}")
    print(f"Wrote endpoint summary: {endpoint_csv}")
    print(f"Wrote figures to: {out_dir}")
    print(endpoint.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
