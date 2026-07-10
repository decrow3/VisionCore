#!/usr/bin/env python3
"""Diagnose near-zero static SSI denominators for RR100 along=0 unit curves."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from declan.vernier_active_sensing.plot_rr100_along0_polarity_group_averages import (
    DEFAULT_ENDPOINT_UNIT_DIR,
    DEFAULT_REAL_TRACE_UNIT_DIR,
    cache_path,
    classify_unit_polarity,
    condition_sequence,
    json_ready,
    load_stats_by_condition,
    parse_scale_list,
)
from declan.vernier_active_sensing.plot_rr100_endpoint_along0_unit_ssi import summarize_units
from declan.vernier_active_sensing.run_rr100_real_trace_scale_grid import DEFAULT_SCALES


EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["both", "real_trace", "endpoint"], default="both")
    parser.add_argument("--real-trace-dir", type=Path, default=DEFAULT_REAL_TRACE_UNIT_DIR)
    parser.add_argument("--endpoint-dir", type=Path, default=DEFAULT_ENDPOINT_UNIT_DIR)
    parser.add_argument("--across-scales", type=str, default=DEFAULT_SCALES)
    parser.add_argument("--along-scale", type=float, default=0.0)
    parser.add_argument("--fd-step-arcmin", type=float, default=0.25)
    parser.add_argument("--real-trace-max-frames", type=int, default=60)
    parser.add_argument("--denominator-floor-bits", type=float, default=0.01)
    parser.add_argument(
        "--static-ssi-thresholds",
        type=str,
        default="0,0.001,0.005,0.01,0.02,0.05",
        help="Comma-separated static SSI floors for leave-out threshold sweeps.",
    )
    parser.add_argument("--focus-scale", type=float, default=3.0)
    parser.add_argument("--annotate-top-negative", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def output_prefix(mode: str) -> str:
    if mode == "real_trace":
        return "rr100_real_trace_along0"
    if mode == "endpoint":
        return "rr100_endpoint_along0"
    raise ValueError(f"Unknown mode: {mode}")


def load_or_classify_polarity(
    out_dir: Path,
    *,
    prefix: str,
    static_maps: np.ndarray,
) -> pd.DataFrame:
    path = out_dir / f"{prefix}_polarity_unit_table.csv"
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return classify_unit_polarity(static_maps, low_percentile=5.0, high_percentile=95.0)


def _mean_sem(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(arr, axis=axis)
    n = arr.shape[axis]
    if n > 1:
        sem = np.nanstd(arr, axis=axis, ddof=1) / math.sqrt(float(n))
    else:
        sem = np.zeros_like(mean)
    return mean, sem


def condition_metrics(stats: dict[str, Any]) -> dict[str, np.ndarray | float]:
    bits = np.asarray(stats["unit_bits_per_trace"], dtype=np.float64)
    rates = np.asarray(stats["unit_mean_rate_per_trace"], dtype=np.float64)
    pop = np.asarray(stats["population_bits_per_trace"], dtype=np.float64)
    budget = np.sum(rates * bits, axis=1)
    total_rate = np.sum(rates, axis=1)
    bits_mean, bits_sem = _mean_sem(bits, axis=0)
    rates_mean, rates_sem = _mean_sem(rates, axis=0)
    budget_unit = rates * bits
    budget_unit_mean, budget_unit_sem = _mean_sem(budget_unit, axis=0)
    pop_mean, pop_sem = _mean_sem(pop, axis=0)
    budget_mean, budget_sem = _mean_sem(budget, axis=0)
    total_rate_mean, total_rate_sem = _mean_sem(total_rate, axis=0)
    return {
        "unit_bits_mean": bits_mean,
        "unit_bits_sem": bits_sem,
        "unit_rates_mean": rates_mean,
        "unit_rates_sem": rates_sem,
        "unit_budget_mean": budget_unit_mean,
        "unit_budget_sem": budget_unit_sem,
        "population_bits_mean": float(pop_mean),
        "population_bits_sem": float(pop_sem),
        "budget_proxy_mean": float(budget_mean),
        "budget_proxy_sem": float(budget_sem),
        "total_rate_mean": float(total_rate_mean),
        "total_rate_sem": float(total_rate_sem),
    }


def build_tables(
    *,
    mode: str,
    out_dir: Path,
    rows: list[dict[str, Any]],
    stats_by_condition: dict[str, dict[str, Any]],
    denominator_floor_bits: float,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unit_df, top_df, diagnostics = summarize_units(stats_by_condition, rows)
    polarity_df = load_or_classify_polarity(
        out_dir,
        prefix=prefix,
        static_maps=np.asarray(stats_by_condition["static_center"]["mean_rate_map"], dtype=np.float32),
    )[["unit_index", "polarity", "polarity_score_positive_minus_negative"]]
    top_small = top_df[
        [
            "unit_index",
            "max_abs_log2_unit_ssi_vs_static",
            "max_abs_leave_one_out_population_ratio_delta",
        ]
    ]
    unit_meta = polarity_df.merge(top_small, on="unit_index", how="left")

    metrics_by_condition = {condition: condition_metrics(stats) for condition, stats in stats_by_condition.items()}
    static_metrics = metrics_by_condition["static_center"]
    nonstatic_rows = [row for row in rows if not bool(row["is_static_baseline"])]
    zero_condition = next(
        (str(row["condition"]) for row in nonstatic_rows if np.isclose(float(row["across_scale"]), 0.0)),
        str(nonstatic_rows[0]["condition"]),
    )
    zero_metrics = metrics_by_condition[zero_condition]
    static_bits = np.asarray(static_metrics["unit_bits_mean"], dtype=np.float64)
    zero_bits = np.asarray(zero_metrics["unit_bits_mean"], dtype=np.float64)
    static_budget = np.asarray(static_metrics["unit_budget_mean"], dtype=np.float64)
    zero_budget = np.asarray(zero_metrics["unit_budget_mean"], dtype=np.float64)
    static_rate = np.asarray(static_metrics["unit_rates_mean"], dtype=np.float64)
    zero_rate = np.asarray(zero_metrics["unit_rates_mean"], dtype=np.float64)

    long_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    unit_indices = np.arange(static_bits.size)

    for row in nonstatic_rows:
        condition = str(row["condition"])
        across = float(row["across_scale"])
        metrics = metrics_by_condition[condition]
        bits = np.asarray(metrics["unit_bits_mean"], dtype=np.float64)
        bits_sem = np.asarray(metrics["unit_bits_sem"], dtype=np.float64)
        rates = np.asarray(metrics["unit_rates_mean"], dtype=np.float64)
        budget = np.asarray(metrics["unit_budget_mean"], dtype=np.float64)
        log2_ratio_static = np.log2((bits + EPS) / (static_bits + EPS))
        log2_ratio_zero = np.log2((bits + EPS) / (zero_bits + EPS))
        log2_ratio_static_floor = np.log2(
            (bits + float(denominator_floor_bits)) / (static_bits + float(denominator_floor_bits))
        )
        log2_ratio_zero_floor = np.log2(
            (bits + float(denominator_floor_bits)) / (zero_bits + float(denominator_floor_bits))
        )
        for unit_index in unit_indices:
            unit_int = int(unit_index)
            meta = unit_meta[unit_meta["unit_index"].astype(int).eq(unit_int)]
            meta_row = meta.iloc[0].to_dict() if not meta.empty else {}
            long_rows.append(
                {
                    "mode": mode,
                    "condition": condition,
                    "across_scale": across,
                    "along_scale": 0.0,
                    "unit_index": unit_int,
                    "polarity": meta_row.get("polarity", "unknown"),
                    "polarity_score_positive_minus_negative": meta_row.get(
                        "polarity_score_positive_minus_negative", np.nan
                    ),
                    "max_abs_log2_unit_ssi_vs_static": meta_row.get("max_abs_log2_unit_ssi_vs_static", np.nan),
                    "max_abs_leave_one_out_population_ratio_delta": meta_row.get(
                        "max_abs_leave_one_out_population_ratio_delta", np.nan
                    ),
                    "static_ssi_bits_per_spike_mean": float(static_bits[unit_int]),
                    "zero_x_ssi_bits_per_spike_mean": float(zero_bits[unit_int]),
                    "unit_ssi_bits_per_spike_mean": float(bits[unit_int]),
                    "unit_ssi_bits_per_spike_sem": float(bits_sem[unit_int]),
                    "delta_ssi_vs_static": float(bits[unit_int] - static_bits[unit_int]),
                    "delta_ssi_vs_0x": float(bits[unit_int] - zero_bits[unit_int]),
                    "log2_ratio_vs_static": float(log2_ratio_static[unit_int]),
                    "log2_ratio_vs_0x": float(log2_ratio_zero[unit_int]),
                    "log2_ratio_vs_static_floor": float(log2_ratio_static_floor[unit_int]),
                    "log2_ratio_vs_0x_floor": float(log2_ratio_zero_floor[unit_int]),
                    "denominator_floor_bits": float(denominator_floor_bits),
                    "static_mean_rate_mean": float(static_rate[unit_int]),
                    "zero_x_mean_rate_mean": float(zero_rate[unit_int]),
                    "unit_mean_rate_mean": float(rates[unit_int]),
                    "static_budget_proxy_mean": float(static_budget[unit_int]),
                    "zero_x_budget_proxy_mean": float(zero_budget[unit_int]),
                    "unit_budget_proxy_mean": float(budget[unit_int]),
                    "delta_budget_proxy_vs_static": float(budget[unit_int] - static_budget[unit_int]),
                    "delta_budget_proxy_vs_0x": float(budget[unit_int] - zero_budget[unit_int]),
                }
            )

        summary_rows.append(
            {
                "mode": mode,
                "condition": condition,
                "across_scale": across,
                "along_scale": 0.0,
                "population_ssi_bits_per_spike_mean": float(metrics["population_bits_mean"]),
                "population_ssi_bits_per_spike_sem": float(metrics["population_bits_sem"]),
                "static_population_ssi_bits_per_spike_mean": float(static_metrics["population_bits_mean"]),
                "zero_x_population_ssi_bits_per_spike_mean": float(zero_metrics["population_bits_mean"]),
                "delta_population_ssi_vs_static": float(
                    metrics["population_bits_mean"] - static_metrics["population_bits_mean"]
                ),
                "delta_population_ssi_vs_0x": float(metrics["population_bits_mean"] - zero_metrics["population_bits_mean"]),
                "budget_proxy_mean": float(metrics["budget_proxy_mean"]),
                "budget_proxy_sem": float(metrics["budget_proxy_sem"]),
                "static_budget_proxy_mean": float(static_metrics["budget_proxy_mean"]),
                "zero_x_budget_proxy_mean": float(zero_metrics["budget_proxy_mean"]),
                "delta_budget_proxy_vs_static": float(metrics["budget_proxy_mean"] - static_metrics["budget_proxy_mean"]),
                "delta_budget_proxy_vs_0x": float(metrics["budget_proxy_mean"] - zero_metrics["budget_proxy_mean"]),
                "total_rate_mean": float(metrics["total_rate_mean"]),
                "mean_unit_log2_ratio_vs_static": float(np.nanmean(log2_ratio_static)),
                "mean_unit_log2_ratio_vs_static_floor": float(np.nanmean(log2_ratio_static_floor)),
                "mean_unit_delta_ssi_vs_static": float(np.nanmean(bits - static_bits)),
                "sum_unit_delta_ssi_vs_static": float(np.nansum(bits - static_bits)),
                "mean_unit_delta_ssi_vs_0x": float(np.nanmean(bits - zero_bits)),
                "sum_unit_delta_ssi_vs_0x": float(np.nansum(bits - zero_bits)),
                "denominator_floor_bits": float(denominator_floor_bits),
            }
        )

        for group_name in ["all", "positive", "negative"]:
            if group_name == "all":
                mask = np.ones_like(unit_indices, dtype=bool)
            else:
                group_units = unit_meta[unit_meta["polarity"].eq(group_name)]["unit_index"].astype(int).to_numpy()
                mask = np.isin(unit_indices, group_units)
            group_rows.append(
                {
                    "mode": mode,
                    "polarity": group_name,
                    "condition": condition,
                    "across_scale": across,
                    "along_scale": 0.0,
                    "n_units": int(np.sum(mask)),
                    "sum_unit_ssi_bits_per_spike": float(np.nansum(bits[mask])),
                    "sum_static_unit_ssi_bits_per_spike": float(np.nansum(static_bits[mask])),
                    "sum_zero_x_unit_ssi_bits_per_spike": float(np.nansum(zero_bits[mask])),
                    "sum_delta_ssi_vs_static": float(np.nansum(bits[mask] - static_bits[mask])),
                    "sum_delta_ssi_vs_0x": float(np.nansum(bits[mask] - zero_bits[mask])),
                    "mean_unit_log2_ratio_vs_static": float(np.nanmean(log2_ratio_static[mask])),
                    "mean_unit_log2_ratio_vs_static_floor": float(np.nanmean(log2_ratio_static_floor[mask])),
                    "mean_unit_delta_ssi_vs_static": float(np.nanmean(bits[mask] - static_bits[mask])),
                    "mean_unit_delta_ssi_vs_0x": float(np.nanmean(bits[mask] - zero_bits[mask])),
                    "sum_unit_budget_proxy": float(np.nansum(budget[mask])),
                    "sum_static_unit_budget_proxy": float(np.nansum(static_budget[mask])),
                    "sum_zero_x_unit_budget_proxy": float(np.nansum(zero_budget[mask])),
                    "sum_delta_budget_proxy_vs_static": float(np.nansum(budget[mask] - static_budget[mask])),
                    "sum_delta_budget_proxy_vs_0x": float(np.nansum(budget[mask] - zero_budget[mask])),
                    "denominator_floor_bits": float(denominator_floor_bits),
                }
            )

    long_df = pd.DataFrame(long_rows)
    group_df = pd.DataFrame(group_rows)
    summary_df = pd.DataFrame(summary_rows)
    diagnostics_df = pd.DataFrame(
        {
            "unit_index": np.arange(static_bits.size, dtype=int),
            "static_ssi_bits_per_spike_mean": static_bits,
            "zero_x_ssi_bits_per_spike_mean": zero_bits,
            "static_budget_proxy_mean": static_budget,
            "zero_x_budget_proxy_mean": zero_budget,
        }
    ).merge(unit_meta, on="unit_index", how="left")
    diagnostics_df["mode"] = mode
    diagnostics_df["static_population_ssi_bits_per_spike_mean"] = float(diagnostics["static_population_ssi_bits_per_spike_mean"])
    return long_df, group_df, summary_df, diagnostics_df


def build_static_threshold_sweep(
    long_df: pd.DataFrame,
    *,
    thresholds: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name in ["all", "positive", "negative"]:
        base = long_df if group_name == "all" else long_df[long_df["polarity"].eq(group_name)]
        for threshold in thresholds:
            kept = base[base["static_ssi_bits_per_spike_mean"].ge(float(threshold))]
            for across in sorted(kept["across_scale"].dropna().unique()):
                sub = kept[np.isclose(kept["across_scale"], float(across))]
                if sub.empty:
                    continue
                log_mean = float(np.nanmean(sub["log2_ratio_vs_static"]))
                floor_log_mean = float(np.nanmean(sub["log2_ratio_vs_static_floor"]))
                rows.append(
                    {
                        "mode": str(sub.iloc[0]["mode"]),
                        "polarity": group_name,
                        "static_ssi_min_bits": float(threshold),
                        "across_scale": float(across),
                        "along_scale": 0.0,
                        "n_units": int(sub["unit_index"].nunique()),
                        "mean_log2_ratio_vs_static": log_mean,
                        "geometric_mean_ratio_vs_static": float(2.0**log_mean),
                        "mean_log2_ratio_vs_static_floor": floor_log_mean,
                        "geometric_mean_ratio_vs_static_floor": float(2.0**floor_log_mean),
                        "arithmetic_mean_ratio_vs_static": float(
                            np.nanmean(
                                (sub["unit_ssi_bits_per_spike_mean"] + EPS)
                                / (sub["static_ssi_bits_per_spike_mean"] + EPS)
                            )
                        ),
                        "mean_delta_ssi_vs_static": float(np.nanmean(sub["delta_ssi_vs_static"])),
                        "sum_delta_ssi_vs_static": float(np.nansum(sub["delta_ssi_vs_static"])),
                        "sum_static_ssi_bits_per_spike": float(np.nansum(sub["static_ssi_bits_per_spike_mean"])),
                        "sum_current_ssi_bits_per_spike": float(np.nansum(sub["unit_ssi_bits_per_spike_mean"])),
                    }
                )
    return pd.DataFrame(rows)


def draw_threshold_sweep_figure(
    *,
    mode: str,
    out_dir: Path,
    prefix: str,
    threshold_df: pd.DataFrame,
    focus_scale: float,
    dpi: int,
) -> Path:
    focus = threshold_df[np.isclose(threshold_df["across_scale"], float(focus_scale))].copy()
    if focus.empty:
        closest = float(
            threshold_df.iloc[
                np.abs(threshold_df["across_scale"].to_numpy(dtype=float) - float(focus_scale)).argmin()
            ]["across_scale"]
        )
        focus = threshold_df[np.isclose(threshold_df["across_scale"], closest)].copy()
    actual_focus_scale = float(focus.iloc[0]["across_scale"])
    thresholds = sorted(focus["static_ssi_min_bits"].dropna().unique())
    x = np.arange(len(thresholds), dtype=float)
    labels = [f"{threshold:g}" for threshold in thresholds]
    colors = {"all": "#111111", "positive": "#555555", "negative": "#9a9a9a"}

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.8), dpi=int(dpi), constrained_layout=True)
    mode_label = "real-trace" if mode == "real_trace" else "endpoint-history"
    for group_name in ["all", "positive", "negative"]:
        sub = focus[focus["polarity"].eq(group_name)].sort_values("static_ssi_min_bits")
        if sub.empty:
            continue
        axes[0].plot(
            x[: len(sub)],
            sub["geometric_mean_ratio_vs_static"],
            marker="o",
            linewidth=1.9,
            color=colors[group_name],
            label=group_name,
        )
        axes[1].plot(
            x[: len(sub)],
            sub["sum_delta_ssi_vs_static"],
            marker="o",
            linewidth=1.9,
            color=colors[group_name],
            label=group_name,
        )
    axes[0].axhline(1.0, color="#777777", linestyle="--", linewidth=0.8)
    axes[1].axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_xlabel("minimum static unit SSI retained (bits/spike)")
        ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.65)
    axes[0].set_ylabel("geometric mean SSI ratio")
    axes[0].set_title(f"Fold-change after dropping weak-static units at {actual_focus_scale:g}x")
    axes[1].set_ylabel("sum delta unit SSI vs static")
    axes[1].set_title("Absolute SSI change of retained units")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"RR100 {mode_label} static-SSI threshold sweep", fontsize=12.0)
    png = out_dir / f"{prefix}_denominator_static_floor_sweep.png"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


def draw_diagnostic_figure(
    *,
    mode: str,
    out_dir: Path,
    prefix: str,
    long_df: pd.DataFrame,
    group_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    focus_scale: float,
    denominator_floor_bits: float,
    annotate_top_negative: int,
    dpi: int,
) -> Path:
    focus = long_df[np.isclose(long_df["across_scale"], float(focus_scale))].copy()
    if focus.empty:
        closest = float(
            long_df.iloc[np.abs(long_df["across_scale"].to_numpy(dtype=float) - float(focus_scale)).argmin()][
                "across_scale"
            ]
        )
        focus = long_df[np.isclose(long_df["across_scale"], closest)].copy()
    actual_focus_scale = float(focus.iloc[0]["across_scale"])
    x = focus["static_ssi_bits_per_spike_mean"].to_numpy(dtype=float)
    y = focus["log2_ratio_vs_static"].to_numpy(dtype=float)
    delta = focus["delta_ssi_vs_static"].to_numpy(dtype=float)
    max_abs_delta = max(float(np.nanmax(np.abs(delta))), EPS)
    norm = TwoSlopeNorm(vmin=-max_abs_delta, vcenter=0.0, vmax=max_abs_delta)
    sizes = 20.0 + 620.0 * np.sqrt(np.clip(np.abs(delta) / max_abs_delta, 0.0, 1.0))

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), dpi=int(dpi), constrained_layout=True)
    mode_label = "real-trace" if mode == "real_trace" else "endpoint-history"

    ax = axes[0, 0]
    for polarity, marker, edge in [("positive", "o", "#333333"), ("negative", "s", "#111111")]:
        sub = focus[focus["polarity"].eq(polarity)]
        if sub.empty:
            continue
        idx = sub.index.to_numpy()
        ax.scatter(
            sub["static_ssi_bits_per_spike_mean"],
            sub["log2_ratio_vs_static"],
            c=sub["delta_ssi_vs_static"],
            s=sizes[focus.index.get_indexer(idx)],
            cmap="coolwarm",
            norm=norm,
            marker=marker,
            edgecolors=edge,
            linewidths=0.45,
            alpha=0.86,
            label=polarity,
        )
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
    ax.axvline(float(denominator_floor_bits), color="#777777", linestyle=":", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("static unit SSI (bits/spike)")
    ax.set_ylabel("log2 SSI(scale) / SSI(static)")
    ax.set_title(f"Denominator diagnostic at across={actual_focus_scale:g}x")
    ax.grid(True, color="#e5e5e5", linewidth=0.65)
    ax.legend(frameon=False, fontsize=8, loc="best")
    top_neg = (
        focus[focus["polarity"].eq("negative")]
        .sort_values("max_abs_leave_one_out_population_ratio_delta", ascending=False)
        .head(int(annotate_top_negative))
    )
    for _, row in top_neg.iterrows():
        ax.annotate(
            f"u{int(row['unit_index']):03d}",
            (float(row["static_ssi_bits_per_spike_mean"]), float(row["log2_ratio_vs_static"])),
            xytext=(3.0, 3.0),
            textcoords="offset points",
            fontsize=6.3,
            color="#222222",
        )
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="coolwarm"),
        ax=ax,
        fraction=0.046,
        pad=0.02,
    )
    cbar.set_label("absolute delta SSI vs static")

    ax = axes[0, 1]
    for group_name, color in [("all", "#111111"), ("positive", "#555555"), ("negative", "#9a9a9a")]:
        sub = group_df[group_df["polarity"].eq(group_name)].sort_values("across_scale")
        ax.plot(
            sub["across_scale"],
            sub["mean_unit_log2_ratio_vs_static"],
            marker="o",
            linewidth=1.8,
            color=color,
            label=f"{group_name}, raw",
        )
        ax.plot(
            sub["across_scale"],
            sub["mean_unit_log2_ratio_vs_static_floor"],
            marker=".",
            linewidth=1.15,
            linestyle="--",
            color=color,
            alpha=0.82,
            label=f"{group_name}, floor",
        )
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
    ax.axvline(1.0, color="#999999", linestyle=":", linewidth=0.9)
    ax.set_xlabel("across-contour motion scale, along=0")
    ax.set_ylabel("mean unit log2 ratio")
    ax.set_title(f"Fold-change sensitivity to floor lambda={denominator_floor_bits:g}")
    ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.65)
    ax.legend(frameon=False, fontsize=6.7, ncols=2)

    ax = axes[1, 0]
    for group_name, color in [("positive", "#333333"), ("negative", "#888888")]:
        sub = group_df[group_df["polarity"].eq(group_name)].sort_values("across_scale")
        ax.plot(
            sub["across_scale"],
            sub["sum_unit_ssi_bits_per_spike"],
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"{group_name} sum SSI",
        )
        ax.axhline(
            float(sub.iloc[0]["sum_static_unit_ssi_bits_per_spike"]),
            color=color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.55,
        )
        ax.axhline(
            float(sub.iloc[0]["sum_zero_x_unit_ssi_bits_per_spike"]),
            color=color,
            linewidth=1.0,
            linestyle=":",
            alpha=0.55,
        )
    ax.axvline(1.0, color="#999999", linestyle=":", linewidth=0.9)
    ax.set_xlabel("across-contour motion scale, along=0")
    ax.set_ylabel("sum unit SSI (bits/spike)")
    ax.set_title("Absolute unit SSI budget by polarity")
    ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.65)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    summary = summary_df.sort_values("across_scale")
    ax.plot(
        summary["across_scale"],
        summary["population_ssi_bits_per_spike_mean"],
        marker="o",
        linewidth=2.0,
        color="#111111",
        label="population SSI bits/spike",
    )
    ax.axhline(
        float(summary.iloc[0]["static_population_ssi_bits_per_spike_mean"]),
        color="#111111",
        linestyle="--",
        linewidth=0.9,
        alpha=0.6,
        label="static population SSI",
    )
    ax.axhline(
        float(summary.iloc[0]["zero_x_population_ssi_bits_per_spike_mean"]),
        color="#111111",
        linestyle=":",
        linewidth=0.9,
        alpha=0.6,
        label="0x population SSI",
    )
    ax2 = ax.twinx()
    ax2.plot(
        summary["across_scale"],
        summary["budget_proxy_mean"],
        marker="s",
        linewidth=1.7,
        color="#666666",
        label="sum rate * unit SSI",
    )
    ax2.axhline(
        float(summary.iloc[0]["static_budget_proxy_mean"]),
        color="#666666",
        linestyle="--",
        linewidth=0.9,
        alpha=0.6,
        label="static budget proxy",
    )
    ax2.axhline(
        float(summary.iloc[0]["zero_x_budget_proxy_mean"]),
        color="#666666",
        linestyle=":",
        linewidth=0.9,
        alpha=0.6,
        label="0x budget proxy",
    )
    ax.axvline(1.0, color="#999999", linestyle=":", linewidth=0.9)
    ax.set_xlabel("across-contour motion scale, along=0")
    ax.set_ylabel("population SSI (bits/spike)")
    ax2.set_ylabel("rate-weighted budget proxy")
    ax.set_title("Absolute population quantities")
    ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.65)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=7.2, loc="best")

    fig.suptitle(
        f"RR100 {mode_label} along=0 denominator diagnostics",
        fontsize=12.5,
    )
    png = out_dir / f"{prefix}_denominator_diagnostics.png"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


def run_one(args: argparse.Namespace, mode: str) -> None:
    out_dir = Path(args.real_trace_dir if mode == "real_trace" else args.endpoint_dir)
    prefix = output_prefix(mode)
    across_scales = parse_scale_list(args.across_scales)
    rows = condition_sequence(across_scales, float(args.along_scale))
    stats_by_condition = load_stats_by_condition(
        mode=mode,
        out_dir=out_dir,
        rows=rows,
        fd_step_arcmin=float(args.fd_step_arcmin),
        real_trace_max_frames=int(args.real_trace_max_frames),
    )
    long_df, group_df, summary_df, diagnostics_df = build_tables(
        mode=mode,
        out_dir=out_dir,
        rows=rows,
        stats_by_condition=stats_by_condition,
        denominator_floor_bits=float(args.denominator_floor_bits),
        prefix=prefix,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    unit_csv = out_dir / f"{prefix}_denominator_diagnostic_units.csv"
    group_csv = out_dir / f"{prefix}_denominator_diagnostic_groups.csv"
    summary_csv = out_dir / f"{prefix}_denominator_diagnostic_summary.csv"
    diagnostics_csv = out_dir / f"{prefix}_denominator_diagnostic_unit_static_metadata.csv"
    threshold_csv = out_dir / f"{prefix}_denominator_static_floor_sweep.csv"
    long_df.to_csv(unit_csv, index=False)
    group_df.to_csv(group_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    diagnostics_df.to_csv(diagnostics_csv, index=False)
    threshold_df = build_static_threshold_sweep(
        long_df,
        thresholds=parse_scale_list(str(args.static_ssi_thresholds)),
    )
    threshold_df.to_csv(threshold_csv, index=False)
    png = draw_diagnostic_figure(
        mode=mode,
        out_dir=out_dir,
        prefix=prefix,
        long_df=long_df,
        group_df=group_df,
        summary_df=summary_df,
        focus_scale=float(args.focus_scale),
        denominator_floor_bits=float(args.denominator_floor_bits),
        annotate_top_negative=int(args.annotate_top_negative),
        dpi=int(args.dpi),
    )
    threshold_png = draw_threshold_sweep_figure(
        mode=mode,
        out_dir=out_dir,
        prefix=prefix,
        threshold_df=threshold_df,
        focus_scale=float(args.focus_scale),
        dpi=int(args.dpi),
    )
    manifest = out_dir / f"{prefix}_denominator_diagnostic_manifest.json"
    payload = {
        "analysis": f"{prefix}_denominator_diagnostics",
        "mode": mode,
        "figure_png": png,
        "threshold_sweep_png": threshold_png,
        "unit_csv": unit_csv,
        "group_csv": group_csv,
        "summary_csv": summary_csv,
        "diagnostics_csv": diagnostics_csv,
        "threshold_csv": threshold_csv,
        "focus_scale": float(args.focus_scale),
        "denominator_floor_bits": float(args.denominator_floor_bits),
        "static_ssi_thresholds": parse_scale_list(str(args.static_ssi_thresholds)),
        "interpretation": (
            "Compares unit-wise fold-change against absolute SSI and rate-weighted "
            "budget proxies to diagnose near-zero static denominators."
        ),
    }
    manifest.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {mode} denominator figure: {png}", flush=True)
    print(f"Wrote {mode} denominator unit table: {unit_csv}", flush=True)
    print(f"Wrote {mode} denominator group table: {group_csv}", flush=True)
    print(f"Wrote {mode} denominator summary table: {summary_csv}", flush=True)
    print(f"Wrote {mode} static-floor threshold table: {threshold_csv}", flush=True)


def main() -> None:
    args = parse_args()
    modes = ["real_trace", "endpoint"] if str(args.mode) == "both" else [str(args.mode)]
    for mode in modes:
        run_one(args, mode)


if __name__ == "__main__":
    main()
