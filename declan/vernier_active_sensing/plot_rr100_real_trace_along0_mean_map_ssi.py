#!/usr/bin/env python3
"""Cache-only RR100 along=0 SSI computed from time-averaged activation maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from declan.vernier_active_sensing.forward import STIMULUS_NORMALIZATION
from declan.vernier_active_sensing.plot_rr100_endpoint_along0_unit_ssi import (
    draw_leave_one_out,
    draw_unit_lines,
    draw_unit_lines_with_activation_rows,
    order_units_by_y_at_x,
    summarize_units,
    unit_ssi_single_frame,
)
from declan.vernier_active_sensing.plot_rr100_real_trace_along0_unit_ssi import (
    DEFAULT_OUT_DIR as DEFAULT_SOURCE_DIR,
    DEFAULT_SCALES,
    EPS,
    parse_scale_list,
    stats_cache_path,
    write_json,
)
from declan.vernier_active_sensing.run_rr100_real_trace_scale_grid import scale_token


DEFAULT_OUT_DIR = DEFAULT_SOURCE_DIR / "mean_map_ssi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--across-scales", type=str, default=DEFAULT_SCALES)
    parser.add_argument("--along-scale", type=float, default=0.0)
    parser.add_argument("--fd-step-arcmin", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--top-units", type=int, default=12)
    parser.add_argument(
        "--baseline",
        choices=("grid_0_0", "static_center"),
        default="grid_0_0",
        help="Normalization reference for SSI ratios, matching plot_rr100_real_trace_scale_grid_rows.py.",
    )
    parser.add_argument("--map-vmin-percentile", type=float, default=0.5)
    parser.add_argument("--map-vmax-percentile", type=float, default=99.5)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _condition_rows(across_scales: list[float], along_scale: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "condition": "static_center",
            "label": "static center",
            "across_scale": np.nan,
            "along_scale": float(along_scale),
            "is_static_baseline": True,
        }
    ]
    for across in across_scales:
        rows.append(
            {
                "condition": f"real_aniso_across_{scale_token(float(across))}_along_{scale_token(float(along_scale))}",
                "label": f"across {float(across):g}; along {float(along_scale):g}",
                "across_scale": float(across),
                "along_scale": float(along_scale),
                "is_static_baseline": False,
            }
        )
    return rows


def _baseline_condition(rows: list[dict[str, Any]], baseline: str) -> str:
    if str(baseline) == "static_center":
        return "static_center"
    for row in rows:
        if bool(row["is_static_baseline"]):
            continue
        if np.isclose(float(row["across_scale"]), 0.0):
            return str(row["condition"])
    raise ValueError("--baseline grid_0_0 requires 0 in --across-scales.")


def _baseline_label(baseline: str) -> str:
    if baseline == "grid_0_0":
        return "trace-mean static catalog"
    if baseline == "static_center":
        return "centered static oracle"
    return str(baseline)


def _rows_for_baseline(rows: list[dict[str, Any]], baseline_condition: str) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows]
    if baseline_condition != "static_center":
        out[0]["label"] = "trace-mean static catalog"
        out[0]["baseline_condition"] = baseline_condition
    return out


def _stats_for_baseline(
    stats_by_condition: dict[str, dict[str, Any]],
    baseline_condition: str,
) -> dict[str, dict[str, Any]]:
    if baseline_condition == "static_center":
        return stats_by_condition
    if baseline_condition not in stats_by_condition:
        raise KeyError(f"Missing baseline condition: {baseline_condition}")
    out = dict(stats_by_condition)
    out["static_center"] = stats_by_condition[baseline_condition]
    return out


def _load_mean_map_stats(source_dir: Path, rows: list[dict[str, Any]], fd_step_arcmin: float, max_frames: int) -> dict[str, dict[str, Any]]:
    stats_by_condition: dict[str, dict[str, Any]] = {}
    for row in rows:
        condition = str(row["condition"])
        path = stats_cache_path(source_dir, condition, float(fd_step_arcmin), int(max_frames))
        if not path.exists():
            raise FileNotFoundError(f"Missing cached mean map for {condition}: {path}")
        with np.load(path) as data:
            if "mean_rate_map" not in data:
                raise KeyError(f"Cache does not contain mean_rate_map: {path}")
            mean_map = np.asarray(data["mean_rate_map"], dtype=np.float32)
            std_map = np.asarray(data["std_rate_map"], dtype=np.float32) if "std_rate_map" in data else np.zeros_like(mean_map)
        ssi = unit_ssi_single_frame(mean_map, eps=EPS)
        stats_by_condition[condition] = {
            "unit_bits_per_trace": np.asarray(ssi["unit_bits_per_spike"], dtype=np.float32)[None, :],
            "unit_mean_rate_per_trace": np.asarray(ssi["unit_mean_rate"], dtype=np.float32)[None, :],
            "population_bits_per_trace": np.asarray([float(ssi["population_bits_per_spike"])], dtype=np.float32),
            "mean_rate_map": mean_map,
            "std_rate_map": std_map,
            "source_cache": str(path),
        }
    return stats_by_condition


def _population_summary(
    unit_df: pd.DataFrame,
    rows: list[dict[str, Any]],
    old_unit_csv: Path,
    *,
    baseline_condition: str,
) -> pd.DataFrame:
    old_df = pd.read_csv(old_unit_csv) if old_unit_csv.exists() else pd.DataFrame()
    old_pop_by_condition = {}
    if not old_df.empty:
        old_pop_by_condition = (
            old_df[["condition", "population_ssi_bits_per_spike_mean"]]
            .drop_duplicates("condition")
            .set_index("condition")["population_ssi_bits_per_spike_mean"]
            .astype(float)
            .to_dict()
        )
    old_ref_pop = old_pop_by_condition.get(baseline_condition, np.nan)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        condition = str(row["condition"])
        this = unit_df[unit_df["condition"].eq(condition)]
        pop = float(this["population_ssi_bits_per_spike_mean"].iloc[0]) if not this.empty else np.nan
        old_condition = baseline_condition if bool(row["is_static_baseline"]) and baseline_condition != "static_center" else condition
        old_pop = old_pop_by_condition.get(old_condition, np.nan)
        old_ratio = float(old_pop / old_ref_pop) if np.isfinite(old_pop) and np.isfinite(old_ref_pop) and old_ref_pop > 0 else np.nan
        out_rows.append(
            {
                "condition": condition,
                "label": row["label"],
                "across_scale": row["across_scale"],
                "along_scale": row["along_scale"],
                "is_static_baseline": bool(row["is_static_baseline"]),
                "baseline_condition": baseline_condition,
                "condition_stat_source": old_condition,
                "population_ssi_bits_per_spike_mean_map": pop,
                "population_ssi_vs_static_mean_map": float(this["population_ssi_vs_static"].iloc[0]) if not this.empty else np.nan,
                "population_ssi_bits_per_spike_frame_mean": old_pop,
                "population_ssi_vs_static_frame_mean": old_ratio,
                "mean_map_minus_frame_mean_bits": pop - old_pop if np.isfinite(old_pop) else np.nan,
            }
        )
    return pd.DataFrame(out_rows)


def _write_comparison_plot(pop_df: pd.DataFrame, path: Path, dpi: int, *, baseline_label: str) -> None:
    import matplotlib.pyplot as plt

    grid = pop_df[~pop_df["is_static_baseline"].astype(bool)].copy()
    x = grid["across_scale"].to_numpy(dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), dpi=int(dpi), constrained_layout=True)
    axes[0].plot(
        x,
        grid["population_ssi_vs_static_frame_mean"],
        marker="o",
        linewidth=1.8,
        label="mean of framewise SSI",
    )
    axes[0].plot(
        x,
        grid["population_ssi_vs_static_mean_map"],
        marker="o",
        linewidth=1.8,
        label="SSI of mean map",
    )
    axes[0].axhline(1.0, color="0.45", linestyle="--", linewidth=0.9)
    axes[0].axvline(1.0, color="0.6", linestyle=":", linewidth=0.9)
    axes[0].set_xlabel("across-contour motion scale, along=0")
    axes[0].set_ylabel(f"population SSI / {baseline_label}")
    axes[0].set_title("Population ratio")
    axes[0].grid(True, axis="y", color="0.9")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(
        x,
        grid["population_ssi_bits_per_spike_frame_mean"],
        marker="o",
        linewidth=1.8,
        label="mean of framewise SSI",
    )
    axes[1].plot(
        x,
        grid["population_ssi_bits_per_spike_mean_map"],
        marker="o",
        linewidth=1.8,
        label="SSI of mean map",
    )
    axes[1].axvline(1.0, color="0.6", linestyle=":", linewidth=0.9)
    axes[1].set_xlabel("across-contour motion scale, along=0")
    axes[1].set_ylabel("population SSI (bits/spike)")
    axes[1].set_title("Absolute population SSI")
    axes[1].grid(True, axis="y", color="0.9")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"RR100 real-trace along=0 SSI aggregation comparison ({baseline_label})", fontsize=12)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    across_scales = parse_scale_list(args.across_scales)
    rows = _condition_rows(across_scales, float(args.along_scale))
    baseline = str(args.baseline)
    baseline_condition = _baseline_condition(rows, baseline)
    summary_rows = _rows_for_baseline(rows, baseline_condition)
    baseline_label = _baseline_label(baseline)
    stats_by_condition = _load_mean_map_stats(Path(args.source_dir), rows, float(args.fd_step_arcmin), int(args.max_frames))
    summary_stats_by_condition = _stats_for_baseline(stats_by_condition, baseline_condition)

    unit_df, top_df, diagnostics = summarize_units(summary_stats_by_condition, summary_rows)
    unit_df["baseline"] = baseline
    unit_df["baseline_condition"] = baseline_condition
    top_df["baseline"] = baseline
    top_df["baseline_condition"] = baseline_condition
    top_n = max(1, int(args.top_units))
    top_by_influence = top_df.head(top_n)["unit_index"].astype(int).tolist()
    top_for_plot = order_units_by_y_at_x(diagnostics, top_by_influence, x_value=1.0)

    unit_csv = Path(args.out_dir) / "rr100_real_trace_along0_mean_map_ssi_unit_table.csv"
    top_csv = Path(args.out_dir) / "rr100_real_trace_along0_mean_map_ssi_top_units.csv"
    pop_csv = Path(args.out_dir) / "rr100_real_trace_along0_mean_map_ssi_population_summary.csv"
    comparison_png = Path(args.out_dir) / "rr100_real_trace_along0_mean_map_ssi_population_comparison.png"
    unit_lines_png = Path(args.out_dir) / "rr100_real_trace_along0_mean_map_ssi_lines_top_influence.png"
    unit_lines_with_maps_png = Path(args.out_dir) / "rr100_real_trace_along0_mean_map_ssi_lines_top_influence_with_activation_rows.png"
    loo_png = Path(args.out_dir) / "rr100_real_trace_along0_mean_map_ssi_leave_one_out.png"

    unit_df.to_csv(unit_csv, index=False)
    top_df.to_csv(top_csv, index=False)
    pop_df = _population_summary(
        unit_df,
        summary_rows,
        Path(args.source_dir) / "rr100_real_trace_along0_unit_ssi_table.csv",
        baseline_condition=baseline_condition,
    )
    pop_df.to_csv(pop_csv, index=False)
    _write_comparison_plot(pop_df, comparison_png, int(args.dpi), baseline_label=baseline_label)

    figure_title = f"RR100 real-trace unit SSI along the along=0 scale line: SSI of mean map ({baseline_label})"
    draw_unit_lines(
        diagnostics,
        top_for_plot,
        unit_lines_png,
        int(args.dpi),
        highlight_note="largest leave-one-out influences highlighted",
        figure_title=figure_title,
    )
    draw_unit_lines_with_activation_rows(
        diagnostics=diagnostics,
        highlighted_units=top_for_plot,
        rows=summary_rows,
        stats_by_condition=stats_by_condition,
        path=unit_lines_with_maps_png,
        dpi=int(args.dpi),
        highlight_note="largest leave-one-out influences highlighted",
        map_vmin_percentile=float(args.map_vmin_percentile),
        map_vmax_percentile=float(args.map_vmax_percentile),
        figure_title=figure_title,
        figure_subtitle=(
            "activation maps and SSI numbers both use the trace/time mean finite-difference midpoint map; "
            "rows and legend are ordered by y at across=1"
        ),
    )
    draw_leave_one_out(
        diagnostics,
        top_by_influence,
        top_df,
        loo_png,
        int(args.dpi),
        figure_title="Does any single RR100 unit drive mean-map along=0 SSI?",
    )

    manifest_path = Path(args.out_dir) / "rr100_real_trace_along0_mean_map_ssi_manifest.json"
    write_json(
        manifest_path,
        {
            "analysis": "rr100_real_trace_along0_mean_map_ssi",
            "source_dir": Path(args.source_dir),
            "out_dir": Path(args.out_dir),
            "conditions": [str(row["condition"]) for row in rows],
            "across_scales": across_scales,
            "along_scale": float(args.along_scale),
            "baseline": baseline,
            "baseline_condition": baseline_condition,
            "fd_step_arcmin": float(args.fd_step_arcmin),
            "max_frames": int(args.max_frames),
            "stimulus_normalization": STIMULUS_NORMALIZATION,
            "ssi_contract": "compute SSI on the trace/time mean finite-difference midpoint activation map",
            "source_cache_contract": "mean_rate_map from plot_rr100_real_trace_along0_unit_ssi caches",
            "unit_table_csv": unit_csv,
            "top_units_csv": top_csv,
            "population_summary_csv": pop_csv,
            "population_comparison_png": comparison_png,
            "unit_lines_png": unit_lines_png,
            "unit_lines_with_activation_rows_png": unit_lines_with_maps_png,
            "leave_one_out_png": loo_png,
            "top_units_by_leave_one_out_influence": top_by_influence,
            "top_units_plot_order_at_across1": top_for_plot,
            "source_caches": {condition: stats["source_cache"] for condition, stats in stats_by_condition.items()},
        },
    )
    print(f"Wrote mean-map SSI population summary: {pop_csv}", flush=True)
    print(f"Wrote mean-map SSI comparison plot: {comparison_png}", flush=True)
    print(f"Wrote mean-map SSI activation-row plot: {unit_lines_with_maps_png}", flush=True)
    print(f"Wrote manifest: {manifest_path}", flush=True)
    print(
        pop_df[
            [
                "condition",
                "population_ssi_vs_static_frame_mean",
                "population_ssi_vs_static_mean_map",
                "population_ssi_bits_per_spike_frame_mean",
                "population_ssi_bits_per_spike_mean_map",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6g}"),
        flush=True,
    )


if __name__ == "__main__":
    main()
