"""Plot trace-level Fisher diagnostics for the RR100 endpoint scale grid."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path("outputs/vernier_endpoint_history_last_frame_tutorial/rr100_endpoint_history_scale_grid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--along-scales", type=str, default="0,0.5,1,2")
    parser.add_argument("--file-prefix", type=str, default="rr100_endpoint_fisher_trace_diagnostics")
    return parser.parse_args()


def _parse_scales(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _metric_at(summary: pd.DataFrame, across: float, along: float, column: str) -> float:
    rows = summary[
        np.isclose(pd.to_numeric(summary["across_scale"], errors="coerce"), across)
        & np.isclose(pd.to_numeric(summary["along_scale"], errors="coerce"), along)
    ]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0][column])


def _plot_panel(
    ax: Any,
    trace_table: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    along: float,
    across_scales: list[float],
    pose_ref: float,
    hidden_ref: float,
    normalized: bool,
) -> None:
    x = np.asarray(across_scales, dtype=float)
    trace_subset = trace_table[np.isclose(pd.to_numeric(trace_table["along_scale"], errors="coerce"), along)]
    for trace_idx, rows in trace_subset.groupby("trace_index"):
        values = []
        for across in across_scales:
            point = rows[np.isclose(pd.to_numeric(rows["across_scale"], errors="coerce"), across)]
            value = float(point.iloc[0]["pose_aware_fisher"]) if not point.empty else float("nan")
            values.append(value / pose_ref if normalized else value)
        ax.plot(x, values, color="0.78", linewidth=0.75, alpha=0.65, zorder=1)

    mean = np.asarray(
        [_metric_at(summary, across, along, "pose_aware_fisher_mean") for across in across_scales],
        dtype=float,
    )
    sem = np.asarray(
        [_metric_at(summary, across, along, "pose_aware_fisher_sem") for across in across_scales],
        dtype=float,
    )
    hidden = np.asarray(
        [_metric_at(summary, across, along, "pose_hidden_fisher") for across in across_scales],
        dtype=float,
    )
    if normalized:
        mean = mean / pose_ref
        sem = sem / pose_ref
        hidden = hidden / hidden_ref

    ax.fill_between(x, mean - sem, mean + sem, color="black", alpha=0.14, linewidth=0.0, zorder=2)
    ax.plot(x, mean, color="black", marker="o", markersize=3.0, linewidth=1.8, label="known-trace mean", zorder=3)
    ax.plot(
        x,
        hidden,
        color="#d62728",
        marker="o",
        markersize=2.8,
        linewidth=1.4,
        linestyle="--",
        label="hidden-trace",
        zorder=4,
    )
    ax.axhline(1.0 if normalized else pose_ref, color="0.45", linestyle=":", linewidth=1.0)
    if any(np.isclose(scale, 1.0) for scale in across_scales):
        ax.axvline(1.0, color="0.65", linestyle=":", linewidth=0.9)
    major_ticks = [scale for scale in [0.0, 0.25, 0.5, 1.0, 2.0, 3.0] if any(np.isclose(scale, x))]
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([f"{scale:g}" for scale in major_ticks])
    ax.set_xlim(min(across_scales) - 0.05, max(across_scales) + 0.08)
    ax.grid(True, axis="y", color="0.9", linewidth=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"along {along:g}x", fontsize=10)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary = pd.read_csv(run_dir / "rr100_endpoint_history_scale_grid_summary.csv")
    trace_table = pd.read_csv(run_dir / "rr100_endpoint_history_scale_grid_trace_table.csv")
    grid = summary[~summary["is_static_baseline"].astype(bool)].copy()
    across_scales = [
        float(v)
        for v in sorted(
            np.unique(pd.to_numeric(grid["across_scale"], errors="coerce").dropna().to_numpy(dtype=float))
        )
    ]
    along_scales = _parse_scales(args.along_scales)
    static = summary[summary["condition"].eq("static_center")].iloc[0]
    pose_ref = float(static["pose_aware_fisher_mean"])
    hidden_ref = float(static["pose_hidden_fisher"])

    fig, axes = plt.subplots(2, len(along_scales), figsize=(3.9 * len(along_scales), 7.0), dpi=220, sharex=True)
    if len(along_scales) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for col_idx, along in enumerate(along_scales):
        _plot_panel(
            axes[0, col_idx],
            trace_table,
            summary,
            along=along,
            across_scales=across_scales,
            pose_ref=pose_ref,
            hidden_ref=hidden_ref,
            normalized=False,
        )
        _plot_panel(
            axes[1, col_idx],
            trace_table,
            summary,
            along=along,
            across_scales=across_scales,
            pose_ref=pose_ref,
            hidden_ref=hidden_ref,
            normalized=True,
        )
        axes[1, col_idx].set_xlabel("across-contour motion scale")
    axes[0, 0].set_ylabel("raw Fisher")
    axes[1, 0].set_ylabel("Fisher / static")
    axes[0, -1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("RR100 endpoint-history Fisher trace diagnostics", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    png = run_dir / f"{args.file_prefix}.png"
    pdf = run_dir / f"{args.file_prefix}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
