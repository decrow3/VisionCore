#!/usr/bin/env python3
"""Summarize spatial-frequency localization of the FEM information gain.

This is an existing-output analysis: it reads the production Figure 5
``05_lagcube_information_summary.csv`` and contrasts each spatial-frequency
condition with its stabilized counterpart.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from non_circular_fem_common import (
    DEFAULT_STACK_OUT_DIR,
    DEFAULT_TWININFO_RUN_DIR,
    PRIMARY_FINAL_METRIC,
    load_summary_rows,
    paired_contrast_rows,
    summarize_groups,
    write_csv_rows,
    write_json,
)


DEFAULT_OUT_DIR = DEFAULT_STACK_OUT_DIR / "active_sensing_movie_information_figure"
SF_CONTRASTS = (
    ("sf_low_minus_stabilized_sf_low", "sf_low", "stabilized_sf_low"),
    ("sf_mid_low_minus_stabilized_sf_mid_low", "sf_mid_low", "stabilized_sf_mid_low"),
    ("sf_mid_high_minus_stabilized_sf_mid_high", "sf_mid_high", "stabilized_sf_mid_high"),
    ("sf_high_minus_stabilized_sf_high", "sf_high", "stabilized_sf_high"),
)
SF_ORDER = ("sf_low", "sf_mid_low", "sf_mid_high", "sf_high")
SF_LABELS = {
    "sf_low": "low",
    "sf_mid_low": "mid-low",
    "sf_mid_high": "mid-high",
    "sf_high": "high",
}


def contrast_to_band(contrast: str) -> str:
    return str(contrast).removesuffix("_minus_stabilized_" + str(contrast).split("_minus_stabilized_")[-1])


def add_band_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        condition = str(item.get("condition", ""))
        item["sf_band"] = condition
        item["sf_band_label"] = SF_LABELS.get(condition, condition)
        out.append(item)
    return out


def plot_sf_gain(out_dir: Path, contrast_rows: list[dict[str, Any]]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    xs = np.arange(len(SF_ORDER))
    means = []
    sems = []
    labels = []
    for band in SF_ORDER:
        vals = [float(row["delta"]) for row in contrast_rows if row.get("condition") == band]
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        means.append(float(np.mean(arr)) if arr.size else float("nan"))
        sems.append(float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0)
        labels.append(SF_LABELS[band])
        if arr.size:
            jitter = np.linspace(-0.12, 0.12, arr.size) if arr.size > 1 else np.asarray([0.0])
            ax.scatter(xs[len(labels) - 1] + jitter, arr, color="#2f6fa5", alpha=0.45, s=18, linewidth=0)
    ax.bar(xs, means, yerr=sems, color="#9fb8cc", edgecolor="#315f7d", linewidth=0.8, capsize=3)
    ax.axhline(0, color="#bbbbbb", linewidth=0.8)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("real-motion gain over stabilized (bits/spike)")
    ax.set_title("SF-localized information gain")
    fig.tight_layout()
    fig.savefig(fig_dir / "sf_localization_gain.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(out_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Spatial-Frequency Localization Summary",
        "",
        f"Primary metric: `{PRIMARY_FINAL_METRIC}`.",
        "",
        "## Paired SF Gains",
        "",
    ]
    for row in summary_rows:
        lines.append(
            f"- {row['sf_band_label']}: mean={row['mean']:.6g}, SEM={row['sem']:.6g}, n={row['n']}"
        )
    lines.extend(
        [
            "",
            "Interpretation guardrail: high-SF-localized gain supports a fine-structure mechanism; SF-flat gain would weaken that specificity.",
            "",
        ]
    )
    (out_dir / "sf_localization_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_TWININFO_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metric", default=PRIMARY_FINAL_METRIC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_summary_rows(Path(args.run_dir))
    contrasts = add_band_columns(paired_contrast_rows(rows, metric=str(args.metric), contrasts=SF_CONTRASTS))
    summary = summarize_groups(contrasts, ("sf_band", "sf_band_label"), "delta")
    write_csv_rows(out_dir / "sf_localization_paired_contrasts.csv", contrasts)
    write_csv_rows(out_dir / "sf_localization_summary.csv", summary)
    plot_sf_gain(out_dir, contrasts)
    write_summary(out_dir, summary)
    write_json(
        out_dir / "sf_localization_manifest.json",
        {
            "run_dir": Path(args.run_dir),
            "out_dir": out_dir,
            "metric": str(args.metric),
            "n_paired_contrasts": len(contrasts),
        },
    )
    print(f"Wrote SF localization summary to {out_dir}")


if __name__ == "__main__":
    main()
