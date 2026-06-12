#!/usr/bin/env python3
"""Summarize sustained information accumulation from a production twininfo run.

This script does not run the twin.  It reads
``cache/cumulative_information_series.npz`` and the matching row metadata from
an existing Figure 5 run, then quantifies whether real retinal motion sustains
information accumulation relative to stabilized input.
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
    PRIMARY_SERIES_METRIC,
    canonical_condition,
    load_series,
    paired_key,
    robust_slope,
    summarize_groups,
    time_to_fraction,
    write_csv_rows,
    write_json,
)


DEFAULT_OUT_DIR = DEFAULT_STACK_OUT_DIR / "information_accumulation"


def window_masks(time_s: np.ndarray) -> dict[str, np.ndarray]:
    t = np.asarray(time_s, dtype=np.float64)
    if t.size == 0:
        return {"early": np.asarray([], dtype=bool), "mid": np.asarray([], dtype=bool), "late": np.asarray([], dtype=bool)}
    lo = float(np.nanmin(t))
    hi = float(np.nanmax(t))
    span = max(hi - lo, 1e-12)
    frac = (t - lo) / span
    return {
        "early": frac <= 0.25,
        "mid": (frac >= 0.25) & (frac <= 0.75),
        "late": frac >= 0.75,
    }


def slope_row(
    *,
    row_id: int,
    record: dict[str, str],
    condition: str,
    time_s: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    masks = window_masks(time_s)
    early = robust_slope(time_s[masks["early"]], y[masks["early"]])
    mid = robust_slope(time_s[masks["mid"]], y[masks["mid"]])
    late = robust_slope(time_s[masks["late"]], y[masks["late"]])
    return {
        "row_id": row_id,
        "example_id": record.get("example_id", ""),
        "kind": record.get("kind", ""),
        "image_index": int(record.get("image_index", 0)),
        "crop_rank": int(record.get("crop_rank", 0)),
        "condition": condition,
        "metric": PRIMARY_SERIES_METRIC,
        "early_slope": early,
        "mid_slope": mid,
        "late_slope": late,
        "late_minus_early_slope": late - early,
        "final_value": float(y[-1]) if y.size else float("nan"),
        "time_to_50pct_final_s": time_to_fraction(time_s, y, 0.5),
        "time_to_80pct_final_s": time_to_fraction(time_s, y, 0.8),
    }


def build_tables(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    series = load_series(run_dir)
    arrays = series.arrays
    metric = np.asarray(arrays[PRIMARY_SERIES_METRIC], dtype=np.float64)
    time_s = np.asarray(arrays.get("time_s", np.arange(metric.shape[1])), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    by_pair: dict[tuple[str, str, int, int], dict[str, tuple[int, np.ndarray, dict[str, str]]]] = {}
    for i, record in enumerate(series.records):
        condition = canonical_condition(str(record.get("condition", "")))
        y = metric[i]
        rows.append(slope_row(row_id=i, record=record, condition=condition, time_s=time_s, y=y))
        by_pair.setdefault(paired_key(record), {})[condition] = (i, y, record)

    contrasts: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []
    for key, conds in by_pair.items():
        if "real" not in conds or "stabilized" not in conds:
            continue
        example_id, kind, image_index, crop_rank = key
        real_i, real_y, _real_record = conds["real"]
        stable_i, stable_y, _stable_record = conds["stabilized"]
        gain = real_y - stable_y
        masks = window_masks(time_s)
        contrast_row: dict[str, Any] = {
            "contrast": "real_minus_stabilized",
            "example_id": example_id,
            "kind": kind,
            "image_index": image_index,
            "crop_rank": crop_rank,
            "real_row_id": real_i,
            "stabilized_row_id": stable_i,
            "final_gain": float(gain[-1]),
            "area_under_gain_curve": float(np.trapz(gain, time_s)),
            "mean_gain": float(np.nanmean(gain)),
        }
        for name, mask in masks.items():
            contrast_row[f"{name}_gain_slope"] = robust_slope(time_s[mask], gain[mask])
            contrast_row[f"{name}_mean_gain"] = float(np.nanmean(gain[mask])) if np.any(mask) else float("nan")
        contrasts.append(contrast_row)
        thresholds.append(
            {
                "contrast": "real_minus_stabilized",
                "example_id": example_id,
                "kind": kind,
                "image_index": image_index,
                "crop_rank": crop_rank,
                "real_time_to_50pct_final_s": time_to_fraction(time_s, real_y, 0.5),
                "stabilized_time_to_50pct_final_s": time_to_fraction(time_s, stable_y, 0.5),
                "real_minus_stabilized_time_to_50pct_s": time_to_fraction(time_s, real_y, 0.5)
                - time_to_fraction(time_s, stable_y, 0.5),
                "real_time_to_80pct_final_s": time_to_fraction(time_s, real_y, 0.8),
                "stabilized_time_to_80pct_final_s": time_to_fraction(time_s, stable_y, 0.8),
                "real_minus_stabilized_time_to_80pct_s": time_to_fraction(time_s, real_y, 0.8)
                - time_to_fraction(time_s, stable_y, 0.8),
            }
        )
    return rows, contrasts, thresholds, time_s


def plot_outputs(
    *,
    out_dir: Path,
    slope_rows: list[dict[str, Any]],
    contrast_rows: list[dict[str, Any]],
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    kinds = sorted({str(row.get("kind", "")) for row in slope_rows if row.get("condition") in {"real", "stabilized"}})
    fig, axs = plt.subplots(1, max(1, len(kinds)), figsize=(4.0 * max(1, len(kinds)), 3.4), squeeze=False)
    for ax, kind in zip(axs[0], kinds or [""], strict=True):
        vals = [
            (row["condition"], float(row["early_slope"]), float(row["late_slope"]))
            for row in slope_rows
            if row.get("kind") == kind and row.get("condition") in {"real", "stabilized"}
        ]
        for condition in ("stabilized", "real"):
            xs = [0, 1]
            pairs = [(early, late) for cond, early, late in vals if cond == condition]
            color = "#777777" if condition == "stabilized" else "#2f6fa5"
            for early, late in pairs:
                ax.plot(xs, [early, late], color=color, alpha=0.25, linewidth=0.8)
            if pairs:
                mean = np.nanmean(np.asarray(pairs, dtype=np.float64), axis=0)
                ax.plot(xs, mean, color=color, linewidth=2.0, label=condition)
        ax.set_xticks([0, 1], ["early", "late"])
        ax.set_title(kind or "all")
        ax.set_ylabel("slope bits/spike/s")
        ax.axhline(0, color="#bbbbbb", linewidth=0.8)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "accumulation_slope_pairs.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    by_kind: dict[str, list[float]] = {}
    for row in contrast_rows:
        by_kind.setdefault(str(row.get("kind", "")), []).append(float(row["area_under_gain_curve"]))
    labels = sorted(by_kind)
    data = [by_kind[label] for label in labels]
    if data:
        ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.axhline(0, color="#bbbbbb", linewidth=0.8)
    ax.set_ylabel("area under real-stabilized gain")
    ax.set_title("information gain over fixation")
    fig.tight_layout()
    fig.savefig(fig_dir / "accumulation_gain_over_time.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(out_dir: Path, contrast_rows: list[dict[str, Any]]) -> None:
    summary = summarize_groups(contrast_rows, ("contrast", "kind"), "final_gain")
    lines = [
        "# Information Accumulation Summary",
        "",
        "Primary contrast: real minus stabilized cumulative spatial SSI bits/spike.",
        "",
        "## Final Gain",
        "",
    ]
    for row in summary:
        lines.append(
            f"- {row['contrast']} / {row['kind']}: mean={row['mean']:.6g}, "
            f"SEM={row['sem']:.6g}, n={row['n']}"
        )
    lines.extend(
        [
            "",
            "Interpretation guardrail: this summarizes existing pose-aware model-side cumulative information; it is not a recorded-cortex decoder.",
            "",
        ]
    )
    (out_dir / "accumulation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_TWININFO_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slope_rows, contrast_rows, threshold_rows, _time_s = build_tables(Path(args.run_dir))
    write_csv_rows(out_dir / "accumulation_slope_metrics.csv", slope_rows)
    write_csv_rows(out_dir / "accumulation_paired_contrasts.csv", contrast_rows)
    write_csv_rows(out_dir / "accumulation_time_to_threshold.csv", threshold_rows)
    write_csv_rows(out_dir / "accumulation_contrast_summary.csv", summarize_groups(contrast_rows, ("contrast", "kind"), "final_gain"))
    plot_outputs(out_dir=out_dir, slope_rows=slope_rows, contrast_rows=contrast_rows)
    write_summary(out_dir, contrast_rows)
    write_json(
        out_dir / "accumulation_manifest.json",
        {
            "run_dir": Path(args.run_dir),
            "out_dir": out_dir,
            "metric": PRIMARY_SERIES_METRIC,
            "n_slope_rows": len(slope_rows),
            "n_paired_contrasts": len(contrast_rows),
        },
    )
    print(f"Wrote information accumulation summary to {out_dir}")


if __name__ == "__main__":
    main()
