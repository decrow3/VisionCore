#!/usr/bin/env python3
"""Merge and plot BackImage RR100 along/across SSI grid summaries."""

from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_ACROSS_RUN = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_contour_axis_rr100_spatial_ssi_mean_map_primary_n128_across_sweep_v1"
)
DEFAULT_ISOTROPIC_RUN = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_contour_axis_rr100_spatial_ssi_mean_map_isotropic_n128_scales_0_0p5_1_2_v1"
)
DEFAULT_MISSING_PAIRS_RUN = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_contour_axis_rr100_spatial_ssi_mean_map_grid5_missing_pairs_n128_v1"
)
DEFAULT_OUT_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_contour_axis_rr100_spatial_ssi_mean_map_grid5_merged_v1"
)
DEFAULT_SCALES = "0,0.25,0.5,1,2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--across-run-dir", type=Path, default=DEFAULT_ACROSS_RUN)
    parser.add_argument("--isotropic-run-dir", type=Path, default=DEFAULT_ISOTROPIC_RUN)
    parser.add_argument("--missing-pairs-run-dir", type=Path, default=DEFAULT_MISSING_PAIRS_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--scales", type=str, default=DEFAULT_SCALES)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one scale is required.")
    return values


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
        return value if np.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_condition_summary(run_dir: Path, source_run: str) -> pd.DataFrame:
    path = Path(run_dir) / "condition_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["source_run"] = source_run
    df["source_run_dir"] = str(Path(run_dir))
    if "motion_scale" not in df.columns:
        df["motion_scale"] = df["across_scale"]
    if "sweep_mode" not in df.columns:
        df["sweep_mode"] = "across"
    return df


def source_priority(row: pd.Series) -> int:
    source = str(row["source_run"])
    along = float(row["along_scale"])
    across = float(row["across_scale"])
    if source == "across" and np.isclose(along, 1.0):
        return 0
    if source == "isotropic" and np.isclose(along, across):
        return 1
    if source == "missing_pairs":
        return 2
    if source == "across":
        return 3
    return 4


def select_grid_rows(frames: list[pd.DataFrame], scales: list[float]) -> pd.DataFrame:
    merged = pd.concat(frames, ignore_index=True)
    selected: list[pd.Series] = []
    for along in scales:
        for across in scales:
            sub = merged[
                np.isclose(pd.to_numeric(merged["along_scale"], errors="coerce"), float(along))
                & np.isclose(pd.to_numeric(merged["across_scale"], errors="coerce"), float(across))
            ].copy()
            if sub.empty:
                raise ValueError(f"Missing grid cell along={along:g}, across={across:g}")
            sub["_priority"] = sub.apply(source_priority, axis=1)
            sub = sub.sort_values(["_priority", "condition_index"], kind="mergesort")
            selected.append(sub.iloc[0].drop(labels=["_priority"]))
    out = pd.DataFrame(selected).reset_index(drop=True)
    out["grid_along_scale"] = out["along_scale"].astype(float)
    out["grid_across_scale"] = out["across_scale"].astype(float)
    static = out[
        np.isclose(out["grid_along_scale"].astype(float), 0.0)
        & np.isclose(out["grid_across_scale"].astype(float), 0.0)
    ]
    if not static.empty:
        static_mean = float(static.iloc[0]["population_mean_map_ssi_bits_per_spike_mean"])
    else:
        static_mean = float(np.nanmin(out["population_mean_map_ssi_bits_per_spike_mean"].to_numpy(dtype=float)))
    values = out["population_mean_map_ssi_bits_per_spike_mean"].to_numpy(dtype=float)
    out["grid_mean_map_delta_vs_0x0x"] = values - static_mean
    out["grid_mean_map_population_z"] = (values - float(np.nanmean(values))) / max(float(np.nanstd(values)), 1e-12)
    return out


def grid_array(grid: pd.DataFrame, scales: list[float], column: str) -> np.ndarray:
    arr = np.full((len(scales), len(scales)), np.nan, dtype=np.float64)
    for _, row in grid.iterrows():
        y = int(np.flatnonzero(np.isclose(scales, float(row["grid_along_scale"])))[0])
        x = int(np.flatnonzero(np.isclose(scales, float(row["grid_across_scale"])))[0])
        arr[y, x] = float(row[column])
    return arr


def annotate_heatmap(ax: plt.Axes, values: np.ndarray, *, fmt: str) -> None:
    finite = values[np.isfinite(values)]
    threshold = float(np.nanmean(finite)) if finite.size else 0.0
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            val = values[y, x]
            if not np.isfinite(val):
                continue
            color = "white" if val < threshold else "black"
            ax.text(x, y, format(float(val), fmt), ha="center", va="center", fontsize=6.5, color=color)


def plot_grid(out_dir: Path, grid: pd.DataFrame, scales: list[float], *, dpi: int) -> tuple[Path, Path]:
    mean_map = grid_array(grid, scales, "population_mean_map_ssi_bits_per_spike_mean")
    time_resolved = grid_array(grid, scales, "population_time_resolved_ssi_bits_per_spike_mean")
    static = float(mean_map[0, 0]) if np.isclose(scales[0], 0.0) else float(np.nanmin(mean_map))
    delta = mean_map - static
    z = (mean_map - np.nanmean(mean_map)) / max(float(np.nanstd(mean_map)), 1e-12)
    panels = [
        ("mean-map SSI", mean_map, "viridis", ".4f"),
        ("mean-map SSI minus 0x/0x", delta, "coolwarm", "+.4f"),
        ("population z-score", z, "coolwarm", "+.2f"),
        ("legacy time-resolved SSI", time_resolved, "magma", ".4f"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.8), constrained_layout=True)
    for ax, (title, values, cmap, fmt) in zip(axes.ravel(), panels, strict=True):
        if cmap == "coolwarm":
            vmax = float(np.nanpercentile(np.abs(values), 98.0))
            vmax = max(vmax, 1e-6)
            im = ax.imshow(values, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(values, origin="lower", cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("across-contour scale")
        ax.set_ylabel("along-contour scale")
        ax.set_xticks(np.arange(len(scales)))
        ax.set_yticks(np.arange(len(scales)))
        labels = [f"{scale:g}" for scale in scales]
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        annotate_heatmap(ax, values, fmt=fmt)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=7)
    fig.suptitle("BackImage RR100 mean-map SSI: crossed along/across scale grid", fontsize=12)
    png = out_dir / "backimage_contour_axis_rr100_grid5_mean_map_ssi_heatmaps.png"
    pdf = out_dir / "backimage_contour_axis_rr100_grid5_mean_map_ssi_heatmaps.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scales = parse_float_list(str(args.scales))
    frames = [
        load_condition_summary(Path(args.across_run_dir), "across"),
        load_condition_summary(Path(args.isotropic_run_dir), "isotropic"),
        load_condition_summary(Path(args.missing_pairs_run_dir), "missing_pairs"),
    ]
    grid = select_grid_rows(frames, scales)
    rows = grid.to_dict(orient="records")
    condition_csv = out_dir / "grid_condition_summary.csv"
    write_csv(condition_csv, rows)
    png, pdf = plot_grid(out_dir, grid, scales, dpi=int(args.dpi))
    mean_map = grid_array(grid, scales, "population_mean_map_ssi_bits_per_spike_mean")
    best = grid.iloc[int(np.nanargmax(mean_map.ravel()))]
    write_json(
        out_dir / "summary.json",
        {
            "grid_condition_summary_csv": condition_csv,
            "plot_png": png,
            "plot_pdf": pdf,
            "scales": scales,
            "source_run_dirs": {
                "across": Path(args.across_run_dir),
                "isotropic": Path(args.isotropic_run_dir),
                "missing_pairs": Path(args.missing_pairs_run_dir),
            },
            "best_mean_map_condition": {
                "along_scale": float(best["grid_along_scale"]),
                "across_scale": float(best["grid_across_scale"]),
                "population_mean_map_ssi_bits_per_spike_mean": float(best["population_mean_map_ssi_bits_per_spike_mean"]),
                "grid_mean_map_delta_vs_0x0x": float(best["grid_mean_map_delta_vs_0x0x"]),
                "grid_mean_map_population_z": float(best["grid_mean_map_population_z"]),
                "source_run": str(best["source_run"]),
                "condition_id": str(best["condition_id"]),
            },
        },
    )
    print(f"Wrote {condition_csv}")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
