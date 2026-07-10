#!/usr/bin/env python3
"""Posthoc z-scored unit-curve plots for BackImage contour-axis RR100 SSI."""

from __future__ import annotations

import argparse
import csv
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


DEFAULT_RUN_DIR = Path(
    "outputs/active_sensing_movie_information/backimage_contour_axis_rr100_spatial_ssi_n128_across_sweep_v1"
)
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--min-unit-std",
        type=float,
        default=1e-4,
        help="Drop units whose absolute SSI curve std across sweep conditions is below this threshold.",
    )
    parser.add_argument("--top-units", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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


def sem(x: np.ndarray, axis: int = 0) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    n = arr.shape[axis]
    if n <= 1:
        return np.zeros_like(np.nanmean(arr, axis=axis), dtype=np.float64)
    return np.nanstd(arr, axis=axis, ddof=1) / np.sqrt(float(n))


def zscore_rows(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(curves, axis=1, keepdims=True)
    sd = np.nanstd(curves, axis=1, ddof=0, keepdims=True)
    z = (curves - mean) / np.maximum(sd, EPS)
    return z, mean[:, 0], sd[:, 0]


def scale_token(value: float) -> str:
    return f"{float(value):.9g}".replace("-", "m").replace(".", "p")


def plot_zscore_curves(
    run_dir: Path,
    *,
    out_dir: Path | None = None,
    min_unit_std: float = 1e-4,
    top_units: int = 12,
    dpi: int = 220,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    unit = pd.read_csv(run_dir / "unit_ssi_table.csv")
    cond = pd.read_csv(run_dir / "condition_summary.csv")
    sweep_mode = str(cond["sweep_mode"].dropna().iloc[0]) if "sweep_mode" in cond.columns else "across"
    scale_column = "motion_scale" if sweep_mode == "isotropic" and "motion_scale" in cond.columns else "across_scale"
    scale_label = "isotropic measured-trace motion scale" if sweep_mode == "isotropic" else "across-contour motion scale, along=1"
    sweep = cond[cond["is_across_sweep"].astype(bool)].sort_values(scale_column).copy()
    sweep_ids = sweep["condition_id"].astype(str).to_list()
    scales = sweep[scale_column].to_numpy(dtype=np.float64)

    wide = (
        unit[unit["condition_id"].astype(str).isin(sweep_ids)]
        .pivot(index="unit_index", columns="condition_id", values="unit_ssi_bits_per_spike_mean")
        .loc[:, sweep_ids]
        .sort_index()
    )
    curves = wide.to_numpy(dtype=np.float64)
    unit_indices = wide.index.to_numpy(dtype=int)
    z, curve_mean, curve_sd = zscore_rows(curves)
    keep = np.isfinite(z).all(axis=1) & (curve_sd >= float(min_unit_std))
    kept_z = z[keep]
    kept_curves = curves[keep]
    kept_units = unit_indices[keep]
    kept_sd = curve_sd[keep]

    if kept_z.size == 0:
        raise ValueError(
            f"No units survived --min-unit-std={float(min_unit_std):g}; "
            "try a smaller threshold."
        )

    slopes = np.asarray([np.polyfit(scales, row, deg=1)[0] for row in kept_z], dtype=np.float64)
    dynamic_range = np.nanmax(kept_curves, axis=1) - np.nanmin(kept_curves, axis=1)
    order_by_slope = np.argsort(slopes)
    top_order = np.argsort(dynamic_range)[::-1][: max(1, int(top_units))]

    population_curve = sweep["population_ssi_bits_per_spike_mean"].to_numpy(dtype=np.float64)
    population_z = (population_curve - np.mean(population_curve)) / max(float(np.std(population_curve)), EPS)
    mean_z = np.nanmean(kept_z, axis=0)
    sem_z = sem(kept_z, axis=0)

    z_rows: list[dict[str, Any]] = []
    for row_idx, unit_index in enumerate(kept_units):
        payload: dict[str, Any] = {
            "unit_index": int(unit_index),
            "unit_label": f"u{int(unit_index):03d}",
            "sweep_mode": sweep_mode,
            "scale_column": scale_column,
            "absolute_curve_mean": float(np.mean(kept_curves[row_idx])),
            "absolute_curve_std": float(kept_sd[row_idx]),
            "z_slope_vs_scale": float(slopes[row_idx]),
            "absolute_dynamic_range": float(dynamic_range[row_idx]),
        }
        for scale, value, z_value in zip(scales, kept_curves[row_idx], kept_z[row_idx], strict=True):
            token = scale_token(float(scale))
            payload[f"ssi_at_scale_{token}"] = float(value)
            payload[f"z_at_scale_{token}"] = float(z_value)
        z_rows.append(payload)
    write_csv_rows(out_dir / "backimage_contour_axis_rr100_unit_zscore_curves.csv", z_rows)

    fig = plt.figure(figsize=(12.4, 8.6))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.0, 1.25], hspace=0.36, wspace=0.22)
    ax_all = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[1, :])

    for row in kept_z:
        ax_all.plot(scales, row, color="#9b9b9b", linewidth=0.65, alpha=0.25, zorder=1)
    ax_all.fill_between(scales, mean_z - sem_z, mean_z + sem_z, color="#222222", alpha=0.12, linewidth=0)
    ax_all.plot(scales, mean_z, color="black", linewidth=2.2, marker="o", label="mean unit z", zorder=4)
    ax_all.plot(scales, population_z, color="#1f77b4", linewidth=2.0, marker="s", label="population z", zorder=5)
    ax_all.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
    ax_all.axvline(1.0, color="#999999", linestyle=":", linewidth=0.9)
    ax_all.set_title(f"All retained RR100 units (n={kept_z.shape[0]}/{curves.shape[0]})")
    ax_all.set_xlabel(scale_label)
    ax_all.set_ylabel("within-unit z-scored SSI")
    ax_all.grid(True, color="#e7e7e7", linewidth=0.7)
    ax_all.legend(frameon=False, fontsize=8)

    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(len(top_order), 1)))
    for pos, row_idx in enumerate(top_order):
        ax_top.plot(
            scales,
            kept_z[row_idx],
            color=colors[pos],
            marker="o",
            linewidth=1.5,
            markersize=3.5,
            label=f"u{int(kept_units[row_idx]):03d}",
        )
    ax_top.plot(scales, mean_z, color="black", linewidth=2.2, marker="o", label="mean")
    ax_top.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
    ax_top.axvline(1.0, color="#999999", linestyle=":", linewidth=0.9)
    ax_top.set_title(f"Top {len(top_order)} by absolute dynamic range")
    ax_top.set_xlabel(scale_label)
    ax_top.set_ylabel("within-unit z-scored SSI")
    ax_top.grid(True, color="#e7e7e7", linewidth=0.7)
    ax_top.legend(frameon=False, fontsize=6.5, ncol=3)

    vmax = float(np.nanpercentile(np.abs(kept_z), 98.0))
    vmax = max(vmax, 1.0)
    im = ax_heat.imshow(
        kept_z[order_by_slope],
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        extent=[float(scales[0]), float(scales[-1]), 0, kept_z.shape[0]],
        origin="lower",
    )
    ax_heat.axvline(1.0, color="#222222", linestyle=":", linewidth=0.9)
    ax_heat.set_title("Unit z-score curves ordered by linear slope")
    ax_heat.set_xlabel(scale_label)
    ax_heat.set_ylabel("RR100 units ordered by z-slope")
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.015)
    cbar.set_label("within-unit z-scored SSI")

    fig.suptitle(
        f"BackImage RR100 contour-axis SSI curve shapes ({sweep_mode} sweep)\n"
        f"z-score computed within each unit across sweep conditions; units with std < {float(min_unit_std):g} bits/spike omitted",
        fontsize=12,
        y=0.985,
    )
    png = out_dir / "backimage_contour_axis_rr100_unit_zscore_curves.png"
    pdf = out_dir / "backimage_contour_axis_rr100_unit_zscore_curves.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "png": png,
        "pdf": pdf,
        "csv": out_dir / "backimage_contour_axis_rr100_unit_zscore_curves.csv",
        "sweep_mode": sweep_mode,
        "scale_column": scale_column,
        "scale_values": scales,
        "retained_units": int(kept_z.shape[0]),
        "total_units": int(curves.shape[0]),
        "mean_z": mean_z,
        "population_z": population_z,
    }


def main() -> None:
    args = parse_args()
    result = plot_zscore_curves(
        Path(args.run_dir),
        out_dir=Path(args.out_dir) if args.out_dir is not None else None,
        min_unit_std=float(args.min_unit_std),
        top_units=int(args.top_units),
        dpi=int(args.dpi),
    )

    print(f"Wrote {result['png']}")
    print(f"Wrote {result['pdf']}")
    print(f"Retained units: {int(result['retained_units'])}/{int(result['total_units'])}")
    print(f"Mean z curve: {', '.join(f'{float(v):.3f}' for v in np.asarray(result['mean_z'], dtype=float))}")
    print(f"Population z curve: {', '.join(f'{float(v):.3f}' for v in np.asarray(result['population_z'], dtype=float))}")


if __name__ == "__main__":
    main()
