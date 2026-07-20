#!/usr/bin/env python3
"""Diagnostics for contour-axis/image-stat and unit-orientation confounds."""

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
DEFAULT_LONG_RUN_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_contour_axis_rr100_sf_contour_alignment_long_axis30_n576_low0p05_high0p5_v1"
)
DEFAULT_SF_GROUPS_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/"
    "sf_group_ssi_modulation_dynamic_log_gaussian_marginal_threshold_low0p05_high0p5_v1/"
    "dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv"
)


IMAGE_FEATURES = [
    "image_patch_std",
    "image_patch_rms_contrast",
    "image_gradient_energy",
    "image_edge_density",
    "image_orientation_coherence",
    "image_spectrum_anisotropy",
    "image_high_freq_power_fraction",
    "image_power_8plus_cpd_fraction",
    "image_oriented_8plus_power_proxy",
]

ORIENTATION_COLUMNS = [
    ("prior", "prior_preferred_orientation_deg"),
    ("static_peak", "static_peak_orientation_deg_by_mean_rate"),
    ("dynamic_peak", "dynamic_peak_orientation_deg_by_amp"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-run-dir", type=Path, default=DEFAULT_LONG_RUN_DIR)
    parser.add_argument("--sf-groups-csv", type=Path, default=DEFAULT_SF_GROUPS_CSV)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--axis-bin-width-deg", type=float, default=30.0)
    parser.add_argument("--dpi", type=int, default=180)
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


def axis_bins(width_deg: float) -> np.ndarray:
    width = float(width_deg)
    if width <= 0.0 or 180.0 % width > 1e-9:
        raise ValueError("--axis-bin-width-deg must be a positive divisor of 180")
    return np.arange(0.0, 180.0 + 0.5 * width, width, dtype=float)


def assign_axis_bin(axis_deg: pd.Series | np.ndarray, edges: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis_deg, dtype=float) % 180.0
    return np.clip(np.digitize(axis, edges, right=False) - 1, 0, len(edges) - 2)


def axial_distance_deg(a_deg: np.ndarray, b_deg: float) -> np.ndarray:
    delta = np.abs((np.asarray(a_deg, dtype=float) % 180.0) - float(b_deg))
    return np.minimum(delta, 180.0 - delta)


def summarize_features_by_axis(windows: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    work = windows.copy()
    if "axis_balance_bin" in work.columns:
        work["axis_bin"] = pd.to_numeric(work["axis_balance_bin"], errors="coerce").astype(int)
    else:
        work["axis_bin"] = assign_axis_bin(work["image_edge_axis_deg"], edges)
    rows: list[dict[str, Any]] = []
    for feature in IMAGE_FEATURES:
        if feature not in work.columns:
            continue
        values = pd.to_numeric(work[feature], errors="coerce")
        for idx in range(len(edges) - 1):
            sub = values[work["axis_bin"].to_numpy(dtype=int) == idx].dropna().to_numpy(dtype=float)
            if sub.size == 0:
                continue
            rows.append(
                {
                    "axis_bin": int(idx),
                    "axis_bin_start_deg": float(edges[idx]),
                    "axis_bin_stop_deg": float(edges[idx + 1]),
                    "feature": feature,
                    "n": int(sub.size),
                    "mean": float(np.mean(sub)),
                    "median": float(np.median(sub)),
                    "q25": float(np.quantile(sub, 0.25)),
                    "q75": float(np.quantile(sub, 0.75)),
                    "std": float(np.std(sub, ddof=1)) if sub.size > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def summarize_near_modes(windows: pd.DataFrame) -> pd.DataFrame:
    axis = pd.to_numeric(windows["image_edge_axis_deg"], errors="coerce").to_numpy(dtype=float) % 180.0
    masks = {
        "near_horizontal_axis0pm15": axial_distance_deg(axis, 0.0) <= 15.0,
        "near_vertical_axis90pm15": axial_distance_deg(axis, 90.0) <= 15.0,
    }
    rows: list[dict[str, Any]] = []
    for mode, mask in masks.items():
        sub_frame = windows.loc[mask]
        for feature in IMAGE_FEATURES:
            if feature not in sub_frame.columns:
                continue
            sub = pd.to_numeric(sub_frame[feature], errors="coerce").dropna().to_numpy(dtype=float)
            if sub.size == 0:
                continue
            rows.append(
                {
                    "mode": mode,
                    "feature": feature,
                    "n": int(sub.size),
                    "mean": float(np.mean(sub)),
                    "median": float(np.median(sub)),
                    "q25": float(np.quantile(sub, 0.25)),
                    "q75": float(np.quantile(sub, 0.75)),
                    "std": float(np.std(sub, ddof=1)) if sub.size > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def summarize_unit_orientation_density(units: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_name, column in ORIENTATION_COLUMNS:
        if column not in units.columns:
            continue
        angles = pd.to_numeric(units[column], errors="coerce").to_numpy(dtype=float) % 180.0
        bins = assign_axis_bin(angles, edges)
        for sf_group, sub in units.groupby("sf_group", sort=True):
            sub_idx = sub.index.to_numpy(dtype=int)
            finite = np.isfinite(angles[sub_idx])
            for idx in range(len(edges) - 1):
                keep = finite & (bins[sub_idx] == idx)
                picked = sub.loc[sub.index[keep]]
                rows.append(
                    {
                        "orientation_source": source_name,
                        "orientation_column": column,
                        "sf_group": sf_group,
                        "axis_bin": int(idx),
                        "axis_bin_start_deg": float(edges[idx]),
                        "axis_bin_stop_deg": float(edges[idx + 1]),
                        "n_units": int(picked.shape[0]),
                        "mean_sf_cpd": float(
                            pd.to_numeric(picked["dynamic_log_gaussian_marginal_sf_cpd"], errors="coerce").mean()
                        )
                        if picked.shape[0]
                        else float("nan"),
                        "mean_prior_osi": float(
                            pd.to_numeric(picked["prior_orientation_selectivity_index"], errors="coerce").mean()
                        )
                        if picked.shape[0]
                        else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def summarize_high_sf_axis_gaps(long_run_dir: Path) -> pd.DataFrame:
    path = long_run_dir / "orientation_stratified_population" / "orientation_stratified_weighted_population_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    sub = df[(df["sf_group"].astype(str) == "high_sf") & df["band"].astype(str).str.startswith("axis_bin_")]
    for (band, view), group in sub.groupby(["band", "view"], sort=True):
        xs = sorted(pd.to_numeric(group["x_scale"], errors="coerce").dropna().unique())
        if not xs:
            continue
        x0, x1 = float(xs[0]), float(xs[-1])

        def value(alignment: str, x: float) -> float:
            picked = group[
                (group["alignment_group"].astype(str) == alignment)
                & np.isclose(pd.to_numeric(group["x_scale"], errors="coerce").to_numpy(dtype=float), x)
            ]
            return float(picked["accumulated_bits_per_spike"].iloc[0]) if not picked.empty else float("nan")

        aligned0 = value("contour_aligned", x0)
        orth0 = value("contour_orthogonal", x0)
        aligned1 = value("contour_aligned", x1)
        orth1 = value("contour_orthogonal", x1)
        rows.append(
            {
                "band": band,
                "view": view,
                "x_start": x0,
                "x_stop": x1,
                "aligned_start": aligned0,
                "orthogonal_start": orth0,
                "aligned_stop": aligned1,
                "orthogonal_stop": orth1,
                "aligned_minus_orthogonal_start": aligned0 - orth0,
                "aligned_minus_orthogonal_stop": aligned1 - orth1,
                "aligned_delta": aligned1 - aligned0,
                "orthogonal_delta": orth1 - orth0,
            }
        )
    return pd.DataFrame(rows)


def plot_image_feature_axis_summary(summary: pd.DataFrame, out_dir: Path, *, dpi: int) -> list[str]:
    outputs: list[str] = []
    if summary.empty:
        return outputs
    features = [feature for feature in IMAGE_FEATURES if feature in set(summary["feature"].astype(str))]
    ncols = 3
    nrows = int(math.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 3.1 * nrows), squeeze=False)
    for ax, feature in zip(axes.ravel(), features, strict=False):
        sub = summary[summary["feature"].astype(str) == feature].sort_values("axis_bin_start_deg")
        x = 0.5 * (sub["axis_bin_start_deg"].to_numpy(float) + sub["axis_bin_stop_deg"].to_numpy(float))
        y = sub["median"].to_numpy(float)
        y0 = sub["q25"].to_numpy(float)
        y1 = sub["q75"].to_numpy(float)
        ax.plot(x, y, marker="o", color="#2c7fb8")
        ax.fill_between(x, y0, y1, color="#2c7fb8", alpha=0.16, linewidth=0)
        ax.set_title(feature, fontsize=10)
        ax.set_xlabel("contour axis bin center (deg)")
        ax.grid(True, alpha=0.3)
    for ax in axes.ravel()[len(features) :]:
        ax.axis("off")
    fig.suptitle("Selected BackImage windows: image statistics by contour-axis bin", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = out_dir / "image_statistics_by_contour_axis_bin.png"
    pdf = out_dir / "image_statistics_by_contour_axis_bin.pdf"
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    outputs.extend([str(png), str(pdf)])
    return outputs


def plot_near_mode_feature_summary(summary: pd.DataFrame, out_dir: Path, *, dpi: int) -> list[str]:
    outputs: list[str] = []
    if summary.empty:
        return outputs
    features = [feature for feature in IMAGE_FEATURES if feature in set(summary["feature"].astype(str))]
    ncols = 3
    nrows = int(math.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 3.1 * nrows), squeeze=False)
    colors = {"near_horizontal_axis0pm15": "#7b3294", "near_vertical_axis90pm15": "#008837"}
    labels = {"near_horizontal_axis0pm15": "horizontal", "near_vertical_axis90pm15": "vertical"}
    for ax, feature in zip(axes.ravel(), features, strict=False):
        sub = summary[summary["feature"].astype(str) == feature].copy()
        for xpos, mode in enumerate(["near_horizontal_axis0pm15", "near_vertical_axis90pm15"]):
            row = sub[sub["mode"].astype(str) == mode]
            if row.empty:
                continue
            ax.errorbar(
                [xpos],
                row["median"].to_numpy(float),
                yerr=[
                    row["median"].to_numpy(float) - row["q25"].to_numpy(float),
                    row["q75"].to_numpy(float) - row["median"].to_numpy(float),
                ],
                fmt="o",
                color=colors[mode],
                capsize=4,
            )
        ax.set_xticks([0, 1], [labels["near_horizontal_axis0pm15"], labels["near_vertical_axis90pm15"]])
        ax.set_title(feature, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    for ax in axes.ravel()[len(features) :]:
        ax.axis("off")
    fig.suptitle("Selected BackImage windows: near-horizontal vs near-vertical image statistics", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = out_dir / "image_statistics_horizontal_vs_vertical_modes.png"
    pdf = out_dir / "image_statistics_horizontal_vs_vertical_modes.pdf"
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    outputs.extend([str(png), str(pdf)])
    return outputs


def plot_unit_orientation_density(summary: pd.DataFrame, out_dir: Path, *, dpi: int) -> list[str]:
    outputs: list[str] = []
    if summary.empty:
        return outputs
    sources = [source for source, _ in ORIENTATION_COLUMNS if source in set(summary["orientation_source"].astype(str))]
    sf_groups = [g for g in ["low_sf", "middle_sf", "high_sf"] if g in set(summary["sf_group"].astype(str))]
    colors = {"low_sf": "#2c7fb8", "middle_sf": "0.60", "high_sf": "#d34848"}
    fig, axes = plt.subplots(len(sources), 1, figsize=(10, 3.0 * len(sources)), squeeze=False)
    for ax, source in zip(axes.ravel(), sources, strict=True):
        sub = summary[summary["orientation_source"].astype(str) == source].copy()
        centers = sorted(
            0.5
            * (
                sub.drop_duplicates("axis_bin")["axis_bin_start_deg"].to_numpy(float)
                + sub.drop_duplicates("axis_bin")["axis_bin_stop_deg"].to_numpy(float)
            )
        )
        width = 7.0
        offsets = np.linspace(-width, width, num=max(len(sf_groups), 1))
        for offset, sf_group in zip(offsets, sf_groups, strict=False):
            g = sub[sub["sf_group"].astype(str) == sf_group].sort_values("axis_bin_start_deg")
            x = 0.5 * (g["axis_bin_start_deg"].to_numpy(float) + g["axis_bin_stop_deg"].to_numpy(float)) + offset
            ax.bar(x, g["n_units"].to_numpy(float), width=width * 0.9, color=colors.get(sf_group, "0.4"), label=sf_group)
        ax.set_xticks(centers, [f"{int(c - 15)}-{int(c + 15)}" for c in centers])
        ax.set_ylabel("units")
        ax.set_title(f"{source} preferred orientation density")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(frameon=False, ncols=len(sf_groups), fontsize=9)
    axes[-1, 0].set_xlabel("preferred orientation bin (deg, axial)")
    fig.tight_layout()
    png = out_dir / "unit_orientation_density_by_sf_group.png"
    pdf = out_dir / "unit_orientation_density_by_sf_group.pdf"
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    outputs.extend([str(png), str(pdf)])
    return outputs


def plot_high_sf_axis_gap(gaps: pd.DataFrame, out_dir: Path, *, dpi: int) -> list[str]:
    outputs: list[str] = []
    if gaps.empty:
        return outputs
    views = ["across_along0", "across_along1", "along_across0", "along_across1"]
    bands = sorted(gaps["band"].astype(str).unique())
    matrix = np.full((len(views), len(bands)), np.nan, dtype=float)
    for i, view in enumerate(views):
        for j, band in enumerate(bands):
            row = gaps[(gaps["view"].astype(str) == view) & (gaps["band"].astype(str) == band)]
            if not row.empty:
                matrix[i, j] = float(row["aligned_minus_orthogonal_start"].iloc[0])
    fig, ax = plt.subplots(figsize=(10.5, 3.7))
    vmax = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(np.arange(len(views)), views)
    labels = [band.replace("axis_bin_", "").replace("_", "-") for band in bands]
    ax.set_xticks(np.arange(len(bands)), labels, rotation=0)
    ax.set_title("High-SF weighted SSI baseline gap: aligned minus orthogonal")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="bits/spike")
    fig.tight_layout()
    png = out_dir / "high_sf_aligned_minus_orthogonal_gap_by_contour_axis_bin.png"
    pdf = out_dir / "high_sf_aligned_minus_orthogonal_gap_by_contour_axis_bin.pdf"
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    outputs.extend([str(png), str(pdf)])
    return outputs


def main() -> None:
    args = parse_args()
    long_run_dir = args.long_run_dir
    out_dir = args.out_dir or (long_run_dir / "confounder_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = axis_bins(float(args.axis_bin_width_deg))

    selected_csv = long_run_dir / "balanced_source_windows" / "selected_windows.csv"
    windows = pd.read_csv(selected_csv)
    units = pd.read_csv(args.sf_groups_csv)

    image_axis = summarize_features_by_axis(windows, edges)
    image_axis_csv = out_dir / "selected_image_feature_by_axis_bin_summary.csv"
    image_axis.to_csv(image_axis_csv, index=False)

    image_modes = summarize_near_modes(windows)
    image_modes_csv = out_dir / "selected_image_feature_horizontal_vs_vertical_summary.csv"
    image_modes.to_csv(image_modes_csv, index=False)

    unit_density = summarize_unit_orientation_density(units, edges)
    unit_density_csv = out_dir / "unit_orientation_density_by_sf_group.csv"
    unit_density.to_csv(unit_density_csv, index=False)

    gaps = summarize_high_sf_axis_gaps(long_run_dir)
    gaps_csv = out_dir / "high_sf_axis_bin_alignment_gap_summary.csv"
    gaps.to_csv(gaps_csv, index=False)

    figure_outputs: list[str] = []
    figure_outputs.extend(plot_image_feature_axis_summary(image_axis, out_dir, dpi=int(args.dpi)))
    figure_outputs.extend(plot_near_mode_feature_summary(image_modes, out_dir, dpi=int(args.dpi)))
    figure_outputs.extend(plot_unit_orientation_density(unit_density, out_dir, dpi=int(args.dpi)))
    figure_outputs.extend(plot_high_sf_axis_gap(gaps, out_dir, dpi=int(args.dpi)))

    summary_payload = {
        "analysis": "backimage_rr100_long_run_confounder_diagnostics",
        "long_run_dir": long_run_dir,
        "selected_windows_csv": selected_csv,
        "sf_groups_csv": args.sf_groups_csv,
        "axis_bin_width_deg": float(args.axis_bin_width_deg),
        "n_selected_windows": int(windows.shape[0]),
        "n_units_by_sf_group": units.groupby("sf_group", sort=True).size().to_dict(),
        "csv_outputs": {
            "image_axis": image_axis_csv,
            "image_horizontal_vs_vertical": image_modes_csv,
            "unit_orientation_density": unit_density_csv,
            "high_sf_axis_gaps": gaps_csv,
        },
        "figure_outputs": figure_outputs,
    }
    write_json(out_dir / "summary.json", summary_payload)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
