#!/usr/bin/env python3
"""Plot alternate views of the BackImage RR100 mean-map SSI grid."""

from __future__ import annotations

import argparse
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
DEFAULT_GRID_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_contour_axis_rr100_spatial_ssi_mean_map_grid5_merged_v1/"
    "grid_condition_summary.csv"
)
DEFAULT_OUT_DIR = DEFAULT_GRID_CSV.parent / "alternate_views"
SSI_COL = "population_mean_map_ssi_bits_per_spike_mean"
SEM_COL = "population_mean_map_ssi_bits_per_spike_sem"
DELTA_COL = "grid_mean_map_delta_vs_0x0x"
Z_COL = "grid_mean_map_population_z"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
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
        return value if np.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def scale_label(value: float) -> str:
    return f"{float(value):g}x"


def load_grid(path: Path) -> tuple[pd.DataFrame, list[float]]:
    df = pd.read_csv(path)
    required = {"grid_along_scale", "grid_across_scale", SSI_COL, SEM_COL}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    if DELTA_COL not in df.columns:
        static = df[
            np.isclose(df["grid_along_scale"].astype(float), 0.0)
            & np.isclose(df["grid_across_scale"].astype(float), 0.0)
        ]
        static_value = float(static.iloc[0][SSI_COL]) if not static.empty else float(df[SSI_COL].min())
        df[DELTA_COL] = df[SSI_COL].astype(float) - static_value
    if Z_COL not in df.columns:
        values = df[SSI_COL].astype(float).to_numpy()
        df[Z_COL] = (values - float(np.nanmean(values))) / max(float(np.nanstd(values)), 1e-12)
    along_scales = sorted(float(v) for v in df["grid_along_scale"].dropna().unique())
    across_scales = sorted(float(v) for v in df["grid_across_scale"].dropna().unique())
    if along_scales != across_scales:
        raise ValueError("Expected matching along/across scale values for a square grid.")
    return df, along_scales


def pivot_grid(df: pd.DataFrame, scales: list[float], column: str) -> pd.DataFrame:
    table = df.pivot(index="grid_along_scale", columns="grid_across_scale", values=column)
    return table.reindex(index=scales, columns=scales).astype(float)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int) -> dict[str, str]:
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def plot_line_slices(df: pd.DataFrame, scales: list[float], out_dir: Path, dpi: int) -> dict[str, str]:
    mean_grid = pivot_grid(df, scales, SSI_COL)
    sem_grid = pivot_grid(df, scales, SEM_COL)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(scales)))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)

    for color, along in zip(colors, scales, strict=True):
        y = mean_grid.loc[along].to_numpy(dtype=float)
        e = sem_grid.loc[along].to_numpy(dtype=float)
        axes[0].plot(scales, y, marker="o", color=color, linewidth=1.8, label=f"along {scale_label(along)}")
        axes[0].fill_between(scales, y - e, y + e, color=color, alpha=0.13, linewidth=0)
    axes[0].set_title("Across slices")
    axes[0].set_xlabel("across-contour scale")
    axes[0].set_ylabel("population mean-map SSI")
    axes[0].axvline(1.0, color="0.55", linestyle=":", linewidth=1.0)
    axes[0].grid(True, color="0.9", linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=8)

    for color, across in zip(colors, scales, strict=True):
        y = mean_grid[across].to_numpy(dtype=float)
        e = sem_grid[across].to_numpy(dtype=float)
        axes[1].plot(scales, y, marker="o", color=color, linewidth=1.8, label=f"across {scale_label(across)}")
        axes[1].fill_between(scales, y - e, y + e, color=color, alpha=0.13, linewidth=0)
    axes[1].set_title("Along slices")
    axes[1].set_xlabel("along-contour scale")
    axes[1].set_ylabel("population mean-map SSI")
    axes[1].axvline(1.0, color="0.55", linestyle=":", linewidth=1.0)
    axes[1].grid(True, color="0.9", linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle("BackImage RR100 mean-map SSI: line slices through the 5x5 grid", fontsize=12)
    return save_figure(fig, out_dir, "grid5_mean_map_ssi_line_slices", dpi)


def plot_delta_slices(df: pd.DataFrame, scales: list[float], out_dir: Path, dpi: int) -> dict[str, str]:
    delta_grid = pivot_grid(df, scales, DELTA_COL)
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(scales)))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)

    for color, along in zip(colors, scales, strict=True):
        axes[0].plot(
            scales,
            delta_grid.loc[along].to_numpy(dtype=float),
            marker="o",
            color=color,
            linewidth=1.8,
            label=f"along {scale_label(along)}",
        )
    axes[0].axhline(0.0, color="0.35", linewidth=1.0)
    axes[0].axvline(1.0, color="0.55", linestyle=":", linewidth=1.0)
    axes[0].set_title("Across slices")
    axes[0].set_xlabel("across-contour scale")
    axes[0].set_ylabel("mean-map SSI minus 0x/0x")
    axes[0].grid(True, color="0.9", linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=8)

    for color, across in zip(colors, scales, strict=True):
        axes[1].plot(
            scales,
            delta_grid[across].to_numpy(dtype=float),
            marker="o",
            color=color,
            linewidth=1.8,
            label=f"across {scale_label(across)}",
        )
    axes[1].axhline(0.0, color="0.35", linewidth=1.0)
    axes[1].axvline(1.0, color="0.55", linestyle=":", linewidth=1.0)
    axes[1].set_title("Along slices")
    axes[1].set_xlabel("along-contour scale")
    axes[1].set_ylabel("mean-map SSI minus 0x/0x")
    axes[1].grid(True, color="0.9", linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle("BackImage RR100 mean-map SSI: deltas from static 0x/0x", fontsize=12)
    return save_figure(fig, out_dir, "grid5_mean_map_ssi_delta_slices", dpi)


def plot_main_effects_and_optima(df: pd.DataFrame, scales: list[float], out_dir: Path, dpi: int) -> dict[str, str]:
    mean_grid = pivot_grid(df, scales, SSI_COL)
    delta_grid = pivot_grid(df, scales, DELTA_COL)
    row_mean = mean_grid.mean(axis=1)
    col_mean = mean_grid.mean(axis=0)
    best_across = mean_grid.idxmax(axis=1)
    best_along = mean_grid.idxmax(axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), constrained_layout=True)

    axes[0, 0].plot(scales, row_mean.to_numpy(dtype=float), marker="o", color="#2866b0", linewidth=2.0)
    axes[0, 0].set_title("Mean over across scales")
    axes[0, 0].set_xlabel("along-contour scale")
    axes[0, 0].set_ylabel("population mean-map SSI")
    axes[0, 0].grid(True, color="0.9", linewidth=0.8)

    axes[0, 1].plot(scales, col_mean.to_numpy(dtype=float), marker="o", color="#b04c28", linewidth=2.0)
    axes[0, 1].set_title("Mean over along scales")
    axes[0, 1].set_xlabel("across-contour scale")
    axes[0, 1].set_ylabel("population mean-map SSI")
    axes[0, 1].grid(True, color="0.9", linewidth=0.8)

    axes[1, 0].plot(scales, [float(best_across.loc[a]) for a in scales], marker="o", color="#2866b0", linewidth=2.0)
    axes[1, 0].set_title("Best across scale at each along scale")
    axes[1, 0].set_xlabel("along-contour scale")
    axes[1, 0].set_ylabel("argmax across scale")
    axes[1, 0].set_yticks(scales)
    axes[1, 0].grid(True, color="0.9", linewidth=0.8)

    axes[1, 1].plot(scales, [float(best_along.loc[c]) for c in scales], marker="o", color="#b04c28", linewidth=2.0)
    axes[1, 1].set_title("Best along scale at each across scale")
    axes[1, 1].set_xlabel("across-contour scale")
    axes[1, 1].set_ylabel("argmax along scale")
    axes[1, 1].set_yticks(scales)
    axes[1, 1].grid(True, color="0.9", linewidth=0.8)

    static_delta = float(delta_grid.loc[0.0, 0.0]) if 0.0 in delta_grid.index and 0.0 in delta_grid.columns else 0.0
    for ax in axes.ravel():
        ax.axvline(1.0, color="0.55", linestyle=":", linewidth=1.0)
    axes[0, 0].text(
        0.02,
        0.02,
        f"static delta = {static_delta:+.4f}",
        transform=axes[0, 0].transAxes,
        fontsize=8,
        color="0.35",
    )

    fig.suptitle("BackImage RR100 mean-map SSI: main effects and grid optima", fontsize=12)
    return save_figure(fig, out_dir, "grid5_mean_map_ssi_main_effects_and_optima", dpi)


def annotate_matrix(ax: plt.Axes, values: np.ndarray, fmt: str) -> None:
    finite = values[np.isfinite(values)]
    threshold = float(np.nanmean(finite)) if finite.size else 0.0
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            if not np.isfinite(value):
                continue
            text_color = "white" if value < threshold else "black"
            ax.text(x, y, format(float(value), fmt), ha="center", va="center", fontsize=6.8, color=text_color)


def plot_interaction_residuals(df: pd.DataFrame, scales: list[float], out_dir: Path, dpi: int) -> dict[str, str]:
    mean_grid = pivot_grid(df, scales, SSI_COL)
    values = mean_grid.to_numpy(dtype=float)
    grand = float(np.nanmean(values))
    row_effect = np.nanmean(values, axis=1) - grand
    col_effect = np.nanmean(values, axis=0) - grand
    additive = grand + row_effect[:, None] + col_effect[None, :]
    residual = values - additive
    panels = [
        ("observed mean-map SSI", values, "viridis", ".4f"),
        ("additive row+column expectation", additive, "viridis", ".4f"),
        ("interaction residual", residual, "coolwarm", "+.4f"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.4), constrained_layout=True)
    labels = [scale_label(s) for s in scales]
    for ax, (title, matrix, cmap, fmt) in zip(axes, panels, strict=True):
        if cmap == "coolwarm":
            limit = max(float(np.nanpercentile(np.abs(matrix), 98.0)), 1e-6)
            im = ax.imshow(matrix, origin="lower", cmap=cmap, vmin=-limit, vmax=limit)
        else:
            im = ax.imshow(matrix, origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("across scale")
        ax.set_ylabel("along scale")
        ax.set_xticks(np.arange(len(scales)), labels=labels, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(scales)), labels=labels)
        annotate_matrix(ax, matrix, fmt)
        cbar = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.03)
        cbar.ax.tick_params(labelsize=7)
    fig.suptitle("BackImage RR100 mean-map SSI: additive structure and interaction residual", fontsize=12)
    return save_figure(fig, out_dir, "grid5_mean_map_ssi_interaction_residuals", dpi)


def plot_bubble_scatter(df: pd.DataFrame, scales: list[float], out_dir: Path, dpi: int) -> dict[str, str]:
    plot_df = df.copy()
    deltas = plot_df[DELTA_COL].astype(float).to_numpy()
    centered = np.abs(deltas)
    size_scale = centered / max(float(np.nanmax(centered)), 1e-12)
    sizes = 180.0 + 900.0 * size_scale

    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    limit = max(float(np.nanpercentile(np.abs(deltas), 98.0)), 1e-6)
    scatter = ax.scatter(
        plot_df["grid_across_scale"].astype(float),
        plot_df["grid_along_scale"].astype(float),
        c=deltas,
        s=sizes,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        edgecolors="0.2",
        linewidths=0.7,
    )
    for _, row in plot_df.iterrows():
        ax.text(
            float(row["grid_across_scale"]),
            float(row["grid_along_scale"]),
            f"{float(row[DELTA_COL]):+.4f}",
            ha="center",
            va="center",
            fontsize=7,
            color="black",
        )
    ax.axhline(1.0, color="0.55", linestyle=":", linewidth=1.0)
    ax.axvline(1.0, color="0.55", linestyle=":", linewidth=1.0)
    ax.set_xticks(scales, [scale_label(s) for s in scales])
    ax.set_yticks(scales, [scale_label(s) for s in scales])
    ax.set_xlim(min(scales) - 0.18, max(scales) + 0.18)
    ax.set_ylim(min(scales) - 0.18, max(scales) + 0.18)
    ax.set_xlabel("across-contour scale")
    ax.set_ylabel("along-contour scale")
    ax.set_title("Mean-map SSI delta from static 0x/0x")
    ax.grid(True, color="0.9", linewidth=0.8)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.048, pad=0.03)
    cbar.set_label("SSI delta")
    return save_figure(fig, out_dir, "grid5_mean_map_ssi_delta_bubble_scatter", dpi)


def plot_ranked_conditions(df: pd.DataFrame, out_dir: Path, dpi: int) -> dict[str, str]:
    ranked = df.sort_values(SSI_COL, ascending=False).reset_index(drop=True)
    labels = [
        f"a{float(row.grid_along_scale):g}/c{float(row.grid_across_scale):g}"
        for row in ranked.itertuples(index=False)
    ]
    values = ranked[SSI_COL].astype(float).to_numpy()
    deltas = ranked[DELTA_COL].astype(float).to_numpy()
    colors = plt.cm.coolwarm(
        (deltas - float(np.nanmin(deltas))) / max(float(np.nanmax(deltas) - np.nanmin(deltas)), 1e-12)
    )

    fig, ax = plt.subplots(figsize=(11.0, 5.2), constrained_layout=True)
    ax.bar(np.arange(len(ranked)), values, color=colors, edgecolor="0.25", linewidth=0.4)
    static = float(df.loc[np.isclose(df["grid_along_scale"], 0.0) & np.isclose(df["grid_across_scale"], 0.0), SSI_COL].iloc[0])
    ax.axhline(static, color="0.25", linestyle="--", linewidth=1.0, label="0x/0x static")
    ax.set_xticks(np.arange(len(ranked)), labels=labels, rotation=60, ha="right")
    ax.set_ylabel("population mean-map SSI")
    ax.set_title("BackImage RR100 mean-map SSI: ranked grid conditions")
    ax.grid(True, axis="y", color="0.9", linewidth=0.8)
    ax.legend(frameon=False)
    return save_figure(fig, out_dir, "grid5_mean_map_ssi_ranked_conditions", dpi)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, scales = load_grid(Path(args.grid_csv))
    outputs = {
        "line_slices": plot_line_slices(df, scales, out_dir, int(args.dpi)),
        "delta_slices": plot_delta_slices(df, scales, out_dir, int(args.dpi)),
        "main_effects_and_optima": plot_main_effects_and_optima(df, scales, out_dir, int(args.dpi)),
        "interaction_residuals": plot_interaction_residuals(df, scales, out_dir, int(args.dpi)),
        "delta_bubble_scatter": plot_bubble_scatter(df, scales, out_dir, int(args.dpi)),
        "ranked_conditions": plot_ranked_conditions(df, out_dir, int(args.dpi)),
    }
    mean_grid = pivot_grid(df, scales, SSI_COL)
    best = df.iloc[int(df[SSI_COL].astype(float).argmax())]
    summary = {
        "grid_csv": Path(args.grid_csv),
        "out_dir": out_dir,
        "scales": scales,
        "best_condition": {
            "along_scale": float(best["grid_along_scale"]),
            "across_scale": float(best["grid_across_scale"]),
            "population_mean_map_ssi": float(best[SSI_COL]),
            "delta_vs_0x0x": float(best[DELTA_COL]),
            "z": float(best[Z_COL]),
        },
        "row_mean_by_along_scale": {
            scale_label(scale): float(value) for scale, value in mean_grid.mean(axis=1).items()
        },
        "column_mean_by_across_scale": {
            scale_label(scale): float(value) for scale, value in mean_grid.mean(axis=0).items()
        },
        "outputs": outputs,
    }
    summary_path = out_dir / "summary.json"
    write_json(summary_path, summary)
    print(f"Wrote alternate grid views to {out_dir}")
    for name, paths in outputs.items():
        print(f"{name}: {paths['png']}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
