#!/usr/bin/env python3
"""Show the dense SF/TF group tuning surfaces used for overlay contours."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_eye_movement_power_spectrum_shift import ROOT
from plot_ssi_sftf_mechanism_bridge import (
    DEFAULT_TUNING_POINTS_CSV,
    TUNING_GROUP_COMBINE,
    TUNING_GROUP_COLORS,
    TUNING_GROUP_LABELS,
    TUNING_GROUP_ORDER,
)


PANEL_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels"
DEFAULT_OUT_DIR = PANEL_DIR / "sftf_group_tuning_surfaces"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def matrix_from_group(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sf = np.asarray(sorted(group["spatial_cpd"].unique()), dtype=np.float64)
    tf = np.asarray(sorted(group["temporal_hz"].unique()), dtype=np.float64)
    mat = np.full((tf.size, sf.size), np.nan, dtype=np.float64)
    sf_index = {float(v): i for i, v in enumerate(sf)}
    tf_index = {float(v): i for i, v in enumerate(tf)}
    for row in group.itertuples(index=False):
        mat[tf_index[float(row.temporal_hz)], sf_index[float(row.spatial_cpd)]] = float(row.unit_surface_z)
    return sf, tf, mat


def normalize_surface(mat: np.ndarray) -> np.ndarray:
    finite = np.asarray(mat, dtype=np.float64)[np.isfinite(mat)]
    if finite.size == 0:
        return np.asarray(mat, dtype=np.float64) * np.nan
    lo = float(np.nanpercentile(finite, 5.0))
    hi = float(np.nanpercentile(finite, 98.0))
    if hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if hi <= lo:
        return np.asarray(mat, dtype=np.float64) * 0.0
    return np.clip((np.asarray(mat, dtype=np.float64) - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


def load_surfaces(points_csv: Path) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], pd.DataFrame]:
    points = pd.read_csv(points_csv)
    required = {"unit_index", "sf_group", "spatial_cpd", "temporal_hz", "unit_surface_z"}
    missing = required.difference(points.columns)
    if missing:
        raise ValueError(f"Tuning points file is missing columns: {sorted(missing)}")
    points["tuning_group"] = points["sf_group"].map(TUNING_GROUP_COMBINE).fillna(points["sf_group"])
    grouped = (
        points.dropna(subset=["sf_group", "spatial_cpd", "temporal_hz", "unit_surface_z"])
        .groupby(["tuning_group", "spatial_cpd", "temporal_hz"], sort=True)["unit_surface_z"]
        .mean()
        .reset_index()
    )
    surfaces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, object]] = []
    for group_id, group in grouped.groupby("tuning_group", sort=False):
        sf, tf, mat = matrix_from_group(group)
        surfaces[str(group_id)] = (sf, tf, mat)
        units = points[points["tuning_group"].eq(group_id)]["unit_index"].nunique()
        peak = np.unravel_index(int(np.nanargmax(mat)), mat.shape)
        rows.append(
            {
                "tuning_group": str(group_id),
                "tuning_group_label": TUNING_GROUP_LABELS.get(str(group_id), str(group_id)),
                "n_units": int(units),
                "peak_spatial_cpd": float(sf[peak[1]]),
                "peak_temporal_hz": float(tf[peak[0]]),
                "raw_surface_min": float(np.nanmin(mat)),
                "raw_surface_max": float(np.nanmax(mat)),
                "raw_surface_mean": float(np.nanmean(mat)),
            }
        )
    return surfaces, pd.DataFrame(rows)


def plot_surfaces(
    surfaces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    contour_level: float,
) -> Path:
    raw_values = np.concatenate([mat[np.isfinite(mat)] for _sf, _tf, mat in surfaces.values()])
    raw_vmin = float(np.nanpercentile(raw_values, 2.0))
    raw_vmax = float(np.nanpercentile(raw_values, 98.0))
    present_groups = [group_id for group_id in TUNING_GROUP_ORDER if group_id in surfaces]
    n_cols = max(len(present_groups), 1)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.9 * n_cols + 1.15, 6.8), squeeze=False)
    fig.subplots_adjust(left=0.105, right=0.86, bottom=0.09, top=0.88, hspace=0.38, wspace=0.24)
    raw_mesh = None
    norm_mesh = None
    for col, group_id in enumerate(present_groups):
        sf, tf, mat = surfaces[group_id]
        row = summary[summary["tuning_group"].eq(group_id)].iloc[0]
        raw_mesh = axes[0, col].pcolormesh(sf, tf, mat, shading="nearest", cmap="viridis", vmin=raw_vmin, vmax=raw_vmax)
        norm = normalize_surface(mat)
        norm_mesh = axes[1, col].pcolormesh(sf, tf, norm, shading="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
        axes[1, col].contour(
            sf,
            tf,
            norm,
            levels=[float(contour_level)],
            colors=[TUNING_GROUP_COLORS[group_id]],
            linewidths=1.8,
        )
        for ax in axes[:, col]:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([0.4, 1, 3, 10], ["0.4", "1", "3", "10"])
            ax.set_yticks([0.4, 1, 3, 10, 30, 50], ["0.4", "1", "3", "10", "30", "50"])
            ax.spines[["top", "right"]].set_visible(False)
        axes[0, col].set_title(
            f"{TUNING_GROUP_LABELS.get(group_id, group_id)}\n"
            f"n={int(row['n_units'])}, peak {float(row['peak_spatial_cpd']):.3g} cpd / "
            f"{float(row['peak_temporal_hz']):.3g} Hz",
            loc="left",
            fontweight="bold",
        )
        axes[1, col].set_xlabel("spatial frequency (cycles/deg)")
    axes[0, 0].set_ylabel("raw mean surface\nTF (Hz)")
    axes[1, 0].set_ylabel("normalized surface\nTF (Hz)")
    if raw_mesh is not None:
        cbar = fig.colorbar(raw_mesh, ax=axes[0, :].tolist(), pad=0.012, shrink=0.86)
        cbar.set_label("mean unit_surface_z")
    if norm_mesh is not None:
        cbar = fig.colorbar(norm_mesh, ax=axes[1, :].tolist(), pad=0.012, shrink=0.86)
        cbar.set_label(f"5-98 percentile normalized; contour={float(contour_level):.2f}")
    fig.suptitle(
        "Dense grating SF/TF group surfaces used for the heatmap contour overlay",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    png = out_dir / "sftf_group_tuning_surfaces.png"
    pdf = out_dir / "sftf_group_tuning_surfaces.pdf"
    svg = out_dir / "sftf_group_tuning_surfaces.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png


def run(args: argparse.Namespace) -> dict[str, Path]:
    configure_matplotlib()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    surfaces, summary = load_surfaces(Path(args.tuning_points_csv))
    summary_csv = out_dir / "sftf_group_tuning_surface_summary.csv"
    summary.to_csv(summary_csv, index=False)
    png = plot_surfaces(surfaces, summary, out_dir, contour_level=float(args.contour_level))
    return {"png": png, "summary_csv": summary_csv}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning-points-csv", type=Path, default=DEFAULT_TUNING_POINTS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--contour-level", type=float, default=0.68)
    return parser.parse_args()


def main() -> None:
    paths = run(parse_args())
    print(f"Wrote {paths['png']}")
    print(f"Wrote {paths['summary_csv']}")


if __name__ == "__main__":
    main()
