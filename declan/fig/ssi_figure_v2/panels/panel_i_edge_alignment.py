#!/usr/bin/env python3
"""Panel I: edge-following alignment by local image coherence.

This is a compact copy of the 4E candidate-3A plotting logic from
``declan/figure4_active_sensing_atlas/scripts/build_panel_e_single_panel_candidates.py``.
It regenerates the panel from the reviewed BackImage FEM-window table rather
than embedding the existing PDF/PNG artifact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels"
WINDOWS_CSV = (
    ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_image_structure_reviewed_v2_screenfiltered_yfix"
    / "backimage_image_fem_windows.csv"
)

EDGE_BLUE = "#244f7a"
COUNT_GRAY = "#dfe3e8"
GRID = "#d8dde3"
INK = "#111111"


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


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_panel_values(windows_csv: Path = WINDOWS_CSV) -> pd.DataFrame:
    """Compute coherence-binned edge-alignment values from reviewed windows."""
    if not windows_csv.exists():
        raise FileNotFoundError(windows_csv)

    windows = pd.read_csv(windows_csv)
    required = ["image_feature_ok", "image_orientation_coherence", "drift_edge_cos2", "session"]
    missing = [col for col in required if col not in windows.columns]
    if missing:
        raise ValueError(f"Missing required columns in {windows_csv}: {missing}")

    work = windows[windows["image_feature_ok"].astype(bool)].copy()
    work["image_orientation_coherence"] = pd.to_numeric(work["image_orientation_coherence"], errors="coerce")
    work["drift_edge_cos2"] = pd.to_numeric(work["drift_edge_cos2"], errors="coerce")
    work = work[
        np.isfinite(work["image_orientation_coherence"]) & np.isfinite(work["drift_edge_cos2"])
    ].copy()

    bins = np.linspace(0.0, 1.0, 11)
    rows: list[dict[str, float | int]] = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        if hi == bins[-1]:
            block = work[(work["image_orientation_coherence"] >= lo) & (work["image_orientation_coherence"] <= hi)]
        else:
            block = work[(work["image_orientation_coherence"] >= lo) & (work["image_orientation_coherence"] < hi)]
        if block.empty:
            continue
        values = block["drift_edge_cos2"].to_numpy(dtype=float)
        mean = float(values.mean())
        sem95 = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        rows.append(
            {
                "bin_low": float(lo),
                "bin_high": float(hi),
                "bin_center": float((lo + hi) / 2.0),
                "mean_edge_alignment_index": mean,
                "ci95_low": mean - sem95,
                "ci95_high": mean + sem95,
                "n_windows": int(len(block)),
                "n_sessions": int(block["session"].nunique()),
            }
        )

    values = pd.DataFrame(rows)
    if values.empty:
        raise ValueError(f"No finite edge-alignment values in {windows_csv}")
    return values


def draw_panel(
    ax: plt.Axes,
    *,
    label: str = "I",
    title: str = "Edge alignment",
    values: pd.DataFrame | None = None,
    show_counts: bool = True,
) -> pd.DataFrame:
    """Draw the compact edge-coherence alignment panel on ``ax``."""
    values = load_panel_values() if values is None else values.copy()

    count_ax = None
    if show_counts:
        count_ax = ax.twinx()
        ax.set_zorder(count_ax.get_zorder() + 1)
        ax.patch.set_visible(False)
        count_ax.bar(
            values["bin_center"],
            values["n_windows"],
            width=0.075,
            color=COUNT_GRAY,
            edgecolor="none",
            alpha=0.82,
            zorder=0,
        )
        count_ax.set_yticks([])
        count_ax.spines["top"].set_visible(False)
        count_ax.spines["right"].set_visible(False)
        count_ax.spines["left"].set_visible(False)
        count_ax.grid(False)

    y = values["mean_edge_alignment_index"].to_numpy(dtype=float)
    lo = values["ci95_low"].to_numpy(dtype=float)
    hi = values["ci95_high"].to_numpy(dtype=float)
    ax.errorbar(
        values["bin_center"],
        y,
        yerr=np.vstack([y - lo, hi - y]),
        color=EDGE_BLUE,
        marker="o",
        markersize=3.8,
        lw=1.55,
        capsize=0,
        zorder=4,
    )
    ax.axhline(0.0, color="#242a2f", lw=0.8)
    ax.set_xlim(0.0, 1.0)
    y_top = max(0.38, float(np.nanmax(hi)) + 0.025)
    ax.set_ylim(-0.04, y_top)
    ax.set_xlabel("local edge coherence")
    ax.set_ylabel("edge-following alignment")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"{label}  {title}", loc="left", fontsize=9.0, fontweight="bold", pad=4, color=INK)
    _clean_axis(ax)
    return values


def build_panel(out_dir: Path = OUT_DIR) -> dict[str, Path]:
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    values = load_panel_values()
    values.to_csv(out_dir / "panel_i_edge_alignment_values.csv", index=False)

    fig, ax = plt.subplots(figsize=(2.35, 2.25), constrained_layout=True)
    draw_panel(ax, values=values)
    paths = {
        "png": out_dir / "panel_i_edge_alignment.png",
        "pdf": out_dir / "panel_i_edge_alignment.pdf",
        "svg": out_dir / "panel_i_edge_alignment.svg",
    }
    fig.savefig(paths["png"], dpi=220)
    fig.savefig(paths["pdf"], dpi=300)
    fig.savefig(paths["svg"], dpi=300)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    paths = build_panel(args.out_dir)
    for key in ("png", "pdf", "svg"):
        print(paths[key])


if __name__ == "__main__":
    main()
