#!/usr/bin/env python3
"""Plot RR100 endpoint final-map options for the SSI schematic."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAP_DIR = ROOT / "outputs" / "fig_ssi" / "rr100_schematic_endpoint_final_maps"
MAP_NPZ = MAP_DIR / "cache" / "schematic_rr100_final_maps.npz"
METRICS_CSV = MAP_DIR / "schematic_rr100_final_map_unit_metrics.csv"
OUT_BASE = MAP_DIR / "schematic_rr100_final_map_unit_options_gray_nearest"

CURRENT_FIGURE_UNIT_INDEX = 56
N_UNIT_COLS = 10
TOP_N = 100
DISPLAY_PERCENTILES = (1.0, 99.5)


def image_limits(image: np.ndarray) -> tuple[float, float]:
    values = np.asarray(image, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(finite, DISPLAY_PERCENTILES)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return 0.0, 1.0
    return float(vmin), float(vmax)


def format_sf_group(value: object) -> str:
    text = str(value) if value is not None and not pd.isna(value) else ""
    return text.replace("_", "-")


def main() -> None:
    if not MAP_NPZ.exists():
        raise FileNotFoundError(MAP_NPZ)
    if not METRICS_CSV.exists():
        raise FileNotFoundError(METRICS_CSV)

    with np.load(MAP_NPZ, allow_pickle=False) as data:
        final_maps = np.maximum(np.asarray(data["final_maps"], dtype=np.float64), 0.0)
        condition_labels = [str(x) for x in np.asarray(data["condition_label"]).astype(str)]

    metrics = pd.read_csv(METRICS_CSV)
    sort_cols = [col for col in ["figure_candidate_score", "real_minus_stable_map_ssi"] if col in metrics.columns]
    if sort_cols:
        metrics = metrics.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    metrics = metrics.head(min(int(TOP_N), int(metrics.shape[0]))).reset_index(drop=True)

    n_units = int(metrics.shape[0])
    n_rows = int(math.ceil(n_units / N_UNIT_COLS))
    fig, axes = plt.subplots(
        n_rows,
        N_UNIT_COLS * 2,
        figsize=(2.08 * N_UNIT_COLS, 1.42 * n_rows),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    for ax in axes.ravel():
        ax.set_axis_off()

    for pos, row in enumerate(metrics.itertuples(index=False)):
        unit = int(row.unit_index)
        r = pos // N_UNIT_COLS
        c = (pos % N_UNIT_COLS) * 2
        delta_ssi = float(getattr(row, "real_minus_stable_map_ssi", np.nan))
        real_ssi = float(getattr(row, "real_final_map_ssi_bits_per_spike", np.nan))
        stable_ssi = float(getattr(row, "stable_final_map_ssi_bits_per_spike", np.nan))
        sf_group = format_sf_group(getattr(row, "sf_group", ""))
        current = " *current*" if unit == CURRENT_FIGURE_UNIT_INDEX else ""

        for j, condition_title in enumerate(["real", "stable"]):
            ax = axes[r, c + j]
            image = final_maps[j, unit]
            vmin, vmax = image_limits(image)
            ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5 if unit != CURRENT_FIGURE_UNIT_INDEX else 1.3)
                spine.set_edgecolor("#444" if unit != CURRENT_FIGURE_UNIT_INDEX else "#c51f27")
            if j == 0:
                ax.set_title(
                    f"u{unit:03d}{current}\nΔSSI {delta_ssi:+.3f}  {sf_group}",
                    fontsize=5.8,
                    pad=2.0,
                )
            else:
                ax.set_title(
                    f"{condition_title}\n{real_ssi:.3f}/{stable_ssi:.3f}",
                    fontsize=5.8,
                    pad=2.0,
                )

    title_labels = ", ".join(condition_labels) if condition_labels else "real/stabilized"
    fig.suptitle(
        f"RR100 schematic endpoint map options ({title_labels}; gray, nearest, per-map 1/99.5%)",
        fontsize=12,
        y=0.997,
    )
    fig.tight_layout(pad=0.42, w_pad=0.04, h_pad=0.42, rect=(0.0, 0.0, 1.0, 0.985))

    png = OUT_BASE.with_suffix(".png")
    pdf = OUT_BASE.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(png)


if __name__ == "__main__":
    main()
