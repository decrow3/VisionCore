#!/usr/bin/env python3
"""Replot checkpoint-20 selected response maps at maximal spatial change."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "outputs/fig4_active_sensing/rr100_corrected_ssi_map_first_smoke_checkpoint_20_v1"
OUT = CHECKPOINT / "checkpoint_20b2_stabilized_moving_difference_at_max_change.png"
TIMING = CHECKPOINT / "checkpoint_20b2_selected_frames.csv"


def main() -> None:
    if OUT.exists() or TIMING.exists():
        raise FileExistsError("Refusing to overwrite checkpoint-20b2 outputs")
    roles = pd.read_csv(CHECKPOINT / "selected_unit_roles.csv")
    maps = np.load(CHECKPOINT / "corrected_smoke_selected_response_maps.npz")
    image_index = 3
    traces = (1, 31)
    fig, axes = plt.subplots(len(roles), 6, figsize=(15.2, 2.5 * len(roles)), constrained_layout=True)
    timing_rows = []
    for row, role in roles.iterrows():
        unit = int(role["rr100_index"])
        baseline = maps[f"baseline_image_{image_index:02d}"][:, row]
        for block, trace in enumerate(traces):
            moving = maps[f"moving_image_{image_index:02d}_trace_{trace:02d}"][:, row]
            delta = moving - baseline
            spatial_change = np.mean(np.abs(delta), axis=(1, 2))
            frame = int(np.argmax(spatial_change))
            timing_rows.append({
                "rr100_index": unit,
                "trace_index": trace,
                "selected_frame": frame,
                "selection_metric": "maximum spatial mean absolute moving-minus-stabilized response",
                "mean_absolute_delta_hz": float(spatial_change[frame]),
                "mean_signed_delta_hz": float(delta[frame].mean()),
            })
            rate_max = max(float(np.percentile(baseline[frame], 99)), float(np.percentile(moving[frame], 99)), 1e-5)
            delta_limit = max(float(np.percentile(np.abs(delta[frame]), 99)), 1e-5)
            start = block * 3
            rate_image = None
            for offset, (label, values) in enumerate((("stabilized", baseline[frame]), ("moving", moving[frame]))):
                rate_image = axes[row, start + offset].imshow(values, cmap="magma", vmin=0, vmax=rate_max, interpolation="nearest")
                axes[row, start + offset].set_title(f"trace {trace} · frame {frame}\n{label}", fontsize=8)
            diff_image = axes[row, start + 2].imshow(
                delta[frame], cmap="RdBu_r",
                norm=TwoSlopeNorm(vmin=-delta_limit, vcenter=0.0, vmax=delta_limit),
                interpolation="nearest",
            )
            axes[row, start + 2].set_title(
                f"moving − stabilized\nmean {delta[frame].mean():+.3f} Hz", fontsize=8
            )
            fig.colorbar(rate_image, ax=axes[row, start:start + 2], shrink=0.65, pad=0.006, label="rate (Hz)")
            fig.colorbar(diff_image, ax=axes[row, start + 2], shrink=0.65, pad=0.006, label="Δ rate (Hz)")
        for col in range(6):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        half = str(role["sf_outer_third"]).replace("sf_", "").replace("_", " ")
        axes[row, 0].set_ylabel(f"u{unit:03d}\n{half}\npref {role['preferred_sf_cpd']:.2f} cpd", fontsize=8)
    fig.suptitle(
        "Corrected response maps at each unit/trace's maximal spatial change\n"
        "strong-contour image 3 · minimum-path versus maximum-path trace",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(OUT, dpi=210)
    plt.close(fig)
    pd.DataFrame(timing_rows).to_csv(TIMING, index=False)
    print(OUT)
    print(TIMING)


if __name__ == "__main__":
    main()
