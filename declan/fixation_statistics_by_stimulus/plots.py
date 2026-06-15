from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_qc_plots(out_dir: Path, window_rows: list[dict[str, Any]]) -> list[str]:
    if not window_rows:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    key_metrics = ["rms_radius_deg", "step_mean_deg", "speed_mean_deg_s", "anisotropy"]
    stimuli = sorted({str(r["stimulus"]) for r in window_rows})
    fig, axs = plt.subplots(1, len(key_metrics), figsize=(4.0 * len(key_metrics), 3.2), squeeze=False)
    for ax, metric in zip(axs[0], key_metrics, strict=True):
        data = []
        labels = []
        for stim in stimuli:
            vals = np.asarray([float(r.get(metric, np.nan)) for r in window_rows if str(r["stimulus"]) == stim], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                data.append(vals)
                labels.append(stim)
        if data:
            ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(color="0.9", lw=0.6)
    fig.tight_layout()
    path = figure_dir / "window_metric_distributions.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(str(path))

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for stim in stimuli:
        xs = [float(r.get("rms_radius_deg", np.nan)) for r in window_rows if str(r["stimulus"]) == stim]
        ys = [float(r.get("speed_mean_deg_s", np.nan)) for r in window_rows if str(r["stimulus"]) == stim]
        ax.scatter(xs, ys, s=10, alpha=0.35, label=stim)
    ax.set_xlabel("RMS radius (deg)")
    ax.set_ylabel("mean speed (deg/s)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="0.9", lw=0.6)
    fig.tight_layout()
    path = figure_dir / "rms_radius_vs_speed.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(str(path))
    return written

