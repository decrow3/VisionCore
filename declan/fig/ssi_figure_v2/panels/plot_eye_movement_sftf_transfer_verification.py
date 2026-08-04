#!/usr/bin/env python3
"""Verify SSI-v3 joint SF-TF heatmaps against the Rucci/Mostofi convention.

The existing joint diagnostic stores both the image-weighted modulation power
``P_image(k) * Q(k, f_t)`` and the motion-only redistribution term ``Q``.  The
Mostofi/Rucci Figure 2 convention is closer to the latter: power available at
each temporal frequency given unit spatial power at each spatial frequency.
"""

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

from plot_eye_movement_joint_sftf_power import (
    DEFAULT_OUT_DIR,
    db,
    spatial_edges_from_centers,
    temporal_edges,
)


DEFAULT_SUMMARY = DEFAULT_OUT_DIR / "eye_movement_joint_sftf_radial_temporal_summary.csv"


def matrix_from_column(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sf = np.asarray(sorted(frame["spatial_frequency_cpd"].unique()), dtype=np.float64)
    tf = np.asarray(sorted(frame["temporal_frequency_hz"].unique()), dtype=np.float64)
    mat = np.full((tf.size, sf.size), np.nan, dtype=np.float64)
    sf_index = {float(v): i for i, v in enumerate(sf)}
    tf_index = {float(v): i for i, v in enumerate(tf)}
    for row in frame.itertuples(index=False):
        mat[tf_index[float(row.temporal_frequency_hz)], sf_index[float(row.spatial_frequency_cpd)]] = float(
            getattr(row, column)
        )
    return sf, tf, mat


def nearest_indices(values: np.ndarray, targets: tuple[float, ...]) -> list[int]:
    return [int(np.argmin(np.abs(values - target))) for target in targets]


def k2_reference(sf: np.ndarray, values: np.ndarray, anchor_count: int = 2) -> np.ndarray:
    valid = np.isfinite(sf) & np.isfinite(values) & (sf > 0) & (values > 0)
    idx = np.where(valid)[0][: max(int(anchor_count), 1)]
    if idx.size == 0:
        return np.full_like(values, np.nan, dtype=np.float64)
    anchor = float(np.nanmedian(values[idx] / np.square(sf[idx])))
    return anchor * np.square(sf)


def critical_frequency_3db(sf: np.ndarray, values: np.ndarray) -> float:
    ref = k2_reference(sf, values)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_db = db(values / ref)
    candidates = np.where(np.isfinite(delta_db) & (delta_db <= -3.0))[0]
    return float(sf[int(candidates[0])]) if candidates.size else float("nan")


def run(summary_csv: Path, out_dir: Path, condition: str) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    radial = pd.read_csv(summary_csv)
    sub = radial[radial["condition"].eq(condition)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for condition {condition!r} in {summary_csv}")

    sf, tf, movie_power = matrix_from_column(sub, "modulation_power_mean")
    _sf_q, _tf_q, transfer_q = matrix_from_column(sub, "motion_q_mean")
    sf_edges = spatial_edges_from_centers(sf)
    tf_edges = temporal_edges(tf)

    q_db = db(transfer_q)
    movie_db = db(movie_power)
    q_vmin, q_vmax = np.nanpercentile(q_db, [5, 98])
    movie_vmin, movie_vmax = np.nanpercentile(movie_db, [5, 98])

    fig = plt.figure(figsize=(12.4, 8.0))
    gs = fig.add_gridspec(2, 2, width_ratios=(1.03, 1.0), hspace=0.38, wspace=0.34)
    ax_movie = fig.add_subplot(gs[0, 0])
    ax_q = fig.add_subplot(gs[0, 1])
    ax_q_sections = fig.add_subplot(gs[1, 0])
    ax_movie_sections = fig.add_subplot(gs[1, 1])

    movie_mesh = ax_movie.pcolormesh(
        sf_edges, tf_edges, movie_db, shading="auto", cmap="hot", vmin=movie_vmin, vmax=movie_vmax
    )
    ax_movie.set_title("Current heatmap: image-weighted retinal power", loc="left", fontsize=10, fontweight="bold")
    ax_movie.set_xscale("log")
    ax_movie.set_yscale("log")
    ax_movie.set_xlabel("spatial frequency (cycles/deg)")
    ax_movie.set_ylabel("temporal frequency (Hz)")
    ax_movie.set_xticks([0.5, 1, 3, 10], ["0.5", "1", "3", "10"])
    ax_movie.set_yticks([3, 10, 30, 60], ["3", "10", "30", "60"])
    cbar_movie = fig.colorbar(movie_mesh, ax=ax_movie, pad=0.02)
    cbar_movie.set_label("10 log10 P_image * Q")

    q_mesh = ax_q.pcolormesh(sf_edges, tf_edges, q_db, shading="auto", cmap="hot", vmin=q_vmin, vmax=q_vmax)
    ax_q.set_title("Rucci/Mostofi comparison: motion transfer Q", loc="left", fontsize=10, fontweight="bold")
    ax_q.set_xscale("log")
    ax_q.set_yscale("log")
    ax_q.set_xlabel("spatial frequency (cycles/deg)")
    ax_q.set_ylabel("temporal frequency (Hz)")
    ax_q.set_xticks([0.5, 1, 3, 10], ["0.5", "1", "3", "10"])
    ax_q.set_yticks([3, 10, 30, 60], ["3", "10", "30", "60"])
    cbar_q = fig.colorbar(q_mesh, ax=ax_q, pad=0.02)
    cbar_q.set_label("10 log10 Q")

    target_tfs = (5.0, 9.0, 16.0, 30.0)
    colors = ("#3f4ab8", "#2d7b39", "#c51f2f", "#65b7bd")
    critical_rows = []
    for idx, color in zip(nearest_indices(tf, target_tfs), colors, strict=True):
        q_vals = transfer_q[idx, :]
        movie_vals = movie_power[idx, :]
        q_ref = k2_reference(sf, q_vals)
        crit = critical_frequency_3db(sf, q_vals)
        critical_rows.append(
            {
                "requested_temporal_frequency_hz": target_tfs[len(critical_rows)],
                "nearest_temporal_frequency_hz": float(tf[idx]),
                "critical_spatial_frequency_3db_cpd": crit,
            }
        )
        ax_q_sections.plot(sf, db(q_vals), color=color, linewidth=2.1, label=f"{tf[idx]:.0f} Hz")
        ax_q_sections.plot(sf, db(q_ref), color="black", linestyle="--", linewidth=1.1, alpha=0.8)
        if np.isfinite(crit):
            ax_q_sections.scatter([crit], [db(q_vals[np.argmin(np.abs(sf - crit))])], color=color, s=26, zorder=5)
        ax_movie_sections.plot(sf, db(movie_vals), color=color, linewidth=2.1, label=f"{tf[idx]:.0f} Hz")

    ax_q_sections.set_title("Motion transfer sections: expected k^2 rise", loc="left", fontsize=10, fontweight="bold")
    ax_q_sections.set_xscale("log")
    ax_q_sections.set_xlabel("spatial frequency (cycles/deg)")
    ax_q_sections.set_ylabel("10 log10 Q")
    ax_q_sections.legend(frameon=False, fontsize=8)

    ax_movie_sections.set_title("Image-weighted sections: natural-image spectrum folded in", loc="left", fontsize=10, fontweight="bold")
    ax_movie_sections.set_xscale("log")
    ax_movie_sections.set_xlabel("spatial frequency (cycles/deg)")
    ax_movie_sections.set_ylabel("10 log10 P_image * Q")
    ax_movie_sections.legend(frameon=False, fontsize=8)

    for ax in (ax_movie, ax_q, ax_q_sections, ax_movie_sections):
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "SSI-v3 SF-TF heatmap verification: movie power vs motion-only transfer",
        x=0.02,
        y=0.99,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    out_png = out_dir / f"{condition}_sftf_transfer_verification.png"
    out_pdf = out_dir / f"{condition}_sftf_transfer_verification.pdf"
    out_csv = out_dir / f"{condition}_sftf_transfer_critical_frequencies.csv"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(critical_rows).to_csv(out_csv, index=False)
    return {"png": out_png, "pdf": out_pdf, "csv": out_csv}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--condition", default="all_real_fem")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run(args.summary_csv, args.out_dir, args.condition)
    print(f"Wrote {paths['png']}")
    print(f"Wrote {paths['csv']}")


if __name__ == "__main__":
    main()
