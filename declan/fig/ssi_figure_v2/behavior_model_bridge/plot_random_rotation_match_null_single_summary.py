#!/usr/bin/env python3
"""Single-panel summary of the random-rotation trace-contour matching null."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig.ssi_figure_v2.behavior_model_bridge import run_behavior_model_bridge as bridge


SUMMARY_CSV = bridge.OUT_DIR / "behavior_model_bridge_random_rotation_match_null_summary.csv"
OUT_STEM = "behavior_model_bridge_random_rotation_match_null_main_point"

POPULATION_ORDER = (
    "high_sf_aligned",
    "high_sf_oblique",
    "high_sf_orthogonal",
    "high_sf_all",
    "low_sf_all",
)

POPULATION_LABELS = {
    "high_sf_aligned": "Aligned high-SF",
    "high_sf_oblique": "Oblique high-SF",
    "high_sf_orthogonal": "Orthogonal high-SF",
    "high_sf_all": "All high-SF",
    "low_sf_all": "All low-SF",
}

METRICS = (
    ("component_rms", "RMS excursion", "#2f6f9f", "o", -0.13),
    ("component_range", "Projected range", "#2d8a66", "s", 0.13),
)


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build(
    *,
    summary_csv: Path = SUMMARY_CSV,
    out_dir: Path = bridge.OUT_DIR,
    subset_key: str = "coh_ge_0p2",
) -> dict[str, Path]:
    bridge.configure_matplotlib()
    df = pd.read_csv(summary_csv)
    frame = df[
        df["subset_key"].astype(str).eq(subset_key)
        & df["score_type"].astype(str).eq("component_mean_marginal")
        & df["population_key"].astype(str).isin(POPULATION_ORDER)
        & df["metric_family"].astype(str).isin([metric[0] for metric in METRICS])
    ].copy()
    if frame.empty:
        raise RuntimeError(f"No rows found for subset_key={subset_key!r} in {summary_csv}")

    y_base = np.arange(len(POPULATION_ORDER), dtype=float)[::-1]
    y_lookup = {population: y for population, y in zip(POPULATION_ORDER, y_base, strict=True)}
    fig, ax = plt.subplots(figsize=(7.2, 4.15), constrained_layout=True)

    ax.axvspan(-0.14, 0.0, color="#b9c1ca", alpha=0.14, lw=0)
    ax.axvspan(0.0, 0.18, color="#6fb58b", alpha=0.10, lw=0)
    ax.axhspan(
        y_lookup["high_sf_aligned"] - 0.42,
        y_lookup["high_sf_aligned"] + 0.42,
        color="#f3df8e",
        alpha=0.22,
        lw=0,
        zorder=0,
    )
    ax.axvline(0.0, color="#333333", lw=1.0, ls=":")

    for metric_key, label, color, marker, offset in METRICS:
        sub = frame[frame["metric_family"].astype(str).eq(metric_key)].set_index("population_key")
        xs: list[float] = []
        xlo: list[float] = []
        xhi: list[float] = []
        ys: list[float] = []
        for population in POPULATION_ORDER:
            row = sub.loc[population]
            x = float(row["observed_minus_rotated_session_mean"])
            lo = float(row["observed_minus_rotated_ci95_low"])
            hi = float(row["observed_minus_rotated_ci95_high"])
            xs.append(x)
            xlo.append(x - lo)
            xhi.append(hi - x)
            ys.append(y_lookup[population] + offset)
        ax.errorbar(
            xs,
            ys,
            xerr=np.vstack([xlo, xhi]),
            fmt=marker,
            ms=6.2,
            lw=1.6,
            capsize=3.0,
            color=color,
            markerfacecolor="white",
            markeredgewidth=1.6,
            label=label,
            zorder=3,
        )

    ax.set_yticks([y_lookup[population] for population in POPULATION_ORDER])
    ax.set_yticklabels([POPULATION_LABELS[population] for population in POPULATION_ORDER])
    ax.set_xlabel("Observed trace-contour matching benefit over random rotation\n(pp SSI prediction)")
    ax.set_xlim(-0.13, 0.17)
    ax.set_ylim(-0.75, len(POPULATION_ORDER) - 0.25)
    ax.grid(axis="x", color=bridge.GRID, lw=0.8)
    _clean_axis(ax)

    ax.text(
        -0.128,
        len(POPULATION_ORDER) - 0.35,
        "random rotation better",
        ha="left",
        va="top",
        fontsize=7.2,
        color="#6B6F75",
    )
    ax.text(
        0.168,
        len(POPULATION_ORDER) - 0.35,
        "observed matching better",
        ha="right",
        va="top",
        fontsize=7.2,
        color="#42734f",
    )
    ax.text(
        0.162,
        y_lookup["high_sf_aligned"],
        "largest high-SF match advantage",
        ha="right",
        va="center",
        fontsize=7.4,
        color="#7a6517",
    )

    title = "Real drift-contour matching predicts higher SSI for high-SF populations"
    subtitle = "0.325 s BackImage snippets; contour coherence >=0.2; component-mean marginal model score"
    fig.suptitle(title, x=0.02, y=1.03, ha="left", fontsize=12.6, fontweight="bold")
    ax.set_title(subtitle, loc="left", fontsize=8.4, color="#4b5563", pad=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{OUT_STEM}.png"
    pdf = out_dir / f"{OUT_STEM}.pdf"
    svg = out_dir / f"{OUT_STEM}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {"png": png, "pdf": pdf, "svg": svg}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--out-dir", type=Path, default=bridge.OUT_DIR)
    parser.add_argument("--subset-key", default="coh_ge_0p2")
    args = parser.parse_args()
    paths = build(summary_csv=args.summary_csv, out_dir=args.out_dir, subset_key=str(args.subset_key))
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
