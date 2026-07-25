#!/usr/bin/env python3
"""Panel H: unwrapped FEM position-spread profile by local edge coherence.

This regenerates a compact, wider-bin version of
``b_position_spread_unwrapped_overlay_by_edge_coherence_zoomed`` from the saved
profile table produced by the BackImage contour-motion component analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels"
PROFILE_CSV = (
    ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_contour_motion_component_plots_v1"
    / "b_position_spread_unwrapped_profiles_by_edge_coherence.csv"
)

WIDE_COHERENCE_BANDS = (
    (0.0, 0.2, "0-0.2"),
    (0.2, 0.5, "0.2-0.5"),
    (0.5, 0.8, "0.5-0.8"),
    (0.8, 1.0, "0.8-1"),
)
COLORS = ("#9aa5b1", "#6c8fb5", "#2c7fb8", "#0b4f83")
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


def _parse_bin_edges(label: str) -> tuple[float, float]:
    lo, hi = str(label).split("-", maxsplit=1)
    return float(lo), float(hi)


def _wide_band_for_source_bin(bin_low: float, bin_high: float) -> tuple[int, float, float, str] | None:
    eps = 1e-9
    for order, (lo, hi, label) in enumerate(WIDE_COHERENCE_BANDS):
        if bin_low >= lo - eps and bin_high <= hi + eps:
            return order, lo, hi, label
    return None


def load_panel_values(profile_csv: Path = PROFILE_CSV) -> pd.DataFrame:
    if not profile_csv.exists():
        raise FileNotFoundError(profile_csv)
    values = pd.read_csv(profile_csv)
    required = ["coherence_bin", "relative_angle_deg", "rms_arcmin", "n_windows"]
    missing = [col for col in required if col not in values.columns]
    if missing:
        raise ValueError(f"Missing required columns in {profile_csv}: {missing}")
    edges = values["coherence_bin"].astype(str).map(_parse_bin_edges)
    values["source_bin_low"] = [edge[0] for edge in edges]
    values["source_bin_high"] = [edge[1] for edge in edges]
    bands = [
        _wide_band_for_source_bin(float(row.source_bin_low), float(row.source_bin_high))
        for row in values.itertuples(index=False)
    ]
    values["wide_band_order"] = [band[0] if band is not None else np.nan for band in bands]
    values["wide_bin_low"] = [band[1] if band is not None else np.nan for band in bands]
    values["wide_bin_high"] = [band[2] if band is not None else np.nan for band in bands]
    values["wide_coherence_bin"] = [band[3] if band is not None else None for band in bands]
    values["relative_angle_deg"] = pd.to_numeric(values["relative_angle_deg"], errors="coerce")
    values["rms_arcmin"] = pd.to_numeric(values["rms_arcmin"], errors="coerce")
    values["n_windows"] = pd.to_numeric(values["n_windows"], errors="coerce")
    values = values[
        values["wide_coherence_bin"].notna()
        & np.isfinite(values["relative_angle_deg"])
        & np.isfinite(values["rms_arcmin"])
        & np.isfinite(values["n_windows"])
    ].copy()
    if values.empty:
        raise ValueError(f"No selected coherence-bin values in {profile_csv}")

    rows: list[dict[str, float | int | str]] = []
    group_cols = ["wide_band_order", "wide_bin_low", "wide_bin_high", "wide_coherence_bin", "relative_angle_deg"]
    for key, sub in values.groupby(group_cols, sort=True):
        order, bin_low, bin_high, label, angle = key
        weights = sub["n_windows"].to_numpy(dtype=float)
        rms = sub["rms_arcmin"].to_numpy(dtype=float)
        valid = np.isfinite(weights) & np.isfinite(rms) & (weights > 0)
        if not np.any(valid):
            continue
        total_n = float(np.sum(weights[valid]))
        merged_rms = float(np.sqrt(np.sum(weights[valid] * rms[valid] ** 2) / total_n))
        rows.append(
            {
                "wide_band_order": int(order),
                "wide_coherence_bin": str(label),
                "wide_bin_low": float(bin_low),
                "wide_bin_high": float(bin_high),
                "relative_angle_deg": float(angle),
                "rms_arcmin": merged_rms,
                "n_windows": int(round(total_n)),
                "n_source_bins": int(sub["coherence_bin"].nunique()),
                "source_bins": ",".join(sorted(sub["coherence_bin"].astype(str).unique())),
            }
        )

    merged = pd.DataFrame(rows).sort_values(["wide_band_order", "relative_angle_deg"])
    if merged.empty:
        raise ValueError(f"No merged wider-bin values in {profile_csv}")
    return merged


def draw_panel(
    ax: plt.Axes,
    *,
    label: str = "H",
    title: str = "FEM spread follows local contours",
    values: pd.DataFrame | None = None,
) -> pd.DataFrame:
    values = load_panel_values() if values is None else values.copy()

    band_labels = [band[2] for band in WIDE_COHERENCE_BANDS]
    for bin_name, color in zip(band_labels, COLORS, strict=True):
        sub = values[values["wide_coherence_bin"].astype(str).eq(bin_name)].sort_values("relative_angle_deg")
        if sub.empty:
            continue
        n_windows = int(pd.to_numeric(sub["n_windows"], errors="coerce").dropna().iloc[0])
        label_text = f"{bin_name} (n={n_windows})"
        ax.plot(
            sub["relative_angle_deg"],
            sub["rms_arcmin"],
            color=color,
            lw=1.75,
            label=label_text,
        )

    ax.axvline(90.0, color="#7d858c", lw=0.75, ls=":")
    ax.set_xlim(0.0, 180.0)
    finite_y = values["rms_arcmin"].to_numpy(dtype=float)
    finite_y = finite_y[np.isfinite(finite_y)]
    if finite_y.size:
        lo, hi = float(np.nanmin(finite_y)), float(np.nanmax(finite_y))
        pad = max(0.06, 0.12 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks([0.0, 90.0, 180.0])
    ax.set_xticklabels(["parallel", "orthogonal", "parallel"])
    ax.set_xlabel("angle from local edge")
    ax.set_ylabel("position spread RMS (arcmin)")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"{label}  {title}", loc="left", fontsize=9.0, fontweight="bold", pad=4, color=INK)
    ax.legend(
        frameon=False,
        fontsize=5.7,
        loc="upper left",
        title="edge coherence",
        title_fontsize=5.9,
        handlelength=1.7,
        borderaxespad=0.2,
    )
    _clean_axis(ax)
    return values


def build_panel(out_dir: Path = OUT_DIR) -> dict[str, Path]:
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    values = load_panel_values()
    values.to_csv(out_dir / "panel_h_unwrapped_edge_coherence_values.csv", index=False)

    fig, ax = plt.subplots(figsize=(2.35, 2.25), constrained_layout=True)
    draw_panel(ax, values=values)
    paths = {
        "png": out_dir / "panel_h_unwrapped_edge_coherence.png",
        "pdf": out_dir / "panel_h_unwrapped_edge_coherence.pdf",
        "svg": out_dir / "panel_h_unwrapped_edge_coherence.svg",
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
