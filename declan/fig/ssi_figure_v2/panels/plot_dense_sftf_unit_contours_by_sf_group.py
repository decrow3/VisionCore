#!/usr/bin/env python3
"""Plot dense SF/TF per-unit contours colored by previous SF group labels."""

from __future__ import annotations

import argparse
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
from matplotlib.lines import Line2D

from plot_eye_movement_power_spectrum_shift import ROOT


DEFAULT_DENSE_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_dense_sf_tf_speed_pref_groups_v1"
)
DEFAULT_POINTS_CSV = DEFAULT_DENSE_DIR / "cycle_valid_dense_sf_tf_points.csv"
DEFAULT_FITS_CSV = DEFAULT_DENSE_DIR / "cycle_valid_dense_sf_tf_fit_unit_summary.csv"
DEFAULT_OUT_DIR = ROOT / "outputs/fig/ssi_figure_v2/panels/dense_sftf_unit_contours_by_sf_group"

FINAL_GROUP_ORDER = ("low_mid_sf", "high_sf")
FINAL_GROUP_LABELS = {"low_mid_sf": "low+middle SF", "high_sf": "high SF"}
FINAL_GROUP_COLORS = {"low_mid_sf": "#0072B2", "high_sf": "#D55E00"}
ORIGINAL_GROUP_ORDER = ("low_sf", "middle_sf", "high_sf")
ORIGINAL_GROUP_LABELS = {"low_sf": "low SF", "middle_sf": "middle SF", "high_sf": "high SF"}
ORIGINAL_GROUP_COLORS = {"low_sf": "#0072B2", "middle_sf": "#559F76", "high_sf": "#D55E00"}
ONE_CYCLE_CPD = 0.37133431851485155


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


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def add_final_group(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["final_sf_group"] = np.where(out["sf_group"].isin(["low_sf", "middle_sf"]), "low_mid_sf", "high_sf")
    return out


def normalize_surface(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr * np.nan
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def surface_matrix(sub: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sf = np.asarray(sorted(sub["spatial_cpd"].unique()), dtype=np.float64)
    tf = np.asarray(sorted(sub["temporal_hz"].unique()), dtype=np.float64)
    grid = sub.pivot_table(index="temporal_hz", columns="spatial_cpd", values=value_col, aggfunc="mean")
    grid = grid.reindex(index=tf, columns=sf)
    return sf, tf, grid.to_numpy(dtype=np.float64)


def setup_sftf_axes(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.36, 17.3)
    ax.set_ylim(0.36, 56.5)
    ax.set_xticks([0.4, 0.8, 1.6, 3.2, 6.4, 12.8], ["0.4", "0.8", "1.6", "3.2", "6.4", "12.8"])
    ax.set_yticks([0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2], ["0.4", "0.8", "1.6", "3.2", "6.4", "12.8", "25.6", "51.2"])
    ax.axvline(ONE_CYCLE_CPD, color="0.55", linestyle="-.", linewidth=0.8)
    ax.grid(True, color="0.91", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def legend_handles(
    *,
    group_order: tuple[str, ...],
    labels: dict[str, str],
    colors: dict[str, str],
    counts: dict[str, int],
) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=colors[group_id],
            linewidth=1.7,
            label=f"{labels[group_id]} (n={int(counts.get(group_id, 0))})",
        )
        for group_id in group_order
    ]


def plot_unit_contours(
    ax: plt.Axes,
    points: pd.DataFrame,
    *,
    group_col: str,
    group_order: tuple[str, ...],
    labels: dict[str, str],
    colors: dict[str, str],
    contour_level: float,
    title: str,
) -> None:
    counts = points.groupby(group_col)["unit_index"].nunique().to_dict()
    for unit_index, sub in points.groupby("unit_index", sort=True):
        group_id = str(sub[group_col].iloc[0])
        if group_id not in colors:
            continue
        sf, tf, response = surface_matrix(sub, "response_amp_rms_mean")
        norm = normalize_surface(response)
        if not np.isfinite(norm).any() or np.nanmax(norm) < float(contour_level):
            continue
        try:
            ax.contour(
                sf,
                tf,
                norm,
                levels=[float(contour_level)],
                colors=[colors[group_id]],
                linewidths=0.78,
                alpha=0.34,
            )
        except ValueError:
            continue
        peak_idx = np.unravel_index(int(np.nanargmax(response)), response.shape)
        ax.scatter(
            [float(sf[peak_idx[1]])],
            [float(tf[peak_idx[0]])],
            color=colors[group_id],
            s=8,
            alpha=0.36,
            linewidth=0,
        )
    setup_sftf_axes(ax)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("spatial frequency (cycles/deg)")
    ax.set_ylabel("temporal frequency (Hz)")
    ax.legend(handles=legend_handles(group_order=group_order, labels=labels, colors=colors, counts=counts), frameon=False, fontsize=7, loc="lower left")


def plot_peak_scatter(
    ax: plt.Axes,
    fits: pd.DataFrame,
    *,
    group_col: str,
    group_order: tuple[str, ...],
    labels: dict[str, str],
    colors: dict[str, str],
    title: str,
) -> None:
    for group_id in group_order:
        sub = fits[fits[group_col].eq(group_id)].copy()
        if sub.empty:
            continue
        edge = sub["fit_edge_sf"].astype(bool) | sub["fit_edge_tf"].astype(bool)
        ax.scatter(
            sub.loc[~edge, "observed_peak_sf_cpd"],
            sub.loc[~edge, "observed_peak_tf_hz"],
            color=colors[group_id],
            s=22,
            alpha=0.58,
            edgecolor="white",
            linewidth=0.35,
            label=f"{labels[group_id]} observed peak",
        )
        ax.scatter(
            sub.loc[edge, "observed_peak_sf_cpd"],
            sub.loc[edge, "observed_peak_tf_hz"],
            facecolor="none",
            edgecolor=colors[group_id],
            s=34,
            alpha=0.75,
            linewidth=1.0,
        )
        ok = sub["fit_ok"].astype(bool)
        ax.scatter(
            sub.loc[ok, "fit_pref_sf_cpd"],
            sub.loc[ok, "fit_pref_tf_hz"],
            marker="x",
            color=colors[group_id],
            s=28,
            alpha=0.9,
            linewidth=1.0,
        )
        med = sub.loc[ok, ["fit_pref_sf_cpd", "fit_pref_tf_hz"]].median(numeric_only=True)
        if np.isfinite(med).all():
            ax.scatter(
                [float(med["fit_pref_sf_cpd"])],
                [float(med["fit_pref_tf_hz"])],
                marker="D",
                color=colors[group_id],
                s=44,
                edgecolor="black",
                linewidth=0.55,
                zorder=10,
            )
    setup_sftf_axes(ax)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("spatial frequency (cycles/deg)")
    ax.set_ylabel("temporal frequency (Hz)")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.45", markeredgecolor="white", markersize=5, label="observed peak"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="0.45", markersize=6, label="edge fit"),
        Line2D([0], [0], marker="x", color="0.25", linewidth=0, markersize=5, label="2D fit preference"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="0.45", markeredgecolor="black", markersize=5, label="group median fit"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6.8, loc="lower left")


def summarize_groups(fits: pd.DataFrame, group_col: str, order: tuple[str, ...], labels: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_id in order:
        sub = fits[fits[group_col].eq(group_id)].copy()
        ok = sub[sub["fit_ok"].astype(bool)]
        interior = ok[~(ok["fit_edge_sf"].astype(bool) | ok["fit_edge_tf"].astype(bool))]
        for subset_id, subset in (("all", sub), ("fit_ok", ok), ("interior_fit", interior)):
            rows.append(
                {
                    "grouping": group_col,
                    "group": group_id,
                    "group_label": labels[group_id],
                    "subset": subset_id,
                    "n_units": int(subset["unit_index"].nunique()),
                    "observed_peak_sf_median_cpd": float(pd.to_numeric(subset["observed_peak_sf_cpd"], errors="coerce").median()) if not subset.empty else float("nan"),
                    "observed_peak_tf_median_hz": float(pd.to_numeric(subset["observed_peak_tf_hz"], errors="coerce").median()) if not subset.empty else float("nan"),
                    "fit_pref_sf_median_cpd": float(pd.to_numeric(subset["fit_pref_sf_cpd"], errors="coerce").median()) if not subset.empty else float("nan"),
                    "fit_pref_tf_median_hz": float(pd.to_numeric(subset["fit_pref_tf_hz"], errors="coerce").median()) if not subset.empty else float("nan"),
                    "fit_edge_fraction": float(
                        np.mean(subset["fit_edge_sf"].astype(bool).to_numpy() | subset["fit_edge_tf"].astype(bool).to_numpy())
                    )
                    if not subset.empty
                    else float("nan"),
                    "fit_r2_median": float(pd.to_numeric(subset["fit_r2"], errors="coerce").median()) if not subset.empty else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def plot_figure(points: pd.DataFrame, fits: pd.DataFrame, out_dir: Path, *, contour_level: float) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.0), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.075, top=0.9, hspace=0.34, wspace=0.24)
    plot_unit_contours(
        axes[0, 0],
        points,
        group_col="final_sf_group",
        group_order=FINAL_GROUP_ORDER,
        labels=FINAL_GROUP_LABELS,
        colors=FINAL_GROUP_COLORS,
        contour_level=float(contour_level),
        title=f"A. Final split: per-unit {float(contour_level):.2g} response contours",
    )
    plot_peak_scatter(
        axes[0, 1],
        fits,
        group_col="final_sf_group",
        group_order=FINAL_GROUP_ORDER,
        labels=FINAL_GROUP_LABELS,
        colors=FINAL_GROUP_COLORS,
        title="B. Final split: peaks and 2D fit preferences",
    )
    plot_unit_contours(
        axes[1, 0],
        points,
        group_col="sf_group",
        group_order=ORIGINAL_GROUP_ORDER,
        labels=ORIGINAL_GROUP_LABELS,
        colors=ORIGINAL_GROUP_COLORS,
        contour_level=float(contour_level),
        title=f"C. Original split: per-unit {float(contour_level):.2g} response contours",
    )
    plot_peak_scatter(
        axes[1, 1],
        fits,
        group_col="sf_group",
        group_order=ORIGINAL_GROUP_ORDER,
        labels=ORIGINAL_GROUP_LABELS,
        colors=ORIGINAL_GROUP_COLORS,
        title="D. Original split: peaks and 2D fit preferences",
    )
    fig.suptitle(
        "Dense SF/TF unit contours colored by earlier SF-tuning labels",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.94,
        "Contours are within-unit normalized dense response amplitude; labels come from the earlier coarse dynamic marginal SF fit.",
        ha="left",
        fontsize=8.5,
        color="0.35",
    )
    png = out_dir / "dense_sftf_unit_contours_by_sf_group.png"
    pdf = out_dir / "dense_sftf_unit_contours_by_sf_group.pdf"
    svg = out_dir / "dense_sftf_unit_contours_by_sf_group.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png


def write_readme(out_dir: Path, args: argparse.Namespace, summaries: pd.DataFrame, png: Path) -> None:
    final = summaries[(summaries["grouping"].eq("final_sf_group")) & (summaries["subset"].eq("fit_ok"))]
    lines = [
        "# Dense SF/TF Unit Contours By SF Group",
        "",
        f"- Figure: `{_relative(png)}`",
        f"- Dense points source: `{_relative(Path(args.points_csv))}`",
        f"- Dense fit source: `{_relative(Path(args.fits_csv))}`",
        "",
        "Contours are one contour per unit, drawn at a fixed level after min-max normalizing that unit's `response_amp_rms_mean` over the dense cycle-valid SF/TF grid.",
        f"The contour level is `{float(args.contour_level):.3g}`.",
        "",
        "The color labels are not re-fit from the dense surface. They are inherited from the earlier coarse dynamic marginal SF fit, then optionally collapsed as low+middle versus high.",
        "",
        "## Fit-Preference Summary",
        "",
    ]
    for row in final.itertuples(index=False):
        lines.append(
            f"- {row.group_label}: n={int(row.n_units)}, median dense fit preference "
            f"{float(row.fit_pref_sf_median_cpd):.3g} cpd / {float(row.fit_pref_tf_median_hz):.3g} Hz; "
            f"edge-fit fraction {float(row.fit_edge_fraction):.2f}."
        )
    lines.extend(
        [
            "",
            "## Interpretation Caution",
            "",
            "The dense cycle-valid grid starts at 0.4 cpd, while the original low-SF labels were often driven by sub-cycle coarse probe points at 0.0125 and 0.05 cpd. A strong overlap between low+middle and high contours in this plot therefore means the old labels do not map cleanly onto compact dense SF/TF tuning islands.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Path]:
    configure_matplotlib()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = add_final_group(pd.read_csv(args.points_csv))
    fits = add_final_group(pd.read_csv(args.fits_csv))
    png = plot_figure(points, fits, out_dir, contour_level=float(args.contour_level))
    summary = pd.concat(
        [
            summarize_groups(fits, "final_sf_group", FINAL_GROUP_ORDER, FINAL_GROUP_LABELS),
            summarize_groups(fits, "sf_group", ORIGINAL_GROUP_ORDER, ORIGINAL_GROUP_LABELS),
        ],
        ignore_index=True,
    )
    summary_csv = out_dir / "dense_sftf_unit_contours_group_summary.csv"
    summary.to_csv(summary_csv, index=False)
    write_readme(out_dir, args, summary, png)
    return {"png": png, "summary_csv": summary_csv, "readme": out_dir / "README.md"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-csv", type=Path, default=DEFAULT_POINTS_CSV)
    parser.add_argument("--fits-csv", type=Path, default=DEFAULT_FITS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--contour-level", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    paths = run(parse_args())
    print(f"Wrote {paths['png']}")
    print(f"Wrote {paths['summary_csv']}")
    print(f"Wrote {paths['readme']}")


if __name__ == "__main__":
    main()
