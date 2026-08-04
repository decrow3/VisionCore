#!/usr/bin/env python3
"""Plot fitted dense SF/TF tuning ellipses colored by previous SF group labels."""

from __future__ import annotations

import argparse
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
from matplotlib.patches import Ellipse

from plot_eye_movement_power_spectrum_shift import ROOT


DEFAULT_DENSE_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_dense_sf_tf_speed_pref_groups_v1"
)
DEFAULT_FITS_CSV = DEFAULT_DENSE_DIR / "cycle_valid_dense_sf_tf_fit_unit_summary.csv"
DEFAULT_OUT_DIR = ROOT / "outputs/fig/ssi_figure_v2/panels/dense_sftf_fit_ellipses_by_sf_group"

FINAL_GROUP_ORDER = ("low_mid_sf", "high_sf")
FINAL_GROUP_LABELS = {"low_mid_sf": "low+middle SF", "high_sf": "high SF"}
FINAL_GROUP_COLORS = {"low_mid_sf": "#0072B2", "high_sf": "#D55E00"}
ORIGINAL_GROUP_ORDER = ("low_sf", "middle_sf", "high_sf")
ORIGINAL_GROUP_LABELS = {"low_sf": "low SF", "middle_sf": "middle SF", "high_sf": "high SF"}
ORIGINAL_GROUP_COLORS = {"low_sf": "#0072B2", "middle_sf": "#559F76", "high_sf": "#D55E00"}

SF_TICKS = np.asarray([0.4, 0.8, 1.6, 3.2, 6.4, 12.8], dtype=float)
TF_TICKS = np.asarray([0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2], dtype=float)
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


def setup_log2_axes(ax: plt.Axes) -> None:
    ax.set_xlim(float(np.log2(0.36)), float(np.log2(17.3)))
    ax.set_ylim(float(np.log2(0.36)), float(np.log2(56.5)))
    ax.set_xticks(np.log2(SF_TICKS), [f"{v:g}" for v in SF_TICKS])
    ax.set_yticks(np.log2(TF_TICKS), [f"{v:g}" for v in TF_TICKS])
    ax.axvline(float(np.log2(ONE_CYCLE_CPD)), color="0.55", linestyle="-.", linewidth=0.8)
    ax.grid(True, color="0.91", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("spatial frequency (cycles/deg)")
    ax.set_ylabel("temporal frequency (Hz)")


def _finite_fit_mask(frame: pd.DataFrame) -> pd.Series:
    cols = ["fit_pref_log2_sf", "fit_pref_log2_tf", "fit_fwhm_sf_octaves", "fit_fwhm_tf_octaves"]
    mask = frame["fit_ok"].astype(bool)
    for col in cols:
        mask &= np.isfinite(pd.to_numeric(frame[col], errors="coerce"))
    return mask


def _legend_handles(
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
            linewidth=1.6,
            label=f"{labels[group_id]} (n={int(counts.get(group_id, 0))})",
        )
        for group_id in group_order
    ]


def plot_fit_ellipses(
    ax: plt.Axes,
    fits: pd.DataFrame,
    *,
    group_col: str,
    group_order: tuple[str, ...],
    labels: dict[str, str],
    colors: dict[str, str],
    interior_only: bool,
    title: str,
) -> None:
    use = fits[_finite_fit_mask(fits)].copy()
    use["is_edge"] = use["fit_edge_sf"].astype(bool) | use["fit_edge_tf"].astype(bool)
    if interior_only:
        use = use[~use["is_edge"]].copy()
    counts = use.groupby(group_col)["unit_index"].nunique().to_dict()
    for _, row in use.iterrows():
        group_id = str(row[group_col])
        if group_id not in colors:
            continue
        edge = bool(row["is_edge"])
        ell = Ellipse(
            (float(row["fit_pref_log2_sf"]), float(row["fit_pref_log2_tf"])),
            width=float(row["fit_fwhm_sf_octaves"]),
            height=float(row["fit_fwhm_tf_octaves"]),
            angle=0.0,
            fill=False,
            edgecolor=colors[group_id],
            linewidth=0.75 if edge else 0.95,
            linestyle="--" if edge else "-",
            alpha=0.16 if edge else 0.28,
        )
        ax.add_patch(ell)
        ax.scatter(
            [float(row["fit_pref_log2_sf"])],
            [float(row["fit_pref_log2_tf"])],
            color=colors[group_id],
            s=9 if edge else 13,
            alpha=0.45 if edge else 0.7,
            linewidth=0,
            zorder=3,
        )

    for group_id in group_order:
        sub = use[use[group_col].eq(group_id)]
        if sub.empty:
            continue
        med = sub[["fit_pref_log2_sf", "fit_pref_log2_tf", "fit_fwhm_sf_octaves", "fit_fwhm_tf_octaves"]].median(
            numeric_only=True
        )
        ell = Ellipse(
            (float(med["fit_pref_log2_sf"]), float(med["fit_pref_log2_tf"])),
            width=float(med["fit_fwhm_sf_octaves"]),
            height=float(med["fit_fwhm_tf_octaves"]),
            angle=0.0,
            fill=False,
            edgecolor=colors[group_id],
            linewidth=2.2,
            alpha=0.95,
            zorder=6,
        )
        ax.add_patch(ell)
        ax.scatter(
            [float(med["fit_pref_log2_sf"])],
            [float(med["fit_pref_log2_tf"])],
            marker="D",
            color=colors[group_id],
            edgecolor="black",
            linewidth=0.5,
            s=40,
            zorder=7,
        )

    setup_log2_axes(ax)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(
        handles=_legend_handles(group_order=group_order, labels=labels, colors=colors, counts=counts),
        frameon=False,
        fontsize=7,
        loc="lower left",
    )


def plot_centers(
    ax: plt.Axes,
    fits: pd.DataFrame,
    *,
    group_col: str,
    group_order: tuple[str, ...],
    labels: dict[str, str],
    colors: dict[str, str],
    title: str,
) -> None:
    use = fits[_finite_fit_mask(fits)].copy()
    use["is_edge"] = use["fit_edge_sf"].astype(bool) | use["fit_edge_tf"].astype(bool)
    for group_id in group_order:
        sub = use[use[group_col].eq(group_id)].copy()
        if sub.empty:
            continue
        ax.scatter(
            sub.loc[~sub["is_edge"], "fit_pref_log2_sf"],
            sub.loc[~sub["is_edge"], "fit_pref_log2_tf"],
            color=colors[group_id],
            s=24,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
            label=labels[group_id],
        )
        ax.scatter(
            sub.loc[sub["is_edge"], "fit_pref_log2_sf"],
            sub.loc[sub["is_edge"], "fit_pref_log2_tf"],
            facecolor="none",
            edgecolor=colors[group_id],
            s=34,
            alpha=0.8,
            linewidth=1.0,
        )
    setup_log2_axes(ax)
    ax.set_title(title, loc="left", fontweight="bold")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.45", markeredgecolor="white", markersize=5, label="interior fit"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="0.45", markersize=6, label="edge fit"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="lower left")


def summarize(fits: pd.DataFrame, group_col: str, order: tuple[str, ...], labels: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    use = fits[_finite_fit_mask(fits)].copy()
    use["is_edge"] = use["fit_edge_sf"].astype(bool) | use["fit_edge_tf"].astype(bool)
    for group_id in order:
        sub = use[use[group_col].eq(group_id)].copy()
        for subset_id, subset in (("all_fit", sub), ("interior_fit", sub[~sub["is_edge"]])):
            rows.append(
                {
                    "grouping": group_col,
                    "group": group_id,
                    "group_label": labels[group_id],
                    "subset": subset_id,
                    "n_units": int(subset["unit_index"].nunique()),
                    "fit_pref_sf_median_cpd": float(np.power(2.0, subset["fit_pref_log2_sf"].median()))
                    if not subset.empty
                    else float("nan"),
                    "fit_pref_tf_median_hz": float(np.power(2.0, subset["fit_pref_log2_tf"].median()))
                    if not subset.empty
                    else float("nan"),
                    "fit_fwhm_sf_median_octaves": float(subset["fit_fwhm_sf_octaves"].median())
                    if not subset.empty
                    else float("nan"),
                    "fit_fwhm_tf_median_octaves": float(subset["fit_fwhm_tf_octaves"].median())
                    if not subset.empty
                    else float("nan"),
                    "fit_r2_median": float(subset["fit_r2"].median()) if not subset.empty else float("nan"),
                    "edge_fraction": float(sub["is_edge"].mean()) if not sub.empty else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def plot_figure(fits: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.0), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.075, top=0.9, hspace=0.34, wspace=0.24)
    plot_fit_ellipses(
        axes[0, 0],
        fits,
        group_col="final_sf_group",
        group_order=FINAL_GROUP_ORDER,
        labels=FINAL_GROUP_LABELS,
        colors=FINAL_GROUP_COLORS,
        interior_only=False,
        title="A. Final split: fitted FWHM ellipse per unit",
    )
    plot_fit_ellipses(
        axes[0, 1],
        fits,
        group_col="final_sf_group",
        group_order=FINAL_GROUP_ORDER,
        labels=FINAL_GROUP_LABELS,
        colors=FINAL_GROUP_COLORS,
        interior_only=True,
        title="B. Final split: interior fits only",
    )
    plot_fit_ellipses(
        axes[1, 0],
        fits,
        group_col="sf_group",
        group_order=ORIGINAL_GROUP_ORDER,
        labels=ORIGINAL_GROUP_LABELS,
        colors=ORIGINAL_GROUP_COLORS,
        interior_only=False,
        title="C. Original split: fitted FWHM ellipse per unit",
    )
    plot_centers(
        axes[1, 1],
        fits,
        group_col="final_sf_group",
        group_order=FINAL_GROUP_ORDER,
        labels=FINAL_GROUP_LABELS,
        colors=FINAL_GROUP_COLORS,
        title="D. Final split: fitted centers only",
    )
    fig.suptitle(
        "Dense SF/TF 2D fit contours colored by earlier SF-tuning labels",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.94,
        "Each ellipse is the FWHM contour of the axis-aligned 2D log-Gaussian fit; dashed outlines indicate SF or TF edge fits.",
        ha="left",
        fontsize=8.5,
        color="0.35",
    )
    png = out_dir / "dense_sftf_fit_ellipses_by_sf_group.png"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "dense_sftf_fit_ellipses_by_sf_group.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "dense_sftf_fit_ellipses_by_sf_group.svg", bbox_inches="tight")
    plt.close(fig)
    return png


def write_readme(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, png: Path) -> None:
    final = summary[(summary["grouping"].eq("final_sf_group")) & (summary["subset"].eq("all_fit"))]
    lines = [
        "# Dense SF/TF Fit Ellipses By SF Group",
        "",
        f"- Figure: `{_relative(png)}`",
        f"- Dense fit source: `{_relative(Path(args.fits_csv))}`",
        "",
        "Ellipses are not contours of the sampled response grid. They are reconstructed from the axis-aligned 2D log-Gaussian fit summary: center = fitted preferred log2 SF/TF, width/height = fitted FWHM in octaves.",
        "",
        "## Final Split Summary",
        "",
    ]
    for row in final.itertuples(index=False):
        lines.append(
            f"- {row.group_label}: n={int(row.n_units)}, median center "
            f"{float(row.fit_pref_sf_median_cpd):.3g} cpd / {float(row.fit_pref_tf_median_hz):.3g} Hz; "
            f"median FWHM {float(row.fit_fwhm_sf_median_octaves):.2g} x {float(row.fit_fwhm_tf_median_octaves):.2g} octaves; "
            f"edge fraction {float(row.edge_fraction):.2f}."
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Path]:
    configure_matplotlib()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fits = add_final_group(pd.read_csv(args.fits_csv))
    png = plot_figure(fits, out_dir)
    summary = pd.concat(
        [
            summarize(fits, "final_sf_group", FINAL_GROUP_ORDER, FINAL_GROUP_LABELS),
            summarize(fits, "sf_group", ORIGINAL_GROUP_ORDER, ORIGINAL_GROUP_LABELS),
        ],
        ignore_index=True,
    )
    summary_csv = out_dir / "dense_sftf_fit_ellipses_group_summary.csv"
    summary.to_csv(summary_csv, index=False)
    write_readme(out_dir, args, summary, png)
    return {"png": png, "summary_csv": summary_csv, "readme": out_dir / "README.md"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits-csv", type=Path, default=DEFAULT_FITS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    paths = run(parse_args())
    print(f"Wrote {paths['png']}")
    print(f"Wrote {paths['summary_csv']}")
    print(f"Wrote {paths['readme']}")


if __name__ == "__main__":
    main()
