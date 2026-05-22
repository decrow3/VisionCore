"""Render a descriptive figure for the Priority 2 causal-alignment result.

This figure is intentionally driven from saved intervention outputs rather than
recomputing any analysis. It focuses on the real-FEM condition and explains the
causal interpretation of the alignment result:

1. Removing U_FEM helps above threshold at LogMAR -0.20.
2. Removing U_FEM is neutral at hyperacuity (LogMAR -0.40).
3. The alpha drop coexists with larger C_signal eigenvalues at -0.40, supporting
   the interpretation that the signal subspace moved away from the translation
   covariance direction rather than FEM covariance becoming informative.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class InterventionPoint:
    logmar: float
    d1_acc: float
    d1_std: float
    d1_clean_acc: float
    d1_clean_std: float
    d1_delta: float
    alpha: float
    top_signal_eigvals: np.ndarray


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_points(npz_path: Path) -> list[InterventionPoint]:
    d = np.load(npz_path, allow_pickle=True)
    prefixes = sorted({k.split("_", 1)[0] for k in d.files if k.endswith("_d1_acc")})
    points: list[InterventionPoint] = []
    for prefix in prefixes:
        logmar = float(prefix[2:])
        points.append(
            InterventionPoint(
                logmar=logmar,
                d1_acc=float(np.asarray(d[f"{prefix}_d1_acc"]).ravel()[0]),
                d1_std=float(np.asarray(d[f"{prefix}_d1_std"]).ravel()[0]),
                d1_clean_acc=float(np.asarray(d[f"{prefix}_d1_clean_acc"]).ravel()[0]),
                d1_clean_std=float(np.asarray(d[f"{prefix}_d1_clean_std"]).ravel()[0]),
                d1_delta=float(np.asarray(d[f"{prefix}_d1_delta"]).ravel()[0]),
                alpha=float(np.asarray(d[f"{prefix}_alpha"]).ravel()[0]),
                top_signal_eigvals=np.asarray(d[f"{prefix}_top_signal_eigvals"], dtype=float),
            )
        )
    points.sort(key=lambda item: item.logmar)
    return points


def make_figure(points: list[InterventionPoint]):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    mpl.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )

    color_raw = "#40444b"
    color_clean = "#177e89"
    color_limit = "#d95d39"
    color_neutral = "#7a9e7e"
    color_alpha = "#7c5ab8"
    color_signal_20 = "#d95d39"
    color_signal_40 = "#177e89"
    panel_bg = "#fbfaf7"
    help_bg = "#f8ece7"
    neutral_bg = "#eaf3eb"

    points = sorted(points, key=lambda item: item.logmar, reverse=True)
    x = np.arange(len(points), dtype=float)
    labels = [f"{p.logmar:+.2f}" for p in points]
    deltas = np.array([p.d1_delta for p in points], dtype=float)
    alphas = np.array([p.alpha for p in points], dtype=float)

    fig = plt.figure(figsize=(9.6, 6.4), constrained_layout=False)
    fig.patch.set_facecolor("white")
    gs = GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        width_ratios=[1.2, 1.0],
        height_ratios=[1.05, 1.0],
        top=0.86,
        bottom=0.13,
        left=0.08,
        right=0.98,
        wspace=0.28,
        hspace=0.36,
    )

    axA = fig.add_subplot(gs[0, 0])
    axA.set_facecolor(panel_bg)
    axA.text(-0.18, 1.02, "A", transform=axA.transAxes, fontweight="bold", va="bottom")

    for xi, point in zip(x, points):
        axA.plot(
            [xi, xi],
            [point.d1_acc, point.d1_clean_acc],
            color="#c9c9c9",
            lw=2.0,
            zorder=1,
        )
        axA.errorbar(
            [xi - 0.08],
            [point.d1_acc],
            yerr=[point.d1_std],
            fmt="o",
            color=color_raw,
            ms=6,
            elinewidth=1.3,
            capsize=3,
            label="D1 original" if xi == 0 else None,
            zorder=3,
        )
        axA.errorbar(
            [xi + 0.08],
            [point.d1_clean_acc],
            yerr=[point.d1_clean_std],
            fmt="o",
            color=color_clean,
            ms=6,
            elinewidth=1.3,
            capsize=3,
            label="D1 after removing U_FEM" if xi == 0 else None,
            zorder=3,
        )
        axA.text(
            xi,
            max(point.d1_acc, point.d1_clean_acc) + 0.035,
            f"{point.d1_delta:+.3f}",
            ha="center",
            va="bottom",
            color=color_limit if point.d1_delta > 0.01 else color_neutral,
            fontsize=8.5,
            fontweight="bold",
        )

    axA.axhline(0.25, color="#8d8d8d", lw=1.0, ls="--")
    axA.set_xticks(x)
    axA.set_xticklabels(labels)
    axA.set_xlabel("Stimulus size (LogMAR)")
    axA.set_ylabel("Orientation decoding accuracy")
    axA.set_ylim(0.22, 1.01)
    axA.set_title("Removing the shared FEM covariance\nimproves decoding only above threshold")
    axA.grid(True, axis="y", alpha=0.18)
    axA.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.60))
    axA.text(
        0.03,
        0.93,
        "Real-FEM condition, 4-way orientation task",
        transform=axA.transAxes,
        color="#666666",
    )
    axA.text(
        0.03,
        0.06,
        "Dashed line: chance = 0.25",
        transform=axA.transAxes,
        color="#666666",
    )

    axB = fig.add_subplot(gs[0, 1])
    axB.set_facecolor(panel_bg)
    axB.text(-0.18, 1.02, "B", transform=axB.transAxes, fontweight="bold", va="bottom")
    ymax = max(0.035, float(deltas.max()) + 0.006)
    axB.axhspan(0.0, ymax, color=help_bg, zorder=0)
    axB.bar(
        x,
        deltas,
        width=0.56,
        color=[color_limit if delta > 0.01 else color_neutral for delta in deltas],
        edgecolor="none",
        zorder=2,
    )
    axB.axhline(0.0, color="#555555", lw=1.0)
    axB.set_xticks(x)
    axB.set_xticklabels(labels)
    axB.set_xlabel("Stimulus size (LogMAR)")
    axB.set_ylabel("Accuracy change after removing U_FEM")
    axB.set_title("Direct causal test:\ndoes the FEM covariance subspace hurt or help?")
    axB.set_ylim(0.0, ymax)
    axB.grid(True, axis="y", alpha=0.18)
    for xi, delta in zip(x, deltas):
        is_limiting = delta > 0.01
        axB.text(
            xi,
            delta + 0.002,
            "information-limiting" if is_limiting else "causally neutral",
            ha="center",
            va="bottom",
            color=color_limit if is_limiting else "#507651",
            fontsize=8,
        )
    axB.text(
        0.03,
        0.92,
        "Positive values mean decoding got better after projection",
        transform=axB.transAxes,
        color="#666666",
    )

    axC = fig.add_subplot(gs[1, 0])
    axC.set_facecolor(panel_bg)
    axC.text(-0.18, 1.02, "C", transform=axC.transAxes, fontweight="bold", va="bottom")
    axC.axhspan(0.60, 0.75, color=help_bg, alpha=0.8, zorder=0)
    axC.axhspan(0.0, 0.60, color=neutral_bg, alpha=0.8, zorder=0)
    axC.plot(x, alphas, color=color_alpha, marker="o", lw=1.8, ms=5)
    for xi, point in zip(x, points):
        axC.text(
            xi,
            point.alpha + 0.025,
            f"alpha={point.alpha:.3f}",
            ha="center",
            color=color_alpha,
        )
    axC.set_xticks(x)
    axC.set_xticklabels(labels)
    axC.set_xlabel("Stimulus size (LogMAR)")
    axC.set_ylabel("Overlap between FEM covariance and signal")
    axC.set_ylim(0.48, 0.74)
    axC.set_title("At hyperacuity, FEM covariance overlaps less\nwith the orientation signal")
    axC.grid(True, axis="y", alpha=0.18)
    axC.text(0.03, 0.88, "higher overlap: shared FEM variability\ncontaminates the decoder", transform=axC.transAxes, color="#8a5644")
    axC.text(0.03, 0.08, "lower overlap: FEM covariance is\nmostly a bystander", transform=axC.transAxes, color="#4e6f4d")

    axD = fig.add_subplot(gs[1, 1])
    axD.set_facecolor(panel_bg)
    axD.text(-0.18, 1.02, "D", transform=axD.transAxes, fontweight="bold", va="bottom")
    eig_index = np.array([1, 2], dtype=float)
    for point, color, offset, label in [
        (points[0], color_signal_20, -0.08, f"{points[0].logmar:+.2f}"),
        (points[1], color_signal_40, +0.08, f"{points[1].logmar:+.2f}"),
    ]:
        vals = np.asarray(point.top_signal_eigvals, dtype=float)
        axD.vlines(eig_index + offset, 1e-6, vals, color=color, lw=2.2, alpha=0.95)
        axD.scatter(eig_index + offset, vals, s=42, color=color, zorder=3, label=label)
        for xi, val in zip(eig_index + offset, vals):
            axD.text(xi, val * 1.17, f"{val:.2e}", ha="center", va="bottom", fontsize=7, color=color)
    axD.set_yscale("log")
    axD.set_ylim(1e-6, 5e-4)
    axD.set_xticks(eig_index)
    axD.set_xticklabels(["eig1", "eig2"])
    axD.set_xlabel("Leading orientation-signal modes")
    axD.set_ylabel("Signal covariance eigenvalue")
    axD.set_title("Orientation means are more separated at -0.40,\neven though covariance is neutral")
    axD.grid(True, axis="y", which="both", alpha=0.18)
    axD.legend(frameon=False, loc="lower left")
    axD.text(
        0.02,
        0.92,
        "Interpretation: the signal subspace moved away\nfrom the FEM translation subspace",
        transform=axD.transAxes,
        color="#555555",
    )

    fig.suptitle("FEM covariance limits decoding above threshold but becomes neutral at hyperacuity", x=0.5, y=0.975, fontsize=12)
    fig.text(
        0.5,
        0.94,
        "D1 = linear decoder on time-averaged population rates. U_FEM = top shared covariance mode induced by fixational eye movements.\nProjecting out U_FEM asks whether that covariance helps decoding, hurts decoding, or is simply neutral.",
        ha="center",
        va="top",
        fontsize=8.0,
        color="#505050",
    )
    fig.text(
        0.5,
        0.02,
        "Take-home message: removing FEM covariance helps at LogMAR -0.20 because that covariance lies in the signal direction, but it has essentially no effect at LogMAR -0.40. The hyperacuity benefit comes from mean-rate sampling across positions, not from covariance carrying extra signal.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#505050",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=_repo_root() / "declan" / "fem_global_intervention_results" / "fem_global_intervention_real.npz",
        help="Saved real-condition intervention summary (.npz).",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=_repo_root() / "declan" / "fem_global_intervention_results" / "fig_priority2_causal_alignment",
        help="Output path without extension.",
    )
    args = parser.parse_args()

    points = _load_points(args.input)
    if len(points) != 2:
        raise RuntimeError(f"Expected exactly 2 logMAR points in {args.input}, got {len(points)}")

    fig = make_figure(points)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output_stem.with_suffix(".png")
    pdf_path = args.output_stem.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"Wrote: {png_path}")
    print(f"Wrote: {pdf_path}")


if __name__ == "__main__":
    main()