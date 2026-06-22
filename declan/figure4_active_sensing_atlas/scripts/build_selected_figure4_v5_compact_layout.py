"""Build a compact-layout Figure 4 design draft.

This version keeps the v4 analysis choices but moves Panel B into the top row
with Panel A. The remaining panels are tightened into a single lower row so the
story reads as:

retinal movie -> feature encoding
hidden-eye recovery -> edge mechanism -> behavior
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:  # pragma: no cover - script-mode import path fallback
    from . import build_selected_figure4_v4_design as v4
except ImportError:  # pragma: no cover
    import build_selected_figure4_v4_design as v4


OUT_DIR = v4.OUT_DIR

COMPACT_HEADERS = {
    "A": ("One image becomes a retinal movie", "recorded eye drift samples different views of the same scene"),
    "B": ("Motion enhances feature encoding", "but only when eye position is known"),
    "C": ("Compact-subspace recovery", "compact removal collapses hidden-eye recovery"),
    "D": ("Along-edge priors recover features", "matched-static hidden-eye decoder"),
    "E": ("Drift follows clear edges", "alignment strengthens with edge coherence"),
}


def _compact_panel_header(ax, label: str, title: str, subtitle: str) -> None:
    title, subtitle = COMPACT_HEADERS.get(label, (title, subtitle))
    title_x = 0.16
    if label == "A":
        title_x = 0.11
    ax.text(
        0.0,
        1.10,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=v4.BLUE,
        fontsize=13.2,
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        title_x,
        1.118,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=v4.INK,
        fontsize=9.5,
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        title_x,
        1.045,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=v4.MUTED,
        fontsize=6.9,
        clip_on=False,
    )


def _write_manifest(path: Path) -> None:
    rows = [
        ("A", "One image becomes a retinal movie", v4.A_PNG.relative_to(v4.ATLAS).as_posix()),
        (
            "B",
            "Motion enhances feature encoding",
            f"{v4.B_GAIN_CSV.relative_to(v4.REPO_ROOT).as_posix()}; "
            f"{v4.B_POSE_UNAWARE_CSV.relative_to(v4.REPO_ROOT).as_posix()}",
        ),
        ("C", "Features survive hidden eye position", v4.C_POSTERIOR_CSV.relative_to(v4.REPO_ROOT).as_posix()),
        (
            "D",
            "Example edge axes plus along-edge feature recovery",
            f"{v4.D_FEATURE_SUMMARY_CSV.relative_to(v4.REPO_ROOT).as_posix()}; "
            f"{v4.D_THUMBNAIL_VALUES_CSV.relative_to(v4.REPO_ROOT).as_posix()}",
        ),
        ("E", "Real drift follows coherent edges", v4.E_WINDOWS_CSV.relative_to(v4.REPO_ROOT).as_posix()),
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["panel", "title", "source"])
        writer.writerows(rows)


def _write_caption(path: Path) -> None:
    caption = """# Figure 4 Selected Composite v5

Status: compact-layout provisional draft, 2026-06-21.

This version keeps the current analysis choices but moves Panel B into the top
row with Panel A. Panels C-E are compressed into one lower row so the composite
reads as retinal movie -> corrected feature-response change, then compact-subspace
recovery -> along-edge feature recovery -> behavior.

Draft legend:

Figure 4. Small fixational eye movements turn a static natural image into an
informative retinal movie. (A) A recorded eye trace samples different retinal
views of the same image. (B) In the V1 twin, recorded drift produces corrected
delta-mean feature-response gain when the eye trace is known, whereas the
pose-unaware empirical proxy falls below static; OU controls are held out of the
main trace set pending the trace/readout audit. (C) Zero-eye
feature recovery falls as motion scale grows, compact-subspace inference
remains stable without being given the eye trace, and compact removal collapses
recovery back toward the zero-eye curve. (D)
An example natural-image edge shows the along/across axes; in the matched-static
hidden-eye feature decoder, along-edge trajectory priors recover more feature
signal than matched across-edge priors. Hard-negative controls keep this as a
scoped axis result rather than a universal motion policy. (E) Measured drift shows the same
contour-following geometry most clearly when the local image supplies a
coherent edge axis; gray bars show the number of sampled windows per coherence
bin. The figure supports convergence between useful retinal-movie structure
and measured behavior, not a completed proof of behavioral optimality.
"""
    path.write_text(caption, encoding="utf-8")


def _plot_a_compact(ax) -> None:
    image = Image.open(v4.A_PNG).convert("RGB")
    crop = image.crop((60, 150, image.width - 72, image.height - 130))
    ax.imshow(crop)
    ax.set_axis_off()
    _compact_panel_header(
        ax,
        "A",
        "One image becomes a retinal movie",
        "recorded eye drift samples different views of the same scene",
    )


def _plot_e_compact(ax):
    values = v4._behavior_bins()
    count_ax = ax.twinx()
    ax.set_zorder(count_ax.get_zorder() + 1)
    ax.patch.set_visible(False)
    count_ax.bar(
        values["bin_center"],
        values["n_windows"],
        width=0.075,
        color=v4.LIGHT_GRAY,
        edgecolor="none",
        zorder=0,
    )
    count_ax.set_yticks([])
    count_ax.set_ylabel("")
    count_ax.spines["top"].set_visible(False)
    count_ax.spines["right"].set_visible(False)

    y = values["mean_edge_alignment_index"].to_numpy(dtype=float)
    lo = values["ci95_low"].to_numpy(dtype=float)
    hi = values["ci95_high"].to_numpy(dtype=float)
    ax.errorbar(
        values["bin_center"],
        y,
        yerr=np.vstack([y - lo, hi - y]),
        color=v4.BLUE,
        marker="o",
        markersize=3.8,
        lw=2.0,
        capsize=0,
        zorder=3,
    )
    ax.axhline(0, color=v4.INK, lw=0.8)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.04, 0.36)
    ax.set_xlabel("local edge coherence")
    ax.set_ylabel("edge-following alignment")
    v4._clean_axis(ax)
    _compact_panel_header(
        ax,
        "E",
        "Real drift follows coherent edges",
        "behavioral alignment strengthens when the local edge is clear",
    )
    return values


def build() -> list[Path]:
    v4._configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11.4, 7.35), constrained_layout=False)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        2,
        6,
        height_ratios=[1.08, 1.0],
        left=0.055,
        right=0.975,
        top=0.82,
        bottom=0.08,
        hspace=0.50,
        wspace=0.70,
    )

    fig.text(
        0.055,
        0.955,
        "Figure 4. Small eye movements turn images into informative retinal movies",
        ha="left",
        va="top",
        fontsize=15.0,
        fontweight="bold",
        color=v4.INK,
    )
    fig.text(
        0.055,
        0.912,
        "Retinal motion changes feature responses; compact subspace recovers hidden-eye features; local edge priors shape which motion helps.",
        ha="left",
        va="top",
        fontsize=8.4,
        color=v4.MUTED,
    )
    fig.add_artist(
        plt.Line2D([0.055, 0.975], [0.886, 0.886], transform=fig.transFigure, color="#c9d0d6", lw=0.8)
    )

    ax_a = fig.add_subplot(gs[0, 0:4])
    ax_b = fig.add_subplot(gs[0, 4:6])
    ax_c = fig.add_subplot(gs[1, 0:2])
    ax_e = fig.add_subplot(gs[1, 4:6])

    original_header = v4._panel_header
    v4._panel_header = _compact_panel_header
    try:
        _plot_a_compact(ax_a)
        b_values = v4._plot_b(ax_b)
        c_values = v4._plot_c(ax_c)
        d_values = v4._plot_d(gs[1, 2:4])
        e_values = _plot_e_compact(ax_e)
    finally:
        v4._panel_header = original_header

    png = OUT_DIR / "figure4_selected_v5.png"
    pdf = OUT_DIR / "figure4_selected_v5.pdf"
    manifest = OUT_DIR / "figure4_selected_v5_manifest.csv"
    caption = OUT_DIR / "figure4_selected_v5_caption.md"
    b_csv = OUT_DIR / "figure4_selected_v5_panel_b_values.csv"
    c_csv = OUT_DIR / "figure4_selected_v5_panel_c_values.csv"
    d_csv = OUT_DIR / "figure4_selected_v5_panel_d_values.csv"
    e_csv = OUT_DIR / "figure4_selected_v5_panel_e_values.csv"

    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    _write_manifest(manifest)
    _write_caption(caption)
    b_values.to_csv(b_csv, index=False)
    c_values.to_csv(c_csv, index=False)
    d_values.to_csv(d_csv, index=False)
    e_values.to_csv(e_csv, index=False)
    return [png, pdf, manifest, caption, b_csv, c_csv, d_csv, e_csv]


def main() -> None:
    for path in build():
        print(path)


if __name__ == "__main__":
    main()
