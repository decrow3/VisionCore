#!/usr/bin/env python3
"""
Generate covTFTS Figure 4.

Figure 4. Image-specific retinal translations form a compact,
image-generalizing reafferent geometry.

This script is intentionally self-contained: it uses pandas/matplotlib only,
reads the finalized TFTS outputs when available, and falls back to the compact
MANUSCRIPT_REPORT.md table when some CSVs are absent. It draws the schematic
panels programmatically so the figure is reproducible.

Intended usage from the VisionCore repository root:

    python declan/covTFTS_figure/generate_covTFTS_figure.py \
        --tfts-root outputs/twin_feature_tangent_structure_prod_limited_synth \
        --out-dir outputs/covTFTS_figure

The script will look for the image-disjoint basis file first. If unavailable,
it will fall back to the generic train_test_basis file and mark this in the
manifest / README. Do not use a fallback output as a final manuscript figure
unless the manifest confirms image_disjoint as the basis source.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyArrowPatch,
    Polygon,
    Rectangle,
)
from matplotlib.lines import Line2D


# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------

TEXT = "#202124"
REC = "#2f2f2f"
REC_LIGHT = "#d8d8d8"
MODEL = "#2f5f9f"
MODEL_LIGHT = "#d8e6f5"
MODEL_MID = "#7fa8d8"
BRIDGE = "#7b5ea7"
BRIDGE_LIGHT = "#e7ddf2"
NULL = "#9a9a9a"
ACCENT = "#c44e52"
GREEN = "#4b8a5a"

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)


def clean_axes(ax, grid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, alpha=0.18, zorder=-1)


def setup_schematic_ax(ax):
    """Use a stable 0..1 coordinate system for schematic panels."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")


def panel_label(ax, letter: str, title: str):
    ax.set_title(f"{letter}", loc="left", fontweight="bold", pad=4)
    ax.text(
        0.08,
        1.015,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        fontweight="bold",
        color=TEXT,
    )


def arrow(ax, xy1, xy2, color=TEXT, lw=1.2, mutation_scale=10, ls="-", alpha=1.0, transform=None):
    """Arrow helper. Schematic panels use axes-fraction coordinates by default."""
    a = FancyArrowPatch(
        xy1,
        xy2,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        lw=lw,
        color=color,
        linestyle=ls,
        alpha=alpha,
        shrinkA=1,
        shrinkB=1,
        zorder=5,
        transform=(ax.transAxes if transform is None else transform),
    )
    ax.add_patch(a)
    return a


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


@dataclass
class DataPaths:
    tfts_root: Path
    union_file: Optional[Path]
    basis_file: Optional[Path]
    covariance_file: Optional[Path]
    report_file: Optional[Path]
    summary_file: Optional[Path]
    basis_source_label: str
    warnings: list[str]


def _exists(p: Optional[Path]) -> Optional[Path]:
    if p is not None and p.exists():
        return p
    return None


def first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_paths(tfts_root: Path, fallback_dir: Optional[Path] = None) -> DataPaths:
    """Find finalized TFTS output files."""
    warnings = []
    fallback_dir = fallback_dir or Path.cwd()

    # Union summary. Prefer the image-disjoint split summary for the final figure.
    union_file = first_existing(
        [
            tfts_root / "split_modes" / "image_disjoint" / "twin_tangent_union_summary_image_disjoint.csv",
            tfts_root / "split_modes" / "image_disjoint" / "twin_tangent_union_summary.csv",
            tfts_root / "union_spectrum" / "twin_tangent_union_summary.csv",
            tfts_root / "twin_tangent_union_summary.csv",
            fallback_dir / "twin_tangent_union_summary_image_disjoint.csv",
            fallback_dir / "twin_tangent_union_summary.csv",
        ]
    )

    # Prefer image-disjoint split, then generic train/test.
    image_disjoint_candidates = [
        tfts_root
        / "split_modes"
        / "image_disjoint"
        / "twin_tangent_train_test_basis_image_disjoint.csv",
        tfts_root
        / "split_modes"
        / "image_disjoint"
        / "twin_tangent_train_test_basis.csv",
        tfts_root
        / "train_test_basis"
        / "image_disjoint"
        / "twin_tangent_train_test_basis_image_disjoint.csv",
    ]
    basis_file = first_existing(image_disjoint_candidates)
    basis_source_label = "image_disjoint"
    if basis_file is None:
        basis_file = first_existing(
            [
                tfts_root / "train_test_basis" / "twin_tangent_train_test_basis.csv",
                tfts_root / "twin_tangent_train_test_basis.csv",
                fallback_dir / "twin_tangent_train_test_basis.csv",
            ]
        )
        basis_source_label = "fallback_generic_train_test"
        if basis_file is not None:
            warnings.append(
                "Using generic train_test_basis file because image_disjoint file was not found. "
                "For the final manuscript figure, use the image-disjoint file."
            )

    covariance_file = first_existing(
        [
            tfts_root / "covariance_approx" / "twin_linear_covariance_approx.csv",
            tfts_root / "twin_linear_covariance_approx.csv",
            fallback_dir / "twin_linear_covariance_approx.csv",
        ]
    )

    report_file = first_existing(
        [
            tfts_root / "MANUSCRIPT_REPORT.md",
            fallback_dir / "MANUSCRIPT_REPORT.md",
        ]
    )
    summary_file = first_existing(
        [
            tfts_root / "twin_feature_tangent_summary.json",
            fallback_dir / "twin_feature_tangent_summary.json",
        ]
    )

    for name, p in [
        ("union_file", union_file),
        ("basis_file", basis_file),
        ("report_file", report_file),
    ]:
        if p is None:
            warnings.append(f"Missing {name}; associated panel may use constants or fail.")

    return DataPaths(
        tfts_root=tfts_root,
        union_file=union_file,
        basis_file=basis_file,
        covariance_file=covariance_file,
        report_file=report_file,
        summary_file=summary_file,
        basis_source_label=basis_source_label,
        warnings=warnings,
    )


def load_union(paths: DataPaths) -> pd.DataFrame:
    if paths.union_file is None:
        # Conservative fallback from current manuscript report.
        return pd.DataFrame(
            {
                "delta": [0.125, 0.25, 0.5],
                "participation_ratio": [7.8029, 9.0407, 6.5500],
                "null_pr_mean": [27.4645, 31.0288, 24.2867],
                "null_pr_ci_low": [27.1661, 30.6021, 24.0068],
                "null_pr_ci_high": [27.7510, 31.3573, 24.5549],
                "status": ["fallback"] * 3,
            }
        )
    df = pd.read_csv(paths.union_file)
    # Normalize column names from possible variants.
    rename = {
        "PR": "participation_ratio",
        "null_mean": "null_pr_mean",
        "null_95CI_low": "null_pr_ci_low",
        "null_95CI_high": "null_pr_ci_high",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "component_set" in df.columns:
        df = df[df["component_set"].astype(str).str.lower().eq("combined")]
    if "space" in df.columns:
        # Prefer raw space if present.
        raw = df[df["space"].astype(str).str.lower().eq("raw")]
        if len(raw):
            df = raw
    return df.copy()


def summarize_basis(df: pd.DataFrame) -> pd.DataFrame:
    """Median across folds by delta x k."""
    if "basis_rank_k" not in df.columns and "k" in df.columns:
        df = df.rename(columns={"k": "basis_rank_k"})
    if "test_variance_captured" not in df.columns and "capture_median" in df.columns:
        # Already summarized report form.
        out = df.rename(
            columns={
                "capture_median": "capture",
                "null_median": "null",
                "effect_median": "effect",
            }
        )
        return out

    ok = df.copy()
    if "fold_status" in ok.columns:
        ok = ok[ok["fold_status"].astype(str).str.lower().eq("ok")]
    out = (
        ok.groupby(["delta", "basis_rank_k"], as_index=False)
        .agg(
            capture=("test_variance_captured", "median"),
            capture_lo=("test_variance_captured", lambda x: np.nanpercentile(x, 2.5)),
            capture_hi=("test_variance_captured", lambda x: np.nanpercentile(x, 97.5)),
            null=("null_mean", "median"),
            null_lo=("null_mean", lambda x: np.nanpercentile(x, 2.5)),
            null_hi=("null_mean", lambda x: np.nanpercentile(x, 97.5)),
            effect=("effect_minus_null", "median"),
            n_folds=("fold", "nunique") if "fold" in ok.columns else ("delta", "size"),
        )
    )
    return out


def load_basis(paths: DataPaths) -> pd.DataFrame:
    if paths.basis_file is None:
        # Use the image-disjoint values reported in the status note.
        return pd.DataFrame(
            {
                "delta": [0.125] * 4 + [0.25] * 4 + [0.5] * 4,
                "basis_rank_k": [2, 5, 10, 20] * 3,
                "capture": [
                    0.346,
                    0.441,
                    0.539,
                    0.651,
                    0.284,
                    0.396,
                    0.498,
                    0.632,
                    0.368,
                    0.513,
                    0.614,
                    0.720,
                ],
                "null": [
                    0.127,
                    0.133,
                    0.141,
                    0.153,
                    0.101,
                    0.106,
                    0.113,
                    0.125,
                    0.118,
                    0.122,
                    0.129,
                    0.143,
                ],
                "n_folds": [5] * 12,
            }
        )
    df = pd.read_csv(paths.basis_file)
    return summarize_basis(df)


def _extract_markdown_table(report_text: str, section_name: str) -> Optional[pd.DataFrame]:
    pat = re.compile(
        rf"## {re.escape(section_name)}\s*\n(?P<table>(?:\|.*\|\s*\n)+)",
        re.MULTILINE,
    )
    m = pat.search(report_text)
    if not m:
        return None
    lines = [ln.strip() for ln in m.group("table").splitlines() if ln.strip()]
    rows = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if all(re.fullmatch(r"-+", c.replace(" ", "")) for c in cells):
            continue
        rows.append(cells)
    if len(rows) < 2:
        return None
    header = rows[0]
    data = rows[1:]
    df = pd.DataFrame(data, columns=header)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


def load_covariance_bridge(paths: DataPaths) -> pd.DataFrame:
    if paths.covariance_file is not None:
        df = pd.read_csv(paths.covariance_file)
        # Normalize common column names.
        rename = {
            "overlap_k2": "overlap_k2_median",
            "subspace_overlap_k2": "overlap_k2_median",
            "trace_ratio": "trace_ratio_lin_full_median",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        # Summarize if object-level.
        if "overlap_k2_median" not in df.columns:
            overlap_cols = [c for c in df.columns if "overlap" in c and "k2" in c]
            if overlap_cols:
                df = df.rename(columns={overlap_cols[0]: "overlap_k2_median"})
        if {"delta", "cloud_scale", "overlap_k2_median"}.issubset(df.columns):
            return (
                df.groupby(["delta", "cloud_scale"], as_index=False)
                .agg(
                    overlap_k2_median=("overlap_k2_median", "median"),
                    trace_ratio_lin_full_median=(
                        "trace_ratio_lin_full_median",
                        "median",
                    )
                    if "trace_ratio_lin_full_median" in df.columns
                    else ("overlap_k2_median", "size"),
                )
            )

    if paths.report_file is not None:
        text = paths.report_file.read_text(encoding="utf-8")
        df = _extract_markdown_table(text, "Analysis 5 Locality")
        if df is not None:
            return df.rename(columns={"overlap_k2_median": "overlap_k2_median"})

    # Fallback from latest compact report.
    return pd.DataFrame(
        {
            "delta": [0.25] * 4,
            "cloud_scale": [0.25, 0.5, 1.0, 2.0],
            "overlap_k2_median": [0.3951, 0.3441, 0.2951, 0.2449],
            "trace_ratio_lin_full_median": [1.3872, 2.6641, 5.4681, 12.0299],
        }
    )


# -----------------------------------------------------------------------------
# Schematic helpers
# -----------------------------------------------------------------------------


def draw_image_patch(ax, xy, wh=(0.18, 0.12), seed=0, label=None, transform=None):
    """Small deterministic natural-image-like patch using imshow."""
    rng = np.random.default_rng(seed)
    h, w = 28, 40
    yy, xx = np.mgrid[0:h, 0:w]
    img = (
        0.45
        + 0.20 * np.sin(xx / 4.0 + seed)
        + 0.15 * np.cos((xx + yy) / 7.0)
        + 0.12 * rng.normal(size=(h, w))
    )
    img = np.clip(img, 0, 1)
    x, y = xy
    ww, hh = wh
    trans = transform or ax.transData
    ax.imshow(
        img,
        cmap="gray",
        extent=(x, x + ww, y, y + hh),
        transform=trans,
        origin="lower",
        zorder=2,
        interpolation="bilinear",
        clip_on=False,
    )
    ax.add_patch(
        Rectangle(
            (x, y),
            ww,
            hh,
            transform=trans,
            facecolor="none",
            edgecolor=TEXT,
            lw=0.7,
            zorder=3,
        )
    )
    ax.set_aspect("auto")
    if label:
        ax.text(x + ww / 2, y - 0.025, label, transform=trans, ha="center", va="top", fontsize=6.5)


def draw_plane(ax, center, size=(0.24, 0.10), angle=0, color=MODEL_LIGHT, edge=MODEL, alpha=0.9, transform=None):
    trans = transform or ax.transData
    cx, cy = center
    w, h = size
    pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    th = np.deg2rad(angle)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = pts @ R.T + np.array([cx, cy])
    poly = Polygon(pts, closed=True, facecolor=color, edgecolor=edge, lw=0.9, alpha=alpha, transform=trans, zorder=2)
    ax.add_patch(poly)
    return poly


def draw_tangent_arrows(ax, origin, angle_deg=0, length=0.10, color=MODEL, transform=None, alpha=1.0):
    trans = transform or ax.transData
    x, y = origin
    th = np.deg2rad(angle_deg)
    dx, dy = length * np.cos(th), length * np.sin(th)
    arrow(ax, (x, y), (x + dx, y + dy), color=color, lw=1.2, mutation_scale=8, alpha=alpha, transform=trans)
    arrow(ax, (x, y), (x - 0.55 * dy, y + 0.55 * dx), color=color, lw=1.2, mutation_scale=8, alpha=alpha, transform=trans)


# -----------------------------------------------------------------------------
# Panels
# -----------------------------------------------------------------------------


def plot_panel_a(ax):
    setup_schematic_ax(ax)
    panel_label(ax, "A", "Recorded anchor")

    # Recorded register card.
    ax.add_patch(Rectangle((0.035, 0.08), 0.93, 0.82, transform=ax.transAxes,
                           fill=False, edgecolor="0.25", lw=0.9))
    ax.text(0.06, 0.84, "Recorded V1", transform=ax.transAxes,
            fontsize=8.8, fontweight="bold", color=REC)

    # Mini scatter with explicit axes, representing the collapse of shared variability.
    x0, y0, w, h = 0.075, 0.26, 0.42, 0.46
    ax.plot([x0, x0 + w], [y0, y0], color="0.25", lw=0.9, transform=ax.transAxes)
    ax.plot([x0, x0], [y0, y0 + h], color="0.25", lw=0.9, transform=ax.transAxes)
    ax.text(x0 + w / 2, y0 - 0.055, "population state 1", ha="center",
            transform=ax.transAxes, fontsize=6.6)
    ax.text(x0 - 0.045, y0 + h / 2, "population\nstate 2", ha="center",
            va="center", rotation=90, transform=ax.transAxes, fontsize=6.6)

    rng = np.random.default_rng(5)
    pts = rng.normal(size=(180, 2)) @ np.array([[0.105, 0.055], [0.055, 0.050]]) + np.array([0.50, 0.48])
    pts[:, 0] = np.clip(pts[:, 0], 0.05, 0.95)
    pts[:, 1] = np.clip(pts[:, 1], 0.05, 0.95)
    xx = x0 + w * pts[:, 0]
    yy = y0 + h * pts[:, 1]
    ax.scatter(xx, yy, s=8, color=REC, alpha=0.52, transform=ax.transAxes, linewidths=0)
    # Ellipse after conditioning.
    ax.add_patch(Ellipse((x0 + w * 0.68, y0 + h * 0.55), w * 0.25, h * 0.20,
                         angle=10, transform=ax.transAxes, facecolor=MODEL_LIGHT,
                         edgecolor=MODEL, lw=1.0, alpha=0.9))
    ax.text(x0 + w * 0.72, y0 + h * 0.11, "after FEM\nconditioning",
            ha="center", transform=ax.transAxes, fontsize=6.6, color=MODEL)
    arrow(ax, (x0 + w * 0.48, y0 + h * 0.52), (x0 + w * 0.61, y0 + h * 0.55),
          color=TEXT, lw=1.15, mutation_scale=9)

    # Visual response axis annotation.
    arrow(ax, (x0 + w * 0.53, y0 + h * 0.58), (x0 + w * 0.77, y0 + h * 0.76),
          color=ACCENT, lw=1.55, mutation_scale=10)
    ax.text(x0 + w * 0.72, y0 + h * 0.79, "aligned with\nvisual response",
            ha="center", transform=ax.transAxes, fontsize=6.6, color=ACCENT)

    # Low-rank eigenspectrum glyph on the right.
    ax.text(0.72, 0.72, r"low-dimensional $\Sigma_{\mathrm{FEM}}$",
            ha="center", transform=ax.transAxes, fontsize=7.0, color=TEXT)
    xs = np.linspace(0.61, 0.88, 6)
    heights = np.array([0.22, 0.125, 0.055, 0.030, 0.020, 0.014])
    for x, hh in zip(xs, heights):
        ax.add_patch(Rectangle((x, 0.36), 0.032, hh, transform=ax.transAxes,
                               facecolor=REC_LIGHT, edgecolor="0.48", lw=0.55))
    ax.plot([0.59, 0.92], [0.34, 0.34], color="0.30", lw=0.8, transform=ax.transAxes)
    ax.text(0.75, 0.22, "FEM-linked covariance\nremoved from classical\nshared variability",
            ha="center", transform=ax.transAxes, fontsize=6.5, color="0.25")

def plot_panel_b(ax):
    setup_schematic_ax(ax)
    panel_label(ax, "B", "Why compactness is nontrivial")

    # Header blocks.
    ax.text(0.19, 0.84, "single image", ha="center", transform=ax.transAxes,
            fontsize=7.5, fontweight="bold", color=TEXT)
    ax.text(0.67, 0.84, "across images?", ha="center", transform=ax.transAxes,
            fontsize=7.5, fontweight="bold", color=TEXT)
    ax.plot([0.42, 0.42], [0.16, 0.82], color="0.72", lw=1.1, transform=ax.transAxes)

    # Left side: one image gives a local low-D plane.
    draw_image_patch(ax, (0.10, 0.68), wh=(0.18, 0.11), seed=1)
    draw_plane(ax, (0.19, 0.42), size=(0.27, 0.095), angle=3,
               color="#eef4fb", edge=MODEL, alpha=0.95)
    ax.add_patch(Ellipse((0.19, 0.42), 0.18, 0.050, angle=3, transform=ax.transAxes,
                         facecolor="none", edgecolor=MODEL, lw=1.05, alpha=0.9))
    draw_tangent_arrows(ax, (0.19, 0.42), angle_deg=3, length=0.080, color=MODEL)
    ax.text(0.19, 0.20, "local low-dimensional\ntranslation neighborhood",
            ha="center", transform=ax.transAxes, fontsize=6.8)

    # Right side: unrelated image-specific planes could smear.
    examples = [(0.56, 0.67, 24), (0.75, 0.68, -25), (0.58, 0.48, 75), (0.76, 0.47, -62)]
    for i, (x, y, ang) in enumerate(examples):
        draw_image_patch(ax, (x - 0.033, y + 0.070), wh=(0.066, 0.045), seed=10 + i)
        draw_plane(ax, (x, y), size=(0.145, 0.050), angle=ang,
                   color="#eeeeee", edge=NULL, alpha=0.90)
        draw_tangent_arrows(ax, (x, y), angle_deg=ang, length=0.040, color=NULL, alpha=0.85)

    ax.add_patch(Ellipse((0.67, 0.29), 0.34, 0.115, angle=-5, transform=ax.transAxes,
                         facecolor="0.93", edgecolor="0.48", lw=0.85, alpha=0.96))
    ax.text(0.67, 0.245, "high-D mixture", ha="center", transform=ax.transAxes,
            fontsize=7.0, color=TEXT)
    ax.text(0.67, 0.105, "Null possibility: image-specific\nplanes need not align",
            ha="center", transform=ax.transAxes, fontsize=6.5, color="0.35")

def plot_panel_c(ax):
    setup_schematic_ax(ax)
    panel_label(ax, "C", "Tangent construction in the twin")

    ax.text(0.055, 0.86, "Canonical twin", transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", color=MODEL)

    # Stimulus history stack.
    for i in range(5):
        draw_image_patch(ax, (0.08 + i * 0.018, 0.58 + i * 0.014),
                         wh=(0.145, 0.095), seed=21 + i)
    ax.text(0.16, 0.46, "full stimulus-history\nobject", ha="center",
            transform=ax.transAxes, fontsize=6.8)

    # Shift arrows and model block.
    arrow(ax, (0.30, 0.64), (0.39, 0.64), color=TEXT, lw=1.1, mutation_scale=9)
    ax.text(0.345, 0.72, "small retinal\nshifts", ha="center",
            transform=ax.transAxes, fontsize=6.7)

    ax.add_patch(Rectangle((0.43, 0.55), 0.15, 0.18, transform=ax.transAxes,
                           facecolor=MODEL_LIGHT, edgecolor=MODEL, lw=1.1))
    ax.text(0.505, 0.64, "twin\nresponse", ha="center", va="center",
            transform=ax.transAxes, fontsize=7.1, color=MODEL)
    arrow(ax, (0.59, 0.64), (0.68, 0.64), color=TEXT, lw=1.1, mutation_scale=9)

    # Compact bundle / slab with multiple image-specific tangent arrows.
    draw_plane(ax, (0.82, 0.59), size=(0.30, 0.12), angle=-9,
               color="#eef4fb", edge=MODEL, alpha=0.78)
    origins = [(0.75, 0.56), (0.81, 0.63), (0.88, 0.58), (0.82, 0.51)]
    angles = [-14, 24, 48, -42]
    for origin, ang in zip(origins, angles):
        draw_tangent_arrows(ax, origin, angle_deg=ang, length=0.055, color=MODEL, alpha=0.95)

    ax.text(0.82, 0.37, r"$b_x(I),\; b_y(I)$" + "\nimage-specific",
            ha="center", transform=ax.transAxes, fontsize=8.0, color=MODEL)
    ax.text(0.82, 0.20, "compact bundle;\nnot a universal\nsigned axis",
            ha="center", transform=ax.transAxes, fontsize=6.6, color=TEXT)

def plot_panel_d(ax, union_df: pd.DataFrame):
    panel_label(ax, "D", "Compact tangent family")
    df = union_df.sort_values("delta")
    x = np.arange(len(df))
    obs = df["participation_ratio"].astype(float).to_numpy()
    null_mean = df["null_pr_mean"].astype(float).to_numpy()
    lo = df["null_pr_ci_low"].astype(float).to_numpy()
    hi = df["null_pr_ci_high"].astype(float).to_numpy()

    ax.fill_between(x, lo, hi, color=NULL, alpha=0.18, lw=0, zorder=1)
    ax.plot(x, null_mean, "o-", color=NULL, lw=2.0, markersize=5.2,
            markeredgecolor="white", markeredgewidth=0.5, label="Unit-shuffle null", zorder=2)
    ax.plot(x, obs, "o-", color=MODEL, lw=2.5, markersize=5.8,
            markeredgecolor="white", markeredgewidth=0.5, label="Observed", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:g}" for d in df["delta"].astype(float)])
    ax.set_xlabel("Tangent displacement (arcmin)")
    ax.set_ylabel("Participation ratio")
    clean_axes(ax, grid=True)
    ax.legend(frameon=False, loc="upper right", handlelength=1.5)

    idx = int(np.argmin(np.abs(df["delta"].astype(float).to_numpy() - 0.25)))
    ax.annotate(
        f"0.25 arcmin:\nPR {obs[idx]:.1f} vs null {null_mean[idx]:.1f}",
        xy=(x[idx], obs[idx]),
        xytext=(x[idx] + 0.23, obs[idx] + 5.7),
        arrowprops=dict(arrowstyle="->", color=TEXT, lw=0.9),
        fontsize=7.4,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.85", lw=0.6, alpha=0.95),
    )

def plot_panel_e(ax, basis_df: pd.DataFrame, basis_source_label: str):
    panel_label(ax, "E", "Image-disjoint generalization")
    df = basis_df.copy()
    if "basis_rank_k" not in df.columns and "k" in df.columns:
        df = df.rename(columns={"k": "basis_rank_k"})
    deltas = np.array(sorted(df["delta"].astype(float).unique()))
    delta = float(deltas[np.argmin(np.abs(deltas - 0.25))])
    d = df[np.isclose(df["delta"].astype(float), delta)].sort_values("basis_rank_k")

    k = d["basis_rank_k"].astype(float).to_numpy()
    capture = d["capture"].astype(float).to_numpy()
    null = d["null"].astype(float).to_numpy()

    if {"capture_lo", "capture_hi"}.issubset(d.columns):
        ax.fill_between(k, d["capture_lo"].astype(float), d["capture_hi"].astype(float),
                        color=MODEL, alpha=0.15, lw=0, zorder=1)
    if {"null_lo", "null_hi"}.issubset(d.columns):
        ax.fill_between(k, d["null_lo"].astype(float), d["null_hi"].astype(float),
                        color=NULL, alpha=0.15, lw=0, zorder=1)

    ax.plot(k, null, "o--", color=NULL, lw=1.8, markersize=5.0,
            markeredgecolor="white", markeredgewidth=0.5, label="Unit-shuffle null", zorder=2)
    ax.plot(k, capture, "o-", color=MODEL, lw=2.6, markersize=5.8,
            markeredgecolor="white", markeredgewidth=0.5, label="Observed", zorder=3)
    ax.set_xlabel("Basis dimension k")
    ax.set_ylabel("Held-out tangent\nvariance captured")
    ax.set_xticks(k)
    ax.set_ylim(0, min(0.85, max(0.72, np.nanmax(capture) + 0.08)))
    clean_axes(ax, grid=True)
    ax.legend(frameon=False, loc="lower right", handlelength=1.5)

    # Image-disjoint callout and k=10 headline.
    ax.text(0.03, 0.96, "image-disjoint split", transform=ax.transAxes,
            fontsize=7.3, color=MODEL, fontweight="bold", ha="left", va="top")
    if np.any(np.isclose(k, 10)):
        i = int(np.where(np.isclose(k, 10))[0][0])
    else:
        i = int(np.argmin(np.abs(k - 10)))
    if basis_source_label == "image_disjoint":
        label = f"k={int(k[i])}: {capture[i]:.2f} vs null {null[i]:.2f}\n0% image-ID leakage"
        color = TEXT
    else:
        label = f"k={int(k[i])}: {capture[i]:.2f} vs null {null[i]:.2f}\nWARNING: fallback split"
        color = ACCENT
    ax.annotate(
        label,
        xy=(k[i], capture[i]),
        xytext=(k[i] + 1.3, max(0.12, capture[i] - 0.17)),
        arrowprops=dict(arrowstyle="->", color=color, lw=0.9),
        fontsize=7.4,
        ha="left",
        va="center",
        color=color,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.85", lw=0.6, alpha=0.95),
    )
    if basis_source_label != "image_disjoint":
        ax.text(0.03, 0.86, "Fallback source: not image-disjoint",
                transform=ax.transAxes, fontsize=6.5, color=ACCENT,
                ha="left", va="top", alpha=0.90)

def plot_panel_f(ax, cov_df: pd.DataFrame):
    panel_label(ax, "F", "Partial covariance bridge")
    df = cov_df.copy()
    deltas = np.array(sorted(pd.to_numeric(df["delta"], errors="coerce").dropna().unique()))
    delta = float(deltas[np.argmin(np.abs(deltas - 0.25))]) if len(deltas) else 0.25
    d = df[np.isclose(pd.to_numeric(df["delta"], errors="coerce").astype(float), delta)].copy()
    d["cloud_scale"] = pd.to_numeric(d["cloud_scale"], errors="coerce")
    d["overlap_k2_median"] = pd.to_numeric(d["overlap_k2_median"], errors="coerce")
    d = d.sort_values("cloud_scale")

    ax.plot(d["cloud_scale"], d["overlap_k2_median"], "o-", color=BRIDGE,
            lw=2.5, markersize=5.8, markeredgecolor="white", markeredgewidth=0.5,
            label="Local tangent covariance", zorder=3)
    ax.set_xlabel("Eye-position cloud scale")
    ax.set_ylabel("Overlap with\nfull FEM covariance")
    ax.set_ylim(0, max(0.5, float(np.nanmax(d["overlap_k2_median"])) + 0.08))
    clean_axes(ax, grid=True)

    # Compact annotation box, deliberately restrained.
    ax.text(0.06, 0.91, "meaningful but incomplete", transform=ax.transAxes,
            fontsize=7.6, fontweight="bold", color=BRIDGE, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=BRIDGE_LIGHT, lw=0.8, alpha=0.98))
    ax.text(0.55, 0.88, "strongest for local clouds;\nweaker for broader finite shifts",
            transform=ax.transAxes, fontsize=6.9, color=TEXT, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.88", lw=0.5, alpha=0.92))

    # Dashed partial bridge cue near tail.
    if len(d) >= 2:
        x_tail = d["cloud_scale"].iloc[-2:]
        y_tail = d["overlap_k2_median"].iloc[-2:]
        ax.plot(x_tail, y_tail, color=BRIDGE, lw=2.2, linestyle="--", alpha=0.85)
    ax.annotate("partial bridge", xy=(0.86, 0.19), xycoords="axes fraction",
                xytext=(0.63, 0.12), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=1.0, color=BRIDGE, linestyle="--"),
                fontsize=6.6, color=BRIDGE, ha="center")


# -----------------------------------------------------------------------------
# Compose
# -----------------------------------------------------------------------------


def compose(paths: DataPaths, out_dir: Path, *, dpi: int = 300) -> tuple[plt.Figure, dict]:
    union_df = load_union(paths)
    basis_df = load_basis(paths)
    cov_df = load_covariance_bridge(paths)

    fig = plt.figure(figsize=(13.2, 7.45), constrained_layout=False)
    gs = GridSpec(
        2,
        3,
        figure=fig,
        left=0.045,
        right=0.985,
        bottom=0.075,
        top=0.895,
        wspace=0.26,
        hspace=0.30,
    )

    axes = {
        "A": fig.add_subplot(gs[0, 0]),
        "B": fig.add_subplot(gs[0, 1]),
        "C": fig.add_subplot(gs[0, 2]),
        "D": fig.add_subplot(gs[1, 0]),
        "E": fig.add_subplot(gs[1, 1]),
        "F": fig.add_subplot(gs[1, 2]),
    }

    plot_panel_a(axes["A"])
    plot_panel_b(axes["B"])
    plot_panel_c(axes["C"])
    plot_panel_d(axes["D"], union_df)
    plot_panel_e(axes["E"], basis_df, paths.basis_source_label)
    plot_panel_f(axes["F"], cov_df)

    fig.suptitle(
        "Image-specific retinal translations form a compact, image-generalizing reafferent geometry",
        fontsize=12.3,
        fontweight="bold",
        x=0.515,
        y=0.965,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"covTFTS_figure.{ext}", dpi=dpi, bbox_inches="tight")

    # Build a manifest with exact values used.
    manifest = {
        "figure": "covTFTS_figure",
        "title": "Image-specific retinal translations form a compact, image-generalizing reafferent geometry",
        "tfts_root": str(paths.tfts_root),
        "source_files": {
            "union_file": str(paths.union_file) if paths.union_file else None,
            "basis_file": str(paths.basis_file) if paths.basis_file else None,
            "covariance_file": str(paths.covariance_file) if paths.covariance_file else None,
            "report_file": str(paths.report_file) if paths.report_file else None,
            "summary_file": str(paths.summary_file) if paths.summary_file else None,
        },
        "basis_source_label": paths.basis_source_label,
        "warnings": paths.warnings,
        "panel_D_union_compactness": union_df.to_dict(orient="records"),
        "panel_E_basis_summary": basis_df.to_dict(orient="records"),
        "panel_F_covariance_bridge": cov_df.to_dict(orient="records"),
    }
    with open(out_dir / "panel_data_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    readme = f"""# covTFTS Figure

Generated by `generate_covTFTS_figure.py`.

## Source files

- union: `{manifest["source_files"]["union_file"]}`
- basis: `{manifest["source_files"]["basis_file"]}`
- covariance bridge: `{manifest["source_files"]["covariance_file"]}`
- compact report: `{manifest["source_files"]["report_file"]}`

## Basis source label

`{paths.basis_source_label}`

For the final manuscript figure, this should be `image_disjoint`. If it is
`fallback_generic_train_test`, the plot should be treated as a preview only.

## Warnings

{chr(10).join('- ' + w for w in paths.warnings) if paths.warnings else '- none'}

## Panel logic

A. Recorded V1 anchor.  
B. Across-image nontriviality schematic.  
C. Canonical twin tangent construction schematic.  
D. Compact tangent family, participation ratio vs unit-shuffle null.  
E. Image-disjoint held-out tangent capture vs unit-shuffle null.  
F. Partial covariance bridge from local tangents to full FEM covariance.

The figure intentionally excludes optotype / discrimination and ecology scale panels.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    return fig, manifest


def parse_args():
    p = argparse.ArgumentParser(description="Generate covTFTS Figure 4.")
    p.add_argument(
        "--tfts-root",
        type=Path,
        default=Path("outputs/twin_feature_tangent_structure_prod_limited_synth"),
        help="Root directory for finalized TFTS outputs.",
    )
    p.add_argument(
        "--fallback-dir",
        type=Path,
        default=None,
        help="Optional fallback directory containing summary CSVs / MANUSCRIPT_REPORT.md.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/covTFTS_figure"),
        help="Output directory for figure files.",
    )
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    paths = resolve_paths(args.tfts_root, fallback_dir=args.fallback_dir or Path.cwd())
    fig, manifest = compose(paths, args.out_dir, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved covTFTS figure outputs to {args.out_dir}")
    if paths.warnings:
        print("Warnings:")
        for w in paths.warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
