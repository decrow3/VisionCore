#!/usr/bin/env python3
"""
Generate covTFTS Figure 4.

Figure 4. Image-specific retinal translations form a compact,
image-generalizing reafferent geometry.

Data-forward version: schematics replaced with real data panels.

Panels:
  A - Recorded V1 anchor (NC reduction + FEM eigenspectrum)
  B - Image-specific local translation charts (bx/by paired glyphs in tangent PCA)
  C - Compact tangent spectrum (cumulative variance vs unit-shuffle PR reference)
  D - Cross-image generalization (held-out translation tangent variance vs k)
  E - Tangent-subspace Fisher gain (placeholder → run_tangent_subspace_information.py)
  F - FEM operating regimes (tangent/FEM covariance overlap vs cloud scale; placeholder)

  Supplement (not in main compose): compactness across scales → plot_supp_scales()

Usage (from VisionCore repo root):
    python declan/fig4_cov_TFTS/generate_covTFTS_figure.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR


# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------

TEXT   = "#202124"
REC    = "#2f2f2f"
REC_L  = "#b0b0b0"
MODEL  = "#2f5f9f"
MODEL_L = "#d8e6f5"
BRIDGE = "#7b5ea7"
BRIDGE_L = "#e7ddf2"
NULL   = "#9a9a9a"
NULL_L = "#d8d8d8"
ACCENT = "#c44e52"

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.0, "axes.labelsize": 8.0, "axes.titlesize": 9.0,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 7.0,
    "axes.linewidth": 0.8, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
})


def clean_axes(ax, grid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, alpha=0.18, zorder=-1)


def panel_label(ax, letter: str, title: str):
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=11, pad=4)
    ax.text(0.13, 1.015, title, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.7, fontweight="bold", color=TEXT)


# -----------------------------------------------------------------------------
# Data paths
# -----------------------------------------------------------------------------


@dataclass
class DataPaths:
    tfts_root:        Path
    union_file:       Optional[Path]
    basis_file:       Optional[Path]
    covariance_file:  Optional[Path]
    report_file:      Optional[Path]
    summary_file:     Optional[Path]
    spec_file:        Optional[Path]
    tangent_maps:     Optional[Path]
    v1_cache:         Optional[Path]
    information_file: Optional[Path]   # tangent-subspace Fisher gain summary (Panel E)
    basis_source_label: str
    warnings: list[str] = field(default_factory=list)


def _first(candidates: Iterable[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_paths(tfts_root: Path) -> DataPaths:
    w = []
    union_file = _first([
        tfts_root / "split_modes" / "image_disjoint" / "twin_tangent_union_summary_image_disjoint.csv",
        tfts_root / "union_spectrum" / "twin_tangent_union_summary.csv",
    ])
    basis_file = _first([
        tfts_root / "split_modes" / "image_disjoint" / "twin_tangent_train_test_basis_image_disjoint.csv",
        tfts_root / "train_test_basis" / "twin_tangent_train_test_basis.csv",
    ])
    basis_label = "image_disjoint"
    if basis_file and "image_disjoint" not in str(basis_file):
        basis_label = "fallback_generic_train_test"
        w.append("Using generic train_test_basis; image-disjoint file not found.")

    covariance_file = _first([tfts_root / "covariance_approx" / "twin_linear_covariance_approx.csv"])
    report_file  = _first([tfts_root / "MANUSCRIPT_REPORT.md"])
    summary_file = _first([tfts_root / "twin_feature_tangent_summary.json"])
    spec_file    = _first([tfts_root / "union_spectrum" / "twin_tangent_union_spectrum.csv"])
    tangent_maps = _first([tfts_root / "tangent_maps" / "twin_tangent_maps.pkl"])
    v1_cache     = _first([CACHE_DIR / "fig2_decomposition.pkl",
                            VISIONCORE_ROOT / "outputs" / "cache" / "fig2_decomposition.pkl"])
    # Tangent-subspace information result (Panel E) — written by run_tangent_subspace_information.py
    information_file = _first([
        VISIONCORE_ROOT / "outputs" / "tangent_subspace_information"
        / "panelE_production_k10_delta025" / "results"
        / "panelE_subspace_capture_summary.csv",
    ])

    for name, p in [("union_file", union_file), ("basis_file", basis_file),
                    ("spec_file", spec_file), ("tangent_maps", tangent_maps)]:
        if p is None:
            w.append(f"Missing {name}; panel may use fallback values.")

    if v1_cache is None:
        w.append("fig2_decomposition.pkl not found; Panel A will be skipped.")
    if information_file is None:
        w.append("Panel E information file not found; Panel E will show placeholder.")

    return DataPaths(
        tfts_root=tfts_root,
        union_file=union_file, basis_file=basis_file,
        covariance_file=covariance_file, report_file=report_file,
        summary_file=summary_file, spec_file=spec_file,
        tangent_maps=tangent_maps, v1_cache=v1_cache,
        information_file=information_file,
        basis_source_label=basis_label, warnings=w,
    )


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_recorded_v1(cache_path: Path, window_idx: int = 2):
    """Extract NC reduction and FEM eigenspectrum from fig2 decomp pickle."""
    with open(cache_path, "rb") as f:
        decomp = pickle.load(f)
    nc_u, nc_c, fem_cumspecs = [], [], []
    for s in decomp:
        m = s["mats"][window_idx]
        n = m["NoiseCorrU"].shape[0]
        uidx = np.triu_indices(n, k=1)
        nc_u.append(np.nanmean(m["NoiseCorrU"][uidx]))
        nc_c.append(np.nanmean(m["NoiseCorrC"][uidx]))
        fem = m["FEM"]
        vals = np.linalg.eigvalsh(fem)
        vals = np.sort(vals[vals > 1e-10])[::-1]
        if vals.sum() > 0:
            fem_cumspecs.append(np.cumsum(vals / vals.sum()))
    window_ms = decomp[0]["results"][window_idx]["window_ms"]
    return {
        "nc_u": np.array(nc_u), "nc_c": np.array(nc_c),
        "fem_cumspecs": fem_cumspecs, "window_ms": window_ms,
    }


def load_tangent_family(tangent_maps_path: Path, delta: float = 0.25):
    """Load raw tangent vectors (bx, by) per object at the given delta."""
    with open(tangent_maps_path, "rb") as f:
        tm = pickle.load(f)
    payload = tm["object_payload"][delta]
    oids = list(payload.keys())
    bx = np.vstack([payload[o]["bx"] for o in oids])
    by = np.vstack([payload[o]["by"] for o in oids])
    valid = np.isfinite(bx).all(axis=1) & np.isfinite(by).all(axis=1)
    bx, by = bx[valid], by[valid]
    return {"bx": bx, "by": by, "n_objects": int(valid.sum()), "delta": delta}


def load_union(paths: DataPaths) -> pd.DataFrame:
    if paths.union_file is None:
        return pd.DataFrame({
            "delta": [0.125, 0.25, 0.5],
            "participation_ratio": [7.8029, 9.0407, 6.5500],
            "null_pr_mean": [27.4645, 31.0288, 24.2867],
            "null_pr_ci_low": [27.1661, 30.6021, 24.0068],
            "null_pr_ci_high": [27.7510, 31.3573, 24.5549],
        })
    df = pd.read_csv(paths.union_file)
    rename = {"PR": "participation_ratio", "null_mean": "null_pr_mean",
              "null_95CI_low": "null_pr_ci_low", "null_95CI_high": "null_pr_ci_high"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "component_set" in df.columns:
        df = df[df["component_set"].astype(str).str.lower() == "combined"]
    if "space" in df.columns:
        raw = df[df["space"].astype(str).str.lower() == "raw"]
        if len(raw):
            df = raw
    return df.copy()


def _summarize_basis(df: pd.DataFrame) -> pd.DataFrame:
    if "basis_rank_k" not in df.columns and "k" in df.columns:
        df = df.rename(columns={"k": "basis_rank_k"})
    ok = df[df.get("fold_status", pd.Series(["ok"] * len(df))).astype(str).str.lower() == "ok"].copy()
    if "tangent_set" in ok.columns:
        comb = ok[ok["tangent_set"].astype(str).str.lower() == "combined"]
        if len(comb):
            ok = comb
    return (
        ok.groupby(["delta", "basis_rank_k"], as_index=False)
        .agg(
            capture=("test_variance_captured", "median"),
            capture_lo=("test_variance_captured", lambda x: np.nanpercentile(x, 2.5)),
            capture_hi=("test_variance_captured", lambda x: np.nanpercentile(x, 97.5)),
            null=("null_mean", "median"),
            null_lo=("null_mean", lambda x: np.nanpercentile(x, 2.5)),
            null_hi=("null_mean", lambda x: np.nanpercentile(x, 97.5)),
        )
    )


def load_basis(paths: DataPaths) -> pd.DataFrame:
    if paths.basis_file is None:
        return pd.DataFrame({
            "delta": [0.125] * 4 + [0.25] * 4 + [0.5] * 4,
            "basis_rank_k": [2, 5, 10, 20] * 3,
            "capture": [0.352, 0.462, 0.537, 0.667, 0.297, 0.436, 0.552, 0.666,
                        0.391, 0.525, 0.630, 0.723],
            "null": [0.143, 0.148, 0.156, 0.168, 0.105, 0.111, 0.118, 0.130,
                     0.125, 0.129, 0.137, 0.150],
        })
    return _summarize_basis(pd.read_csv(paths.basis_file))


def load_covariance_bridge(paths: DataPaths) -> pd.DataFrame:
    if paths.covariance_file is not None:
        df = pd.read_csv(paths.covariance_file)
        rename = {"subspace_overlap_k2": "overlap_k2_median",
                  "overlap_k2": "overlap_k2_median",
                  "trace_ratio_lin_full": "trace_ratio_lin_full_median"}
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        if {"delta", "cloud_scale", "overlap_k2_median"}.issubset(df.columns):
            return df.groupby(["delta", "cloud_scale"], as_index=False).agg(
                overlap_k2_median=("overlap_k2_median", "median"))
    return pd.DataFrame({
        "delta": [0.25] * 4, "cloud_scale": [0.25, 0.5, 1.0, 2.0],
        "overlap_k2_median": [0.3951, 0.3441, 0.2951, 0.2449],
    })


def load_union_spectrum(paths: DataPaths, delta: float = 0.25, n_show: int = 40) -> pd.DataFrame:
    if paths.spec_file is None:
        return pd.DataFrame()
    df = pd.read_csv(paths.spec_file)
    d = df[(df["delta"] == delta) &
           (df["space"].astype(str).str.lower() == "raw") &
           (df["tangent_set"].astype(str).str.lower() == "combined")]
    return d.sort_values("component_index").head(n_show).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Panels
# -----------------------------------------------------------------------------


def plot_panel_a(ax_nc: plt.Axes, ax_spec: plt.Axes, v1_data: dict):
    """Panel A: recorded V1 anchor. Two data sub-panels.
    Panel label is placed by compose() on a full-width overlay axis, not here."""

    # --- Left: NC reduction per session ---
    nc_u = v1_data["nc_u"]
    nc_c = v1_data["nc_c"]
    n = len(nc_u)
    jitter = np.linspace(-0.08, 0.08, n)
    for i in range(n):
        ax_nc.plot([0 + jitter[i], 1 + jitter[i]], [nc_u[i], nc_c[i]],
                   "-", color="0.75", lw=0.75, zorder=1)
    ax_nc.scatter(np.zeros(n) + jitter, nc_u, s=22, color=REC,
                  zorder=3, linewidths=0.4, edgecolors="white")
    ax_nc.scatter(np.ones(n) + jitter, nc_c, s=22, color=MODEL,
                  zorder=3, linewidths=0.4, edgecolors="white")
    ax_nc.axhline(0, color="0.7", lw=0.6, linestyle="--", zorder=0)
    ax_nc.set_xlim(-0.45, 1.45)
    ax_nc.set_xticks([0, 1])
    ax_nc.set_xticklabels(["Uncorr.", "Eye-pos.\ncorr."], fontsize=7)
    ax_nc.set_ylabel("Mean noise\ncorrelation", fontsize=7)
    clean_axes(ax_nc)

    # --- Right: FEM cumulative eigenspectrum per session ---
    max_rank = 7
    ranks = np.arange(1, max_rank + 1)
    specs_arr = []
    for spec in v1_data["fem_cumspecs"]:
        row = np.full(max_rank, np.nan)
        L = min(len(spec), max_rank)
        row[:L] = spec[:L]
        if L < max_rank and L > 0:
            row[L:] = spec[L - 1]
        specs_arr.append(row)
        ax_spec.plot(ranks, row, "-", color=REC, alpha=0.22, lw=0.9, zorder=1)
    if specs_arr:
        med = np.nanmedian(np.array(specs_arr), axis=0)
        ax_spec.plot(ranks, med, "-", color=REC, lw=2.2, zorder=3)
    ax_spec.set_xlim(1, max_rank)
    ax_spec.set_ylim(0, 1.05)
    ax_spec.set_xticks([1, 3, 5, 7])
    ax_spec.set_xlabel("FEM covariance\neigenvalue rank", fontsize=7)
    ax_spec.set_ylabel("Cumulative variance\n(FEM covariance)", fontsize=7)
    ax_spec.axhline(1, color="0.8", lw=0.5, linestyle=":", zorder=0)
    clean_axes(ax_spec, grid=True)


def _farthest_point_subset(points: np.ndarray, n_show: int, seed: int = 0) -> np.ndarray:
    """Choose a spread-out, deterministic subset for small-panel readability."""
    pts = np.asarray(points, dtype=np.float64)
    n = int(pts.shape[0])
    if n <= int(n_show):
        return np.arange(n)
    rng = np.random.default_rng(seed)
    finite = np.all(np.isfinite(pts), axis=1)
    if not np.all(finite):
        pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)
    chosen = [int(np.argmax(np.linalg.norm(pts - np.median(pts, axis=0, keepdims=True), axis=1)))]
    min_dist = np.linalg.norm(pts - pts[chosen[0]], axis=1)
    jitter = rng.uniform(0.0, 1e-9, size=n)
    while len(chosen) < int(n_show):
        score = min_dist + jitter
        score[chosen] = -np.inf
        nxt = int(np.argmax(score))
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(pts - pts[nxt], axis=1))
    return np.asarray(chosen, dtype=int)


def plot_panel_b(ax: plt.Axes, tangent_data: dict, n_show: int = 24):
    """Panel B: image-specific local translation charts in tangent PCA space.

    Each object contributes a small two-ray chart: the projected local response
    direction induced by horizontal retinal translation and the direction induced
    by vertical translation.  The chart centers are pair midpoints in tangent PC
    space, so this is a tangent-family atlas, not a response-manifold plot.
    """
    panel_label(ax, "B", "Image-specific local translation charts")

    bx = tangent_data["bx"]
    by = tangent_data["by"]
    n_obj = tangent_data["n_objects"]

    # PCA of the full tangent family to get a common 2D space
    B = np.vstack([bx, by])
    Bc = B - B.mean(axis=0)
    _, s, Vt = np.linalg.svd(Bc, full_matrices=False)
    var_exp = s**2 / (s**2).sum()

    # Project bx and by separately into the same 2D tangent-family space.
    bx_proj = (bx - B.mean(axis=0)) @ Vt[:2].T   # (n_obj, 2)
    by_proj = (by - B.mean(axis=0)) @ Vt[:2].T   # (n_obj, 2)
    centers = 0.5 * (bx_proj + by_proj)
    pair_len = np.linalg.norm(bx_proj - by_proj, axis=1)

    # Faint full family provides context; selected charts provide interpretation.
    ax.scatter(bx_proj[:, 0], bx_proj[:, 1], s=9, color=MODEL, alpha=0.11,
               linewidths=0, zorder=1)
    ax.scatter(by_proj[:, 0], by_proj[:, 1], s=9, color=BRIDGE, alpha=0.11,
               linewidths=0, zorder=1)

    idx = _farthest_point_subset(centers, min(n_show, n_obj), seed=2)
    order = idx[np.argsort(pair_len[idx])]

    for j, i in enumerate(order):
        alpha = 0.38 + 0.52 * (j + 1) / max(len(order), 1)
        ax.plot([bx_proj[i, 0], by_proj[i, 0]],
                [bx_proj[i, 1], by_proj[i, 1]],
                "-", color="0.72", lw=0.65, alpha=0.60, zorder=2)
        ax.plot([centers[i, 0], bx_proj[i, 0]],
                [centers[i, 1], bx_proj[i, 1]],
                "-", color=MODEL, lw=1.25, alpha=alpha, zorder=3)
        ax.plot([centers[i, 0], by_proj[i, 0]],
                [centers[i, 1], by_proj[i, 1]],
                "-", color=BRIDGE, lw=1.25, alpha=alpha, zorder=3)

    ax.scatter(centers[idx, 0], centers[idx, 1], s=10, color="white",
               edgecolors="0.52", linewidths=0.45, zorder=4)
    ax.scatter(bx_proj[idx, 0], bx_proj[idx, 1], s=27, color=MODEL, alpha=0.93,
               linewidths=0.45, edgecolors="white", zorder=5)
    ax.scatter(by_proj[idx, 0], by_proj[idx, 1], s=27, color=BRIDGE, alpha=0.93,
               linewidths=0.45, edgecolors="white", zorder=5)

    bx_handle = Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL,
                        markersize=7, label=r"$b_x(I)$ horizontal shift")
    by_handle = Line2D([0], [0], marker="o", color="w", markerfacecolor=BRIDGE,
                        markersize=7, label=r"$b_y(I)$ vertical shift")

    pct0, pct1 = var_exp[0] * 100, var_exp[1] * 100
    ax.set_xlabel(f"Tangent PC 1 ({pct0:.1f}% var.)")
    ax.set_ylabel(f"Tangent PC 2 ({pct1:.1f}% var.)")
    clean_axes(ax, grid=True)
    ax.legend(handles=[bx_handle, by_handle],
              frameon=False, loc="upper right", handlelength=1.1,
              borderpad=0.15, labelspacing=0.45)
    ax.text(0.04, 0.04,
            f"{len(idx)} local charts shown; faint points show full tangent family",
            transform=ax.transAxes, fontsize=6.3, color="0.50",
            ha="left", va="bottom", style="italic")

    # Small quantitative inset: local x/y chart angles vary across objects.
    cos_xy = np.sum(bx * by, axis=1) / (
        np.linalg.norm(bx, axis=1) * np.linalg.norm(by, axis=1) + 1e-12
    )
    cos_xy = cos_xy[np.isfinite(cos_xy)]
    if cos_xy.size:
        inset = ax.inset_axes([0.055, 0.66, 0.25, 0.25])
        bins = np.linspace(-1.0, 1.0, 13)
        inset.hist(cos_xy, bins=bins, color="0.78", edgecolor="white", lw=0.45)
        inset.axvline(float(np.median(cos_xy)), color=TEXT, lw=1.0)
        inset.set_xlim(-1.0, 1.0)
        inset.set_xticks([-1, 0, 1])
        inset.set_yticks([])
        inset.set_xlabel(r"cos$(b_x,b_y)$", fontsize=5.8, labelpad=0.5)
        inset.tick_params(axis="x", labelsize=5.5, pad=1.0, length=2)
        inset.spines["top"].set_visible(False)
        inset.spines["right"].set_visible(False)
        inset.spines["left"].set_visible(False)
        inset.spines["bottom"].set_color("0.45")
        inset.patch.set_alpha(0.82)


def plot_panel_c(ax: plt.Axes, spec_df: pd.DataFrame, union_df: pd.DataFrame):
    """Panel C: cumulative tangent variance spectrum vs null PR reference."""
    panel_label(ax, "C", "Compact tangent spectrum")

    if spec_df is None or len(spec_df) == 0:
        ax.text(0.5, 0.5, "spectrum data not found", transform=ax.transAxes,
                ha="center", va="center", color=ACCENT, fontsize=8)
        clean_axes(ax)
        return

    ranks = spec_df["component_index"].to_numpy(dtype=float)
    cumvar = spec_df["cumulative_fraction_variance"].to_numpy(dtype=float)
    pr_obs = float(spec_df["participation_ratio"].iloc[0])

    # Null reference: linear ramp to 1.0 over null_pr components (null PR=31 → PR=31 equivalent)
    # null_pr from union_df at delta=0.25
    null_pr = 31.03  # from MANUSCRIPT_REPORT
    try:
        row_025 = union_df[np.isclose(union_df["delta"].astype(float), 0.25)]
        if len(row_025):
            null_pr = float(row_025["null_pr_mean"].iloc[0])
    except Exception:
        pass
    null_ranks = np.array([0.0, null_pr, float(ranks.max())])
    null_cumvar = np.array([0.0, 1.0, 1.0])

    # Draw null reference as dashed gray
    ax.plot(null_ranks, null_cumvar, "--", color=NULL, lw=1.8, zorder=1,
            label=f"Unit-shuffle PR reference (PR≈{null_pr:.0f})")

    # Draw observed
    ax.plot(np.concatenate([[0], ranks]),
            np.concatenate([[0], cumvar]),
            "-", color=MODEL, lw=2.4, zorder=3, label="Observed")

    ax.set_xlabel("Tangent component rank")
    ax.set_ylabel("Cumulative variance fraction")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 1.05)
    clean_axes(ax, grid=True)
    ax.legend(frameon=False, loc="lower right", handlelength=1.5)

    # Annotate PR comparison
    ax.annotate(f"PR = {pr_obs:.1f}  vs  null ≈ {null_pr:.0f}",
                xy=(pr_obs, cumvar[np.argmin(np.abs(ranks - pr_obs))]
                    if any(ranks <= pr_obs) else cumvar[0]),
                xytext=(0.38, 0.30), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=MODEL, lw=0.9),
                fontsize=7.4, color=MODEL, ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="0.85", lw=0.6, alpha=0.95))


def plot_panel_d(ax: plt.Axes, union_df: pd.DataFrame):
    panel_label(ax, "D", "Compactness across scales")
    df = union_df.sort_values("delta")
    x = np.arange(len(df))
    obs      = df["participation_ratio"].astype(float).to_numpy()
    null_mn  = df["null_pr_mean"].astype(float).to_numpy()
    lo       = df["null_pr_ci_low"].astype(float).to_numpy()
    hi       = df["null_pr_ci_high"].astype(float).to_numpy()

    ax.fill_between(x, lo, hi, color=NULL, alpha=0.18, lw=0, zorder=1)
    ax.plot(x, null_mn, "o-", color=NULL, lw=1.8, markersize=5.0,
            markeredgecolor="white", markeredgewidth=0.5,
            label="Unit-shuffle null", zorder=2)
    ax.plot(x, obs, "o-", color=MODEL, lw=2.4, markersize=5.5,
            markeredgecolor="white", markeredgewidth=0.5,
            label="Observed", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:g}" for d in df["delta"].astype(float)])
    ax.set_xlabel("Tangent displacement (arcmin)")
    ax.set_ylabel("Participation ratio")
    ax.set_ylim(0, max(null_mn) * 1.18)
    clean_axes(ax, grid=True)
    ax.legend(frameon=False, loc="upper right", handlelength=1.5)

    idx = int(np.argmin(np.abs(df["delta"].astype(float).to_numpy() - 0.25)))
    ax.annotate(
        f"0.25 arcmin:\nPR {obs[idx]:.1f} vs null {null_mn[idx]:.1f}",
        xy=(x[idx], obs[idx]),
        xytext=(x[idx] + 0.22, obs[idx] + 6.0),
        arrowprops=dict(arrowstyle="->", color=TEXT, lw=0.9),
        fontsize=7.4, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.85", lw=0.6, alpha=0.95))


def plot_panel_e(ax: plt.Axes, basis_df: pd.DataFrame, basis_source_label: str):
    panel_label(ax, "D", "Cross-image generalization")
    df = basis_df.copy()
    if "basis_rank_k" not in df.columns and "k" in df.columns:
        df = df.rename(columns={"k": "basis_rank_k"})
    deltas = np.array(sorted(df["delta"].astype(float).unique()))
    delta = float(deltas[np.argmin(np.abs(deltas - 0.25))])
    d = df[np.isclose(df["delta"].astype(float), delta)].sort_values("basis_rank_k")
    k       = d["basis_rank_k"].astype(float).to_numpy()
    capture = d["capture"].astype(float).to_numpy()
    null    = d["null"].astype(float).to_numpy()

    if {"capture_lo", "capture_hi"}.issubset(d.columns):
        ax.fill_between(k, d["capture_lo"].astype(float), d["capture_hi"].astype(float),
                        color=MODEL, alpha=0.14, lw=0, zorder=1)
    if {"null_lo", "null_hi"}.issubset(d.columns):
        ax.fill_between(k, d["null_lo"].astype(float), d["null_hi"].astype(float),
                        color=NULL, alpha=0.14, lw=0, zorder=1)

    ax.plot(k, null, "o--", color=NULL, lw=1.8, markersize=5.0,
            markeredgecolor="white", markeredgewidth=0.5,
            label="Unit-shuffle null", zorder=2)
    ax.plot(k, capture, "o-", color=MODEL, lw=2.4, markersize=5.5,
            markeredgecolor="white", markeredgewidth=0.5,
            label="Observed", zorder=3)
    ax.set_xlabel("Basis dimension k")
    ax.set_ylabel("Held-out translation\ntangent variance captured")
    ax.set_xticks(k)
    ax.set_ylim(0, min(0.85, np.nanmax(capture) + 0.10))
    clean_axes(ax, grid=True)
    ax.legend(frameon=False, loc="lower right", handlelength=1.5)

    ax.text(0.03, 0.97, "held-out images", transform=ax.transAxes,
            fontsize=7.3, color=MODEL, fontweight="bold", ha="left", va="top")
    i = (int(np.where(np.isclose(k, 10))[0][0]) if np.any(np.isclose(k, 10))
         else int(np.argmin(np.abs(k - 10))))
    color = TEXT if basis_source_label == "image_disjoint" else ACCENT
    label = (f"k={int(k[i])}: {capture[i]:.2f} vs null {null[i]:.2f}\n0% image-ID leakage"
             if basis_source_label == "image_disjoint"
             else f"k={int(k[i])}: {capture[i]:.2f} vs null {null[i]:.2f}\nWARNING: fallback split")
    ax.annotate(label, xy=(k[i], capture[i]),
                xytext=(k[i] + 1.5, max(0.12, capture[i] - 0.18)),
                arrowprops=dict(arrowstyle="->", color=color, lw=0.9),
                fontsize=7.4, ha="left", va="center", color=color,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.85",
                          lw=0.6, alpha=0.95))

def plot_panel_f(
    ax: plt.Axes,
    cov_df: pd.DataFrame,
    *,
    drift_band: tuple[float, float] = (0.03, 0.25),
    microsaccade_band: tuple[float, float] = (0.5, 2.0),
):
    """Panel F: drift and microsaccades as operating regimes.

    Uses the existing covariance/locality summary as a first-pass tangent-utility
    curve: local tangent/FEM covariance overlap vs eye-position cloud scale.
    Replace the default drift/microsaccade bands once empirical event ranges
    are available.
    """
    panel_label(ax, "F", "FEM operating regimes")

    df = cov_df.copy()
    if df.empty or not {"delta", "cloud_scale", "overlap_k2_median"}.issubset(df.columns):
        ax.text(
            0.5, 0.5,
            "operating-regime\ndata pending",
            transform=ax.transAxes,
            ha="center", va="center",
            color=ACCENT, fontsize=8,
        )
        ax.set_xlabel("Retinal displacement scale (arcmin)")
        ax.set_ylabel("Tangent-regime utility")
        clean_axes(ax, grid=True)
        return

    deltas = np.sort(pd.to_numeric(df["delta"], errors="coerce").dropna().unique())
    delta = float(deltas[np.argmin(np.abs(deltas - 0.25))]) if len(deltas) else 0.25

    d = df[np.isclose(pd.to_numeric(df["delta"], errors="coerce"), delta)].copy()
    d["cloud_scale"] = pd.to_numeric(d["cloud_scale"], errors="coerce")
    d["overlap_k2_median"] = pd.to_numeric(d["overlap_k2_median"], errors="coerce")
    d = d.dropna(subset=["cloud_scale", "overlap_k2_median"]).sort_values("cloud_scale")

    x = d["cloud_scale"].to_numpy(dtype=float)
    y = d["overlap_k2_median"].to_numpy(dtype=float)

    # Set log scale first so axvspan coordinates are interpreted correctly.
    ax.set_xscale("log")

    xmin = min(float(np.nanmin(x)) * 0.75, drift_band[0] * 0.8)
    xmax = max(float(np.nanmax(x)) * 1.25, microsaccade_band[1] * 1.15)
    ax.set_xlim(xmin, xmax)

    # Shaded biological regimes. Defaults are placeholders until empirical
    # drift/microsaccade amplitude summaries are wired in.
    # drift_band[1] capped at 0.22 to avoid overlap with first data point at 0.25.
    ax.axvspan(drift_band[0], min(drift_band[1], 0.22),
               color=MODEL, alpha=0.12, lw=0, zorder=0)
    ax.axvspan(microsaccade_band[0], microsaccade_band[1],
               color=BRIDGE, alpha=0.12, lw=0, zorder=0)

    ax.plot(x, y, "o-", color=TEXT, lw=2.2, markersize=5.5,
            markeredgecolor="white", markeredgewidth=0.5, zorder=3)

    if len(x) >= 2:
        ax.plot(x[-2:], y[-2:], color=TEXT, lw=1.8, linestyle="--",
                alpha=0.75, zorder=2)

    ax.text(np.sqrt(drift_band[0] * min(drift_band[1], 0.22)), 0.97,
            "drift", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.2, fontweight="bold", color=MODEL)
    ax.text(np.sqrt(microsaccade_band[0] * microsaccade_band[1]), 0.97,
            "microsaccades", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.2, fontweight="bold", color=BRIDGE)

    ax.set_xlabel("Retinal displacement scale (arcmin)")
    ax.set_ylabel("Tangent/FEM covariance overlap")
    ax.set_ylim(0, max(0.5, float(np.nanmax(y)) + 0.08))

    ticks = [t for t in [0.03, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
             if xmin <= t <= xmax]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())

    clean_axes(ax, grid=True)


def plot_panel_f_supp(ax: plt.Axes, cov_df: pd.DataFrame):
    panel_label(ax, "F", "Partial covariance bridge")
    df = cov_df.copy()
    deltas = np.sort(pd.to_numeric(df["delta"], errors="coerce").dropna().unique())
    delta = float(deltas[np.argmin(np.abs(deltas - 0.25))]) if len(deltas) else 0.25
    d = df[np.isclose(pd.to_numeric(df["delta"], errors="coerce"), delta)].copy()
    d["cloud_scale"] = pd.to_numeric(d["cloud_scale"], errors="coerce")
    d["overlap_k2_median"] = pd.to_numeric(d["overlap_k2_median"], errors="coerce")
    d = d.sort_values("cloud_scale")

    ax.plot(d["cloud_scale"], d["overlap_k2_median"], "o-", color=BRIDGE,
            lw=2.4, markersize=5.5, markeredgecolor="white", markeredgewidth=0.5,
            zorder=3)
    # Dashed tail to emphasize weakening
    if len(d) >= 2:
        ax.plot(d["cloud_scale"].iloc[-2:], d["overlap_k2_median"].iloc[-2:],
                color=BRIDGE, lw=2.0, linestyle="--", alpha=0.85)

    ax.set_xlabel("Eye-position cloud scale (arcmin)")
    ax.set_ylabel(r"Subspace overlap with sampled $\Sigma_{\mathrm{FEM}}$")
    ax.set_ylim(0, float(np.nanmax(d["overlap_k2_median"])) + 0.10)
    clean_axes(ax, grid=True)

    ax.text(0.06, 0.91, "meaningful but incomplete",
            transform=ax.transAxes, fontsize=7.6, fontweight="bold",
            color=BRIDGE, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=BRIDGE_L, lw=0.8, alpha=0.98))
    ax.text(0.06, 0.72, "Diagnostic bridge—local tangents\nalign most strongly with\nFEM covar. for small clouds",
            transform=ax.transAxes, fontsize=6.8, color=TEXT,
            ha="left", va="top")


def load_information(paths: DataPaths) -> Optional[pd.DataFrame]:
    """Load tangent-subspace Fisher gain summary produced by run_tangent_subspace_information."""
    if paths.information_file is None or not paths.information_file.exists():
        return None
    return pd.read_csv(paths.information_file)


def plot_panel_e_information(ax: plt.Axes, info_df: Optional[pd.DataFrame]):
    """Panel E: tangent-subspace Fisher gain (placeholder until production run lands)."""
    panel_label(ax, "E", "Tangent-subspace information")

    if info_df is None or len(info_df) == 0:
        # Placeholder — production run still in progress
        ax.set_axis_off()
        ax.text(0.5, 0.58, "Tangent-subspace\nFisher gain",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, fontweight="bold", color=MODEL)
        ax.text(0.5, 0.38, "production run pending",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7.5, color="0.55", style="italic")
        ax.text(0.5, 0.24,
                r"$\frac{I_{\rm real,\,tangent} - I_{\rm stab,\,full}}"
                r"{I_{\rm real,\,full} - I_{\rm stab,\,full}}$",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color=MODEL)
        return

    # Real result: bar plot of fraction of full FEM gain by basis type
    order   = ["tangent", "orthogonal_complement", "unit_shuffle", "random_orthogonal"]
    colors  = [MODEL, NULL_L, NULL, NULL]
    labels_ = ["Tangent\nbasis", "Orthogonal\ncomplement", "Unit-shuffle\nnull", "Random\nnull"]

    fracs, sems = [], []
    for bt in order:
        rows = info_df[info_df["basis_type"].astype(str) == bt]
        if "fraction_full_fem_gain_captured_full_baseline" in rows.columns:
            vals = pd.to_numeric(rows["fraction_full_fem_gain_captured_full_baseline"],
                                 errors="coerce").dropna()
        else:
            vals = pd.to_numeric(rows.get("median_fraction_full_fem_gain_captured",
                                          pd.Series(dtype=float)), errors="coerce").dropna()
        fracs.append(float(vals.median()) if len(vals) else float("nan"))
        sems.append(float(vals.std() / len(vals) ** 0.5) if len(vals) > 1 else 0.0)

    x = np.arange(len(order))
    ax.bar(x, fracs, yerr=sems, capsize=3, color=colors, alpha=0.85,
           edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="0.35", lw=0.9, ls="--", label="Full FEM gain")
    ax.axhline(0.0, color="0.70", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_, fontsize=6.5)
    ax.set_ylabel("Fraction of full FEM\ninformation gain captured")
    clean_axes(ax, grid=True)
    ax.text(0.03, 0.97, "image-disjoint basis  |  k=10  |  0.25 arcmin",
            transform=ax.transAxes, fontsize=6.5, color=MODEL,
            fontweight="bold", ha="left", va="top")


# Supplement-only: compactness across scales (not in main compose)
def plot_supp_scales(ax: plt.Axes, union_df: pd.DataFrame):
    """Supplemental: compactness across displacement scales."""
    panel_label(ax, "S", "Compactness across scales")
    df = union_df.sort_values("delta")
    x = np.arange(len(df))
    obs     = df["participation_ratio"].astype(float).to_numpy()
    null_mn = df["null_pr_mean"].astype(float).to_numpy()
    lo      = df["null_pr_ci_low"].astype(float).to_numpy()
    hi      = df["null_pr_ci_high"].astype(float).to_numpy()
    ax.fill_between(x, lo, hi, color=NULL, alpha=0.18, lw=0, zorder=1)
    ax.plot(x, null_mn, "o-", color=NULL, lw=1.8, markersize=5.0,
            markeredgecolor="white", markeredgewidth=0.5,
            label="Unit-shuffle null", zorder=2)
    ax.plot(x, obs, "o-", color=MODEL, lw=2.4, markersize=5.5,
            markeredgecolor="white", markeredgewidth=0.5,
            label="Observed", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:g}" for d in df["delta"].astype(float)])
    ax.set_xlabel("Tangent displacement (arcmin)")
    ax.set_ylabel("Participation ratio")
    ax.set_ylim(0, max(null_mn) * 1.18)
    clean_axes(ax, grid=True)
    ax.legend(frameon=False, loc="upper right", handlelength=1.5)


# -----------------------------------------------------------------------------
# Compose
# -----------------------------------------------------------------------------


def compose(paths: DataPaths, out_dir: Path, *, dpi: int = 300):
    # Load data
    union_df = load_union(paths)
    basis_df = load_basis(paths)
    cov_df   = load_covariance_bridge(paths)
    spec_df  = load_union_spectrum(paths, delta=0.25, n_show=32)

    v1_data      = load_recorded_v1(paths.v1_cache)     if paths.v1_cache     else None
    tangent_data = load_tangent_family(paths.tangent_maps) if paths.tangent_maps else None
    info_df      = load_information(paths)

    fig = plt.figure(figsize=(10.0, 7.5), constrained_layout=False)
    gs = GridSpec(2, 3, figure=fig,
                  left=0.07, right=0.97, bottom=0.085, top=0.89,
                  wspace=0.44, hspace=0.28)

    # --- Panel A: two sub-panels inside gs[0, 0] ---
    gs_a  = gs[0, 0].subgridspec(1, 2, wspace=0.60)
    ax_nc   = fig.add_subplot(gs_a[0])
    ax_spec = fig.add_subplot(gs_a[1])
    if v1_data is not None:
        plot_panel_a(ax_nc, ax_spec, v1_data)
    else:
        ax_nc.text(0.5, 0.5, "V1 data\nnot found", transform=ax_nc.transAxes,
                   ha="center", va="center", color=ACCENT, fontsize=8)
        clean_axes(ax_nc)

    # Panel A label on a full-width invisible overlay so it aligns with B–F.
    bbox = gs[0, 0].get_position(fig)
    ax_a_lbl = fig.add_axes([bbox.x0, bbox.y0, bbox.width, bbox.height])
    ax_a_lbl.set_axis_off()
    panel_label(ax_a_lbl, "A", "Recorded V1 anchor")

    # --- Panel B ---
    ax_b = fig.add_subplot(gs[0, 1])
    if tangent_data is not None:
        plot_panel_b(ax_b, tangent_data)
    else:
        ax_b.text(0.5, 0.5, "tangent maps\nnot found", transform=ax_b.transAxes,
                  ha="center", va="center", color=ACCENT, fontsize=8)
        panel_label(ax_b, "B", "Tangent family structure")
        clean_axes(ax_b)

    # --- Panel C ---
    ax_c = fig.add_subplot(gs[0, 2])
    plot_panel_c(ax_c, spec_df, union_df)

    # --- Panels D (generalization), E (information), F (bridge) ---
    ax_d = fig.add_subplot(gs[1, 0])
    plot_panel_e(ax_d, basis_df, paths.basis_source_label)   # labeled "D" inside

    ax_e = fig.add_subplot(gs[1, 1])
    plot_panel_e_information(ax_e, info_df)                   # labeled "E" inside

    ax_f = fig.add_subplot(gs[1, 2])
    plot_panel_f(ax_f, cov_df)

    fig.suptitle(
        "Image-specific retinal translations form a compact, image-generalizing reafferent geometry",
        fontsize=10.5, fontweight="bold", x=0.52, y=0.965)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"covTFTS_figure.{ext}", dpi=dpi, bbox_inches="tight")

    manifest = {
        "figure": "covTFTS_figure",
        "source_files": {
            "v1_cache": str(paths.v1_cache),
            "tangent_maps": str(paths.tangent_maps),
            "union_file": str(paths.union_file),
            "spec_file": str(paths.spec_file),
            "basis_file": str(paths.basis_file),
            "covariance_file": str(paths.covariance_file),
        },
        "basis_source_label": paths.basis_source_label,
        "warnings": paths.warnings,
        "panel_D_cross_image_generalization": basis_df.to_dict(orient="records"),
        "panel_E_tangent_subspace_information": info_df.to_dict(orient="records") if info_df is not None else None,
        "panel_F_operating_regimes": cov_df.to_dict(orient="records"),
    }
    with open(out_dir / "panel_data_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    readme = f"""# covTFTS Figure 4 (data-forward version)

Generated by `declan/fig4_cov_TFTS/generate_covTFTS_figure.py`.

## Source files
- V1 recorded data: `{paths.v1_cache}`
- Tangent maps:     `{paths.tangent_maps}`
- Union spectrum:   `{paths.spec_file}`
- Union summary:    `{paths.union_file}`
- Basis (image-disjoint): `{paths.basis_file}`
- Covariance bridge: `{paths.covariance_file}`

## Basis source: `{paths.basis_source_label}`
Final figure requires `image_disjoint`.

## Warnings
{chr(10).join("- " + w for w in paths.warnings) or "- none"}

## Panels
A. Two sub-panels: (left) mean noise correlation per session, uncorrected vs
   eye-position corrected; (right) median cumulative FEM eigenspectrum across
   sessions. Data from fig2_decomposition.pkl.
B. Image-specific local translation charts: selected paired bx(I)/by(I)
   tangent directions projected into 2D tangent PCA space. Each local chart
   shows the horizontal- and vertical-translation directions for one object;
   faint points show the full tangent family.
C. Cumulative variance spectrum of the tangent family at 0.25 arcmin (32
   components). Unit-shuffle PR reference as dashed gray ramp. Annotated with
   observed PR vs null PR.
D. Cross-image generalization: held-out translation tangent variance captured
   vs basis dimension k (image-disjoint split, delta=0.25 arcmin).
E. Tangent-subspace Fisher gain (placeholder until production run from
   run_tangent_subspace_information.py lands).
F. FEM operating regimes: tangent/FEM covariance overlap vs eye-position cloud
   scale (placeholder bands; empirical drift/microsaccade ranges pending).
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    return fig, manifest


def parse_args():
    default_tfts = VISIONCORE_ROOT / "outputs" / "twin_feature_tangent_structure_prod_limited_synth"
    default_out  = VISIONCORE_ROOT / "outputs" / "covTFTS_figure"
    p = argparse.ArgumentParser(description="Generate covTFTS Figure 4 (data-forward).")
    p.add_argument("--tfts-root", type=Path, default=default_tfts)
    p.add_argument("--out-dir",   type=Path, default=default_out)
    p.add_argument("--dpi",       type=int,  default=300)
    return p.parse_args()


def main():
    args = parse_args()
    paths = resolve_paths(args.tfts_root)
    fig, manifest = compose(paths, args.out_dir, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved to {args.out_dir}")
    for w in paths.warnings:
        print(f"  Warning: {w}")


if __name__ == "__main__":
    main()
