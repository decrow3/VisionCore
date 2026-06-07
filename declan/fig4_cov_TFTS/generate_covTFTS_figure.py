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
  F - Matched finite-difference closure to recorded FEM covariance

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
    information_file:            Optional[Path]  # capture summary (Panel E)
    information_null_file:       Optional[Path]  # basis_null_summary (Panel E preferred)
    null_spectrum_summary_file:  Optional[Path]  # per-component null band (Panel C)
    panel_f_file:                Optional[Path]  # covariance overlap summary (Panel F)
    panel_f_natural_file:        Optional[Path]  # natural-structure Panel F diagnostic
    panel_f_closure_summary_file: Optional[Path] # finite-difference closure headline rows
    panel_f_closure_metrics_file: Optional[Path] # finite-difference per-session metric rows
    panel_f_closure_audit_file:   Optional[Path] # finite-difference provenance audit
    panel_f_compact_closure_summary_file: Optional[Path] # compact-k10 closure companion
    panel_f_metric:              str             # which metric is in panel_f_file
    panel_f_fem_ranges:          Optional[dict]  # empirical drift/microsaccade bands
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

    covariance_file  = _first([tfts_root / "covariance_approx" / "twin_linear_covariance_approx.csv"])
    _panelf_root     = VISIONCORE_ROOT / "outputs" / "panel_f_covariance_overlap"
    # Prefer the most informative metric; fall back in order.
    _metric_priority = ("fisher_r2", "variance_fraction", "covariance_overlap")
    panel_f_file:   Optional[Path] = None
    panel_f_metric: str            = "covariance_overlap"
    for _m in _metric_priority:
        _candidate = _panelf_root / _m / "panelF_summary.csv"
        if _candidate.exists():
            panel_f_file   = _candidate
            panel_f_metric = _m
            break
    if panel_f_file is None:
        w.append("Panel F information file not found; Panel F will show placeholder. "
                 "Run declan/fig4_cov_TFTS/run_panelF_covariance_overlap.py to generate it.")
    panel_f_natural_file = _first([
        VISIONCORE_ROOT / "outputs" / "panelF_natural_structure" / "panelF_natural_structure_scale_sweep.csv",
    ])
    if panel_f_natural_file is None:
        w.append("Natural-structure Panel F file not found; using diagnostic Panel F if available. "
                 "Run declan/fig4_cov_TFTS/run_panelF_natural_structure.py to generate it.")
    _closure_root = VISIONCORE_ROOT / "outputs" / "matched_twin_covariance_closure_finite_difference"
    panel_f_closure_summary_file = _first([
        _closure_root / "finite_difference_headline_raw_psd_bootstrap.csv",
    ])
    panel_f_closure_metrics_file = _first([
        _closure_root / "finite_difference_capture_metrics.csv",
    ])
    panel_f_closure_audit_file = _first([
        _closure_root / "finite_difference_provenance_audit.json",
    ])
    _compact_closure_root = VISIONCORE_ROOT / "outputs" / "matched_twin_covariance_closure_finite_difference_compact_k10"
    panel_f_compact_closure_summary_file = _first([
        _compact_closure_root / "finite_difference_headline_raw_psd_bootstrap.csv",
    ])
    if panel_f_closure_summary_file is None or panel_f_closure_metrics_file is None:
        w.append("Finite-difference closure Panel F files not found; Panel F will fall back to older diagnostic data. "
                 "Run declan.matched_twin_covariance_closure.run_finite_difference_closure and summarize_finite_difference_results.")
    else:
        w[:] = [
            msg for msg in w
            if "Panel F information file not found" not in msg
            and "Natural-structure Panel F file not found" not in msg
            and "panelF_fem_ranges.json not found" not in msg
        ]
    _fem_ranges_json = _panelf_root / "panelF_fem_ranges.json"
    panel_f_fem_ranges: Optional[dict] = None
    if _fem_ranges_json.exists():
        import json as _json
        with open(_fem_ranges_json, encoding="utf-8") as _fh:
            panel_f_fem_ranges = _json.load(_fh)
    else:
        w.append("panelF_fem_ranges.json not found; Panel F will use placeholder regime bands. "
                 "Run run_panelF_covariance_overlap.py --metric fem_ranges to compute them.")
    report_file  = _first([tfts_root / "MANUSCRIPT_REPORT.md"])
    summary_file = _first([tfts_root / "twin_feature_tangent_summary.json"])
    spec_file    = _first([tfts_root / "union_spectrum" / "twin_tangent_union_spectrum.csv"])
    tangent_maps = _first([tfts_root / "tangent_maps" / "twin_tangent_maps.pkl"])
    null_spectrum_summary_file = _first([
        tfts_root / "union_spectrum" / "twin_tangent_union_null_spectrum_summary.csv",
    ])
    v1_cache     = _first([CACHE_DIR / "fig2_decomposition.pkl",
                            VISIONCORE_ROOT / "outputs" / "cache" / "fig2_decomposition.pkl"])
    # Tangent-subspace information result (Panel E) — written by run_tangent_subspace_information.py
    information_file = _first([
        VISIONCORE_ROOT / "outputs" / "tangent_subspace_information"
        / "panelE_production_fisher" / "results"
        / "panelE_subspace_capture_summary.csv",
        VISIONCORE_ROOT / "outputs" / "tangent_subspace_information"
        / "panelE_production_k10_delta025" / "results"
        / "panelE_subspace_capture_summary.csv",
    ])
    information_null_file = _first([
        VISIONCORE_ROOT / "outputs" / "tangent_subspace_information"
        / "panelE_production_fisher" / "results"
        / "panelE_basis_null_summary.csv",
        VISIONCORE_ROOT / "outputs" / "tangent_subspace_information"
        / "panelE_production_k10_delta025" / "results"
        / "panelE_basis_null_summary.csv",
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
        null_spectrum_summary_file=null_spectrum_summary_file,
        panel_f_file=panel_f_file, panel_f_natural_file=panel_f_natural_file,
        panel_f_closure_summary_file=panel_f_closure_summary_file,
        panel_f_closure_metrics_file=panel_f_closure_metrics_file,
        panel_f_closure_audit_file=panel_f_closure_audit_file,
        panel_f_compact_closure_summary_file=panel_f_compact_closure_summary_file,
        panel_f_metric=panel_f_metric,
        panel_f_fem_ranges=panel_f_fem_ranges,
        information_null_file=information_null_file,
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
    """Load tangent vectors and (when available) base responses per object."""
    with open(tangent_maps_path, "rb") as f:
        tm = pickle.load(f)
    available = sorted(tm["object_payload"].keys())
    delta_key = min(available, key=lambda d: abs(d - delta))
    payload = tm["object_payload"][delta_key]
    oids = list(payload.keys())
    bx = np.vstack([np.asarray(payload[o]["bx"], dtype=np.float32) for o in oids])
    by = np.vstack([np.asarray(payload[o]["by"], dtype=np.float32) for o in oids])
    has_r0 = "r0" in payload[oids[0]]
    r0 = np.vstack([np.asarray(payload[o]["r0"], dtype=np.float32) for o in oids]) if has_r0 else None
    delta_model_px = float(payload[oids[0]].get("delta_model_px", 1.0))
    valid = np.isfinite(bx).all(axis=1) & np.isfinite(by).all(axis=1)
    if r0 is not None:
        valid = valid & np.isfinite(r0).all(axis=1)
    bx, by = bx[valid], by[valid]
    r0 = r0[valid] if r0 is not None else None
    return {
        "r0": r0, "bx": bx, "by": by,
        "n_objects": int(valid.sum()),
        "delta": delta_key,
        "delta_model_px": delta_model_px,
    }


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


def load_panel_f(paths: DataPaths, k: int = 2, delta: float = 0.25) -> Optional[pd.DataFrame]:
    """Load Panel F summary produced by run_panelF_covariance_overlap.py."""
    if paths.panel_f_file is None or not paths.panel_f_file.exists():
        return None
    df = pd.read_csv(paths.panel_f_file)
    d = df[
        (df["k"].astype(int) == k) &
        (np.isclose(df["delta"].astype(float), delta))
    ].sort_values("cloud_scale").reset_index(drop=True)
    return d if len(d) >= 2 else None


def load_panel_f_natural(paths: DataPaths) -> Optional[pd.DataFrame]:
    """Load natural-structure Panel F scale sweep."""
    if paths.panel_f_natural_file is None or not paths.panel_f_natural_file.exists():
        return None
    df = pd.read_csv(paths.panel_f_natural_file, low_memory=False)
    return df if len(df) else None


def load_panel_f_closure(paths: DataPaths) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[dict]]:
    """Load finite-difference closure summary, per-session rows, and audit."""
    if (
        paths.panel_f_closure_summary_file is None
        or paths.panel_f_closure_metrics_file is None
        or not paths.panel_f_closure_summary_file.exists()
        or not paths.panel_f_closure_metrics_file.exists()
    ):
        return None, None, None
    summary = pd.read_csv(paths.panel_f_closure_summary_file)
    metrics = pd.read_csv(paths.panel_f_closure_metrics_file)
    audit = None
    if paths.panel_f_closure_audit_file is not None and paths.panel_f_closure_audit_file.exists():
        with open(paths.panel_f_closure_audit_file, encoding="utf-8") as f:
            audit = json.load(f)
    return summary, metrics, audit


def load_panel_f_compact_closure(paths: DataPaths) -> Optional[pd.DataFrame]:
    """Load compact-k10 finite-difference closure headline rows for Panel F inset."""
    if (
        paths.panel_f_compact_closure_summary_file is None
        or not paths.panel_f_compact_closure_summary_file.exists()
    ):
        return None
    df = pd.read_csv(paths.panel_f_compact_closure_summary_file)
    return df if len(df) else None


def load_union_spectrum(paths: DataPaths, delta: float = 0.25, n_show: int = 40) -> pd.DataFrame:
    if paths.spec_file is None:
        return pd.DataFrame()
    df = pd.read_csv(paths.spec_file)
    d = df[(df["delta"] == delta) &
           (df["space"].astype(str).str.lower() == "raw") &
           (df["tangent_set"].astype(str).str.lower() == "combined")]
    return d.sort_values("component_index").head(n_show).reset_index(drop=True)


def load_null_spectrum_summary(paths: DataPaths, delta: float = 0.25, n_show: int = 32) -> Optional[pd.DataFrame]:
    """Load per-component null spectrum band from unit-shuffle repeats."""
    if paths.null_spectrum_summary_file is None:
        return None
    df = pd.read_csv(paths.null_spectrum_summary_file)
    d = df[
        (np.isclose(df["delta"].astype(float), delta)) &
        (df["space"].astype(str).str.lower() == "raw") &
        (df["tangent_set"].astype(str).str.lower() == "combined")
    ]
    return d.sort_values("component_index").head(n_show).reset_index(drop=True) if len(d) else None


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
    """Panel B: image-specific local translation charts.

    When r0 is available (new pickle): projects base responses into response PCA
    and draws bx/by tangent arrows from each anchor point — a true response-
    manifold chart.  Falls back to the tangent-PCA paired-glyph view otherwise.
    """
    panel_label(ax, "B", "Image-specific local translation charts")

    bx = np.asarray(tangent_data["bx"], dtype=np.float64)
    by = np.asarray(tangent_data["by"], dtype=np.float64)
    r0 = tangent_data.get("r0")
    if r0 is not None:
        r0 = np.asarray(r0, dtype=np.float64)
    n_obj = tangent_data["n_objects"]

    if r0 is not None:
        # ----------------------------------------------------------------
        # Response-manifold version: PCA of r0, quiver arrows for bx/by
        # ----------------------------------------------------------------
        r0_c = r0 - r0.mean(axis=0)
        _, s_r, Vt_r = np.linalg.svd(r0_c, full_matrices=False)
        var_exp = s_r**2 / (s_r**2).sum()

        r0_proj = r0_c @ Vt_r[:2].T        # (n_obj, 2)  response manifold coords
        bx_proj = bx @ Vt_r[:2].T          # tangent directions in same space
        by_proj = by @ Vt_r[:2].T

        # Normalise arrows to a fixed fraction of the r0 cloud spread.
        spread  = float(np.std(r0_proj)) * 0.30
        bx_u    = bx_proj / (np.linalg.norm(bx_proj, axis=1, keepdims=True) + 1e-12)
        by_u    = by_proj / (np.linalg.norm(by_proj, axis=1, keepdims=True) + 1e-12)

        # Faint background: full r0 cloud
        ax.scatter(r0_proj[:, 0], r0_proj[:, 1], s=6, color="0.82",
                   linewidths=0, zorder=1, alpha=0.55)

        idx = _farthest_point_subset(r0_proj, min(n_show, n_obj), seed=2)

        # Quiver: all selected arrows in one call each
        ax.quiver(r0_proj[idx, 0], r0_proj[idx, 1],
                  bx_u[idx, 0] * spread, bx_u[idx, 1] * spread,
                  color=MODEL, scale=1.0, scale_units="xy", angles="xy",
                  width=0.004, headwidth=4, headlength=5,
                  alpha=0.88, zorder=4)
        ax.quiver(r0_proj[idx, 0], r0_proj[idx, 1],
                  by_u[idx, 0] * spread, by_u[idx, 1] * spread,
                  color=BRIDGE, scale=1.0, scale_units="xy", angles="xy",
                  width=0.004, headwidth=4, headlength=5,
                  alpha=0.88, zorder=4)

        ax.scatter(r0_proj[idx, 0], r0_proj[idx, 1], s=16, color="white",
                   edgecolors="0.45", linewidths=0.5, zorder=5)

        bx_handle = Line2D([0], [0], color=MODEL,   lw=1.5, label=r"$b_x(I)$  horizontal")
        by_handle = Line2D([0], [0], color=BRIDGE,  lw=1.5, label=r"$b_y(I)$  vertical")

        pct0, pct1 = var_exp[0] * 100, var_exp[1] * 100
        ax.set_xlabel(f"Response PC 1 ({pct0:.1f}% var.)")
        ax.set_ylabel(f"Response PC 2 ({pct1:.1f}% var.)")
        ax.text(0.04, 0.04,
                f"{len(idx)} charts in response PCA  |  arrows show local translation directions",
                transform=ax.transAxes, fontsize=6.3, color="0.50",
                ha="left", va="bottom", style="italic")
    else:
        # ----------------------------------------------------------------
        # Fallback: tangent-PCA paired-glyph view (no r0 in pickle)
        # ----------------------------------------------------------------
        B   = np.vstack([bx, by])
        Bc  = B - B.mean(axis=0)
        _, s, Vt = np.linalg.svd(Bc, full_matrices=False)
        var_exp = s**2 / (s**2).sum()

        bx_proj = (bx - B.mean(axis=0)) @ Vt[:2].T
        by_proj = (by - B.mean(axis=0)) @ Vt[:2].T
        centers  = 0.5 * (bx_proj + by_proj)
        pair_len = np.linalg.norm(bx_proj - by_proj, axis=1)

        ax.scatter(bx_proj[:, 0], bx_proj[:, 1], s=9, color=MODEL,   alpha=0.11, linewidths=0, zorder=1)
        ax.scatter(by_proj[:, 0], by_proj[:, 1], s=9, color=BRIDGE,  alpha=0.11, linewidths=0, zorder=1)

        idx   = _farthest_point_subset(centers, min(n_show, n_obj), seed=2)
        order = idx[np.argsort(pair_len[idx])]
        for j, i in enumerate(order):
            alpha = 0.38 + 0.52 * (j + 1) / max(len(order), 1)
            ax.plot([bx_proj[i, 0], by_proj[i, 0]], [bx_proj[i, 1], by_proj[i, 1]],
                    "-", color="0.72", lw=0.65, alpha=0.60, zorder=2)
            ax.plot([centers[i, 0], bx_proj[i, 0]], [centers[i, 1], bx_proj[i, 1]],
                    "-", color=MODEL,  lw=1.25, alpha=alpha, zorder=3)
            ax.plot([centers[i, 0], by_proj[i, 0]], [centers[i, 1], by_proj[i, 1]],
                    "-", color=BRIDGE, lw=1.25, alpha=alpha, zorder=3)
        ax.scatter(centers[idx, 0],   centers[idx, 1],   s=10, color="white",  edgecolors="0.52", linewidths=0.45, zorder=4)
        ax.scatter(bx_proj[idx, 0],   bx_proj[idx, 1],   s=27, color=MODEL,    alpha=0.93, linewidths=0.45, edgecolors="white", zorder=5)
        ax.scatter(by_proj[idx, 0],   by_proj[idx, 1],   s=27, color=BRIDGE,   alpha=0.93, linewidths=0.45, edgecolors="white", zorder=5)

        bx_handle = Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL,   markersize=7, label=r"$b_x(I)$ horizontal")
        by_handle = Line2D([0], [0], marker="o", color="w", markerfacecolor=BRIDGE,  markersize=7, label=r"$b_y(I)$ vertical")

        pct0, pct1 = var_exp[0] * 100, var_exp[1] * 100
        ax.set_xlabel(f"Tangent PC 1 ({pct0:.1f}% var.)")
        ax.set_ylabel(f"Tangent PC 2 ({pct1:.1f}% var.)")
        ax.text(0.04, 0.04,
                f"{len(idx)} local charts shown; faint points show full tangent family",
                transform=ax.transAxes, fontsize=6.3, color="0.50",
                ha="left", va="bottom", style="italic")

    clean_axes(ax, grid=True)
    ax.legend(handles=[bx_handle, by_handle],
              frameon=False, loc="upper right", handlelength=1.1,
              borderpad=0.15, labelspacing=0.45)

    # Inset: distribution of cos(bx, by) angles across objects
    cos_xy = np.sum(bx * by, axis=1) / (np.linalg.norm(bx, axis=1) * np.linalg.norm(by, axis=1) + 1e-12)
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


def plot_panel_c(ax: plt.Axes, spec_df: pd.DataFrame, union_df: pd.DataFrame,
                 null_spec_df: Optional[pd.DataFrame] = None):
    """Panel C: cumulative tangent variance spectrum vs unit-shuffle null band."""
    panel_label(ax, "C", "Compact tangent spectrum")

    if spec_df is None or len(spec_df) == 0:
        ax.text(0.5, 0.5, "spectrum data not found", transform=ax.transAxes,
                ha="center", va="center", color=ACCENT, fontsize=8)
        clean_axes(ax)
        return

    ranks  = spec_df["component_index"].to_numpy(dtype=float)
    cumvar = spec_df["cumulative_fraction_variance"].to_numpy(dtype=float)
    pr_obs = float(spec_df["participation_ratio"].iloc[0])

    # Null PR for annotation (from union summary)
    null_pr = 31.03
    try:
        row_025 = union_df[np.isclose(union_df["delta"].astype(float), 0.25)]
        if len(row_025):
            null_pr = float(row_025["null_pr_mean"].iloc[0])
    except Exception:
        pass

    # --- Null reference ---
    if null_spec_df is not None and len(null_spec_df) >= 2:
        n_reps   = int(null_spec_df["n_null_repeats"].iloc[0])
        n_ranks  = null_spec_df["component_index"].to_numpy(dtype=float)
        n_med    = null_spec_df["cumvar_median"].to_numpy(dtype=float)
        n_lo     = null_spec_df["cumvar_ci_low"].to_numpy(dtype=float)
        n_hi     = null_spec_df["cumvar_ci_high"].to_numpy(dtype=float)
        n_ranks  = np.concatenate([[0], n_ranks])
        n_med    = np.concatenate([[0], n_med])
        n_lo     = np.concatenate([[0], n_lo])
        n_hi     = np.concatenate([[0], n_hi])
        ax.fill_between(n_ranks, n_lo, n_hi, color=NULL, alpha=0.22, lw=0, zorder=1)
        ax.plot(n_ranks, n_med, "--", color=NULL, lw=1.6, zorder=2,
                label=f"Unit-shuffle null  (n={n_reps}, 95% CI)")
    else:
        # Fallback: synthetic linear ramp to null PR
        n_ranks  = np.array([0.0, null_pr, float(ranks.max())])
        n_cumvar = np.array([0.0, 1.0,     1.0])
        ax.plot(n_ranks, n_cumvar, "--", color=NULL, lw=1.8, zorder=1,
                label=f"Unit-shuffle PR reference (PR≈{null_pr:.0f})")

    # --- Observed spectrum ---
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


def plot_panel_f_natural(
    ax: plt.Axes,
    panel_df: pd.DataFrame,
    *,
    drift_band: tuple[float, float] = (1.0, 3.0),
    microsaccade_band: tuple[float, float] = (10.0, 50.0),
):
    """Panel F: natural-image spatial-content diagnostic."""
    panel_label(ax, "F", "Spatial content recruits compact geometry")

    obj = panel_df[
        (panel_df["metric_name"].astype(str) == "tangent_subspace_fraction") &
        (panel_df["bootstrap_id_or_fold"].astype(str) == "object") &
        (panel_df["image_condition"].astype(str) == "intact_natural")
    ].copy()
    if len(obj) == 0:
        ax.text(0.5, 0.5, "natural-structure data missing",
                transform=ax.transAxes, ha="center", va="center",
                color=ACCENT, fontsize=8)
        clean_axes(ax, grid=True)
        return
    obj["displacement_arcmin"] = pd.to_numeric(obj["displacement_arcmin"], errors="coerce")
    obj["metric_value"] = pd.to_numeric(obj["metric_value"], errors="coerce")

    def _object_summary(basis_type: str, group: Optional[str] = None) -> pd.DataFrame:
        d = obj[obj["basis_type"].astype(str) == basis_type].copy()
        if group is not None:
            d = d[d["structure_group"].astype(str) == group]
        if len(d) == 0:
            return pd.DataFrame()
        return (
            d.groupby("displacement_arcmin", as_index=False)
            .agg(
                mean=("metric_value", "mean"),
                lo=("metric_value", lambda x: np.nanpercentile(x, 25)),
                hi=("metric_value", lambda x: np.nanpercentile(x, 75)),
            )
            .sort_values("displacement_arcmin")
        )

    diff = panel_df[
        (panel_df["metric_name"].astype(str) == "high_minus_low_fraction") &
        (panel_df["structure_group"].astype(str) == "high_minus_low") &
        (panel_df["image_condition"].astype(str) == "intact_natural")
    ].copy()
    if len(diff) == 0:
        ax.text(0.5, 0.5, "high-minus-low summary missing",
                transform=ax.transAxes, ha="center", va="center",
                color=ACCENT, fontsize=8)
        clean_axes(ax, grid=True)
        return
    diff["displacement_arcmin"] = pd.to_numeric(diff["displacement_arcmin"], errors="coerce")
    diff["metric_value"] = pd.to_numeric(diff["metric_value"], errors="coerce")

    ax.set_xscale("log")
    for d_lo, d_hi, color, label in [
        (drift_band[0], drift_band[1], MODEL, "drift"),
        (microsaccade_band[0], microsaccade_band[1], BRIDGE, "microsaccade"),
    ]:
        if np.isfinite(d_lo) and np.isfinite(d_hi) and d_lo < d_hi:
            ax.axvspan(d_lo, d_hi, color=color, alpha=0.10, lw=0, zorder=0)
            ax.text(np.sqrt(d_lo * d_hi), 0.97, label,
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=6.7, color=color)

    def _diff_summary(basis_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        d = diff[diff["basis_type"].astype(str) == basis_type].copy()
        if len(d) == 0:
            return pd.DataFrame(), pd.DataFrame()
        obs = d[d["bootstrap_id_or_fold"].astype(str) == "observed"].copy()
        boot = d[d["bootstrap_id_or_fold"].astype(str) != "observed"].copy()
        obs = obs.sort_values("displacement_arcmin")
        if len(boot):
            boot = (
                boot.groupby("displacement_arcmin", as_index=False)
                .agg(lo=("metric_value", lambda x: np.nanpercentile(x, 2.5)),
                     hi=("metric_value", lambda x: np.nanpercentile(x, 97.5)))
                .sort_values("displacement_arcmin")
            )
        return obs, boot

    def _plot_diff(
        basis_type: str,
        color: str,
        label: str,
        *,
        ls: str = "-",
        lw: float = 2.2,
        band_alpha: float = 0.16,
        marker: str = "o",
        zorder: int = 3,
    ) -> pd.DataFrame:
        obs, boot = _diff_summary(basis_type)
        if len(obs) == 0:
            return obs
        if len(boot):
            ax.fill_between(boot["displacement_arcmin"], boot["lo"], boot["hi"],
                            color=color, alpha=band_alpha, lw=0, zorder=1)
        ax.plot(obs["displacement_arcmin"], obs["metric_value"],
                linestyle=ls, color=color, lw=lw, marker=marker, markersize=4.2,
                markeredgecolor="white", markeredgewidth=0.45,
                label=label, zorder=zorder)
        return obs

    true_obs = _plot_diff("true_tangent", MODEL, "True tangent basis", lw=2.3, band_alpha=0.18)
    _plot_diff("unit_shuffle", NULL, "Unit-shuffle", ls="--", lw=1.45,
               band_alpha=0.08, marker="s", zorder=2)
    _plot_diff("random_subspace", "0.68", "Random subspace", ls=":", lw=1.35,
               band_alpha=0.05, marker=".", zorder=2)

    ax.axhline(0, color="0.45", lw=0.75, ls=":", zorder=0)
    ax.set_xlabel("Retinal displacement scale (arcmin)")
    ax.set_ylabel("High - low tangent-subspace fraction")

    finite_y = diff["metric_value"].to_numpy(float)
    finite_y = finite_y[np.isfinite(finite_y)]
    ymin = min(-0.06, float(np.nanpercentile(finite_y, 1)) - 0.015) if finite_y.size else -0.06
    ymax = max(0.16, float(np.nanpercentile(finite_y, 99)) + 0.025) if finite_y.size else 0.16
    ax.set_ylim(ymin, ymax)
    xmin = float(np.nanmin(diff["displacement_arcmin"])) * 0.78
    xmax = max(float(np.nanmax(diff["displacement_arcmin"])) * 1.35, microsaccade_band[1] * 1.05)
    ax.set_xlim(xmin, xmax)
    ticks = [0.0625, 0.25, 1, 4, 16, 50]
    ticks = [t for t in ticks if xmin <= t <= xmax]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1/16" if np.isclose(t, 0.0625) else f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    clean_axes(ax, grid=True)
    ax.legend(frameon=False, loc="upper left", handlelength=1.4, labelspacing=0.3)

_PANEL_F_YLABELS = {
    "fisher_r2":          "Fisher-weighted $R^2$ (tangent vs. actual)",
    "variance_fraction":  "FEM variance fraction in tangent subspace",
    "covariance_overlap": "Tangent / FEM covariance overlap",
}


def plot_panel_f(
    ax: plt.Axes,
    panel_f_df: Optional[pd.DataFrame],
    metric: str = "covariance_overlap",
    *,
    drift_band: tuple[float, float] = (1.0, 3.0),
    microsaccade_band: tuple[float, float] = (10.0, 50.0),
    arcmin_per_cloud_scale: Optional[float] = None,
):
    """Panel F: drift and microsaccades as operating regimes.

    panel_f_df is the output of run_panelF_covariance_overlap.py (median + 95% CI
    per cloud_scale).  If None, shows a placeholder with a run-script hint.

    drift_band / microsaccade_band are always in arcmin.  For metrics where
    cloud_scale is dimensionless, pass arcmin_per_cloud_scale to convert the
    x-axis before plotting.  For fisher_r2, cloud_scale is already arcmin.
    """
    title = (
        "Tangent R² decays across FEM scales"
        if metric == "fisher_r2"
        else "Tangent / FEM overlap across scales"
    )
    panel_label(ax, "F", title)

    if panel_f_df is None or len(panel_f_df) == 0:
        ax.text(0.5, 0.52, "Panel F data not yet computed.",
                transform=ax.transAxes, ha="center", va="center",
                color=ACCENT, fontsize=8)
        ax.text(0.5, 0.40,
                "Run:\n"
                "python -m declan.fig4_cov_TFTS.run_panelF_covariance_overlap\n"
                "    --metric covariance_overlap,variance_fraction,fisher_r2",
                transform=ax.transAxes, ha="center", va="center",
                color="0.50", fontsize=6.2, family="monospace",
                linespacing=1.6)
        ax.set_xlabel("Retinal displacement scale (arcmin)")
        ax.set_ylabel(_PANEL_F_YLABELS.get(metric, "Tangent-regime utility"))
        clean_axes(ax, grid=True)
        return

    x_raw = panel_f_df["cloud_scale"].to_numpy(dtype=float)
    # Convert dimensionless cloud_scale → arcmin when requested.  Fisher R²
    # stores cloud_scale directly in arcmin, so compose() passes None there.
    x = x_raw * arcmin_per_cloud_scale if arcmin_per_cloud_scale is not None else x_raw
    y   = panel_f_df["median"].to_numpy(dtype=float)
    lo  = panel_f_df["ci_low"].to_numpy(dtype=float)
    hi  = panel_f_df["ci_high"].to_numpy(dtype=float)

    ax.set_xscale("log")
    xmin = float(np.nanmin(x)) * 0.72
    xmax_data = float(np.nanmax(x)) * 1.40
    # Keep the measured curve in its native units, but let the axis widen to
    # show where empirical FEM regimes sit.  This avoids the misleading
    # alternative of stretching the data to reach the microsaccade band.
    finite_band_hi = [
        float(v) for v in (drift_band[1], microsaccade_band[1])
        if np.isfinite(v) and v > 0
    ]
    xmax = max([xmax_data, *[v * 1.10 for v in finite_band_hi]]) if finite_band_hi else xmax_data
    ax.set_xlim(xmin, xmax)

    # Clip biological bands to the visible range; skip if entirely outside
    d_lo = max(drift_band[0], xmin)
    d_hi = min(drift_band[1], xmax)
    m_lo = max(microsaccade_band[0], xmin)
    m_hi = min(microsaccade_band[1], xmax)

    if d_lo < d_hi:
        ax.axvspan(d_lo, d_hi, color=MODEL, alpha=0.12, lw=0, zorder=0)
        ax.text(np.sqrt(d_lo * d_hi), 0.97, "drift scale",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.8, color=MODEL)

    if m_lo < m_hi:
        ax.axvspan(m_lo, m_hi, color=BRIDGE, alpha=0.12, lw=0, zorder=0)
        # Add "→" if band was clipped on the right
        label = "microsaccade scale" + (" →" if microsaccade_band[1] > xmax else "")
        label_x = np.sqrt(m_lo * m_hi)
        ax.text(label_x, 0.97, label,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.8, color=BRIDGE)

    # CI band + median line
    ax.fill_between(x, lo, hi, color=TEXT, alpha=0.12, lw=0, zorder=2)
    ax.plot(x, y, "o-", color=TEXT, lw=2.2, markersize=5.5,
            markeredgecolor="white", markeredgewidth=0.5, zorder=3)

    ax.set_xlabel("Retinal displacement scale (arcmin)")
    ax.set_ylabel(_PANEL_F_YLABELS.get(metric, "Tangent-regime utility"))
    ax.set_ylim(0, min(1.05, max(0.5, float(np.nanmax(hi)) + 0.08)))

    tick_candidates = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 20.0, 50.0]
    ticks = [t for t in tick_candidates if xmin <= t <= xmax]
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


def plot_panel_f_closure(
    ax: plt.Axes,
    closure_summary: Optional[pd.DataFrame],
    closure_metrics: Optional[pd.DataFrame],
    closure_audit: Optional[dict],
    compact_summary: Optional[pd.DataFrame] = None,
):
    """Panel F: matched finite-difference retinal-translation covariance closure."""
    panel_label(ax, "F", "Translation tangents capture\nrecorded FEM covariance")

    if closure_summary is None or closure_metrics is None or len(closure_summary) == 0:
        ax.text(0.5, 0.52, "finite-difference closure\nnot found",
                transform=ax.transAxes, ha="center", va="center",
                color=ACCENT, fontsize=8)
        ax.text(0.5, 0.35,
                "Run matched_twin_covariance_closure\nfinite-difference + bootstrap summaries",
                transform=ax.transAxes, ha="center", va="center",
                color="0.50", fontsize=6.4)
        ax.set_ylabel(r"Excess $\Sigma_{\mathrm{FEM}}$ capture" "\nover unit-shuffle null")
        clean_axes(ax, grid=True)
        return

    source = "fd_sample_eye_trace_cov"
    k = 2
    target = "psd"
    controls = ["none", "global_rate", "target_pc1", "global_rate+target_pc1"]
    labels = ["none", "global", "PC1", "global\n+ PC1"]

    s = closure_summary[
        (closure_summary["target_variant"].astype(str) == target) &
        (closure_summary["basis_source"].astype(str) == source) &
        (closure_summary["k"].astype(int) == k)
    ].copy()
    m = closure_metrics[
        (closure_metrics["target_variant"].astype(str) == target) &
        (closure_metrics["basis_source"].astype(str) == source) &
        (closure_metrics["k"].astype(int) == k) &
        (closure_metrics["row_status"].astype(str) == "ok")
    ].copy()

    rows = []
    all_vals: list[float] = []
    rng = np.random.default_rng(4)
    for i, control in enumerate(controls):
        sr = s[s["projection_control"].astype(str) == control]
        mr = m[m["projection_control"].astype(str) == control]
        if len(sr) == 0 or len(mr) == 0:
            continue
        mean = float(sr["effect_unit_mean"].iloc[0])
        lo = float(sr["effect_unit_boot_ci_low"].iloc[0])
        hi = float(sr["effect_unit_boot_ci_high"].iloc[0])
        vals = pd.to_numeric(mr["effect_minus_unit_shuffle_median"], errors="coerce").dropna().to_numpy(float)
        all_vals.extend(float(v) for v in vals if np.isfinite(v))
        all_vals.extend([lo, hi, mean])
        jitter = rng.uniform(-0.11, 0.11, size=vals.size)
        ax.scatter(np.full(vals.size, i) + jitter, vals,
                   s=13, color="0.25", alpha=0.22, linewidths=0, zorder=2)
        ax.errorbar(i, mean,
                    yerr=[[max(mean - lo, 0.0)], [max(hi - mean, 0.0)]],
                    fmt="o", color=BRIDGE, ecolor=BRIDGE, elinewidth=1.5,
                    capsize=3.5, markersize=6.0, markeredgecolor="white",
                    markeredgewidth=0.7, zorder=4)
        rows.append((control, i, mean, lo, hi, vals.size,
                     int(sr["n_effect_positive"].iloc[0]),
                     float(sr["sign_test_p_two_sided"].iloc[0])))

    ax.axhline(0, color="0.48", lw=0.75, ls=":", zorder=1)
    ax.set_xticks(np.arange(len(controls)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Excess $\Sigma_{\mathrm{FEM}}$ capture" "\nover unit-shuffle null")
    finite_vals = np.asarray(all_vals, dtype=float)
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    ymax = max(0.47, float(np.nanmax(finite_vals)) + 0.045) if finite_vals.size else 0.47
    ax.set_ylim(-0.05, ymax)
    clean_axes(ax, grid=True)

    controlled = next((r for r in rows if r[0] == "global_rate+target_pc1"), None)
    if controlled is not None:
        _, _, mean, lo, hi, n, n_pos, pval = controlled
        ax.text(0.04, 0.96,
                f"global + PC1 removed:\n"
                f"+{mean:.3f} [{lo:.3f}, {hi:.3f}],\n"
                f"{n_pos}/{n} sessions",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=6.3, color=BRIDGE, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=BRIDGE_L, lw=0.7, alpha=0.96))

    if compact_summary is not None and len(compact_summary):
        inset_sources = [
            ("fd_sample_eye_trace_cov", "full\nFD", BRIDGE),
            ("fd_sample_eye_trace_xfit_compact_k10_cov", "compact\nk=10", MODEL),
        ]
        inset_rows = []
        for src, label, color in inset_sources:
            rr = compact_summary[
                (compact_summary["target_variant"].astype(str) == "psd") &
                (compact_summary["basis_source"].astype(str) == src) &
                (compact_summary["projection_control"].astype(str) == "global_rate+target_pc1") &
                (compact_summary["k"].astype(int) == 2)
            ]
            if len(rr) == 0:
                continue
            r0 = rr.iloc[0]
            inset_rows.append(
                (
                    label,
                    color,
                    float(r0["effect_unit_mean"]),
                    float(r0["effect_unit_boot_ci_low"]),
                    float(r0["effect_unit_boot_ci_high"]),
                )
            )
        if len(inset_rows) == 2:
            iax = ax.inset_axes([0.57, 0.54, 0.36, 0.33])
            inset_vals = []
            for i, (label, color, mean, lo, hi) in enumerate(inset_rows):
                iax.errorbar(
                    i, mean,
                    yerr=[[max(mean - lo, 0.0)], [max(hi - mean, 0.0)]],
                    fmt="o", color=color, ecolor=color, elinewidth=1.15,
                    capsize=2.4, markersize=4.4, markeredgecolor="white",
                    markeredgewidth=0.55, zorder=3,
                )
                inset_vals.extend([mean, lo, hi])
            ratio = inset_rows[1][2] / inset_rows[0][2] if inset_rows[0][2] else float("nan")
            iax.axhline(0, color="0.55", lw=0.55, ls=":", zorder=1)
            iax.set_xticks([0, 1])
            iax.set_xticklabels([r[0] for r in inset_rows], fontsize=5.6)
            iax.tick_params(axis="y", labelsize=5.6, length=2)
            iax.set_title("controlled", fontsize=5.9, pad=1.5)
            top = max(0.23, max(inset_vals) + 0.025)
            iax.set_ylim(-0.02, top)
            iax.text(0.50, 0.94, f"{ratio:.2f}x", transform=iax.transAxes,
                     ha="center", va="top", fontsize=5.9, color=TEXT,
                     fontweight="bold")
            clean_axes(iax, grid=True)

    if closure_audit:
        raw_trace = closure_audit.get("target_trace_raw_total", float("nan"))
        psd_trace = closure_audit.get("target_trace_psd_total", float("nan"))
        device = closure_audit.get("manifest_device", "")
        ax.text(0.98, 0.04,
                f"PSD target; raw also positive\nraw trace {raw_trace:.0f}, PSD {psd_trace:.0f}  |  {device}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5.9, color="0.38")


def load_information(paths: DataPaths) -> Optional[pd.DataFrame]:
    """Load per-window capture summary (fallback source for Panel E)."""
    if paths.information_file is None or not paths.information_file.exists():
        return None
    return pd.read_csv(paths.information_file)


def load_information_null_summary(paths: DataPaths) -> Optional[pd.DataFrame]:
    """Load pre-aggregated basis_null_summary (preferred source for Panel E bars)."""
    if paths.information_null_file is None or not paths.information_null_file.exists():
        return None
    return pd.read_csv(paths.information_null_file)


def _panel_e_bars_from_null_summary(null_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Extract 'all' rows from basis_null_summary in the order we want to plot."""
    d = null_df[null_df["kind"].astype(str) == "all"].copy()
    if len(d) == 0:
        return None
    # Map basis_label_group → display order / label (orthogonal complement omitted: always 1-tangent)
    order = ["tangent", "unit_shuffle", "random_orthogonal"]
    rows = []
    for grp in order:
        r = d[d["basis_label_group"].astype(str) == grp]
        if len(r) == 0:
            continue
        rows.append({
            "group":    grp,
            "mean":     float(r["mean_fraction_full_fem_gain_captured"].iloc[0]),
            "ci_low":   float(r["ci_low_fraction"].iloc[0]),
            "ci_high":  float(r["ci_high_fraction"].iloc[0]),
            "n":        int(r["n_windows_positive_gain"].iloc[0]),
        })
    return pd.DataFrame(rows) if rows else None


def _panel_e_bars_from_capture_df(capture_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Aggregate fraction_full_fem_gain_captured from per-window CSV (fallback)."""
    pos = capture_df[capture_df["full_gain_positive"].astype(str).str.lower() == "true"].copy()
    if len(pos) == 0:
        pos = capture_df.copy()

    # Map basis columns to group labels
    def _group(row) -> str:
        bl = str(row.get("basis_label", ""))
        bt = str(row.get("basis_type",  ""))
        if bl == "tangent":               return "tangent"
        if bl == "orthogonal_complement": return "orthogonal_complement"
        if bt == "unit_shuffle" or bl.startswith("unit_shuffle"):  return "unit_shuffle"
        if bt == "random_orthogonal" or bl.startswith("random_orthogonal"): return "random_orthogonal"
        return "other"

    pos = pos.copy()
    pos["group"] = pos.apply(_group, axis=1)
    pos = pos[pos["group"] != "other"]

    rows = []
    for grp in ["tangent", "unit_shuffle", "random_orthogonal"]:
        sub = pos[pos["group"] == grp]
        if len(sub) == 0:
            continue
        vals = pd.to_numeric(sub["fraction_full_fem_gain_captured"], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        rows.append({
            "group":   grp,
            "mean":    float(vals.mean()),
            "ci_low":  float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)),
            "n":       len(vals),
        })
    return pd.DataFrame(rows) if rows else None


def plot_panel_e_information(
    ax: plt.Axes,
    null_summary_df: Optional[pd.DataFrame],
    capture_df: Optional[pd.DataFrame],
):
    """Panel E: tangent-subspace Fisher gain fraction, bar chart."""
    panel_label(ax, "E", "FEM-related local displacement sensitivity")

    # Prefer pre-aggregated null summary; fall back to per-window capture CSV
    bar_df = None
    if null_summary_df is not None and len(null_summary_df) > 0:
        bar_df = _panel_e_bars_from_null_summary(null_summary_df)
    if bar_df is None and capture_df is not None and len(capture_df) > 0:
        bar_df = _panel_e_bars_from_capture_df(capture_df)

    if bar_df is None or len(bar_df) == 0:
        ax.set_axis_off()
        ax.text(0.5, 0.55, "Tangent-subspace\nFisher gain",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, fontweight="bold", color=MODEL)
        ax.text(0.5, 0.35, "production run pending",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7.5, color="0.55", style="italic")
        return

    _labels = {
        "tangent":           "Tangent\nbasis",
        "unit_shuffle":      "Unit-shuffle\nnull",
        "random_orthogonal": "Random\nnull",
    }
    _colors = {
        "tangent":           MODEL,
        "unit_shuffle":      NULL,
        "random_orthogonal": NULL_L,
    }

    x      = np.arange(len(bar_df))
    means  = bar_df["mean"].to_numpy(dtype=float)
    ci_lo  = bar_df["ci_low"].to_numpy(dtype=float)
    ci_hi  = bar_df["ci_high"].to_numpy(dtype=float)
    err_lo = np.clip(means - ci_lo, 0, None)
    err_hi = np.clip(ci_hi - means, 0, None)
    colors = [_colors.get(g, NULL) for g in bar_df["group"]]
    xlbls  = [_labels.get(g, g)    for g in bar_df["group"]]

    ax.bar(x, means,
           yerr=[err_lo, err_hi], capsize=3.5, error_kw=dict(lw=1.0, ecolor="0.35"),
           color=colors, alpha=0.88, edgecolor="white", linewidth=0.5, zorder=3)

    # Subtle 1.0 reference — dotted, thin, does not dominate
    ax.axhline(1.0, color="0.65", lw=0.7, ls=":", zorder=1)
    ax.axhline(0.0, color="0.80", lw=0.5, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, fontsize=6.5)
    ax.set_ylabel("Fraction of FEM\ninformation gain captured")
    ax.set_ylim(0, 0.75)
    clean_axes(ax, grid=True)

    ax.text(0.03, 0.97, "image-disjoint basis  |  k=10  |  0.25 arcmin",
            transform=ax.transAxes, fontsize=6.0, color=MODEL,
            fontweight="bold", ha="left", va="top")

    # Annotate tangent bar value
    tang_idx = list(bar_df["group"]).index("tangent") if "tangent" in list(bar_df["group"]) else None
    if tang_idx is not None:
        ax.text(tang_idx, means[tang_idx] + err_hi[tang_idx] + 0.025,
                f"{means[tang_idx]:.2f}",
                ha="center", va="bottom", fontsize=7.0, color=MODEL, fontweight="bold")


def _fmt(x: object, digits: int = 3) -> str:
    try:
        xf = float(x)
    except Exception:
        return "NA"
    if not np.isfinite(xf):
        return "NA"
    return f"{xf:.{digits}f}"


def _first_summary_row(df: Optional[pd.DataFrame], **filters) -> Optional[pd.Series]:
    if df is None or len(df) == 0:
        return None
    d = df.copy()
    for col, val in filters.items():
        if col not in d.columns:
            return None
        if isinstance(val, float):
            d = d[np.isclose(d[col].astype(float), val)]
        elif isinstance(val, int):
            d = d[d[col].astype(int) == val]
        else:
            d = d[d[col].astype(str) == str(val)]
    if len(d) == 0:
        return None
    return d.iloc[0]


def build_methods_text(
    *,
    paths: DataPaths,
    v1_data: Optional[dict],
    tangent_data: Optional[dict],
    union_df: pd.DataFrame,
    basis_df: pd.DataFrame,
    info_null_sum_df: Optional[pd.DataFrame],
    info_df: Optional[pd.DataFrame],
    closure_summary_df: Optional[pd.DataFrame],
    closure_audit: Optional[dict],
    compact_closure_summary_df: Optional[pd.DataFrame],
) -> str:
    """Build a panel-by-panel methods sidecar for the rendered figure."""
    lines: list[str] = []
    lines.append("# Figure 4 Methods and Calculation Audit")
    lines.append("")
    lines.append("Generated by `declan/fig4_cov_TFTS/generate_covTFTS_figure.py`.")
    lines.append("")
    lines.append("## Source Files")
    source_rows = [
        ("Recorded V1 cache", paths.v1_cache),
        ("Tangent maps", paths.tangent_maps),
        ("Union spectrum", paths.spec_file),
        ("Union summary", paths.union_file),
        ("Image-disjoint basis summary", paths.basis_file),
        ("Panel E Fisher summary", paths.information_file),
        ("Panel E basis/null summary", paths.information_null_file),
        ("Panel F finite-difference headline", paths.panel_f_closure_summary_file),
        ("Panel F finite-difference metrics", paths.panel_f_closure_metrics_file),
        ("Panel F finite-difference audit", paths.panel_f_closure_audit_file),
        ("Panel F compact-k10 headline", paths.panel_f_compact_closure_summary_file),
    ]
    for label, path in source_rows:
        lines.append(f"- {label}: `{path}`")

    lines.append("")
    lines.append("## Shared Conventions")
    lines.append("- All covariance matrices are symmetrized before eigendecomposition.")
    lines.append("- For a covariance matrix `Sigma`, eigenvalue spectra use positive eigenvalues sorted descending.")
    lines.append("- A subspace capture fraction is `tr(U.T @ Sigma @ U) / tr(Sigma)`, where columns of `U` are orthonormal basis vectors.")
    lines.append("- Unit-shuffle nulls permute unit identity in the source basis/covariance while preserving its loading structure.")
    lines.append("- Projection controls remove nuisance modes with `P = I - Q Q.T`; target and source covariances are replaced by `P @ Sigma @ P` before capture is computed.")

    lines.append("")
    lines.append("## Panel A. Recorded V1 Anchor")
    lines.append("Source: `fig2_decomposition.pkl`, window index 2.")
    lines.append("For each recorded session, the left subpanel computes the mean off-diagonal noise correlation before and after conditioning on measured eye position:")
    lines.append("")
    lines.append("```text")
    lines.append("mean_noise_corr = mean_{i<j} NoiseCorr[i,j]")
    lines.append("```")
    lines.append("")
    lines.append("The right subpanel plots the cumulative FEM-linked covariance spectrum:")
    lines.append("")
    lines.append("```text")
    lines.append("lambda = positive_eigenvalues(Sigma_FEM)")
    lines.append("cumulative_fraction(k) = sum_{r<=k} lambda_r / sum_r lambda_r")
    lines.append("```")
    if v1_data is not None:
        nc_u = np.asarray(v1_data.get("nc_u", []), dtype=float)
        nc_c = np.asarray(v1_data.get("nc_c", []), dtype=float)
        lines.append("")
        lines.append("Audit:")
        lines.append(f"- sessions: `{len(nc_u)}`")
        lines.append(f"- window: `{_fmt(v1_data.get('window_ms'), 3)} ms`")
        lines.append(f"- mean uncorrected noise correlation: `{_fmt(np.nanmean(nc_u), 4)}`")
        lines.append(f"- mean eye-position-corrected noise correlation: `{_fmt(np.nanmean(nc_c), 4)}`")
        lines.append(f"- mean corrected-minus-uncorrected change: `{_fmt(np.nanmean(nc_c - nc_u), 4)}`")

    lines.append("")
    lines.append("## Panel B. Image-Specific Local Translation Charts")
    lines.append("Source: `tangent_maps/twin_tangent_maps.pkl`, using the delta closest to 0.25 arcmin.")
    lines.append("For each image/history object, the fitted twin is evaluated under small horizontal and vertical retinal translations. Central finite differences define local tangent vectors:")
    lines.append("")
    lines.append("```text")
    lines.append("b_x(I) = [r(I + dx) - r(I - dx)] / (2 dx)")
    lines.append("b_y(I) = [r(I + dy) - r(I - dy)] / (2 dy)")
    lines.append("```")
    lines.append("")
    lines.append("The panel projects full stimulus-history response objects into response PCA space and overlays the paired tangent arrows. The inset summarizes signed local chart geometry with `cos(b_x, b_y)`.")
    if tangent_data is not None:
        bx = np.asarray(tangent_data["bx"], dtype=float)
        by = np.asarray(tangent_data["by"], dtype=float)
        cos_xy = np.sum(bx * by, axis=1) / (np.linalg.norm(bx, axis=1) * np.linalg.norm(by, axis=1) + 1e-12)
        lines.append("")
        lines.append("Audit:")
        lines.append(f"- valid objects: `{int(tangent_data['n_objects'])}`")
        lines.append(f"- plotted delta: `{_fmt(tangent_data['delta'], 3)} arcmin`")
        lines.append(f"- median `cos(b_x,b_y)`: `{_fmt(np.nanmedian(cos_xy), 3)}`")
        lines.append(f"- IQR `cos(b_x,b_y)`: `[{_fmt(np.nanpercentile(cos_xy, 25), 3)}, {_fmt(np.nanpercentile(cos_xy, 75), 3)}]`")

    lines.append("")
    lines.append("## Panel C. Compact Tangent Spectrum")
    lines.append("Source: `union_spectrum/twin_tangent_union_spectrum.csv` and image-disjoint union summary.")
    lines.append("The pooled tangent family stacks horizontal and vertical local tangents across valid objects:")
    lines.append("")
    lines.append("```text")
    lines.append("B = [b_x(I_1), b_y(I_1), ..., b_x(I_N), b_y(I_N)]")
    lines.append("Sigma_tangent = B @ B.T")
    lines.append("participation_ratio = (sum lambda)^2 / sum(lambda^2)")
    lines.append("```")
    lines.append("")
    lines.append("The observed cumulative spectrum is compared with unit-shuffled tangent controls that preserve per-vector loading magnitudes but break unit identity.")
    row_c = _first_summary_row(union_df, delta=0.25)
    if row_c is not None:
        lines.append("")
        lines.append("Audit at 0.25 arcmin:")
        lines.append(f"- valid objects: `{int(row_c.get('n_objects', 0))}`")
        lines.append(f"- observed PR: `{_fmt(row_c.get('participation_ratio'), 3)}`")
        lines.append(f"- unit-shuffle PR mean: `{_fmt(row_c.get('null_pr_mean'), 3)}`")
        lines.append(f"- unit-shuffle PR 95% interval: `[{_fmt(row_c.get('null_pr_ci_low'), 3)}, {_fmt(row_c.get('null_pr_ci_high'), 3)}]`")

    lines.append("")
    lines.append("## Panel D. Cross-Image Generalization")
    lines.append("Source: image-disjoint train/test basis summary.")
    lines.append("For each split, a compact tangent basis is learned from one set of image identities and tested on held-out image identities:")
    lines.append("")
    lines.append("```text")
    lines.append("U_train,k = top k eigenvectors of B_train @ B_train.T")
    lines.append("heldout_capture(k) = ||U_train,k.T @ B_test||_F^2 / ||B_test||_F^2")
    lines.append("```")
    lines.append("")
    lines.append("The plotted null uses unit-shuffled training bases evaluated on the same held-out tangent matrix.")
    if basis_df is not None and len(basis_df):
        d = basis_df[np.isclose(basis_df["delta"].astype(float), 0.25)].copy()
        if len(d):
            lines.append("")
            lines.append("Audit at 0.25 arcmin:")
            lines.append("")
            lines.append("| k | observed held-out capture | unit-shuffle null |")
            lines.append("|---:|---:|---:|")
            for _, rr in d.sort_values("basis_rank_k").iterrows():
                lines.append(f"| {int(rr['basis_rank_k'])} | {_fmt(rr['capture'], 3)} | {_fmt(rr['null'], 3)} |")

    lines.append("")
    lines.append("## Panel E. Tangent Subspace Captures FEM-Related Displacement Sensitivity")
    lines.append("Source: `tangent_subspace_information` production Fisher summaries.")
    lines.append("For each held-out image/window, a derivative-projection Poisson Fisher analysis compares local displacement sensitivity along real-FEM and stabilized histories. With predicted rate `mu` and displacement derivative `dmu/dalpha`, the local Poisson Fisher form is:")
    lines.append("")
    lines.append("```text")
    lines.append("F = (dmu/dalpha).T @ diag(1 / mu) @ (dmu/dalpha)")
    lines.append("```")
    lines.append("")
    lines.append("For a basis `U`, derivatives are projected into the basis before computing the gain. The plotted quantity is the fraction of the full real-versus-stabilized FEM Fisher gain captured by each basis family.")
    bar_df = None
    if info_null_sum_df is not None and len(info_null_sum_df):
        bar_df = _panel_e_bars_from_null_summary(info_null_sum_df)
    if bar_df is None and info_df is not None and len(info_df):
        bar_df = _panel_e_bars_from_capture_df(info_df)
    if bar_df is not None and len(bar_df):
        lines.append("")
        lines.append("Audit:")
        lines.append("")
        lines.append("| basis | mean fraction | 95% interval | n |")
        lines.append("|---|---:|---:|---:|")
        for _, rr in bar_df.iterrows():
            lines.append(
                f"| {rr['group']} | {_fmt(rr['mean'], 3)} | "
                f"[{_fmt(rr['ci_low'], 3)}, {_fmt(rr['ci_high'], 3)}] | {int(rr['n'])} |"
            )

    lines.append("")
    lines.append("## Panel F. Translation Tangents Capture Recorded FEM Covariance")
    lines.append("Source: finite-difference matched twin/recorded closure outputs.")
    lines.append("The recorded target is the matched-unit FEM covariance from `fig2_decomposition_ryan.pkl`, window index 1. PSD target rows use eigenvalue clipping:")
    lines.append("")
    lines.append("```text")
    lines.append("Sigma_FEM_psd = V @ diag(max(lambda, 0)) @ V.T")
    lines.append("```")
    lines.append("")
    lines.append("For each matched session, fitted-twin retinal translation Jacobians are computed by central finite differences:")
    lines.append("")
    lines.append("```text")
    lines.append("J_i[:,x] = [r_i(stim shifted +step_x) - r_i(stim shifted -step_x)] / (2 step)")
    lines.append("J_i[:,y] = [r_i(stim shifted +step_y) - r_i(stim shifted -step_y)] / (2 step)")
    lines.append("delta_r_i = J_i @ centered_eye_position_i")
    lines.append("Sigma_FD = cov_i(delta_r_i)")
    lines.append("```")
    lines.append("")
    lines.append("The source eigenspace `U_source,k` is obtained from `Sigma_FD`; the plotted effect is target capture above the median unit-shuffle source-basis null:")
    lines.append("")
    lines.append("```text")
    lines.append("capture = tr(U_source,k.T @ Sigma_FEM @ U_source,k) / tr(Sigma_FEM)")
    lines.append("effect = capture - median(capture_unit_shuffle)")
    lines.append("```")
    lines.append("")
    lines.append("Projection controls are applied to both source and target before the source eigenspace is learned. `global_rate` removes the all-ones mode, `target_pc1` removes the leading target covariance eigenvector, and `global_rate+target_pc1` removes both.")
    lines.append("")
    lines.append("The compact-k10 inset repeats this analysis after restricting finite-difference response predictions to a cross-fit compact tangent subspace. For each trial-disjoint fold, the basis is learned from the other trials:")
    lines.append("")
    lines.append("```text")
    lines.append("B_train = [J_x(train_1), J_y(train_1), ..., J_x(train_N), J_y(train_N)]")
    lines.append("U_compact,10 = top 10 eigenvectors of B_train @ B_train.T")
    lines.append("delta_r_i,compact = U_compact,10 @ U_compact,10.T @ delta_r_i  for held-out trials")
    lines.append("Sigma_FD,compact = cov_i(delta_r_i,compact)")
    lines.append("```")
    lines.append("")
    lines.append("Thus the inset asks whether the recorded-covariance prediction survives after the FD source is forced to live in the compact tangent geometry, rather than merely in the full FD translation covariance.")
    if closure_summary_df is not None and len(closure_summary_df):
        lines.append("")
        lines.append("Main Panel F audit, PSD target, full finite-difference source, source eigenspace k=2:")
        lines.append("")
        lines.append("| projection control | capture | effect over unit-shuffle | 95% bootstrap interval | sign |")
        lines.append("|---|---:|---:|---:|---:|")
        for control in ["none", "global_rate", "target_pc1", "global_rate+target_pc1"]:
            rr = _first_summary_row(
                closure_summary_df,
                target_variant="psd",
                basis_source="fd_sample_eye_trace_cov",
                projection_control=control,
                k=2,
            )
            if rr is not None:
                lines.append(
                    f"| {control} | {_fmt(rr['capture_mean'], 3)} | {_fmt(rr['effect_unit_mean'], 3)} | "
                    f"[{_fmt(rr['effect_unit_boot_ci_low'], 3)}, {_fmt(rr['effect_unit_boot_ci_high'], 3)}] | "
                    f"{int(rr['n_effect_positive'])}/{int(rr['n_effect_nonzero'])} |"
                )
    if compact_closure_summary_df is not None and len(compact_closure_summary_df):
        full = _first_summary_row(
            compact_closure_summary_df,
            target_variant="psd",
            basis_source="fd_sample_eye_trace_cov",
            projection_control="global_rate+target_pc1",
            k=2,
        )
        comp = _first_summary_row(
            compact_closure_summary_df,
            target_variant="psd",
            basis_source="fd_sample_eye_trace_xfit_compact_k10_cov",
            projection_control="global_rate+target_pc1",
            k=2,
        )
        if full is not None and comp is not None:
            ratio = float(comp["effect_unit_mean"]) / float(full["effect_unit_mean"])
            lines.append("")
            lines.append("Inset audit, PSD target, global-rate + target-PC1 removed, source eigenspace k=2:")
            lines.append("")
            lines.append("| source | effect over unit-shuffle | 95% bootstrap interval | sign |")
            lines.append("|---|---:|---:|---:|")
            for label, rr in [("full finite difference", full), ("cross-fit compact k=10", comp)]:
                lines.append(
                    f"| {label} | {_fmt(rr['effect_unit_mean'], 3)} | "
                    f"[{_fmt(rr['effect_unit_boot_ci_low'], 3)}, {_fmt(rr['effect_unit_boot_ci_high'], 3)}] | "
                    f"{int(rr['n_effect_positive'])}/{int(rr['n_effect_nonzero'])} |"
                )
            lines.append(f"- compact/full effect ratio: `{_fmt(ratio, 3)}`")
    if closure_audit:
        lines.append("")
        lines.append("Provenance audit:")
        lines.append(f"- CUDA/device: `{closure_audit.get('manifest_device', 'NA')}`")
        lines.append(f"- sessions: `{closure_audit.get('n_sessions_manifest', 'NA')}`")
        lines.append(f"- max samples per session: `{closure_audit.get('manifest_max_samples', 'NA')}`")
        lines.append(f"- null repeats: `{closure_audit.get('manifest_n_nulls', 'NA')}`")
        lines.append(f"- raw target total trace: `{_fmt(closure_audit.get('target_trace_raw_total'), 3)}`")
        lines.append(f"- PSD target total trace: `{_fmt(closure_audit.get('target_trace_psd_total'), 3)}`")
        lines.append(f"- clipped negative eigenspectrum mass: `{_fmt(closure_audit.get('target_negative_eigenvalue_mass_total_raw'), 3)}`")

    lines.append("")
    lines.append("## Interpretation Guardrails")
    lines.append("- Panel F supports a robust first-order retinal-translation component of recorded FEM shared variability; it does not claim that all recorded FEM covariance is explained.")
    lines.append("- The compact-k10 inset tests whether the covariance bridge survives restriction to the compact tangent geometry. In the current run, the controlled compact effect is essentially the same size as the controlled full finite-difference effect.")
    lines.append("- PSD-clipped targets are shown in the main panel; raw target rows are retained in the finite-difference CSVs and were also positive.")
    lines.append("")
    return "\n".join(lines)


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
    union_df    = load_union(paths)
    basis_df    = load_basis(paths)
    cov_df      = load_covariance_bridge(paths)
    panel_f_df  = load_panel_f(paths, k=2, delta=0.25)
    panel_f_natural_df = load_panel_f_natural(paths)
    panel_f_closure_summary_df, panel_f_closure_metrics_df, panel_f_closure_audit = load_panel_f_closure(paths)
    panel_f_compact_closure_summary_df = load_panel_f_compact_closure(paths)
    spec_df     = load_union_spectrum(paths, delta=0.25, n_show=32)

    v1_data       = load_recorded_v1(paths.v1_cache)        if paths.v1_cache     else None
    tangent_data  = load_tangent_family(paths.tangent_maps) if paths.tangent_maps else None
    null_spec_df      = load_null_spectrum_summary(paths, delta=0.25, n_show=32)
    info_df           = load_information(paths)
    info_null_sum_df  = load_information_null_summary(paths)

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
    plot_panel_c(ax_c, spec_df, union_df, null_spec_df=null_spec_df)

    # --- Panels D (generalization), E (information), F (bridge) ---
    ax_d = fig.add_subplot(gs[1, 0])
    plot_panel_e(ax_d, basis_df, paths.basis_source_label)   # labeled "D" inside

    ax_e = fig.add_subplot(gs[1, 1])
    plot_panel_e_information(ax_e, info_null_sum_df, info_df)

    ax_f = fig.add_subplot(gs[1, 2])
    _fem = paths.panel_f_fem_ranges or {}
    _metric = paths.panel_f_metric
    # Bands always in arcmin; pass local_eye_sd so cloud_scale x-axes are converted
    _drift_band = tuple(_fem["drift_band_arcmin"]) if "drift_band_arcmin" in _fem else (1.0, 3.0)
    _msac_band  = tuple(_fem["msac_band_arcmin"])  if "msac_band_arcmin"  in _fem else (10.0, 50.0)
    if panel_f_closure_summary_df is not None and panel_f_closure_metrics_df is not None:
        plot_panel_f_closure(
            ax_f,
            panel_f_closure_summary_df,
            panel_f_closure_metrics_df,
            panel_f_closure_audit,
            compact_summary=panel_f_compact_closure_summary_df,
        )
    elif panel_f_natural_df is not None and len(panel_f_natural_df):
        plot_panel_f_natural(ax_f, panel_f_natural_df,
                             drift_band=_drift_band,
                             microsaccade_band=_msac_band)
    else:
        # fisher_r2 cloud_scale is already an arcmin displacement sigma; covariance
        # style metrics use dimensionless cloud_scale multipliers.
        _arcmin_per_cs = None if _metric == "fisher_r2" else _fem.get("local_eye_sd_arcmin")
        plot_panel_f(ax_f, panel_f_df, metric=_metric,
                     drift_band=_drift_band, microsaccade_band=_msac_band,
                     arcmin_per_cloud_scale=_arcmin_per_cs)

    fig.suptitle(
        "Image-specific retinal translations form a compact, image-generalizing reafferent geometry",
        fontsize=10.5, fontweight="bold", x=0.52, y=0.965)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"covTFTS_figure.{ext}", dpi=dpi, bbox_inches="tight")

    caption = """Figure 4. Image-specific retinal translations form a compact, image-generalizing reafferent geometry.

(A) Recorded V1 anchor. Conditioning on measured eye position reduced mean noise correlations, and the FEM-linked covariance component removed by this conditioning was low-dimensional.
(B) Image-specific local translation charts in the digital twin. Each point is a full stimulus-history object projected into response PCA space. Arrows show local response tangents produced by small horizontal and vertical retinal translations, \\(b_x(I)\\) and \\(b_y(I)\\). Tangent directions were content-dependent and did not define a universal signed x/y population axis.
(C) Compact tangent spectrum. The cumulative variance spectrum of the pooled tangent family was substantially more compact than unit-shuffled controls. At 0.25 arcmin, the observed participation ratio was approximately 9, compared with approximately 31 for the unit-shuffled null.
(D) Cross-image generalization. A tangent basis learned from one set of image identities captured held-out translation tangent variance above unit-shuffled nulls. A 10-dimensional image-disjoint basis captured approximately 0.50 of held-out tangent variance versus approximately 0.11 under the null, with no image-ID leakage.
(E) Tangent subspace captures FEM-related displacement sensitivity. A derivative-projection Poisson Fisher analysis measured local sensitivity to small counterfactual retinal translations along real-FEM and stabilized stimulus histories. The image-disjoint k=10 tangent basis captured approximately 0.53 of the real-versus-stabilized local spatial-displacement Fisher gain on held-out images, far above unit-shuffled and random null bases. The orthogonal complement contains the remaining partition of the full Fisher gain.
(F) Finite-difference translation covariances computed in matched twin/recorded unit space captured recorded FEM covariance above a unit-shuffle null. The effect persisted after removing global-rate and target-PC1 modes, indicating a reliable first-order retinal-translation component of recorded FEM shared variability. Points show sessions; purple markers show means and confidence intervals. The y-axis is excess covariance capture over a unit-shuffled source-basis null. Capture remained positive across 24/24 sessions for all projection controls, including after removing both global-rate and target-PC1 components (mean +0.177 [0.144, 0.212]; sign p = 1.2e-07). Inset compares the controlled full finite-difference result with finite-difference responses cross-fit through the compact k=10 tangent subspace. PSD-clipped targets are shown; raw targets were also positive.
"""
    (out_dir / "caption.md").write_text(caption, encoding="utf-8")
    (out_dir / "caption.txt").write_text(caption, encoding="utf-8")
    methods = build_methods_text(
        paths=paths,
        v1_data=v1_data,
        tangent_data=tangent_data,
        union_df=union_df,
        basis_df=basis_df,
        info_null_sum_df=info_null_sum_df,
        info_df=info_df,
        closure_summary_df=panel_f_closure_summary_df,
        closure_audit=panel_f_closure_audit,
        compact_closure_summary_df=panel_f_compact_closure_summary_df,
    )
    (out_dir / "methods.md").write_text(methods, encoding="utf-8")
    (out_dir / "methods.txt").write_text(methods, encoding="utf-8")

    manifest = {
        "figure": "covTFTS_figure",
        "caption_file": str(out_dir / "caption.md"),
        "methods_file": str(out_dir / "methods.md"),
        "source_files": {
            "v1_cache": str(paths.v1_cache),
            "tangent_maps": str(paths.tangent_maps),
            "union_file": str(paths.union_file),
            "spec_file": str(paths.spec_file),
            "basis_file": str(paths.basis_file),
            "covariance_file": str(paths.covariance_file),
            "information_file": str(paths.information_file),
            "information_null_file": str(paths.information_null_file),
            "panel_f_natural_file": str(paths.panel_f_natural_file),
            "panel_f_closure_summary_file": str(paths.panel_f_closure_summary_file),
            "panel_f_closure_metrics_file": str(paths.panel_f_closure_metrics_file),
            "panel_f_closure_audit_file": str(paths.panel_f_closure_audit_file),
            "panel_f_compact_closure_summary_file": str(paths.panel_f_compact_closure_summary_file),
        },
        "basis_source_label": paths.basis_source_label,
        "warnings": paths.warnings,
        "panel_D_cross_image_generalization": basis_df.to_dict(orient="records"),
        "panel_E_tangent_subspace_information": info_df.to_dict(orient="records") if info_df is not None else None,
        "panel_F_operating_regimes": {
            "metric": (
                "finite_difference_closure"
                if panel_f_closure_summary_df is not None and panel_f_closure_metrics_df is not None
                else "natural_structure" if panel_f_natural_df is not None
                else _metric
            ),
            "summary_file": str(paths.panel_f_file) if paths.panel_f_file is not None else None,
            "closure_summary_file": str(paths.panel_f_closure_summary_file) if paths.panel_f_closure_summary_file is not None else None,
            "closure_metrics_file": str(paths.panel_f_closure_metrics_file) if paths.panel_f_closure_metrics_file is not None else None,
            "natural_structure_file": str(paths.panel_f_natural_file) if paths.panel_f_natural_file is not None else None,
            "fem_ranges": paths.panel_f_fem_ranges,
            "rows": (
                panel_f_closure_summary_df.to_dict(orient="records")
                if panel_f_closure_summary_df is not None and panel_f_closure_metrics_df is not None
                else panel_f_natural_df.to_dict(orient="records")
                if panel_f_natural_df is not None
                else panel_f_df.to_dict(orient="records") if panel_f_df is not None else None
            ),
        },
    }
    with open(out_dir / "panel_data_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    readme = f"""# covTFTS Figure 4 (data-forward version)

Generated by `declan/fig4_cov_TFTS/generate_covTFTS_figure.py`.

Full caption sidecars: `caption.md`, `caption.txt`.
Panel methods sidecars: `methods.md`, `methods.txt`.

## Source files
- V1 recorded data: `{paths.v1_cache}`
- Tangent maps:     `{paths.tangent_maps}`
- Union spectrum:   `{paths.spec_file}`
- Union summary:    `{paths.union_file}`
- Basis (image-disjoint): `{paths.basis_file}`
- Covariance bridge: `{paths.covariance_file}`
- Panel E Fisher summary: `{paths.information_file}`
- Panel E basis/null summary: `{paths.information_null_file}`
- Panel F natural structure: `{paths.panel_f_natural_file}`
- Panel F finite-difference closure summary: `{paths.panel_f_closure_summary_file}`
- Panel F finite-difference closure metrics: `{paths.panel_f_closure_metrics_file}`
- Panel F compact-k10 closure summary: `{paths.panel_f_compact_closure_summary_file}`

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
E. Tangent subspace captures FEM-related local displacement sensitivity from
   the production Panel E derivative-projection Fisher output when available;
   otherwise a neutral pending-production placeholder.
F. Finite-difference translation covariances computed in matched twin/recorded
   unit space captured recorded FEM covariance above a unit-shuffle null. The
   effect persisted after removing global-rate and target-PC1 modes, indicating
   a reliable first-order retinal-translation component of recorded FEM shared
   variability. Shows session dots and session-bootstrap mean/CI across
   projection controls. Inset compares the controlled full finite-difference
   source with the cross-fit compact-k10 restricted source when available. Falls
   back to older Panel F diagnostics only if the finite-difference closure CSVs
   are absent.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    return fig, manifest


def parse_args():
    default_tfts = VISIONCORE_ROOT / "outputs" / "twin_feature_tangent_structure_prod_v2"
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
