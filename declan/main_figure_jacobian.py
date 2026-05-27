"""
Main multipanel figure: Jacobian / identity-transformation geometry.

Panels A-E communicate the central Jacobian-mimicry-phase story for the FEM
V1 paper. All data loads from pre-cached NPZ/CSV files; no model inference.

Layout (183 mm x 120 mm, Nature double-column):
  Row 0: A (schematic)  B (subspace alignment)  C (mimicry matrices)
  Row 1: D (crossover)  E (phase heatmaps, spans cols 1-2)

Usage:
    python declan/main_figure_jacobian.py

Outputs to: outputs/figures/jacobian_main_figure/
  main_figure.pdf   -- vector PDF for submission
  main_figure.png   -- 300 dpi raster for review
  manifest.json     -- provenance: exact paths, versions, seed, git commit
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VisionCore.paths import FIGURES_DIR
from declan.geometry_utils import (
    subspace_overlap,
    maybe_git_commit,
    dump_json,
    format_logmar,
    ORIENTATIONS,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DECLAN = ROOT / "declan"
JACOBIAN_DIR = DECLAN / "jacobian_results"
MIMICRY_NPZ = (
    DECLAN / "results" / "translation_mimicry_primary" / "translation_mimicry_by_logmar.npz"
)
PHASE_NPZ = DECLAN / "results" / "phase_landscape_fine" / "phase_landscape_metrics.npz"
LOGMARS_B = [-0.20, -0.25, -0.30, -0.35, -0.40]

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

REAL_HEX = "#1F4E79"   # deep blue
STAB_HEX = "#C45A11"   # rust orange
NULL_HEX = "#9CA3AF"   # medium gray
HMAP_CMAP = "viridis"   # phase landscape heatmap
HMAP_CMAP_C = "Blues"   # mimicry matrices — blue at high end, consistent with Panel A
PPD = 37.5             # model pixels per degree


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
    })


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def add_panel_label(ax: plt.Axes, label: str, x: float = -0.22, y: float = 1.04) -> None:
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def add_subtitle(ax: plt.Axes, text: str, y: float = 1.003) -> None:
    """Conceptual one-liner below the main panel title, in muted italic."""
    ax.text(
        0.5, y, text,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=5.5, color="#888888", fontstyle="italic",
        clip_on=False,
    )


def _lm_ticks(lm_arr: np.ndarray) -> list[str]:
    return [f"{v:.2f}" for v in lm_arr]


# ---------------------------------------------------------------------------
# Panel A — conceptual schematic
# ---------------------------------------------------------------------------

def _draw_e_glyph(
    ax: plt.Axes,
    cx: float,
    cy: float,
    size: float,
    color: str = "k",
    alpha: float = 1.0,
) -> None:
    """Snellen-E glyph (prongs pointing right) drawn as filled rectangles."""
    w, h = size, size
    bw = w * 0.22   # backbone (vertical bar) width
    bh = h * 0.18   # bar height

    parts = [
        # vertical backbone
        Rectangle((cx - w / 2, cy - h / 2), bw, h),
        # top prong
        Rectangle((cx - w / 2 + bw, cy + h / 2 - bh), w * 0.78, bh),
        # middle prong (slightly shorter)
        Rectangle((cx - w / 2 + bw, cy - bh / 2), w * 0.62, bh),
        # bottom prong
        Rectangle((cx - w / 2 + bw, cy - h / 2), w * 0.78, bh),
    ]
    for p in parts:
        p.set_facecolor(color)
        p.set_edgecolor("none")
        p.set_alpha(alpha)
        ax.add_patch(p)


def draw_panel_a(fig: plt.Figure, gs_cell) -> None:
    """Pure schematic: E-optotype translations → Jacobian decomposition diagram."""
    ax_outer = fig.add_subplot(gs_cell)
    ax_outer.set_axis_off()

    # Sub-axes: left (E glyphs), middle (mapping arrow), right (decomposition).
    # Bottom margin reserved in ax_r for component labels below the band.
    ax_l = ax_outer.inset_axes([0.00, 0.08, 0.28, 0.88])
    ax_m = ax_outer.inset_axes([0.30, 0.28, 0.09, 0.46])
    ax_r = ax_outer.inset_axes([0.42, 0.05, 0.58, 0.91])

    for ax in (ax_l, ax_m, ax_r):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # ---- Left: reference E + two orthogonal translations ----
    s = 0.20
    E_ref = (0.28, 0.42)    # reference
    E_dx  = (0.68, 0.42)    # x-translated (same y)
    E_dy  = (0.28, 0.75)    # y-translated (same x)

    _draw_e_glyph(ax_l, *E_ref, s, "#111111", 1.0)
    _draw_e_glyph(ax_l, *E_dx,  s, "#666666", 0.72)
    _draw_e_glyph(ax_l, *E_dy,  s, "#666666", 0.72)

    ax_l.text(E_ref[0], E_ref[1] - s / 2 - 0.07,
              "Reference", fontsize=5.8, ha="center", va="top", color="#111111")

    ak = dict(arrowstyle="-|>", color="#999999", lw=0.9, mutation_scale=6,
              shrinkA=5, shrinkB=5)
    ax_l.annotate("", xy=E_dx, xytext=E_ref, arrowprops=ak)
    ax_l.annotate("", xy=E_dy, xytext=E_ref, arrowprops=ak)

    # Label each translation separately (no merged "Δx, Δy")
    ax_l.text((E_ref[0] + E_dx[0]) / 2, E_ref[1] + 0.09,
              "+Δx", fontsize=6.0, ha="center", va="bottom", color="#777777")
    ax_l.text(E_dy[0] + 0.09, (E_ref[1] + E_dy[1]) / 2,
              "+Δy", fontsize=6.0, ha="left", va="center", color="#777777")

    ax_l.text(0.50, 0.01, "Retinal translations",
              fontsize=6.0, ha="center", va="bottom", color="#555555")

    # ---- Middle: mapping arrow ----
    ax_m.annotate(
        "", xy=(0.82, 0.50), xytext=(0.18, 0.50),
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.3, mutation_scale=8),
    )
    ax_m.text(0.50, 0.68, "f (·)", fontsize=6.5, ha="center", va="bottom",
              color="#555555", style="italic")

    # ---- Right: 2D vector decomposition ----
    #
    # Geometry (axes coords 0–1):
    #   Tangent plane band: y ∈ [0.30, 0.46]  (raised to give label space below)
    #   A: source orientation, inside band at (0.13, 0.38)
    #   B: target orientation, above band at (0.76, 0.91)
    #   F: foot of perpendicular on plane (same y as A, same x as B) = (0.76, 0.38)

    A = np.array([0.13, 0.38])
    B = np.array([0.76, 0.91])
    F = np.array([B[0], A[1]])

    band_y0, band_y1 = 0.30, 0.46
    band = mpatches.FancyBboxPatch(
        (0.03, band_y0), 0.91, band_y1 - band_y0,
        boxstyle="round,pad=0.005",
        facecolor="#D6E8F7", edgecolor="#7AAED4", linewidth=0.8, zorder=1,
    )
    ax_r.add_patch(band)

    fa = dict(mutation_scale=7, zorder=6, shrinkA=0, shrinkB=0)

    # Basis arrows inside the band
    ax_r.annotate("", xy=(0.56, A[1]), xytext=A,
                  arrowprops=dict(**fa, arrowstyle="-|>", color="#5A9DC8", lw=1.0))
    ax_r.text(0.58, A[1] + 0.03, "$J_x$", fontsize=6.5, ha="left", va="bottom",
              color="#3A7AAD", fontweight="bold")

    ax_r.annotate("", xy=(0.28, A[1] + 0.10), xytext=A,
                  arrowprops=dict(**fa, arrowstyle="-|>", color="#5A9DC8", lw=1.0))
    ax_r.text(0.29, A[1] + 0.12, "$J_y$", fontsize=6.5, ha="left", va="bottom",
              color="#3A7AAD", fontweight="bold")

    # Total identity vector d_{a→b} (drawn behind components)
    ax_r.annotate("", xy=B, xytext=A,
                  arrowprops=dict(**fa, arrowstyle="-|>", color="#1A1A1A", lw=1.3))
    mid_d = (A + B) / 2 + np.array([-0.12, 0.01])
    ax_r.text(*mid_d, r"$d_{a \to b}$", fontsize=7.5, ha="center", va="center",
              color="#1A1A1A", rotation=41)

    # In-plane component J_a Δp* (A → F, REAL_HEX)
    ax_r.annotate("", xy=F, xytext=A,
                  arrowprops=dict(**fa, arrowstyle="-|>", color=REAL_HEX, lw=1.7))
    # All labels for the below-band region stack vertically:
    #   1) "Jacobian tangent plane" — structural label for the blue region
    #   2) bold formula J_a Δp*
    #   3) italic descriptor
    # Placing all three below the band keeps the central region (where the arrows
    # are) free of competing text.
    mid_af_x = (A[0] + F[0]) / 2
    ax_r.text(0.50, band_y0 - 0.03, "Jacobian tangent plane",
              fontsize=6.0, ha="center", va="top", color="#3A6E9A",
              fontweight="semibold", clip_on=False)
    ax_r.text(mid_af_x, band_y0 - 0.13,
              r"$J_a \Delta p^*$",
              fontsize=7.5, ha="center", va="top", color=REAL_HEX, fontweight="bold",
              clip_on=False)
    ax_r.text(mid_af_x, band_y0 - 0.24,
              "mimicked by translation",
              fontsize=5.8, ha="center", va="top", color=REAL_HEX, style="italic",
              clip_on=False)

    # Orthogonal component d⊥ (F → B, STAB_HEX)
    ax_r.annotate("", xy=B, xytext=F,
                  arrowprops=dict(**fa, arrowstyle="-|>", color=STAB_HEX, lw=1.7))
    mid_fb_y = (F[1] + B[1]) / 2
    ax_r.text(F[0] + 0.05, mid_fb_y + 0.05,
              r"$d^\perp$",
              fontsize=7.5, ha="left", va="center", color=STAB_HEX, fontweight="bold",
              clip_on=False)
    # Two-line wrap keeps each line short enough to avoid right-margin clipping
    ax_r.text(F[0] + 0.05, mid_fb_y - 0.06,
              "irreducibly\nidentity",
              fontsize=5.8, ha="left", va="top", color=STAB_HEX, style="italic",
              linespacing=1.25, clip_on=False)

    # Dashed projection lines
    ax_r.plot([A[0], F[0]], [A[1], A[1]], "--", color="#CCCCCC", lw=0.8, zorder=4)
    ax_r.plot([F[0], F[0]], [F[1], B[1]], "--", color="#CCCCCC", lw=0.8, zorder=4)

    # Right-angle indicator at F
    sq = 0.032
    ax_r.plot([F[0] - sq, F[0] - sq, F[0]],
              [F[1], F[1] + sq, F[1] + sq],
              color="#BBBBBB", lw=0.8, zorder=5)

    # Points: only A and B — consistent filled black circles, no gray F dot
    ax_r.plot(*A, "o", color="#1A1A1A", ms=5.5, zorder=9)
    ax_r.plot(*B, "o", color="#1A1A1A", ms=5.5, zorder=9)
    ax_r.text(A[0] - 0.05, A[1] + 0.05, "Ori A", fontsize=6.0, ha="right",
              va="bottom", color="#1A1A1A")
    ax_r.text(B[0] + 0.03, B[1], "Ori B", fontsize=6.0, ha="left",
              va="center", color="#1A1A1A")

    add_panel_label(ax_l, "A", x=-0.06, y=1.04)
    # Conceptual subtitle anchored to the right sub-axis (where the core diagram is)
    ax_r.text(0.50, 1.03, "Translation = local tangent plane",
              transform=ax_r.transAxes, ha="center", va="bottom",
              fontsize=5.5, color="#888888", fontstyle="italic", clip_on=False)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_test3(logmar: float) -> tuple[dict, str]:
    lm_str = format_logmar(logmar)
    for name in (f"test3_lm{lm_str}.npz", f"test3_lm{lm_str}_grid7.npz"):
        p = JACOBIAN_DIR / name
        if p.exists():
            raw = np.load(p, allow_pickle=True)
            return {k: raw[k] for k in raw.files}, str(p)
    raise FileNotFoundError(
        f"Jacobian bundle not found for lm={lm_str} in {JACOBIAN_DIR}"
    )


def _compute_null_alignment(
    U_jac: np.ndarray,
    C_FEM: np.ndarray,
    n_samples: int = 500,
    var_threshold: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Mean ± std of subspace overlap between U_jac and random 2-D subspaces
    drawn from the top-K eigenvectors of C_FEM (matched null).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    vals, vecs = np.linalg.eigh(C_FEM)
    # Sort descending
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vals = np.clip(vals, 0.0, None)
    total = vals.sum() + 1e-12
    cumvar = np.cumsum(vals) / total
    K = int(np.searchsorted(cumvar, var_threshold)) + 1
    K = max(K, 4)
    U_top = vecs[:, :K]   # (n_neurons, K)

    # Pre-compute U_top.T @ U_jac once; sample in K-space for speed
    UTU = U_top.T @ U_jac   # (K, 2)
    overlaps = np.empty(n_samples)
    for i in range(n_samples):
        Z = rng.standard_normal((K, 2))
        Q, _ = np.linalg.qr(Z)         # (K, 2) — random 2-D subspace in K-space
        M = Q.T @ UTU                   # (2, 2)
        sv = np.linalg.svd(M, compute_uv=False)
        sv = np.clip(sv, 0.0, 1.0)
        overlaps[i] = float(np.mean(sv ** 2))
    return float(overlaps.mean()), float(overlaps.std())


def load_panel_b_data(rng: np.random.Generator) -> dict:
    records: list[dict] = []
    paths: list[str] = []
    for lm in LOGMARS_B:
        data, path = _load_test3(lm)
        paths.append(path)
        for ori in ORIENTATIONS:
            U_jac = np.asarray(data[f"U_jac_ori{ori}"], dtype=np.float64)
            U_pca = np.asarray(data[f"U_pca2_ori{ori}"], dtype=np.float64)
            C_FEM = np.asarray(data[f"C_FEM_ori{ori}"], dtype=np.float64)
            align = subspace_overlap(U_jac, U_pca)
            null_mean, null_std = _compute_null_alignment(U_jac, C_FEM, rng=rng)
            records.append({
                "logmar": float(lm),
                "orientation": int(ori),
                "alignment": float(align),
                "null_mean": float(null_mean),
                "null_std": float(null_std),
            })
    return {"records": records, "paths": paths}


def load_panel_cde_data() -> dict:
    raw = np.load(MIMICRY_NPZ, allow_pickle=True)
    return {
        "logmars": np.asarray(raw["logmars"], dtype=np.float64),
        "conditions": list(raw["conditions"]),
        "mimicry": np.asarray(raw["mimicry_raw"], dtype=np.float64),   # (5,2,4,4)
        "path": str(MIMICRY_NPZ),
    }


def load_panel_e_data() -> dict:
    raw = np.load(PHASE_NPZ, allow_pickle=True)
    files = set(raw.files)

    def _get(key: str):
        return np.asarray(raw[key]) if key in files else None

    return {
        "logmars": np.asarray(raw["logmars"], dtype=np.float64),
        "mean_mimicry": np.asarray(raw["mean_mimicry"], dtype=np.float64),
        "offset_x_pix": np.asarray(raw["offset_x_pix"], dtype=np.float64),
        "offset_y_pix": np.asarray(raw["offset_y_pix"], dtype=np.float64),
        "eye_trial_mean_xy": _get("eye_trial_mean_xy"),
        "eye_frame_xy": _get("eye_frame_xy"),
        "grand_mean_eye_pos": _get("grand_mean_eye_pos"),
        "path": str(PHASE_NPZ),
    }


# ---------------------------------------------------------------------------
# Panel B — subspace alignment vs null
# ---------------------------------------------------------------------------

def draw_panel_b(ax: plt.Axes, b_data: dict) -> None:
    records = b_data["records"]
    lm_vals = sorted({r["logmar"] for r in records})   # ascending (most neg first)
    lm_arr = np.array(lm_vals)

    align_grid = np.array(
        [[r["alignment"] for r in records if r["logmar"] == lm] for lm in lm_vals]
    )   # (5, 4)
    null_mean_grid = np.array(
        [[r["null_mean"] for r in records if r["logmar"] == lm] for lm in lm_vals]
    )
    null_std_grid = np.array(
        [[r["null_std"] for r in records if r["logmar"] == lm] for lm in lm_vals]
    )

    align_mean = align_grid.mean(axis=1)
    null_mean = null_mean_grid.mean(axis=1)
    null_std = null_std_grid.mean(axis=1)

    # Null band (no separate legend entry for dashed mean — explained by label)
    ax.fill_between(
        lm_arr,
        null_mean - 2 * null_std,
        null_mean + 2 * null_std,
        color=NULL_HEX, alpha=0.22, zorder=2,
    )
    ax.plot(lm_arr, null_mean, "--", color=NULL_HEX, lw=0.9, zorder=3,
            label="Null mean ± 2 SD")

    # Per-orientation scatter (thin, unlabeled)
    for col in range(align_grid.shape[1]):
        ax.plot(lm_arr, align_grid[:, col], "o", color=REAL_HEX,
                ms=3.0, alpha=0.40, zorder=4)

    # Mean line (bold)
    ax.plot(lm_arr, align_mean, "o-", color=REAL_HEX, lw=1.8, ms=5.0,
            zorder=5, label="FEM (mean over ori.)")

    # Annotation: "N× above null" — placed mid-panel to avoid title collision
    ratio = align_mean / (null_mean + 1e-9)
    if ratio.min() > 2.0:
        lo, hi = ratio.min(), ratio.max()
        ax.text(0.97, 0.55, f"{lo:.0f}–{hi:.0f}×\nabove null",
                transform=ax.transAxes, ha="right", va="center",
                fontsize=6.0, color=REAL_HEX)

    ax.set_xlabel("LogMAR")
    ax.set_ylabel("Subspace alignment")
    ax.set_title("Subspace alignment vs null", pad=12)
    add_subtitle(ax, "FEM responses lie along the Jacobian plane")
    ax.set_xticks(lm_arr)
    ax.set_xticklabels(_lm_ticks(lm_vals), rotation=40, ha="right")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, loc="upper right", handlelength=1.2)
    add_panel_label(ax, "B")


# ---------------------------------------------------------------------------
# Panel C — pairwise mimicry matrices
# ---------------------------------------------------------------------------

def draw_panel_c(fig: plt.Figure, gs_cell, c_data: dict) -> None:
    mimicry = c_data["mimicry"]    # (5, 2, 4, 4)
    logmars = c_data["logmars"]
    ORI_LABELS = ["0°", "90°", "180°", "270°"]

    # Journal-size decision rule (hardcoded geometry)
    target_mm = 183.0
    panel_c_frac = 1.3 / (1.5 + 1.0 + 1.3)
    panel_c_mm = target_mm * panel_c_frac
    matrix_mm = (panel_c_mm - 10.0) / 2.0
    USE_SINGLE = matrix_mm < 20.0
    print(f"  Panel C: estimated matrix size = {matrix_mm:.1f} mm "
          f"→ {'single matrix' if USE_SINGLE else '2×2 layout'}")

    lm_020 = int(np.argmin(np.abs(logmars - (-0.20))))
    lm_035 = int(np.argmin(np.abs(logmars - (-0.35))))

    # Normalize colormap to actual data range (not 0–1) for better discrimination
    all_off_diag = []
    for li in (lm_020, lm_035):
        for ci in (0, 1):
            m = mimicry[li, ci].copy()
            np.fill_diagonal(m, np.nan)
            vals = m.ravel()
            all_off_diag.extend(vals[~np.isnan(vals)].tolist())
    vmin = max(0.0, float(np.percentile(all_off_diag, 2)))
    vmax = min(1.0, float(np.percentile(all_off_diag, 98)))

    def _masked(lm_i: int, cond_i: int) -> np.ndarray:
        m = mimicry[lm_i, cond_i].copy()
        np.fill_diagonal(m, np.nan)
        return m

    def _decorate(ax: plt.Axes, title: str, show_xlabel: bool, show_ylabel: bool) -> None:
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(ORI_LABELS, fontsize=5.5)
        ax.set_yticklabels(ORI_LABELS, fontsize=5.5)
        ax.set_title(title, fontsize=6.8, pad=2)
        if show_xlabel:
            ax.set_xlabel("Target ori.", fontsize=6.5)
        if show_ylabel:
            ax.set_ylabel("Source ori.", fontsize=6.5)

    if USE_SINGLE:
        ax = fig.add_subplot(gs_cell)
        im = ax.imshow(_masked(lm_020, 0), vmin=vmin, vmax=vmax,
                       cmap=HMAP_CMAP_C, aspect="equal")
        _decorate(ax, "Pairwise mimicry\n(real FEM, lm=−0.20)", True, True)
        plt.colorbar(im, ax=ax, shrink=0.82, label="Mimicry")
        add_panel_label(ax, "C")
        return

    sub_gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_cell,
                                     hspace=0.40, wspace=0.22)
    configs = [
        (lm_020, 0, "Real, lm=−0.20",   False, True),
        (lm_020, 1, "Stab., lm=−0.20",  False, False),
        (lm_035, 0, "Real, lm=−0.35",   True,  True),
        (lm_035, 1, "Stab., lm=−0.35",  True,  False),
    ]
    axes_c: list[plt.Axes] = []
    ims_c = []
    for k, (li, ci, title, xl, yl) in enumerate(configs):
        row, col = divmod(k, 2)
        ax = fig.add_subplot(sub_gs[row, col])
        im = ax.imshow(_masked(li, ci), vmin=vmin, vmax=vmax,
                       cmap=HMAP_CMAP_C, aspect="equal")
        _decorate(ax, title, xl, yl)
        axes_c.append(ax)
        ims_c.append(im)

    # Pair callouts on the top-left matrix (Real, lm=−0.20).
    # Box around the cell; text label outside the matrix with a curved arrow.
    m0 = _masked(lm_020, 0)
    mask_valid = ~np.isnan(m0)
    if mask_valid.any():
        flat_vals = np.where(mask_valid, m0, np.nan).ravel()
        r_hi, c_hi = np.unravel_index(int(np.nanargmax(flat_vals)), m0.shape)
        r_lo, c_lo = np.unravel_index(int(np.nanargmin(flat_vals)), m0.shape)
        for (r, c, lbl, text_corner, corner_ha, corner_va) in [
            (r_hi, c_hi, "high", (0.98, 0.98), "right", "top"),
            (r_lo, c_lo, "low",  (0.02, 0.02), "left",  "bottom"),
        ]:
            # Thin neutral box so the cell color is still visible
            rect = mpatches.Rectangle(
                (c - 0.49, r - 0.49), 0.98, 0.98,
                fill=False, edgecolor="#555555", linewidth=0.9, zorder=10,
            )
            axes_c[0].add_patch(rect)
            # Text at an axis corner with a curved arrow pointing to the cell
            axes_c[0].annotate(
                lbl,
                xy=(c, r), xycoords="data",
                xytext=text_corner, textcoords="axes fraction",
                ha=corner_ha, va=corner_va,
                fontsize=6.0, fontweight="bold", color="#222222",
                arrowprops=dict(
                    arrowstyle="-|>", color="#555555", lw=0.8,
                    mutation_scale=5, shrinkA=2, shrinkB=7,
                    connectionstyle="arc3,rad=0.25",
                ),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
                zorder=12, clip_on=False,
            )

    # Colorbar first so its position is known before we use get_position()
    fig.colorbar(ims_c[-1], ax=axes_c, shrink=0.55, label="Mimicry",
                 fraction=0.045, pad=0.08)

    # Subtitle centered above the entire 2×2 matrix block (figure coordinates)
    x0 = axes_c[0].get_position().x0
    x1 = axes_c[1].get_position().x1
    y1 = max(axes_c[0].get_position().y1, axes_c[1].get_position().y1)
    fig.text(
        (x0 + x1) / 2, y1 + 0.005,
        "Some identity contrasts are mimicked by translation",
        ha="center", va="bottom",
        fontsize=5.5, color="#888888", fontstyle="italic",
        transform=fig.transFigure,
    )

    add_panel_label(axes_c[0], "C", x=-0.42, y=1.22)


# ---------------------------------------------------------------------------
# Panel D — crossover line plot
# ---------------------------------------------------------------------------

def draw_panel_d(ax: plt.Axes, c_data: dict) -> None:
    mimicry = c_data["mimicry"]    # (5, 2, 4, 4)
    logmars = np.asarray(c_data["logmars"])

    # Mask diagonal, flatten, compute mean + SEM per logmar × condition
    m = mimicry.copy()
    for i in range(4):
        m[:, :, i, i] = np.nan
    flat = m.reshape(5, 2, 16)    # 12 non-NaN entries per matrix

    real_mean = np.nanmean(flat[:, 0], axis=-1)
    stab_mean = np.nanmean(flat[:, 1], axis=-1)
    n_valid = np.sum(~np.isnan(flat[:, 0]), axis=-1).astype(float)
    real_sem = np.nanstd(flat[:, 0], axis=-1) / np.sqrt(np.clip(n_valid, 1, None))
    stab_sem = np.nanstd(flat[:, 1], axis=-1) / np.sqrt(np.clip(n_valid, 1, None))

    # Separate primary range (lm ≥ −0.35) from saturation control (lm = −0.40)
    primary = logmars >= -0.35 - 1e-6   # True for indices 0..3
    sat_idx = ~primary                   # index 4

    # Shade saturation control region
    sat_x = float(logmars[sat_idx][0])
    ax.axvspan(sat_x - 0.025, sat_x + 0.025, color="#EBEBEB", zorder=0)

    kw = dict(capsize=2.5, capthick=0.7, elinewidth=0.7, zorder=5)
    ax.errorbar(logmars[primary], real_mean[primary], yerr=real_sem[primary],
                color=REAL_HEX, marker="o", ms=4.5, lw=1.8,
                label="Real FEM", **kw)
    ax.errorbar(logmars[primary], stab_mean[primary], yerr=stab_sem[primary],
                color=STAB_HEX, marker="o", ms=4.5, lw=1.8,
                label="Trial-mean stab.", **kw)

    # Continuation lines into saturation (lighter)
    for arr, col in [(real_mean, REAL_HEX), (stab_mean, STAB_HEX)]:
        ax.plot(logmars[-2:], arr[-2:], "o--", color=col, ms=3.5,
                lw=0.9, alpha=0.40, zorder=4)

    # y_top: highest data value, used to place in-plot annotations near the top
    y_top = max(float(np.nanmax(real_mean)), float(np.nanmax(stab_mean)))

    # Saturation label — inside the gray shaded region near the top
    ax.text(sat_x, y_top * 0.96, "sat.\nctrl.",
            ha="center", va="top",
            fontsize=5.5, color="#333333", fontweight="bold", zorder=2)

    # Crossover annotation (single point, computed from data)
    signs = np.sign(real_mean[primary] - stab_mean[primary])
    lm_primary = logmars[primary]
    crossover_idx = None
    for i in range(len(signs) - 1):
        if signs[i] != 0 and signs[i + 1] != 0 and signs[i] * signs[i + 1] < 0:
            crossover_idx = i
            break
    if crossover_idx is not None:
        x_cross = float((lm_primary[crossover_idx] + lm_primary[crossover_idx + 1]) / 2)
        ax.axvline(x_cross, color="#AAAAAA", lw=0.9, linestyle=":", zorder=3)
        # In-plot label at top of data range, right of the line
        ax.text(x_cross + 0.005, y_top * 0.96,
                f"crossover\n(lm ≈ {x_cross:.2f})",
                ha="left", va="top", fontsize=5.5, color="#333333", fontweight="bold",
                linespacing=1.2, zorder=6)

    ax.set_xlabel("LogMAR")
    ax.set_ylabel("Mean pairwise mimicry")
    ax.set_title("Mean pairwise mimicry", pad=12)
    add_subtitle(ax, "Scale-dependent confusability")
    ax.set_xticks(logmars)
    ax.set_xticklabels(_lm_ticks(logmars), rotation=40, ha="right")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, loc="lower right", handlelength=1.2)
    add_panel_label(ax, "D")


# ---------------------------------------------------------------------------
# Panel E — fine phase heatmaps
# ---------------------------------------------------------------------------

def draw_panel_e(fig: plt.Figure, gs_cell, e_data: dict) -> None:
    mm_arr = e_data["mean_mimicry"]        # (2, 33, 33)
    ox_pix = e_data["offset_x_pix"]        # (33,)
    oy_pix = e_data["offset_y_pix"]        # (33,)
    logmars = e_data["logmars"]            # (2,)
    eye_tm = e_data["eye_trial_mean_xy"]
    eye_frames = e_data["eye_frame_xy"]

    # Print shapes for debugging / QC
    print(f"  Phase landscape logmars: {logmars}")
    print(f"  mean_mimicry shape: {mm_arr.shape}")
    if eye_tm is not None:
        print(f"  eye_trial_mean_xy shape: {eye_tm.shape}")
    if eye_frames is not None:
        print(f"  eye_frame_xy shape: {eye_frames.shape}")

    # Convert to arcmin (37.5 ppd)
    ox_am = ox_pix / PPD * 60.0
    oy_am = oy_pix / PPD * 60.0
    extent = [float(ox_am[0]), float(ox_am[-1]),
              float(oy_am[0]), float(oy_am[-1])]

    finite = mm_arr[np.isfinite(mm_arr)]
    vmin, vmax = float(finite.min()), float(finite.max())

    lm_020 = int(np.argmin(np.abs(logmars - (-0.20))))
    lm_035 = int(np.argmin(np.abs(logmars - (-0.35))))
    lm_indices = [lm_020, lm_035]
    lm_labels = [f"lm = {logmars[i]:+.2f}" for i in lm_indices]

    sub_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_cell, wspace=0.18)
    axes_e: list[plt.Axes] = []
    ims_e = []

    for k, (li, lbl) in enumerate(zip(lm_indices, lm_labels)):
        ax = fig.add_subplot(sub_gs[0, k])

        # Heatmap: mean_mimicry[li] is (33, 33); dim 0 = x, dim 1 = y.
        # For imshow origin='lower': rows = y (bottom-up), cols = x.
        hmap = mm_arr[li].T   # (y, x) = (33, 33)
        im = ax.imshow(
            hmap,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=HMAP_CMAP,
            vmin=vmin,
            vmax=vmax,
        )
        ims_e.append(im)

        # --- Mandatory overlay 1: trial-mean stabilized positions ---
        if eye_tm is not None:
            tm_2d = eye_tm.reshape(-1, 2) if eye_tm.ndim != 2 else eye_tm
            if tm_2d.shape[1] == 2:
                tm_am = tm_2d / PPD * 60.0
                # Subtle scatter for raw cloud
                ax.scatter(
                    tm_am[:, 0], tm_am[:, 1],
                    s=3, color=STAB_HEX, alpha=0.12,
                    linewidths=0, zorder=7, rasterized=True,
                )
                # Contour summarizing the distribution — more legible than raw dots
                try:
                    H, xedge, yedge = np.histogram2d(
                        tm_am[:, 0], tm_am[:, 1], bins=20,
                        range=[[extent[0], extent[1]], [extent[2], extent[3]]],
                    )
                    xc = (xedge[:-1] + xedge[1:]) / 2
                    yc = (yedge[:-1] + yedge[1:]) / 2
                    ax.contour(
                        xc, yc, H.T,
                        levels=3, colors=[STAB_HEX], alpha=0.75,
                        linewidths=1.1, zorder=8,
                    )
                except Exception:
                    pass
                if k == 0:
                    # Invisible proxy for the legend
                    ax.plot([], [], color=STAB_HEX, lw=1.1, label="Trial-mean pos.")

        # --- Mandatory overlay 2: fixed-center marker at (0, 0) ---
        ax.plot(0.0, 0.0, "o", ms=7, color="white",
                markeredgecolor="black", markeredgewidth=1.3, zorder=9,
                label="Fixed center" if k == 0 else "_nolegend_")

        # --- Optional overlay: real FEM density contour ---
        if eye_frames is not None and eye_frames.ndim == 2:
            fr_am = eye_frames / PPD * 60.0
            if len(fr_am) < 30_000:
                try:
                    H, xedge, yedge = np.histogram2d(
                        fr_am[:, 0], fr_am[:, 1], bins=33,
                        range=[[extent[0], extent[1]], [extent[2], extent[3]]],
                    )
                    xc = (xedge[:-1] + xedge[1:]) / 2
                    yc = (yedge[:-1] + yedge[1:]) / 2
                    ax.contour(xc, yc, H.T, levels=3,
                               colors=[REAL_HEX], alpha=0.40,
                               linewidths=0.6, zorder=6)
                except Exception:
                    pass

        ax.set_title(f"Phase landscape\n{lbl}", pad=12)
        if k == 0:
            add_subtitle(ax, "Subpixel phase reshapes confusability")
        ax.set_xlabel("Retinal phase x (arcmin)")
        if k == 0:
            ax.set_ylabel("Retinal phase y (arcmin)")
            ax.legend(frameon=True, loc="lower left", fontsize=5.5,
                      handlelength=1.2, handletextpad=0.4,
                      facecolor="white", edgecolor="none", framealpha=0.85,
                      markerscale=0.9)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        axes_e.append(ax)

    fig.colorbar(ims_e[-1], ax=axes_e, shrink=0.80, label="Mean mimicry",
                 fraction=0.04, pad=0.04)
    add_panel_label(axes_e[0], "E")


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def build_figure(b_data: dict, c_data: dict, e_data: dict) -> plt.Figure:
    fig = plt.figure(
        figsize=(183 / 25.4, 120 / 25.4),  # 183 mm × 120 mm
        constrained_layout=False,
    )

    gs = GridSpec(
        2, 3,
        figure=fig,
        width_ratios=[1.5, 1.0, 1.3],
        height_ratios=[1.0, 1.0],
        hspace=0.62,
        wspace=0.38,
        left=0.07,
        right=0.93,   # pulled in from 0.96 to give Panel C colorbar breathing room
        top=0.94,
        bottom=0.14,
    )

    print("Drawing Panel A...")
    draw_panel_a(fig, gs[0, 0])

    print("Drawing Panel B...")
    ax_b = fig.add_subplot(gs[0, 1])
    draw_panel_b(ax_b, b_data)

    print("Drawing Panel C...")
    draw_panel_c(fig, gs[0, 2], c_data)

    print("Drawing Panel D...")
    ax_d = fig.add_subplot(gs[1, 0])
    draw_panel_d(ax_d, c_data)

    print("Drawing Panel E...")
    draw_panel_e(fig, gs[1, 1:], e_data)

    return fig


# ---------------------------------------------------------------------------
# Save + manifest
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "main_figure.pdf"
    png_path = out_dir / "main_figure.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")


def write_manifest(
    out_dir: Path,
    b_paths: list[str],
    rng_seed: int,
) -> None:
    import matplotlib as mpl_mod
    import numpy as np_mod
    try:
        import scipy as sp_mod
        scipy_ver = sp_mod.__version__
    except ImportError:
        scipy_ver = "not installed"

    manifest = {
        "entry_point": "python declan/main_figure_jacobian.py",
        "random_seed": rng_seed,
        "git_commit": maybe_git_commit(ROOT),
        "software": {
            "matplotlib": mpl_mod.__version__,
            "numpy": np_mod.__version__,
            "scipy": scipy_ver,
            "python": sys.version,
        },
        "data_sources": {
            "panel_A": "none — pure schematic (no data loaded)",
            "panel_B": b_paths,
            "panel_C": str(MIMICRY_NPZ),
            "panel_D": str(MIMICRY_NPZ),
            "panel_E": str(PHASE_NPZ),
        },
        "null_methodology": (
            "Panel B null: random 2-D subspaces drawn from top-K eigenvectors "
            "of C_FEM (K chosen to capture 95% of FEM variance), "
            f"n_samples=500, seed={rng_seed}."
        ),
        "panel_c_decision": "See stdout for matrix-size decision (single vs 2x2).",
        "color_constants": {
            "real_fem": REAL_HEX,
            "trial_mean_stabilized": STAB_HEX,
            "null": NULL_HEX,
            "heatmap_cmap": HMAP_CMAP,
        },
        "figure_dimensions_mm": {"width": 183, "height": 120},
        "ppd": PPD,
    }
    path = out_dir / "manifest.json"
    dump_json(path, manifest)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_style()

    RNG_SEED = 42
    rng = np.random.default_rng(RNG_SEED)

    print("Loading Panel B data (Jacobian bundles + null)...")
    b_data = load_panel_b_data(rng)

    print("Loading Panels C/D data (translation mimicry)...")
    c_data = load_panel_cde_data()

    print("Loading Panel E data (fine phase landscape)...")
    e_data = load_panel_e_data()

    print("Building figure...")
    fig = build_figure(b_data, c_data, e_data)

    out_dir = FIGURES_DIR / "jacobian_main_figure"
    print(f"Saving to {out_dir} ...")
    save_figure(fig, out_dir)
    write_manifest(out_dir, b_data["paths"], RNG_SEED)
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
