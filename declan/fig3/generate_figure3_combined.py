"""Simplified Figure 3: digital twin mechanism plus compact reafferent geometry.

This compositor keeps only the main-text mechanism chain:

  A  Retinal-input digital twin schematic
  B  Empirical vs model FEM modulation
  C  Extraretinal-pathway zeroing control
  D  No universal translation axis
  E  Compact translation-tangent subspace
  F  Image-disjoint generalization
  G  Translation-predicted recorded FEM covariance

The older full figures remain useful as source/supplemental figures. This file
only selects the panels needed for the combined main-text story.

Usage:
    uv run declan/fig3/generate_figure3_combined.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Annotation
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from VisionCore.paths import VISIONCORE_ROOT

from _fig3_data import FIG_DIR, configure_matplotlib, load_fig3_data
from _fig3_ablation_data import CACHE_PATH as ABLATION_CACHE_PATH
from _fig3_ablation_data import load_ablation_data
from generate_fig3d import compute_model_one_minus_alpha


GEOM_DIR = VISIONCORE_ROOT / "declan" / "fig4_cov_TFTS"
if str(GEOM_DIR) not in sys.path:
    sys.path.insert(0, str(GEOM_DIR))

import generate_covTFTS_figure as geom  # noqa: E402


POOLED_COLOR = "0.25"
POOLED_FILL = "0.55"
PANEL_LETTER_SIZE = 11
PANEL_TITLE_SIZE = 9.0


def _panel_title(ax, letter: str):
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=11, pad=4)


def _clear_panel_heading(ax):
    """Remove source-panel headings so the combined figure can place them uniformly."""
    ax.set_title("", loc="left")
    ax.set_title("", loc="center")
    ax.set_title("", loc="right")
    for txt in list(ax.texts):
        if txt.get_transform() == ax.transAxes:
            x, y = txt.get_position()
            if y >= 0.98 and x <= 0.28:
                txt.remove()


def _standard_panel_heading(
    ax,
    letter: str,
    title: str,
    *,
    y: float = 1.045,
    title_x: float = 0.08,
    title_size: float = PANEL_TITLE_SIZE,
):
    """Place a consistent panel letter/title just above the axes."""
    _clear_panel_heading(ax)
    ax.text(
        -0.035,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_LETTER_SIZE,
        fontweight="bold",
        color="#202124",
        clip_on=False,
    )
    ax.text(
        title_x,
        y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=title_size,
        fontweight="bold",
        color="#202124",
        linespacing=0.9,
        clip_on=False,
    )


def _standard_group_heading(fig, axes, letter: str, title: str):
    """Place one heading over a multi-axis panel."""
    for ax in axes:
        _clear_panel_heading(ax)
    boxes = [ax.get_position() for ax in axes]
    x0 = min(b.x0 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    fig.text(
        x0 - 0.010,
        y1 + 0.018,
        letter,
        ha="left",
        va="bottom",
        fontsize=PANEL_LETTER_SIZE,
        fontweight="bold",
        color="#202124",
    )
    fig.text(
        x0 + 0.030,
        y1 + 0.018,
        title,
        ha="left",
        va="bottom",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        color="#202124",
        linespacing=0.9,
    )


def _box(ax, xy, wh, text, *, fc="#f7f7f7", ec="#444444", color="#202124"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=0.9,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=8.2, color=color, fontweight="bold", linespacing=1.0)
    return patch


def _arrow(ax, start, end, *, color="#444444", lw=1.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    ))


def _plot_twin_mechanism_schematic(ax):
    """Small mechanism schematic for the merged figure."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    _box(ax, (0.06, 0.35), (0.18, 0.30),
         "Gaze-contingent\nretinal movie",
         fc="#f2f6fb", ec=geom.MODEL, color=geom.MODEL)
    _box(ax, (0.35, 0.35), (0.20, 0.30),
         "Digital twin\nResNet + GRU core",
         fc="#f7f7f7", ec="#333333")
    _box(ax, (0.68, 0.35), (0.16, 0.30),
         "V1 readouts\n(rate)",
         fc="#f4eef8", ec=geom.BRIDGE, color=geom.BRIDGE)

    _arrow(ax, (0.24, 0.50), (0.35, 0.50), color=geom.MODEL, lw=1.4)
    _arrow(ax, (0.55, 0.50), (0.68, 0.50), color="#333333", lw=1.4)
    _arrow(ax, (0.84, 0.50), (0.94, 0.50), color=geom.BRIDGE, lw=1.2)

    _box(ax, (0.34, 0.02), (0.22, 0.14),
         "optional eye-state\nbehavior pathway",
         fc="#fbf8fd", ec=geom.BRIDGE, color=geom.BRIDGE)
    _arrow(ax, (0.45, 0.16), (0.45, 0.35), color=geom.BRIDGE, lw=1.1)

    ax.text(0.94, 0.50, "recorded\nspikes", fontsize=7.1, color="#555555",
            ha="left", va="center")
    ax.text(
        0.06,
        0.82,
        "counterfactual engine for retinal stabilization, eye-state ablation, and translation probes",
        fontsize=7.3,
        color="#555555",
        ha="left",
        va="center",
    )

def _plot_existing_schematic(ax, schematic_image: Path | None):
    """Use the existing draft schematic when available."""
    if schematic_image is None or not schematic_image.exists():
        _plot_twin_mechanism_schematic(ax)
        return False

    img = plt.imread(schematic_image)
    rgb = img[..., :3]
    nonwhite = np.any(rgb < 0.985, axis=2)
    ys, xs = np.where(nonwhite)
    if len(xs) and len(ys):
        pad_y = max(5, int(0.03 * (ys.max() - ys.min() + 1)))
        pad_x = max(5, int(0.03 * (xs.max() - xs.min() + 1)))
        y0 = max(0, int(ys.min()) - pad_y)
        y1 = min(img.shape[0], int(ys.max()) + pad_y + 1)
        x0 = max(0, int(xs.min()) - pad_x)
        x1 = min(img.shape[1], int(xs.max()) + pad_x + 1)
        img = img[y0:y1, x0:x1]
    img = _clean_schematic_heading_and_lift_right_half(img)
    ax.imshow(img)
    ax.set_anchor("N")
    ax.set_axis_off()
    ax.text(
        0.01,
        0.98,
        "A",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#202124",
    )
    return True


def _clean_schematic_heading_and_lift_right_half(img: np.ndarray) -> np.ndarray:
    """Remove the embedded right-half heading and nudge that half upward."""
    out = np.ones_like(img)
    h, w = img.shape[:2]
    split_x = int(0.43 * w)
    clear_y = int(0.105 * h)
    shift_y = int(0.065 * h)
    work = img.copy()
    work[:clear_y, split_x:, :3] = 1.0
    if work.shape[-1] == 4:
        work[:clear_y, split_x:, 3] = 1.0
    out[:, :split_x] = work[:, :split_x]
    out[: h - shift_y, split_x:] = work[shift_y:, split_x:]
    return out


def _plot_validation_panel_pooled(ax, data, *, letter: str = "C"):
    """Validation panel: pooled ccnorm histogram across Allen and Logan."""
    ccnorm = data["ccnorm"]
    good = data["good"]
    vals = ccnorm[good & np.isfinite(ccnorm)]
    bins = np.linspace(0, 1, 21)
    med = float(np.nanmedian(vals))
    q25, q75 = np.nanpercentile(vals, [25, 75])
    ax.hist(vals, bins=bins, color=POOLED_FILL, edgecolor="white", alpha=0.55)
    ax.axvline(
        med,
        color=POOLED_COLOR,
        linewidth=2,
        ls=(0, (1, 1)),
    )
    ax.text(0.05, 0.92, f"median {med:.2f} [{q25:.2f}, {q75:.2f}]",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=5.8, color=POOLED_COLOR)
    ax.set_xlabel("Normalized correlation (ccnorm)")
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"{letter}  Held-out responses", loc="left",
                 fontweight="bold", fontsize=10, pad=4)
    print(
        f"Panel {letter} — pooled (N={len(vals)}): median ccnorm={med:.2f}, "
        f"IQR=[{q25:.2f}, {q75:.2f}]"
    )


def _plot_fem_modulation_pooled(ax, data, *, letter: str = "D"):
    """Pooled model-vs-empirical 1-alpha scatter."""
    comp = compute_model_one_minus_alpha(data)
    x = comp["emp"]
    y = comp["model"]
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    rho = spearmanr(x, y).correlation if len(x) >= 3 else np.nan
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.5)
    ax.scatter(x, y, s=5, alpha=0.38, color=POOLED_COLOR)
    ax.text(0.06, 0.94, f"ρ={rho:.2f}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=5.8, color=POOLED_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Empirical $1-\alpha$")
    ax.set_ylabel(r"Model $1-\alpha$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"{letter}  FEM modulation", loc="left",
                 fontweight="bold", fontsize=10, pad=4)
    print(f"Panel {letter} — pooled (N={len(x)}): Spearman ρ={rho:.3f}")


def _plot_ablation_r2_pooled(ax, data, *, cond: str = "zeroed", letter: str = "E"):
    """Pooled intact-vs-zeroed single-trial r2 scatter."""
    x = data["ve"]["intact"]
    y = data["ve"][cond]
    good = data["good"]
    m = good & np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[m], y[m], s=5, alpha=0.38, color=POOLED_COLOR)
    lims = [0, 0.35]
    ax.plot(lims, lims, "k--", lw=0.5, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Single-trial $r^2$ (intact)")
    ax.set_ylabel("Single-trial $r^2$ (zeroed)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    d = y[m] - x[m]
    med_delta = float(np.nanmedian(d))
    pct = 100.0 * med_delta / float(np.nanmedian(x[m]))
    ax.text(0.97, 0.08,
            f"{pct:+.0f}% of intact median\nmedian Δ$r^2$={med_delta:+.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="0.25")
    ax.set_title(f"{letter}  Eye-state zeroing", loc="left",
                 fontweight="bold", fontsize=10, pad=4)
    print(f"Panel {letter} — pooled (N={m.sum()}): median Δr²={med_delta:+.4f}")


def _replace_panel_label_text(ax, old_prefix: str, new_text: str, fontsize: float = 7.3):
    for txt in ax.texts:
        if txt.get_text().startswith(old_prefix):
            txt.set_text(new_text)
            txt.set_fontsize(fontsize)
            txt.set_linespacing(0.9)
            return


def _load_geometry_data(paths: geom.DataPaths):
    """Load the geometry datasets used by selected panels."""
    union_df = geom.load_union(paths)
    basis_df = geom.load_basis(paths)
    spec_df = geom.load_union_spectrum(paths, delta=0.25, n_show=32)
    null_spec_df = geom.load_null_spectrum_summary(paths, delta=0.25, n_show=32)
    tangent_data = geom.load_tangent_family(paths.tangent_maps) if paths.tangent_maps else None
    closure_summary_df, closure_metrics_df, closure_audit = geom.load_panel_f_closure(paths)
    return {
        "union_df": union_df,
        "basis_df": basis_df,
        "spec_df": spec_df,
        "null_spec_df": null_spec_df,
        "tangent_data": tangent_data,
        "closure_summary_df": closure_summary_df,
        "closure_metrics_df": closure_metrics_df,
        "closure_audit": closure_audit,
    }


def _load_ablation_cache():
    """Load ablation data without triggering a heavy inference run."""
    if not ABLATION_CACHE_PATH.exists():
        return None
    return load_ablation_data(recompute=False)


def _plot_ablation_placeholder(ax):
    ax.set_axis_off()
    ax.set_title("C  Extraretinal-pathway zeroing control", loc="left",
                 fontweight="bold", fontsize=10, pad=4)
    ax.text(0.5, 0.58, "ablation cache not found",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8.5, color=geom.ACCENT, fontweight="bold")
    ax.text(0.5, 0.42, f"Missing: {ABLATION_CACHE_PATH.name}",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7.0, color="0.45")


def _plot_translation_cosine_panel(ax, tangent_data: dict | None, *, letter: str = "D"):
    """Compact main-text version of the local translation direction result."""
    geom.panel_label(ax, letter, "No universal translation axis")
    if tangent_data is None:
        ax.text(0.5, 0.52, "tangent maps\nnot found",
                transform=ax.transAxes, ha="center", va="center",
                color=geom.ACCENT, fontsize=8)
        geom.clean_axes(ax, grid=True)
        return

    bx = np.asarray(tangent_data["bx"], dtype=np.float64)
    by = np.asarray(tangent_data["by"], dtype=np.float64)
    cos_xx = geom._sampled_pairwise_cosines(bx, seed=13)
    cos_yy = geom._sampled_pairwise_cosines(by, seed=17)
    bins = np.linspace(-1.0, 1.0, 29)

    if cos_xx.size:
        ax.hist(
            cos_xx,
            bins=bins,
            density=True,
            histtype="step",
            color=geom.MODEL,
            lw=1.9,
            label=r"$b_x(I), b_x(J)$",
        )
        ax.axvline(float(np.median(cos_xx)), color=geom.MODEL, lw=1.0, alpha=0.85)
    if cos_yy.size:
        ax.hist(
            cos_yy,
            bins=bins,
            density=True,
            histtype="step",
            color=geom.BRIDGE,
            lw=1.9,
            label=r"$b_y(I), b_y(J)$",
        )
        ax.axvline(float(np.median(cos_yy)), color=geom.BRIDGE, lw=1.0, alpha=0.85)

    ax.axvline(0, color="0.45", lw=0.8, ls=":")
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Cross-image cosine")
    ax.set_ylabel("Density")
    ax.text(
        0.05,
        0.88,
        "same-axis\nacross images",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        color="0.35",
    )
    ax.text(0.96, 0.91, r"$b_x$", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.0, color=geom.MODEL, fontweight="bold")
    ax.text(0.96, 0.82, r"$b_y$", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.0, color=geom.BRIDGE, fontweight="bold")
    geom.clean_axes(ax, grid=True)


def _plot_covariance_closure_subset(
    ax,
    closure_summary: pd.DataFrame | None,
    closure_metrics: pd.DataFrame | None,
    *,
    letter: str = "I",
):
    """Two-control horizontal version of geometry Figure 4E."""
    geom.panel_label(ax, letter, "Translation-predicted\nFEM covariance")

    if closure_summary is None or closure_metrics is None or len(closure_summary) == 0:
        ax.text(0.5, 0.52, "finite-difference closure\nnot found",
                transform=ax.transAxes, ha="center", va="center",
                color=geom.ACCENT, fontsize=8)
        geom.clean_axes(ax, grid=True)
        return

    source = "fd_sample_eye_trace_cov"
    target = "psd"
    k = 2
    controls = ["none", "global_rate+target_pc1"]
    labels = ["none", "global + PC1\nremoved"]
    ypos = np.array([1.0, 0.0])

    s = closure_summary[
        (closure_summary["target_variant"].astype(str) == target)
        & (closure_summary["basis_source"].astype(str) == source)
        & (closure_summary["k"].astype(int) == k)
    ].copy()
    m = closure_metrics[
        (closure_metrics["target_variant"].astype(str) == target)
        & (closure_metrics["basis_source"].astype(str) == source)
        & (closure_metrics["k"].astype(int) == k)
        & (closure_metrics["row_status"].astype(str) == "ok")
    ].copy()

    rows = []
    finite_vals: list[float] = []
    rng = np.random.default_rng(4)
    for i, control in enumerate(controls):
        sr = s[s["projection_control"].astype(str) == control]
        mr = m[m["projection_control"].astype(str) == control]
        if len(sr) == 0 or len(mr) == 0:
            continue

        mean = float(sr["effect_unit_mean"].iloc[0])
        lo = float(sr["effect_unit_boot_ci_low"].iloc[0])
        hi = float(sr["effect_unit_boot_ci_high"].iloc[0])
        vals = pd.to_numeric(
            mr["effect_minus_unit_shuffle_median"], errors="coerce"
        ).dropna().to_numpy(float)
        finite_vals.extend([mean, lo, hi, *vals])

        jitter = rng.uniform(-0.08, 0.08, size=vals.size)
        ax.scatter(vals, np.full(vals.size, ypos[i]) + jitter,
                   s=13, color="0.25", alpha=0.24, linewidths=0, zorder=2)
        ax.errorbar(mean, ypos[i],
                    xerr=[[max(mean - lo, 0.0)], [max(hi - mean, 0.0)]],
                    fmt="o", color=geom.BRIDGE, ecolor=geom.BRIDGE,
                    elinewidth=1.5, capsize=3.4, markersize=6.0,
                    markeredgecolor="white", markeredgewidth=0.7, zorder=4)
        rows.append((control, mean, lo, hi, vals.size,
                     int(sr["n_effect_positive"].iloc[0])))

    ax.axvline(0, color="0.48", lw=0.75, ls=":", zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    finite = np.asarray(finite_vals, dtype=float)
    finite = finite[np.isfinite(finite)]
    xmax = max(0.46, float(np.nanmax(finite)) + 0.065) if finite.size else 0.46
    ax.set_xlim(-0.04, xmax)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("Excess capture over shuffle")
    geom.clean_axes(ax, grid=True)

    controlled = next((r for r in rows if r[0] == "global_rate+target_pc1"), None)
    if controlled is not None:
        _, mean, lo, hi, n, n_pos = controlled
        ax.text(0.97, 0.16,
                f"global+PC1 removed:\n+{mean:.3f} [{lo:.3f}, {hi:.3f}]\n{n_pos}/{n} sessions",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0,
                color=geom.BRIDGE, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.23", fc="white",
                          ec=geom.BRIDGE_L, lw=0.7, alpha=0.96))


def _shift_axes_y(axes, dy: float):
    """Translate axes vertically in figure coordinates without touching row peers."""
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + dy, pos.width, pos.height])


def _shift_axis_x(ax, dx: float, dw: float = 0.0):
    """Translate one axis horizontally in figure coordinates."""
    pos = ax.get_position()
    ax.set_position([pos.x0 + dx, pos.y0, pos.width + dw, pos.height])


def _pad_axis_limits(ax, *, xfrac: float = 0.05, yfrac: float = 0.05):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = (x1 - x0) * xfrac
    dy = (y1 - y0) * yfrac
    ax.set_xlim(x0 - dx, x1 + dx)
    ax.set_ylim(y0 - dy, y1 + dy)


def _write_sidecars(out_dir: Path, paths: geom.DataPaths, manifest: dict):
    caption = """Figure 3. A retinal-input digital twin reveals a compact reafferent geometry underlying FEM-linked V1 variability.

(A) Minimal schematic of the gaze-contingent digital twin: measured FEMs render a retinal movie, the movie drives an image-computable twin, and a separate eye-state pathway can be zeroed. (B) The twin recapitulates each cell's empirical FEM modulation, measured as \\(1-\\alpha\\), pooling Allen and Logan cells and linking the model to the covariance decomposition in Figure 2. (C) Single-trial prediction is nearly unchanged when the separate extraretinal eye-state pathway is zeroed, pooled across Allen and Logan, supporting a retinal-input route for FEM-linked variability. (D) Same-axis local translation vectors have low cross-image cosine similarity, showing that FEM translations do not correspond to a universal signed population axis. (E) Pooling local retinal-translation tangents reveals a compact translation subspace. (F) An image-disjoint basis captures held-out translation tangent variance above unit-shuffled controls. (G) Finite-difference fitted-twin translation covariances capture the recorded FEM covariance component above a unit-shuffled source-basis null, shown for no projection control and after removing both global-rate and target-PC1 components.
"""
    (out_dir / "fig3_combined_mechanism_caption.md").write_text(caption, encoding="utf-8")
    (out_dir / "fig3_combined_mechanism_caption.txt").write_text(caption, encoding="utf-8")

    readme = f"""# Combined Figure 3

Generated by `declan/fig3/generate_figure3_combined.py`.

This is the simplified merged digital-twin/mechanism figure. It keeps the old
Figure 3 and covTFTS Figure 4 scripts as source and supplemental material, but
promotes only the panels needed for the main-text mechanism chain:

retinal-input twin schematic -> FEM modulation capture -> extraretinal route
control -> no universal translation axis -> compact tangent geometry ->
image-disjoint generalization -> translation-predicted recorded FEM covariance
closure.

## Outputs
- `fig3_combined_mechanism.png`
- `fig3_combined_mechanism.pdf`
- `fig3_combined_mechanism.svg`
- `fig3_combined_mechanism_caption.md`
- `fig3_combined_mechanism_manifest.json`

## Geometry Source
- Tangent maps: `{paths.tangent_maps}`
- Union spectrum: `{paths.spec_file}`
- Union summary: `{paths.union_file}`
- Basis file: `{paths.basis_file}`
- Ablation cache: `{ABLATION_CACHE_PATH}`
- Finite-difference closure summary: `{paths.panel_f_closure_summary_file}`
- Finite-difference closure metrics: `{paths.panel_f_closure_metrics_file}`
- Finite-difference closure audit: `{paths.panel_f_closure_audit_file}`

## Warnings
{chr(10).join("- " + w for w in paths.warnings) or "- none"}
"""
    (out_dir / "fig3_combined_mechanism_README.md").write_text(readme, encoding="utf-8")

    with open(out_dir / "fig3_combined_mechanism_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)


def compose(
    *,
    recompute: bool = False,
    tfts_root: Path | None = None,
    out_dir: Path = FIG_DIR,
    schematic_image: Path | None = None,
    dpi: int = 300,
):
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_fig3_data(recompute=recompute)
    abl = _load_ablation_cache()

    if tfts_root is None:
        tfts_root = VISIONCORE_ROOT / "outputs" / "twin_feature_tangent_structure_prod_v2"
    paths = geom.resolve_paths(tfts_root)
    g = _load_geometry_data(paths)

    if schematic_image is None:
        schematic_image = out_dir / "panel_a_schematic_draft.png"

    fig = plt.figure(figsize=(8.4, 9.0), constrained_layout=False)
    gs = GridSpec(
        3, 4,
        figure=fig,
        left=0.090,
        right=0.985,
        bottom=0.070,
        top=0.925,
        height_ratios=[0.68, 1.0, 0.92],
        wspace=0.62,
        hspace=0.58,
    )

    # Row 1. Model object.
    ax_a = fig.add_subplot(gs[0, :])
    _plot_twin_mechanism_schematic(ax_a)
    used_existing_schematic = False
    _standard_panel_heading(ax_a, "A", "Retinal-input twin")

    # Row 2. Model relevance and retinal route.
    ax_b = fig.add_subplot(gs[1, 0:2])
    _plot_fem_modulation_pooled(ax_b, data, letter="B")
    _standard_panel_heading(ax_b, "B", "FEM modulation")

    ax_c = fig.add_subplot(gs[1, 2:4])
    if abl is not None:
        _plot_ablation_r2_pooled(ax_c, abl, cond="zeroed", letter="C")
    else:
        _plot_ablation_placeholder(ax_c)
    _standard_panel_heading(ax_c, "C", "Eye-state zeroing")

    # Row 3. Translation geometry and recorded covariance closure.
    ax_d = fig.add_subplot(gs[2, 0])
    _plot_translation_cosine_panel(ax_d, g["tangent_data"], letter="D")
    _standard_panel_heading(
        ax_d, "D", "No universal\ntranslation axis",
        y=1.060, title_x=0.11, title_size=8.2,
    )

    ax_e = fig.add_subplot(gs[2, 1])
    geom.plot_panel_c(
        ax_e,
        g["spec_df"],
        g["union_df"],
        null_spec_df=g["null_spec_df"],
        letter="E",
    )
    _replace_panel_label_text(ax_e, "Pooled translation", "Compact\nsubspace")
    leg = ax_e.get_legend()
    if leg is not None:
        leg.remove()
    for artist in list(ax_e.texts):
        if isinstance(artist, Annotation) and "PR =" in artist.get_text():
            artist.remove()
    ax_e.text(
        0.42,
        0.22,
        "PR 9.0\nnull ~31",
        transform=ax_e.transAxes,
        ha="left",
        va="center",
        fontsize=6.1,
        color=geom.MODEL,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.85", lw=0.55, alpha=0.94),
    )
    ax_e.set_xlabel("Rank")
    ax_e.set_ylabel("Cum. variance")
    ax_e.tick_params(labelsize=7)
    _standard_panel_heading(
        ax_e, "E", "Compact\nsubspace",
        y=1.060, title_x=0.11, title_size=8.2,
    )

    ax_f = fig.add_subplot(gs[2, 2])
    geom.plot_panel_e(ax_f, g["basis_df"], paths.basis_source_label, letter="F")
    _replace_panel_label_text(ax_f, "Compactness", "Image-disjoint\ngeneralization")
    leg = ax_f.get_legend()
    if leg is not None:
        leg.remove()
    for artist in list(ax_f.texts):
        if isinstance(artist, Annotation) and "k=" in artist.get_text():
            artist.remove()
    ax_f.text(
        0.50,
        0.20,
        "k=10: 0.50\nnull 0.11",
        transform=ax_f.transAxes,
        ha="left",
        va="center",
        fontsize=6.3,
        color="#202124",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.85", lw=0.55, alpha=0.94),
    )
    ax_f.set_ylabel("Held-out variance")
    ax_f.set_xlabel("Basis k")
    ax_f.tick_params(labelsize=7)
    _standard_panel_heading(
        ax_f, "F", "Image-disjoint\ngeneralization",
        y=1.060, title_x=0.11, title_size=8.2,
    )

    ax_g = fig.add_subplot(gs[2, 3])
    _plot_covariance_closure_subset(
        ax_g,
        g["closure_summary_df"],
        g["closure_metrics_df"],
        letter="G",
    )
    leg = ax_g.get_legend()
    if leg is not None:
        leg.remove()
    _standard_panel_heading(
        ax_g, "G", "Predicted FEM\ncovariance",
        y=1.060, title_x=0.11, title_size=8.2,
    )

    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"fig3_combined_mechanism.{ext}",
                    dpi=dpi, bbox_inches="tight")

    manifest = {
        "figure": "fig3_combined_mechanism",
        "digital_twin_cache": str(VISIONCORE_ROOT / "outputs" / "cache" / "fig3_digitaltwin.pkl"),
        "ablation_cache": str(ABLATION_CACHE_PATH),
        "ablation_cache_present": abl is not None,
        "geometry_root": str(tfts_root),
        "schematic_image": str(schematic_image),
        "used_existing_schematic": used_existing_schematic,
        "geometry_warnings": paths.warnings,
        "source_files": {
            "combined_script": str(Path(__file__).resolve()),
            "digital_twin_compositor": str(VISIONCORE_ROOT / "declan" / "fig3" / "generate_figure3.py"),
            "geometry_compositor": str(VISIONCORE_ROOT / "declan" / "fig4_cov_TFTS" / "generate_covTFTS_figure.py"),
            "tangent_maps": str(paths.tangent_maps),
            "union_file": str(paths.union_file),
            "spec_file": str(paths.spec_file),
            "basis_file": str(paths.basis_file),
            "panel_f_closure_summary_file": str(paths.panel_f_closure_summary_file),
            "panel_f_closure_metrics_file": str(paths.panel_f_closure_metrics_file),
            "panel_f_closure_audit_file": str(paths.panel_f_closure_audit_file),
        },
        "panel_mapping": {
            "A": "minimal retinal-input digital twin schematic",
            "B": "current Figure 3D empirical vs model 1-alpha",
            "C": "current Figure 3I zeroed extraretinal input vs intact single-trial r2",
            "D": "same-axis cross-image cosine histograms from current geometry Figure 4A tangent data",
            "E": "current geometry Figure 4B compact translation-tangent subspace",
            "F": "current geometry Figure 4C image-disjoint generalization",
            "G": "current geometry Figure 4E covariance closure subset (none and global_rate+target_pc1)",
            "moved_to_extended": [
                "current Figure 3B example reliable-neuron PSTH",
                "current Figure 3C held-out response validation histogram",
                "full architecture schematic raster",
            ],
        },
    }
    _write_sidecars(out_dir, paths, manifest)
    return fig, manifest


def parse_args():
    p = argparse.ArgumentParser(description="Generate merged digital-twin/mechanism Figure 3.")
    p.add_argument("--recompute", action="store_true",
                   help="Force digital-twin recomputation instead of cached results.")
    p.add_argument("--tfts-root", type=Path,
                   default=VISIONCORE_ROOT / "outputs" / "twin_feature_tangent_structure_prod_v2",
                   help="Root containing twin feature tangent structure outputs.")
    p.add_argument("--out-dir", type=Path, default=FIG_DIR,
                   help="Directory for figure outputs.")
    p.add_argument("--schematic-image", type=Path, default=None,
                   help="Existing draft schematic image to use for Panel A.")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    fig, _manifest = compose(
        recompute=args.recompute,
        tfts_root=args.tfts_root,
        out_dir=args.out_dir,
        schematic_image=args.schematic_image,
        dpi=args.dpi,
    )
    plt.close(fig)
    print(f"Saved combined Figure 3 to: {args.out_dir}")


if __name__ == "__main__":
    main()
