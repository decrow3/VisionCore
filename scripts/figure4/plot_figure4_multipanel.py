#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FormatStrFormatter

from VisionCore.paths import VISIONCORE_ROOT


EXPECTED_EFFECTS = {
    (-0.35, 60): (0.050955414012738856, 0.02971072186836518, 0.0727176220806794),
    (-0.30, 60): (-0.03662420382165605, -0.059447983014861996, -0.014331210191082803),
    (-0.25, 60): (-0.1040339702760085, -0.1321656050955414, -0.07534501061571128),
    (-0.35, 1): (-0.45116772823779194, -0.4755971337579618, -0.42461518046709135),
}


@dataclass
class CovarianceBranch:
    available: bool
    missing_items: list[str]
    noise_metrics: pd.DataFrame | None
    geometry_metrics: pd.DataFrame | None
    master_summary: pd.DataFrame | None
    quantity_map: dict[str, Any]
    source_paths: list[str]


def configure_style(style: str = "manuscript") -> None:
    """Ryan-style plotting defaults: Arial stack, embedded fonts, compact labels."""
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    if style == "talk":
        mpl.rcParams["font.size"] = 9
        mpl.rcParams["axes.labelsize"] = 9
        mpl.rcParams["axes.titlesize"] = 10
        mpl.rcParams["xtick.labelsize"] = 8
        mpl.rcParams["ytick.labelsize"] = 8
    else:
        mpl.rcParams["font.size"] = 8
        mpl.rcParams["axes.labelsize"] = 8
        mpl.rcParams["axes.titlesize"] = 8
        mpl.rcParams["xtick.labelsize"] = 7
        mpl.rcParams["ytick.labelsize"] = 7


def _ensure_dirs(out_dir: Path) -> dict[str, Path]:
    paths = {
        "root": out_dir,
        "panels": out_dir / "panels",
        "source_tables": out_dir / "source_tables",
        "qc": out_dir / "qc",
        "exports": out_dir / "exports",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def _panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.005, 1.02, letter, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=12)


def _condition_norm(cond: str) -> str:
    c = str(cond).strip().lower()
    if c in {"real", "real_fem", "real fem"}:
        return "real"
    if c in {"stabilized", "stable"}:
        return "stabilized"
    return c


def _save_figure(fig: plt.Figure, stem: Path, export_pdf: bool, export_svg: bool, export_png: bool, dpi: int) -> None:
    if export_pdf:
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", dpi=dpi)
    if export_svg:
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", dpi=dpi)
    if export_png:
        fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=dpi)


def _choose_example_trial(trials: pd.DataFrame) -> pd.Series:
    t = trials.copy()
    t["condition_norm"] = t["condition"].map(_condition_norm)
    t = t[(t["condition_norm"] == "real") & (t["window"].astype(int) == 60)]
    t = t[np.isclose(t["logmar"].astype(float), -0.35)]
    t = t[t["valid"].astype(int) == 1]
    if t.empty:
        raise RuntimeError("No valid real-FEM trial found for Panel A selection.")
    rms = t["eye_rms"].astype(float)
    med = float(np.nanmedian(rms.values))
    idx = (rms - med).abs().idxmin()
    return t.loc[idx]


def _choose_compact_trace_segment(xy: np.ndarray, max_len: int = 18) -> np.ndarray:
    n = xy.shape[0]
    if n <= max_len:
        return xy
    best_i = 0
    best_score = float("inf")
    for i in range(0, n - max_len + 1):
        seg = xy[i : i + max_len]
        span_x = float(np.ptp(seg[:, 0]))
        span_y = float(np.ptp(seg[:, 1]))
        score = span_x * span_y
        if score < best_score:
            best_score = score
            best_i = i
    return xy[best_i : best_i + max_len]


def _plot_panel_a(ax: plt.Axes, trial_manifest: pd.DataFrame, eye_npz: Path) -> dict[str, Any]:
    row = _choose_example_trial(trial_manifest)
    trace_id = int(row["trace_id"])
    d = np.load(eye_npz, allow_pickle=True)
    traces = np.asarray(d["traces"], dtype=np.float64)
    durations = np.asarray(d["durations"], dtype=np.int32)
    if trace_id >= traces.shape[0]:
        raise RuntimeError(f"trace_id {trace_id} out of bounds for eye trace bundle")
    dur = int(max(1, durations[trace_id]))
    xy = traces[trace_id, :dur, :]
    xy = _choose_compact_trace_segment(xy, max_len=18)

    # Stimulus proxy (schematic Tumbling-E) + trajectory overlay.
    # Keep stimulus icon separate so it does not compete with trajectory geometry.
    stim_ax = ax.inset_axes([0.04, 0.76, 0.17, 0.18])
    stim_ax.axis("off")
    stim_ax.text(0.50, 0.52, "E", fontsize=28, fontweight="bold", ha="center", va="center", color="0.30")

    ax.plot(xy[:, 0], xy[:, 1], color="#0072B2", lw=1.0, alpha=0.9, label="Real FEM")
    mu = np.nanmean(xy, axis=0)
    ax.scatter([mu[0]], [mu[1]], s=24, color="#4D4D4D", label="Stabilized")

    # Enforce equal x/y scaling with a square plotting box without collapsing layout.
    span_x = float(np.ptp(xy[:, 0]))
    span_y = float(np.ptp(xy[:, 1]))
    span = max(span_x, span_y, 0.0015)
    half = 0.62 * span
    ax.set_xlim(float(mu[0] - half), float(mu[0] + half))
    ax.set_ylim(float(mu[1] - half), float(mu[1] + half))

    ax.set_xlabel("Eye x (deg)")
    ax.set_ylabel("Eye y (deg)")
    ax.set_box_aspect(1.0)
    ax.legend(loc="upper right", fontsize=5.8, frameon=False)
    ax.text(0.02, 0.10, "Real FEM: retinal image moves", transform=ax.transAxes, fontsize=6, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.0})
    ax.text(0.02, 0.05, "Stabilized: motion removed at trial mean", transform=ax.transAxes, fontsize=6, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.0})
    return {
        "trace_id": trace_id,
        "duration": dur,
        "logmar": float(row["logmar"]),
        "window": int(row["window"]),
    }


def _plot_panel_b(ax: plt.Axes, decoder_metrics: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    ax.axis("off")
    top = ax.inset_axes([0.0, 0.46, 1.0, 0.54])
    bot = ax.inset_axes([0.0, 0.03, 1.0, 0.36])

    dm = decoder_metrics.copy()
    dm["condition_norm"] = dm["condition"].map(_condition_norm)
    dm = dm[dm["window"].astype(int) == 60]
    colors = {"real": "#0072B2", "stabilized": "#4D4D4D"}
    for cond in ["real", "stabilized"]:
        s = dm[dm["condition_norm"] == cond].sort_values("logmar")
        x = s["logmar"].astype(float).values
        y = s["heldout_accuracy"].astype(float).values
        lo = s["accuracy_ci_low"].astype(float).values
        hi = s["accuracy_ci_high"].astype(float).values
        top.plot(x, y, marker="o", lw=1.0, ms=3.4, color=colors[cond], label=cond)
        top.fill_between(x, lo, hi, color=colors[cond], alpha=0.15)
    top.set_ylabel("Accuracy")
    top.set_xticklabels([])
    top.grid(alpha=0.16, lw=0.5)
    top.legend(fontsize=5.9, frameon=False, loc="lower left")

    c = contrasts[contrasts["window"].astype(int) == 60].sort_values("logmar")
    x = c["logmar"].astype(float).values
    y = c["delta_accuracy"].astype(float).values
    lo = c["delta_ci_low"].astype(float).values
    hi = c["delta_ci_high"].astype(float).values
    bot.axhline(0.0, color="0.4", lw=0.9)
    bot.plot(x, y, color="black", marker="o", lw=1.0, ms=3.2)
    bot.fill_between(x, lo, hi, color="black", alpha=0.14)
    bot.set_ylabel("real - stab")
    bot.set_xlabel("LogMAR")
    bot.tick_params(labelsize=6)
    xt = np.array([-0.40, -0.35, -0.30, -0.25, -0.20], dtype=float)
    bot.set_xticks(xt)
    bot.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    bot.grid(alpha=0.15, lw=0.45)
    bot.text(0.02, 0.95, "scale-dependent sign-changing effect", transform=bot.transAxes, fontsize=5.9, ha="left", va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.6})


def _plot_panel_c(ax: plt.Axes, sweep: pd.DataFrame) -> None:
    s = sweep[np.isclose(sweep["logmar"].astype(float), -0.35)].sort_values("window")
    x = s["window"].astype(int).values
    y = s["delta_accuracy"].astype(float).values
    lo = s["delta_ci_low"].astype(float).values
    hi = s["delta_ci_high"].astype(float).values
    ax.axhline(0.0, color="0.2", lw=1.1)
    ax.plot(x, y, color="black", marker="o", lw=1.0, ms=3.5)
    ax.fill_between(x, lo, hi, color="black", alpha=0.15)
    ax.set_xlabel("Integration window (frames)")
    ax.set_ylabel("real - stabilized")
    ax.grid(alpha=0.16, lw=0.5)
    ax.text(0.02, 0.92, "single frame: worse", transform=ax.transAxes, fontsize=6)
    ax.text(0.02, 0.84, "60 frames: better", transform=ax.transAxes, fontsize=6)


def _status_color(status: str) -> str:
    s = str(status)
    if "validated" in s:
        return "#1b9e77"
    if "not_run" in s:
        return "#7f7f7f"
    if "unreliable" in s:
        return "#d95f02"
    return "#555555"


def _plot_panel_d(ax: plt.Axes, claims: pd.DataFrame) -> None:
    ax.axis("off")
    ax.text(0.02, 0.94, "First-order account remains partial", fontsize=8.4, fontweight="bold", transform=ax.transAxes)

    # Mechanism-adjudication flow (manuscript panel), with full observer-status details
    # left to QC/supplement outputs.
    top = FancyBboxPatch((0.08, 0.73), 0.84, 0.16, boxstyle="round,pad=0.012,rounding_size=0.02", linewidth=0.9, edgecolor="0.70", facecolor="#f8f8f8", transform=ax.transAxes)
    left = FancyBboxPatch((0.08, 0.46), 0.38, 0.18, boxstyle="round,pad=0.012,rounding_size=0.02", linewidth=0.9, edgecolor="#4d9221", facecolor="#f1f8e9", transform=ax.transAxes)
    right = FancyBboxPatch((0.54, 0.46), 0.38, 0.18, boxstyle="round,pad=0.012,rounding_size=0.02", linewidth=0.9, edgecolor="#d95f02", facecolor="#fff5eb", transform=ax.transAxes)
    bottom = FancyBboxPatch((0.08, 0.18), 0.84, 0.18, boxstyle="round,pad=0.012,rounding_size=0.02", linewidth=0.9, edgecolor="#4575b4", facecolor="#eef5fb", transform=ax.transAxes)
    for patch in [top, left, right, bottom]:
        ax.add_patch(patch)

    ax.text(0.50, 0.81, "Canonical sign-changing effect", ha="center", va="center", fontsize=7.0, fontweight="bold", transform=ax.transAxes)
    ax.text(0.50, 0.75, r"$\Delta$accuracy: + at -0.35, - at -0.30 / -0.25", ha="center", va="center", fontsize=6.3, transform=ax.transAxes)

    ax.text(0.27, 0.56, "Fine benefit captured", ha="center", va="center", fontsize=6.8, fontweight="bold", color="#2e7d32", transform=ax.transAxes)
    ax.text(0.27, 0.50, "by mean-rate observer", ha="center", va="center", fontsize=6.1, transform=ax.transAxes)

    ax.text(0.73, 0.56, "Coarser cost unresolved", ha="center", va="center", fontsize=6.8, fontweight="bold", color="#d95f02", transform=ax.transAxes)
    ax.text(0.73, 0.50, "for mean-rate observer", ha="center", va="center", fontsize=6.1, transform=ax.transAxes)

    ax.text(0.50, 0.29, "Current claim: first-order account established for\nfine-scale benefit, not the full canonical curve", ha="center", va="center", fontsize=6.4, transform=ax.transAxes)

    ax.annotate("", xy=(0.28, 0.64), xytext=(0.46, 0.73), arrowprops=dict(arrowstyle="->", lw=1.0, color="0.35"), xycoords=ax.transAxes)
    ax.annotate("", xy=(0.72, 0.64), xytext=(0.54, 0.73), arrowprops=dict(arrowstyle="->", lw=1.0, color="0.35"), xycoords=ax.transAxes)
    ax.annotate("", xy=(0.50, 0.36), xytext=(0.50, 0.46), arrowprops=dict(arrowstyle="->", lw=1.0, color="0.35"), xycoords=ax.transAxes)


def _plot_panel_e(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.text(0.50, 0.90, r"$R(t,e)=\mathbb{E}[Y\mid t,e]$", ha="center", fontsize=8.8, fontweight="bold", transform=ax.transAxes)

    left_box = FancyBboxPatch((0.05, 0.30), 0.40, 0.34, boxstyle="round,pad=0.015,rounding_size=0.02", linewidth=0.9, edgecolor="#6baed6", facecolor="#edf6fd", transform=ax.transAxes)
    right_box = FancyBboxPatch((0.55, 0.30), 0.40, 0.34, boxstyle="round,pad=0.015,rounding_size=0.02", linewidth=0.9, edgecolor="#969696", facecolor="#f5f5f5", transform=ax.transAxes)
    ax.add_patch(left_box)
    ax.add_patch(right_box)

    ax.annotate("", xy=(0.24, 0.66), xytext=(0.46, 0.86), arrowprops=dict(arrowstyle="->", lw=1.0, color="0.35"), xycoords=ax.transAxes)
    ax.annotate("", xy=(0.76, 0.66), xytext=(0.54, 0.86), arrowprops=dict(arrowstyle="->", lw=1.0, color="0.35"), xycoords=ax.transAxes)

    ax.text(0.25, 0.59, "First moment", fontsize=6.6, fontweight="bold", transform=ax.transAxes, ha="center")
    ax.text(0.25, 0.52, "mean over eye trajectory", fontsize=5.6, transform=ax.transAxes, ha="center")
    ax.text(0.25, 0.46, "class means shift", fontsize=5.6, transform=ax.transAxes, ha="center")
    ax.text(0.25, 0.40, "supports discrimination", fontsize=5.6, transform=ax.transAxes, ha="center")

    ax.text(0.75, 0.59, "Second moment", fontsize=6.6, fontweight="bold", transform=ax.transAxes, ha="center")
    ax.text(0.75, 0.52, "covariance over eye states", fontsize=5.6, transform=ax.transAxes, ha="center")
    ax.text(0.75, 0.46, "reafferent shared variability", fontsize=5.6, transform=ax.transAxes, ha="center")
    ax.text(0.50, 0.24, r"$\Sigma_{FEM}=\mathbb{E}_{t}[\mathrm{Cov}_{e}(R(t,e)\mid t)]$", fontsize=5.5, transform=ax.transAxes, ha="center")


def _load_covariance_branch(cov_dir: Path) -> CovarianceBranch:
    noise_path = cov_dir / "noise_correlations" / "noise_correlation_session_metrics.csv"
    geom_path = cov_dir / "covariance_geometry" / "covariance_geometry_session_metrics.csv"
    master_path = cov_dir / "summaries" / "phase1_master_summary.csv"

    noise = pd.read_csv(noise_path) if noise_path.exists() else None
    geom = pd.read_csv(geom_path) if geom_path.exists() else None
    master = pd.read_csv(master_path) if master_path.exists() else None
    source_paths = [str(p) for p in [noise_path, geom_path, master_path] if p.exists()]

    required = {
        "noise_corr_raw_median_by_session": noise is not None and "median_corr_raw" in noise.columns,
        "noise_corr_corrected_median_by_session": noise is not None and "median_corr_eye_corrected" in noise.columns,
        "noise_corr_delta_by_session": noise is not None and "median_delta_raw_to_corrected" in noise.columns,
        "fem_covariance_participation_ratio": geom is not None and ("mc_participation_ratio" in geom.columns or "b_emp_participation_ratio" in geom.columns),
        "fem_covariance_signal_alignment": geom is not None and "model_alignment" in geom.columns,
        "eye_position_decoding_metric": master is not None and "aggregation_reliability_at_max_N" in master.columns,
    }
    missing = [k for k, ok in required.items() if not ok]

    quantity_map: dict[str, Any] = {}
    if noise is not None:
        quantity_map["noise_corr_raw_median_by_session"] = noise[["session", "median_corr_raw"]].to_dict(orient="records")
        quantity_map["noise_corr_corrected_median_by_session"] = noise[["session", "median_corr_eye_corrected"]].to_dict(orient="records")
        quantity_map["noise_corr_delta_by_session"] = noise[["session", "median_delta_raw_to_corrected"]].to_dict(orient="records")
    if geom is not None:
        if "mc_participation_ratio" in geom.columns:
            quantity_map["fem_covariance_participation_ratio"] = float(np.nanmedian(geom["mc_participation_ratio"].astype(float).values))
        elif "b_emp_participation_ratio" in geom.columns:
            quantity_map["fem_covariance_participation_ratio"] = float(np.nanmedian(geom["b_emp_participation_ratio"].astype(float).values))
        if "model_alignment" in geom.columns:
            quantity_map["fem_covariance_signal_alignment"] = float(np.nanmedian(geom["model_alignment"].astype(float).values))
    if master is not None and "aggregation_reliability_at_max_N" in master.columns:
        quantity_map["eye_position_decoding_metric"] = float(np.nanmedian(master["aggregation_reliability_at_max_N"].astype(float).values))

    return CovarianceBranch(
        available=len(missing) == 0,
        missing_items=missing,
        noise_metrics=noise,
        geometry_metrics=geom,
        master_summary=master,
        quantity_map=quantity_map,
        source_paths=source_paths,
    )


def _plot_panel_f(ax: plt.Axes, cov: CovarianceBranch) -> None:
    if not cov.available or cov.noise_metrics is None:
        ax.axis("off")
        ax.text(0.5, 0.55, "Covariance-branch values pending", ha="center", va="center", fontsize=8, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.40, "Panel F requires dedicated covariance outputs.", ha="center", va="center", fontsize=6, transform=ax.transAxes)
        return
    n = cov.noise_metrics
    x0, x1 = 0.0, 1.0
    for _, r in n.iterrows():
        y0 = float(r["median_corr_raw"])
        y1 = float(r["median_corr_eye_corrected"])
        ax.plot([x0, x1], [y0, y1], marker="o", lw=1.0, ms=3, color="0.35", alpha=0.8)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["raw", "eye-corrected"])
    ax.set_ylabel("Median noise corr")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.grid(alpha=0.15, lw=0.5)
    if cov.geometry_metrics is not None:
        g = cov.geometry_metrics
        pr_col = "mc_participation_ratio" if "mc_participation_ratio" in g.columns else "b_emp_participation_ratio"
        pr = float(np.nanmedian(g[pr_col].astype(float).values))
        al = float(np.nanmedian(g["model_alignment"].astype(float).values))
        txt = f"median across sessions\nparticipation ratio: {pr:.2f}\nsignal alignment: {al:.2f}"
        ax.text(0.55, 0.95, txt, transform=ax.transAxes, va="top", ha="left", fontsize=5.8, bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.9, "pad": 1.5})


def _plot_panel_g(ax: plt.Axes, occupancy: pd.DataFrame) -> dict[str, Any]:
    required_cols = {"center_mimicry", "weighted_mimicry_mean", "logmar"}
    if not required_cols.issubset(set(occupancy.columns)):
        raise RuntimeError("validated mimicry occupancy table missing required columns")
    x = occupancy["center_mimicry"].astype(float).values
    y = occupancy["weighted_mimicry_mean"].astype(float).values
    lm = occupancy["logmar"].astype(float).values
    sc = ax.scatter(x, y, c=lm, cmap="viridis", s=14, alpha=0.8)
    lo = min(np.nanmin(x), np.nanmin(y))
    hi = max(np.nanmax(x), np.nanmax(y))
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    lo_lim = lo - pad
    hi_lim = hi + pad
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8, ls="--")
    ax.set_xlim(lo_lim, hi_lim)
    ax.set_ylim(lo_lim, hi_lim)
    ax.set_box_aspect(1.0)
    ax.set_anchor("W")
    ax.set_xlabel("center mimicry")
    ax.set_ylabel("occupancy-weighted mimicry")
    ax.text(0.62, 1.02, "Pose/identity mimicry is phase- and scale-dependent", transform=ax.transAxes, ha="center", va="bottom", fontsize=13)
    ax.text(0.02, 0.95, "dashed: y=x (weighted equals center)", transform=ax.transAxes, fontsize=5.6, ha="left", va="top")
    plt.colorbar(sc, ax=ax, label="LogMAR")
    ax.grid(alpha=0.12, lw=0.45)
    return {
        "mimicry_fraction_range": [float(np.nanmin(y)), float(np.nanmax(y))],
        "n_points": int(len(y)),
    }


def _check_consistency(contrast: pd.DataFrame, sweep: pd.DataFrame) -> tuple[bool, float]:
    a = contrast[["logmar", "window", "delta_accuracy"]].copy()
    b = sweep[["logmar", "window", "delta_accuracy"]].copy()
    m = a.merge(b, on=["logmar", "window"], suffixes=("_a", "_b"))
    if m.empty:
        return False, float("nan")
    max_abs = float(np.nanmax(np.abs(m["delta_accuracy_a"].astype(float) - m["delta_accuracy_b"].astype(float))))
    return max_abs < 1e-12, max_abs


def _check_effect_values(contrast: pd.DataFrame, sweep: pd.DataFrame) -> tuple[bool, list[str]]:
    errs: list[str] = []
    for (lm, w), (ed, elo, ehi) in EXPECTED_EFFECTS.items():
        src = contrast if w == 60 else sweep
        row = src[np.isclose(src["logmar"].astype(float), lm) & (src["window"].astype(int) == int(w))]
        if row.empty:
            errs.append(f"missing ({lm},{w})")
            continue
        r = row.iloc[0]
        got = (float(r["delta_accuracy"]), float(r["delta_ci_low"]), float(r["delta_ci_high"]))
        exp = (float(ed), float(elo), float(ehi))
        if any(abs(g - e) > 5e-4 for g, e in zip(got, exp)):
            errs.append(f"mismatch ({lm},{w}) got={got} expected={exp}")
    return len(errs) == 0, errs


def _write_qc_reports(
    qc_dir: Path,
    *,
    canonical_ok: bool,
    canonical_max_abs_diff: float,
    effect_ok: bool,
    effect_errs: list[str],
    observer_claims: pd.DataFrame,
    cov: CovarianceBranch,
    panel_g_qc: dict[str, Any],
    include_panel_g: bool,
    final_status: str,
    single_frame_real_acc: float,
    single_frame_qc_status: str,
) -> None:
    lines = [
        "# Figure 4 QC report",
        "",
        "## Data availability",
        "- canonical_decoder_metrics.csv loaded: yes",
        "- canonical_real_minus_stabilized.csv loaded: yes",
        "- integration_window_sweep.csv loaded: yes",
        "- observer_claim_validation.csv loaded: yes",
        f"- covariance branch loaded: {'yes' if cov.available else 'no'}",
        "- validated mimicry tables loaded: yes",
        "",
        "## Canonical consistency",
        f"- contrast_vs_sweep_exact_match: {'yes' if canonical_ok else 'no'}",
        f"- max_abs_delta_difference: {canonical_max_abs_diff}",
        "",
        "## Effect-size checks",
        f"- status: {'ok' if effect_ok else 'effect_size_mismatch'}",
        f"- single_frame_real_accuracy: {single_frame_real_acc:.4f}",
        f"- single_frame_above_4way_chance: {single_frame_qc_status}",
    ]
    if effect_errs:
        lines.extend([f"- {e}" for e in effect_errs])
    lines.extend([
        "",
        "## Claim validation",
    ])
    for _, r in observer_claims.iterrows():
        lines.append(f"- {r['observer_name']}: {r['status']}")
    lines.extend([
        "",
        "## Mimicry validation",
        f"- include_panel_g: {'yes' if include_panel_g else 'no'}",
        f"- panel_g_points: {panel_g_qc.get('n_points', 0)}",
        f"- mimicry_range: {panel_g_qc.get('mimicry_fraction_range', [])}",
        "",
        "## Covariance branch",
        f"- missing_items: {', '.join(cov.missing_items) if cov.missing_items else 'none'}",
        f"- source_files: {', '.join(cov.source_paths) if cov.source_paths else 'none'}",
        "- participation/signal values in Panel F are medians across session-level covariance tables",
    ])
    if not cov.available:
        lines.append("- Panel F requires dedicated covariance branch outputs before manuscript submission.")
    lines.extend([
        "",
        f"## Final status\n- {final_status}",
    ])
    (qc_dir / "figure4_qc_report.md").write_text("\n".join(lines) + "\n")

    g_lines = [
        "# Panel G mimicry QC",
        "",
        f"- n_points: {panel_g_qc.get('n_points', 0)}",
        f"- mimicry_fraction_range: {panel_g_qc.get('mimicry_fraction_range', [])}",
        "- expected_mimicry_bounds: [0,1]",
    ]
    (qc_dir / "panelG_mimicry_qc.md").write_text("\n".join(g_lines) + "\n")


def _write_methods_and_caption(qc_dir: Path, mean_status: str) -> None:
    methods = [
        "# Figure 4 methods snippet",
        "",
        "We rendered four Tumbling-E orientations across a LogMAR ladder and passed each stimulus through the frozen V1 digital twin under measured real-FEM trajectories or a stabilized counterfactual that held gaze at the trial mean. Four-way orientation accuracy was estimated from held-out trials using the canonical time-averaged mean-rate feature representation. Real-minus-stabilized contrasts were bootstrapped over trials or splits using the same cross-validation policy as the decoder.",
        "",
        "We define the eye-conditioned mean population response as R(t,e)=E[Y|t,e]. The first-order discrimination analysis depends on the within-trial average of R(t,e) over the sampled eye trajectory. The FEM-linked covariance is the second moment of this same modulation, Sigma_FEM=E_t[Cov_e(R(t,e)|t)], and corresponds to the component of shared variability removed by conditioning on eye position.",
        "",
        "Translation mimicry was computed as a first-order alignment between orientation identity-difference directions and local retinal-translation directions in the validated mono Model A population. This analysis was used to characterize when reafferent covariance should be recoverable as pose variation versus confounding for pose-ignorant readouts. It was not used to estimate Sigma_FEM via a local J Sigma_eye J^T approximation.",
    ]
    (qc_dir / "figure4_methods_snippet.md").write_text("\n".join(methods) + "\n")

    d_sentence = (
        "(D) The fine-scale benefit was captured by the time-averaged population mean. The coarser-scale cost was not fully explained by the mean-only observer and is not assigned to the first-order mechanism here."
        if mean_status == "mean_only_reproduces_benefit_only"
        else "(D) The first-order observer summary is shown from canonical observer-claim validation."
    )
    caption = [
        "# Figure 4 caption draft",
        "",
        "Figure 4. Fixational eye movements produce a scale-dependent discrimination effect whose second moment is reafferent V1 shared variability.",
        "",
        "(A) Counterfactual digital-twin manipulation. Under real FEMs, fine optotypes sweep across retinal positions during fixation. Under stabilization, retinal motion is removed while preserving trial-mean gaze position.",
        "(B) Four-way Tumbling-E orientation accuracy across LogMAR shows a scale-dependent sign-changing effect.",
        "(C) At LogMAR -0.35, real FEMs are worse at a single frame but better after 60-frame integration.",
        d_sentence,
        "(E) First-moment / second-moment bridge for R(t,e): mean shifts support discrimination while Sigma_FEM captures reafferent shared variability.",
        "(F) Covariance-branch panel summarizes reduction in positive shared variability after eye conditioning (or remains placeholder if branch values are missing).",
        "(G) Translation mimicry summarizes recoverability versus confusability in model geometry.",
    ]
    (qc_dir / "figure4_caption_draft.md").write_text("\n".join(caption) + "\n")


def _write_missing_covariance_note(qc_dir: Path, missing: list[str]) -> None:
    lines = [
        "# Missing covariance branch values",
        "",
        "Covariance-branch values required for Panel F were not fully available.",
        "",
        "Missing quantities:",
    ]
    lines.extend([f"- {m}" for m in missing])
    (qc_dir / "missing_covariance_branch_values.md").write_text("\n".join(lines) + "\n")


def _save_panel_individuals(
    panels_dir: Path,
    panel_data: dict[str, Any],
    dpi: int,
) -> None:
    for letter, drawer in panel_data.items():
        fig, ax = plt.subplots(figsize=(3.5, 2.2), constrained_layout=True)
        drawer(ax)
        _panel_letter(ax, letter)
        stem = panels_dir / f"Fig4{letter}_{panel_data[letter].__name__.replace('_draw_', '')}"
        fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Figure 4 multipanel outputs from reconciliation artifacts.")
    p.add_argument("--reconciliation-dir", type=Path, default=VISIONCORE_ROOT / "outputs" / "figure4_reconciliation")
    p.add_argument("--covariance-dir", type=Path, default=VISIONCORE_ROOT / "outputs" / "phase1_fem_covariance")
    p.add_argument("--out-dir", type=Path, default=VISIONCORE_ROOT / "outputs" / "figure4_reconciliation" / "final_figure4")
    p.add_argument("--include-panel-g", action="store_true")
    p.add_argument("--skip-panel-g", action="store_true")
    p.add_argument("--panel-f-placeholder-ok", action="store_true")
    p.add_argument("--make-supplement-s4", action="store_true")
    p.add_argument("--style", type=str, default="manuscript", choices=["manuscript", "talk"])
    p.add_argument("--export-pdf", action="store_true")
    p.add_argument("--export-svg", action="store_true")
    p.add_argument("--export-png", action="store_true")
    p.add_argument("--dpi", type=int, default=600)
    args = p.parse_args()

    configure_style(args.style)
    paths = _ensure_dirs(args.out_dir)

    recon = args.reconciliation_dir
    canonical_dir = recon / "canonical_discrimination"
    mimicry_dir = recon / "validated_mimicry"
    bundle_dir = recon / "manuscript_bundle"

    decoder_metrics = pd.read_csv(_require_file(canonical_dir / "canonical_decoder_metrics.csv"))
    contrasts = pd.read_csv(_require_file(canonical_dir / "canonical_real_minus_stabilized.csv"))
    sweep = pd.read_csv(_require_file(canonical_dir / "integration_window_sweep.csv"))
    claims = pd.read_csv(_require_file(canonical_dir / "observer_claim_validation.csv"))
    trial_manifest = pd.read_csv(_require_file(canonical_dir / "eoptotype_trial_manifest.csv"))
    occupancy = pd.read_csv(_require_file(mimicry_dir / "occupancy_weighted_mimicry.csv"))
    phase_summary = pd.read_csv(_require_file(mimicry_dir / "phase_landscape_summary.csv"))
    pairwise = pd.read_csv(_require_file(mimicry_dir / "pairwise_mimicry_by_phase.csv"))
    translation = pd.read_csv(_require_file(mimicry_dir / "translation_direction_metrics.csv"))
    numbers = pd.read_csv(_require_file(bundle_dir / "figure4_numbers_for_text.csv"))
    checklist = pd.read_csv(_require_file(bundle_dir / "figure4_claim_checklist.csv"))

    manifest_path = recon / "model_population_manifest.json"
    eye_npz = VISIONCORE_ROOT / "scripts" / "temporal_decoding" / "data" / "eye_traces.npz"
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        eye_src = m.get("dataset_or_trace_source", "")
        if eye_src:
            eye_npz = VISIONCORE_ROOT / str(eye_src)
    _require_file(eye_npz)

    cov = _load_covariance_branch(args.covariance_dir)
    if not cov.available:
        _write_missing_covariance_note(paths["qc"], cov.missing_items)
        if not args.panel_f_placeholder_ok:
            raise RuntimeError(
                "Covariance branch values missing and --panel-f-placeholder-ok was not set."
            )

    # Save source tables used by the plot.
    decoder_metrics.to_csv(paths["source_tables"] / "panel_b_decoder_metrics.csv", index=False)
    contrasts.to_csv(paths["source_tables"] / "panel_b_contrasts.csv", index=False)
    sweep.to_csv(paths["source_tables"] / "panel_c_sweep.csv", index=False)
    claims.to_csv(paths["source_tables"] / "panel_d_claims.csv", index=False)
    occupancy.to_csv(paths["source_tables"] / "panel_g_occupancy.csv", index=False)

    include_g = args.include_panel_g or not args.skip_panel_g

    # Main multipanel composition in Ryan-like layout.
    fig = plt.figure(figsize=(10.4, 12.0 if include_g else 9.5), constrained_layout=False)
    if include_g:
        mosaic = [["A", "B"], ["C", "D"], ["E", "F"], ["G", "G"]]
        gridspec_kw = {"height_ratios": [1.05, 1.05, 1.0, 1.5], "hspace": 0.40, "wspace": 0.34}
    else:
        mosaic = [["A", "B"], ["C", "D"], ["E", "F"]]
        gridspec_kw = {"height_ratios": [1.05, 1.05, 1.0], "hspace": 0.40, "wspace": 0.34}
    axd = fig.subplot_mosaic(mosaic, gridspec_kw=gridspec_kw)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.06)

    a_meta = _plot_panel_a(axd["A"], trial_manifest, eye_npz)
    _plot_panel_b(axd["B"], decoder_metrics, contrasts)
    _plot_panel_c(axd["C"], sweep)
    _plot_panel_d(axd["D"], claims)
    _plot_panel_e(axd["E"])
    _plot_panel_f(axd["F"], cov)
    panel_g_qc: dict[str, Any] = {}
    if include_g:
        panel_g_qc = _plot_panel_g(axd["G"], occupancy)

    for letter in [k for k in ["A", "B", "C", "D", "E", "F", "G"] if k in axd]:
        if letter == "G":
            continue
        _panel_letter(axd[letter], letter)
    if "G" in axd:
        axd["G"].text(-0.14, 1.11, "G", transform=axd["G"].transAxes, ha="left", va="bottom", fontweight="bold", fontsize=12)

    fig.suptitle(
        "Fixational eye movements produce a scale-dependent discrimination effect whose second moment is reafferent V1 shared variability",
        y=0.985,
        fontsize=14,
    )

    _save_figure(
        fig,
        paths["exports"] / "Figure4_multipanel",
        export_pdf=args.export_pdf,
        export_svg=args.export_svg,
        export_png=args.export_png,
        dpi=args.dpi,
    )
    plt.close(fig)

    # Individual panel exports (PNG + SVG).
    panel_specs: dict[str, tuple[str, Any]] = {
        "A": ("counterfactual_input", lambda ax: _plot_panel_a(ax, trial_manifest, eye_npz)),
        "B": ("accuracy_vs_logmar", lambda ax: _plot_panel_b(ax, decoder_metrics, contrasts)),
        "C": ("integration_dependence", lambda ax: _plot_panel_c(ax, sweep)),
        "D": ("observer_mechanism", lambda ax: _plot_panel_d(ax, claims)),
        "E": ("moments_bridge", lambda ax: _plot_panel_e(ax)),
        "F": ("reafferent_covariance_recorded_v1", lambda ax: _plot_panel_f(ax, cov)),
        "G": ("recoverability_geometry", lambda ax: _plot_panel_g(ax, occupancy)),
    }
    for letter, (name, draw) in panel_specs.items():
        if letter == "G" and not include_g:
            continue
        pf = plt.figure(figsize=(3.2, 2.4), constrained_layout=True)
        pax = pf.add_subplot(1, 1, 1)
        draw(pax)
        _panel_letter(pax, letter)
        stem = paths["panels"] / f"Fig4{letter}_{name}"
        pf.savefig(stem.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
        pf.savefig(stem.with_suffix(".svg"), dpi=args.dpi, bbox_inches="tight")
        plt.close(pf)

    canonical_ok, canonical_max_abs_diff = _check_consistency(contrasts, sweep)
    effect_ok, effect_errs = _check_effect_values(contrasts, sweep)
    if not effect_ok:
        raise RuntimeError("effect_size_mismatch: " + " | ".join(effect_errs))

    mean_status_rows = claims[claims["observer_name"] == "mean_only_observer"]
    mean_status = str(mean_status_rows.iloc[0]["status"]) if not mean_status_rows.empty else "unknown"

    sf_real = decoder_metrics[
        np.isclose(decoder_metrics["logmar"].astype(float), -0.35)
        & (decoder_metrics["window"].astype(int) == 1)
        & (decoder_metrics["condition"].map(_condition_norm) == "real")
    ]
    single_frame_real_acc = float(sf_real.iloc[0]["heldout_accuracy"]) if not sf_real.empty else float("nan")
    chance = 0.25
    single_frame_qc_status = "yes" if np.isfinite(single_frame_real_acc) and (single_frame_real_acc > chance) else "no_or_missing"

    if not cov.available:
        final_status = "ready_except_covariance_branch_values"
    elif include_g:
        final_status = "ready_for_main_figure"
    else:
        final_status = "ready_for_main_figure"

    _write_qc_reports(
        paths["qc"],
        canonical_ok=canonical_ok,
        canonical_max_abs_diff=canonical_max_abs_diff,
        effect_ok=effect_ok,
        effect_errs=effect_errs,
        observer_claims=claims,
        cov=cov,
        panel_g_qc=panel_g_qc,
        include_panel_g=include_g,
        final_status=final_status,
        single_frame_real_acc=single_frame_real_acc,
        single_frame_qc_status=single_frame_qc_status,
    )
    _write_methods_and_caption(paths["qc"], mean_status=mean_status)

    if args.make_supplement_s4:
        # Lightweight S4 output from existing validated mimicry artifacts.
        sf = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
        s_ax = sf.subplot_mosaic([["S4A", "S4B"], ["S4C", "S4D"], ["S4E", "S4E"]])

        # S4A: translation norms scatter.
        s_ax["S4A"].scatter(
            translation["j_norm_x"].astype(float).values,
            translation["j_norm_y"].astype(float).values,
            s=3,
            alpha=0.2,
            color="#0072B2",
        )
        s_ax["S4A"].set_xlabel("J norm x")
        s_ax["S4A"].set_ylabel("J norm y")

        # S4B: pair matrix at primary logmar.
        p = phase_summary[np.isclose(phase_summary["logmar"].astype(float), -0.35)].copy()
        pairs = sorted(p["orientation_pair"].unique())
        vals = p.set_index("orientation_pair")["mimicry_mean"].reindex(pairs).fillna(np.nan).values
        s_ax["S4B"].bar(np.arange(len(vals)), vals, color="#4D4D4D")
        s_ax["S4B"].set_xticks(np.arange(len(vals)))
        s_ax["S4B"].set_xticklabels(pairs, rotation=90, fontsize=6)
        s_ax["S4B"].set_ylabel("mimicry mean")

        # S4C: representative landscape.
        pair0 = pairwise[(np.isclose(pairwise["logmar"].astype(float), -0.35)) & (pairwise["orientation_pair"] == "0_vs_90")]
        if not pair0.empty:
            sc = s_ax["S4C"].scatter(pair0["phase_x"], pair0["phase_y"], c=pair0["mimicry_fraction"], cmap="viridis", s=6)
            plt.colorbar(sc, ax=s_ax["S4C"], label="mimicry")
        s_ax["S4C"].set_xlabel("phase_x")
        s_ax["S4C"].set_ylabel("phase_y")

        # S4D: occupancy-weighted vs center.
        s_ax["S4D"].scatter(occupancy["center_mimicry"], occupancy["weighted_mimicry_mean"], s=10, alpha=0.8)
        lo = min(float(occupancy["center_mimicry"].min()), float(occupancy["weighted_mimicry_mean"].min()))
        hi = max(float(occupancy["center_mimicry"].max()), float(occupancy["weighted_mimicry_mean"].max()))
        s_ax["S4D"].plot([lo, hi], [lo, hi], ls="--", lw=0.8, color="0.5")
        s_ax["S4D"].set_xlabel("center")
        s_ax["S4D"].set_ylabel("weighted")

        # S4E: textual schematic.
        s_ax["S4E"].axis("off")
        s_ax["S4E"].text(0.5, 0.60, "Recoverability schematic", ha="center", fontsize=8, fontweight="bold", transform=s_ax["S4E"].transAxes)
        s_ax["S4E"].text(0.5, 0.42, "Low mimicry: separable pose variation", ha="center", fontsize=7, transform=s_ax["S4E"].transAxes)
        s_ax["S4E"].text(0.5, 0.28, "High mimicry: potential identity/pose confound", ha="center", fontsize=7, transform=s_ax["S4E"].transAxes)

        for key, ax in s_ax.items():
            _panel_letter(ax, key)

        _save_figure(
            sf,
            paths["exports"] / "FigureS4_mimicry",
            export_pdf=args.export_pdf,
            export_svg=args.export_svg,
            export_png=args.export_png,
            dpi=args.dpi,
        )
        plt.close(sf)

    print(f"Saved Figure 4 outputs to: {paths['root']}")


if __name__ == "__main__":
    main()
