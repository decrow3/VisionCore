"""Build the BackImage active-sensing collaborator figure pack.

This script is intentionally cache-first.  It reads the reviewed BackImage
analysis outputs and assembles simple PNG/PDF figures for collaborators without
rerunning the twin, feature screens, or aggregate decoders.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "fixation_statistics_by_stimulus_all_sessions_after_review"

DRIFT_DIR = BASE / "backimage_twin_drift_geometry_scaled_n256_twin_axis_only"
DRIFT_ALT_DIR = BASE / "backimage_twin_drift_geometry_scaled_n256_twin_axis_only_yfix"
STABILITY_DIR = BASE / "backimage_edge_parallel_stability_screen_yfix_n256_pop256"
TWIN_AUDIT_DIR = BASE / "backimage_twin_stability_metric_audit"
TWIN_CHEAP_DIR = TWIN_AUDIT_DIR / "cheap_synthesis"
LATENT_DIR = BASE / "backimage_latent_information_scalesweep_n256_rel0125-2_rand8_delta"
AGG_N128_DIR = BASE / "backimage_aggregate_fem_information_n128_k4_rel025-1_gabor_pyramid"
AGG_N128_INC_DIR = AGG_N128_DIR / "incremental_static_plus_motion"
PATHFINDER_DIR = BASE / "backimage_aggregate_fem_information_pathfinder_n64_k2_drift_only_common_unclipped_rel025-2_not_final"
PATHFINDER_INC_DIR = PATHFINDER_DIR / "incremental_static_plus_motion"

COLORS = {
    "empirical": "#1f3a5f",
    "ou": "#d9822b",
    "brownian": "#8f8f8f",
    "rotated": "#7a5ea8",
    "static": "#c7c7c7",
    "edge_parallel": "#2f8f6a",
    "edge_orthogonal": "#c8553d",
    "gabor": "#34699a",
    "pyramid": "#2f8f6a",
}


@dataclass
class ProvenanceRow:
    panel: str
    claim: str
    script: str
    output_folder: str
    source_files_read: str
    main_metric: str
    sample_size: str
    unit_space: str
    known_caveats: str


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=220)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _label_panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def _scale_key(scale_id: str) -> float:
    return float(scale_id.replace("rel_", "").replace("p", ".").replace("x", ""))


def _format_scale(scale_id: str) -> str:
    value = _scale_key(scale_id)
    return f"{value:g}x"


def _errbar_lines(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    lo_col: str,
    hi_col: str,
    label: str,
    color: str,
) -> None:
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    lo = df[lo_col].to_numpy(dtype=float)
    hi = df[hi_col].to_numpy(dtype=float)
    ax.errorbar(
        x,
        y,
        yerr=np.vstack([y - lo, hi - y]),
        marker="o",
        linewidth=1.8,
        capsize=3,
        color=color,
        label=label,
    )


def _synthetic_patch(angle_deg: float, size: int = 80, noise: float = 0.20) -> np.ndarray:
    rng = np.random.default_rng(int(abs(angle_deg) * 13 + size))
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    theta = np.deg2rad(angle_deg)
    axis = x * np.cos(theta) + y * np.sin(theta)
    patch = 0.52 + 0.28 * np.sin(18 * axis) + 0.10 * np.sin(7 * (x - y))
    patch += noise * rng.normal(size=(size, size))
    return np.clip(patch, 0, 1)


def _arrow(ax: plt.Axes, xy: tuple[float, float], angle_deg: float, length: float, color: str, label: str | None = None) -> None:
    theta = np.deg2rad(angle_deg)
    dx = length * np.cos(theta)
    dy = length * np.sin(theta)
    ax.annotate(
        "",
        xy=(xy[0] + dx, xy[1] + dy),
        xytext=(xy[0] - dx, xy[1] - dy),
        arrowprops={"arrowstyle": "<->", "color": color, "linewidth": 2.0, "shrinkA": 0, "shrinkB": 0},
    )
    if label:
        ax.text(xy[0] + dx * 1.08, xy[1] + dy * 1.08, label, color=color, fontsize=7, ha="center", va="center")


def _schematic_patch_panel(ax: plt.Axes, title: str, examples: pd.DataFrame) -> None:
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_axis_off()
    labels = ["strong", "weak", "low coherence"]
    for i, (_, row) in enumerate(examples.iterrows()):
        x0 = 0.03 + i * 0.32
        y0 = 0.18
        extent = [x0, x0 + 0.27, y0, y0 + 0.62]
        edge = float(row["predicted_axis_deg"])
        drift = float(row["real_drift_axis_deg"])
        ax.imshow(_synthetic_patch(edge), cmap="gray", extent=extent, origin="lower", interpolation="bilinear")
        cx = x0 + 0.135
        cy = y0 + 0.31
        _arrow(ax, (cx, cy), edge, 0.09, COLORS["edge_parallel"], "edge")
        _arrow(ax, (cx, cy), drift, 0.075, "#111111", "drift")
        rng = np.random.default_rng(int(row["window_id"]) + 1)
        cov_angle = np.deg2rad(drift)
        rot = np.array([[np.cos(cov_angle), -np.sin(cov_angle)], [np.sin(cov_angle), np.cos(cov_angle)]])
        cloud = rng.normal(size=(80, 2)) @ np.diag([0.035, 0.010]) @ rot.T
        ax.scatter(cx + cloud[:, 0], cy + cloud[:, 1], s=2, c="#111111", alpha=0.25, linewidths=0)
        ax.text(cx, 0.07, labels[i], fontsize=8, ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def make_figure1(out_dir: Path) -> list[ProvenanceRow]:
    align = _read_csv(DRIFT_DIR / "real_vs_predicted_axis_alignment.csv")
    summary = _read_csv(DRIFT_DIR / "alignment_by_objective_summary.csv")
    nulls = _read_csv(DRIFT_DIR / "summary" / "key_null_summary.csv")
    raw = align[(align["objective"] == "raw_edge_axis") & (align["status"] == "ok")].copy()
    if raw.empty:
        raise ValueError("No raw_edge_axis rows found in drift alignment table.")

    strong = raw.sort_values("cos2_alignment", ascending=False).head(1)
    weak = raw.iloc[(raw["cos2_alignment"].abs()).argsort()[:1]]
    low = raw.sort_values("image_orientation_coherence", ascending=True).head(1)
    examples = pd.concat([strong, weak, low], ignore_index=True)

    session_means = raw.groupby("session")["cos2_alignment"].mean().sort_values()
    raw_summary = summary.loc[summary["objective"] == "raw_edge_axis"].iloc[0]
    null_row = nulls[
        (nulls["objective"] == "raw_edge_axis") & (nulls["null_type"] == "random_axis_candidate_grid")
    ]
    p_text = ""
    if not null_row.empty:
        p_text = f", random-axis p={float(null_row.iloc[0]['p_greater_equal']):.4f}"

    fig = plt.figure(figsize=(10.5, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    fig.suptitle("Measured drift is biased toward local image geometry", fontsize=13, fontweight="bold", x=0.02, ha="left")

    _label_panel(ax0, "A")
    _schematic_patch_panel(ax0, "Example window schematic", examples)

    _label_panel(ax1, "B")
    ax1.hist(raw["cos2_alignment"], bins=np.linspace(-1, 1, 25), color="#688fca", edgecolor="white")
    ax1.axvline(0, color="black", linewidth=0.9)
    ax1.axvline(raw["cos2_alignment"].mean(), color=COLORS["edge_parallel"], linewidth=2)
    ax1.set_xlabel("cos(2 delta theta): drift vs edge")
    ax1.set_ylabel("windows")
    ax1.set_title("Window-level alignment", loc="left", fontsize=10)
    _clean_axis(ax1)

    _label_panel(ax2, "C")
    y = np.arange(session_means.size)
    ax2.scatter(session_means.to_numpy(), y, c=np.where(session_means.to_numpy() > 0, COLORS["edge_parallel"], "#b55d60"), s=24)
    ax2.axvline(0, color="black", linewidth=0.9)
    ax2.set_yticks([])
    ax2.set_xlabel("session mean cos(2 delta theta)")
    ax2.set_title("Session summary", loc="left", fontsize=10)
    ax2.text(
        0.02,
        0.04,
        f"{int(raw_summary['n_sessions_positive'])}/{int(raw_summary['n_sessions'])} sessions positive\n"
        f"mean={float(raw_summary['mean_cos2_session_mean']):+.3f}, weighted={float(raw_summary['weighted_cos2_session_mean']):+.3f}{p_text}",
        transform=ax2.transAxes,
        fontsize=8,
        va="bottom",
    )
    _clean_axis(ax2)

    _save(fig, out_dir, "figure1_drift_orientation_geometry")
    return [
        ProvenanceRow(
            "Figure 1A-C",
            "Measured drift/fixation-cloud orientation is modestly aligned with local raw image edge geometry.",
            "run_backimage_twin_drift_geometry.py; summarize_backimage_twin_drift_geometry.py",
            str(DRIFT_DIR),
            "real_vs_predicted_axis_alignment.csv; alignment_by_objective_summary.csv; summary/key_null_summary.csv",
            "cos(2 * (theta_drift - theta_edge)) for raw_edge_axis",
            f"{int(raw_summary['n_windows'])} windows, {int(raw_summary['n_sessions'])} sessions",
            "eye-position geometry and pixel edge/spectral axis",
            f"Raw edge geometry is the biological baseline; model PA/PB/Pareto objectives are not foregrounded. The yfix-named sibling cache is {DRIFT_ALT_DIR.name} and reports a more conservative raw-edge summary.",
        )
    ]


def make_figure2(out_dir: Path) -> list[ProvenanceRow]:
    window = _read_csv(STABILITY_DIR / "edge_parallel_stability_by_window.csv")
    stability = _read_csv(STABILITY_DIR / "stability_summary.csv")
    first_order = _read_csv(TWIN_CHEAP_DIR / "first_order_signed_stability_advantage_session_ci.csv")
    corr = _read_csv(TWIN_CHEAP_DIR / "pixel_vs_twin_signed_stability_correlations.csv")
    twin_window = _read_csv(TWIN_AUDIT_DIR / "twin_stability_metric_by_window.csv")

    fig = plt.figure(figsize=(11.5, 7.2))
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    fig.suptitle("Edge-parallel motion preserves pixels and V1-twin responses", fontsize=13, fontweight="bold", x=0.02, ha="left")

    _label_panel(ax0, "A")
    ax0.set_title("Local displacement test", loc="left", fontsize=10)
    ax0.imshow(_synthetic_patch(35, size=120, noise=0.15), cmap="gray", extent=[0, 1, 0, 1], origin="lower")
    _arrow(ax0, (0.52, 0.52), 35, 0.22, COLORS["edge_parallel"], "parallel")
    _arrow(ax0, (0.52, 0.52), 125, 0.18, COLORS["edge_orthogonal"], "orthogonal")
    ax0.text(0.03, 0.04, "positive = orthogonal disruption - parallel disruption", fontsize=8, color="white")
    ax0.set_axis_off()

    _label_panel(ax1, "B")
    sess = window.groupby("session")["pixel_stability_advantage"].mean().sort_values()
    pix = stability.loc[stability["screen"] == "pixel"].iloc[0]
    ax1.scatter(sess.to_numpy(), np.arange(sess.size), c=np.where(sess.to_numpy() > 0, COLORS["edge_parallel"], COLORS["edge_orthogonal"]), s=24)
    ax1.axvline(0, color="black", linewidth=0.9)
    ax1.errorbar(
        float(pix["mean_advantage_session_mean"]),
        sess.size + 1.0,
        xerr=np.array([[float(pix["mean_advantage_session_mean"] - pix["ci95_low_session_mean"])], [float(pix["ci95_high_session_mean"] - pix["mean_advantage_session_mean"])]]),
        fmt="o",
        color="#111111",
        capsize=4,
    )
    ax1.set_yticks([])
    ax1.set_xlabel("pixel stability advantage")
    ax1.set_title("Pixel edge-parallel advantage", loc="left", fontsize=10)
    ax1.text(0.02, 0.05, f"{int(pix['n_sessions_positive_advantage'])}/{int(pix['n_sessions'])} sessions positive", transform=ax1.transAxes, fontsize=8)
    _clean_axis(ax1)

    _label_panel(ax2, "C")
    metric_order = [
        ("original_twin_relative_screen_metric_raw_mse", "raw MSE"),
        ("response_norm_mse", "response-norm"),
        ("per_rate_mse", "per-rate"),
        ("full_cov_whitened_mse", "full-cov whitened"),
    ]
    rows = []
    for metric, label in metric_order:
        rec = first_order.loc[first_order["metric"] == metric]
        if not rec.empty:
            row = rec.iloc[0].copy()
            row["label"] = label
            rows.append(row)
    metrics = pd.DataFrame(rows)
    x = np.arange(metrics.shape[0])
    y = metrics["mean_session"].to_numpy(dtype=float)
    lo = metrics["ci_low"].to_numpy(dtype=float)
    hi = metrics["ci_high"].to_numpy(dtype=float)
    ax2.bar(x, y, color="#6f9ecf")
    ax2.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), fmt="none", ecolor="black", capsize=3, linewidth=1)
    ax2.axhline(0, color="black", linewidth=0.9)
    for xpos, mean, low, high in zip(x, y, lo, hi, strict=True):
        ax2.text(xpos, max(mean, high) + 0.006, f"{mean:.3g}\n[{low:.3g}, {high:.3g}]", ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics["label"], rotation=20, ha="right")
    ax2.set_ylabel("twin stability advantage, native units")
    ax2.set_title("V1-twin metrics", loc="left", fontsize=10)
    ax2.text(0.02, 0.94, "metric units differ; labels show mean and CI", transform=ax2.transAxes, fontsize=8, va="top")
    _clean_axis(ax2)

    _label_panel(ax3, "D")
    xcol = "pixel_stability_advantage"
    ycol = "full_cov_whitened_mse_stability_advantage"
    ax3.scatter(twin_window[xcol], twin_window[ycol], s=15, c="#1f3a5f", alpha=0.45, linewidths=0)
    ax3.axhline(0, color="#777777", linewidth=0.8)
    ax3.axvline(0, color="#777777", linewidth=0.8)
    ax3.set_xlabel("pixel advantage")
    ax3.set_ylabel("full-cov twin advantage")
    ax3.set_title("Pixel/twin agreement", loc="left", fontsize=10)
    crow = corr[
        (corr["metric"] == "full_cov_whitened_mse") & (corr["level"] == "window_within_session")
    ]
    if not crow.empty:
        row = crow.iloc[0]
        ax3.text(0.03, 0.95, f"within-session r={float(row['r']):+.3f}\nCI [{float(row['ci_low']):+.3f}, {float(row['ci_high']):+.3f}]", transform=ax3.transAxes, fontsize=8, va="top")
    _clean_axis(ax3)

    _save(fig, out_dir, "figure2_edge_parallel_preservation")
    return [
        ProvenanceRow(
            "Figure 2A-D",
            "Motion along the local edge direction is less disruptive than orthogonal motion in pixels and V1-twin responses.",
            "run_backimage_edge_parallel_stability_screen.py; posthoc_backimage_twin_stability_metric_audit.py; summarize_backimage_twin_stability_metric_audit.py",
            f"{STABILITY_DIR}; {TWIN_AUDIT_DIR}",
            "edge_parallel_stability_by_window.csv; stability_summary.csv; cheap_synthesis/*.csv; twin_stability_metric_by_window.csv",
            "edge-orthogonal disruption - edge-parallel disruption",
            f"{int(pix['n_windows'])} windows, {int(pix['n_sessions'])} sessions",
            "pixels and V1-twin response metrics",
            "This tests edge-parallel preservation, not whether a response-stability optimizer predicts the measured drift axis.",
        )
    ]


def make_figure3(out_dir: Path) -> list[ProvenanceRow]:
    contrasts = _read_csv(LATENT_DIR / "posthoc_candidate_contrast_summary.csv")
    clipping = _read_csv(LATENT_DIR / "posthoc_scale_clipping_summary.csv")
    contrasts = contrasts[contrasts["observer"] == "pose_blind_delta_mean"].copy()

    specs = [
        ("gabor_local_field", 4, "Gabor k=4", COLORS["gabor"]),
        ("pyramid_local_field", 8, "Pyramid k=8", COLORS["pyramid"]),
    ]
    fig = plt.figure(figsize=(11.5, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.85, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    fig.suptitle("Local feature readout shows a small-scale real-vs-random signal", fontsize=13, fontweight="bold", x=0.02, ha="left")

    _label_panel(ax0, "A")
    for latent, k, label, color in specs:
        sub = contrasts[
            (contrasts["latent_name"] == latent)
            & (contrasts["pca_k"] == k)
            & (contrasts["contrast"] == "real_minus_random")
        ].sort_values("motion_scale_value")
        _errbar_lines(ax0, sub, "motion_scale_value", "mean_score_delta", "ci_low", "ci_high", label, color)
    ax0.axhline(0, color="black", linewidth=0.9)
    ax0.axvspan(0.22, 0.28, color="#f3d27a", alpha=0.25, linewidth=0)
    ax0.set_xscale("log", base=2)
    ax0.set_xticks([0.125, 0.25, 0.5, 1, 2])
    ax0.set_xticklabels(["0.125x", "0.25x", "0.5x", "1x", "2x"])
    ax0.set_xlabel("observed RMS scale")
    ax0.set_ylabel("real - random score")
    ax0.set_title("Real drift axis vs random axes", loc="left", fontsize=10)
    ax0.legend(frameon=False, fontsize=8)
    _clean_axis(ax0)

    _label_panel(ax1, "B")
    clip = clipping.sort_values("motion_scale_value")
    clipped = clip["fraction_rms_clipped_high"].to_numpy(dtype=float) + clip["fraction_rms_clipped_low"].to_numpy(dtype=float)
    ax1.plot(clip["motion_scale_value"], 100 * clipped, marker="o", color="#9a6a20")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks([0.125, 0.25, 0.5, 1, 2])
    ax1.set_xticklabels(["0.125x", "0.25x", "0.5x", "1x", "2x"], rotation=20)
    ax1.set_ylabel("clipped windows (%)")
    ax1.set_title("Scale clipping", loc="left", fontsize=10)
    ax1.text(0.03, 0.93, "2x is clipping-heavy", transform=ax1.transAxes, fontsize=8, va="top")
    _clean_axis(ax1)

    _label_panel(ax2, "C")
    for latent, k, label, color in specs:
        sub = contrasts[
            (contrasts["latent_name"] == latent)
            & (contrasts["pca_k"] == k)
            & (contrasts["contrast"] == "real_minus_edge")
        ].sort_values("motion_scale_value")
        _errbar_lines(ax2, sub, "motion_scale_value", "mean_score_delta", "ci_low", "ci_high", label, color)
    ax2.axhline(0, color="black", linewidth=0.9)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks([0.125, 0.25, 0.5, 1, 2])
    ax2.set_xticklabels(["0.125x", "0.25x", "0.5x", "1x", "2x"])
    ax2.set_xlabel("observed RMS scale")
    ax2.set_ylabel("real - edge score")
    ax2.set_title("Real does not robustly beat edge", loc="left", fontsize=10)
    _clean_axis(ax2)

    _save(fig, out_dir, "figure3_local_Iz_scale_screen")
    return [
        ProvenanceRow(
            "Figure 3A-C",
            "Local feature-information screens show their cleanest real-vs-random support near 0.25x, with scale/clipping caveats.",
            "run_backimage_latent_information_screen.py; summarize_backimage_latent_information_screen.py; audit_backimage_latent_real_random.py",
            str(LATENT_DIR),
            "posthoc_candidate_contrast_summary.csv; posthoc_scale_clipping_summary.csv",
            "real minus random and real minus raw edge decoding score",
            "256 windows, 29 sessions",
            "Gabor/pyramid local feature readouts",
            "This is a regime-dependent local readout signal, not a global local-feature infomax proof.",
        )
    ]


def _plot_design_schematic(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_title("Aggregate design", loc="left", fontsize=10)
    for i, lab in enumerate(["image patches", "trace bank", "motion families", "decode features"]):
        y = 0.78 - i * 0.21
        ax.add_patch(plt.Rectangle((0.08, y - 0.055), 0.36, 0.11, color="#edf1f5", ec="#777777", lw=0.8))
        ax.text(0.26, y, lab, ha="center", va="center", fontsize=8)
        if i < 3:
            ax.annotate("", xy=(0.26, y - 0.12), xytext=(0.26, y - 0.06), arrowprops={"arrowstyle": "->", "lw": 1})
    families = [("empirical", COLORS["empirical"]), ("OU", COLORS["ou"]), ("Brownian", COLORS["brownian"]), ("rotated", COLORS["rotated"])]
    for j, (lab, color) in enumerate(families):
        ax.plot([0.55, 0.92], [0.67 - j * 0.12] * 2, color=color, lw=2)
        ax.text(0.94, 0.67 - j * 0.12, lab, fontsize=8, va="center")
    ax.text(0.55, 0.10, "R_static vs R_static + R_motion", fontsize=8)


def make_figure4(out_dir: Path) -> list[ProvenanceRow]:
    gains = _read_csv(AGG_N128_INC_DIR / "incremental_gain_vs_static.csv")
    contrasts = _read_csv(AGG_N128_INC_DIR / "incremental_gain_contrasts.csv")
    pf_gains = _read_csv(PATHFINDER_INC_DIR / "incremental_gain_vs_static.csv")
    pf_contrasts = _read_csv(PATHFINDER_INC_DIR / "incremental_gain_contrasts.csv")

    fig = plt.figure(figsize=(12, 7.4))
    gs = fig.add_gridspec(2, 3)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    fig.suptitle("Aggregate feature-information signals are promising but trace-bank/readout dependent", fontsize=13, fontweight="bold", x=0.02, ha="left")

    _label_panel(axes[0], "A")
    _plot_design_schematic(axes[0])

    _label_panel(axes[1], "B")
    sub = gains[
        (gains["family"] == "empirical")
        & (gains["latent"] == "pyramid_local_field")
        & (gains["k"] == 8)
        & (gains["motion_summary"].isin(["temporal_pca", "temporal_dct"]))
    ].copy()
    sub["scale"] = sub["scale_id"].map(_scale_key)
    for mode, color in [("temporal_pca", "#34699a"), ("temporal_dct", "#2f8f6a")]:
        block = sub[sub["motion_summary"] == mode].sort_values("scale")
        _errbar_lines(axes[1], block, "scale", "incremental_gain_neg_mse", "ci95_low", "ci95_high", mode.replace("_", " "), color)
    axes[1].axhline(0, color="black", linewidth=0.9)
    axes[1].set_xticks([0.25, 0.5, 1.0])
    axes[1].set_xticklabels(["0.25x", "0.5x", "1x"])
    axes[1].set_ylabel("incremental gain vs static")
    axes[1].set_title("n=128 pyramid k=8 empirical", loc="left", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    _clean_axis(axes[1])

    _label_panel(axes[2], "C")
    sub = contrasts[
        (contrasts["lhs_family"] == "empirical")
        & (contrasts["rhs_family"] == "ou")
        & (contrasts["latent"] == "pyramid_local_field")
        & (contrasts["k"] == 8)
        & (contrasts["motion_summary"].isin(["temporal_pca", "temporal_dct"]))
    ].copy()
    sub["scale"] = sub["scale_id"].map(_scale_key)
    for mode, color in [("temporal_pca", "#34699a"), ("temporal_dct", "#2f8f6a")]:
        block = sub[sub["motion_summary"] == mode].sort_values("scale")
        _errbar_lines(axes[2], block, "scale", "incremental_gain_delta_neg_mse", "ci95_low", "ci95_high", mode.replace("_", " "), color)
    axes[2].axhline(0, color="black", linewidth=0.9)
    axes[2].set_xticks([0.25, 0.5, 1.0])
    axes[2].set_xticklabels(["0.25x", "0.5x", "1x"])
    axes[2].set_ylabel("empirical - OU gain")
    axes[2].set_title("n=128 empirical exceeds OU", loc="left", fontsize=10)
    _clean_axis(axes[2])

    _label_panel(axes[3], "D")
    pf = pf_gains[
        (pf_gains["family"] == "empirical")
        & (pf_gains["latent"] == "pyramid_local_field")
        & (pf_gains["k"] == 8)
        & (pf_gains["motion_summary"].isin(["temporal_pca", "temporal_dct"]))
    ].copy()
    pf["scale"] = pf["scale_id"].map(_scale_key)
    for mode, color in [("temporal_pca", "#34699a"), ("temporal_dct", "#2f8f6a")]:
        block = pf[pf["motion_summary"] == mode].sort_values("scale")
        _errbar_lines(axes[3], block, "scale", "incremental_gain_neg_mse", "ci95_low", "ci95_high", mode.replace("_", " "), color)
    axes[3].axhline(0, color="black", linewidth=0.9)
    axes[3].set_xticks([0.25, 0.5, 1, 1.5, 2])
    axes[3].set_xticklabels(["0.25x", "0.5x", "1x", "1.5x", "2x"])
    axes[3].set_ylabel("incremental gain vs static")
    axes[3].set_title("n=64 drift-only pathfinder, not final", loc="left", fontsize=10)
    _clean_axis(axes[3])

    _label_panel(axes[4], "E")
    pf = pf_contrasts[
        (pf_contrasts["lhs_family"] == "empirical")
        & (pf_contrasts["rhs_family"].isin(["ou", "rotated"]))
        & (pf_contrasts["latent"] == "pyramid_local_field")
        & (pf_contrasts["k"] == 8)
        & (pf_contrasts["motion_summary"] == "temporal_pca")
    ].copy()
    pf["scale"] = pf["scale_id"].map(_scale_key)
    for rhs, color in [("ou", COLORS["ou"]), ("rotated", COLORS["rotated"])]:
        block = pf[pf["rhs_family"] == rhs].sort_values("scale")
        _errbar_lines(axes[4], block, "scale", "incremental_gain_delta_neg_mse", "ci95_low", "ci95_high", f"empirical - {rhs}", color)
    axes[4].axhline(0, color="black", linewidth=0.9)
    axes[4].set_xticks([0.25, 0.5, 1, 1.5, 2])
    axes[4].set_xticklabels(["0.25x", "0.5x", "1x", "1.5x", "2x"])
    axes[4].set_ylabel("contrast in gain")
    axes[4].set_title("Controls in cleaned pathfinder", loc="left", fontsize=10)
    axes[4].legend(frameon=False, fontsize=8)
    _clean_axis(axes[4])

    _label_panel(axes[5], "F")
    png = PATHFINDER_DIR / "figures" / "incremental_empirical_control_contrasts_primary.png"
    axes[5].set_title("Existing pathfinder diagnostic", loc="left", fontsize=10)
    if png.exists():
        axes[5].imshow(mpimg.imread(png))
        axes[5].set_axis_off()
    else:
        axes[5].text(0.5, 0.5, "diagnostic PNG missing", ha="center", va="center")
        axes[5].set_axis_off()

    _save(fig, out_dir, "figure4_aggregate_information_pathfinder")
    return [
        ProvenanceRow(
            "Figure 4A-F",
            "Aggregate FEM feature-information signals are positive in the n=128 run but become unsettled under the cleaned drift-only pathfinder.",
            "run_backimage_aggregate_fem_information.py; summarize_backimage_aggregate_fem_information.py",
            f"{AGG_N128_DIR}; {PATHFINDER_DIR}",
            "incremental_static_plus_motion/incremental_gain_vs_static.csv; incremental_gain_contrasts.csv; figures/incremental_empirical_control_contrasts_primary.png",
            "incremental feature-decoding gain and empirical-minus-control gain",
            "n=128 aggregate run; n=64/K=2 drift-only pathfinder",
            "aggregate Gabor/pyramid feature readouts from V1-twin summaries",
            "The n=64 drift-only pathfinder is explicitly not final and should not be used as a load-bearing conclusion.",
        )
    ]


def make_figure_s1(out_dir: Path) -> list[ProvenanceRow]:
    trace = _read_csv(PATHFINDER_DIR / "trace_bank_metadata.csv")
    manifest = _read_csv(PATHFINDER_DIR / "pathfinder_n64_manifest.csv")
    metadata = _read_json(PATHFINDER_DIR / "run_metadata.json")
    cfg = metadata.get("config", {})

    fig = plt.figure(figsize=(11.5, 6.5))
    gs = fig.add_gridspec(2, 3)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    fig.suptitle("Drift-only source bank QC", fontsize=13, fontweight="bold", x=0.02, ha="left")

    _label_panel(axes[0], "A")
    rms_values = pd.concat([manifest["source_observed_rms_radius_deg"], trace["source_rms_radius_deg"]]).astype(float)
    rms_values = rms_values[np.isfinite(rms_values) & (rms_values > 0)]
    rms_bins = np.geomspace(rms_values.min() * 0.8, rms_values.max() * 1.2, 18)
    axes[0].hist(manifest["source_observed_rms_radius_deg"], bins=rms_bins, color="#7796c7", edgecolor="white", alpha=0.75, label="manifest")
    axes[0].hist(trace["source_rms_radius_deg"], bins=rms_bins, color="#263b63", alpha=0.45, label="trace bank")
    if "max_trace_source_rms_deg" in cfg:
        axes[0].axvline(float(cfg["max_trace_source_rms_deg"]), color="#b55d60", linewidth=1.5, label="RMS guard")
    axes[0].set_xlabel("source RMS radius (deg)")
    axes[0].set_ylabel("source traces")
    axes[0].set_title("RMS distribution", loc="left", fontsize=10)
    axes[0].set_xscale("log")
    axes[0].legend(frameon=False, fontsize=8)
    _clean_axis(axes[0])

    _label_panel(axes[1], "B")
    counts = trace["n_microsaccade_events"].value_counts().sort_index()
    axes[1].bar(counts.index.astype(str), counts.values, color="#6f9ecf")
    axes[1].set_xlabel("detected microsaccade events")
    axes[1].set_ylabel("source traces")
    axes[1].set_title("Microsaccade screen metadata", loc="left", fontsize=10)
    axes[1].text(0.03, 0.95, f"requested max events: {cfg.get('max_trace_source_microsaccade_events', 'NA')}", transform=axes[1].transAxes, fontsize=8, va="top")
    _clean_axis(axes[1])

    _label_panel(axes[2], "C")
    axes[2].scatter(trace["source_rms_radius_deg"], trace["source_speed_p95_deg_s"], c=trace["n_microsaccade_events"], cmap="viridis", s=30)
    if "max_trace_source_rms_deg" in cfg:
        axes[2].axvline(float(cfg["max_trace_source_rms_deg"]), color="#b55d60", linewidth=1.2)
    if "max_trace_source_speed_p95_deg_s" in cfg:
        axes[2].axhline(float(cfg["max_trace_source_speed_p95_deg_s"]), color="#b55d60", linewidth=1.2)
    axes[2].set_xlabel("RMS radius (deg)")
    axes[2].set_ylabel("p95 speed (deg/s)")
    axes[2].set_title("Physical guards", loc="left", fontsize=10)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    rms_guard = float(cfg.get("max_trace_source_rms_deg", np.inf))
    speed_guard = float(cfg.get("max_trace_source_speed_p95_deg_s", np.inf))
    n_guard_violations = int(((trace["source_rms_radius_deg"] > rms_guard) | (trace["source_speed_p95_deg_s"] > speed_guard)).sum())
    axes[2].text(0.03, 0.95, f"{n_guard_violations} rows outside saved guards", transform=axes[2].transAxes, fontsize=8, va="top")
    _clean_axis(axes[2])

    _label_panel(axes[3], "D")
    axes[3].scatter(trace["source_max_radius_deg"], trace["source_path_length_deg"], c="#4c78a8", s=26, alpha=0.75)
    if "max_trace_source_radius_deg" in cfg:
        axes[3].axvline(float(cfg["max_trace_source_radius_deg"]), color="#b55d60", linewidth=1.2)
    if "max_trace_source_path_length_deg" in cfg:
        axes[3].axhline(float(cfg["max_trace_source_path_length_deg"]), color="#b55d60", linewidth=1.2)
    axes[3].set_xlabel("max radius (deg)")
    axes[3].set_ylabel("path length (deg)")
    axes[3].set_title("Radius/path-length guards", loc="left", fontsize=10)
    axes[3].set_xscale("log")
    axes[3].set_yscale("log")
    radius_guard = float(cfg.get("max_trace_source_radius_deg", np.inf))
    path_guard = float(cfg.get("max_trace_source_path_length_deg", np.inf))
    n_path_violations = int(((trace["source_max_radius_deg"] > radius_guard) | (trace["source_path_length_deg"] > path_guard)).sum())
    axes[3].text(0.03, 0.95, f"{n_path_violations} rows outside saved guards", transform=axes[3].transAxes, fontsize=8, va="top")
    _clean_axis(axes[3])

    _label_panel(axes[4], "E")
    axes[4].axis("off")
    rows = [
        ("manifest rows", len(manifest)),
        ("unique source rows", trace["source_row"].nunique()),
        ("sessions", trace["session"].nunique()),
        ("zero-event rows", int((trace["n_microsaccade_events"] == 0).sum())),
        ("same-source reuse across scales", "enabled" if cfg.get("reuse_trace_sources_across_scales") else "not recorded"),
    ]
    for i, (key, val) in enumerate(rows):
        axes[4].text(0.05, 0.88 - i * 0.16, f"{key}: {val}", fontsize=10)
    axes[4].set_title("Source count and reuse", loc="left", fontsize=10)

    _label_panel(axes[5], "F")
    axes[5].hist(trace["lag1_autocorr"], bins=16, color="#5e8c61", edgecolor="white")
    axes[5].set_xlabel("lag-1 autocorrelation")
    axes[5].set_ylabel("source traces")
    axes[5].set_title("Temporal smoothness", loc="left", fontsize=10)
    _clean_axis(axes[5])

    _save(fig, out_dir, "figureS1_trace_bank_qc")
    return [
        ProvenanceRow(
            "Figure S1A-F",
            "Trace-bank QC documents source RMS, speed, microsaccade metadata, and same-source reuse for the drift-only pathfinder.",
            "run_backimage_aggregate_fem_information.py; jake/twininfo/eye_controls.py; jake/twininfo/trace_selection.py; declan/vernier_active_sensing/trajectories.py",
            str(PATHFINDER_DIR),
            "trace_bank_metadata.csv; pathfinder_n64_manifest.csv; run_metadata.json",
            "trace-source RMS, speed, radius, path length, microsaccade count, lag-1 autocorrelation",
            f"{len(manifest)} manifest rows, {trace['source_row'].nunique()} unique source rows",
            "eye trace source bank",
            "The saved metadata does not match the note's 40 accepted source-trace anchor; the figure reports the cache contents rather than rerunning the detector.",
        )
    ]


def make_figure_s2(out_dir: Path) -> list[ProvenanceRow]:
    motion = _read_csv(PATHFINDER_DIR / "aggregate_motion_summary.csv")
    motion["scale"] = motion["scale_id"].map(_scale_key)
    families = ["empirical", "ou", "brownian", "rotated"]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4))
    axes = axes.ravel()
    fig.suptitle("Motion matching and scale sanity checks", fontsize=13, fontweight="bold", x=0.02, ha="left")

    panels = [
        ("A", "median_effective_to_requested_rms", "effective/requested RMS", "RMS ratio"),
        ("B", "clipped_fraction", "clipped fraction", "fraction clipped"),
        ("C", "median_path_length_deg", "path length", "deg"),
        ("D", "median_speed_mean_deg_s", "mean speed", "deg/s"),
    ]
    for ax, (label, col, title, ylabel) in zip(axes, panels, strict=True):
        _label_panel(ax, label)
        for fam in families:
            block = motion[motion["family"] == fam].sort_values("scale")
            ax.plot(block["scale"], block[col], marker="o", label=fam, color=COLORS[fam], linewidth=1.8)
        ax.set_xticks([0.25, 0.5, 1, 1.5, 2])
        ax.set_xticklabels(["0.25x", "0.5x", "1x", "1.5x", "2x"])
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("requested observed-RMS scale")
        if col in {"median_effective_to_requested_rms", "clipped_fraction"}:
            ax.axhline(1.0 if col == "median_effective_to_requested_rms" else 0.0, color="black", linewidth=0.8)
        _clean_axis(ax)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    _save(fig, out_dir, "figureS2_motion_family_sanity")
    return [
        ProvenanceRow(
            "Figure S2A-D",
            "The pathfinder motion families have matched effective scale and zero clipping in the common-unclipped drift-only cache.",
            "run_backimage_aggregate_fem_information.py; summarize_backimage_aggregate_fem_information.py",
            str(PATHFINDER_DIR),
            "aggregate_motion_summary.csv; figures/motion_sanity.png",
            "effective/requested RMS ratio, clipping, path length, speed",
            "4 motion families across 5 scales",
            "aggregate generated motion families",
            "This is a sanity check for the pathfinder cache, not an endpoint information result.",
        )
    ]


def write_provenance(out_dir: Path, rows: Iterable[ProvenanceRow]) -> None:
    columns = [
        "panel",
        "claim",
        "script",
        "output folder",
        "source files read",
        "main metric",
        "sample size",
        "unit space",
        "known caveats",
    ]
    lines = ["# BackImage Active-Sensing Collaborator Figure Provenance", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        vals = [
            row.panel,
            row.claim,
            row.script,
            row.output_folder,
            row.source_files_read,
            row.main_metric,
            row.sample_size,
            row.unit_space,
            row.known_caveats,
        ]
        vals = [str(v).replace("\n", " ").replace("|", "/") for v in vals]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("Central message: The active-sensing evidence is strongest for image-structured preservation. Feature-information analyses add support, but they are not yet a single unified optimizer proof.")
    (out_dir / "provenance_table.md").write_text("\n".join(lines), encoding="utf-8")


def build_pack(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[ProvenanceRow] = []
    rows.extend(make_figure1(out_dir))
    rows.extend(make_figure2(out_dir))
    rows.extend(make_figure3(out_dir))
    rows.extend(make_figure4(out_dir))
    rows.extend(make_figure_s1(out_dir))
    rows.extend(make_figure_s2(out_dir))
    write_provenance(out_dir, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make BackImage active-sensing collaborator figure pack.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to dated collaborator pack under reviewed outputs.",
    )
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y%m%d"),
        help="Date stamp for the default output directory, format YYYYMMDD.",
    )
    args = parser.parse_args()
    out_dir = args.out_dir or (BASE / f"backimage_active_sensing_collab_figures_{args.date}")
    build_pack(out_dir)
    print(f"Wrote BackImage active-sensing collaborator figures to {out_dir}")


if __name__ == "__main__":
    main()
