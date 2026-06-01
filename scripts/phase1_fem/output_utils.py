"""
Phase 1 FEM/V1 covariance analysis: output utilities.

Handles CSV writing and figure generation for all Phase 1 analyses.
Every figure has an accompanying CSV (plan requirement).
"""
from __future__ import annotations

import csv
import io
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# Use Agg backend to avoid display issues on headless machines
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

PHASE1_SUBDIRS = (
    "configs",
    "qc",
    "covariance_geometry",
    "noise_correlations",
    "aggregation_scaling",
    "controls",
    "summaries",
    "figures",
    "logs",
)


def make_output_tree(root: Path, session: str | None = None) -> dict[str, Path]:
    """
    Create the Phase 1 output directory tree and return a dict of paths.

    If session is given, create per-session subdirs under root/session/.
    Also creates top-level shared dirs.
    """
    dirs: dict[str, Path] = {}

    # Top-level dirs
    for d in PHASE1_SUBDIRS:
        p = root / d
        p.mkdir(parents=True, exist_ok=True)
        dirs[d] = p

    # Per-session dirs (mirror structure under root/session/)
    if session is not None:
        sess_root = root / session
        for d in PHASE1_SUBDIRS:
            p = sess_root / d
            p.mkdir(parents=True, exist_ok=True)
            dirs[f"session_{d}"] = p

    return dirs


# ---------------------------------------------------------------------------
# CSV utilities
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    """Write a list of dicts to a CSV file, creating parent dirs as needed."""
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dict_to_csv_row(d: dict) -> dict:
    """Convert a dict so all values are CSV-safe (no arrays, no None→empty)."""
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v) if np.isfinite(float(v)) else "nan"
        elif isinstance(v, float):
            out[k] = v if np.isfinite(v) else "nan"
        elif isinstance(v, np.ndarray):
            out[k] = f"<array shape={v.shape}>"
        elif isinstance(v, (list, dict)):
            out[k] = f"<{type(v).__name__} len={len(v)}>"
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Config YAML writing
# ---------------------------------------------------------------------------

def write_config(path: Path, cfg: dict) -> None:
    """Write a flat dict as a simple YAML-like config file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Auto-generated phase1 config"]
    for k, v in cfg.items():
        if isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item}")
        elif isinstance(v, dict):
            lines.append(f"{k}:")
            for ik, iv in v.items():
                lines.append(f"  {ik}: {iv}")
        else:
            lines.append(f"{k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# QC figures
# ---------------------------------------------------------------------------

def figure_session_qc_overview(
    session_qc_rows: list[dict],
    path: Path,
) -> None:
    """Multi-panel session QC summary figure."""
    if not session_qc_rows:
        return

    sessions = [r["session"] for r in session_qc_rows]
    n = len(sessions)
    x = np.arange(n)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Phase 1 session QC overview", fontsize=12)

    def _bar(ax, vals, title, ylabel):
        ax.bar(x, vals, color="steelblue", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(sessions, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)

    _bar(axes[0, 0],
         [r.get("n_units_primary", 0) for r in session_qc_rows],
         "Primary units", "count")
    _bar(axes[0, 1],
         [r.get("valid_bin_fraction", 0) for r in session_qc_rows],
         "Valid bin fraction", "fraction")
    _bar(axes[0, 2],
         [r.get("frac_image_time_cells_ge2_repeats", 0) for r in session_qc_rows],
         "Image-time cells ≥2 repeats", "fraction")
    _bar(axes[1, 0],
         [r.get("mean_rate_median", 0) for r in session_qc_rows],
         "Median firing rate", "Hz")
    _bar(axes[1, 1],
         [r.get("eye_x_std", 0) for r in session_qc_rows],
         "Eye X std", "deg")
    _bar(axes[1, 2],
         [r.get("median_repeats_per_image_time_cell", 0) for r in session_qc_rows],
         "Median repeats per image-time", "count")

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Covariance geometry figures
# ---------------------------------------------------------------------------

def figure_eigenspectra_by_session(
    eigenspectrum_records: list[dict],
    path: Path,
    n_to_show: int = 20,
) -> None:
    """
    Eigenspectrum curves for each session (B_emp estimator and McFarland).
    eigenspectrum_records: list of dicts with keys 'session', 'eigenvalues',
    'estimator', and standard eigenspectrum metric fields.
    """
    if not eigenspectrum_records:
        return

    sessions = list(dict.fromkeys(r["session"] for r in eigenspectrum_records))
    n_sess = len(sessions)
    fig, axes = plt.subplots(1, n_sess, figsize=(4 * n_sess, 4), squeeze=False)
    fig.suptitle("FEM covariance eigenspectra by session", fontsize=11)

    colors = {"B_emp_regression": "steelblue", "mcfarland": "darkorange"}

    for col, sess in enumerate(sessions):
        ax = axes[0, col]
        for rec in eigenspectrum_records:
            if rec["session"] != sess:
                continue
            evals = rec.get("eigenvalues")
            if evals is None or (hasattr(evals, "__len__") and len(evals) == 0):
                continue
            evals = np.asarray(evals)
            total = evals.sum()
            if total <= 0:
                continue
            n_show = min(n_to_show, len(evals))
            estimator = rec.get("estimator", "unknown")
            ax.plot(
                np.arange(1, n_show + 1),
                evals[:n_show] / total,
                marker="o",
                markersize=3,
                label=estimator,
                color=colors.get(estimator, "gray"),
            )
        ax.set_title(sess.replace("Allen_", ""), fontsize=9)
        ax.set_xlabel("PC index", fontsize=8)
        ax.set_ylabel("Variance fraction", fontsize=8)
        ax.legend(fontsize=7)
        ax.set_xlim(0.5, n_to_show + 0.5)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_fem_dimensionality_summary(
    session_metrics_rows: list[dict],
    eye_shuffle_records: list[dict],
    path: Path,
) -> None:
    """
    Summary figure: top2 variance fraction and participation ratio per session,
    with eye-shuffle p95 overlaid.
    """
    if not session_metrics_rows:
        return

    sessions = [r["session"] for r in session_metrics_rows]
    n = len(sessions)
    x = np.arange(n)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("FEM covariance dimensionality summary", fontsize=11)

    top2_obs = [r.get("b_emp_top2_fraction", float("nan")) for r in session_metrics_rows]
    top2_shuf = [r.get("eye_shuffle_p95_top2", float("nan")) for r in session_metrics_rows]
    pr_obs = [r.get("b_emp_participation_ratio", float("nan")) for r in session_metrics_rows]

    ax = axes[0]
    ax.bar(x, top2_obs, color="steelblue", label="Observed", alpha=0.8)
    ax.scatter(x, top2_shuf, color="tomato", zorder=5, label="Eye shuffle p95", marker="_", s=200, linewidths=2)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("Allen_", "") for s in sessions], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Top-2 variance fraction", fontsize=9)
    ax.set_title("B_emp top-2 variance fraction", fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x, pr_obs, color="steelblue", alpha=0.8)
    ax.axhline(2.0, color="gray", linestyle="--", lw=1.5, label="PR=2 (2D translation)")
    ax.axhline(5.0, color="lightgray", linestyle=":", lw=1.0, label="PR=5")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("Allen_", "") for s in sessions], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Participation ratio", fontsize=9)
    ax.set_title("Participation ratio", fontsize=9)
    ax.legend(fontsize=8)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_shared_vs_image_specific(
    session_metrics_rows: list[dict],
    path: Path,
) -> None:
    """Bar chart comparing R² of shared vs image-specific basis per session."""
    if not session_metrics_rows:
        return

    sessions = [r["session"] for r in session_metrics_rows]
    n = len(sessions)
    x = np.arange(n)
    width = 0.35

    r2_shared = [r.get("mean_r2_shared", float("nan")) for r in session_metrics_rows]
    r2_img = [r.get("mean_r2_image_specific", float("nan")) for r in session_metrics_rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, r2_shared, width, label="Shared basis", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, r2_img, width, label="Image-specific basis", color="darkorange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("Allen_", "") for s in sessions], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean R² (cross-validated)", fontsize=9)
    ax.set_title("Shared vs image-specific eye basis R²", fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_image_shuffle_eye_shuffle_controls(
    session_metrics_rows: list[dict],
    path: Path,
) -> None:
    """Observed top-2 fraction vs shuffle controls, per session."""
    if not session_metrics_rows:
        return

    sessions = [r["session"] for r in session_metrics_rows]
    n = len(sessions)
    x = np.arange(n)
    width = 0.25

    obs = [r.get("b_emp_top2_fraction", float("nan")) for r in session_metrics_rows]
    eye_p95 = [r.get("eye_shuffle_p95_top2", float("nan")) for r in session_metrics_rows]
    img_p95 = [r.get("image_shuffle_p95_top2", float("nan")) for r in session_metrics_rows]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width, obs, width, label="Observed", color="steelblue", alpha=0.9)
    ax.bar(x, eye_p95, width, label="Eye shuffle p95", color="tomato", alpha=0.7)
    ax.bar(x + width, img_p95, width, label="Image shuffle p95", color="goldenrod", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("Allen_", "") for s in sessions], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Top-2 variance fraction", fontsize=9)
    ax.set_title("Observed vs shuffle controls", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Aggregation scaling figures
# ---------------------------------------------------------------------------

def figure_aggregation_scaling_curves(
    curve_rows: list[dict],
    path: Path,
) -> None:
    """Plot split-half reliability versus number of windows for each session."""
    if not curve_rows:
        return

    sessions = list(dict.fromkeys(r.get("session", "") for r in curve_rows if r.get("session", "")))
    if not sessions:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.get_cmap("tab10")

    for i, session in enumerate(sessions):
        rows = [r for r in curve_rows if r.get("session") == session]
        rows = sorted(rows, key=lambda r: int(r.get("n_windows", 0)))
        if not rows:
            continue

        x = np.array([float(r.get("n_windows", np.nan)) for r in rows], dtype=np.float64)
        y = np.array([float(r.get("split_half_subspace_overlap_median", np.nan)) for r in rows], dtype=np.float64)
        lo = np.array([float(r.get("split_half_subspace_overlap_ci_lo", np.nan)) for r in rows], dtype=np.float64)
        hi = np.array([float(r.get("split_half_subspace_overlap_ci_hi", np.nan)) for r in rows], dtype=np.float64)

        ok = np.isfinite(x) & np.isfinite(y)
        if int(ok.sum()) == 0:
            continue

        color = cmap(i % 10)
        ax.plot(x[ok], y[ok], marker="o", color=color, label=session.replace("Allen_", ""))

        ok_band = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)
        if int(ok_band.sum()) > 0:
            ax.fill_between(x[ok_band], lo[ok_band], hi[ok_band], color=color, alpha=0.18)

    ax.set_xlabel("Number of aggregated windows", fontsize=9)
    ax.set_ylabel("Split-half subspace overlap", fontsize=9)
    ax.set_title("Aggregation scaling: reliability vs windows", fontsize=10)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(fontsize=8)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_aggregation_scaling_summary(
    session_rows: list[dict],
    path: Path,
) -> None:
    """Bar summaries of full aggregation reliability and n_half per session."""
    if not session_rows:
        return

    sessions = [r.get("session", "") for r in session_rows]
    x = np.arange(len(sessions))

    rel = [r.get("aggregation_reliability_full", float("nan")) for r in session_rows]
    n_half = [r.get("aggregation_n_half", float("nan")) for r in session_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.bar(x, rel, color="seagreen", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("Allen_", "") for s in sessions], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Reliability", fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Full aggregation reliability", fontsize=10)

    ax = axes[1]
    ax.bar(x, n_half, color="slateblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("Allen_", "") for s in sessions], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("n_half (windows)", fontsize=9)
    ax.set_title("Aggregation scale (half-maximum)", fontsize=10)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pilot report
# ---------------------------------------------------------------------------

def write_pilot_report(
    path: Path,
    session: str,
    session_qc: dict,
    session_metrics: dict,
    shared_vs_img: dict,
    reliability: dict,
    sanity_passed: bool,
    sanity_details: str,
) -> None:
    """Write the pilot milestone Markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def _fmt(v):
        if isinstance(v, float) and np.isfinite(v):
            return f"{v:.4f}"
        return str(v)

    lines = [
        f"# Phase 1 pilot report — {session}",
        "",
        "Generated by `run_phase1_fem_covariance.py` (pilot milestone).",
        "",
        "---",
        "",
        "## Session QC",
        "",
        f"- Trials: {session_qc.get('n_trials')}",
        f"- Time bins per trial: {session_qc.get('n_time_bins')}",
        f"- Total units: {session_qc.get('n_units_total')}",
        f"- Primary units: {session_qc.get('n_units_primary')}",
        f"- Inferred bin ms: {_fmt(session_qc.get('bin_ms', float('nan')))}",
        f"- Valid bin fraction: {_fmt(session_qc.get('valid_bin_fraction', float('nan')))}",
        f"- Unique images: {session_qc.get('n_unique_images')}",
        f"- Median repeats per image-time cell: {_fmt(session_qc.get('median_repeats_per_image_time_cell', float('nan')))}",
        f"- Image-time cells ≥2 repeats: {_fmt(session_qc.get('frac_image_time_cells_ge2_repeats', float('nan')))}",
        f"- Median firing rate: {_fmt(session_qc.get('mean_rate_median', float('nan')))} Hz",
        f"- QC recommendation: {session_qc.get('analysis_recommendation')}",
        "",
        "---",
        "",
        "## McFarland estimator (primary estimator)",
        "",
        f"- n image contexts contributing: {session_metrics.get('mc_n_contexts')}",
        f"- Fit OK: {session_metrics.get('mc_fit_ok')}",
        f"- Top-2 variance fraction: {_fmt(session_metrics.get('mc_top2_fraction', float('nan')))}",
        f"- Participation ratio: {_fmt(session_metrics.get('mc_participation_ratio', float('nan')))}",
        f"- Estimator covariance correlation (McFarland vs B_emp): "
        f"{_fmt(session_metrics.get('primary_vs_sensitivity_cov_corr', float('nan')))}",
        "",
        "## Covariance geometry (B_emp sensitivity estimator)",
        "",
        f"- Top-2 variance fraction: {_fmt(session_metrics.get('b_emp_top2_fraction', float('nan')))}",
        f"- Participation ratio: {_fmt(session_metrics.get('b_emp_participation_ratio', float('nan')))}",
        f"- Total variance: {_fmt(session_metrics.get('b_emp_total_variance', float('nan')))}",
        f"- Effective dimensionality: {_fmt(session_metrics.get('b_emp_effective_dim', float('nan')))}",
        f"- n samples used for fit: {session_metrics.get('b_emp_n_samples')}",
        "",
        "### Reliability",
        "",
        f"- Split-half subspace overlap (reliability ceiling): {_fmt(reliability.get('reliability_ceiling', float('nan')))}",
        f"- Split-half B cosine: {_fmt(reliability.get('split_half_b_cosine', float('nan')))}",
        f"- Model basis used: {session_metrics.get('model_alignment_basis')}",
        f"- Model alignment (matched): {_fmt(session_metrics.get('model_alignment', float('nan')))}",
        f"- Model alignment (shuffle): {_fmt(session_metrics.get('model_shuffle_alignment', float('nan')))}",
        f"- Model reliability ceiling: {_fmt(session_metrics.get('model_reliability_ceiling', float('nan')))}",
        f"- Ceiling-normalised model alignment: {_fmt(session_metrics.get('ceiling_normalized_alignment', float('nan')))}",
        f"- Alignment status: {session_metrics.get('alignment_norm_status')}",
        f"- Ceiling-normalised empirical excess (top2 minus eye-shuffle p95): "
        f"{_fmt(session_metrics.get('ceiling_normalized_empirical_excess', float('nan')))}",
        f"- Empirical excess status: {session_metrics.get('empirical_excess_status')}",
        f"- Model alignment status: {session_metrics.get('model_alignment_status')}",
        "",
        "### Controls",
        "",
        f"- Eye shuffle p95 top-2 fraction: {_fmt(session_metrics.get('eye_shuffle_p95_top2', float('nan')))}",
        f"- Image shuffle p95 top-2 fraction: {_fmt(session_metrics.get('image_shuffle_p95_top2', float('nan')))}",
        "",
        "---",
        "",
        "## Shared vs image-specific geometry",
        "",
        f"- Images evaluated (cross-validated): {shared_vs_img.get('n_images_evaluated')}",
        f"- Mean R² shared basis: {_fmt(shared_vs_img.get('mean_r2_shared', float('nan')))}",
        f"- Mean R² image-specific basis: {_fmt(shared_vs_img.get('mean_r2_image_specific', float('nan')))}",
        f"- R² delta (image-specific − shared): {_fmt(shared_vs_img.get('r2_delta_image_vs_shared', float('nan')))}",
        f"- Status: {shared_vs_img.get('model_1_vs_2_status')}",
        "",
        "---",
        "",
        "## Sanity check: empirical bridge reproduction",
        "",
        f"- **Passed: {sanity_passed}**",
        f"- Details: {sanity_details}",
        "",
        "---",
        "",
        "## Pilot verdict",
        "",
    ]

    # Interpret results
    top2 = session_metrics.get("mc_top2_fraction", float("nan"))
    pr = session_metrics.get("mc_participation_ratio", float("nan"))
    eye_p95 = session_metrics.get("eye_shuffle_p95_top2", float("nan"))

    if not np.isfinite(top2):
        verdict = "INCOMPLETE — McFarland fit failed."
    elif top2 > 0.50 or (np.isfinite(pr) and pr <= 4):
        verdict = "LOW-DIMENSIONAL COVARIANCE observed (top2 fraction high or PR ≤ 4)."
    else:
        verdict = "Covariance not strongly low-dimensional at this threshold."

    if np.isfinite(top2) and np.isfinite(eye_p95) and top2 > eye_p95:
        verdict += " Observed top-2 fraction exceeds eye-shuffle p95."
    elif np.isfinite(top2) and np.isfinite(eye_p95):
        verdict += " Observed top-2 fraction does NOT exceed eye-shuffle p95."

    lines.append(verdict)
    lines.append("")
    lines.append("Proceed to batch only after verifying data shapes, residuals, and this sanity check.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
