#!/usr/bin/env python3
"""Publication-style plots for the v11 curvature/amplitude-law analysis."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "outputs" / "covTFTS_v11_remaining_analysis" / "curvature_amplitude_law_all_sessions_ms128"
FULL = "full_linear_tangent"
COMPACT = "compact_k10_tangent"

COLORS = {
    FULL: "#2f5f9f",
    COMPACT: "#7b5ea7",
    "null": "#9a9a9a",
    "eye_step": "#2f7f5f",
    "eye_cloud": "#c44e52",
}
LABELS = {
    FULL: "full tangent",
    COMPACT: "compact k=10",
}
BIN_LABELS = {
    "drift_scale": "drift",
    "intermediate": "intermediate",
    "microsaccade_scale": "micro",
    "larger_offsets": "large",
}


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _sem_ci(values: pd.Series) -> tuple[float, float, float]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, mean, mean
    sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


def _bin_order(metrics: pd.DataFrame) -> list[str]:
    tmp = metrics[["amplitude_bin", "bin_low_arcmin"]].drop_duplicates()
    tmp = tmp.sort_values("bin_low_arcmin")
    return tmp["amplitude_bin"].astype(str).tolist()


def _summary_rows(metrics: pd.DataFrame, cov: pd.DataFrame, sessions: pd.DataFrame, bins: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric_name, df, group_col, value_col in [
        ("pointwise_r2", metrics, "prediction_variant", "pointwise_r2"),
        ("residual_norm_ratio", metrics, "prediction_variant", "median_residual_norm_fraction"),
        ("covariance_capture_k10", cov[(cov["k"].astype(int) == 10)], "source_variant", "capture_fraction"),
        ("unit_shuffle_null_k10", cov[(cov["k"].astype(int) == 10)], "source_variant", "unit_shuffle_null_median"),
        ("random_subspace_null_k10", cov[(cov["k"].astype(int) == 10)], "source_variant", "random_subspace_null_median"),
    ]:
        for source in [FULL, COMPACT]:
            for b in bins:
                g = df[
                    (df["amplitude_bin"].astype(str) == b)
                    & (df[group_col].astype(str) == source)
                    & (df["row_status"].astype(str) == "ok")
                ]
                mean, low, high = _sem_ci(g[value_col])
                rows.append(
                    {
                        "panel_metric": metric_name,
                        "source_variant": source,
                        "source_label": LABELS[source],
                        "amplitude_bin": b,
                        "x_index": bins.index(b),
                        "x_label": BIN_LABELS.get(b, b),
                        "n_sessions": int(g["session"].nunique()) if "session" in g else 0,
                        "mean": mean,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    for key, label in [
        ("empirical_eye_step_arcmin_p50", "eye step p50"),
        ("empirical_eye_step_arcmin_p90", "eye step p90"),
        ("empirical_eye_cloud_radius_arcmin_p50", "eye-cloud radius p50"),
        ("empirical_eye_cloud_radius_arcmin_p90", "eye-cloud radius p90"),
    ]:
        mean, low, high = _mean_ci(sessions[key])
        rows.append(
            {
                "panel_metric": "empirical_eye_anchor_arcmin",
                "source_variant": key,
                "source_label": label,
                "amplitude_bin": "",
                "x_index": float("nan"),
                "x_label": "",
                "n_sessions": int(sessions["session"].nunique()),
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def _x_positions(bins: list[str]) -> np.ndarray:
    return np.arange(len(bins), dtype=float)


def _draw_session_traces(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    bins: list[str],
    source_col: str,
    value_col: str,
    source: str,
    offset: float,
    color: str,
) -> None:
    x = _x_positions(bins) + offset
    for _session, g in df[(df[source_col].astype(str) == source) & (df["row_status"].astype(str) == "ok")].groupby("session"):
        vals = []
        for b in bins:
            row = g[g["amplitude_bin"].astype(str) == b]
            vals.append(float(row[value_col].iloc[0]) if len(row) else np.nan)
        ax.plot(x, vals, color=color, alpha=0.16, linewidth=0.8, zorder=1)
        ax.scatter(x, vals, color=color, alpha=0.20, s=10, edgecolor="none", zorder=2)


def _draw_mean_ci(
    ax: plt.Axes,
    plot_data: pd.DataFrame,
    *,
    bins: list[str],
    metric: str,
    source: str,
    offset: float,
    color: str,
) -> None:
    g = plot_data[(plot_data["panel_metric"] == metric) & (plot_data["source_variant"] == source)]
    x = _x_positions(bins) + offset
    means = [float(g[g["amplitude_bin"] == b]["mean"].iloc[0]) for b in bins]
    lows = [float(g[g["amplitude_bin"] == b]["ci_low"].iloc[0]) for b in bins]
    highs = [float(g[g["amplitude_bin"] == b]["ci_high"].iloc[0]) for b in bins]
    ax.fill_between(x, lows, highs, color=color, alpha=0.16, linewidth=0, zorder=3)
    ax.plot(x, means, color=color, linewidth=2.2, marker="o", markersize=4.5, zorder=4)


def _draw_cov_null_bands(ax: plt.Axes, plot_data: pd.DataFrame, *, bins: list[str]) -> None:
    x = _x_positions(bins)
    for metric, label, alpha in [
        ("unit_shuffle_null_k10", "unit shuffle null", 0.18),
        ("random_subspace_null_k10", "random subspace null", 0.10),
    ]:
        vals = []
        lows = []
        highs = []
        for b in bins:
            rows = plot_data[(plot_data["panel_metric"] == metric) & (plot_data["amplitude_bin"] == b)]
            vals.append(float(rows["mean"].mean()))
            lows.append(float(rows["ci_low"].mean()))
            highs.append(float(rows["ci_high"].mean()))
        ax.fill_between(x, lows, highs, color=COLORS["null"], alpha=alpha, linewidth=0, zorder=0)
        ax.plot(x, vals, color=COLORS["null"], linewidth=1.0, alpha=0.85, linestyle="--", zorder=1)
    ax.text(0.02, 0.06, "gray: null bands", transform=ax.transAxes, fontsize=6.5, color="#666666")


def _draw_eye_anchors(ax: plt.Axes, plot_data: pd.DataFrame, *, bins: list[str]) -> None:
    ax.set_xlim(-0.25, len(bins) - 0.75)
    ax.set_ylim(0, 32)
    ax.set_xticks(_x_positions(bins))
    ax.set_xticklabels([BIN_LABELS.get(b, b) for b in bins], fontsize=7)
    ax.set_ylabel("arcmin", fontsize=8)
    ax.set_title("D. Empirical FEM scale anchors", loc="left", fontweight="bold", fontsize=9)
    for y, color, label, linestyle in [
        (
            float(plot_data[(plot_data["panel_metric"] == "empirical_eye_anchor_arcmin") & (plot_data["source_variant"] == "empirical_eye_step_arcmin_p50")]["mean"].iloc[0]),
            COLORS["eye_step"],
            "step p50",
            "-",
        ),
        (
            float(plot_data[(plot_data["panel_metric"] == "empirical_eye_anchor_arcmin") & (plot_data["source_variant"] == "empirical_eye_step_arcmin_p90")]["mean"].iloc[0]),
            COLORS["eye_step"],
            "step p90",
            "--",
        ),
        (
            float(plot_data[(plot_data["panel_metric"] == "empirical_eye_anchor_arcmin") & (plot_data["source_variant"] == "empirical_eye_cloud_radius_arcmin_p50")]["mean"].iloc[0]),
            COLORS["eye_cloud"],
            "cloud radius p50",
            "-",
        ),
        (
            float(plot_data[(plot_data["panel_metric"] == "empirical_eye_anchor_arcmin") & (plot_data["source_variant"] == "empirical_eye_cloud_radius_arcmin_p90")]["mean"].iloc[0]),
            COLORS["eye_cloud"],
            "cloud radius p90",
            "--",
        ),
    ]:
        ax.axhline(y, color=color, linestyle=linestyle, linewidth=1.3, alpha=0.9)
        ax.text(len(bins) - 0.78, y + 0.45, label, ha="right", va="bottom", fontsize=6.2, color=color)
    controlled = [0.5, 1.5, 3.5, 8.0]
    ax.scatter(_x_positions(bins), controlled, s=34, color="#202124", zorder=4)
    for xi, yi in zip(_x_positions(bins), controlled, strict=True):
        ax.text(xi, yi + 0.55, f"{yi:g}", ha="center", va="bottom", fontsize=6.4, color="#202124")
    ax.text(0.02, 0.96, "black dots: controlled offsets", transform=ax.transAxes, ha="left", va="top", fontsize=6.5)


def _style_panel(ax: plt.Axes, bins: list[str]) -> None:
    ax.set_xticks(_x_positions(bins))
    ax.set_xticklabels([BIN_LABELS.get(b, b) for b in bins], fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_main(input_root: Path, out_dir: Path) -> pd.DataFrame:
    metrics = pd.read_csv(input_root / "curvature_amplitude_law_metrics.csv")
    cov = pd.read_csv(input_root / "curvature_amplitude_law_covariance_capture.csv")
    sessions = pd.read_csv(input_root / "curvature_session_summary.csv")
    bins = _bin_order(metrics)
    plot_data = _summary_rows(metrics, cov, sessions, bins)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_data.to_csv(out_dir / "curvature_amplitude_law_main_plot_data.csv", index=False)

    fig = plt.figure(figsize=(7.2, 5.4), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95])
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    offsets = {FULL: -0.04, COMPACT: 0.04}

    ax = axes[0]
    for source in [FULL, COMPACT]:
        _draw_session_traces(ax, metrics, bins=bins, source_col="prediction_variant", value_col="pointwise_r2", source=source, offset=offsets[source], color=COLORS[source])
        _draw_mean_ci(ax, plot_data, bins=bins, metric="pointwise_r2", source=source, offset=offsets[source], color=COLORS[source])
    ax.axhline(0, color="#202124", linewidth=0.8, linestyle=":", zorder=0)
    ax.set_ylabel("pointwise R2", fontsize=8)
    ax.set_title("A. Pointwise prediction degrades", loc="left", fontweight="bold", fontsize=9)
    _style_panel(ax, bins)

    ax = axes[1]
    for source in [FULL, COMPACT]:
        _draw_session_traces(ax, metrics, bins=bins, source_col="prediction_variant", value_col="median_residual_norm_fraction", source=source, offset=offsets[source], color=COLORS[source])
        _draw_mean_ci(ax, plot_data, bins=bins, metric="residual_norm_ratio", source=source, offset=offsets[source], color=COLORS[source])
    ax.axhline(1, color="#202124", linewidth=0.8, linestyle=":", zorder=0)
    ax.set_ylabel("residual norm / actual", fontsize=8)
    ax.set_title("B. Residual grows with amplitude", loc="left", fontweight="bold", fontsize=9)
    _style_panel(ax, bins)

    ax = axes[2]
    _draw_cov_null_bands(ax, plot_data, bins=bins)
    for source in [FULL, COMPACT]:
        df = cov[cov["k"].astype(int) == 10]
        _draw_session_traces(ax, df, bins=bins, source_col="source_variant", value_col="capture_fraction", source=source, offset=offsets[source], color=COLORS[source])
        _draw_mean_ci(ax, plot_data, bins=bins, metric="covariance_capture_k10", source=source, offset=offsets[source], color=COLORS[source])
    ax.set_ylabel("covariance captured, k=10", fontsize=8)
    ax.set_title("C. Covariance footprint remains stable", loc="left", fontweight="bold", fontsize=9)
    _style_panel(ax, bins)
    ax.set_ylim(0.0, 1.0)

    _draw_eye_anchors(axes[3], plot_data, bins=bins)
    _style_panel(axes[3], bins)

    handles = [
        Line2D([0], [0], color=COLORS[FULL], marker="o", linewidth=2.2, label=LABELS[FULL]),
        Line2D([0], [0], color=COLORS[COMPACT], marker="o", linewidth=2.2, label=LABELS[COMPACT]),
        Patch(facecolor=COLORS["null"], alpha=0.18, label="null bands"),
    ]
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.10, top=0.84, hspace=0.42, wspace=0.30)
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.91))
    fig.suptitle("Local translation tangents are pointwise local but covariance-stable", fontsize=10, y=0.98)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"curvature_amplitude_law_main.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_data


def plot_audit(input_root: Path, out_dir: Path) -> None:
    metrics = pd.read_csv(input_root / "curvature_amplitude_law_metrics.csv")
    cov = pd.read_csv(input_root / "curvature_amplitude_law_covariance_capture.csv")
    sessions = pd.read_csv(input_root / "curvature_session_summary.csv")
    bins = _bin_order(metrics)
    subjects = sessions.set_index("session")["subject"].to_dict()
    sessions_order = sorted(sessions["session"].astype(str).tolist(), key=lambda s: (subjects.get(s, ""), s))
    x = _x_positions(bins)
    fig, axes = plt.subplots(6, 4, figsize=(10, 8.5), sharex=True, sharey=False, constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, session in zip(axes_flat, sessions_order, strict=False):
        for source in [FULL, COMPACT]:
            g = metrics[(metrics["session"] == session) & (metrics["prediction_variant"] == source)]
            vals = [float(g[g["amplitude_bin"] == b]["pointwise_r2"].iloc[0]) for b in bins]
            ax.plot(x, vals, color=COLORS[source], marker="o", markersize=2.4, linewidth=1.0)
        g_cov = cov[(cov["session"] == session) & (cov["source_variant"] == COMPACT) & (cov["k"].astype(int) == 10)]
        cap = [float(g_cov[g_cov["amplitude_bin"] == b]["capture_fraction"].iloc[0]) for b in bins]
        ax2 = ax.twinx()
        ax2.plot(x, cap, color="#202124", linestyle="--", linewidth=0.9, alpha=0.75)
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis="y", labelsize=5, length=2)
        ax.axhline(0, color="#666666", linewidth=0.5, linestyle=":")
        row = sessions[sessions["session"] == session].iloc[0]
        ax.set_title(f"{session}\nn={int(row['n_samples_used'])}, units={int(row['n_common_units'])}", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels([BIN_LABELS.get(b, b) for b in bins], fontsize=5.5)
        ax.tick_params(axis="y", labelsize=5.5)
        ax.grid(axis="y", alpha=0.16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes_flat[len(sessions_order):]:
        ax.axis("off")
    fig.suptitle("Curvature/amplitude law session audit: R2 curves and compact covariance capture", fontsize=10)
    handles = [
        Line2D([0], [0], color=COLORS[FULL], marker="o", linewidth=1.2, label="full R2"),
        Line2D([0], [0], color=COLORS[COMPACT], marker="o", linewidth=1.2, label="compact R2"),
        Line2D([0], [0], color="#202124", linestyle="--", linewidth=1.0, label="compact covariance capture"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.02))
    fig.savefig(out_dir / "curvature_amplitude_law_session_audit.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "curvature_amplitude_law_session_audit.pdf", bbox_inches="tight")
    plt.close(fig)


def write_readme(input_root: Path, out_dir: Path) -> None:
    manifest = {}
    if (input_root / "run_manifest.json").exists():
        manifest = json.loads((input_root / "run_manifest.json").read_text(encoding="utf-8"))
    lines = [
        "# Curvature / Amplitude Law Publication Figures",
        "",
        f"Input root: `{input_root}`",
        f"Sessions completed: {len(manifest.get('sessions_completed', []))}",
        f"Max samples per session: {manifest.get('max_samples', 'NA')}",
        f"Nulls: {manifest.get('n_nulls', 'NA')}",
        "",
        "## Outputs",
        "",
        "- `curvature_amplitude_law_main.png/pdf/svg`: four-panel publication figure.",
        "- `curvature_amplitude_law_main_plot_data.csv`: plotted means/CIs and empirical FEM anchors.",
        "- `curvature_amplitude_law_session_audit.png/pdf`: per-session audit figure.",
        "",
        "Main message: pointwise tangent prediction degrades with displacement amplitude, while covariance capture remains stable and above null.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot publication figures for v11 curvature/amplitude-law outputs.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    input_root = Path(args.input_root)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else input_root / "figures" / "curvature_amplitude_law_main_and_session_audit"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_main(input_root, out_dir)
    plot_audit(input_root, out_dir)
    write_readme(input_root, out_dir)
    print(json.dumps({"status": "ok", "out_dir": str(out_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
