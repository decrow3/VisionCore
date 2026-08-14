#!/usr/bin/env python3
"""Pair tuning-split distributions with corrected Figure-4B path curves."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig.ssi_figure_v2 import compose_ssi_figure_v4_corrected_sf_quartiles as figure


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/fig4_active_sensing/rr100_panel_b_grouping_comparison_v2"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_panel_b_split_distribution_pairs_v3"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GOLD = "#E69F00"
PINK = "#CC79A7"
PURPLE = "#7B61A8"
GRAY = "#A7A9AC"

CONFIG = {
    "sf_quartiles": {
        "title": "Preferred-SF quartiles", "metric": "preferred_sf_cpd",
        "xlabel": "preferred SF (cycles/degree)", "log": True,
        "groups": ("sf_q1", "sf_q2", "sf_q3", "sf_q4"),
        "display_groups": ("sf_q1", "sf_q2", "sf_q3", "sf_q4"),
        "labels": {"sf_q1": "Q1", "sf_q2": "Q2", "sf_q3": "Q3", "sf_q4": "Q4"},
        "colors": {"sf_q1": BLUE, "sf_q2": GREEN, "sf_q3": GOLD, "sf_q4": PINK},
    },
    "sf_halves": {
        "title": "Preferred-SF halves", "metric": "preferred_sf_cpd",
        "xlabel": "preferred SF (cycles/degree)", "log": True,
        "groups": ("low", "high"), "display_groups": ("low", "high"),
        "labels": {"low": "low SF", "high": "high SF"},
        "colors": {"low": BLUE, "high": ORANGE},
    },
    "sf_outer_thirds": {
        "title": "Preferred-SF outer thirds", "metric": "preferred_sf_cpd",
        "xlabel": "preferred SF (cycles/degree)", "log": True,
        "groups": ("bottom", "top"), "display_groups": ("bottom", "excluded", "top"),
        "labels": {"bottom": "bottom SF", "excluded": "middle excluded", "top": "top SF"},
        "colors": {"bottom": BLUE, "excluded": GRAY, "top": ORANGE},
    },
    "tf_halves": {
        "title": "Preferred-TF halves", "metric": "preferred_tf_hz",
        "xlabel": "preferred TF (Hz)", "log": True,
        "groups": ("low", "high"), "display_groups": ("low", "high"),
        "labels": {"low": "low TF", "high": "high TF"},
        "colors": {"low": PURPLE, "high": GREEN},
    },
    "speed_halves": {
        "title": "Preferred-speed halves", "metric": "preferred_speed_dps",
        "xlabel": "preferred retinal speed = TF/SF (degrees/s)", "log": True,
        "groups": ("slow", "fast"), "display_groups": ("slow", "fast"),
        "labels": {"slow": "slow preference", "fast": "fast preference"},
        "colors": {"slow": PURPLE, "fast": GOLD},
    },
    "tf_adjusted_sf_halves": {
        "title": "TF-adjusted preferred-SF halves", "metric": "tf_adjusted_log2_sf",
        "xlabel": "SF residual after log₂(SF) ~ log₂(TF) (octaves)", "log": False,
        "groups": ("low", "high"), "display_groups": ("low", "high"),
        "labels": {"low": "lower SF | TF", "high": "higher SF | TF"},
        "colors": {"low": BLUE, "high": ORANGE},
    },
}


def bins_for(values: np.ndarray, use_log: bool) -> np.ndarray:
    lo, hi = float(np.min(values)), float(np.max(values))
    if use_log:
        return np.geomspace(lo * 0.94, hi * 1.06, 13)
    pad = 0.06 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, 13)


def boundaries(frame: pd.DataFrame, groups: tuple[str, ...], metric: str, use_log: bool) -> list[float]:
    output = []
    for left, right in zip(groups[:-1], groups[1:]):
        left_max = float(frame.loc[frame.group.eq(left), metric].max())
        right_min = float(frame.loc[frame.group.eq(right), metric].min())
        output.append(float(np.sqrt(left_max * right_min)) if use_log else 0.5 * (left_max + right_min))
    return output


def plot_distribution(ax: plt.Axes, frame: pd.DataFrame, cfg: dict[str, object]) -> None:
    metric = str(cfg["metric"]); use_log = bool(cfg["log"])
    values = frame[metric].to_numpy(float)
    bins = bins_for(values, use_log)
    for group in cfg["display_groups"]:
        sub = frame[frame.group.eq(group)]
        ax.hist(
            sub[metric], bins=bins, color=cfg["colors"][group], alpha=0.28,
            edgecolor=cfg["colors"][group], linewidth=1.1,
            label=f"{cfg['labels'][group]} (n={len(sub)})",
        )
        ymax = max(1.0, ax.get_ylim()[1])
        ax.scatter(
            sub[metric], np.full(len(sub), -0.045 * ymax), s=17,
            color=cfg["colors"][group], edgecolor="white", linewidth=0.35,
            clip_on=False, zorder=4,
        )
    for value in boundaries(frame, tuple(cfg["display_groups"]), metric, use_log):
        ax.axvline(value, color="0.25", lw=0.85, ls="--")
    if use_log:
        ax.set_xscale("log", base=2)
        if metric == "preferred_sf_cpd":
            ticks = [1, 1.5, 2, 3, 4, 6]
        elif metric == "preferred_tf_hz":
            ticks = [0.5, 1, 2, 4, 8, 16, 32]
        else:
            ticks = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
        visible = [tick for tick in ticks if bins[0] <= tick <= bins[-1]]
        ax.set_xticks(visible, [f"{tick:g}" for tick in visible])
    ax.set_xlabel(str(cfg["xlabel"]))
    ax.set_ylabel("validated units")
    ax.legend(frameon=False, fontsize=7.1, ncol=2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.92", lw=0.6)
    ax.set_axisbelow(True)


def plot_path(ax: plt.Axes, curves: pd.DataFrame, cfg: dict[str, object], ylim: tuple[float, float]) -> None:
    groups = tuple(cfg["groups"]); colors = cfg["colors"]; labels = cfg["labels"]
    figure.add_segmented_zero_anchor(ax, [colors[group] for group in groups])
    for group in groups:
        for context, filled in (("drift_only", False), ("microsaccade", True)):
            sub = curves[curves.sf_quartile.eq(group) & curves.context.eq(context)].sort_values("bin")
            yerr = np.vstack([sub.delta_percent - sub.ci95_low, sub.ci95_high - sub.delta_percent])
            ax.errorbar(
                figure.path_broken_log(sub.x_median), sub.delta_percent, yerr=yerr,
                color=colors[group], marker="o", mfc=colors[group] if filled else "white",
                mec=colors[group], ms=3.9, lw=1.35, ls="-", capsize=1.5,
                label=f"{labels[group]} (n={int(sub.n_units.iloc[0])})" if context == "drift_only" else None,
            )
    figure.format_broken_path_axis(ax)
    ax.axhline(0, color="0.45", lw=0.7, ls=":")
    ax.set_ylim(*ylim)
    ax.set_xlabel("corrected retinal path length (arcmin)")
    ax.set_ylabel("SSI change (%) vs matched stabilized")
    ax.legend(frameon=False, fontsize=7.1, ncol=2, loc="best")
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    assignments = pd.read_csv(SOURCE / "unit_assignments_all_schemes.csv")
    curves = pd.read_csv(SOURCE / "panel_b_curves_all_schemes.csv")
    lo, hi = float(curves.ci95_low.min()), float(curves.ci95_high.max())
    pad = 0.06 * (hi - lo)
    ylim = (lo - pad, hi + pad)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2, "axes.titlesize": 10,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(6, 2, figsize=(13.2, 22.0), constrained_layout=True)
    for row, (scheme, cfg) in enumerate(CONFIG.items()):
        left = assignments[assignments.scheme.eq(scheme)].copy()
        right = curves[curves.scheme.eq(scheme)].copy()
        plot_distribution(axes[row, 0], left, cfg)
        plot_path(axes[row, 1], right, cfg, ylim)
        axes[row, 0].set_title(f"{chr(65 + 2 * row)}  {cfg['title']}: tuning distribution", loc="left", weight="bold")
        axes[row, 1].set_title(f"{chr(66 + 2 * row)}  Corresponding path-length response", loc="left", weight="bold")
    fig.suptitle(
        "Outcome-blind tuning splits and their corrected Figure-4B path curves\n"
        "left: validated tuning distribution and split · right: identical population SSI estimand",
        fontsize=15, weight="bold",
    )
    fig.text(0.995, 0.002, "open: drift only · filled: scored-window microsaccade", ha="right", fontsize=7.5)
    png = OUT / "rr100_panel_b_split_distribution_pairs.png"
    pdf = OUT / "rr100_panel_b_split_distribution_pairs.pdf"
    fig.savefig(png, dpi=180, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "split_distribution_path_curve_pairs_complete",
        "source": str(SOURCE.resolve()),
        "schemes": list(CONFIG),
        "outputs": {"png": str(png.resolve()), "pdf": str(pdf.resolve())},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
