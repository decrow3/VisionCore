#!/usr/bin/env python3
"""Plot end-to-end Vernier continuous pose-profile diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_ORDER = ("static_center", "scaled_real_0.5", "real_fem")
CONDITION_LABELS = {
    "static_center": "static\ncenter",
    "scaled_real_0.5": "0.5x\nFEM",
    "real_fem": "1x\nFEM",
}
OBSERVER_COLORS = {
    "zero": "#6b7280",
    "neutral": "#8b5cf6",
    "profile": "#2563eb",
    "profile_best": "#93c5fd",
    "known": "#111827",
}
LABEL_COLORS = {"plus": "#2563eb", "minus": "#dc2626"}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 260,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.23,
            "grid.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _conditions_present(df: pd.DataFrame) -> list[str]:
    present = set(df["condition"].astype(str))
    ordered = [condition for condition in CONDITION_ORDER if condition in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _condition_labels(conditions: Iterable[str]) -> list[str]:
    return [CONDITION_LABELS.get(condition, condition.replace("_", "\n")) for condition in conditions]


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")


def _binomial_err(p: np.ndarray, n: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = np.maximum(np.asarray(n, dtype=float), 1.0)
    return 1.96 * np.sqrt(np.clip(p * (1.0 - p), 0.0, None) / n)


def _plot_accuracy(summary: pd.DataFrame, out_dir: Path, ax: plt.Axes | None = None) -> plt.Figure:
    conditions = _conditions_present(summary)
    block = summary.set_index("condition").reindex(conditions)
    observers = [
        ("zero_accuracy", "zero", "static / zero"),
        ("neutral_accuracy", "neutral", "shared neutral"),
        ("profile_accuracy", "profile", "continuous profile"),
        ("known_accuracy", "known", "known trace"),
    ]
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7.1, 3.7), constrained_layout=True)
    else:
        fig = ax.figure
    x = np.arange(len(conditions), dtype=float)
    width = 0.18
    n = block["n"].to_numpy(dtype=float)
    for idx, (column, key, label) in enumerate(observers):
        values = block[column].to_numpy(dtype=float)
        xpos = x + (idx - 1.5) * width
        ax.bar(xpos, values, width=width, color=OBSERVER_COLORS[key], label=label)
        ax.errorbar(xpos, values, yerr=_binomial_err(values, n), color="#111827", lw=0.8, fmt="none", capsize=2)
    if "profile_best_margin_accuracy" in block:
        ax.scatter(
            x,
            block["profile_best_margin_accuracy"].to_numpy(dtype=float),
            s=38,
            facecolors="white",
            edgecolors=OBSERVER_COLORS["profile"],
            linewidths=1.4,
            marker="D",
            label="profile best threshold",
            zorder=5,
        )
    ax.axhline(0.5, color="#9ca3af", lw=1.0, linestyle="--")
    ax.set_title("End-to-end Vernier decoding")
    ax.set_ylabel("accuracy")
    ax.set_xticks(x, _condition_labels(conditions))
    ax.set_ylim(0.35, 1.04)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    if own_fig:
        _save(fig, out_dir, "continuous_profile_accuracy")
        plt.close(fig)
    return fig


def _plot_profile_margins(trials: pd.DataFrame, out_dir: Path, ax: plt.Axes | None = None) -> plt.Figure:
    conditions = _conditions_present(trials)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    else:
        fig = ax.figure
    rng = np.random.default_rng(10)
    for idx, condition in enumerate(conditions):
        cond = trials[trials["condition"].eq(condition)]
        for label, offset in [("minus", -0.08), ("plus", 0.08)]:
            vals = cond.loc[cond["true_label"].eq(label), "profile_margin_plus_minus"].to_numpy(dtype=float)
            jitter = rng.normal(0.0, 0.018, size=vals.size)
            ax.scatter(
                np.full(vals.size, idx + offset) + jitter,
                vals,
                s=13,
                color=LABEL_COLORS[label],
                alpha=0.58,
                edgecolors="none",
                label=label if idx == 0 else None,
            )
            if vals.size:
                ax.plot([idx + offset - 0.06, idx + offset + 0.06], [np.median(vals)] * 2, color="#111827", lw=1.1)
    ax.axhline(0.0, color="#111827", lw=0.9, linestyle="--")
    ax.set_title("Continuous profile margin")
    ax.set_ylabel("score plus - score minus")
    ax.set_xticks(np.arange(len(conditions)), _condition_labels(conditions))
    ax.legend(frameon=False, loc="upper right")
    if own_fig:
        _save(fig, out_dir, "continuous_profile_margin_scatter")
        plt.close(fig)
    return fig


def _plot_margin_by_observer(trials: pd.DataFrame, out_dir: Path) -> None:
    conditions = _conditions_present(trials)
    observers = [
        ("zero_margin_plus_minus", "static / zero"),
        ("profile_margin_plus_minus", "continuous profile"),
        ("known_margin_plus_minus", "known trace"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4), sharey=True, constrained_layout=True)
    rng = np.random.default_rng(14)
    for ax, (column, title) in zip(axes, observers, strict=True):
        for idx, condition in enumerate(conditions):
            cond = trials[trials["condition"].eq(condition)]
            for label, offset in [("minus", -0.08), ("plus", 0.08)]:
                vals = cond.loc[cond["true_label"].eq(label), column].to_numpy(dtype=float)
                jitter = rng.normal(0.0, 0.018, size=vals.size)
                ax.scatter(
                    np.full(vals.size, idx + offset) + jitter,
                    vals,
                    s=10,
                    color=LABEL_COLORS[label],
                    alpha=0.42,
                    edgecolors="none",
                    label=label if idx == 0 and column == observers[0][0] else None,
                )
                if vals.size:
                    ax.plot(
                        [idx + offset - 0.055, idx + offset + 0.055],
                        [np.median(vals)] * 2,
                        color="#111827",
                        lw=1.0,
                    )
        ax.axhline(0.0, color="#111827", lw=0.8, linestyle="--")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(conditions)), _condition_labels(conditions))
    axes[0].set_ylabel("score plus - score minus")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Margin separation by observer")
    _save(fig, out_dir, "continuous_profile_margin_by_observer")
    plt.close(fig)


def _plot_score_gain(trials: pd.DataFrame, out_dir: Path, ax: plt.Axes | None = None) -> plt.Figure:
    conditions = _conditions_present(trials)
    gain_columns = [
        ("neutral_score", "neutral", "shared neutral"),
        ("profile_score", "profile", "continuous profile"),
        ("known_score", "known", "known trace"),
    ]
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6.4, 3.5), constrained_layout=True)
    else:
        fig = ax.figure
    width = 0.22
    x = np.arange(len(conditions), dtype=float)
    for idx, (column, key, label) in enumerate(gain_columns):
        gains = [trials.loc[trials["condition"].eq(condition), column] - trials.loc[trials["condition"].eq(condition), "zero_score"] for condition in conditions]
        positions = x + (idx - 1.0) * width
        bp = ax.boxplot(
            [np.asarray(gain, dtype=float) for gain in gains],
            positions=positions,
            widths=width * 0.74,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111827", "lw": 1.1},
            boxprops={"lw": 0.8, "color": "#111827"},
            whiskerprops={"lw": 0.8, "color": "#111827"},
            capprops={"lw": 0.8, "color": "#111827"},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(OBSERVER_COLORS[key])
            patch.set_alpha(0.72)
        ax.scatter([], [], color=OBSERVER_COLORS[key], label=label)
    ax.axhline(0.0, color="#111827", lw=0.9, linestyle="--")
    ax.set_title("Score gain over static / zero")
    ax.set_ylabel("true-vs-other score gain")
    ax.set_xticks(x, _condition_labels(conditions))
    ax.legend(frameon=False, loc="upper right")
    if own_fig:
        _save(fig, out_dir, "continuous_profile_score_gain")
        plt.close(fig)
    return fig


def _zero_pose_rmse_by_condition(source_dir: Path, conditions: list[str], n_timebins: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for condition in conditions:
        paths = sorted((source_dir / "cache").glob(f"rates_{condition}_fd*arcmin.npz"))
        if not paths:
            continue
        with np.load(paths[0], allow_pickle=True) as npz:
            poses = np.asarray(npz["poses"], dtype=float)[:, :n_timebins] * 60.0
        out[condition] = np.sqrt(np.mean(np.sum(poses * poses, axis=2), axis=1))
    return out


def _plot_pose_rmse(
    trials: pd.DataFrame,
    out_dir: Path,
    source_dir: Path | None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    conditions = _conditions_present(trials)
    n_timebins = int(trials["n_timebins"].max())
    zero_rmse = _zero_pose_rmse_by_condition(source_dir, conditions, n_timebins) if source_dir else {}
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6.2, 3.5), constrained_layout=True)
    else:
        fig = ax.figure
    width = 0.24
    x = np.arange(len(conditions), dtype=float)
    series = [
        ("neutral_pose_rmse_arcmin_true", "neutral", "shared neutral"),
        ("profile_pose_rmse_arcmin_true", "profile", "continuous profile"),
    ]
    for idx, (column, key, label) in enumerate(series):
        vals = [trials.loc[trials["condition"].eq(condition), column].to_numpy(dtype=float) for condition in conditions]
        positions = x + (idx - 0.5) * width
        bp = ax.boxplot(
            vals,
            positions=positions,
            widths=width * 0.72,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111827", "lw": 1.1},
            boxprops={"lw": 0.8, "color": "#111827"},
            whiskerprops={"lw": 0.8, "color": "#111827"},
            capprops={"lw": 0.8, "color": "#111827"},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(OBSERVER_COLORS[key])
            patch.set_alpha(0.72)
        ax.scatter([], [], color=OBSERVER_COLORS[key], label=label)
    if zero_rmse:
        medians = [float(np.median(zero_rmse[condition])) if condition in zero_rmse else np.nan for condition in conditions]
        ax.plot(x, medians, color="#111827", marker="o", lw=1.2, linestyle=":", label="zero-pose baseline")
    ax.set_title("Continuous trace recovery")
    ax.set_ylabel("trajectory RMSE (arcmin)")
    ax.set_xticks(x, _condition_labels(conditions))
    ax.legend(frameon=False, loc="upper left")
    if own_fig:
        _save(fig, out_dir, "continuous_profile_pose_rmse")
        plt.close(fig)
    return fig


def _plot_sheet(trials: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, source_dir: Path | None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)
    _plot_accuracy(summary, out_dir, axes[0, 0])
    _plot_profile_margins(trials, out_dir, axes[0, 1])
    _plot_score_gain(trials, out_dir, axes[1, 0])
    _plot_pose_rmse(trials, out_dir, source_dir, axes[1, 1])
    fig.suptitle("Vernier continuous hidden-pose diagnostic (24 bins, 64 units)", fontsize=12)
    _save(fig, out_dir, "continuous_profile_diagnostic_sheet")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "figures"
    trials = pd.read_csv(run_dir / "end_to_end_pose_profile_trials.csv")
    summary = pd.read_csv(run_dir / "end_to_end_pose_profile_summary.csv")
    _set_style()
    _plot_accuracy(summary, out_dir)
    _plot_profile_margins(trials, out_dir)
    _plot_margin_by_observer(trials, out_dir)
    _plot_score_gain(trials, out_dir)
    _plot_pose_rmse(trials, out_dir, args.source_dir)
    _plot_sheet(trials, summary, out_dir, args.source_dir)
    print(f"Wrote end-to-end pose-profile figures to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
