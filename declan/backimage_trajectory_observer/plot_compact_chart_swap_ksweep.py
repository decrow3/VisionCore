"""Plot compact chart-swap k-sweep diagnostics.

The input is the output of ``analyze_compact_chart_swap.py``.  The figure is
meant to answer the compact-routing question directly: does the correct
image-conditioned chart beat wrong image charts and a universal/global chart,
and is that advantage specific to the compact basis?
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKIMAGE_BASE = REPO_ROOT / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
DEFAULT_MATCHED_STATIC_DIR = (
    BACKIMAGE_BASE / "backimage_compact_chart_swap_matched_static_n64_ksweep_foldstatic_v2"
)
DEFAULT_HARD_NEGATIVE_DIR = (
    BACKIMAGE_BASE / "backimage_compact_chart_swap_hard_negative_n128_ksweep_foldstatic_v2"
)
DEFAULT_OUTPUT_DIR = BACKIMAGE_BASE / "backimage_compact_chart_swap_ksweep_figures_v1"

METRIC = "mean_chart_true_score_lhs_minus_rhs"
CI_LOW = f"{METRIC}_ci_low"
CI_HIGH = f"{METRIC}_ci_high"
P_VALUE = f"{METRIC}_permutation_p_two_sided"

CONTRAST_ORDER = ["wrong_chart_roll", "wrong_chart_pool", "global_chart"]
CONTRAST_LABELS = {
    "wrong_chart_roll": "correct - wrong roll",
    "wrong_chart_pool": "correct - wrong pool",
    "global_chart": "correct - global",
}
BASIS_ORDER = ["compact", "static_pc_k", "random_k", "unit_shuffle_compact", "gain_axis"]
BASIS_LABELS = {
    "compact": "compact",
    "static_pc_k": "static PCs",
    "random_k": "random",
    "unit_shuffle_compact": "unit shuffle",
    "gain_axis": "gain",
}
BASIS_COLORS = {
    "compact": "#1b9e77",
    "static_pc_k": "#7570b3",
    "random_k": "#6f7a83",
    "unit_shuffle_compact": "#d95f02",
    "gain_axis": "#1f78b4",
}
BASIS_MARKERS = {
    "compact": "o",
    "static_pc_k": "s",
    "random_k": "D",
    "unit_shuffle_compact": "^",
    "gain_axis": "X",
}
DATASET_LABELS = {
    "matched_static": "matched-static",
    "hard_negative": "hard-negative",
}
CANDIDATE_LABELS = {
    "matched_static_response": "matched-static",
    "hard_negative_structure": "hard-negative",
}
PRIOR_LABELS = {
    "axis_edge_orthogonal": "orthogonal",
    "axis_edge_parallel": "parallel",
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0.0, color="#bac3ca", linewidth=0.8, zorder=0)
    ax.grid(axis="y", color="#e7ebee", linewidth=0.7, zorder=0)


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _format_scale(value: object) -> str:
    num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(num):
        return str(value)
    return f"{float(num):g}x"


def _condition_label(row: pd.Series) -> str:
    candidate = CANDIDATE_LABELS.get(str(row["candidate_set_mode"]), str(row["candidate_set_mode"]))
    prior = PRIOR_LABELS.get(str(row["prior_family"]), str(row["prior_family"]))
    return f"{candidate}; {prior}; {_format_scale(row['motion_scale'])}"


def _load_contrasts(path: Path, dataset: str) -> pd.DataFrame:
    csv_path = path / "compact_chart_swap_contrasts.csv"
    df = pd.read_csv(csv_path)
    required = {
        "candidate_set_mode",
        "prior_family",
        "motion_scale",
        "basis_type",
        "effective_k_dim",
        "lhs_chart_family",
        "rhs_chart_family",
        METRIC,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    work = df[
        (df["lhs_chart_family"] == "correct_chart")
        & (df["rhs_chart_family"].isin(CONTRAST_ORDER))
        & (df["basis_type"].isin(BASIS_ORDER))
    ].copy()
    work["source_dir"] = str(path)
    work["dataset"] = str(dataset)
    work["dataset_label"] = work["dataset"].map(DATASET_LABELS).fillna(work["dataset"])
    work["contrast_label"] = work["rhs_chart_family"].map(CONTRAST_LABELS).fillna(work["rhs_chart_family"])
    work["basis_label"] = work["basis_type"].map(BASIS_LABELS).fillna(work["basis_type"])
    work["condition_label"] = work.apply(_condition_label, axis=1)
    for col in ["motion_scale", "effective_k_dim", METRIC, CI_LOW, CI_HIGH, P_VALUE]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    return work


def load_all(matched_static_dir: Path, hard_negative_dir: Path) -> pd.DataFrame:
    frames = [
        _load_contrasts(matched_static_dir, "matched_static"),
        _load_contrasts(hard_negative_dir, "hard_negative"),
    ]
    out = pd.concat(frames, ignore_index=True)
    out["dataset"] = pd.Categorical(out["dataset"], ["matched_static", "hard_negative"], ordered=True)
    out["rhs_chart_family"] = pd.Categorical(out["rhs_chart_family"], CONTRAST_ORDER, ordered=True)
    out["basis_type"] = pd.Categorical(out["basis_type"], BASIS_ORDER, ordered=True)
    out = out.sort_values(
        [
            "dataset",
            "candidate_set_mode",
            "prior_family",
            "motion_scale",
            "rhs_chart_family",
            "basis_type",
            "effective_k_dim",
        ]
    ).reset_index(drop=True)
    return out


def _plot_lines(ax: plt.Axes, data: pd.DataFrame, *, show_label: bool = False) -> None:
    for basis in BASIS_ORDER:
        sub = data[data["basis_type"] == basis].sort_values("effective_k_dim")
        if sub.empty:
            continue
        x = sub["effective_k_dim"].to_numpy(dtype=float)
        y = sub[METRIC].to_numpy(dtype=float)
        label = BASIS_LABELS[basis] if show_label else None
        linewidth = 2.1 if basis == "compact" else 1.25
        alpha = 1.0 if basis in {"compact", "static_pc_k", "random_k"} else 0.75
        ax.plot(
            x,
            y,
            color=BASIS_COLORS[basis],
            marker=BASIS_MARKERS[basis],
            markersize=4.2,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
        )
        if P_VALUE in sub.columns:
            sig = sub[np.isfinite(sub[P_VALUE]) & (sub[P_VALUE] < 0.05)]
            if not sig.empty:
                ax.scatter(
                    sig["effective_k_dim"],
                    sig[METRIC],
                    s=46,
                    facecolors="none",
                    edgecolors=BASIS_COLORS[basis],
                    linewidths=1.1,
                    alpha=alpha,
                    zorder=4,
                )


def _aggregate_for_summary(data: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "dataset_label", "rhs_chart_family", "contrast_label", "basis_type", "basis_label", "effective_k_dim"]
    summary = (
        data.groupby(group_cols, observed=True)
        .agg(
            mean_contrast=(METRIC, "mean"),
            min_contrast=(METRIC, "min"),
            max_contrast=(METRIC, "max"),
            n_conditions=(METRIC, "size"),
            n_positive=(METRIC, lambda s: int(np.sum(np.asarray(s, dtype=float) > 0.0))),
        )
        .reset_index()
    )
    return summary


def plot_summary(data: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    summary = _aggregate_for_summary(data)
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.7), sharex=True, constrained_layout=True)

    for row_idx, dataset in enumerate(["matched_static", "hard_negative"]):
        for col_idx, rhs in enumerate(CONTRAST_ORDER):
            ax = axes[row_idx, col_idx]
            _clean_axis(ax)
            sub = summary[(summary["dataset"] == dataset) & (summary["rhs_chart_family"] == rhs)]
            for basis in BASIS_ORDER:
                bsub = sub[sub["basis_type"] == basis].sort_values("effective_k_dim")
                if bsub.empty:
                    continue
                x = bsub["effective_k_dim"].to_numpy(dtype=float)
                y = bsub["mean_contrast"].to_numpy(dtype=float)
                lo = bsub["min_contrast"].to_numpy(dtype=float)
                hi = bsub["max_contrast"].to_numpy(dtype=float)
                ax.fill_between(x, lo, hi, color=BASIS_COLORS[basis], alpha=0.08, linewidth=0)
                ax.plot(
                    x,
                    y,
                    color=BASIS_COLORS[basis],
                    marker=BASIS_MARKERS[basis],
                    markersize=4.3,
                    linewidth=2.2 if basis == "compact" else 1.3,
                    alpha=1.0 if basis in {"compact", "static_pc_k", "random_k"} else 0.75,
                    label=BASIS_LABELS[basis] if (row_idx == 0 and col_idx == 0) else None,
                )
            if row_idx == 0:
                ax.set_title(CONTRAST_LABELS[rhs])
            if col_idx == 0:
                ax.set_ylabel(f"{DATASET_LABELS[dataset]}\nmean score contrast")
            if row_idx == 1:
                ax.set_xlabel("projection dimension k")
            ax.set_xticks([1, 2, 5, 10, 20, 30])
            ax.set_xscale("log")
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Correct image-conditioned chart advantage across projection dimension", y=1.08, fontsize=11)
    _save(fig, out_dir, "compact_chart_swap_ksweep_summary")
    return summary


def plot_by_condition(data: pd.DataFrame, out_dir: Path) -> None:
    condition_order = (
        data[["dataset", "candidate_set_mode", "prior_family", "motion_scale", "condition_label"]]
        .drop_duplicates()
        .sort_values(["dataset", "candidate_set_mode", "prior_family", "motion_scale"])
    )
    conditions = condition_order["condition_label"].tolist()
    n_rows = len(conditions)
    fig_height = max(2.0, 1.55 * n_rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(10.2, fig_height), sharex=True, constrained_layout=True)
    if n_rows == 1:
        axes = np.asarray([axes])

    for row_idx, condition in enumerate(conditions):
        for col_idx, rhs in enumerate(CONTRAST_ORDER):
            ax = axes[row_idx, col_idx]
            _clean_axis(ax)
            sub = data[(data["condition_label"] == condition) & (data["rhs_chart_family"] == rhs)]
            _plot_lines(ax, sub, show_label=(row_idx == 0 and col_idx == 0))
            if row_idx == 0:
                ax.set_title(CONTRAST_LABELS[rhs])
            if col_idx == 0:
                ax.set_ylabel("score contrast")
                ax.text(
                    0.02,
                    0.95,
                    condition,
                    ha="left",
                    va="top",
                    fontsize=7.4,
                    transform=ax.transAxes,
                    bbox={
                        "boxstyle": "round,pad=0.18,rounding_size=0.06",
                        "facecolor": "white",
                        "edgecolor": "#d6dde2",
                        "linewidth": 0.6,
                        "alpha": 0.92,
                    },
                )
            if row_idx == n_rows - 1:
                ax.set_xlabel("projection dimension k")
            ax.set_xticks([1, 2, 5, 10, 20, 30])
            ax.set_xscale("log")
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Correct chart contrasts by task condition", y=1.04, fontsize=11)
    _save(fig, out_dir, "compact_chart_swap_ksweep_by_condition")


def _best_k30_rows(data: pd.DataFrame) -> pd.DataFrame:
    rows = data[data["effective_k_dim"] == 30].copy()
    rows = rows[rows["rhs_chart_family"].isin(CONTRAST_ORDER)]
    keep = [
        "dataset_label",
        "condition_label",
        "rhs_chart_family",
        "contrast_label",
        "basis_type",
        "basis_label",
        "effective_k_dim",
        METRIC,
        CI_LOW,
        CI_HIGH,
        P_VALUE,
    ]
    keep = [col for col in keep if col in rows.columns]
    return rows[keep].sort_values(["dataset_label", "condition_label", "rhs_chart_family", "basis_type"])


def write_report(data: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    compact = summary[summary["basis_type"] == "compact"].copy()
    compact_k30 = compact[compact["effective_k_dim"] == 30].copy()
    basis_k30 = summary[summary["effective_k_dim"] == 30].copy()

    lines = [
        "# Compact Chart-Swap K-Sweep Figures",
        "",
        "Positive values mean the correct image-conditioned chart gives the true image a higher score than the comparison chart.",
        "The shaded bands in the summary figure show the min-to-max range across task conditions within each dataset.",
        "",
        "## Files",
        "",
        "- `compact_chart_swap_ksweep_summary.png/pdf`: condition-mean curves by dataset.",
        "- `compact_chart_swap_ksweep_by_condition.png/pdf`: one row per task condition.",
        "- `compact_chart_swap_ksweep_plot_values.csv`: filtered source rows used for plotting.",
        "- `compact_chart_swap_ksweep_summary_values.csv`: condition-mean summary used for the main figure.",
        "",
        "## Compact k=30 Summary",
        "",
        "```text",
    ]
    for _, row in compact_k30.sort_values(["dataset", "rhs_chart_family"]).iterrows():
        lines.append(
            f"{row['dataset_label']:>14s} {row['contrast_label']:<22s} "
            f"mean={row['mean_contrast']:+.3f} range=[{row['min_contrast']:+.3f}, {row['max_contrast']:+.3f}] "
            f"positive={int(row['n_positive'])}/{int(row['n_conditions'])}"
        )
    lines.extend(
        [
            "```",
            "",
            "## k=30 Basis Ordering",
            "",
            "```text",
        ]
    )
    for (dataset, rhs), sub in basis_k30.groupby(["dataset_label", "contrast_label"], observed=True):
        ranked = sub.sort_values("mean_contrast", ascending=False)
        parts = [f"{r['basis_label']} {r['mean_contrast']:+.2f}" for _, r in ranked.iterrows()]
        lines.append(f"{dataset}; {rhs}: " + ", ".join(parts))
    lines.extend(["```", ""])
    (out_dir / "compact_chart_swap_ksweep_figure_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-static-dir", type=Path, default=DEFAULT_MATCHED_STATIC_DIR)
    parser.add_argument("--hard-negative-dir", type=Path, default=DEFAULT_HARD_NEGATIVE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    _configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_all(args.matched_static_dir, args.hard_negative_dir)
    data.to_csv(args.output_dir / "compact_chart_swap_ksweep_plot_values.csv", index=False)
    summary = plot_summary(data, args.output_dir)
    summary.to_csv(args.output_dir / "compact_chart_swap_ksweep_summary_values.csv", index=False)
    plot_by_condition(data, args.output_dir)
    _best_k30_rows(data).to_csv(args.output_dir / "compact_chart_swap_ksweep_k30_detail_values.csv", index=False)
    write_report(data, summary, args.output_dir)
    _write_json(
        args.output_dir / "compact_chart_swap_ksweep_figure_metadata.json",
        {
            "matched_static_dir": args.matched_static_dir,
            "hard_negative_dir": args.hard_negative_dir,
            "output_dir": args.output_dir,
            "metric": METRIC,
            "n_plot_rows": int(data.shape[0]),
            "n_summary_rows": int(summary.shape[0]),
            "contrast_order": CONTRAST_ORDER,
            "basis_order": BASIS_ORDER,
        },
    )
    print(f"[compact-chart-swap-plot] wrote figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
