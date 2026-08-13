#!/usr/bin/env python3
"""Scorecard for the first-order SF/TF power-shift explanation.

This reads the saved SF/TF proxy summaries and asks the deliberately simple
question: how much of the normal-motion SSI change is explained by image SF
power plus TF landing preference?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / (
    "outputs/active_sensing_movie_information/temporal_remapping/"
    "backimage_rr100_retiming_medium_figure4pool_n16_t32_fullgrid_cuda0_v1"
)
DEFAULT_EXPLANATION_DIR = DEFAULT_RUN_DIR / "sftf_power_explanation_normal_first"
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "map_first_power_shift_checkpoint_05_linear_power_scorecard_v1"

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "vermillion": "#D55E00",
    "grey": "#777777",
}
GROUP_ORDER = ["low_sf", "middle_sf", "high_sf"]
GROUP_LABELS = {"low_sf": "Low SF", "middle_sf": "Middle SF", "high_sf": "High SF"}
GROUP_COLORS = {"low_sf": OKABE_ITO["blue"], "middle_sf": OKABE_ITO["green"], "high_sf": OKABE_ITO["orange"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--explanation-dir", type=Path, default=DEFAULT_EXPLANATION_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def standardize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    sd = float(np.nanstd(arr))
    if not np.isfinite(sd) or sd <= 0.0:
        return np.zeros_like(arr)
    return (arr - float(np.nanmean(arr))) / sd


def fit_ols(table: pd.DataFrame, predictors: list[str]) -> tuple[np.ndarray, float]:
    work = table[["unit_ssi_delta_absolute", *predictors]].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(work.to_numpy(dtype=float)).all(axis=1)
    work = work.loc[mask].copy()
    y = work["unit_ssi_delta_absolute"].to_numpy(dtype=float)
    x = np.column_stack([standardize(work[col].to_numpy(dtype=float)) for col in predictors])
    design = np.column_stack([np.ones(y.size), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    full_pred = np.full(table.shape[0], np.nan, dtype=float)
    full_pred[np.flatnonzero(mask)] = pred
    return full_pred, r2


def read_unit_metadata(run_dir: Path) -> pd.DataFrame:
    cols = ["unit_index", "unit_label", "sf_group", "preferred_sf_cpd"]
    meta = pd.read_csv(run_dir / "retiming_unit_observations.csv", usecols=cols)
    return meta.drop_duplicates("unit_index").sort_values("unit_index").reset_index(drop=True)


def scale_summary(model_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    exact_specs = [
        ("Raw unit x movie samples", "normal_vs_static_unit_observations", "sf_power_plus_tf_match"),
        ("Within-unit samples", "normal_vs_static_within_unit", "sf_power_plus_tf_match"),
        ("Movie means", "normal_vs_static_movie_means", "sf_power_plus_tf_match"),
        ("Unit means", "normal_vs_static_unit_means", "sf_power_plus_tf_match"),
    ]
    for label, scale, model in exact_specs:
        row = model_summary[(model_summary["analysis_scale"].eq(scale)) & (model_summary["model"].eq(model))].iloc[0]
        rows.append(
            {
                "question": "normal_vs_stabilized",
                "view": label,
                "analysis_scale": scale,
                "model": model,
                "n_rows": int(row["n_rows"]),
                "r2": float(row["r2"]),
                "percent_variance_explained": 100.0 * float(row["r2"]),
            }
        )
    row = model_summary[
        (model_summary["analysis_scale"].eq("condition_means"))
        & (model_summary["model"].eq("sf_power_x_tf_match"))
    ].iloc[0]
    rows.append(
        {
            "question": "retiming_condition_grid_context",
            "view": "Broad retiming condition means",
            "analysis_scale": "condition_means",
            "model": "sf_power_x_tf_match",
            "n_rows": int(row["n_rows"]),
            "r2": float(row["r2"]),
            "percent_variance_explained": 100.0 * float(row["r2"]),
        }
    )
    return pd.DataFrame(rows)


def group_summary(unit_means: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name in GROUP_ORDER:
        group = unit_means[unit_means["sf_group"].eq(group_name)]
        y = group["unit_ssi_delta_absolute"].to_numpy(dtype=float)
        pred = group["linear_power_predicted_delta"].to_numpy(dtype=float)
        n = int(np.isfinite(y).sum())
        rows.append(
            {
                "sf_group": group_name,
                "sf_group_label": GROUP_LABELS[group_name],
                "n_units": n,
                "observed_mean_delta": float(np.nanmean(y)),
                "observed_sem_delta": float(np.nanstd(y, ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
                "predicted_mean_delta": float(np.nanmean(pred)),
                "predicted_sem_delta": float(np.nanstd(pred, ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
                "mean_residual_delta": float(np.nanmean(y - pred)),
            }
        )
    return pd.DataFrame(rows)


def scatter_with_identity(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    finite = np.isfinite(x) & np.isfinite(y)
    lo = float(np.nanmin([np.nanmin(x[finite]), np.nanmin(y[finite])]))
    hi = float(np.nanmax([np.nanmax(x[finite]), np.nanmax(y[finite])]))
    pad = 0.06 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#333333", lw=1.0, alpha=0.8)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)


def plot_scorecard(
    scale_table: pd.DataFrame,
    unit_means: pd.DataFrame,
    unit_r2: float,
    group_table: pd.DataFrame,
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.3))
    ax = axes[0, 0]
    primary = scale_table[scale_table["question"].eq("normal_vs_stabilized")].copy()
    context = scale_table[scale_table["question"].eq("retiming_condition_grid_context")].iloc[0]
    primary_labels = ["Raw\nexamples", "Within\nunit", "Movie\naverage", "Unit\naverage"]
    bar_colors = [OKABE_ITO["blue"], OKABE_ITO["sky"], OKABE_ITO["green"], OKABE_ITO["orange"]]
    x = np.arange(primary.shape[0])
    values = primary["percent_variance_explained"].to_numpy(dtype=float)
    ax.bar(x, values, color=bar_colors, alpha=0.9)
    for xx, value in zip(x, values, strict=True):
        ax.text(xx, value + 0.35, f"{value:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(primary_labels, fontsize=10)
    ax.set_ylabel("variance explained (%)")
    ax.set_title("Primary test: normal motion vs stabilized")
    ax.set_ylim(0.0, max(14.0, float(np.nanmax(values)) + 2.0))
    ax.grid(True, axis="y", color="#e6e6e6", lw=0.7)
    ax.text(
        0.02,
        0.93,
        "Main readout:\nsmall at example level,\n~10% after averaging.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.95},
    )
    ax.text(
        0.98,
        0.93,
        f"Context only:\nretiming-grid\ncondition means = {context['percent_variance_explained']:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f3f3f3", "edgecolor": "#aaaaaa", "alpha": 0.95},
    )

    ax = axes[0, 1]
    for group_name in GROUP_ORDER:
        group = unit_means[unit_means["sf_group"].eq(group_name)]
        ax.scatter(
            group["linear_power_predicted_delta"],
            group["unit_ssi_delta_absolute"],
            s=42,
            alpha=0.82,
            color=GROUP_COLORS[group_name],
            edgecolor="white",
            linewidth=0.35,
            label=f"{GROUP_LABELS[group_name]} (n={group.shape[0]})",
        )
    scatter_with_identity(
        ax,
        unit_means["linear_power_predicted_delta"].to_numpy(dtype=float),
        unit_means["unit_ssi_delta_absolute"].to_numpy(dtype=float),
    )
    ax.set_xlabel("predicted SSI change")
    ax.set_ylabel("observed SSI change")
    ax.set_title(f"Unit means: broad spread remains unexplained (R2={unit_r2:.3f})")
    ax.grid(True, color="#e6e6e6", lw=0.7)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1, 0]
    residual = unit_means.copy()
    residual["residual_delta"] = residual["unit_ssi_delta_absolute"] - residual["linear_power_predicted_delta"]
    rng = np.random.default_rng(0)
    for idx, group_name in enumerate(GROUP_ORDER):
        group = residual[residual["sf_group"].eq(group_name)]
        jitter = rng.uniform(-0.12, 0.12, size=group.shape[0])
        ax.scatter(
            np.full(group.shape[0], idx) + jitter,
            group["residual_delta"],
            s=34,
            alpha=0.78,
            color=GROUP_COLORS[group_name],
            edgecolor="white",
            linewidth=0.35,
        )
        q1, med, q3 = np.nanpercentile(group["residual_delta"], [25, 50, 75])
        ax.plot([idx - 0.18, idx + 0.18], [med, med], color="#222222", lw=1.5)
        ax.plot([idx, idx], [q1, q3], color="#222222", lw=2.2)
    ax.axhline(0.0, color="#555555", lw=0.9)
    ax.set_xticks(np.arange(len(GROUP_ORDER)))
    ax.set_xticklabels([GROUP_LABELS[name] for name in GROUP_ORDER])
    ax.set_ylabel("observed - predicted SSI change")
    ax.set_title("Residual unit effects after linear power proxy")
    ax.grid(True, axis="y", color="#e6e6e6", lw=0.7)

    ax = axes[1, 1]
    gx = np.arange(group_table.shape[0])
    ax.errorbar(
        gx - 0.08,
        group_table["observed_mean_delta"],
        yerr=group_table["observed_sem_delta"],
        fmt="o",
        ms=8,
        color="#222222",
        ecolor="#222222",
        capsize=3,
        label="observed",
    )
    ax.errorbar(
        gx + 0.08,
        group_table["predicted_mean_delta"],
        yerr=group_table["predicted_sem_delta"],
        fmt="s",
        ms=7,
        color=OKABE_ITO["vermillion"],
        ecolor=OKABE_ITO["vermillion"],
        capsize=3,
        label="linear power proxy",
    )
    ax.axhline(0.0, color="#555555", lw=0.8)
    ax.set_xticks(gx)
    ax.set_xticklabels(group_table["sf_group_label"])
    ax.set_ylabel("normal - stabilized SSI change")
    ax.set_title("SF group means: proxy captures rough scale")
    ax.grid(True, axis="y", color="#e6e6e6", lw=0.7)
    ax.legend(frameon=False, fontsize=9)
    for _, row in group_table.iterrows():
        idx = GROUP_ORDER.index(row["sf_group"])
        residual_value = row["mean_residual_delta"]
        ax.text(
            idx,
            max(row["observed_mean_delta"], row["predicted_mean_delta"]) + 0.0025,
            f"resid {residual_value:+.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )

    fig.suptitle("Linear power-shift scorecard", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = out_dir / "checkpoint_05_linear_power_scorecard.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    explanation_dir = Path(args.explanation_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_summary = pd.read_csv(explanation_dir / "sftf_power_explanation_model_summary.csv")
    unit_means = pd.read_csv(explanation_dir / "sftf_power_explanation_normal_vs_static_unit_means.csv")
    movie_means = pd.read_csv(explanation_dir / "sftf_power_explanation_normal_vs_static_movie_means.csv")
    unit_meta = read_unit_metadata(run_dir)

    unit_means = unit_means.merge(unit_meta, on="unit_index", how="left")
    unit_means["linear_power_predicted_delta"], unit_r2 = fit_ols(
        unit_means, ["unit_sf_power_abs", "tf_match_fixed"]
    )
    movie_means["linear_power_predicted_delta"], movie_r2 = fit_ols(
        movie_means, ["unit_sf_power_abs", "tf_match_fixed"]
    )

    scale_table = scale_summary(model_summary)
    group_table = group_summary(unit_means)

    scale_table.to_csv(out_dir / "checkpoint_05_linear_power_scale_summary.csv", index=False)
    unit_means.to_csv(out_dir / "checkpoint_05_normal_vs_static_unit_means_with_groups.csv", index=False)
    movie_means.to_csv(out_dir / "checkpoint_05_normal_vs_static_movie_means_with_predictions.csv", index=False)
    group_table.to_csv(out_dir / "checkpoint_05_linear_power_unit_group_summary.csv", index=False)
    figure_path = plot_scorecard(scale_table, unit_means, unit_r2, group_table, out_dir)
    print(f"wrote scorecard to {figure_path}")
    print(scale_table.to_string(index=False))
    print(group_table.to_string(index=False))


if __name__ == "__main__":
    main()
