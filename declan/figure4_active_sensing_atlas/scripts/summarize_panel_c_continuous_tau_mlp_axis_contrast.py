"""Summarize along/across contrasts for the continuous-tau MLP decoder."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "declan"
    / "figure4_active_sensing_atlas"
    / "figures"
    / "panel_C"
    / "diagnostics"
    / "continuous_tau_mlp_feature_decoder_residual"
)
AXIS_FAMILIES = ("axis_edge_parallel", "axis_edge_orthogonal")


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    if vals.size == 1 or int(n_bootstrap) <= 0:
        value = float(np.mean(vals))
        return value, value, value
    draws = rng.choice(vals, size=(int(n_bootstrap), vals.size), replace=True).mean(axis=1)
    return float(np.mean(vals)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _axis_summary(trials: pd.DataFrame) -> pd.DataFrame:
    axis = trials[trials["prior_family"].isin(AXIS_FAMILIES)].copy()
    group_cols = ["input_mode", "decoder_mode", "prior_scale", "prior_family"]
    summary = (
        axis.groupby(group_cols, as_index=False)
        .agg(
            n=("feature_cosine", "size"),
            mean_feature_cosine=("feature_cosine", "mean"),
            median_feature_cosine=("feature_cosine", "median"),
            mean_feature_mse=("feature_mse", "mean"),
            median_feature_rmse=("feature_rmse", "median"),
        )
        .sort_values(["input_mode", "prior_scale", "prior_family"])
    )
    overall = (
        axis.groupby(["input_mode", "decoder_mode", "prior_family"], as_index=False)
        .agg(
            n=("feature_cosine", "size"),
            mean_feature_cosine=("feature_cosine", "mean"),
            median_feature_cosine=("feature_cosine", "median"),
            mean_feature_mse=("feature_mse", "mean"),
            median_feature_rmse=("feature_rmse", "median"),
        )
        .sort_values(["input_mode", "prior_family"])
    )
    overall["prior_scale"] = "all"
    return pd.concat([summary, overall[summary.columns]], ignore_index=True)


def _axis_contrasts(trials: pd.DataFrame, *, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    metrics = {
        "feature_cosine": {
            "positive_means": "parallel higher feature cosine",
            "delta": lambda pivot: pivot["axis_edge_parallel"] - pivot["axis_edge_orthogonal"],
        },
        "feature_mse": {
            "positive_means": "parallel lower feature MSE",
            "delta": lambda pivot: pivot["axis_edge_orthogonal"] - pivot["axis_edge_parallel"],
        },
        "feature_rmse": {
            "positive_means": "parallel lower feature RMSE",
            "delta": lambda pivot: pivot["axis_edge_orthogonal"] - pivot["axis_edge_parallel"],
        },
    }
    key_cols = ["trial_id", "prior_scale", "true_source_row", "input_mode", "decoder_mode"]
    for metric, spec in metrics.items():
        if metric not in trials.columns:
            continue
        pivot = trials.pivot_table(index=key_cols, columns="prior_family", values=metric, aggfunc="first")
        if not set(AXIS_FAMILIES).issubset(set(pivot.columns)):
            continue
        paired = pivot.dropna(subset=list(AXIS_FAMILIES)).reset_index()
        paired["delta"] = spec["delta"](paired)
        for (input_mode, decoder_mode), mode_rows in paired.groupby(["input_mode", "decoder_mode"], sort=True):
            for scale_value, scale_rows in mode_rows.groupby("prior_scale", sort=True):
                values = scale_rows["delta"].to_numpy(dtype=float)
                mean, lo, hi = _bootstrap_mean(values, rng, int(n_bootstrap))
                rows.append(
                    {
                        "input_mode": str(input_mode),
                        "decoder_mode": str(decoder_mode),
                        "metric": metric,
                        "positive_means": spec["positive_means"],
                        "prior_scale": float(scale_value),
                        "mean_parallel_minus_orthogonal": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "fraction_positive": float(np.mean(values > 0.0)) if values.size else float("nan"),
                        "n_pairs": int(values.size),
                    }
                )
            values = mode_rows["delta"].to_numpy(dtype=float)
            mean, lo, hi = _bootstrap_mean(values, rng, int(n_bootstrap))
            rows.append(
                {
                    "input_mode": str(input_mode),
                    "decoder_mode": str(decoder_mode),
                    "metric": metric,
                    "positive_means": spec["positive_means"],
                    "prior_scale": "all",
                    "mean_parallel_minus_orthogonal": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "fraction_positive": float(np.mean(values > 0.0)) if values.size else float("nan"),
                    "n_pairs": int(values.size),
                }
            )
    return pd.DataFrame(rows)


def _write_readme(out_dir: Path, summary: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    all_contrasts = contrasts[
        (contrasts["metric"].eq("feature_cosine")) & (contrasts["prior_scale"].astype(str).eq("all"))
    ].set_index("input_mode")

    def row(mode: str) -> tuple[float, float, float]:
        if mode not in all_contrasts.index:
            return float("nan"), float("nan"), float("nan")
        item = all_contrasts.loc[mode]
        return (
            float(item["mean_parallel_minus_orthogonal"]),
            float(item["ci_low"]),
            float(item["ci_high"]),
        )

    compact_delta, compact_lo, compact_hi = row("augmented_observed_compact")
    true_delta, true_lo, true_hi = row("augmented_true_tau_residual")
    tau_delta, tau_lo, tau_hi = row("augmented_continuous_tau_residual")
    raw_tau_delta, raw_tau_lo, raw_tau_hi = row("augmented_continuous_tau")
    lines = [
        "# Continuous-Tau MLP Axis Contrast",
        "",
        "Paired along/across contrast for the residual continuous-tau MLP decoder.",
        "The paired rows share the same observed response and true eye trace;",
        "the axis label changes the trajectory prior/catalog family.",
        "",
        "All-scale feature-cosine contrast, `axis_edge_parallel - axis_edge_orthogonal`:",
        "",
        "```text",
        f"augmented compact-only:        {compact_delta:+.5f}  CI [{compact_lo:+.5f}, {compact_hi:+.5f}]",
        f"true-tau residual:            {true_delta:+.5f}  CI [{true_lo:+.5f}, {true_hi:+.5f}]",
        f"tau_hat residual:             {tau_delta:+.5f}  CI [{tau_lo:+.5f}, {tau_hi:+.5f}]",
        f"raw tau_hat concatenation:    {raw_tau_delta:+.5f}  CI [{raw_tau_lo:+.5f}, {raw_tau_hi:+.5f}]",
        "```",
        "",
        "Interpretation: this method does not show a meaningful along-contour",
        "advantage. Response-only and true-eye residual modes are exactly or",
        "effectively tied because their paired test inputs are the same across",
        "axis labels. The only possible axis-dependent signal is through the",
        "estimated `tau_hat`; the residual `tau_hat` contrast is near zero.",
        "",
        "Outputs:",
        "",
        "- `continuous_tau_mlp_axis_summary.csv`",
        "- `continuous_tau_mlp_axis_contrasts.csv`",
    ]
    (out_dir / "continuous_tau_mlp_axis_contrast_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260624)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    trials_path = run_dir / "continuous_tau_mlp_feature_decoder_trials.csv"
    trials = pd.read_csv(trials_path)
    summary = _axis_summary(trials)
    contrasts = _axis_contrasts(trials, n_bootstrap=int(args.n_bootstrap), seed=int(args.seed))
    summary.to_csv(run_dir / "continuous_tau_mlp_axis_summary.csv", index=False)
    contrasts.to_csv(run_dir / "continuous_tau_mlp_axis_contrasts.csv", index=False)
    _write_readme(run_dir, summary, contrasts)
    print(f"Wrote axis contrasts to {run_dir}")


if __name__ == "__main__":
    main()
