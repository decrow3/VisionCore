"""Plot endpoint-history feature readout sweeps across feature dimensions."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.figure4_active_sensing_atlas.scripts.build_panel_c_continuous_feature_embedding_reconstruction import (
    _configure_matplotlib,
)


def _feature_dim_from_dir(path: Path) -> int:
    match = re.search(r"fdim(\d+)", path.name)
    if match is None:
        raise ValueError(f"Could not infer feature dimension from directory name: {path}")
    return int(match.group(1))


def _read_sweep(run_dirs: list[Path], scale: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    beta_rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        feature_dim = _feature_dim_from_dir(run_dir)
        summary = pd.read_csv(run_dir / "endpoint_history_feature_readout_summary.csv")
        summary = summary[summary["observation_scale"].astype(str).eq(str(scale))]
        for row in summary.itertuples(index=False):
            summary_rows.append(
                {
                    "feature_dim": feature_dim,
                    "observer_mode": str(row.observer_mode),
                    "R2_cv": float(row.R2_cv),
                    "mean_feature_cosine": float(row.mean_feature_cosine),
                    "median_feature_pred_norm": float(row.median_feature_pred_norm),
                }
            )

        contrasts = pd.read_csv(run_dir / "endpoint_history_feature_readout_contrasts.csv")
        contrasts = contrasts[contrasts["observation_scale"].astype(str).eq(str(scale))]
        for row in contrasts.itertuples(index=False):
            contrast_rows.append(
                {
                    "feature_dim": feature_dim,
                    "contrast": str(row.contrast),
                    "mean_feature_cosine_delta": float(row.mean_feature_cosine_delta),
                    "ci_low": float(row.ci_low),
                    "ci_high": float(row.ci_high),
                    "fraction_positive": float(row.fraction_positive),
                }
            )

        models = pd.read_csv(run_dir / "endpoint_history_feature_readout_models.csv")
        if "known_history_generative_beta" in models:
            beta = models.loc[
                models["observer_mode"].astype(str).eq("known_history_generative"),
                "known_history_generative_beta",
            ].dropna()
            beta_rows.append(
                {
                    "feature_dim": feature_dim,
                    "mean_known_history_generative_beta": float(beta.mean()) if len(beta) else np.nan,
                    "selected_betas": ",".join(f"{float(value):g}" for value in beta.to_numpy(dtype=float)),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows), pd.DataFrame(beta_rows)


def plot_sweep(summary: pd.DataFrame, contrasts: pd.DataFrame, betas: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    _configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.1), constrained_layout=True)
    colors = {
        "static_history": "#66717d",
        "known_history_generative": "#0f172a",
        "joint_history_generative": "#0f766e",
        "zero_history_generative_on_motion": "#b45309",
        "joint_history_response_only": "#235789",
    }
    labels = {
        "static_history": "static",
        "known_history_generative": "known gen",
        "joint_history_generative": "joint gen",
        "zero_history_generative_on_motion": "zero gen",
        "joint_history_response_only": "joint resp-only",
    }
    for observer in labels:
        block = summary[summary["observer_mode"].eq(observer)].sort_values("feature_dim")
        if block.empty:
            continue
        axes[0].plot(
            block["feature_dim"],
            block["R2_cv"],
            marker="o",
            lw=1.8,
            color=colors[observer],
            label=labels[observer],
        )
    axes[0].axhline(0.0, color="#94a3b8", lw=0.8)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([2, 4, 8, 16, 32], [2, 4, 8, 16, 32])
    axes[0].set_xlabel("feature dim")
    axes[0].set_title("pooled R2_cv")
    axes[0].legend(frameon=False, fontsize=7)

    contrast_labels = {
        "joint_generative_minus_zero_generative": "joint gen - zero gen",
        "joint_generative_motion_minus_static_history": "joint gen - static",
        "known_generative_minus_joint_generative": "known gen - joint gen",
    }
    offsets = {
        "joint_generative_minus_zero_generative": -0.12,
        "joint_generative_motion_minus_static_history": 0.0,
        "known_generative_minus_joint_generative": 0.12,
    }
    x_labels = sorted(summary["feature_dim"].drop_duplicates().astype(int).tolist())
    x_lookup = {feature_dim: idx for idx, feature_dim in enumerate(x_labels)}
    for contrast, label in contrast_labels.items():
        block = contrasts[contrasts["contrast"].eq(contrast)].sort_values("feature_dim")
        if block.empty:
            continue
        x = np.asarray([x_lookup[int(value)] for value in block["feature_dim"]], dtype=float) + offsets[contrast]
        y = block["mean_feature_cosine_delta"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y - block["ci_low"].to_numpy(dtype=float),
                block["ci_high"].to_numpy(dtype=float) - y,
            ]
        )
        axes[1].errorbar(x, y, yerr=yerr, marker="o", lw=1.4, capsize=2.5, label=label)
    axes[1].axhline(0.0, color="#94a3b8", lw=0.8)
    axes[1].set_xticks(np.arange(len(x_labels)), x_labels)
    axes[1].set_xlabel("feature dim")
    axes[1].set_ylabel("delta cosine")
    axes[1].set_title("paired cosine contrasts")
    axes[1].legend(frameon=False, fontsize=7)

    betas = betas.sort_values("feature_dim")
    axes[2].bar(
        betas["feature_dim"].astype(str),
        betas["mean_known_history_generative_beta"],
        color="#0f172a",
    )
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_xlabel("feature dim")
    axes[2].set_ylabel("inner-val beta")
    axes[2].set_title("known-history beta")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    png = out_dir / "endpoint_history_feature_dim_sweep.png"
    pdf = out_dir / "endpoint_history_feature_dim_sweep.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scale", default="1.0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary, contrasts, betas = _read_sweep(list(args.run_dirs), scale=str(args.scale))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "endpoint_history_feature_dim_sweep_summary.csv", index=False)
    contrasts.to_csv(args.out_dir / "endpoint_history_feature_dim_sweep_contrasts.csv", index=False)
    betas.to_csv(args.out_dir / "endpoint_history_feature_dim_sweep_betas.csv", index=False)
    png, _pdf = plot_sweep(summary, contrasts, betas, args.out_dir)
    print(f"[endpoint-history-sweep] wrote {png}", flush=True)


if __name__ == "__main__":
    main()
