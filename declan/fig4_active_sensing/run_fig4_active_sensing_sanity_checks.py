#!/usr/bin/env python3
"""Cache-first sanity checks for the Figure 4 active-sensing headline result."""
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

from .generate_fig4_active_sensing import DEFAULT_AGGREGATE_DIR, DEFAULT_INCREMENTAL_DIR


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "fig4_active_sensing" / "sanity_checks"

COLORS = {
    "empirical": "#244f7a",
    "ou": "#d07a22",
    "brownian": "#707070",
    "rotated": "#8064a2",
}
ORDER = ["empirical", "ou", "brownian", "rotated"]


def _scale_value(scale_id: str) -> float:
    return float(str(scale_id).replace("rel_", "").replace("p", ".").replace("x", ""))


def _scale_label(scale: float) -> str:
    return f"{scale:g}x"


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _errbar(ax: plt.Axes, block: pd.DataFrame, y: str, lo: str, hi: str, label: str, color: str) -> None:
    block = block.sort_values("scale")
    x = block["scale"].to_numpy(dtype=float)
    yy = block[y].to_numpy(dtype=float)
    low = block[lo].to_numpy(dtype=float)
    high = block[hi].to_numpy(dtype=float)
    ax.errorbar(
        x,
        yy,
        yerr=np.vstack([yy - low, high - yy]),
        marker="o",
        linewidth=1.8,
        markersize=4,
        capsize=3,
        label=label,
        color=color,
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _absolute_gain_panel(
    gains: pd.DataFrame,
    *,
    motion_summary: str,
    latent: str,
    k: int,
    out_dir: Path,
) -> Path:
    block = gains[
        (gains["motion_summary"] == motion_summary)
        & (gains["latent"] == latent)
        & (gains["k"] == int(k))
        & (gains["family"].isin(ORDER))
    ].copy()
    block["scale"] = block["scale_id"].map(_scale_value)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for fam in ORDER:
        sub = block[block["family"] == fam]
        if sub.empty:
            continue
        _errbar(
            ax,
            sub,
            "incremental_gain_neg_mse",
            "ci95_low",
            "ci95_high",
            fam,
            COLORS[fam],
        )
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks([0.25, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels([_scale_label(v) for v in [0.25, 0.5, 1.0, 1.5, 2.0]])
    ax.set_xlabel("requested observed-RMS scale")
    ax.set_ylabel("incremental gain over static (-MSE)")
    ax.set_title(f"Absolute gains: {latent}, k={k}, {motion_summary}", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    _clean_axis(ax)
    fig.tight_layout()
    path = out_dir / f"absolute_gains_{latent}_k{k}_{motion_summary}.png"
    fig.savefig(path, dpi=220, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return path


def _motion_metric_panels(
    gains: pd.DataFrame,
    motion: pd.DataFrame,
    covariance: pd.DataFrame,
    *,
    out_dir: Path,
) -> Path:
    primary = gains[
        (gains["motion_summary"] == "temporal_pca")
        & (gains["latent"] == "gabor_local_field")
        & (gains["k"] == 4)
        & (gains["family"].isin(ORDER))
    ].copy()
    primary["scale"] = primary["scale_id"].map(_scale_value)
    motion = motion.copy()
    motion["scale"] = motion["scale_id"].map(_scale_value)
    cov = covariance[
        (covariance["summary"] == "temporal_pca") & (covariance["family"].isin(ORDER))
    ].copy()
    cov["scale"] = cov["scale_id"].map(_scale_value)

    fig, axs = plt.subplots(2, 2, figsize=(9.0, 6.4))
    panels = [
        (axs[0, 0], primary, "incremental_gain_neg_mse", "absolute gain", "gain over static (-MSE)"),
        (axs[0, 1], motion, "median_path_length_deg", "path length", "median path length (deg)"),
        (axs[1, 0], motion, "median_speed_mean_deg_s", "mean speed", "median mean speed (deg/s)"),
        (axs[1, 1], cov, "motion_cov_trace", "motion covariance", "motion covariance trace"),
    ]
    for ax, df, col, title, ylabel in panels:
        for fam in ORDER:
            sub = df[df["family"] == fam].sort_values("scale")
            if sub.empty:
                continue
            ax.plot(sub["scale"], sub[col], marker="o", linewidth=1.8, color=COLORS[fam], label=fam)
        ax.set_xticks([0.25, 0.5, 1.0, 1.5, 2.0])
        ax.set_xticklabels([_scale_label(v) for v in [0.25, 0.5, 1.0, 1.5, 2.0]])
        ax.set_xlabel("scale")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        _clean_axis(ax)
    axs[0, 0].axhline(0, color="#222222", linewidth=0.8)
    axs[0, 0].legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "motion_metric_sanity_panels.png"
    fig.savefig(path, dpi=220, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return path


def _joined_metric_table(gains: pd.DataFrame, motion: pd.DataFrame, covariance: pd.DataFrame) -> pd.DataFrame:
    primary = gains[
        (gains["motion_summary"] == "temporal_pca")
        & (gains["latent"] == "gabor_local_field")
        & (gains["k"] == 4)
        & (gains["family"].isin(ORDER))
    ].copy()
    motion_cols = [
        "family",
        "scale_id",
        "median_effective_to_requested_rms",
        "median_path_length_deg",
        "median_speed_mean_deg_s",
        "median_generated_lag1_autocorr",
        "clipped_fraction",
    ]
    cov = covariance[covariance["summary"] == "temporal_pca"][
        [
            "family",
            "scale_id",
            "motion_cov_trace",
            "signal_motion_trace_ratio",
            "signal_motion_subspace_overlap",
        ]
    ].copy()
    joined = primary.merge(motion[motion_cols], on=["family", "scale_id"], how="left")
    joined = joined.merge(cov, on=["family", "scale_id"], how="left")
    joined["scale"] = joined["scale_id"].map(_scale_value)
    return joined.sort_values(["family", "scale"])


def _correlation_table(joined: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "median_path_length_deg",
        "median_speed_mean_deg_s",
        "median_generated_lag1_autocorr",
        "motion_cov_trace",
        "signal_motion_trace_ratio",
        "signal_motion_subspace_overlap",
    ]
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        valid = joined[["incremental_gain_neg_mse", metric]].dropna()
        rows.append(
            {
                "metric": metric,
                "pearson_r_across_family_scale": float(valid.corr().iloc[0, 1]) if valid.shape[0] >= 3 else np.nan,
                "n": int(valid.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def _code_audit() -> list[str]:
    return [
        "Temporal response PCA basis: shared across all raw response movies in the aggregate run, not fit separately by family. It is fit once before decoder CV (`run_backimage_aggregate_fem_information.py`, `_fit_temporal_basis`; call at the temporal-basis stage).",
        "Temporal PCA caveat: the response basis is unsupervised but global to the run, not fit inside each decoder fold. The DCT summaries are the fixed-basis check for this issue.",
        "Feature-target PCA: fit inside each outer decoding fold in `_cross_validated_decode`; test targets use the fold's training PCA transform.",
        "Feature/target standardization: train-fold means and SDs are used for each fold via `_standardize_train_test`.",
        "Static-only and static-plus-motion decoders: use the same decode groups, same grouped-by-image folds, and fixed ridge alpha.",
        "Ridge alpha: fixed at 10.0 for these summaries, so family differences are not from per-family alpha selection.",
    ]


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text_df = df.copy()
    for col in text_df.columns:
        if pd.api.types.is_float_dtype(text_df[col]):
            text_df[col] = text_df[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4g}")
        else:
            text_df[col] = text_df[col].map(lambda v: "" if pd.isna(v) else str(v))
    headers = [str(c) for c in text_df.columns]
    rows = text_df.values.tolist()
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def _write_report(
    *,
    out_dir: Path,
    joined: pd.DataFrame,
    corr: pd.DataFrame,
    abs_temporal: Path,
    abs_dct: Path,
    metric_plot: Path,
) -> Path:
    primary = joined[joined["scale"].isin([0.25, 1.0, 2.0])][
        [
            "family",
            "scale_id",
            "incremental_gain_neg_mse",
            "ci95_low",
            "ci95_high",
            "median_path_length_deg",
            "median_speed_mean_deg_s",
            "motion_cov_trace",
        ]
    ]
    lines = [
        "# Figure 4 Active-Sensing Sanity Checks",
        "",
        "## Generated Plots",
        "",
        f"- Absolute temporal-PC gains: `{abs_temporal.name}`",
        f"- Absolute temporal-DCT gains: `{abs_dct.name}`",
        f"- Motion/gain metric panels: `{metric_plot.name}`",
        "",
        "## Implementation Audit",
        "",
    ]
    lines.extend(f"- {item}" for item in _code_audit())
    lines.extend(
        [
            "",
            "## Primary Absolute-Gain Read",
            "",
            "For Gabor k=4 temporal-PC readout, the absolute gain table shows that the empirical curve is not merely an artifact of subtracting a bad OU control. At 0.25x, empirical is strongly positive, Brownian is modestly positive, rotated is near zero, and OU is negative. At 1x-2x, Brownian/rotated approach empirical while OU remains poor.",
            "",
            "The fixed temporal-DCT plot is the response-basis sanity check. It keeps the same qualitative concern alive rather than resolving it away: empirical remains positive, OU remains weak, and Brownian/rotated are broadly positive under a fixed temporal basis. This argues against a family-specific temporal-PCA-basis artifact, while still pointing to generic phase/temporal diversity as a plausible contributor.",
            "",
            _markdown_table(primary),
            "",
            "## Motion-Metric Correlations",
            "",
            "These correlations are descriptive across the 20 family x scale points, not causal tests.",
            "",
            _markdown_table(corr),
            "",
            "## Interpretation",
            "",
            "- The pattern supports the skepticism in the note: small motion can provide generic derivative/phase-sampling benefit, and empirical specificity is strongest at small scale.",
            "- Brownian catching empirical at large scales is compatible with generic phase coverage or temporal diversity, not unique biological trajectory optimality.",
            "- OU remaining weak suggests the OU control is missing a feature-relevant aspect of the sampled displacement sequence, despite matched effective RMS and lag-1 summary.",
            "- A stronger next control would preserve empirical step-size/temporal spectrum more closely than OU while disrupting trace geometry.",
            "",
        ]
    )
    path = out_dir / "fig4_active_sensing_sanity_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir = Path(args.aggregate_dir)
    incremental_dir = Path(args.incremental_dir)

    gains = _read_csv(incremental_dir / "incremental_gain_vs_static.csv")
    motion = _read_csv(aggregate_dir / "aggregate_motion_summary.csv")
    covariance = _read_csv(aggregate_dir / "covariance_summary.csv")

    abs_temporal = _absolute_gain_panel(
        gains,
        motion_summary="temporal_pca",
        latent="gabor_local_field",
        k=4,
        out_dir=out_dir,
    )
    abs_dct = _absolute_gain_panel(
        gains,
        motion_summary="temporal_dct",
        latent="gabor_local_field",
        k=4,
        out_dir=out_dir,
    )
    metric_plot = _motion_metric_panels(gains, motion, covariance, out_dir=out_dir)

    joined = _joined_metric_table(gains, motion, covariance)
    corr = _correlation_table(joined)
    joined_path = out_dir / "absolute_gain_motion_metrics_joined.csv"
    corr_path = out_dir / "gain_motion_metric_correlations.csv"
    joined.to_csv(joined_path, index=False)
    corr.to_csv(corr_path, index=False)
    report = _write_report(
        out_dir=out_dir,
        joined=joined,
        corr=corr,
        abs_temporal=abs_temporal,
        abs_dct=abs_dct,
        metric_plot=metric_plot,
    )
    metadata = {
        "aggregate_dir": str(aggregate_dir),
        "incremental_dir": str(incremental_dir),
        "outputs": {
            "report": str(report),
            "absolute_temporal_pc_gains": str(abs_temporal),
            "absolute_temporal_dct_gains": str(abs_dct),
            "motion_metric_panels": str(metric_plot),
            "joined_table": str(joined_path),
            "correlation_table": str(corr_path),
        },
    }
    (out_dir / "fig4_active_sensing_sanity_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, default=DEFAULT_AGGREGATE_DIR)
    parser.add_argument("--incremental-dir", type=Path, default=DEFAULT_INCREMENTAL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser


def main() -> None:
    metadata = run(build_parser().parse_args())
    for key, value in metadata["outputs"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
