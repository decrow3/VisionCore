"""Build readout-variant diagnostic panels for aggregate BackImage posthocs.

This consumes an `incremental_staticmean_plus_motion` posthoc folder produced by
`summarize_backimage_aggregate_incremental_motion.py`. It is intentionally
cache-only: no V1-twin responses are recomputed here.
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


READOUT_ORDER = [
    "mean",
    "delta_mean",
    "temporal_pca",
    "temporal_delta_pca",
    "temporal_dct",
    "temporal_dct_delta",
]


def _parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _scale_label(scale_id: str) -> str:
    label = str(scale_id).replace("rel_", "").replace("p", ".")
    return label if label.endswith("x") else f"{label}x"


def _read_metadata(posthoc_dir: Path) -> dict[str, Any]:
    path = posthoc_dir / "run_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _ordered(values: list[Any], preferred: list[Any]) -> list[Any]:
    seen = set(values)
    out = [value for value in preferred if value in seen]
    out.extend(sorted(value for value in seen if value not in set(out)))
    return out


def _load_posthoc(posthoc_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    decode = pd.read_csv(posthoc_dir / "incremental_decode_summary.csv")
    gain = pd.read_csv(posthoc_dir / "incremental_gain_vs_static.csv")
    contrast = pd.read_csv(posthoc_dir / "incremental_gain_contrasts.csv")
    return decode, gain, contrast, _read_metadata(posthoc_dir)


def _build_long_diagnostic(
    decode: pd.DataFrame,
    gain: pd.DataFrame,
    contrast: pd.DataFrame,
    *,
    primary_scales: list[str],
) -> pd.DataFrame:
    aug = decode.loc[decode["model"] == "static_plus_motion"].copy()
    aug_cols = [
        "motion_summary",
        "family",
        "scale_id",
        "latent",
        "k",
        "mean_neg_mse",
        "r2",
        "chosen_alpha_median",
        "feature_dim",
        "target_dim",
        "ridge_alpha_mode",
        "fixed_ridge_alpha",
    ]
    available_aug_cols = [col for col in aug_cols if col in aug.columns]
    gain_aug = gain.merge(
        aug[available_aug_cols],
        on=["motion_summary", "family", "scale_id", "latent", "k"],
        how="left",
    )
    gain_aug["model"] = "static_plus_motion"
    gain_aug["control_contrast"] = ""
    gain_aug["value"] = gain_aug["incremental_gain_neg_mse"].astype(float)
    gain_aug["metric"] = "gain_vs_static_mean"

    contrast_long = contrast.copy()
    contrast_long["control_contrast"] = contrast_long["lhs_family"].astype(str) + "-" + contrast_long["rhs_family"].astype(str)
    contrast_long["family"] = contrast_long["lhs_family"].astype(str)
    contrast_long["model"] = "contrast"
    contrast_long["value"] = contrast_long["incremental_gain_delta_neg_mse"].astype(float)
    contrast_long["metric"] = "incremental_gain_contrast"
    contrast_long["mean_neg_mse"] = np.nan
    contrast_long["r2"] = np.nan
    contrast_long["chosen_alpha_median"] = np.nan
    contrast_long["feature_dim"] = np.nan
    contrast_long["target_dim"] = np.nan
    contrast_long["ridge_alpha_mode"] = ""
    contrast_long["fixed_ridge_alpha"] = np.nan

    cols = [
        "motion_summary",
        "latent",
        "k",
        "model",
        "family",
        "scale_id",
        "metric",
        "control_contrast",
        "value",
        "ci95_low",
        "ci95_high",
        "mean_neg_mse",
        "r2",
        "chosen_alpha_median",
        "feature_dim",
        "target_dim",
        "ridge_alpha_mode",
        "fixed_ridge_alpha",
    ]
    long = pd.concat([gain_aug, contrast_long], ignore_index=True, sort=False)
    long = long.loc[long["scale_id"].isin(primary_scales), [col for col in cols if col in long.columns]]
    return long.sort_values(["latent", "k", "motion_summary", "model", "family", "scale_id"])


def _ci_pass_count(rows: pd.DataFrame) -> int:
    if rows.empty or "ci95_low" not in rows:
        return 0
    return int(np.sum(rows["ci95_low"].astype(float) > 0.0))


def _build_primary_summary(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    readouts = _ordered(long["motion_summary"].dropna().astype(str).unique().tolist(), READOUT_ORDER)
    for latent in sorted(long["latent"].dropna().astype(str).unique()):
        for k in sorted(long.loc[long["latent"] == latent, "k"].dropna().astype(int).unique()):
            for readout in readouts:
                sub = long.loc[
                    (long["latent"] == latent)
                    & (long["k"].astype(int) == int(k))
                    & (long["motion_summary"] == readout)
                ]
                empirical = sub.loc[
                    (sub["model"] == "static_plus_motion")
                    & (sub["family"] == "empirical")
                    & (sub["metric"] == "gain_vs_static_mean")
                ]
                row: dict[str, Any] = {
                    "latent": latent,
                    "k": int(k),
                    "motion_summary": readout,
                    "empirical_mean_gain_primary": float(empirical["value"].mean()) if not empirical.empty else np.nan,
                    "empirical_ci_pass_primary_n": _ci_pass_count(empirical),
                    "empirical_chosen_alpha_median": float(np.nanmedian(empirical["chosen_alpha_median"])) if not empirical.empty else np.nan,
                    "empirical_feature_dim": float(np.nanmedian(empirical["feature_dim"])) if not empirical.empty else np.nan,
                }
                for control in ["ou", "brownian", "rotated"]:
                    control_rows = sub.loc[
                        (sub["model"] == "contrast")
                        & (sub["control_contrast"] == f"empirical-{control}")
                    ]
                    row[f"emp_minus_{control}_mean_primary"] = (
                        float(control_rows["value"].mean()) if not control_rows.empty else np.nan
                    )
                    row[f"emp_minus_{control}_ci_pass_primary_n"] = _ci_pass_count(control_rows)
                rows.append(row)
    return pd.DataFrame(rows)


def _pivot_for_heatmap(summary: pd.DataFrame, latent: str, value_col: str) -> pd.DataFrame:
    sub = summary.loc[summary["latent"] == latent].copy()
    sub["motion_summary"] = pd.Categorical(sub["motion_summary"], categories=READOUT_ORDER, ordered=True)
    pivot = sub.pivot_table(index="motion_summary", columns="k", values=value_col, aggfunc="mean", observed=False)
    return pivot.reindex([r for r in READOUT_ORDER if r in pivot.index])


def _plot_heatmaps(summary: pd.DataFrame, *, value_col: str, title: str, out_path: Path) -> None:
    latents = sorted(summary["latent"].dropna().astype(str).unique())
    if not latents:
        return
    fig_width = max(7.0, 4.8 * len(latents))
    fig, axes = plt.subplots(1, len(latents), figsize=(fig_width, 5.0), squeeze=False, constrained_layout=True)
    matrices = [_pivot_for_heatmap(summary, latent, value_col) for latent in latents]
    finite = np.concatenate([m.to_numpy(dtype=float).ravel() for m in matrices if not m.empty])
    finite = finite[np.isfinite(finite)]
    vmax = float(np.nanpercentile(np.abs(finite), 95)) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    for ax, latent, matrix in zip(axes.ravel(), latents, matrices, strict=True):
        im = ax.imshow(matrix.to_numpy(dtype=float), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(latent.replace("_", " "), fontsize=10)
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels([str(c) for c in matrix.columns])
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels([str(i) for i in matrix.index])
        ax.set_xlabel("target k")
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                val = matrix.iloc[y, x]
                if np.isfinite(val):
                    ax.text(x, y, f"{val:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.78, pad=0.06, label=value_col)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_scale_grid(
    long: pd.DataFrame,
    *,
    value_metric: str,
    family: str | None,
    control_contrast: str | None,
    title: str,
    out_path: Path,
    primary_scales: list[str],
) -> None:
    latents = sorted(long["latent"].dropna().astype(str).unique())
    ks = sorted(long["k"].dropna().astype(int).unique())
    readouts = _ordered(long["motion_summary"].dropna().astype(str).unique().tolist(), READOUT_ORDER)
    if not latents or not ks:
        return
    nrows = len(latents)
    ncols = len(ks)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.55 * nrows), sharex=True, sharey=True, squeeze=False)
    x = np.arange(len(primary_scales))
    colors = plt.get_cmap("tab10")
    for r, latent in enumerate(latents):
        for c, k in enumerate(ks):
            ax = axes[r, c]
            for idx, readout in enumerate(readouts):
                sub = long.loc[
                    (long["latent"] == latent)
                    & (long["k"].astype(int) == int(k))
                    & (long["motion_summary"] == readout)
                    & (long["metric"] == value_metric)
                ]
                if family is not None:
                    sub = sub.loc[sub["family"] == family]
                if control_contrast is not None:
                    sub = sub.loc[sub["control_contrast"] == control_contrast]
                vals = []
                for scale in primary_scales:
                    row = sub.loc[sub["scale_id"] == scale]
                    vals.append(float(row["value"].iloc[0]) if not row.empty else np.nan)
                ax.plot(x, vals, marker="o", linewidth=1.2, markersize=3, color=colors(idx), label=readout)
            ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.7)
            ax.set_title(f"{latent.replace('_', ' ')} k={k}", fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels([_scale_label(s) for s in primary_scales], rotation=25)
            ax.grid(alpha=0.25, linewidth=0.6)
            if c == 0:
                ax.set_ylabel("gain")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _write_report(out_dir: Path, posthoc_dir: Path, metadata: dict[str, Any], summary: pd.DataFrame) -> None:
    top_abs = summary.sort_values("empirical_mean_gain_primary", ascending=False).head(8)
    top_ou = summary.sort_values("emp_minus_ou_mean_primary", ascending=False).head(8)
    abs_cols = ["latent", "k", "motion_summary", "empirical_mean_gain_primary", "empirical_ci_pass_primary_n"]
    ou_cols = ["latent", "k", "motion_summary", "emp_minus_ou_mean_primary", "emp_minus_ou_ci_pass_primary_n"]
    lines = [
        "# Aggregate Readout Variant Diagnostics",
        "",
        f"Posthoc source: `{posthoc_dir}`",
        "",
        "Decoder:",
        f"- ridge alpha mode: `{metadata.get('ridge_alpha_mode', 'unknown')}`",
        f"- fixed ridge alpha: `{metadata.get('fixed_ridge_alpha', 'n/a')}`",
        f"- pca k list: `{metadata.get('pca_k_list', 'unknown')}`",
        f"- latent names: `{metadata.get('latent_names', 'unknown')}`",
        "",
        "Top absolute empirical gain variants:",
        "",
        _markdown_table(top_abs[abs_cols]),
        "",
        "Top empirical-minus-OU variants:",
        "",
        _markdown_table(top_ou[ou_cols]),
        "",
        "Interpretive guardrail:",
        "",
        "Positive empirical-minus-control temporal scores are order-sensitive diagnostics. "
        "They are not the same as positive absolute gain beyond the static mean response.",
    ]
    (out_dir / "readout_variant_diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    cols = list(frame.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                values.append(f"{val:.3f}" if np.isfinite(val) else "nan")
            else:
                values.append(str(val))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posthoc-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--primary-scales", default="rel_0p25x,rel_0p5x,rel_1x")
    return parser


def run(args: argparse.Namespace) -> Path:
    posthoc_dir = Path(args.posthoc_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else posthoc_dir / "readout_variant_diagnostic_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    primary_scales = _parse_csv(args.primary_scales)
    decode, gain, contrast, metadata = _load_posthoc(posthoc_dir)
    long = _build_long_diagnostic(decode, gain, contrast, primary_scales=primary_scales)
    summary = _build_primary_summary(long)
    long.to_csv(out_dir / "nested_alpha_variant_primary_scale_diagnostic.csv", index=False)
    summary.to_csv(out_dir / "nested_alpha_variant_primary_scale_score_table.csv", index=False)
    _plot_heatmaps(
        summary,
        value_col="empirical_mean_gain_primary",
        title="Empirical gain over static mean, primary scales",
        out_path=out_dir / "variant_heatmap_empirical_gain_over_static.png",
    )
    _plot_heatmaps(
        summary,
        value_col="emp_minus_ou_mean_primary",
        title="Empirical minus OU incremental gain, primary scales",
        out_path=out_dir / "variant_heatmap_empirical_minus_ou.png",
    )
    _plot_scale_grid(
        long,
        value_metric="gain_vs_static_mean",
        family="empirical",
        control_contrast=None,
        title="Empirical gain over static mean across primary scales",
        out_path=out_dir / "variant_scale_grid_empirical_gain_over_static.png",
        primary_scales=primary_scales,
    )
    _plot_scale_grid(
        long,
        value_metric="incremental_gain_contrast",
        family=None,
        control_contrast="empirical-ou",
        title="Empirical minus OU incremental gain across primary scales",
        out_path=out_dir / "variant_scale_grid_empirical_minus_ou.png",
        primary_scales=primary_scales,
    )
    _write_report(out_dir, posthoc_dir, metadata, summary)
    print(f"Wrote readout variant diagnostics to {out_dir}", flush=True)
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
