#!/usr/bin/env python3
"""Plot a standalone covariance-closure response-window sweep panel."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from VisionCore.paths import VISIONCORE_ROOT


TEXT = "#202124"
MODEL = "#2f5f9f"
BRIDGE = "#7b5ea7"
BRIDGE_L = "#e7ddf2"
NULL = "#9a9a9a"

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.8,
})


def clean_axes(ax: plt.Axes, *, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, alpha=0.18, zorder=-1)


def window_ms_from_fig2(fig2_cache: Path) -> dict[int, float]:
    with open(fig2_cache, "rb") as f:
        rows = pickle.load(f)
    out: dict[int, float] = {}
    for idx, res in enumerate(rows[0].get("results", [])):
        if "window_ms" in res:
            out[int(idx)] = float(res["window_ms"])
    return out


def default_roots(sweep_root: Path) -> dict[int, Path]:
    return {
        0: sweep_root / "window_0",
        1: VISIONCORE_ROOT / "outputs" / "matched_twin_covariance_closure_finite_difference",
        2: sweep_root / "window_2",
        3: sweep_root / "window_3",
    }


def parse_root_specs(specs: list[str], sweep_root: Path) -> dict[int, Path]:
    roots = default_roots(sweep_root)
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Root spec must be window_idx=path, got {spec!r}")
        key, val = spec.split("=", 1)
        roots[int(key)] = Path(val)
    return roots


def _finite(vals: np.ndarray) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float64)
    return arr[np.isfinite(arr)]


def bootstrap_mean(vals: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float, float]:
    arr = _finite(vals)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if arr.size < 2 or n_boot <= 0:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    boot = np.mean(arr[idx], axis=1)
    return mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def load_window_rows(
    roots: dict[int, Path],
    window_ms: dict[int, float],
    *,
    target_variant: str,
    projection_control: str,
    basis_source: str,
    k: int,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    metric_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for window_idx, root in sorted(roots.items()):
        metrics_path = root / "finite_difference_capture_metrics.csv"
        if not metrics_path.exists():
            warnings.append(f"missing metrics for window_idx={window_idx}: {metrics_path}")
            continue
        metrics = pd.read_csv(metrics_path)
        d = metrics[
            (metrics["row_status"].astype(str) == "ok")
            & (metrics["target_variant"].astype(str) == str(target_variant))
            & (metrics["projection_control"].astype(str) == str(projection_control))
            & (metrics["basis_source"].astype(str) == str(basis_source))
            & (metrics["k"].astype(int) == int(k))
        ].copy()
        if len(d) == 0:
            warnings.append(f"no matching rows for window_idx={window_idx}: {metrics_path}")
            continue
        d["window_idx"] = int(window_idx)
        d["window_ms"] = float(window_ms.get(int(window_idx), np.nan))
        d["source_root"] = str(root)
        metric_rows.append(d)
        vals = d["effect_minus_unit_shuffle_median"].to_numpy(dtype=np.float64)
        cap = d["capture"].to_numpy(dtype=np.float64)
        mean, lo, hi = bootstrap_mean(vals, n_boot=n_boot, seed=seed + int(window_idx) * 1009)
        cap_mean, cap_lo, cap_hi = bootstrap_mean(cap, n_boot=n_boot, seed=seed + int(window_idx) * 1013)
        finite_vals = _finite(vals)
        summary_rows.append({
            "window_idx": int(window_idx),
            "window_ms": float(window_ms.get(int(window_idx), np.nan)),
            "n_sessions": int(d["session"].nunique()),
            "effect_unit_mean": mean,
            "effect_unit_boot_ci_low": lo,
            "effect_unit_boot_ci_high": hi,
            "capture_mean": cap_mean,
            "capture_boot_ci_low": cap_lo,
            "capture_boot_ci_high": cap_hi,
            "n_effect_positive": int(np.sum(finite_vals > 0.0)),
            "n_effect_nonzero": int(finite_vals.size),
            "effect_unit_min": float(np.min(finite_vals)) if finite_vals.size else float("nan"),
            "effect_unit_max": float(np.max(finite_vals)) if finite_vals.size else float("nan"),
            "source_root": str(root),
        })
    metrics_out = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    summary_out = pd.DataFrame(summary_rows).sort_values("window_idx") if summary_rows else pd.DataFrame()
    return metrics_out, summary_out, warnings


def plot_panel(metrics: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.55), constrained_layout=True)
    if metrics is None or len(metrics) == 0 or summary is None or len(summary) == 0:
        ax.text(0.5, 0.5, "window sweep data not found", transform=ax.transAxes,
                ha="center", va="center", color=TEXT)
        clean_axes(ax)
    else:
        summary = summary.sort_values("window_ms")
        x = summary["window_ms"].to_numpy(dtype=np.float64)
        y = summary["effect_unit_mean"].to_numpy(dtype=np.float64)
        lo = summary["effect_unit_boot_ci_low"].to_numpy(dtype=np.float64)
        hi = summary["effect_unit_boot_ci_high"].to_numpy(dtype=np.float64)

        pivot = metrics.pivot_table(
            index="session",
            columns="window_ms",
            values="effect_minus_unit_shuffle_median",
            aggfunc="mean",
        )
        for _, row in pivot.iterrows():
            yy = row.reindex(x).to_numpy(dtype=np.float64)
            ok = np.isfinite(yy)
            if np.sum(ok) >= 2:
                ax.plot(x[ok], yy[ok], "-", color="0.78", lw=0.75, alpha=0.55, zorder=1)
            ax.scatter(x[ok], yy[ok], s=8, color="0.78", alpha=0.70, linewidths=0, zorder=2)

        ax.axhline(0.0, color="0.55", lw=0.8, ls=":", zorder=0)
        ax.fill_between(x, lo, hi, color=BRIDGE_L, alpha=0.8, lw=0, zorder=3)
        ax.plot(x, y, "o-", color=BRIDGE, lw=2.1, markersize=5.2,
                markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        for xx, yy, nn, pp in zip(
            x,
            y,
            summary["n_effect_nonzero"].astype(int),
            summary["n_effect_positive"].astype(int),
        ):
            ax.text(xx, yy + 0.022, f"{pp}/{nn}", ha="center", va="bottom",
                    fontsize=6.1, color=BRIDGE, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x])
        ax.set_xlabel("Spike-count window for recorded $\\Sigma_{\\mathrm{FEM}}$ (ms)")
        ax.set_ylabel("Excess $\\Sigma_{\\mathrm{FEM}}$ capture\nover unit-shuffle null")
        ax.set_title(title, loc="left", fontsize=8.8, fontweight="bold", color=TEXT)
        ymax = max(float(np.nanmax(hi)) + 0.06, 0.24)
        ymin = min(float(np.nanmin(metrics["effect_minus_unit_shuffle_median"])) - 0.04, -0.02)
        ax.set_ylim(ymin, ymax)
        clean_axes(ax, grid=True)

        ax.text(0.02, 0.04, "PSD target | global + PC1 removed | k=2",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=6.4, color="0.40")

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"covariance_binning_sweep_panel.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    default_sweep = VISIONCORE_ROOT / "outputs" / "matched_twin_covariance_closure_window_sweep"
    default_out = VISIONCORE_ROOT / "outputs" / "covTFTS_figure" / "covariance_binning_sweep_panel"
    p = argparse.ArgumentParser(description="Plot standalone covariance-closure binning-window sweep panel.")
    p.add_argument("--sweep-root", type=Path, default=default_sweep)
    p.add_argument("--root", action="append", default=[],
                   help="Override closure root as window_idx=path. Can be repeated.")
    p.add_argument("--fig2-cache", type=Path, default=VISIONCORE_ROOT / "outputs" / "cache" / "fig2_decomposition_ryan.pkl")
    p.add_argument("--out-dir", type=Path, default=default_out)
    p.add_argument("--target-variant", type=str, default="psd")
    p.add_argument("--projection-control", type=str, default="global_rate+target_pc1")
    p.add_argument("--basis-source", type=str, default="fd_sample_eye_trace_cov")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--title", type=str, default="Covariance prediction is stable across count windows")
    return p


def main() -> None:
    args = build_parser().parse_args()
    roots = parse_root_specs(args.root, args.sweep_root)
    win_ms = window_ms_from_fig2(args.fig2_cache)
    metrics, summary, warnings = load_window_rows(
        roots,
        win_ms,
        target_variant=args.target_variant,
        projection_control=args.projection_control,
        basis_source=args.basis_source,
        k=int(args.k),
        n_boot=int(args.n_boot),
        seed=int(args.seed),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if len(metrics):
        metrics.to_csv(args.out_dir / "covariance_binning_sweep_metrics.csv", index=False)
    if len(summary):
        summary.to_csv(args.out_dir / "covariance_binning_sweep_summary.csv", index=False)
    (args.out_dir / "covariance_binning_sweep_manifest.json").write_text(
        json.dumps(
            {
                "roots": {str(k): str(v) for k, v in roots.items()},
                "fig2_cache": str(args.fig2_cache),
                "target_variant": args.target_variant,
                "projection_control": args.projection_control,
                "basis_source": args.basis_source,
                "k": int(args.k),
                "n_boot": int(args.n_boot),
                "warnings": warnings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    readme = [
        "# Covariance Binning Sweep Panel",
        "",
        "Standalone companion panel for cov_TFTS Figure 4.",
        "",
        f"- target variant: `{args.target_variant}`",
        f"- projection control: `{args.projection_control}`",
        f"- source basis: `{args.basis_source}`",
        f"- source eigenspace dimension: `k={int(args.k)}`",
        "- y-axis: session-level excess recorded FEM covariance capture over unit-shuffle null.",
        "- x-axis: spike-count window used to estimate the recorded FEM covariance target.",
        "",
        "Files:",
        "- `covariance_binning_sweep_panel.png/pdf/svg`",
        "- `covariance_binning_sweep_summary.csv`",
        "- `covariance_binning_sweep_metrics.csv`",
        "- `covariance_binning_sweep_manifest.json`",
    ]
    if warnings:
        readme.extend(["", "Warnings:", *[f"- {w}" for w in warnings]])
    (args.out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    plot_panel(metrics, summary, args.out_dir, title=args.title)
    print(f"Saved to {args.out_dir}")
    for warning in warnings:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
