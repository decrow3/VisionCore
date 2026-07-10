"""Make compact main-results figures for the endpoint-history readout assay."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path(
    "outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_fdim4_hpc8_primary_beta_scale1_cached_v1"
)
DEFAULT_SWEEP_DIR = Path(
    "outputs/figure4_endpoint_history_feature_readout_rr100_n64_feature_dim_sweep_v1"
)


COLORS = {
    "known": "#0f172a",
    "joint": "#0f766e",
    "zero": "#b45309",
    "response": "#235789",
    "static": "#66717d",
    "neutral": "#475569",
    "grid": "#cbd5e1",
}


def _configure_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=COLORS["grid"], alpha=0.55, lw=0.8)


def _gate_row(path: Path) -> pd.Series:
    table = pd.read_csv(path)
    block = table[
        table["observation_scale"].astype(str).eq("1.0")
        & table["group_kind"].astype(str).eq("all")
    ]
    if block.empty:
        block = table[table["group_kind"].astype(str).eq("all")]
    if block.empty:
        block = table
    return block.iloc[0]


def _summary_rows(run_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(run_dir / "endpoint_history_feature_readout_summary.csv")
    block = summary[summary["observation_scale"].astype(str).eq("1.0")].copy()
    if block.empty:
        block = summary[summary["observation_scale"].astype(str).eq("all")].copy()
    return block


def _plot_scores(ax: plt.Axes, summary: pd.DataFrame) -> None:
    labels = [
        ("Static", "static_history", COLORS["static"]),
        ("Response-only\nunknown", "joint_history_response_only", COLORS["response"]),
        ("Zero-history\nunknown", "zero_history_generative_on_motion", COLORS["zero"]),
        ("Joint latent\nhistory", "joint_history_generative", COLORS["joint"]),
        ("Known\nhistory", "known_history_generative", COLORS["known"]),
    ]
    rows = []
    for label, mode, color in labels:
        matched = summary[summary["observer_mode"].astype(str).eq(mode)]
        if matched.empty:
            continue
        rows.append((label, float(matched.iloc[0]["R2_cv"]), color))

    y = np.arange(len(rows))
    values = np.asarray([row[1] for row in rows], dtype=float)
    ax.barh(y, values, color=[row[2] for row in rows], height=0.7)
    ax.set_yticks(y, [row[0] for row in rows])
    ax.invert_yaxis()
    ax.axvline(0.0, color="#94a3b8", lw=1.0)
    ax.set_xlabel("pooled R2_cv (higher is better)")
    ax.set_title("A. Endpoint feature recovery, n=128")
    for yi, value in zip(y, values):
        ax.text(-0.06, yi, f"{value:.2f}", ha="right", va="center", color="white", fontsize=8)
    _configure_axes(ax)


def _contrast_rows(run_dir: Path) -> list[dict[str, object]]:
    zero_gate = _gate_row(
        run_dir / "gates_known_joint_zero_static" / "unified_feature_observer_gate_table.csv"
    )
    response_gate = _gate_row(
        run_dir / "gates_known_joint_responseonly_static" / "unified_feature_observer_gate_table.csv"
    )
    return [
        {
            "label": "Joint - static",
            "value": float(zero_gate["joint_minus_response"]),
            "low": float(zero_gate["joint_minus_response_ci_low"]),
            "high": float(zero_gate["joint_minus_response_ci_high"]),
            "color": COLORS["joint"],
        },
        {
            "label": "Joint - zero-history",
            "value": float(zero_gate["joint_minus_zero"]),
            "low": float(zero_gate["joint_minus_zero_ci_low"]),
            "high": float(zero_gate["joint_minus_zero_ci_high"]),
            "color": COLORS["joint"],
        },
        {
            "label": "Known - zero-history",
            "value": float(zero_gate["known_minus_zero"]),
            "low": float(zero_gate["known_minus_zero_ci_low"]),
            "high": float(zero_gate["known_minus_zero_ci_high"]),
            "color": COLORS["known"],
        },
        {
            "label": "Joint - response-only",
            "value": float(response_gate["joint_minus_zero"]),
            "low": float(response_gate["joint_minus_zero_ci_low"]),
            "high": float(response_gate["joint_minus_zero_ci_high"]),
            "color": COLORS["joint"],
        },
        {
            "label": "Known - response-only",
            "value": float(response_gate["known_minus_zero"]),
            "low": float(response_gate["known_minus_zero_ci_low"]),
            "high": float(response_gate["known_minus_zero_ci_high"]),
            "color": COLORS["known"],
        },
        {
            "label": "Known - joint",
            "value": float(zero_gate["known_minus_joint"]),
            "low": float(zero_gate["known_minus_joint_ci_low"]),
            "high": float(zero_gate["known_minus_joint_ci_high"]),
            "color": COLORS["neutral"],
        },
    ]


def _plot_contrasts(ax: plt.Axes, run_dir: Path) -> None:
    rows = _contrast_rows(run_dir)
    y = np.arange(len(rows))
    for yi, row in zip(y, rows):
        value = float(row["value"])
        low = float(row["low"])
        high = float(row["high"])
        ax.errorbar(
            value,
            yi,
            xerr=[[value - low], [high - value]],
            fmt="o",
            color=str(row["color"]),
            ecolor=str(row["color"]),
            elinewidth=2.0,
            capsize=3,
            markersize=7,
        )
        ax.text(high + 0.035, yi, f"{value:.2f}", va="center", fontsize=8)
    ax.axvline(0.0, color="#0f172a", lw=1.0)
    ax.set_yticks(y, [str(row["label"]) for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Delta pooled R2_cv")
    ax.set_title("B. Gate contrasts with source-bootstrap CIs")
    ax.set_xlim(-0.12, 1.42)
    _configure_axes(ax)


def _plot_feature_dim_sweep(ax: plt.Axes, sweep_dir: Path) -> None:
    summary = pd.read_csv(sweep_dir / "endpoint_history_feature_dim_sweep_summary.csv")
    labels = [
        ("Joint latent history", "joint_history_generative", "#0072B2", "o", "-"),
        ("Zero-history unknown", "zero_history_generative_on_motion", "#D55E00", "s", "--"),
        ("Static", "static_history", "#000000", "^", "-."),
        ("Response-only unknown", "joint_history_response_only", "#CC79A7", "D", ":"),
    ]
    for label, mode, color, marker, linestyle in labels:
        block = summary[summary["observer_mode"].astype(str).eq(mode)].sort_values("feature_dim")
        if block.empty:
            continue
        ax.plot(
            block["feature_dim"].to_numpy(dtype=float),
            block["R2_cv"].to_numpy(dtype=float),
            marker=marker,
            ls=linestyle,
            lw=2.4,
            markersize=6,
            label=label,
            color=color,
        )
    ax.axvline(4.0, color=COLORS["known"], ls="--", lw=1.2)
    ax.text(
        4.15,
        0.96,
        "promoted dim",
        transform=ax.get_xaxis_transform(),
        fontsize=8,
        va="top",
        ha="left",
    )
    ax.axhline(0.0, color="#94a3b8", lw=1.0)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32], ["2", "4", "8", "16", "32"])
    ax.set_xlabel("feature dimension")
    ax.set_ylabel("pooled R2_cv")
    ax.set_title("C. n=64 feature-dimension screen")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _configure_axes(ax)


def _plot_contract_qc(ax: plt.Axes, run_dir: Path) -> None:
    traces = pd.read_csv(run_dir / "endpoint_history_trace_metrics.csv")
    block = traces[traces["condition"].astype(str).ne("static_endpoint_history")].copy()
    families = ["empirical", "ou", "brownian"]
    positions = np.arange(len(families))
    data = [
        block[block["family"].astype(str).eq(family)]["history_path_length_deg"].to_numpy(dtype=float)
        for family in families
    ]
    ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#e2e8f0", "edgecolor": "#475569"},
        medianprops={"color": "#0f172a", "linewidth": 1.5},
        whiskerprops={"color": "#475569"},
        capprops={"color": "#475569"},
    )
    endpoint_norm = block["endpoint_norm_deg"].to_numpy(dtype=float)
    max_endpoint_norm = float(np.nanmax(endpoint_norm)) if len(endpoint_norm) else np.nan
    ax.text(
        0.02,
        0.95,
        f"max final displacement = {max_endpoint_norm:.2e} deg",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#0f172a",
    )
    ax.set_xticks(positions, ["Empirical", "OU", "Brownian"])
    ax.set_ylabel("history path length (deg)")
    ax.set_title("D. Endpoint-alignment QC")
    _configure_axes(ax)


def plot_main_results(run_dir: Path, sweep_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary_rows(run_dir)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 130,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)
    _plot_scores(axes[0, 0], summary)
    _plot_contrasts(axes[0, 1], run_dir)
    _plot_feature_dim_sweep(axes[1, 0], sweep_dir)
    _plot_contract_qc(axes[1, 1], run_dir)
    fig.suptitle(
        "Endpoint-aligned history readout: main results",
        fontsize=13,
        fontweight="bold",
    )
    png = out_dir / "endpoint_history_main_results.png"
    pdf = out_dir / "endpoint_history_main_results.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir
    out_dir = args.out_dir if args.out_dir is not None else run_dir / "main_results_figures"
    png, pdf = plot_main_results(run_dir=run_dir, sweep_dir=args.sweep_dir, out_dir=out_dir)
    print(f"[endpoint-history-main-results] wrote {png}")
    print(f"[endpoint-history-main-results] wrote {pdf}")


if __name__ == "__main__":
    main()
