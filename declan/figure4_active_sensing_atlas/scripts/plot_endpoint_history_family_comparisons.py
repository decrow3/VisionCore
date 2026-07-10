"""Plot endpoint-history comparisons across empirical, OU, and Brownian primaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EMPIRICAL_RUN_DIR = Path(
    "outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_fdim4_hpc8_primary_beta_scale1_cached_v1"
)
OU_RUN_DIR = Path(
    "outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_primary_ou_fdim4_hpc8_scale1_cached_v1"
)
BROWNIAN_RUN_DIR = Path(
    "outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_primary_brownian_fdim4_hpc8_scale1_cached_v1"
)

FAMILY_RUNS = {
    "Empirical": EMPIRICAL_RUN_DIR,
    "OU": OU_RUN_DIR,
    "Brownian": BROWNIAN_RUN_DIR,
}

FAMILY_CONDITIONS = {
    "Empirical": "empirical_endpoint_history",
    "OU": "ou_endpoint_history",
    "Brownian": "brownian_endpoint_history",
}

COLORS = {
    "known": "#0f172a",
    "joint": "#0072B2",
    "zero": "#D55E00",
    "static": "#66717d",
    "along": "#0072B2",
    "across": "#CC79A7",
    "grid": "#d7dee8",
    "text": "#111827",
}


def _configure_axes(ax: plt.Axes, *, xgrid: bool = False, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xgrid:
        ax.grid(axis="x", color=COLORS["grid"], alpha=0.75, lw=0.8)
    if ygrid:
        ax.grid(axis="y", color=COLORS["grid"], alpha=0.65, lw=0.8)


def _gate_row(run_dir: Path) -> pd.Series:
    table = pd.read_csv(run_dir / "gates_known_joint_zero_static" / "unified_feature_observer_gate_table.csv")
    block = table[
        table["observation_scale"].astype(str).eq("1.0")
        & table["group_kind"].astype(str).eq("all")
    ]
    if block.empty:
        block = table[table["group_kind"].astype(str).eq("all")]
    if block.empty:
        block = table
    return block.iloc[0]


def _summary_row(run_dir: Path, observer_mode: str) -> pd.Series:
    table = pd.read_csv(run_dir / "endpoint_history_feature_readout_summary.csv")
    block = table[
        table["observation_scale"].astype(str).eq("1.0")
        & table["observer_mode"].astype(str).eq(observer_mode)
    ]
    if block.empty:
        block = table[table["observer_mode"].astype(str).eq(observer_mode)]
    if block.empty:
        raise ValueError(f"Missing observer_mode={observer_mode!r} in {run_dir}")
    return block.iloc[0]


def _family_scores() -> pd.DataFrame:
    modes = [
        ("Static", "static_history", COLORS["static"]),
        ("Zero-history", "zero_history_generative_on_motion", COLORS["zero"]),
        ("Joint latent", "joint_history_generative", COLORS["joint"]),
        ("Known", "known_history_generative", COLORS["known"]),
    ]
    rows: list[dict[str, object]] = []
    for family, run_dir in FAMILY_RUNS.items():
        for label, mode, color in modes:
            row = _summary_row(run_dir, mode)
            rows.append(
                {
                    "primary_family": family,
                    "observer": label,
                    "observer_mode": mode,
                    "color": color,
                    "R2_cv": float(row["R2_cv"]),
                    "mean_feature_cosine": float(row["mean_feature_cosine"]),
                }
            )
    return pd.DataFrame(rows)


def _family_gates() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    contrast_specs = [
        ("Joint - static", "joint_minus_response", "joint_minus_response_ci_low", "joint_minus_response_ci_high", COLORS["joint"]),
        ("Joint - zero", "joint_minus_zero", "joint_minus_zero_ci_low", "joint_minus_zero_ci_high", COLORS["joint"]),
        ("Known - zero", "known_minus_zero", "known_minus_zero_ci_low", "known_minus_zero_ci_high", COLORS["known"]),
        ("Known - joint", "known_minus_joint", "known_minus_joint_ci_low", "known_minus_joint_ci_high", "#475569"),
    ]
    for family, run_dir in FAMILY_RUNS.items():
        row = _gate_row(run_dir)
        for label, value_col, low_col, high_col, color in contrast_specs:
            rows.append(
                {
                    "primary_family": family,
                    "contrast": label,
                    "value": float(row[value_col]),
                    "ci_low": float(row[low_col]),
                    "ci_high": float(row[high_col]),
                    "color": color,
                }
            )
    return pd.DataFrame(rows)


def _trajectory_metrics(run_dir: Path) -> pd.DataFrame:
    arrays = np.load(run_dir / "endpoint_history_dataset_arrays.npz")
    rows = pd.read_csv(run_dir / "endpoint_history_dataset_rows.csv")
    source_rows = rows["source_row"].to_numpy(dtype=int)
    metrics: list[dict[str, object]] = []
    for family, condition in FAMILY_CONDITIONS.items():
        key = f"tau__{condition}"
        if key not in arrays:
            continue
        tau = np.asarray(arrays[key], dtype=np.float64)
        if tau.ndim != 2 or tau.shape[1] % 2:
            raise ValueError(f"{key} should have shape (n, 2 * history_frames), got {tau.shape}")
        histories = tau.reshape(tau.shape[0], tau.shape[1] // 2, 2)
        endpoint = np.zeros((histories.shape[0], 1, 2), dtype=np.float64)
        traces = np.concatenate([histories, endpoint], axis=1)
        for sample_idx, trace in enumerate(traces):
            steps = np.diff(trace, axis=0)
            path_length = float(np.sum(np.linalg.norm(steps, axis=1)))
            rms = float(np.sqrt(np.mean(np.sum(trace * trace, axis=1))))
            centered = trace - np.mean(trace, axis=0, keepdims=True)
            cov = centered.T @ centered / max(1, centered.shape[0])
            evals, evecs = np.linalg.eigh(cov)
            order = np.argsort(evals)[::-1]
            along_axis = evecs[:, order[0]]
            across_axis = evecs[:, order[1]]
            along = trace @ along_axis
            across = trace @ across_axis
            along_rms = float(np.sqrt(np.mean(along * along)))
            across_rms = float(np.sqrt(np.mean(across * across)))
            anisotropy = float((along_rms**2 - across_rms**2) / (along_rms**2 + across_rms**2 + 1e-12))
            metrics.append(
                {
                    "primary_family": family,
                    "sample_index": int(sample_idx),
                    "source_row": int(source_rows[sample_idx]),
                    "history_rms_deg": rms,
                    "history_path_length_deg": path_length,
                    "along_principal_rms_deg": along_rms,
                    "across_principal_rms_deg": across_rms,
                    "principal_anisotropy": anisotropy,
                    "along_across_rms_ratio": float(along_rms / (across_rms + 1e-12)),
                }
            )
    return pd.DataFrame(metrics)


def _plot_family_scores(ax: plt.Axes, scores: pd.DataFrame) -> None:
    families = list(FAMILY_RUNS)
    observers = ["Static", "Zero-history", "Joint latent", "Known"]
    width = 0.19
    x = np.arange(len(families), dtype=float)
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(observers))
    for offset, observer in zip(offsets, observers):
        block = scores[scores["observer"].eq(observer)].set_index("primary_family").loc[families]
        ax.bar(
            x + offset,
            block["R2_cv"].to_numpy(dtype=float),
            width=width,
            color=str(block["color"].iloc[0]),
            label=observer,
        )
    ax.axhline(0.0, color="#94a3b8", lw=1.0)
    ax.set_xticks(x, families)
    ax.set_ylabel("pooled R2_cv")
    ax.set_title("A. Primary history family: observer scores")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    _configure_axes(ax)


def _plot_family_gates(ax: plt.Axes, gates: pd.DataFrame) -> None:
    families = list(FAMILY_RUNS)
    contrasts = ["Joint - static", "Joint - zero", "Known - zero", "Known - joint"]
    y_positions = np.arange(len(contrasts), dtype=float)
    offsets = {"Empirical": -0.22, "OU": 0.0, "Brownian": 0.22}
    markers = {"Empirical": "o", "OU": "s", "Brownian": "^"}
    for family in families:
        block = gates[gates["primary_family"].eq(family)].set_index("contrast").loc[contrasts]
        for yi, (contrast, row) in enumerate(block.iterrows()):
            value = float(row["value"])
            low = float(row["ci_low"])
            high = float(row["ci_high"])
            ypos = y_positions[yi] + offsets[family]
            ax.errorbar(
                value,
                ypos,
                xerr=[[value - low], [high - value]],
                marker=markers[family],
                color=str(row["color"]),
                ecolor=str(row["color"]),
                lw=1.8,
                capsize=2.5,
                markersize=6,
                linestyle="none",
                label=family if yi == 0 else None,
            )
    ax.axvline(0.0, color=COLORS["text"], lw=1.0)
    ax.set_yticks(y_positions, contrasts)
    ax.invert_yaxis()
    ax.set_xlabel("Delta pooled R2_cv")
    ax.set_title("B. Gate contrasts by primary family")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlim(-0.35, 2.55)
    _configure_axes(ax, xgrid=True, ygrid=False)


def _plot_trajectory_magnitude(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    families = list(FAMILY_RUNS)
    data = [
        metrics[metrics["primary_family"].eq(family)]["history_path_length_deg"].to_numpy(dtype=float)
        for family in families
    ]
    ax.boxplot(
        data,
        tick_labels=families,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
        boxprops={"facecolor": "#e2e8f0", "edgecolor": "#475569"},
        medianprops={"color": "#111827", "linewidth": 1.6},
        whiskerprops={"color": "#475569"},
        capprops={"color": "#475569"},
    )
    means = metrics.groupby("primary_family")["history_path_length_deg"].mean().reindex(families)
    for idx, value in enumerate(means.to_numpy(dtype=float), start=1):
        ax.scatter(idx, value, s=28, color="#D55E00", zorder=3)
    ax.set_ylabel("endpoint-history path length (deg)")
    ax.set_title("C. Rendered history magnitude")
    _configure_axes(ax)


def _plot_along_across(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    families = list(FAMILY_RUNS)
    x = np.arange(len(families), dtype=float)
    width = 0.34
    grouped = metrics.groupby("primary_family")
    along_mean = grouped["along_principal_rms_deg"].mean().reindex(families).to_numpy(dtype=float)
    across_mean = grouped["across_principal_rms_deg"].mean().reindex(families).to_numpy(dtype=float)
    along_sem = grouped["along_principal_rms_deg"].sem().reindex(families).to_numpy(dtype=float)
    across_sem = grouped["across_principal_rms_deg"].sem().reindex(families).to_numpy(dtype=float)
    ax.bar(
        x - width / 2,
        along_mean,
        width=width,
        color=COLORS["along"],
        yerr=along_sem,
        capsize=2.5,
        label="along principal axis",
    )
    ax.bar(
        x + width / 2,
        across_mean,
        width=width,
        color=COLORS["across"],
        yerr=across_sem,
        capsize=2.5,
        label="across principal axis",
    )
    ratio = grouped["along_across_rms_ratio"].mean().reindex(families).to_numpy(dtype=float)
    for xi, value in zip(x, ratio):
        ax.text(xi, max(along_mean.max(), across_mean.max()) * 1.05, f"{value:.1f}x", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, families)
    ax.set_ylabel("RMS displacement (deg)")
    ax.set_title("D. Principal-axis along/across, not image-edge")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    _configure_axes(ax)


def plot_family_comparisons(out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = _family_scores()
    gates = _family_gates()
    metrics = _trajectory_metrics(EMPIRICAL_RUN_DIR)
    scores.to_csv(out_dir / "endpoint_history_family_scores.csv", index=False)
    gates.to_csv(out_dir / "endpoint_history_family_gates.csv", index=False)
    metrics.to_csv(out_dir / "endpoint_history_family_trajectory_metrics.csv", index=False)

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
    _plot_family_scores(axes[0, 0], scores)
    _plot_family_gates(axes[0, 1], gates)
    _plot_trajectory_magnitude(axes[1, 0], metrics)
    _plot_along_across(axes[1, 1], metrics)
    fig.suptitle(
        "Endpoint-history family controls and trajectory geometry",
        fontsize=13,
        fontweight="bold",
    )
    png = out_dir / "endpoint_history_family_comparisons.png"
    pdf = out_dir / "endpoint_history_family_comparisons.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf, out_dir / "endpoint_history_family_trajectory_metrics.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=EMPIRICAL_RUN_DIR / "main_results_figures",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    png, pdf, metrics_csv = plot_family_comparisons(args.out_dir)
    print(f"[endpoint-history-family-comparisons] wrote {png}")
    print(f"[endpoint-history-family-comparisons] wrote {pdf}")
    print(f"[endpoint-history-family-comparisons] wrote {metrics_csv}")


if __name__ == "__main__":
    main()
