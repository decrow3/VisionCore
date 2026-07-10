"""Plot endpoint-history edge-parallel versus edge-orthogonal controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PARALLEL_RUN_DIR = Path(
    "outputs/figure4_endpoint_history_feature_readout_rr100_n128_axis_parallel_orthogonal_fdim4_hpc8_scale1_v1"
)
ORTHOGONAL_RUN_DIR = Path(
    "outputs/figure4_endpoint_history_feature_readout_rr100_n128_axis_parallel_orthogonal_fdim4_hpc8_primary_orthogonal_scale1_cached_v1"
)

PRIMARY_RUNS = {
    "Edge-parallel": PARALLEL_RUN_DIR,
    "Edge-orthogonal": ORTHOGONAL_RUN_DIR,
}

OBSERVER_MODES = {
    "Static": "static_history",
    "Zero-history": "zero_history_generative_on_motion",
    "Response-only": "joint_history_response_only",
    "Joint latent": "joint_history_generative",
    "Known": "known_history_generative",
}

COLORS = {
    "Static": "#66717d",
    "Zero-history": "#D55E00",
    "Response-only": "#CC79A7",
    "Joint latent": "#0072B2",
    "Known": "#0f172a",
    "parallel": "#0072B2",
    "orthogonal": "#D55E00",
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


def _summary_scores() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for primary, run_dir in PRIMARY_RUNS.items():
        summary = pd.read_csv(run_dir / "endpoint_history_feature_readout_summary.csv")
        block = summary[summary["observation_scale"].astype(str).eq("1.0")]
        for label, mode in OBSERVER_MODES.items():
            match = block[block["observer_mode"].astype(str).eq(mode)]
            if match.empty:
                continue
            row = match.iloc[0]
            rows.append(
                {
                    "primary": primary,
                    "observer": label,
                    "observer_mode": mode,
                    "R2_cv": float(row["R2_cv"]),
                    "mean_feature_cosine": float(row["mean_feature_cosine"]),
                }
            )
    return pd.DataFrame(rows)


def _trial_mode_table(run_dir: Path, modes: list[str]) -> pd.DataFrame:
    trials = pd.read_csv(run_dir / "endpoint_history_feature_readout_trials.csv")
    block = trials[
        trials["observation_scale"].astype(str).eq("1.0")
        & trials["observer_mode"].astype(str).isin(modes)
    ].copy()
    return block


def _pooled_r2(block: pd.DataFrame, mode: str) -> float:
    values = block[block["observer_mode"].astype(str).eq(mode)]
    if values.empty:
        return float("nan")
    sse = float(values["feature_sse"].sum())
    sst = float(values["feature_sst_train_baseline"].sum())
    return 1.0 - sse / sst


def _bootstrap_delta_within_run(
    run_dir: Path,
    *,
    lhs_mode: str,
    rhs_mode: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    block = _trial_mode_table(run_dir, [lhs_mode, rhs_mode])
    sources = np.asarray(sorted(block["true_source_row"].astype(int).unique()), dtype=int)
    observed = _pooled_r2(block, lhs_mode) - _pooled_r2(block, rhs_mode)
    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_bootstrap), dtype=float)
    grouped = {source: group for source, group in block.groupby(block["true_source_row"].astype(int))}
    for idx in range(int(n_bootstrap)):
        sample = rng.choice(sources, size=len(sources), replace=True)
        sampled = pd.concat([grouped[int(source)] for source in sample], ignore_index=True)
        boot[idx] = _pooled_r2(sampled, lhs_mode) - _pooled_r2(sampled, rhs_mode)
    return observed, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _bootstrap_delta_across_primaries(
    *,
    mode: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    parallel = _trial_mode_table(PARALLEL_RUN_DIR, [mode])
    orthogonal = _trial_mode_table(ORTHOGONAL_RUN_DIR, [mode])
    common = sorted(
        set(parallel["true_source_row"].astype(int).unique()).intersection(
            set(orthogonal["true_source_row"].astype(int).unique())
        )
    )
    if not common:
        raise ValueError("No common sources between parallel and orthogonal primary runs")
    parallel = parallel[parallel["true_source_row"].astype(int).isin(common)]
    orthogonal = orthogonal[orthogonal["true_source_row"].astype(int).isin(common)]
    observed = _pooled_r2(orthogonal, mode) - _pooled_r2(parallel, mode)
    grouped_parallel = {source: group for source, group in parallel.groupby(parallel["true_source_row"].astype(int))}
    grouped_orthogonal = {source: group for source, group in orthogonal.groupby(orthogonal["true_source_row"].astype(int))}
    sources = np.asarray(common, dtype=int)
    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_bootstrap), dtype=float)
    for idx in range(int(n_bootstrap)):
        sample = rng.choice(sources, size=len(sources), replace=True)
        p_sample = pd.concat([grouped_parallel[int(source)] for source in sample], ignore_index=True)
        o_sample = pd.concat([grouped_orthogonal[int(source)] for source in sample], ignore_index=True)
        boot[idx] = _pooled_r2(o_sample, mode) - _pooled_r2(p_sample, mode)
    return observed, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _contrast_table(n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    within_specs = [
        ("Joint - static", "joint_history_generative", "static_history"),
        ("Joint - zero", "joint_history_generative", "zero_history_generative_on_motion"),
        ("Known - joint", "known_history_generative", "joint_history_generative"),
    ]
    for primary, run_dir in PRIMARY_RUNS.items():
        for label, lhs, rhs in within_specs:
            value, low, high = _bootstrap_delta_within_run(
                run_dir,
                lhs_mode=lhs,
                rhs_mode=rhs,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            rows.append(
                {
                    "comparison": "within-primary gate",
                    "primary": primary,
                    "contrast": label,
                    "value": value,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    for label, mode in [
        ("Joint", "joint_history_generative"),
        ("Known", "known_history_generative"),
        ("Zero-history", "zero_history_generative_on_motion"),
        ("Response-only", "joint_history_response_only"),
    ]:
        value, low, high = _bootstrap_delta_across_primaries(
            mode=mode,
            n_bootstrap=n_bootstrap,
            seed=seed + 17,
        )
        rows.append(
            {
                "comparison": "orthogonal-minus-parallel",
                "primary": "Edge-orthogonal - Edge-parallel",
                "contrast": label,
                "value": value,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def _edge_projection_metrics() -> pd.DataFrame:
    rows = pd.read_csv(PARALLEL_RUN_DIR / "endpoint_history_dataset_rows.csv")
    arrays = np.load(PARALLEL_RUN_DIR / "endpoint_history_dataset_arrays.npz")
    specs = {
        "Edge-parallel": "tau__axis_edge_parallel_endpoint_history",
        "Edge-orthogonal": "tau__axis_edge_orthogonal_endpoint_history",
    }
    out: list[dict[str, object]] = []
    edge_deg = rows["image_edge_axis_deg"].to_numpy(dtype=float)
    for label, key in specs.items():
        tau = np.asarray(arrays[key], dtype=float)
        histories = tau.reshape(tau.shape[0], tau.shape[1] // 2, 2)
        endpoint = np.zeros((histories.shape[0], 1, 2), dtype=float)
        traces = np.concatenate([histories, endpoint], axis=1)
        for idx, trace in enumerate(traces):
            theta = np.deg2rad(edge_deg[idx])
            edge_axis = np.asarray([np.cos(theta), np.sin(theta)], dtype=float)
            orth_axis = np.asarray([-np.sin(theta), np.cos(theta)], dtype=float)
            along = trace @ edge_axis
            across = trace @ orth_axis
            steps = np.diff(trace, axis=0)
            out.append(
                {
                    "condition": label,
                    "sample_index": int(rows.iloc[idx]["sample_index"]),
                    "source_row": int(rows.iloc[idx]["source_row"]),
                    "edge_axis_deg": float(edge_deg[idx]),
                    "edge_along_rms_deg": float(np.sqrt(np.mean(along * along))),
                    "edge_across_rms_deg": float(np.sqrt(np.mean(across * across))),
                    "history_path_length_deg": float(np.sum(np.linalg.norm(steps, axis=1))),
                    "history_rms_deg": float(np.sqrt(np.mean(np.sum(trace * trace, axis=1)))),
                }
            )
    return pd.DataFrame(out)


def _plot_scores(ax: plt.Axes, scores: pd.DataFrame) -> None:
    primaries = list(PRIMARY_RUNS)
    observers = ["Static", "Zero-history", "Response-only", "Joint latent", "Known"]
    width = 0.16
    x = np.arange(len(primaries), dtype=float)
    offsets = np.linspace(-2.0 * width, 2.0 * width, len(observers))
    for offset, observer in zip(offsets, observers):
        block = scores[scores["observer"].eq(observer)].set_index("primary").loc[primaries]
        ax.bar(
            x + offset,
            block["R2_cv"].to_numpy(dtype=float),
            width=width,
            color=COLORS[observer],
            label=observer,
        )
    ax.axhline(0.0, color="#94a3b8", lw=1.0)
    ax.set_xticks(x, primaries)
    ax.set_ylabel("pooled R2_cv")
    ax.set_title("A. Endpoint observer scores by edge-axis history")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    _configure_axes(ax)


def _plot_within_gates(ax: plt.Axes, contrasts: pd.DataFrame) -> None:
    block = contrasts[contrasts["comparison"].eq("within-primary gate")]
    labels = ["Joint - static", "Joint - zero", "Known - joint"]
    y = np.arange(len(labels), dtype=float)
    offsets = {"Edge-parallel": -0.16, "Edge-orthogonal": 0.16}
    markers = {"Edge-parallel": "o", "Edge-orthogonal": "s"}
    colors = {"Edge-parallel": COLORS["parallel"], "Edge-orthogonal": COLORS["orthogonal"]}
    for primary in PRIMARY_RUNS:
        sub = block[block["primary"].eq(primary)].set_index("contrast").loc[labels]
        for yi, (contrast, row) in enumerate(sub.iterrows()):
            value = float(row["value"])
            low = float(row["ci_low"])
            high = float(row["ci_high"])
            ax.errorbar(
                value,
                y[yi] + offsets[primary],
                xerr=[[value - low], [high - value]],
                marker=markers[primary],
                color=colors[primary],
                ecolor=colors[primary],
                capsize=2.5,
                lw=1.8,
                linestyle="none",
                label=primary if yi == 0 else None,
            )
    ax.axvline(0.0, color=COLORS["text"], lw=1.0)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Delta pooled R2_cv")
    ax.set_title("B. Within-axis history gates")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _configure_axes(ax, xgrid=True, ygrid=False)


def _plot_primary_deltas(ax: plt.Axes, contrasts: pd.DataFrame) -> None:
    block = contrasts[contrasts["comparison"].eq("orthogonal-minus-parallel")].copy()
    labels = block["contrast"].tolist()
    y = np.arange(len(labels), dtype=float)
    for yi, row in enumerate(block.itertuples(index=False)):
        value = float(row.value)
        low = float(row.ci_low)
        high = float(row.ci_high)
        color = COLORS["orthogonal"] if value > 0 else COLORS["parallel"]
        ax.errorbar(
            value,
            yi,
            xerr=[[value - low], [high - value]],
            marker="o",
            color=color,
            ecolor=color,
            capsize=2.5,
            lw=1.8,
            linestyle="none",
        )
        ax.text(high + 0.015, yi, f"{value:+.2f}", va="center", fontsize=8)
    ax.axvline(0.0, color=COLORS["text"], lw=1.0)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Edge-orthogonal minus edge-parallel R2_cv")
    ax.set_title("C. Direct primary-axis contrast")
    _configure_axes(ax, xgrid=True, ygrid=False)


def _plot_edge_geometry(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    conditions = list(PRIMARY_RUNS)
    x = np.arange(len(conditions), dtype=float)
    width = 0.34
    grouped = metrics.groupby("condition")
    along = grouped["edge_along_rms_deg"].mean().reindex(conditions)
    across = grouped["edge_across_rms_deg"].mean().reindex(conditions)
    along_sem = grouped["edge_along_rms_deg"].sem().reindex(conditions)
    across_sem = grouped["edge_across_rms_deg"].sem().reindex(conditions)
    ax.bar(
        x - width / 2,
        along.to_numpy(dtype=float),
        width=width,
        yerr=along_sem.to_numpy(dtype=float),
        capsize=2.5,
        color=COLORS["parallel"],
        label="along image edge",
    )
    ax.bar(
        x + width / 2,
        across.to_numpy(dtype=float),
        width=width,
        yerr=across_sem.to_numpy(dtype=float),
        capsize=2.5,
        color="#CC79A7",
        label="across image edge",
    )
    ax.set_xticks(x, conditions)
    ax.set_ylabel("RMS displacement (deg)")
    ax.set_title("D. Image-edge trajectory projections")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    _configure_axes(ax)


def plot_axis_comparison(out_dir: Path, *, n_bootstrap: int, seed: int) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = _summary_scores()
    contrasts = _contrast_table(n_bootstrap=n_bootstrap, seed=seed)
    metrics = _edge_projection_metrics()
    scores.to_csv(out_dir / "endpoint_history_axis_edge_scores.csv", index=False)
    contrasts.to_csv(out_dir / "endpoint_history_axis_edge_contrasts.csv", index=False)
    metrics.to_csv(out_dir / "endpoint_history_axis_edge_geometry.csv", index=False)

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
    fig, axes = plt.subplots(2, 2, figsize=(10.9, 7.5), constrained_layout=True)
    _plot_scores(axes[0, 0], scores)
    _plot_within_gates(axes[0, 1], contrasts)
    _plot_primary_deltas(axes[1, 0], contrasts)
    _plot_edge_geometry(axes[1, 1], metrics)
    fig.suptitle("Endpoint-history edge-axis controls", fontsize=13, fontweight="bold")
    png = out_dir / "endpoint_history_axis_edge_comparison.png"
    pdf = out_dir / "endpoint_history_axis_edge_comparison.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PARALLEL_RUN_DIR / "main_results_figures",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260706)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    png, pdf = plot_axis_comparison(args.out_dir, n_bootstrap=int(args.n_bootstrap), seed=int(args.seed))
    print(f"[endpoint-history-axis-edge] wrote {png}")
    print(f"[endpoint-history-axis-edge] wrote {pdf}")


if __name__ == "__main__":
    main()
