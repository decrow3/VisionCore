"""Build publication-style endpoint-history Figure 4 result and methods plates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_fdim4_hpc8_primary_beta_scale1_cached_v1"
)
DEFAULT_SWEEP_DIR = REPO_ROOT / "outputs/figure4_endpoint_history_feature_readout_rr100_n64_feature_dim_sweep_v1"
DEFAULT_DIMENSION_RUN_DIRS = [
    REPO_ROOT / "outputs/figure4_endpoint_history_feature_readout_rr100_n64_fdim2_scale1_endpoint_cache_v1",
    REPO_ROOT / "outputs/figure4_endpoint_history_feature_readout_rr100_n64_fdim4_scale1_endpoint_cached_v1",
    REPO_ROOT / "outputs/figure4_endpoint_history_feature_readout_rr100_n64_fdim8_scale1_endpoint_cached_v1",
    REPO_ROOT / "outputs/figure4_endpoint_history_feature_readout_rr100_n64_fdim16_scale1_endpoint_cached_v1",
    REPO_ROOT / "outputs/figure4_endpoint_history_feature_readout_rr100_n64_fdim32_scale1_endpoint_cached_v1",
]
OU_RUN_DIR = (
    REPO_ROOT
    / "outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_primary_ou_fdim4_hpc8_scale1_cached_v1"
)
BROWNIAN_RUN_DIR = (
    REPO_ROOT
    / "outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_primary_brownian_fdim4_hpc8_scale1_cached_v1"
)
EDGE_PARALLEL_RUN_DIR = (
    REPO_ROOT
    / "outputs/figure4_endpoint_history_feature_readout_rr100_n128_axis_parallel_orthogonal_fdim4_hpc8_scale1_v1"
)
EDGE_ORTHOGONAL_RUN_DIR = (
    REPO_ROOT
    / "outputs/figure4_endpoint_history_feature_readout_rr100_n128_axis_parallel_orthogonal_fdim4_hpc8_primary_orthogonal_scale1_cached_v1"
)


COLORS = {
    "ink": "#111827",
    "muted": "#64748b",
    "grid": "#dbe3ec",
    "hairline": "#94a3b8",
    "paper": "#ffffff",
    "known": "#0f172a",
    "joint": "#0f766e",
    "zero": "#c65f00",
    "response": "#2364aa",
    "static": "#6b7280",
    "ou": "#7c3aed",
    "brownian": "#db2777",
    "edge_parallel": "#0b79b7",
    "edge_orthogonal": "#c65f00",
    "positive": "#ecfdf5",
    "negative": "#f8fafc",
}

MODE_SPECS = [
    ("known_history_generative", "Known path", "known"),
    ("joint_history_generative", "Joint latent", "joint"),
    ("static_history", "Static", "static"),
    ("joint_history_response_only", "Response-only", "response"),
    ("zero_history_generative_on_motion", "Zero-history", "zero"),
]

FAMILY_RUNS = {
    "Empirical": DEFAULT_RUN_DIR,
    "OU": OU_RUN_DIR,
    "Brownian": BROWNIAN_RUN_DIR,
}


@dataclass(frozen=True)
class Contrast:
    label: str
    value: float
    low: float
    high: float
    color_key: str


def _as_repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["paper"],
            "axes.facecolor": COLORS["paper"],
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "axes.linewidth": 0.9,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "savefig.bbox": "tight",
            "savefig.facecolor": COLORS["paper"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _clean_axis(ax: plt.Axes, *, xgrid: bool = True, ygrid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3.5, width=0.8, color=COLORS["ink"])
    if xgrid:
        ax.grid(axis="x", color=COLORS["grid"], lw=0.8, alpha=0.9)
    if ygrid:
        ax.grid(axis="y", color=COLORS["grid"], lw=0.8, alpha=0.85)


def _panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(f"{label}. {title}", loc="left", fontweight="bold", color=COLORS["ink"], pad=8)


def _save(fig: plt.Figure, out_dir: Path, stem: str, *, dpi: int) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def _scale_block(table: pd.DataFrame, *, scale: str = "1.0") -> pd.DataFrame:
    block = table[table["observation_scale"].astype(str).eq(scale)].copy()
    if block.empty:
        block = table[table["observation_scale"].astype(str).eq("all")].copy()
    if block.empty:
        block = table.copy()
    return block


def _summary_rows(run_dir: Path) -> pd.DataFrame:
    return _scale_block(pd.read_csv(run_dir / "endpoint_history_feature_readout_summary.csv"))


def _mode_score(summary: pd.DataFrame, mode: str, metric: str = "R2_cv") -> float:
    row = summary[summary["observer_mode"].astype(str).eq(mode)]
    if row.empty:
        raise ValueError(f"Missing observer_mode={mode!r}")
    return float(row.iloc[0][metric])


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
    if block.empty:
        raise ValueError(f"Empty gate table: {path}")
    return block.iloc[0]


def _main_contrasts(run_dir: Path) -> list[Contrast]:
    zero_gate = _gate_row(run_dir / "gates_known_joint_zero_static" / "unified_feature_observer_gate_table.csv")
    response_gate = _gate_row(
        run_dir / "gates_known_joint_responseonly_static" / "unified_feature_observer_gate_table.csv"
    )
    return [
        Contrast(
            "Joint minus zero-history",
            float(zero_gate["joint_minus_zero"]),
            float(zero_gate["joint_minus_zero_ci_low"]),
            float(zero_gate["joint_minus_zero_ci_high"]),
            "joint",
        ),
        Contrast(
            "Known minus zero-history",
            float(zero_gate["known_minus_zero"]),
            float(zero_gate["known_minus_zero_ci_low"]),
            float(zero_gate["known_minus_zero_ci_high"]),
            "known",
        ),
        Contrast(
            "Joint minus response-only",
            float(response_gate["joint_minus_zero"]),
            float(response_gate["joint_minus_zero_ci_low"]),
            float(response_gate["joint_minus_zero_ci_high"]),
            "joint",
        ),
        Contrast(
            "Known minus response-only",
            float(response_gate["known_minus_zero"]),
            float(response_gate["known_minus_zero_ci_low"]),
            float(response_gate["known_minus_zero_ci_high"]),
            "known",
        ),
        Contrast(
            "Joint minus static",
            float(zero_gate["joint_minus_response"]),
            float(zero_gate["joint_minus_response_ci_low"]),
            float(zero_gate["joint_minus_response_ci_high"]),
            "joint",
        ),
        Contrast(
            "Known minus joint",
            float(zero_gate["known_minus_joint"]),
            float(zero_gate["known_minus_joint_ci_low"]),
            float(zero_gate["known_minus_joint_ci_high"]),
            "muted",
        ),
    ]


def _family_contrasts() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family, run_dir in FAMILY_RUNS.items():
        gate = _gate_row(run_dir / "gates_known_joint_zero_static" / "unified_feature_observer_gate_table.csv")
        for label, value_col, low_col, high_col, color_key in [
            (
                "Joint minus zero-history",
                "joint_minus_zero",
                "joint_minus_zero_ci_low",
                "joint_minus_zero_ci_high",
                "joint",
            ),
            (
                "Joint minus static",
                "joint_minus_response",
                "joint_minus_response_ci_low",
                "joint_minus_response_ci_high",
                "static",
            ),
        ]:
            rows.append(
                {
                    "family": family,
                    "contrast": label,
                    "value": float(gate[value_col]),
                    "ci_low": float(gate[low_col]),
                    "ci_high": float(gate[high_col]),
                    "color_key": color_key,
                }
            )
    return pd.DataFrame(rows)


def _pooled_r2(block: pd.DataFrame, mode: str) -> float:
    values = block[block["observer_mode"].astype(str).eq(mode)]
    if values.empty:
        return float("nan")
    sse = float(values["feature_sse"].sum())
    sst = float(values["feature_sst_train_baseline"].sum())
    return 1.0 - sse / sst


def _trial_mode_table(run_dir: Path, modes: list[str]) -> pd.DataFrame:
    trials = pd.read_csv(run_dir / "endpoint_history_feature_readout_trials.csv")
    return trials[
        trials["observation_scale"].astype(str).eq("1.0")
        & trials["observer_mode"].astype(str).isin(modes)
    ].copy()


def _bootstrap_delta_across_primaries(*, mode: str, n_bootstrap: int, seed: int) -> tuple[float, float, float]:
    parallel = _trial_mode_table(EDGE_PARALLEL_RUN_DIR, [mode])
    orthogonal = _trial_mode_table(EDGE_ORTHOGONAL_RUN_DIR, [mode])
    common = sorted(
        set(parallel["true_source_row"].astype(int).unique()).intersection(
            set(orthogonal["true_source_row"].astype(int).unique())
        )
    )
    if not common:
        raise ValueError("No common sources between edge-axis primary runs")
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


def _edge_axis_contrasts(n_bootstrap: int, seed: int) -> pd.DataFrame:
    cached = EDGE_PARALLEL_RUN_DIR / "main_results_figures/endpoint_history_axis_edge_contrasts.csv"
    if cached.exists():
        table = pd.read_csv(cached)
        block = table[table["comparison"].astype(str).eq("orthogonal-minus-parallel")].copy()
        if not block.empty:
            return block

    rows: list[dict[str, object]] = []
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


def _score_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mode, label, color_key in MODE_SPECS:
        rows.append(
            {
                "observer_mode": mode,
                "label": label,
                "color_key": color_key,
                "R2_cv": _mode_score(summary, mode, "R2_cv"),
                "mean_feature_cosine": _mode_score(summary, mode, "mean_feature_cosine"),
            }
        )
    return pd.DataFrame(rows)


def _feature_dim_from_dir(path: Path) -> int:
    import re

    match = re.search(r"fdim(\d+)", path.name)
    if match is None:
        raise ValueError(f"Could not infer feature dimension from {path}")
    return int(match.group(1))


def _dimension_summary_rows(run_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        feature_dim = _feature_dim_from_dir(run_dir)
        table = pd.read_csv(run_dir / "endpoint_history_feature_readout_summary.csv")
        block = _scale_block(table)
        for row in block.itertuples(index=False):
            rows.append(
                {
                    "feature_dim": int(feature_dim),
                    "observer_mode": str(row.observer_mode),
                    "R2_cv": float(row.R2_cv),
                    "feature_sse": float(row.feature_sse),
                    "feature_sst_train_baseline": float(row.feature_sst_train_baseline),
                    "mean_feature_cosine": float(row.mean_feature_cosine),
                }
            )
    return pd.DataFrame(rows)


def _dimension_r2_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot_table(index="feature_dim", columns="observer_mode", values="R2_cv", aggfunc="first")
    specs = [
        ("Joint - zero-history", "joint_history_generative", "zero_history_generative_on_motion", "joint"),
        ("Joint - static", "joint_history_generative", "static_history", "static"),
        ("Known - joint", "known_history_generative", "joint_history_generative", "known"),
    ]
    rows: list[dict[str, object]] = []
    for label, lhs, rhs, color_key in specs:
        if lhs not in pivot or rhs not in pivot:
            continue
        for feature_dim, value in (pivot[lhs] - pivot[rhs]).dropna().items():
            rows.append(
                {
                    "feature_dim": int(feature_dim),
                    "contrast": label,
                    "value": float(value),
                    "color_key": color_key,
                }
            )
    return pd.DataFrame(rows)


def _dimension_band_rows(summary: pd.DataFrame) -> pd.DataFrame:
    band_labels = {
        2: "PC 1-2",
        4: "PC 3-4",
        8: "PC 5-8",
        16: "PC 9-16",
        32: "PC 17-32",
    }
    rows: list[dict[str, object]] = []
    for observer_mode, block in summary.groupby("observer_mode"):
        block = block.sort_values("feature_dim")
        prev_dim = 0
        prev_sse = 0.0
        prev_sst = 0.0
        for row in block.itertuples(index=False):
            feature_dim = int(row.feature_dim)
            sse = float(row.feature_sse)
            sst = float(row.feature_sst_train_baseline)
            band_sse = sse - prev_sse
            band_sst = sst - prev_sst
            band_r2 = 1.0 - band_sse / band_sst if band_sst > 0 else np.nan
            rows.append(
                {
                    "observer_mode": str(observer_mode),
                    "feature_dim": feature_dim,
                    "band_start": int(prev_dim + 1),
                    "band_end": feature_dim,
                    "band": band_labels.get(feature_dim, f"PC {prev_dim + 1}-{feature_dim}"),
                    "band_sse": float(band_sse),
                    "band_sst": float(band_sst),
                    "R2_cv_band": float(band_r2),
                }
            )
            prev_dim = feature_dim
            prev_sse = sse
            prev_sst = sst
    return pd.DataFrame(rows)


def _dimension_band_contrasts(bands: pd.DataFrame) -> pd.DataFrame:
    pivot = bands.pivot_table(index=["feature_dim", "band"], columns="observer_mode", values="R2_cv_band", aggfunc="first")
    specs = [
        ("Joint - zero-history", "joint_history_generative", "zero_history_generative_on_motion", "joint"),
        ("Joint - static", "joint_history_generative", "static_history", "static"),
    ]
    rows: list[dict[str, object]] = []
    for label, lhs, rhs, color_key in specs:
        if lhs not in pivot or rhs not in pivot:
            continue
        values = (pivot[lhs] - pivot[rhs]).dropna()
        for (feature_dim, band), value in values.items():
            rows.append(
                {
                    "feature_dim": int(feature_dim),
                    "band": str(band),
                    "contrast": label,
                    "value": float(value),
                    "color_key": color_key,
                }
            )
    return pd.DataFrame(rows)


def _plot_scores(ax: plt.Axes, summary: pd.DataFrame) -> None:
    scores = _score_rows(summary).sort_values("R2_cv", ascending=True)
    y = np.arange(len(scores), dtype=float)
    values = scores["R2_cv"].to_numpy(dtype=float)
    colors = [COLORS[str(key)] for key in scores["color_key"]]
    ax.barh(y, values, color=colors, height=0.66, edgecolor="none")
    ax.axvline(0.0, color=COLORS["ink"], lw=0.9)
    ax.set_yticks(y, scores["label"].tolist())
    ax.set_xlim(min(-2.65, float(values.min()) - 0.16), 0.18)
    ax.set_xlabel("pooled R2_cv")
    for yi, value in zip(y, values):
        ax.text(-0.07, yi, f"{value:.2f}", ha="right", va="center", color="white", fontsize=8.2)
    _panel_label(ax, "A", "Endpoint feature recovery")
    _clean_axis(ax, xgrid=True, ygrid=False)


def _plot_main_contrasts(ax: plt.Axes, contrasts: list[Contrast]) -> None:
    y = np.arange(len(contrasts), dtype=float)
    ax.axvspan(0, 1.35, color=COLORS["positive"], zorder=0)
    for yi, contrast in zip(y, contrasts):
        value = contrast.value
        low = contrast.low
        high = contrast.high
        color = COLORS[contrast.color_key]
        ax.plot([low, high], [yi, yi], color=color, lw=2.2, solid_capstyle="round")
        ax.plot([low, low], [yi - 0.055, yi + 0.055], color=color, lw=1.5)
        ax.plot([high, high], [yi - 0.055, yi + 0.055], color=color, lw=1.5)
        ax.scatter([value], [yi], s=48, color=color, zorder=3)
        label_x = min(high + 0.055, 1.31)
        ax.text(label_x, yi, f"{value:+.2f}", ha="left", va="center", fontsize=8.4, color=COLORS["ink"])
    ax.axvline(0.0, color=COLORS["ink"], lw=0.95)
    ax.set_yticks(y, [item.label for item in contrasts])
    ax.invert_yaxis()
    ax.set_xlim(-0.12, 1.38)
    ax.set_xlabel("Delta pooled R2_cv")
    _panel_label(ax, "B", "Paired gate contrasts")
    ax.text(
        0.99,
        1.02,
        "source-bootstrap 95% CI",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=COLORS["muted"],
        fontsize=8,
    )
    _clean_axis(ax, xgrid=True, ygrid=False)


def _plot_feature_dim_sweep(ax: plt.Axes, sweep_dir: Path) -> None:
    table = pd.read_csv(sweep_dir / "endpoint_history_feature_dim_sweep_summary.csv")
    labels = [
        ("Joint latent", "joint_history_generative", "joint", "o", "-"),
        ("Response-only", "joint_history_response_only", "response", "D", ":"),
        ("Static", "static_history", "static", "^", "-."),
        ("Zero-history", "zero_history_generative_on_motion", "zero", "s", "--"),
    ]
    for label, mode, color_key, marker, linestyle in labels:
        block = table[table["observer_mode"].astype(str).eq(mode)].sort_values("feature_dim")
        if block.empty:
            continue
        ax.plot(
            block["feature_dim"].to_numpy(dtype=float),
            block["R2_cv"].to_numpy(dtype=float),
            marker=marker,
            ls=linestyle,
            color=COLORS[color_key],
            lw=2.1 if color_key == "joint" else 1.8,
            markersize=5.5,
            label=label,
        )
    ax.axhline(0.0, color=COLORS["hairline"], lw=0.9)
    ax.axvline(4, color=COLORS["ink"], lw=1.0, ls=(0, (3, 2)))
    ax.text(4.18, 0.92, "main dim", transform=ax.get_xaxis_transform(), fontsize=8, color=COLORS["ink"], va="top")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32], ["2", "4", "8", "16", "32"])
    ax.set_xlabel("feature dimension")
    ax.set_ylabel("pooled R2_cv")
    ax.set_ylim(-5.85, 0.25)
    ax.legend(frameon=False, ncol=2, loc="lower left", handlelength=2.2)
    _panel_label(ax, "C", "Feature-dimension screen")
    _clean_axis(ax, xgrid=True, ygrid=True)


def _plot_feature_dim_contrasts(ax: plt.Axes, sweep_dir: Path) -> None:
    summary = pd.read_csv(sweep_dir / "endpoint_history_feature_dim_sweep_summary.csv")
    contrasts = _dimension_r2_contrasts(
        summary.rename(
            columns={
                "R2_cv": "R2_cv",
            }
        )
    )
    specs = [
        ("Joint - zero-history", "joint", "o", "-"),
        ("Joint - static", "static", "s", "--"),
        ("Known - joint", "known", "D", ":"),
    ]
    for label, color_key, marker, linestyle in specs:
        block = contrasts[contrasts["contrast"].eq(label)].sort_values("feature_dim")
        if block.empty:
            continue
        ax.plot(
            block["feature_dim"].to_numpy(dtype=float),
            block["value"].to_numpy(dtype=float),
            marker=marker,
            ls=linestyle,
            color=COLORS[color_key],
            lw=2.0,
            markersize=5.5,
            label=label,
        )
    ax.axhline(0.0, color=COLORS["ink"], lw=0.9)
    ax.axvline(4, color=COLORS["ink"], lw=1.0, ls=(0, (3, 2)))
    ax.text(4.18, 0.95, "main dim", transform=ax.get_xaxis_transform(), fontsize=8, color=COLORS["ink"], va="top")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32], ["2", "4", "8", "16", "32"])
    ax.set_xlabel("cumulative feature dimension")
    ax.set_ylabel("Delta pooled R2_cv")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    _panel_label(ax, "C", "Dimension stress test")
    _clean_axis(ax, xgrid=True, ygrid=True)


def _plot_family_controls(ax: plt.Axes, family: pd.DataFrame) -> None:
    families = ["Empirical", "OU", "Brownian"]
    contrasts = ["Joint minus zero-history", "Joint minus static"]
    offsets = {"Joint minus zero-history": -0.13, "Joint minus static": 0.13}
    markers = {"Joint minus zero-history": "o", "Joint minus static": "s"}
    ybase = np.arange(len(families), dtype=float)
    for contrast in contrasts:
        block = family[family["contrast"].eq(contrast)].set_index("family").loc[families]
        color = COLORS["joint"] if "zero" in contrast else COLORS["static"]
        for family_name, row in block.iterrows():
            yi = ybase[families.index(family_name)] + offsets[contrast]
            value = float(row["value"])
            low = float(row["ci_low"])
            high = float(row["ci_high"])
            ax.plot([low, high], [yi, yi], color=color, lw=2.0, solid_capstyle="round")
            ax.scatter([value], [yi], color=color, marker=markers[contrast], s=38, zorder=3)
        ax.scatter([], [], color=color, marker=markers[contrast], s=38, label=contrast.replace("Joint minus ", "vs "))
    ax.axvline(0.0, color=COLORS["ink"], lw=0.9)
    ax.set_yticks(ybase, families)
    ax.invert_yaxis()
    ax.set_xlim(-0.12, 2.45)
    ax.set_xlabel("Delta pooled R2_cv")
    ax.legend(frameon=False, loc="lower right")
    _panel_label(ax, "D", "Primary history family controls")
    _clean_axis(ax, xgrid=True, ygrid=False)


def _plot_edge_axis_control(ax: plt.Axes, edge: pd.DataFrame) -> None:
    order = ["Joint", "Known", "Zero-history", "Response-only"]
    block = edge.set_index("contrast").loc[order].reset_index()
    y = np.arange(len(block), dtype=float)
    ax.axvspan(-0.08, 0.08, color=COLORS["negative"], zorder=0)
    for yi, row in enumerate(block.itertuples(index=False)):
        value = float(row.value)
        low = float(row.ci_low)
        high = float(row.ci_high)
        color = COLORS["edge_orthogonal"] if value > 0 else COLORS["edge_parallel"]
        ax.plot([low, high], [yi, yi], color=color, lw=2.0, solid_capstyle="round")
        ax.scatter([value], [yi], color=color, s=42, zorder=3)
        ax.text(high + 0.015, yi, f"{value:+.2f}", va="center", fontsize=8.2, color=COLORS["ink"])
    ax.axvline(0.0, color=COLORS["ink"], lw=0.95)
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlim(-0.45, 0.35)
    ax.set_xlabel("Edge-orthogonal minus edge-parallel R2_cv")
    _panel_label(ax, "E", "Edge-axis primary control")
    _clean_axis(ax, xgrid=True, ygrid=False)


def plot_results_plate(
    *,
    run_dir: Path,
    sweep_dir: Path,
    out_dir: Path,
    dpi: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[Path, Path]:
    _configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary_rows(run_dir)
    contrasts = _main_contrasts(run_dir)
    family = _family_contrasts()
    edge = _edge_axis_contrasts(n_bootstrap=n_bootstrap, seed=seed)

    values = _score_rows(summary)
    values.to_csv(out_dir / "endpoint_history_polished_observer_scores.csv", index=False)
    family.to_csv(out_dir / "endpoint_history_polished_family_contrasts.csv", index=False)
    edge.to_csv(out_dir / "endpoint_history_polished_edge_axis_contrasts.csv", index=False)
    pd.DataFrame([item.__dict__ for item in contrasts]).to_csv(
        out_dir / "endpoint_history_polished_main_contrasts.csv", index=False
    )

    fig = plt.figure(figsize=(13.5, 8.9), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        3,
        left=0.075,
        right=0.99,
        bottom=0.08,
        top=0.86,
        wspace=0.55,
        hspace=0.5,
        width_ratios=[1.0, 1.25, 1.08],
        height_ratios=[1.08, 1.0],
    )
    ax_scores = fig.add_subplot(gs[0, 0])
    ax_gates = fig.add_subplot(gs[0, 1:])
    ax_sweep = fig.add_subplot(gs[1, 0])
    ax_family = fig.add_subplot(gs[1, 1])
    ax_edge = fig.add_subplot(gs[1, 2])

    _plot_scores(ax_scores, summary)
    _plot_main_contrasts(ax_gates, contrasts)
    _plot_feature_dim_contrasts(ax_sweep, sweep_dir)
    _plot_family_controls(ax_family, family)
    _plot_edge_axis_control(ax_edge, edge)

    fig.text(0.025, 0.965, "Endpoint-aligned history readout", ha="left", va="top", fontsize=17, fontweight="bold")
    fig.text(
        0.025,
        0.925,
        "Matched terminal view, terminal response readout, source-paired observer gates",
        ha="left",
        va="top",
        fontsize=9.5,
        color=COLORS["muted"],
    )
    return _save(fig, out_dir, "endpoint_history_figure4_results_polished", dpi=dpi)


def _trajectory_arrays(run_dir: Path) -> tuple[np.lib.npyio.NpzFile, pd.DataFrame, int]:
    arrays = np.load(run_dir / "endpoint_history_dataset_arrays.npz")
    metrics = pd.read_csv(run_dir / "endpoint_history_trace_metrics.csv")
    empirical = metrics[metrics["condition"].astype(str).eq("empirical_endpoint_history")].copy()
    median_path = float(empirical["history_path_length_deg"].median())
    empirical["distance_to_median"] = (empirical["history_path_length_deg"] - median_path).abs()
    sample_index = int(empirical.sort_values("distance_to_median").iloc[0]["sample_index"])
    return arrays, metrics, sample_index


def _trace_from_arrays(arrays: np.lib.npyio.NpzFile, key: str, sample_index: int) -> np.ndarray:
    tau = np.asarray(arrays[key][sample_index], dtype=float)
    history = tau.reshape(tau.shape[0] // 2, 2)
    return np.concatenate([history, np.zeros((1, 2), dtype=float)], axis=0)


def _plot_methods_endpoint_alignment(ax: plt.Axes, run_dir: Path) -> int:
    arrays, metrics, sample_index = _trajectory_arrays(run_dir)
    specs = [
        ("static_endpoint_history", "Static", "static"),
        ("empirical_endpoint_history", "Empirical", "joint"),
        ("ou_endpoint_history", "OU", "ou"),
        ("brownian_endpoint_history", "Brownian", "brownian"),
    ]
    max_radius = 0.0
    for condition, label, color_key in specs:
        trace = _trace_from_arrays(arrays, f"tau__{condition}", sample_index)
        max_radius = max(max_radius, float(np.max(np.abs(trace))))
        color = COLORS[color_key]
        if condition == "static_endpoint_history":
            ax.scatter([0], [0], marker="x", s=54, color=color, lw=2.0, label=label, zorder=5)
            continue
        ax.plot(trace[:, 0], trace[:, 1], color=color, lw=1.8, alpha=0.95, label=label)
        ax.scatter(trace[0, 0], trace[0, 1], s=24, facecolor=COLORS["paper"], edgecolor=color, lw=1.4, zorder=4)
    ax.scatter([0], [0], s=72, color=COLORS["ink"], marker="*", zorder=6)
    ax.text(0.0, 0.0, " endpoint", ha="left", va="bottom", fontsize=8, color=COLORS["ink"])
    pad = max(0.08, max_radius * 1.25)
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("horizontal displacement (deg)")
    ax.set_ylabel("vertical displacement (deg)")
    selected = metrics[metrics["sample_index"].astype(int).eq(sample_index)]
    max_endpoint = float(selected["endpoint_norm_deg"].max())
    ax.text(
        0.02,
        0.94,
        f"sample {sample_index}; terminal error {max_endpoint:.1e} deg",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        color=COLORS["muted"],
    )
    ax.legend(frameon=False, loc="lower right")
    _panel_label(ax, "1", "Endpoint-align each history")
    _clean_axis(ax, xgrid=True, ygrid=True)
    return sample_index


def _plot_methods_response_matrix(ax: plt.Axes, run_dir: Path, sample_index: int) -> None:
    arrays = np.load(run_dir / "endpoint_history_dataset_arrays.npz")
    conditions = [
        ("static_endpoint_history", "Static"),
        ("empirical_endpoint_history", "Empirical"),
        ("ou_endpoint_history", "OU"),
        ("brownian_endpoint_history", "Brownian"),
    ]
    vectors = []
    for condition, _ in conditions:
        vectors.append(np.asarray(arrays[f"x__{condition}"][sample_index], dtype=float))
    mat = np.vstack(vectors)
    mat = (mat - mat.mean(axis=1, keepdims=True)) / (mat.std(axis=1, keepdims=True) + 1e-12)
    vmax = float(np.quantile(np.abs(mat), 0.98))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(np.arange(len(conditions)), [label for _, label in conditions])
    ax.set_xticks([0, 24, 49, 74, 99], ["1", "25", "50", "75", "100"])
    ax.set_xlabel("terminal response unit")
    ax.set_ylabel("matched source")
    ax.text(0.99, 0.98, "z-scored rows", transform=ax.transAxes, ha="right", va="top", fontsize=7.8, color="white")
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.outline.set_linewidth(0.6)
    _panel_label(ax, "2", "Read only the terminal response")
    _clean_axis(ax, xgrid=False, ygrid=False)


def _draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str | None = None,
    textcolor: str = COLORS["ink"],
    fontsize: float = 8.0,
) -> None:
    edge = edgecolor if edgecolor is not None else facecolor
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edge,
        linewidth=1.0,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        linespacing=1.15,
    )


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = COLORS["muted"]) -> None:
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, linewidth=1.1, color=color)
    ax.add_patch(patch)


def _plot_methods_observer_ladder(ax: plt.Axes, summary: pd.DataFrame) -> None:
    ax.set_axis_off()
    rows = [
        ("Static", "r_T static", "static"),
        ("Response-only", "r_T motion", "response"),
        ("Zero-history", "r_T motion\n+ tau = 0", "zero"),
        ("Joint latent", "r_T motion\n+ latent tau", "joint"),
        ("Known path", "r_T motion\n+ true tau", "known"),
    ]
    y_positions = np.linspace(0.82, 0.16, len(rows))
    for (label, input_text, color_key), y in zip(rows, y_positions):
        color = COLORS[color_key]
        pale = "#f8fafc" if color_key in {"static", "known"} else "#eefcf8"
        if color_key == "zero":
            pale = "#fff7ed"
        if color_key == "response":
            pale = "#eff6ff"
        _draw_box(ax, (0.02, y - 0.045), 0.19, 0.09, label, facecolor=color, textcolor="white", fontsize=7.7)
        _draw_box(ax, (0.29, y - 0.045), 0.22, 0.09, input_text, facecolor=pale, edgecolor=color, fontsize=7.6)
        _draw_box(ax, (0.63, y - 0.045), 0.16, 0.09, "linear\nobserver", facecolor="#f8fafc", edgecolor=COLORS["hairline"], fontsize=7.3)
        mode = next((mode for mode, spec_label, _ in MODE_SPECS if spec_label == label), None)
        score = _mode_score(summary, mode) if mode is not None else float("nan")
        _draw_box(
            ax,
            (0.86, y - 0.045),
            0.11,
            0.09,
            f"{score:.2f}",
            facecolor="#ffffff",
            edgecolor=color,
            fontsize=7.8,
        )
        _arrow(ax, (0.21, y), (0.29, y), color=color)
        _arrow(ax, (0.51, y), (0.63, y), color=color)
        _arrow(ax, (0.79, y), (0.86, y), color=color)
    ax.text(0.29, 0.95, "input contract", ha="center", va="center", fontsize=8, color=COLORS["muted"])
    ax.text(0.915, 0.95, "R2_cv", ha="center", va="center", fontsize=8, color=COLORS["muted"])
    ax.text(0.02, 1.02, "3. Observer ladder", ha="left", va="bottom", fontsize=10.5, fontweight="bold", color=COLORS["ink"])


def _plot_methods_gate(ax: plt.Axes, contrasts: list[Contrast]) -> None:
    selected = [
        item
        for item in contrasts
        if item.label
        in {
            "Joint minus zero-history",
            "Joint minus response-only",
            "Joint minus static",
            "Known minus joint",
        }
    ]
    y = np.arange(len(selected), dtype=float)
    ax.axvspan(0, 1.25, color=COLORS["positive"], zorder=0)
    for yi, contrast in zip(y, selected):
        color = COLORS[contrast.color_key]
        ax.plot([contrast.low, contrast.high], [yi, yi], color=color, lw=2.0, solid_capstyle="round")
        ax.scatter([contrast.value], [yi], color=color, s=42, zorder=3)
        ax.text(contrast.high + 0.035, yi, f"{contrast.value:+.2f}", va="center", fontsize=8, color=COLORS["ink"])
    ax.axvline(0, color=COLORS["ink"], lw=0.9)
    ax.set_yticks(y, [item.label.replace("Joint minus ", "Joint vs ").replace("Known minus ", "Known vs ") for item in selected])
    ax.invert_yaxis()
    ax.set_xlim(-0.12, 1.32)
    ax.set_xlabel("paired Delta R2_cv")
    _panel_label(ax, "4", "Gate the history contribution")
    _clean_axis(ax, xgrid=True, ygrid=False)


def plot_methods_plate(*, run_dir: Path, out_dir: Path, dpi: int) -> tuple[Path, Path]:
    _configure_matplotlib()
    summary = _summary_rows(run_dir)
    contrasts = _main_contrasts(run_dir)
    fig = plt.figure(figsize=(12.6, 8.5), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.97,
        bottom=0.08,
        top=0.86,
        wspace=0.33,
        hspace=0.5,
        width_ratios=[1.02, 1.0],
        height_ratios=[1.0, 1.0],
    )
    ax_trace = fig.add_subplot(gs[0, 0])
    sample_index = _plot_methods_endpoint_alignment(ax_trace, run_dir)
    ax_matrix = fig.add_subplot(gs[0, 1])
    _plot_methods_response_matrix(ax_matrix, run_dir, sample_index)
    ax_ladder = fig.add_subplot(gs[1, 0])
    _plot_methods_observer_ladder(ax_ladder, summary)
    ax_gate = fig.add_subplot(gs[1, 1])
    _plot_methods_gate(ax_gate, contrasts)
    fig.text(
        0.025,
        0.965,
        "Endpoint-history readout: incremental analysis steps",
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.025,
        0.925,
        "Each source has the same terminal view; only the preceding trajectory and observer access change.",
        ha="left",
        va="top",
        fontsize=9.3,
        color=COLORS["muted"],
    )
    return _save(fig, out_dir, "endpoint_history_incremental_methods_polished", dpi=dpi)


def _plot_dimension_absolute(ax: plt.Axes, summary: pd.DataFrame) -> None:
    labels = [
        ("Joint", "joint_history_generative", "joint", "o", "-"),
        ("Static", "static_history", "static", "^", "-."),
        ("Zero-history", "zero_history_generative_on_motion", "zero", "s", "--"),
        ("Response-only", "joint_history_response_only", "response", "D", ":"),
    ]
    for label, mode, color_key, marker, linestyle in labels:
        block = summary[summary["observer_mode"].eq(mode)].sort_values("feature_dim")
        if block.empty:
            continue
        ax.plot(
            block["feature_dim"].to_numpy(dtype=float),
            block["R2_cv"].to_numpy(dtype=float),
            marker=marker,
            ls=linestyle,
            color=COLORS[color_key],
            lw=2.0,
            markersize=5.5,
            label=label,
        )
    ax.axhline(0.0, color=COLORS["hairline"], lw=0.9)
    ax.axvline(4, color=COLORS["ink"], lw=1.0, ls=(0, (3, 2)))
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32], ["2", "4", "8", "16", "32"])
    ax.set_xlabel("cumulative feature dimension")
    ax.set_ylabel("pooled R2_cv")
    ax.legend(frameon=False, ncol=2, loc="lower left")
    _panel_label(ax, "A", "Cumulative score")
    _clean_axis(ax, xgrid=True, ygrid=True)


def _plot_dimension_contrast_panel(ax: plt.Axes, contrasts: pd.DataFrame, *, panel: str) -> None:
    specs = [
        ("Joint - zero-history", "joint", "o", "-"),
        ("Joint - static", "static", "s", "--"),
        ("Known - joint", "known", "D", ":"),
    ]
    for label, color_key, marker, linestyle in specs:
        block = contrasts[contrasts["contrast"].eq(label)].sort_values("feature_dim")
        if block.empty:
            continue
        ax.plot(
            block["feature_dim"].to_numpy(dtype=float),
            block["value"].to_numpy(dtype=float),
            marker=marker,
            ls=linestyle,
            color=COLORS[color_key],
            lw=2.0,
            markersize=5.5,
            label=label,
        )
    ax.axhline(0.0, color=COLORS["ink"], lw=0.9)
    ax.axvline(4, color=COLORS["ink"], lw=1.0, ls=(0, (3, 2)))
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32], ["2", "4", "8", "16", "32"])
    ax.set_xlabel("cumulative feature dimension")
    ax.set_ylabel("Delta pooled R2_cv")
    ax.legend(frameon=False, loc="best")
    _panel_label(ax, panel, "Cumulative contrasts")
    _clean_axis(ax, xgrid=True, ygrid=True)


def _plot_dimension_band_r2(ax: plt.Axes, bands: pd.DataFrame) -> None:
    labels = [
        ("Joint", "joint_history_generative", "joint", "o", "-"),
        ("Static", "static_history", "static", "^", "-."),
        ("Zero-history", "zero_history_generative_on_motion", "zero", "s", "--"),
    ]
    band_order = ["PC 1-2", "PC 3-4", "PC 5-8", "PC 9-16", "PC 17-32"]
    x_lookup = {band: idx for idx, band in enumerate(band_order)}
    for label, mode, color_key, marker, linestyle in labels:
        block = bands[bands["observer_mode"].eq(mode)].copy()
        if block.empty:
            continue
        block["x"] = block["band"].map(x_lookup)
        block = block.dropna(subset=["x"]).sort_values("x")
        ax.plot(
            block["x"].to_numpy(dtype=float),
            block["R2_cv_band"].to_numpy(dtype=float),
            marker=marker,
            ls=linestyle,
            color=COLORS[color_key],
            lw=2.0,
            markersize=5.5,
            label=label,
        )
    ax.axhline(0.0, color=COLORS["hairline"], lw=0.9)
    ax.set_xticks(np.arange(len(band_order)), band_order, rotation=18, ha="right")
    ax.set_ylabel("band pooled R2_cv")
    ax.legend(frameon=False, loc="lower left")
    _panel_label(ax, "C", "Incremental PC-band score")
    _clean_axis(ax, xgrid=False, ygrid=True)


def _plot_dimension_band_contrasts(ax: plt.Axes, contrasts: pd.DataFrame) -> None:
    band_order = ["PC 1-2", "PC 3-4", "PC 5-8", "PC 9-16", "PC 17-32"]
    x_lookup = {band: idx for idx, band in enumerate(band_order)}
    specs = [
        ("Joint - zero-history", "joint", "o", "-"),
        ("Joint - static", "static", "s", "--"),
    ]
    for label, color_key, marker, linestyle in specs:
        block = contrasts[contrasts["contrast"].eq(label)].copy()
        if block.empty:
            continue
        block["x"] = block["band"].map(x_lookup)
        block = block.dropna(subset=["x"]).sort_values("x")
        ax.plot(
            block["x"].to_numpy(dtype=float),
            block["value"].to_numpy(dtype=float),
            marker=marker,
            ls=linestyle,
            color=COLORS[color_key],
            lw=2.0,
            markersize=5.5,
            label=label,
        )
    ax.axhline(0.0, color=COLORS["ink"], lw=0.9)
    ax.set_xticks(np.arange(len(band_order)), band_order, rotation=18, ha="right")
    ax.set_ylabel("Delta band R2_cv")
    ax.legend(frameon=False, loc="best")
    _panel_label(ax, "D", "Incremental PC-band contrasts")
    _clean_axis(ax, xgrid=False, ygrid=True)


def plot_dimension_stress_plate(*, dimension_run_dirs: list[Path], out_dir: Path, dpi: int) -> tuple[Path, Path]:
    _configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _dimension_summary_rows(dimension_run_dirs)
    contrasts = _dimension_r2_contrasts(summary)
    bands = _dimension_band_rows(summary)
    band_contrasts = _dimension_band_contrasts(bands)
    summary.to_csv(out_dir / "endpoint_history_dimension_cumulative_scores.csv", index=False)
    contrasts.to_csv(out_dir / "endpoint_history_dimension_cumulative_r2_contrasts.csv", index=False)
    bands.to_csv(out_dir / "endpoint_history_dimension_pc_band_scores.csv", index=False)
    band_contrasts.to_csv(out_dir / "endpoint_history_dimension_pc_band_contrasts.csv", index=False)

    fig = plt.figure(figsize=(12.4, 8.2), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.985,
        bottom=0.09,
        top=0.84,
        wspace=0.32,
        hspace=0.52,
    )
    _plot_dimension_absolute(fig.add_subplot(gs[0, 0]), summary)
    _plot_dimension_contrast_panel(fig.add_subplot(gs[0, 1]), contrasts, panel="B")
    _plot_dimension_band_r2(fig.add_subplot(gs[1, 0]), bands)
    _plot_dimension_band_contrasts(fig.add_subplot(gs[1, 1]), band_contrasts)
    fig.text(0.025, 0.96, "Endpoint-history feature dimension stress test", ha="left", va="top", fontsize=16, fontweight="bold")
    fig.text(
        0.025,
        0.918,
        "Cumulative scores can hide band-specific behavior; PC-band panels difference nested SSE/SST totals.",
        ha="left",
        va="top",
        fontsize=9.3,
        color=COLORS["muted"],
    )
    return _save(fig, out_dir, "endpoint_history_dimension_stress_polished", dpi=dpi)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--dimension-run-dirs", nargs="+", type=Path, default=DEFAULT_DIMENSION_RUN_DIRS)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260706)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir.resolve()
    sweep_dir = args.sweep_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else run_dir / "main_results_figures/polished"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_png, result_pdf = plot_results_plate(
        run_dir=run_dir,
        sweep_dir=sweep_dir,
        out_dir=out_dir,
        dpi=int(args.dpi),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    methods_png, methods_pdf = plot_methods_plate(run_dir=run_dir, out_dir=out_dir, dpi=int(args.dpi))
    dimension_png, dimension_pdf = plot_dimension_stress_plate(
        dimension_run_dirs=[path.resolve() for path in args.dimension_run_dirs],
        out_dir=out_dir,
        dpi=int(args.dpi),
    )
    for path in [result_png, result_pdf, methods_png, methods_pdf, dimension_png, dimension_pdf]:
        print(f"[endpoint-history-polished] wrote {_as_repo_relative(path)}")


if __name__ == "__main__":
    main()
