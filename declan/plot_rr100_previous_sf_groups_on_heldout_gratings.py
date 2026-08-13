#!/usr/bin/env python3
"""Test whether previous RR100 SF groups transfer to held-out grating tuning."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREVIOUS = ROOT / "outputs/fig/ssi_figure_v2/panels/previous_sf_tuning_groups/previous_sf_tuning_unit_summary.csv"
DEFAULT_COMMON_DIR = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_native_readout_common_sf_probe_v1/common_sf_comparison"
GROUPS = ["low_sf", "middle_sf", "high_sf"]
GROUP_LABELS = {"low_sf": "previous low", "middle_sf": "previous middle", "high_sf": "previous high"}
GROUP_COLORS = {"low_sf": "#0072B2", "middle_sf": "#559F76", "high_sf": "#D55E00"}
TARGETS = {
    "heldout_recorded": "heldout_recorded_preferred_sf_cpd",
    "heldout_fitted_twin": "heldout_fitted_twin_preferred_sf_cpd",
    "synthetic_native_twin": "synthetic_native_twin_preferred_sf_cpd",
}
SF_BINS = [1.0, 2.0, 4.0, 8.0, 16.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument("--common-dir", type=Path, default=DEFAULT_COMMON_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_COMMON_DIR / "previous_sf_group_transfer")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    stat = path.resolve().stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def prepare_join(previous_path: Path, units_path: Path) -> pd.DataFrame:
    previous = pd.read_csv(previous_path)
    units = pd.read_csv(units_path)
    keep = previous[
        [
            "unit_index",
            "unit_label",
            "sf_group",
            "sf_group_label",
            "dynamic_log_gaussian_marginal_sf_cpd",
            "dynamic_log_gaussian_marginal_r2",
            "dynamic_log_gaussian_marginal_fit_ok",
        ]
    ].rename(
        columns={
            "unit_index": "rr100_index",
            "unit_label": "previous_unit_label",
            "dynamic_log_gaussian_marginal_sf_cpd": "previous_synthetic_fit_preferred_sf_cpd",
            "dynamic_log_gaussian_marginal_r2": "previous_synthetic_fit_r2",
            "dynamic_log_gaussian_marginal_fit_ok": "previous_synthetic_fit_ok",
        }
    )
    joined = keep.merge(units, on="rr100_index", how="inner", validate="one_to_one")
    if len(joined) != 100 or joined["rr100_index"].nunique() != 100:
        raise ValueError("Expected a one-to-one join of all 100 RR100 units")
    joined["sf_group_order"] = joined["sf_group"].map({group: idx for idx, group in enumerate(GROUPS)})
    return joined


def group_summary(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target, col in TARGETS.items():
        for group in GROUPS:
            sub = joined[joined["sf_group"] == group]
            valid = pd.to_numeric(sub[col], errors="coerce").dropna()
            row: dict[str, object] = {
                "target": target,
                "sf_group": group,
                "sf_group_label": GROUP_LABELS[group],
                "n_group": int(len(sub)),
                "n_valid": int(len(valid)),
                "n_invalid": int(len(sub) - len(valid)),
                "invalid_fraction": float(1.0 - len(valid) / len(sub)),
                "median_preferred_sf_cpd": float(valid.median()) if len(valid) else np.nan,
                "mean_log2_preferred_sf": float(np.log2(valid).mean()) if len(valid) else np.nan,
            }
            for sf in SF_BINS:
                row[f"n_pref_{sf:g}_cpd"] = int(np.isclose(valid.to_numpy(dtype=float), sf).sum())
                row[f"fraction_valid_pref_{sf:g}_cpd"] = float(np.isclose(valid.to_numpy(dtype=float), sf).mean()) if len(valid) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def transfer_stats(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target, col in TARGETS.items():
        valid = joined[col].notna()
        sub = joined[valid].copy()
        x = sub["sf_group_order"].to_numpy(dtype=float)
        y = np.log2(sub[col].to_numpy(dtype=float))
        group_rho, group_p = spearmanr(x, y)
        fit_rho, fit_p = spearmanr(np.log2(sub["previous_synthetic_fit_preferred_sf_cpd"].to_numpy(dtype=float)), y)
        group_values = {
            group: np.log2(sub.loc[sub["sf_group"] == group, col].to_numpy(dtype=float))
            for group in GROUPS
        }
        kw = kruskal(*(group_values[group] for group in GROUPS))
        low = group_values["low_sf"]
        high = group_values["high_sf"]
        rows.append(
            {
                "target": target,
                "n_valid": int(len(sub)),
                "sf_group_order_spearman_rho": float(group_rho),
                "sf_group_order_spearman_p": float(group_p),
                "previous_continuous_fit_spearman_rho": float(fit_rho),
                "previous_continuous_fit_spearman_p": float(fit_p),
                "kruskal_wallis_h": float(kw.statistic),
                "kruskal_wallis_p": float(kw.pvalue),
                "low_group_median_sf_cpd": float(2 ** np.median(low)) if low.size else np.nan,
                "high_group_median_sf_cpd": float(2 ** np.median(high)) if high.size else np.nan,
                "high_minus_low_mean_log2_sf": float(np.mean(high) - np.mean(low)) if low.size and high.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def select_examples(joined: pd.DataFrame) -> pd.DataFrame:
    valid = joined[joined["heldout_recorded_preferred_sf_cpd"].notna()].copy()
    roles = [
        ("low_group_low_recorded", "low_sf", "min"),
        ("low_group_high_recorded_dissociation", "low_sf", "max"),
        ("high_group_low_recorded_dissociation", "high_sf", "min"),
        ("high_group_high_recorded", "high_sf", "max"),
    ]
    rows: list[pd.Series] = []
    used: set[int] = set()
    for role, group, direction in roles:
        candidates = valid[(valid["sf_group"] == group) & ~valid["rr100_index"].isin(used)].copy()
        extreme = candidates["heldout_recorded_preferred_sf_cpd"].min() if direction == "min" else candidates["heldout_recorded_preferred_sf_cpd"].max()
        candidates = candidates[np.isclose(candidates["heldout_recorded_preferred_sf_cpd"], extreme)].copy()
        candidates["selection_score"] = candidates[
            ["heldout_recorded_curve_strength", "heldout_fitted_twin_curve_strength"]
        ].mean(axis=1)
        chosen = candidates.sort_values(["selection_score", "rr100_index"], ascending=[False, True]).iloc[0].copy()
        chosen["selection_role"] = role
        chosen["selection_criterion"] = f"{group}; recorded preference {direction}; strongest mean held-out curve modulation at that extreme"
        rows.append(chosen)
        used.add(int(chosen["rr100_index"]))
    return pd.DataFrame(rows)


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return values
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    return (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values)


def plot_examples(out_dir: Path, selected: pd.DataFrame, curves: pd.DataFrame, dpi: int) -> Path:
    fig, axes = plt.subplots(len(selected), 1, figsize=(8.7, 2.3 * len(selected)), constrained_layout=True, squeeze=False)
    source_style = {
        "heldout_recorded": ("#222222", "recorded gratings"),
        "heldout_fitted_twin": ("#D55E00", "fitted twin on gratings"),
        "synthetic_native_twin": ("#0072B2", "native synthetic twin"),
    }
    for ax, (_, meta) in zip(axes[:, 0], selected.iterrows(), strict=True):
        unit = int(meta["rr100_index"])
        support = {float(v) for v in str(meta["common_support_sf_cpd"]).split(",")}
        for source, (color, label) in source_style.items():
            sub = curves[
                (curves["rr100_index"] == unit)
                & (curves["source"] == source)
                & curves["sf_cpd"].isin(support)
            ].sort_values("sf_cpd")
            ax.plot(sub["sf_cpd"], normalize(sub["value"].to_numpy(dtype=float)), marker="o", color=color, lw=1.8, label=label)
        ax.set_xscale("log", base=2)
        ticks = sorted(support)
        ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
        ax.set_ylim(-0.06, 1.06)
        ax.grid(alpha=0.18)
        ax.set_ylabel("normalized tuning")
        ax.set_title(
            f"u{unit:03d} · {meta['selection_role']} · old fit={float(meta['previous_synthetic_fit_preferred_sf_cpd']):.3g} cpd\n"
            f"recorded={float(meta['heldout_recorded_preferred_sf_cpd']):g}, fitted twin={float(meta['heldout_fitted_twin_preferred_sf_cpd']):g} cpd",
            loc="left", fontsize=9.3,
        )
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8, loc="upper center")
    axes[-1, 0].set_xlabel("spatial frequency (cpd; per-unit common support)")
    path = out_dir / "previous_sf_group_transfer_examples.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def distribution_matrix(joined: pd.DataFrame, col: str) -> np.ndarray:
    matrix = np.zeros((len(GROUPS), len(SF_BINS)), dtype=float)
    for row, group in enumerate(GROUPS):
        values = joined.loc[joined["sf_group"] == group, col].dropna().to_numpy(dtype=float)
        for column, sf in enumerate(SF_BINS):
            matrix[row, column] = np.isclose(values, sf).mean() if values.size else np.nan
    return matrix


def plot_population(out_dir: Path, joined: pd.DataFrame, summary: pd.DataFrame, stats: pd.DataFrame, dpi: int) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.7), constrained_layout=True)
    ax = axes[0, 0]
    rng = np.random.default_rng(13)
    for row, group in enumerate(GROUPS):
        values = joined.loc[joined["sf_group"] == group, "previous_synthetic_fit_preferred_sf_cpd"].to_numpy(dtype=float)
        ax.scatter(values, row + rng.uniform(-0.12, 0.12, len(values)), s=20, color=GROUP_COLORS[group], alpha=0.65)
        ax.scatter([np.median(values)], [row], marker="D", s=70, color=GROUP_COLORS[group], edgecolor="white", zorder=4)
    ax.set_xscale("log", base=2)
    ax.set_yticks(range(len(GROUPS)), [GROUP_LABELS[g] for g in GROUPS])
    ax.set_xlabel("previous synthetic fitted preferred SF (cpd)")
    ax.set_title("A  Original separation (defined by this fit)", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.18)

    for ax, target, title in [
        (axes[0, 1], "heldout_recorded", "B  Recorded gratings"),
        (axes[1, 0], "heldout_fitted_twin", "C  Fitted twin on gratings"),
    ]:
        matrix = distribution_matrix(joined, TARGETS[target])
        image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=max(0.5, np.nanmax(matrix)))
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                if np.isfinite(matrix[row, column]) and matrix[row, column] > 0:
                    ax.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if matrix[row, column] > 0.28 else "black")
        stat = stats[stats["target"] == target].iloc[0]
        ax.set_xticks(range(len(SF_BINS)), [f"{v:g}" for v in SF_BINS])
        ax.set_yticks(range(len(GROUPS)), [GROUP_LABELS[g] for g in GROUPS])
        ax.set_xlabel("preferred SF (cpd)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.text(0.03, 0.97, f"group-order ρ={stat.sf_group_order_spearman_rho:.2f}, p={stat.sf_group_order_spearman_p:.2g}\n"
                f"Kruskal–Wallis p={stat.kruskal_wallis_p:.2g}", transform=ax.transAxes, va="top", fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "none", "pad": 2})
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="fraction within previous group")

    ax = axes[1, 1]
    x = np.arange(len(GROUPS), dtype=float)
    width = 0.33
    for offset, target, color, label in [
        (-width / 2, "heldout_recorded", "#222222", "recorded"),
        (width / 2, "heldout_fitted_twin", "#D55E00", "fitted twin"),
    ]:
        sub = summary[summary["target"] == target].set_index("sf_group").reindex(GROUPS)
        ax.bar(x + offset, sub["mean_log2_preferred_sf"], width=width, color=color, label=label)
    ax.set_xticks(x, [GROUP_LABELS[g] for g in GROUPS], rotation=15, ha="right")
    ax.set_yticks(np.log2(SF_BINS), [f"{v:g}" for v in SF_BINS])
    ax.set_ylabel("mean preferred SF (cpd; log2 averaging)")
    ax.set_title("D  No monotonic low → high transfer", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)
    fig.suptitle("Do previous synthetic low/middle/high SF groups map onto gratings?", fontsize=12)
    path = out_dir / "previous_sf_groups_on_heldout_gratings.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    previous_path = args.previous.resolve()
    common_dir = args.common_dir.resolve()
    units_path = common_dir / "common_sf_unit_summary.csv"
    curves_path = common_dir / "common_sf_curves_long.csv"
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    joined = prepare_join(previous_path, units_path)
    curves = pd.read_csv(curves_path)
    summary = group_summary(joined)
    stats = transfer_stats(joined)
    selected = select_examples(joined)
    joined_path = out_dir / "previous_sf_group_transfer_unit_join.csv"
    summary_path = out_dir / "previous_sf_group_transfer_summary.csv"
    stats_path = out_dir / "previous_sf_group_transfer_stats.csv"
    selected_path = out_dir / "previous_sf_group_transfer_selected_examples.csv"
    joined.to_csv(joined_path, index=False)
    summary.to_csv(summary_path, index=False)
    stats.to_csv(stats_path, index=False)
    selected.to_csv(selected_path, index=False)
    example_figure = plot_examples(out_dir, selected, curves, int(args.dpi))
    population_figure = plot_population(out_dir, joined, summary, stats, int(args.dpi))
    manifest = {
        "analysis": "previous_rr100_sf_group_transfer_to_heldout_gratings",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Does the population separation into previous synthetic low/middle/high SF groups transfer to held-out grating preferences?",
        "previous_group_contract": "groups are thresholds on the earlier center-pixel synthetic dynamic log-Gaussian preferred SF",
        "heldout_contract": "per-unit common cycle-valid support; discrete preferred SF from recorded or fitted-twin mean response maximized over orientation",
        "sources": [file_identity(previous_path), file_identity(units_path), file_identity(curves_path)],
        "outputs": {
            "unit_join": str(joined_path), "group_summary": str(summary_path), "stats": str(stats_path),
            "selected_examples": str(selected_path), "example_figure": str(example_figure), "population_figure": str(population_figure),
        },
    }
    (out_dir / "previous_sf_group_transfer_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(stats.to_string(index=False))
    print("\nSelected examples:")
    print(selected[["selection_role", "rr100_index", "sf_group", "previous_synthetic_fit_preferred_sf_cpd", "heldout_recorded_preferred_sf_cpd"]].to_string(index=False))
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
