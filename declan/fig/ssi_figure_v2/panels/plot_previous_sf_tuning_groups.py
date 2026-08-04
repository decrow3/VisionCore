#!/usr/bin/env python3
"""Plot the earlier SF tuning curves used to define low/middle/high SF groups."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TUNING_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1"
)
DEFAULT_UNIT_GROUPS_CSV = DEFAULT_TUNING_DIR / (
    "sf_group_ssi_modulation_dynamic_log_gaussian_marginal_threshold_low0p05_high0p5_v1/"
    "dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv"
)
DEFAULT_BIMODAL_GROUPS_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_microsaccade_snippet_rr100_spatial_ssi_isotropic_padded_event_scaled_full_amp1sd_n40_v1/"
    "bimodal_unit_curve_groups/bimodal_unit_curve_groups.csv"
)
DEFAULT_OUT_DIR = ROOT / "outputs/fig/ssi_figure_v2/panels/previous_sf_tuning_groups"

GROUP_ORDER = ("low_sf", "middle_sf", "high_sf")
GROUP_LABELS = {"low_sf": "low SF", "middle_sf": "middle SF", "high_sf": "high SF"}
GROUP_COLORS = {"low_sf": "#0072B2", "middle_sf": "#559F76", "high_sf": "#D55E00"}
FINAL_GROUP_ORDER = ("low_mid_sf", "high_sf")
FINAL_GROUP_LABELS = {"low_mid_sf": "low+middle SF", "high_sf": "high SF"}
FINAL_GROUP_COLORS = {"low_mid_sf": "#0072B2", "high_sf": "#D55E00"}
SF_TICKS = (0.0125, 0.05, 0.2, 0.8, 3.2, 12.8)
LOW_THRESHOLD_CPD = 0.05
HIGH_THRESHOLD_CPD = 0.5


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sem(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return 0.0
    return float(np.std(vals, ddof=1) / math.sqrt(float(vals.size)))


def normalize_curve(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return vals * np.nan
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros_like(vals, dtype=np.float64)
    return (vals - lo) / (hi - lo)


def log_gaussian_curve(log2_sf: np.ndarray, baseline: float, amplitude: float, mu: float, sigma_oct: float) -> np.ndarray:
    sigma = max(float(sigma_oct), 1e-6)
    return float(baseline) + float(amplitude) * np.exp(-0.5 * ((np.asarray(log2_sf, dtype=float) - float(mu)) / sigma) ** 2)


def load_previous_tuning(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tuning_dir = Path(args.tuning_dir)
    grouped = pd.read_csv(tuning_dir / "frequency_tuning_grouped.csv")
    units = pd.read_csv(args.unit_groups_csv)
    grouped["temporal_hz"] = pd.to_numeric(grouped["temporal_hz"], errors="coerce")
    grouped["spatial_cpd"] = pd.to_numeric(grouped["spatial_cpd"], errors="coerce")
    grouped["response_amp_rms"] = pd.to_numeric(grouped["response_amp_rms"], errors="coerce").clip(lower=0.0)
    dynamic = grouped[(grouped["temporal_hz"] > 0.0) & (grouped["spatial_cpd"] > 0.0)].copy()
    curves = (
        dynamic.groupby(["unit_index", "unit_label", "spatial_cpd"], sort=True)["response_amp_rms"]
        .mean()
        .reset_index()
        .rename(columns={"response_amp_rms": "dynamic_marginal_response_amp_rms"})
    )
    curves = curves.merge(
        units[
            [
                "unit_index",
                "unit_label",
                "sf_group",
                "sf_group_label",
                "sf_split_metric",
                "sf_rank_low_to_high",
                "dynamic_log_gaussian_marginal_sf_cpd",
                "dynamic_log_gaussian_marginal_log2_sf",
                "dynamic_log_gaussian_marginal_baseline",
                "dynamic_log_gaussian_marginal_amplitude",
                "dynamic_log_gaussian_marginal_sigma_octaves",
                "dynamic_log_gaussian_marginal_fwhm_octaves",
                "dynamic_log_gaussian_marginal_r2",
                "dynamic_log_gaussian_marginal_fit_ok",
                "dynamic_log_gaussian_marginal_low_subcycle_amp_share",
            ]
        ],
        on=["unit_index", "unit_label"],
        how="inner",
        validate="many_to_one",
    )
    curves["final_sf_group"] = np.where(curves["sf_group"].isin(["low_sf", "middle_sf"]), "low_mid_sf", "high_sf")
    norm_parts: list[pd.DataFrame] = []
    for unit_index, sub in curves.groupby("unit_index", sort=False):
        out = sub.copy()
        out["dynamic_marginal_response_norm"] = normalize_curve(out["dynamic_marginal_response_amp_rms"].to_numpy(dtype=float))
        norm_parts.append(out)
    curves = pd.concat(norm_parts, ignore_index=True)

    retained = pd.DataFrame(columns=["unit_index", "retained_in_microsaccade_curve_group", "curve_group"])
    if Path(args.bimodal_groups_csv).exists():
        bimodal = pd.read_csv(args.bimodal_groups_csv)
        retained = bimodal[["unit_index", "curve_group"]].copy()
        retained["retained_in_microsaccade_curve_group"] = True
    units = units.merge(retained, on="unit_index", how="left")
    units["retained_in_microsaccade_curve_group"] = units["retained_in_microsaccade_curve_group"].fillna(False).astype(bool)
    units["curve_group"] = units["curve_group"].fillna("not_retained")
    units["final_sf_group"] = np.where(units["sf_group"].isin(["low_sf", "middle_sf"]), "low_mid_sf", "high_sf")
    curves = curves.merge(
        units[["unit_index", "retained_in_microsaccade_curve_group", "curve_group"]],
        on="unit_index",
        how="left",
        validate="many_to_one",
    )
    meta: dict[str, Any] = {}
    identity_path = tuning_dir / "frequency_tuning_request_identity.json"
    if identity_path.exists():
        import json

        meta = json.loads(identity_path.read_text(encoding="utf-8"))
    return curves, units, meta


def summarize_group_curves(curves: pd.DataFrame, group_col: str, order: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (group_id, sf), sub in curves.groupby([group_col, "spatial_cpd"], sort=True):
        vals = sub["dynamic_marginal_response_norm"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "grouping": group_col,
                "group": str(group_id),
                "spatial_cpd": float(sf),
                "mean_norm_response": float(np.nanmean(vals)),
                "sem_norm_response": sem(vals),
                "n_units": int(sub["unit_index"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    out["_order"] = out["group"].map({group: idx for idx, group in enumerate(order)}).fillna(99)
    return out.sort_values(["_order", "spatial_cpd"]).drop(columns="_order").reset_index(drop=True)


def plot_group_curve_panel(
    ax: plt.Axes,
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    group_col: str,
    order: tuple[str, ...],
    labels: dict[str, str],
    colors: dict[str, str],
    title: str,
    retained_only: bool,
) -> None:
    use = curves[curves["retained_in_microsaccade_curve_group"].astype(bool)].copy() if retained_only else curves.copy()
    summ = summary.copy()
    if retained_only:
        summ = summarize_group_curves(use, group_col, order)
    for group_id in order:
        sub = use[use[group_col].eq(group_id)].copy()
        for _unit, unit_curve in sub.groupby("unit_index", sort=False):
            unit_curve = unit_curve.sort_values("spatial_cpd")
            ax.plot(
                unit_curve["spatial_cpd"],
                unit_curve["dynamic_marginal_response_norm"],
                color=colors[group_id],
                alpha=0.13,
                linewidth=0.75,
            )
        mean_sub = summ[summ["group"].eq(group_id)].sort_values("spatial_cpd")
        if mean_sub.empty:
            continue
        x = mean_sub["spatial_cpd"].to_numpy(dtype=float)
        y = mean_sub["mean_norm_response"].to_numpy(dtype=float)
        e = mean_sub["sem_norm_response"].to_numpy(dtype=float)
        ax.plot(x, y, color=colors[group_id], linewidth=2.4, marker="o", markersize=4, label=f"{labels[group_id]} (n={int(mean_sub['n_units'].max())})")
        ax.fill_between(x, y - e, y + e, color=colors[group_id], alpha=0.16, linewidth=0)
    add_frequency_guides(ax)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("probe spatial frequency (cycles/deg)")
    ax.set_ylabel("within-unit normalized dynamic response")
    ax.legend(frameon=False, fontsize=7)


def add_frequency_guides(ax: plt.Axes) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.0105, 15.4)
    ax.set_xticks(SF_TICKS, [f"{v:g}" for v in SF_TICKS])
    ax.axvline(LOW_THRESHOLD_CPD, color="0.35", linestyle=":", linewidth=1.0)
    ax.axvline(HIGH_THRESHOLD_CPD, color="0.35", linestyle="--", linewidth=1.0)
    ax.axvline(0.37133431851485155, color="0.55", linestyle="-.", linewidth=0.9)
    ax.grid(True, color="0.9", linewidth=0.65)
    ax.spines[["top", "right"]].set_visible(False)


def plot_preference_distribution(ax: plt.Axes, units: pd.DataFrame) -> None:
    rng = np.random.default_rng(7)
    y_base = {"low_sf": 0.1, "middle_sf": 0.2, "high_sf": 0.3}
    for group_id in GROUP_ORDER:
        sub = units[units["sf_group"].eq(group_id)].copy()
        x = pd.to_numeric(sub["sf_split_metric"], errors="coerce").to_numpy(dtype=float)
        y = np.full(x.shape, y_base[group_id], dtype=float) + rng.uniform(-0.028, 0.028, size=x.shape)
        retained = sub["retained_in_microsaccade_curve_group"].astype(bool).to_numpy()
        ax.scatter(x[retained], y[retained], color=GROUP_COLORS[group_id], s=32, alpha=0.86, edgecolor="white", linewidth=0.35)
        ax.scatter(x[~retained], y[~retained], facecolor="none", edgecolor=GROUP_COLORS[group_id], s=48, linewidth=1.1)
    add_frequency_guides(ax)
    ax.set_ylim(0.0, 0.4)
    ax.set_yticks([y_base[g] for g in GROUP_ORDER], [GROUP_LABELS[g] for g in GROUP_ORDER])
    ax.set_xlabel("fit-derived preferred SF (cycles/deg)")
    ax.set_ylabel("original SF group")
    ax.set_title("B. Preferred-SF metric and thresholds", loc="left", fontweight="bold")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.35", markeredgecolor="white", markersize=5, label="retained"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="0.35", markersize=6, label="not retained"),
        Line2D([0], [0], color="0.35", linestyle=":", linewidth=1.0, label="low <= 0.05"),
        Line2D([0], [0], color="0.35", linestyle="--", linewidth=1.0, label="high >= 0.5"),
        Line2D([0], [0], color="0.55", linestyle="-.", linewidth=0.9, label="1 cycle/window"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6.7, loc="upper left")


def representative_units(units: pd.DataFrame) -> list[int]:
    reps: list[int] = []
    for group_id in GROUP_ORDER:
        sub = units[units["sf_group"].eq(group_id) & units["retained_in_microsaccade_curve_group"].astype(bool)].copy()
        sub = sub[pd.to_numeric(sub["dynamic_log_gaussian_marginal_fit_ok"], errors="coerce").astype(bool)]
        if sub.empty:
            sub = units[units["sf_group"].eq(group_id)].copy()
        metric = pd.to_numeric(sub["sf_split_metric"], errors="coerce").to_numpy(dtype=float)
        target = float(np.nanmedian(metric))
        idx = int(np.nanargmin(np.abs(metric - target)))
        reps.append(int(sub.iloc[idx]["unit_index"]))
    return reps


def plot_representative_fits(ax: plt.Axes, curves: pd.DataFrame, units: pd.DataFrame) -> None:
    x_grid = np.geomspace(float(min(SF_TICKS)), float(max(SF_TICKS)), 256)
    for unit_index in representative_units(units):
        unit_curve = curves[curves["unit_index"].eq(unit_index)].sort_values("spatial_cpd").copy()
        if unit_curve.empty:
            continue
        group_id = str(unit_curve["sf_group"].iloc[0])
        color = GROUP_COLORS[group_id]
        y = unit_curve["dynamic_marginal_response_amp_rms"].to_numpy(dtype=float)
        y_norm = normalize_curve(y)
        ax.plot(
            unit_curve["spatial_cpd"],
            y_norm,
            marker="o",
            color=color,
            linewidth=1.4,
            markersize=3.6,
            label=f"{unit_curve['unit_label'].iloc[0]} {GROUP_LABELS[group_id]}",
        )
        row = units[units["unit_index"].eq(unit_index)].iloc[0]
        fit_y = log_gaussian_curve(
            np.log2(x_grid),
            float(row["dynamic_log_gaussian_marginal_baseline"]),
            float(row["dynamic_log_gaussian_marginal_amplitude"]),
            float(row["dynamic_log_gaussian_marginal_log2_sf"]),
            float(row["dynamic_log_gaussian_marginal_sigma_octaves"]),
        )
        ax.plot(x_grid, normalize_curve(fit_y), color=color, linestyle="--", linewidth=1.5)
        ax.axvline(float(row["sf_split_metric"]), color=color, linewidth=0.85, alpha=0.6)
    add_frequency_guides(ax)
    ax.set_title("D. Representative log-Gaussian fits", loc="left", fontweight="bold")
    ax.set_xlabel("probe spatial frequency (cycles/deg)")
    ax.set_ylabel("normalized response / fit")
    ax.legend(frameon=False, fontsize=6.7, loc="upper right")


def plot_figure(curves: pd.DataFrame, units: pd.DataFrame, out_dir: Path) -> Path:
    summary3 = summarize_group_curves(curves, "sf_group", GROUP_ORDER)
    summary2 = summarize_group_curves(curves, "final_sf_group", FINAL_GROUP_ORDER)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.3), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.975, bottom=0.075, top=0.89, hspace=0.38, wspace=0.28)
    plot_group_curve_panel(
        axes[0, 0],
        curves,
        summary3,
        group_col="sf_group",
        order=GROUP_ORDER,
        labels=GROUP_LABELS,
        colors=GROUP_COLORS,
        title="A. Original dynamic marginal SF curves",
        retained_only=False,
    )
    plot_preference_distribution(axes[0, 1], units)
    plot_group_curve_panel(
        axes[1, 0],
        curves,
        summary2,
        group_col="final_sf_group",
        order=FINAL_GROUP_ORDER,
        labels=FINAL_GROUP_LABELS,
        colors=FINAL_GROUP_COLORS,
        title="C. Final collapse, retained overlay subset",
        retained_only=True,
    )
    plot_representative_fits(axes[1, 1], curves, units)
    fig.suptitle(
        "Previous SF tuning labels: coarse dynamic marginal grating fits",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.935,
        "Dynamic response amplitude is averaged over TF>0 and orientation at each SF, fit with a 1D log-Gaussian, then thresholded by preferred SF.",
        ha="left",
        fontsize=8.5,
        color="0.35",
    )
    png = out_dir / "previous_sf_tuning_groups.png"
    pdf = out_dir / "previous_sf_tuning_groups.pdf"
    svg = out_dir / "previous_sf_tuning_groups.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    summary3.to_csv(out_dir / "previous_sf_tuning_group_summary_original.csv", index=False)
    summary2.to_csv(out_dir / "previous_sf_tuning_group_summary_final_collapse.csv", index=False)
    return png


def write_readme(out_dir: Path, args: argparse.Namespace, units: pd.DataFrame, curves: pd.DataFrame, png: Path) -> None:
    counts = units["sf_group"].value_counts().to_dict()
    retained_counts = units[units["retained_in_microsaccade_curve_group"].astype(bool)]["sf_group"].value_counts().to_dict()
    lines = [
        "# Previous SF Tuning Groups",
        "",
        f"- Figure: `{_relative(png)}`",
        f"- Coarse frequency-tuning source: `{_relative(Path(args.tuning_dir))}`",
        f"- Unit grouping source: `{_relative(Path(args.unit_groups_csv))}`",
        f"- Microsaccade-retained source: `{_relative(Path(args.bimodal_groups_csv))}`",
        "",
        "## Definition",
        "",
        "For each unit, dynamic grating responses with `temporal_hz > 0` were averaged over temporal frequency and orientation at each spatial frequency. The resulting one-dimensional SF curve was fit as `baseline + amplitude * Gaussian(log2 SF)`. The preferred SF from that fit is `dynamic_log_gaussian_marginal_sf_cpd`.",
        "",
        f"- low SF: preferred SF <= {LOW_THRESHOLD_CPD:g} cpd",
        f"- high SF: preferred SF >= {HIGH_THRESHOLD_CPD:g} cpd",
        "- middle SF: between those thresholds",
        "",
        "## Counts",
        "",
        f"- Original 100-unit table: low={int(counts.get('low_sf', 0))}, middle={int(counts.get('middle_sf', 0))}, high={int(counts.get('high_sf', 0))}.",
        f"- Microsaccade-retained overlay subset: low={int(retained_counts.get('low_sf', 0))}, middle={int(retained_counts.get('middle_sf', 0))}, high={int(retained_counts.get('high_sf', 0))}.",
        f"- Final overlay collapse: low+middle={int(retained_counts.get('low_sf', 0) + retained_counts.get('middle_sf', 0))}, high={int(retained_counts.get('high_sf', 0))}.",
        "",
        "## Caution",
        "",
        "The lowest SF probe points are sub-cycle for the 101 px grating window: one cycle across the window is about 0.371 cpd. That makes the low-SF label partly a response-to-ramp/flicker label rather than a clean high-cycle grating preference.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    curves.to_csv(out_dir / "previous_sf_tuning_marginal_curves.csv", index=False)
    units.to_csv(out_dir / "previous_sf_tuning_unit_summary.csv", index=False)


def run(args: argparse.Namespace) -> dict[str, Path]:
    configure_matplotlib()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curves, units, _meta = load_previous_tuning(args)
    png = plot_figure(curves, units, out_dir)
    write_readme(out_dir, args, units, curves, png)
    return {"png": png, "readme": out_dir / "README.md"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning-dir", type=Path, default=DEFAULT_TUNING_DIR)
    parser.add_argument("--unit-groups-csv", type=Path, default=DEFAULT_UNIT_GROUPS_CSV)
    parser.add_argument("--bimodal-groups-csv", type=Path, default=DEFAULT_BIMODAL_GROUPS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    paths = run(parse_args())
    print(f"Wrote {paths['png']}")
    print(f"Wrote {paths['readme']}")


if __name__ == "__main__":
    main()
