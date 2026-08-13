#!/usr/bin/env python3
"""Compare prior RR100 dynamic SF fits with held-out recorded/twin preferences."""

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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    ROOT / "outputs/redundancy_resolved_v1_twin/rr100_recorded_twin_gratings_check_v1"
)
DEFAULT_NEW_METRICS = DEFAULT_RUN_DIR / "rr100_grating_tuning_metrics.csv"
DEFAULT_PREVIOUS_UNITS = (
    ROOT / "outputs/fig/ssi_figure_v2/panels/previous_sf_tuning_groups/"
    "previous_sf_tuning_unit_summary.csv"
)
OUT_STEM = "rr100_previous_sf_fits_vs_recorded_twin"
GROUP_COLORS = {"low_sf": "#0072B2", "middle_sf": "#559F76", "high_sf": "#D55E00"}
PREVIOUS_PROBE_SFS = np.asarray([0.0125, 0.05, 0.2, 0.8, 3.2, 12.8], dtype=float)
COMMON_X_TICKS = np.asarray([0.0125, 0.05, 0.2, 0.371, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-metrics", type=Path, default=DEFAULT_NEW_METRICS)
    parser.add_argument("--previous-units", type=Path, default=DEFAULT_PREVIOUS_UNITS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-stem", default=OUT_STEM)
    return parser.parse_args()


def spearman_pair(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    valid = x.notna() & y.notna() & (x > 0) & (y > 0)
    if int(valid.sum()) < 3:
        return float("nan"), int(valid.sum())
    return float(pd.concat([x[valid], y[valid]], axis=1).corr(method="spearman").iloc[0, 1]), int(valid.sum())


def file_identity(path: Path) -> dict[str, object]:
    stat = path.resolve().stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def source_summary(name: str, values: pd.Series, n_all: int) -> dict[str, object]:
    positive = values[values > 0].dropna().to_numpy(dtype=float)
    return {
        "source": name,
        "n_all_rr100": int(n_all),
        "n_positive_sf": int(positive.size),
        "n_zero_sf": int((values == 0).sum()),
        "n_invalid": int(values.isna().sum()),
        "median_positive_sf_cpd": float(np.median(positive)) if positive.size else np.nan,
        "geometric_mean_positive_sf_cpd": (
            float(np.exp(np.mean(np.log(positive)))) if positive.size else np.nan
        ),
        "min_positive_sf_cpd": float(np.min(positive)) if positive.size else np.nan,
        "max_positive_sf_cpd": float(np.max(positive)) if positive.size else np.nan,
    }


def add_distribution_strip(
    ax: plt.Axes,
    values: pd.Series,
    y: float,
    color: str,
    label: str,
    rng: np.random.Generator,
) -> None:
    positive = values[values > 0].dropna().to_numpy(dtype=float)
    jitter = rng.uniform(-0.11, 0.11, size=positive.size)
    ax.scatter(positive, y + jitter, s=17, color=color, alpha=0.58, edgecolor="none")
    if positive.size:
        median = float(np.median(positive))
        ax.scatter([median], [y], s=80, marker="D", color=color, edgecolor="white", linewidth=0.8, zorder=5)
    ax.text(21.0, y, label, color=color, va="center", ha="right", fontsize=9, fontweight="bold")


def plot_cross_probe_scatter(
    ax: plt.Axes,
    joined: pd.DataFrame,
    y_col: str,
    y_label: str,
    panel_title: str,
    rho: float,
    n: int,
) -> None:
    x_col = "previous_dynamic_log_gaussian_sf_cpd"
    valid = (
        joined[x_col].notna()
        & joined[y_col].notna()
        & (joined[x_col] > 0)
        & (joined[y_col] > 0)
    )
    for group, sub in joined[valid].groupby("previous_sf_group", sort=False):
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=31,
            color=GROUP_COLORS.get(str(group), "0.5"),
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
            label=str(group).replace("_", " "),
        )
    lo, hi = 0.01, 20.0
    ax.plot([lo, hi], [lo, hi], "--", color="0.35", lw=1.0, label="identity")
    ax.axvline(0.3713343185, color="#7B3294", ls=":", lw=1.2)
    ax.axhline(1.0, color="0.55", ls=":", lw=1.0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0.7, hi)
    ax.set_xticks(COMMON_X_TICKS)
    ax.set_xticklabels(["0.0125", "0.05", "0.2", "0.371", "1", "2", "4", "8", "16"], rotation=45, ha="right")
    ax.set_yticks([1.0, 2.0, 4.0, 8.0, 16.0])
    ax.set_yticklabels(["1", "2", "4", "8", "16"])
    ax.set_xlabel("previous twin log-Gaussian preferred SF (cpd)")
    ax.set_ylabel(y_label)
    ax.set_title(panel_title, loc="left", fontweight="bold")
    ax.grid(alpha=0.15, which="both")
    ax.text(
        0.04,
        0.94,
        f"Spearman $\\rho$={rho:.2f}, n={n}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2},
    )


def main() -> None:
    args = parse_args()
    new_path = args.new_metrics.resolve()
    previous_path = args.previous_units.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    new = pd.read_csv(new_path)
    previous = pd.read_csv(previous_path)
    if len(new) != 100 or new["rr100_index"].nunique() != 100:
        raise ValueError("New grating metrics must contain exactly 100 unique RR100 indices")
    if len(previous) != 100 or previous["unit_index"].nunique() != 100:
        raise ValueError("Previous SF table must contain exactly 100 unique RR100 indices")

    previous_keep = previous[
        [
            "unit_index",
            "unit_label",
            "dynamic_log_gaussian_marginal_sf_cpd",
            "dynamic_log_gaussian_marginal_r2",
            "dynamic_log_gaussian_marginal_fit_ok",
            "dynamic_log_gaussian_marginal_fit_peak_is_boundary",
            "dynamic_log_gaussian_marginal_low_subcycle_amp_share",
            "dynamic_sf_probe_one_cycle_cpd",
            "sf_group",
            "sf_group_label",
        ]
    ].rename(
        columns={
            "unit_index": "rr100_index",
            "unit_label": "previous_unit_label",
            "dynamic_log_gaussian_marginal_sf_cpd": "previous_dynamic_log_gaussian_sf_cpd",
            "dynamic_log_gaussian_marginal_r2": "previous_fit_r2",
            "dynamic_log_gaussian_marginal_fit_ok": "previous_fit_ok",
            "dynamic_log_gaussian_marginal_fit_peak_is_boundary": "previous_fit_peak_is_boundary",
            "dynamic_log_gaussian_marginal_low_subcycle_amp_share": "previous_low_subcycle_amp_share",
            "dynamic_sf_probe_one_cycle_cpd": "previous_probe_one_cycle_cpd",
            "sf_group": "previous_sf_group",
            "sf_group_label": "previous_sf_group_label",
        }
    )
    joined = new.merge(previous_keep, on="rr100_index", how="inner", validate="one_to_one")
    if len(joined) != 100:
        raise AssertionError(f"RR100 join lost units: {len(joined)}")
    joined["recorded_task_preferred_sf_cpd"] = joined["real_peak_sf"].round(6)
    joined["fitted_twin_task_preferred_sf_cpd"] = joined["twin_peak_sf"].round(6)
    joined["previous_preference_below_one_cycle"] = (
        joined["previous_dynamic_log_gaussian_sf_cpd"] < joined["previous_probe_one_cycle_cpd"]
    )
    joined["previous_preference_below_new_task_min_positive_sf"] = (
        joined["previous_dynamic_log_gaussian_sf_cpd"] < 1.0
    )
    for source in ("recorded_task", "fitted_twin_task"):
        sf_col = f"{source}_preferred_sf_cpd"
        shift = pd.Series(np.nan, index=joined.index, dtype=float)
        valid_shift = (joined[sf_col] > 0) & (joined["previous_dynamic_log_gaussian_sf_cpd"] > 0)
        shift.loc[valid_shift] = np.log2(
            joined.loc[valid_shift, sf_col]
            / joined.loc[valid_shift, "previous_dynamic_log_gaussian_sf_cpd"]
        )
        joined[f"{source}_minus_previous_octaves"] = shift
    joined_path = out_dir / f"{args.out_stem}_joined_units.csv"
    joined.to_csv(joined_path, index=False)

    previous_pref = joined["previous_dynamic_log_gaussian_sf_cpd"]
    twin_pref = joined["fitted_twin_task_preferred_sf_cpd"]
    real_pref = joined["recorded_task_preferred_sf_cpd"]
    rho_twin, n_twin = spearman_pair(previous_pref, twin_pref)
    rho_real, n_real = spearman_pair(previous_pref, real_pref)

    summary = pd.DataFrame(
        [
            source_summary("previous_twin_dynamic_log_gaussian_fit", previous_pref, len(joined)),
            source_summary("heldout_fitted_twin_task_peak", twin_pref, len(joined)),
            source_summary("heldout_recorded_task_peak", real_pref, len(joined)),
        ]
    )
    summary["comparison_note"] = (
        "Previous fit: center-pixel RR100 twin, phase-RMS amplitude marginalized across orientation and TF>0. "
        "Held-out task: mean response maximum on the recorded grating grid at the recorded-selected lag."
    )
    summary.to_csv(out_dir / f"{args.out_stem}_distribution_summary.csv", index=False)

    stats = pd.DataFrame(
        [
            {
                "comparison": "previous_fit_vs_heldout_fitted_twin",
                "spearman_rho_positive_sf": rho_twin,
                "n_positive_pairs": n_twin,
            },
            {
                "comparison": "previous_fit_vs_heldout_recorded",
                "spearman_rho_positive_sf": rho_real,
                "n_positive_pairs": n_real,
            },
        ]
    )
    stats.to_csv(out_dir / f"{args.out_stem}_agreement_stats.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.35), gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})
    ax_dist, ax_twin, ax_real = axes
    rng = np.random.default_rng(7)
    ax_dist.axvspan(0.01, 1.0, color="0.92", zorder=0)
    ax_dist.axvline(0.3713343185, color="#7B3294", ls=":", lw=1.4, label="1 cycle / old window")
    ax_dist.axvline(1.0, color="0.45", ls="--", lw=1.1, label="new task min positive SF")
    for probe_sf in PREVIOUS_PROBE_SFS:
        ax_dist.axvline(probe_sf, color="0.75", ls=":", lw=0.55, zorder=0)
    add_distribution_strip(ax_dist, previous_pref, 2.0, "#7B3294", "previous twin fit", rng)
    add_distribution_strip(ax_dist, twin_pref, 1.0, "#d62728", "held-out fitted twin", rng)
    add_distribution_strip(ax_dist, real_pref, 0.0, "#222222", "held-out recorded", rng)
    ax_dist.set_xscale("log", base=2)
    ax_dist.set_xlim(0.01, 22.0)
    ax_dist.set_xticks(COMMON_X_TICKS)
    ax_dist.set_xticklabels(
        ["0.0125", "0.05", "0.2", "0.371", "1", "2", "4", "8", "16"],
        rotation=45,
        ha="right",
    )
    ax_dist.set_ylim(-0.45, 2.45)
    ax_dist.set_yticks([])
    ax_dist.set_xlabel("preferred spatial frequency (cpd; log scale)")
    ax_dist.set_title("A  Cross-probe distributions", loc="left", fontweight="bold")
    ax_dist.grid(axis="x", alpha=0.14, which="both")
    ax_dist.text(
        0.018,
        -0.30,
        "shaded: below smallest positive SF in held-out task",
        fontsize=7.5,
        color="0.35",
    )
    ax_dist.text(
        0.012,
        2.30,
        (
            f"previous: {int((previous_pref < 0.3713343185).sum())}/100 below 1 cycle; "
            f"{int((previous_pref < 1.0).sum())}/100 below 1 cpd"
        ),
        fontsize=8,
    )
    ax_dist.text(
        0.012,
        0.70,
        f"twin: zero={int((twin_pref == 0).sum())}, invalid={int(twin_pref.isna().sum())}",
        fontsize=7.5,
        color="#d62728",
    )
    ax_dist.text(
        0.012,
        -0.30 + 0.18,
        f"recorded: zero={int((real_pref == 0).sum())}, invalid={int(real_pref.isna().sum())}",
        fontsize=7.5,
        color="#222222",
    )

    plot_cross_probe_scatter(
        ax_twin,
        joined,
        "fitted_twin_task_preferred_sf_cpd",
        "held-out fitted-twin preferred SF (cpd)",
        "B  Previous fit vs held-out twin",
        rho_twin,
        n_twin,
    )
    plot_cross_probe_scatter(
        ax_real,
        joined,
        "recorded_task_preferred_sf_cpd",
        "held-out recorded preferred SF (cpd)",
        "C  Previous fit vs recorded unit",
        rho_real,
        n_real,
    )
    handles, labels = ax_twin.get_legend_handles_labels()
    ax_twin.legend(handles, labels, frameon=False, fontsize=7, loc="lower right")

    fig.suptitle(
        "RR100 SF preference across the previous synthetic twin probe and held-out recorded gratings",
        fontsize=14,
        y=0.99,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Previous: center-pixel twin phase-RMS amplitude, marginalized across orientation and TF>0, "
            "fit with a log-Gaussian on 0.0125–12.8 cpd. Held-out: mean-response maximum on "
            "0/1/2/4/8/16 cpd at the recorded-selected lag. Cross-probe values are not interchangeable."
        ),
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=[0.02, 0.07, 1.0, 0.94], w_pad=2.2)
    png_path = out_dir / f"{args.out_stem}.png"
    pdf_path = out_dir / f"{args.out_stem}.pdf"
    fig.savefig(png_path, dpi=210)
    fig.savefig(pdf_path)
    plt.close(fig)

    manifest = {
        "analysis": "rr100_previous_sf_fits_vs_recorded_twin",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "cross_probe_population_comparison_after_unit_map_checkpoint",
        "new_metrics": file_identity(new_path),
        "previous_sf_units": file_identity(previous_path),
        "join_key": "rr100_index == previous unit_index",
        "n_joined": int(len(joined)),
        "rr100_population_identity_match": True,
        "previous_probe": {
            "response": "center-pixel RR100 twin phase-RMS response amplitude",
            "marginalization": "orientation and temporal_hz > 0",
            "fit": "baseline + amplitude * Gaussian(log2 SF)",
            "spatial_cpds": PREVIOUS_PROBE_SFS.tolist(),
            "one_cycle_cpd": 0.3713343185,
        },
        "heldout_probe": {
            "response": "mean recorded or fitted-twin task response",
            "preference": "maximum SF-orientation cell at recorded-selected lag",
            "spatial_cpds": [0.0, 1.0, 2.0, 4.0, 8.0, 16.0],
            "twin_lag": "same recorded-selected lag",
        },
        "agreement": {
            "previous_vs_heldout_twin_spearman_positive_sf": rho_twin,
            "n_previous_vs_heldout_twin": n_twin,
            "previous_vs_recorded_spearman_positive_sf": rho_real,
            "n_previous_vs_recorded": n_real,
        },
        "support_caveats": {
            "previous_preferences_below_one_cycle": int((previous_pref < 0.3713343185).sum()),
            "previous_preferences_below_new_task_min_positive_sf": int((previous_pref < 1.0).sum()),
            "previous_boundary_fits": int(joined["previous_fit_peak_is_boundary"].astype(bool).sum()),
            "heldout_invalid_pairs": int((real_pref.isna() | twin_pref.isna()).sum()),
        },
        "outputs": {
            "figure_png": str(png_path),
            "figure_pdf": str(pdf_path),
            "joined_units": str(joined_path),
            "distribution_summary": str(out_dir / f"{args.out_stem}_distribution_summary.csv"),
            "agreement_stats": str(out_dir / f"{args.out_stem}_agreement_stats.csv"),
        },
    }
    (out_dir / f"{args.out_stem}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {png_path}")
    print(f"Wrote {joined_path}")


if __name__ == "__main__":
    main()
