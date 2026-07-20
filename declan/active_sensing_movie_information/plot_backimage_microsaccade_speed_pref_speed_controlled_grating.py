#!/usr/bin/env python3
"""Compare speed-controlled grating tuning for microsaccade-derived unit groups."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROBE_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_speed_controlled_grating_probe_v1"
)
DEFAULT_GROUPS_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_microsaccade_snippet_rr100_spatial_ssi_isotropic_padded_event_scaled_full_amp1sd_n40_v1/"
    "bimodal_unit_curve_groups/bimodal_unit_curve_groups.csv"
)
DEFAULT_OUT_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_microsaccade_speed_pref_speed_controlled_grating_v1"
)
EPS = 1e-12
GROUP_MAP = {
    "large_scale_preferring": ("high_speed_preferring", "high-speed preferring"),
    "small_scale_preferring": ("low_speed_preferring", "low-speed preferring"),
}
GROUP_ORDER = ["high_speed_preferring", "low_speed_preferring"]
GROUP_COLORS = {
    "high_speed_preferring": "#1f77b4",
    "low_speed_preferring": "#d62728",
}
FAMILY_ORDER = ["cycle_valid", "subcycle_control"]
FAMILY_LABELS = {
    "cycle_valid": "cycle-valid SFs",
    "subcycle_control": "sub-cycle control SFs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--groups-csv", type=Path, default=DEFAULT_GROUPS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def sem(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size <= 1:
        return 0.0
    return float(np.std(vals, ddof=1) / math.sqrt(vals.size))


def welch_ttest(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    av = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    bv = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if av.size < 2 or bv.size < 2:
        return float("nan"), float("nan")
    try:
        from scipy import stats

        res = stats.ttest_ind(av, bv, equal_var=False, nan_policy="omit")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), float("nan")


def add_speed_pref_labels(groups: pd.DataFrame) -> pd.DataFrame:
    groups = groups.copy()
    mapped = groups["curve_group"].map(GROUP_MAP)
    groups["speed_pref_group"] = [item[0] if isinstance(item, tuple) else None for item in mapped]
    groups["speed_pref_label"] = [item[1] if isinstance(item, tuple) else None for item in mapped]
    groups = groups[groups["speed_pref_group"].notna()].copy()
    return groups


def unit_zscore(frame: pd.DataFrame, value: str) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, idx in frame.groupby(["unit_index", "speed_family"], sort=False).groups.items():
        vals = frame.loc[idx, value].to_numpy(dtype=float)
        mu = float(np.nanmean(vals))
        sd = float(np.nanstd(vals))
        if sd > EPS:
            out.loc[idx] = (vals - mu) / sd
    return out


def summarize_curves(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value_name, source_col in [
        ("unit_speed_z", "unit_speed_z"),
        ("response_amp_rms_mean", "response_amp_rms_mean"),
    ]:
        for keys, sub in curves.groupby(
            ["speed_family", "speed_pref_group", "speed_pref_label", "speed_dps", "log2_speed_dps"],
            sort=True,
        ):
            family, group, label, speed, log_speed = keys
            vals = pd.to_numeric(sub[source_col], errors="coerce").dropna()
            rows.append(
                {
                    "speed_family": family,
                    "speed_pref_group": group,
                    "speed_pref_label": label,
                    "speed_dps": float(speed),
                    "log2_speed_dps": float(log_speed),
                    "value_name": value_name,
                    "n_units": int(vals.shape[0]),
                    "mean": float(vals.mean()) if not vals.empty else float("nan"),
                    "sem": sem(vals),
                    "median": float(vals.median()) if not vals.empty else float("nan"),
                    "n_finite": int(vals.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def summarize_metrics(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "log2_peak_speed_dps",
        "amp_weighted_log2_speed_dps",
        "speed_curve_log_slope_z",
        "speed_curve_dynamic_range",
        "speed_curve_mean_amp",
    ]
    rows: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    for (family, group, label), sub in summary.groupby(
        ["speed_family", "speed_pref_group", "speed_pref_label"], sort=True
    ):
        for metric in metrics:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
            rows.append(
                {
                    "speed_family": family,
                    "speed_pref_group": group,
                    "speed_pref_label": label,
                    "metric": metric,
                    "n_units": int(vals.shape[0]),
                    "mean": float(vals.mean()) if not vals.empty else float("nan"),
                    "sem": sem(vals),
                    "median": float(vals.median()) if not vals.empty else float("nan"),
                    "n_finite": int(vals.shape[0]),
                }
            )
    for family, sub in summary.groupby("speed_family", sort=True):
        high = sub[sub["speed_pref_group"] == "high_speed_preferring"]
        low = sub[sub["speed_pref_group"] == "low_speed_preferring"]
        for metric in metrics:
            t_stat, p_value = welch_ttest(high[metric], low[metric])
            high_vals = pd.to_numeric(high[metric], errors="coerce").dropna()
            low_vals = pd.to_numeric(low[metric], errors="coerce").dropna()
            tests.append(
                {
                    "speed_family": family,
                    "metric": metric,
                    "high_n": int(high_vals.shape[0]),
                    "low_n": int(low_vals.shape[0]),
                    "high_mean": float(high_vals.mean()) if not high_vals.empty else float("nan"),
                    "low_mean": float(low_vals.mean()) if not low_vals.empty else float("nan"),
                    "high_minus_low": float(high_vals.mean() - low_vals.mean())
                    if not high_vals.empty and not low_vals.empty
                    else float("nan"),
                    "welch_t": t_stat,
                    "welch_p": p_value,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(tests)


def plot_group_comparison(
    out_dir: Path,
    curves: pd.DataFrame,
    curve_summary: pd.DataFrame,
    metric_summary: pd.DataFrame,
    metric_tests: pd.DataFrame,
    *,
    dpi: int,
) -> tuple[Path, Path]:
    png = out_dir / "speed_controlled_grating_speed_pref_group_tuning.png"
    pdf = out_dir / "speed_controlled_grating_speed_pref_group_tuning.pdf"
    fig, axes = plt.subplots(2, 4, figsize=(17.5, 9.2), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.075, top=0.84, hspace=0.48, wspace=0.28)
    fig.suptitle(
        "Speed-controlled grating tuning by microsaccade scale-preference group",
        y=0.965,
        fontsize=16,
    )
    fig.text(
        0.5,
        0.925,
        "Groups defined from the previous event-scaled microsaccade SSI curves; mean +/- SEM across units",
        ha="center",
        va="center",
        fontsize=11,
        color="0.35",
    )

    for row, family in enumerate(FAMILY_ORDER):
        family_curves = curves[curves["speed_family"] == family]
        family_curve_summary = curve_summary[curve_summary["speed_family"] == family]
        family_metrics = metric_summary[metric_summary["speed_family"] == family]
        family_tests = metric_tests[metric_tests["speed_family"] == family]

        ax = axes[row, 0]
        for group in GROUP_ORDER:
            sub = family_curve_summary[
                (family_curve_summary["speed_pref_group"] == group)
                & (family_curve_summary["value_name"] == "unit_speed_z")
            ].sort_values("speed_dps")
            if sub.empty:
                continue
            color = GROUP_COLORS[group]
            label = str(sub["speed_pref_label"].iloc[0])
            ax.plot(sub["speed_dps"], sub["mean"], color=color, marker="o", lw=2.2, label=label)
            ax.fill_between(
                sub["speed_dps"],
                sub["mean"] - sub["sem"],
                sub["mean"] + sub["sem"],
                color=color,
                alpha=0.16,
                linewidth=0,
            )
        ax.axhline(0, color="0.55", lw=1, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_title(f"{FAMILY_LABELS.get(family, family)}: within-unit z")
        ax.set_ylabel("within-unit speed z")
        ax.set_xlabel("grating speed (deg/s)")
        ax.grid(True, color="0.9")
        ax.legend(frameon=False, fontsize=9)

        ax = axes[row, 1]
        for group in GROUP_ORDER:
            sub = family_curve_summary[
                (family_curve_summary["speed_pref_group"] == group)
                & (family_curve_summary["value_name"] == "response_amp_rms_mean")
            ].sort_values("speed_dps")
            if sub.empty:
                continue
            color = GROUP_COLORS[group]
            label = str(sub["speed_pref_label"].iloc[0])
            ax.plot(sub["speed_dps"], sub["mean"], color=color, marker="o", lw=2.2, label=label)
            ax.fill_between(
                sub["speed_dps"],
                sub["mean"] - sub["sem"],
                sub["mean"] + sub["sem"],
                color=color,
                alpha=0.16,
                linewidth=0,
            )
        ax.set_xscale("log", base=2)
        ax.set_title("raw response amplitude")
        ax.set_ylabel("RMS modulation amp")
        ax.set_xlabel("grating speed (deg/s)")
        ax.grid(True, color="0.9")

        for col, (metric, title, ylabel) in enumerate(
            [
                ("amp_weighted_log2_speed_dps", "amp-weighted speed", "log2 deg/s"),
                ("speed_curve_log_slope_z", "speed curve slope", "slope of z vs log2 speed"),
            ],
            start=2,
        ):
            ax = axes[row, col]
            metric_rows = []
            for x, group in enumerate(GROUP_ORDER):
                vals = family_curves[
                    family_curves["speed_pref_group"] == group
                ]["unit_index"].drop_duplicates()
                unit_metric = family_metrics[
                    (family_metrics["speed_pref_group"] == group) & (family_metrics["metric"] == metric)
                ]
                unit_summary_vals = curves_to_unit_metric(curves, family, group, metric)
                if unit_summary_vals.empty:
                    continue
                rng = np.random.default_rng(1000 + row * 10 + col * 100 + x)
                jitter = rng.uniform(-0.08, 0.08, size=unit_summary_vals.shape[0])
                color = GROUP_COLORS[group]
                ax.scatter(
                    np.full(unit_summary_vals.shape[0], x) + jitter,
                    unit_summary_vals.to_numpy(dtype=float),
                    s=18,
                    color=color,
                    alpha=0.48,
                    edgecolor="none",
                )
                row_metric = unit_metric.iloc[0] if not unit_metric.empty else None
                if row_metric is not None:
                    mean = float(row_metric["mean"])
                    err = float(row_metric["sem"])
                    ax.errorbar(
                        [x],
                        [mean],
                        yerr=[err],
                        color="black",
                        marker="o",
                        markersize=5,
                        capsize=4,
                        lw=1.5,
                    )
                    metric_rows.append((group, mean, err, int(row_metric["n_units"])))
                ax.set_xticks(range(len(GROUP_ORDER)))
                ax.set_xticklabels(["high-speed\npref.", "low-speed\npref."])
            test = family_tests[family_tests["metric"] == metric]
            p_text = ""
            if not test.empty and np.isfinite(float(test["welch_p"].iloc[0])):
                p_text = f"\nWelch p={float(test['welch_p'].iloc[0]):.3g}"
            ax.set_title(title + p_text)
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", color="0.9")

    for ax in axes.flat:
        if ax.get_xscale() == "log":
            ax.set_xticks([1, 2, 4, 8, 16, 32])
            ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def curves_to_unit_metric(curves: pd.DataFrame, family: str, group: str, metric: str) -> pd.Series:
    summary_cols = [
        "unit_index",
        "speed_family",
        "speed_pref_group",
        "log2_peak_speed_dps",
        "amp_weighted_log2_speed_dps",
        "speed_curve_log_slope_z",
        "speed_curve_dynamic_range",
        "speed_curve_mean_amp",
    ]
    if metric not in summary_cols:
        return pd.Series(dtype=float)
    dedup = curves[summary_cols].drop_duplicates()
    return dedup[
        (dedup["speed_family"] == family) & (dedup["speed_pref_group"] == group)
    ][metric].dropna()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    speed_curves = pd.read_csv(Path(args.probe_dir) / "speed_controlled_speed_curves.csv")
    unit_summary = pd.read_csv(Path(args.probe_dir) / "speed_controlled_unit_speed_summary.csv")
    groups = add_speed_pref_labels(pd.read_csv(args.groups_csv))
    group_cols = [
        "unit_index",
        "curve_group",
        "curve_group_label",
        "speed_pref_group",
        "speed_pref_label",
        "sf_group",
        "sf_group_label",
    ]
    curves = speed_curves.merge(groups[group_cols], on="unit_index", how="inner")
    summary = unit_summary.merge(groups[group_cols], on="unit_index", how="inner")
    curves = curves.merge(
        summary[
            [
                "unit_index",
                "speed_family",
                "log2_peak_speed_dps",
                "amp_weighted_log2_speed_dps",
                "speed_curve_log_slope_z",
                "speed_curve_dynamic_range",
                "speed_curve_mean_amp",
            ]
        ],
        on=["unit_index", "speed_family"],
        how="left",
    )
    curves["unit_speed_z"] = unit_zscore(curves, "response_amp_rms_mean")
    curve_summary = summarize_curves(curves)
    metric_summary, metric_tests = summarize_metrics(summary)

    curves.to_csv(out_dir / "speed_controlled_grating_speed_pref_unit_curves.csv", index=False)
    curve_summary.to_csv(out_dir / "speed_controlled_grating_speed_pref_curve_summary.csv", index=False)
    metric_summary.to_csv(out_dir / "speed_controlled_grating_speed_pref_metric_summary.csv", index=False)
    metric_tests.to_csv(out_dir / "speed_controlled_grating_speed_pref_metric_tests.csv", index=False)
    png, pdf = plot_group_comparison(
        out_dir,
        curves,
        curve_summary,
        metric_summary,
        metric_tests,
        dpi=int(args.dpi),
    )
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(metric_tests.to_string(index=False))


if __name__ == "__main__":
    main()
