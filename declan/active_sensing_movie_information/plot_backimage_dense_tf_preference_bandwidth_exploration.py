#!/usr/bin/env python3
"""Explore TF preference versus TF bandwidth for dense SF/TF grating fits."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DENSE_GROUP_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_dense_sf_tf_speed_pref_groups_v1"
)
DEFAULT_GROUPS_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_microsaccade_snippet_rr100_spatial_ssi_isotropic_padded_event_scaled_full_amp1sd_n40_v1/"
    "bimodal_unit_curve_groups/bimodal_unit_curve_groups.csv"
)
DEFAULT_OUT_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_dense_sf_tf_speed_pref_groups_v1/tf_preference_bandwidth_exploration"
)
GROUP_ORDER = ["high_speed_preferring", "low_speed_preferring"]
GROUP_LABELS = {
    "high_speed_preferring": "high-speed pref.",
    "low_speed_preferring": "low-speed pref.",
}
GROUP_COLORS = {
    "high_speed_preferring": "#1f77b4",
    "low_speed_preferring": "#d62728",
}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-group-dir", type=Path, default=DEFAULT_DENSE_GROUP_DIR)
    parser.add_argument("--groups-csv", type=Path, default=DEFAULT_GROUPS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--speed-family", type=str, default="cycle_valid")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def sem(vals: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def hedges_g(high: pd.Series, low: pd.Series) -> float:
    a = pd.to_numeric(high, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(low, errors="coerce").dropna().to_numpy(dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = math.sqrt(((a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1)) / (a.size + b.size - 2))
    if pooled <= EPS:
        return float("nan")
    d = (float(np.mean(a)) - float(np.mean(b))) / pooled
    correction = 1.0 - 3.0 / (4.0 * (a.size + b.size) - 9.0)
    return float(correction * d)


def welch_p(high: pd.Series, low: pd.Series) -> float:
    try:
        from scipy import stats

        out = stats.ttest_ind(
            pd.to_numeric(high, errors="coerce").dropna(),
            pd.to_numeric(low, errors="coerce").dropna(),
            equal_var=False,
            nan_policy="omit",
        )
        return float(out.pvalue)
    except Exception:
        return float("nan")


def pearson_text(x: pd.Series, y: pd.Series) -> str:
    xv = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    yv = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(xv) & np.isfinite(yv)
    if ok.sum() < 3:
        return "r=nan"
    try:
        from scipy import stats

        r, p = stats.pearsonr(xv[ok], yv[ok])
        return f"r={float(r):.2f}, p={float(p):.3g}"
    except Exception:
        r = float(np.corrcoef(xv[ok], yv[ok])[0, 1])
        return f"r={r:.2f}"


def load_frame(dense_group_dir: Path, groups_csv: Path, *, speed_family: str) -> pd.DataFrame:
    fits = pd.read_csv(dense_group_dir / f"{speed_family}_dense_sf_tf_fit_unit_summary.csv")
    groups = pd.read_csv(groups_csv)
    extra_cols = [
        "unit_index",
        "high_minus_low_z",
        "z_slope_vs_scale",
        "preferred_scale_by_z",
        "anti_preferred_scale_by_z",
        "ssi_delta_3_vs_0",
        "ssi_delta_1_vs_0",
    ]
    frame = fits.merge(groups[extra_cols], on="unit_index", how="left", suffixes=("", "_microsaccade"))
    frame = frame[frame["fit_ok"].astype(bool)].copy()
    frame["is_edge_fit"] = frame["fit_edge_sf"].astype(bool) | frame["fit_edge_tf"].astype(bool)
    frame["group_binary"] = frame["speed_pref_group"].eq("high_speed_preferring").astype(int)
    frame["bandwidth_resid_after_tf_pref"] = np.nan
    ok = np.isfinite(frame["fit_pref_log2_tf"].to_numpy(dtype=float)) & np.isfinite(frame["fit_fwhm_tf_octaves"].to_numpy(dtype=float))
    if ok.sum() >= 3:
        x = frame.loc[ok, "fit_pref_log2_tf"].to_numpy(dtype=float)
        y = frame.loc[ok, "fit_fwhm_tf_octaves"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        frame.loc[ok, "bandwidth_resid_after_tf_pref"] = y - (slope * x + intercept)
    return frame


def group_stats(frame: pd.DataFrame, metric: str) -> dict[str, float]:
    high = frame[frame["speed_pref_group"].eq("high_speed_preferring")][metric]
    low = frame[frame["speed_pref_group"].eq("low_speed_preferring")][metric]
    return {
        "high_mean": float(pd.to_numeric(high, errors="coerce").mean()),
        "low_mean": float(pd.to_numeric(low, errors="coerce").mean()),
        "high_sem": sem(high),
        "low_sem": sem(low),
        "hedges_g": hedges_g(high, low),
        "p": welch_p(high, low),
    }


def add_strip(ax: plt.Axes, frame: pd.DataFrame, metric: str, ylabel: str, *, title: str, rng: np.random.Generator) -> None:
    for xloc, group in enumerate(GROUP_ORDER):
        vals = pd.to_numeric(frame[frame["speed_pref_group"].eq(group)][metric], errors="coerce").dropna()
        jitter = rng.uniform(-0.08, 0.08, size=vals.shape[0])
        ax.scatter(
            np.full(vals.shape[0], xloc) + jitter,
            vals,
            color=GROUP_COLORS[group],
            alpha=0.62,
            s=25,
            edgecolors="none",
        )
        ax.errorbar([xloc], [float(vals.mean())], yerr=[sem(vals)], color="black", marker="o", capsize=4)
    stats = group_stats(frame, metric)
    ax.set_title(f"{title}\ng={stats['hedges_g']:.2f}, p={stats['p']:.3g}")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER])
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color="0.9")


def add_group_regression(ax: plt.Axes, frame: pd.DataFrame, xcol: str, ycol: str) -> None:
    for group in GROUP_ORDER:
        sub = frame[frame["speed_pref_group"].eq(group)]
        x = pd.to_numeric(sub[xcol], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub[ycol], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            continue
        xs = np.linspace(float(np.min(x[ok])), float(np.max(x[ok])), 100)
        slope, intercept = np.polyfit(x[ok], y[ok], 1)
        ax.plot(xs, slope * xs + intercept, color=GROUP_COLORS[group], lw=1.5, alpha=0.78)


def plot_joint_and_distributions(out_dir: Path, frame: pd.DataFrame, *, speed_family: str, dpi: int) -> Path:
    png = out_dir / f"{speed_family}_tf_preference_vs_bandwidth_joint.png"
    pdf = out_dir / f"{speed_family}_tf_preference_vs_bandwidth_joint.pdf"
    subsets = [
        ("all", frame, "All successful fits"),
        ("interior", frame[~frame["is_edge_fit"]].copy(), "Interior fits only"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.3, 8.7), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.86, hspace=0.42, wspace=0.32)
    fig.suptitle("TF preference versus TF bandwidth in dense SF/TF grating fits", y=0.965, fontsize=15)
    fig.text(
        0.5,
        0.925,
        "Groups are fixed from microsaccade scale SSI curves; hollow points in the all-fit scatter are edge-preference fits",
        ha="center",
        color="0.35",
        fontsize=10.2,
    )
    rng = np.random.default_rng(45)
    for row, (_, sub, label) in enumerate(subsets):
        ax = axes[row, 0]
        for group in GROUP_ORDER:
            ss = sub[sub["speed_pref_group"].eq(group)]
            edge = ss["is_edge_fit"].to_numpy(dtype=bool)
            for edge_state, marker_label, alpha, face_mode in [
                (False, "interior", 0.72, "filled"),
                (True, "edge", 0.55, "hollow"),
            ]:
                part = ss[edge == edge_state]
                if part.empty:
                    continue
                color = GROUP_COLORS[group]
                ax.scatter(
                    part["fit_pref_log2_tf"],
                    part["fit_fwhm_tf_octaves"],
                    s=34,
                    facecolors=color if face_mode == "filled" else "none",
                    edgecolors=color,
                    linewidths=0.9,
                    alpha=alpha,
                    label=f"{GROUP_LABELS[group]} {marker_label}" if row == 0 else None,
                )
        add_group_regression(ax, sub, "fit_pref_log2_tf", "fit_fwhm_tf_octaves")
        ax.set_title(f"{label}: joint plane\n{pearson_text(sub['fit_pref_log2_tf'], sub['fit_fwhm_tf_octaves'])}")
        ax.set_xlabel("TF preference (log2 Hz)")
        ax.set_ylabel("TF bandwidth FWHM (octaves)")
        ax.grid(True, color="0.9")
        if row == 0:
            ax.legend(frameon=False, fontsize=7, ncol=2)

        add_strip(
            axes[row, 1],
            sub,
            "fit_pref_log2_tf",
            "TF preference (log2 Hz)",
            title=f"{label}: preference",
            rng=rng,
        )
        add_strip(
            axes[row, 2],
            sub,
            "fit_fwhm_tf_octaves",
            "TF FWHM (octaves)",
            title=f"{label}: bandwidth",
            rng=rng,
        )
    fig.savefig(png, dpi=int(dpi))
    fig.savefig(pdf)
    plt.close(fig)
    return png


def plot_continuous_microsaccade_relations(out_dir: Path, frame: pd.DataFrame, *, speed_family: str, dpi: int) -> Path:
    png = out_dir / f"{speed_family}_tf_features_vs_microsaccade_scale_metric.png"
    pdf = out_dir / f"{speed_family}_tf_features_vs_microsaccade_scale_metric.pdf"
    subsets = [
        (frame, "All successful fits"),
        (frame[~frame["is_edge_fit"]].copy(), "Interior fits only"),
    ]
    columns = [
        ("fit_pref_log2_tf", "TF preference (log2 Hz)", "TF preference"),
        ("fit_fwhm_tf_octaves", "TF FWHM (octaves)", "TF bandwidth"),
        ("bandwidth_resid_after_tf_pref", "TF bandwidth residual", "Bandwidth residual"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.6), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.86, hspace=0.42, wspace=0.32)
    fig.suptitle("Continuous microsaccade scale preference versus dense TF fit features", y=0.965, fontsize=15)
    fig.text(
        0.5,
        0.925,
        "x-axis is the original microsaccade high-minus-low scale score; positive means large-scale/high-speed-preferring",
        ha="center",
        color="0.35",
        fontsize=10.2,
    )
    for row, (sub, row_label) in enumerate(subsets):
        for col, (metric, ylabel, title) in enumerate(columns):
            ax = axes[row, col]
            for group in GROUP_ORDER:
                ss = sub[sub["speed_pref_group"].eq(group)]
                ax.scatter(
                    ss["high_minus_low_z"],
                    ss[metric],
                    color=GROUP_COLORS[group],
                    alpha=0.68,
                    s=32,
                    edgecolors="none",
                    label=GROUP_LABELS[group] if row == 0 and col == 0 else None,
                )
            x = pd.to_numeric(sub["high_minus_low_z"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 3:
                xs = np.linspace(float(np.min(x[ok])), float(np.max(x[ok])), 100)
                slope, intercept = np.polyfit(x[ok], y[ok], 1)
                ax.plot(xs, slope * xs + intercept, color="black", lw=1.6, alpha=0.82)
            ax.axvline(0, color="0.55", lw=1, ls=":")
            if metric == "bandwidth_resid_after_tf_pref":
                ax.axhline(0, color="0.55", lw=1, ls=":")
            ax.set_title(f"{row_label}: {title}\n{pearson_text(sub['high_minus_low_z'], sub[metric])}")
            ax.set_xlabel("microsaccade scale score: high minus low z")
            ax.set_ylabel(ylabel)
            ax.grid(True, color="0.9")
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.savefig(png, dpi=int(dpi))
    fig.savefig(pdf)
    plt.close(fig)
    return png


def write_summary(out_dir: Path, frame: pd.DataFrame, *, speed_family: str) -> Path:
    rows = []
    for subset_name, sub in [
        ("all_fit_ok", frame),
        ("interior_only", frame[~frame["is_edge_fit"]].copy()),
    ]:
        for metric in ["fit_pref_log2_tf", "fit_fwhm_tf_octaves", "bandwidth_resid_after_tf_pref"]:
            stats = group_stats(sub, metric)
            rows.append(
                {
                    "subset": subset_name,
                    "metric": metric,
                    "high_n": int(sub[sub["speed_pref_group"].eq("high_speed_preferring")][metric].notna().sum()),
                    "low_n": int(sub[sub["speed_pref_group"].eq("low_speed_preferring")][metric].notna().sum()),
                    **stats,
                    "pearson_with_microsaccade_high_minus_low": pearson_text(sub["high_minus_low_z"], sub[metric]),
                }
            )
    path = out_dir / f"{speed_family}_tf_preference_bandwidth_exploration_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = load_frame(Path(args.dense_group_dir), Path(args.groups_csv), speed_family=str(args.speed_family))
    frame.to_csv(out_dir / f"{args.speed_family}_tf_preference_bandwidth_exploration_unit_table.csv", index=False)
    joint_png = plot_joint_and_distributions(out_dir, frame, speed_family=str(args.speed_family), dpi=int(args.dpi))
    continuous_png = plot_continuous_microsaccade_relations(out_dir, frame, speed_family=str(args.speed_family), dpi=int(args.dpi))
    summary_csv = write_summary(out_dir, frame, speed_family=str(args.speed_family))
    print(f"Wrote {joint_png}")
    print(f"Wrote {continuous_png}")
    print(f"Wrote {summary_csv}")
    print(pd.read_csv(summary_csv).to_string(index=False))


if __name__ == "__main__":
    main()
