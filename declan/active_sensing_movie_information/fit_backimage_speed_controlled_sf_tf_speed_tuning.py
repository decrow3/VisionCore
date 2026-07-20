#!/usr/bin/env python3
"""Fit SF, TF, and speed tuning curves from the speed-controlled grating probe."""

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
    "backimage_speed_controlled_sf_tf_speed_tuning_fits_v1"
)
DIMENSIONS = [
    ("sf", "spatial_cpd", "log2_spatial_cpd", "spatial frequency (cpd)"),
    ("tf", "temporal_hz", "log2_temporal_hz", "temporal frequency (Hz)"),
    ("speed", "speed_dps", "log2_speed_dps", "speed (deg/s)"),
]
FAMILY_ORDER = ["cycle_valid", "subcycle_control"]
FAMILY_LABELS = {
    "cycle_valid": "cycle-valid SFs",
    "subcycle_control": "sub-cycle controls",
}
GROUP_MAP = {
    "large_scale_preferring": ("high_speed_preferring", "high-speed preferring"),
    "small_scale_preferring": ("low_speed_preferring", "low-speed preferring"),
}
GROUP_ORDER = ["high_speed_preferring", "low_speed_preferring"]
GROUP_COLORS = {
    "high_speed_preferring": "#1f77b4",
    "low_speed_preferring": "#d62728",
}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--groups-csv", type=Path, default=DEFAULT_GROUPS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-fit-points", type=int, default=4)
    parser.add_argument("--edge-frac", type=float, default=0.05)
    parser.add_argument("--min-r2-for-summary", type=float, default=-np.inf)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def sem(values: pd.Series | np.ndarray) -> float:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
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


def add_group_labels(groups: pd.DataFrame) -> pd.DataFrame:
    groups = groups.copy()
    mapped = groups["curve_group"].map(GROUP_MAP)
    groups["speed_pref_group"] = [item[0] if isinstance(item, tuple) else None for item in mapped]
    groups["speed_pref_label"] = [item[1] if isinstance(item, tuple) else None for item in mapped]
    return groups[groups["speed_pref_group"].notna()].copy()


def log_gaussian(log_x: np.ndarray, baseline: float, amplitude: float, mu: float, sigma: float) -> np.ndarray:
    return baseline + amplitude * np.exp(-0.5 * np.square((log_x - mu) / np.maximum(sigma, EPS)))


def fit_log_gaussian(log_x: np.ndarray, y: np.ndarray, *, min_points: int, edge_frac: float) -> dict[str, Any]:
    finite = np.isfinite(log_x) & np.isfinite(y)
    log_x = log_x[finite].astype(float)
    y = y[finite].astype(float)
    order = np.argsort(log_x)
    log_x = log_x[order]
    y = y[order]
    rec: dict[str, Any] = {
        "fit_ok": False,
        "fit_status": "insufficient_points",
        "fit_preferred_log2_value": float("nan"),
        "fit_preferred_value": float("nan"),
        "fit_baseline": float("nan"),
        "fit_amplitude": float("nan"),
        "fit_sigma_log2": float("nan"),
        "fit_r2": float("nan"),
        "fit_rmse": float("nan"),
        "observed_peak_log2_value": float("nan"),
        "observed_peak_value": float("nan"),
        "observed_peak_response": float("nan"),
        "n_fit_points": int(log_x.size),
    }
    if log_x.size:
        peak_idx = int(np.nanargmax(y))
        rec.update(
            {
                "observed_peak_log2_value": float(log_x[peak_idx]),
                "observed_peak_value": float(2.0 ** log_x[peak_idx]),
                "observed_peak_response": float(y[peak_idx]),
            }
        )
    if log_x.size < int(min_points):
        return rec
    x_min = float(np.nanmin(log_x))
    x_max = float(np.nanmax(log_x))
    x_span = max(x_max - x_min, EPS)
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    y_range = max(y_max - y_min, EPS)
    if not np.isfinite(y_range) or y_range <= EPS:
        rec["fit_status"] = "flat_or_invalid"
        return rec
    try:
        from scipy.optimize import curve_fit
    except Exception:
        rec["fit_status"] = "scipy_unavailable"
        return rec

    baseline_hi = max(y_max * 1.5, y_min + y_range * 2.0, EPS)
    amp_hi = max(y_max * 3.0, y_range * 10.0, EPS)
    sigma_hi = max(0.5, x_span * 2.0)
    bounds = (
        [0.0, 0.0, x_min, 0.15],
        [baseline_hi, amp_hi, x_max, sigma_hi],
    )
    peak_mu = float(log_x[int(np.nanargmax(y))])
    weighted_mu = float(np.nansum(log_x * np.maximum(y - y_min, 0.0)) / max(np.nansum(np.maximum(y - y_min, 0.0)), EPS))
    p0s = []
    for mu0 in [peak_mu, weighted_mu, float(np.nanmean(log_x))]:
        for sigma0 in [max(0.25, x_span / 4.0), max(0.5, x_span / 2.0), max(0.75, x_span)]:
            p0s.append([max(y_min, 0.0), y_range, float(np.clip(mu0, x_min, x_max)), min(sigma0, sigma_hi)])
    best: tuple[float, np.ndarray] | None = None
    for p0 in p0s:
        try:
            params, _ = curve_fit(
                log_gaussian,
                log_x,
                y,
                p0=p0,
                bounds=bounds,
                maxfev=20000,
            )
        except Exception:
            continue
        pred = log_gaussian(log_x, *params)
        rss = float(np.nansum(np.square(y - pred)))
        if best is None or rss < best[0]:
            best = (rss, params)
    if best is None:
        rec["fit_status"] = "fit_failed"
        return rec
    rss, params = best
    pred = log_gaussian(log_x, *params)
    tss = float(np.nansum(np.square(y - np.nanmean(y))))
    r2 = float(1.0 - rss / tss) if tss > EPS else float("nan")
    baseline, amplitude, mu, sigma = [float(v) for v in params]
    edge_tol = max(float(edge_frac) * x_span, 1e-6)
    if mu <= x_min + edge_tol:
        status = "lower_edge"
    elif mu >= x_max - edge_tol:
        status = "upper_edge"
    else:
        status = "interior"
    rec.update(
        {
            "fit_ok": True,
            "fit_status": status,
            "fit_preferred_log2_value": mu,
            "fit_preferred_value": float(2.0**mu),
            "fit_baseline": baseline,
            "fit_amplitude": amplitude,
            "fit_sigma_log2": sigma,
            "fit_r2": r2,
            "fit_rmse": float(math.sqrt(rss / max(log_x.size, 1))),
        }
    )
    return rec


def build_tuning_points(grouped: pd.DataFrame) -> pd.DataFrame:
    grouped = grouped.copy()
    grouped["log2_spatial_cpd"] = np.log2(pd.to_numeric(grouped["spatial_cpd"], errors="coerce"))
    grouped["log2_temporal_hz"] = np.log2(pd.to_numeric(grouped["temporal_hz"], errors="coerce"))
    grouped["log2_speed_dps"] = np.log2(pd.to_numeric(grouped["speed_dps"], errors="coerce"))
    rows: list[dict[str, Any]] = []
    for dim_name, value_col, log_col, dim_label in DIMENSIONS:
        keys = ["unit_index", "unit_label", "speed_family", value_col, log_col]
        for key_values, sub in grouped.groupby(keys, sort=True, dropna=True):
            unit, label, family, value, log_value = key_values
            amp = pd.to_numeric(sub["response_amp_rms"], errors="coerce").to_numpy(dtype=float)
            rows.append(
                {
                    "unit_index": int(unit),
                    "unit_label": str(label),
                    "speed_family": str(family),
                    "dimension": dim_name,
                    "dimension_label": dim_label,
                    "stimulus_value": float(value),
                    "log2_stimulus_value": float(log_value),
                    "response_amp_rms_mean": float(np.nanmean(amp)),
                    "response_amp_rms_median": float(np.nanmedian(amp)),
                    "n_rows_averaged": int(sub.shape[0]),
                    "n_pairs": int(sub["pair_id"].nunique()),
                    "n_orientations": int(sub["probe_orientation_deg"].nunique()),
                }
            )
    points = pd.DataFrame(rows)
    z_rows = []
    for _, sub in points.groupby(["unit_index", "speed_family", "dimension"], sort=False):
        vals = sub["response_amp_rms_mean"].to_numpy(dtype=float)
        sd = float(np.nanstd(vals))
        z = (vals - float(np.nanmean(vals))) / sd if sd > EPS else np.full_like(vals, np.nan)
        ss = sub.copy()
        ss["unit_dimension_z"] = z
        z_rows.append(ss)
    return pd.concat(z_rows, ignore_index=True) if z_rows else points.assign(unit_dimension_z=np.nan)


def fit_all_units(points: pd.DataFrame, *, min_points: int, edge_frac: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key_values, sub in points.groupby(["unit_index", "unit_label", "speed_family", "dimension", "dimension_label"], sort=True):
        unit, label, family, dim, dim_label = key_values
        sub = sub.sort_values("log2_stimulus_value")
        fit = fit_log_gaussian(
            sub["log2_stimulus_value"].to_numpy(dtype=float),
            sub["response_amp_rms_mean"].to_numpy(dtype=float),
            min_points=int(min_points),
            edge_frac=float(edge_frac),
        )
        rows.append(
            {
                "unit_index": int(unit),
                "unit_label": str(label),
                "speed_family": str(family),
                "dimension": str(dim),
                "dimension_label": str(dim_label),
                "min_stimulus_value": float(np.nanmin(sub["stimulus_value"].to_numpy(dtype=float))),
                "max_stimulus_value": float(np.nanmax(sub["stimulus_value"].to_numpy(dtype=float))),
                "min_log2_stimulus_value": float(np.nanmin(sub["log2_stimulus_value"].to_numpy(dtype=float))),
                "max_log2_stimulus_value": float(np.nanmax(sub["log2_stimulus_value"].to_numpy(dtype=float))),
                "n_unique_stimuli": int(sub["stimulus_value"].nunique()),
                **fit,
            }
        )
    return pd.DataFrame(rows)


def summarize_groups(fits: pd.DataFrame, *, min_r2: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = fits[(fits["fit_ok"]) & (pd.to_numeric(fits["fit_r2"], errors="coerce") >= float(min_r2))].copy()
    rows: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    for keys, sub in usable.groupby(["speed_family", "dimension", "dimension_label", "speed_pref_group", "speed_pref_label"], sort=True):
        family, dim, dim_label, group, label = keys
        log_vals = pd.to_numeric(sub["fit_preferred_log2_value"], errors="coerce").dropna()
        obs_log_vals = pd.to_numeric(sub["observed_peak_log2_value"], errors="coerce").dropna()
        rows.append(
            {
                "speed_family": family,
                "dimension": dim,
                "dimension_label": dim_label,
                "speed_pref_group": group,
                "speed_pref_label": label,
                "n_units": int(log_vals.shape[0]),
                "fit_pref_log2_mean": float(log_vals.mean()) if not log_vals.empty else float("nan"),
                "fit_pref_log2_sem": sem(log_vals),
                "fit_pref_log2_median": float(log_vals.median()) if not log_vals.empty else float("nan"),
                "fit_pref_geomean_value": float(2.0 ** log_vals.mean()) if not log_vals.empty else float("nan"),
                "fit_pref_median_value": float(2.0 ** log_vals.median()) if not log_vals.empty else float("nan"),
                "observed_peak_log2_median": float(obs_log_vals.median()) if not obs_log_vals.empty else float("nan"),
                "observed_peak_median_value": float(2.0 ** obs_log_vals.median()) if not obs_log_vals.empty else float("nan"),
                "median_fit_r2": float(pd.to_numeric(sub["fit_r2"], errors="coerce").median()),
                "edge_fraction": float(sub["fit_status"].isin(["lower_edge", "upper_edge"]).mean()) if not sub.empty else float("nan"),
                "lower_edge_fraction": float(sub["fit_status"].eq("lower_edge").mean()) if not sub.empty else float("nan"),
                "upper_edge_fraction": float(sub["fit_status"].eq("upper_edge").mean()) if not sub.empty else float("nan"),
            }
        )
    for keys, sub in usable.groupby(["speed_family", "dimension", "dimension_label"], sort=True):
        family, dim, dim_label = keys
        high = sub[sub["speed_pref_group"] == "high_speed_preferring"]
        low = sub[sub["speed_pref_group"] == "low_speed_preferring"]
        t_stat, p_value = welch_ttest(high["fit_preferred_log2_value"], low["fit_preferred_log2_value"])
        high_vals = pd.to_numeric(high["fit_preferred_log2_value"], errors="coerce").dropna()
        low_vals = pd.to_numeric(low["fit_preferred_log2_value"], errors="coerce").dropna()
        tests.append(
            {
                "speed_family": family,
                "dimension": dim,
                "dimension_label": dim_label,
                "high_n": int(high_vals.shape[0]),
                "low_n": int(low_vals.shape[0]),
                "high_fit_pref_log2_mean": float(high_vals.mean()) if not high_vals.empty else float("nan"),
                "low_fit_pref_log2_mean": float(low_vals.mean()) if not low_vals.empty else float("nan"),
                "high_fit_pref_geomean_value": float(2.0 ** high_vals.mean()) if not high_vals.empty else float("nan"),
                "low_fit_pref_geomean_value": float(2.0 ** low_vals.mean()) if not low_vals.empty else float("nan"),
                "high_minus_low_log2": float(high_vals.mean() - low_vals.mean())
                if not high_vals.empty and not low_vals.empty
                else float("nan"),
                "welch_t": t_stat,
                "welch_p": p_value,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(tests)


def plot_curves(out_dir: Path, points: pd.DataFrame, group_summary: pd.DataFrame, *, dpi: int) -> Path:
    png = out_dir / "sf_tf_speed_tuning_curves_by_group.png"
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.4), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.08, top=0.86, hspace=0.5, wspace=0.28)
    fig.suptitle("Speed-controlled grating tuning curves by microsaccade-derived group", y=0.965, fontsize=16)
    fig.text(
        0.5,
        0.925,
        "Curves are within-unit z-scored RMS modulation; dashed lines mark group median fitted preference when available",
        ha="center",
        fontsize=10.5,
        color="0.35",
    )
    for row, family in enumerate(FAMILY_ORDER):
        for col, (dim, _, _, label) in enumerate(DIMENSIONS):
            ax = axes[row, col]
            sub = points[(points["speed_family"] == family) & (points["dimension"] == dim)]
            for group in GROUP_ORDER:
                ss = sub[sub["speed_pref_group"] == group]
                if ss.empty:
                    continue
                summary_rows = []
                for x, xx in ss.groupby(["stimulus_value", "log2_stimulus_value"], sort=True):
                    value, log_value = x
                    vals = pd.to_numeric(xx["unit_dimension_z"], errors="coerce").dropna()
                    summary_rows.append(
                        {
                            "stimulus_value": float(value),
                            "log2_stimulus_value": float(log_value),
                            "mean": float(vals.mean()) if not vals.empty else float("nan"),
                            "sem": sem(vals),
                        }
                    )
                cs = pd.DataFrame(summary_rows).sort_values("stimulus_value")
                color = GROUP_COLORS[group]
                ax.plot(cs["stimulus_value"], cs["mean"], color=color, marker="o", lw=2.2, label=GROUP_MAP_REV[group])
                ax.fill_between(
                    cs["stimulus_value"],
                    cs["mean"] - cs["sem"],
                    cs["mean"] + cs["sem"],
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
                pref = group_summary[
                    (group_summary["speed_family"] == family)
                    & (group_summary["dimension"] == dim)
                    & (group_summary["speed_pref_group"] == group)
                ]
                if not pref.empty and np.isfinite(float(pref["fit_pref_median_value"].iloc[0])):
                    ax.axvline(float(pref["fit_pref_median_value"].iloc[0]), color=color, ls="--", lw=1.4, alpha=0.7)
            ax.axhline(0, color="0.55", ls=":", lw=1)
            ax.set_xscale("log", base=2)
            ax.grid(True, color="0.9")
            ax.set_title(f"{FAMILY_LABELS.get(family, family)}: {label}")
            ax.set_xlabel(label)
            ax.set_ylabel("within-unit z")
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=9)
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    return png


def plot_fit_preferences(out_dir: Path, fits: pd.DataFrame, tests: pd.DataFrame, *, dpi: int) -> Path:
    png = out_dir / "sf_tf_speed_fit_preferences_by_group.png"
    usable = fits[fits["fit_ok"]].copy()
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.4), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.08, top=0.86, hspace=0.5, wspace=0.28)
    fig.suptitle("Fit-derived preferred SF, TF, and speed", y=0.965, fontsize=16)
    fig.text(
        0.5,
        0.925,
        "Bounded log-Gaussian fits; black points show group mean +/- SEM in log2 units",
        ha="center",
        fontsize=10.5,
        color="0.35",
    )
    rng = np.random.default_rng(90210)
    for row, family in enumerate(FAMILY_ORDER):
        for col, (dim, _, _, label) in enumerate(DIMENSIONS):
            ax = axes[row, col]
            sub = usable[(usable["speed_family"] == family) & (usable["dimension"] == dim)]
            for x, group in enumerate(GROUP_ORDER):
                vals = pd.to_numeric(
                    sub[sub["speed_pref_group"] == group]["fit_preferred_log2_value"], errors="coerce"
                ).dropna()
                if vals.empty:
                    continue
                color = GROUP_COLORS[group]
                jitter = rng.uniform(-0.08, 0.08, size=vals.shape[0])
                ax.scatter(
                    np.full(vals.shape[0], x) + jitter,
                    vals.to_numpy(dtype=float),
                    s=18,
                    color=color,
                    alpha=0.48,
                    edgecolor="none",
                )
                ax.errorbar(
                    [x],
                    [float(vals.mean())],
                    yerr=[sem(vals)],
                    color="black",
                    marker="o",
                    markersize=5,
                    capsize=4,
                    lw=1.5,
                )
            test = tests[(tests["speed_family"] == family) & (tests["dimension"] == dim)]
            p_text = ""
            if not test.empty and np.isfinite(float(test["welch_p"].iloc[0])):
                p_text = f"\nWelch p={float(test['welch_p'].iloc[0]):.3g}"
            ax.set_title(f"{FAMILY_LABELS.get(family, family)}: {label}{p_text}")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["high-speed\npref.", "low-speed\npref."])
            ax.set_ylabel(f"log2 {label}")
            ax.grid(True, axis="y", color="0.9")
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    return png


GROUP_MAP_REV = {
    "high_speed_preferring": "high-speed preferring",
    "low_speed_preferring": "low-speed preferring",
}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped = pd.read_csv(Path(args.probe_dir) / "speed_controlled_grouped.csv")
    groups = add_group_labels(pd.read_csv(args.groups_csv))
    group_cols = [
        "unit_index",
        "curve_group",
        "curve_group_label",
        "speed_pref_group",
        "speed_pref_label",
        "sf_group",
        "sf_group_label",
    ]
    points = build_tuning_points(grouped)
    points = points.merge(groups[group_cols], on="unit_index", how="inner")
    fits = fit_all_units(points, min_points=int(args.min_fit_points), edge_frac=float(args.edge_frac))
    fits = fits.merge(groups[group_cols], on="unit_index", how="inner")
    group_summary, group_tests = summarize_groups(fits, min_r2=float(args.min_r2_for_summary))
    points.to_csv(out_dir / "sf_tf_speed_tuning_curve_points.csv", index=False)
    fits.to_csv(out_dir / "sf_tf_speed_tuning_fit_unit_summary.csv", index=False)
    group_summary.to_csv(out_dir / "sf_tf_speed_tuning_fit_group_summary.csv", index=False)
    group_tests.to_csv(out_dir / "sf_tf_speed_tuning_fit_group_tests.csv", index=False)
    curve_png = plot_curves(out_dir, points, group_summary, dpi=int(args.dpi))
    pref_png = plot_fit_preferences(out_dir, fits, group_tests, dpi=int(args.dpi))
    print(f"Wrote {curve_png}")
    print(f"Wrote {pref_png}")
    print(group_tests.to_string(index=False))


if __name__ == "__main__":
    main()
