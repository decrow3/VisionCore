#!/usr/bin/env python3
"""Compare dense SF/TF grating tuning for microsaccade-derived speed groups."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROBE_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_dense_sf_tf_grating_probe_v1"
)
DEFAULT_GROUPS_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_microsaccade_snippet_rr100_spatial_ssi_isotropic_padded_event_scaled_full_amp1sd_n40_v1/"
    "bimodal_unit_curve_groups/bimodal_unit_curve_groups.csv"
)
DEFAULT_OUT_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_dense_sf_tf_speed_pref_groups_v1"
)

EPS = 1e-12
FWHM_FACTOR = float(2.0 * math.sqrt(2.0 * math.log(2.0)))
GROUP_MAP = {
    "large_scale_preferring": ("high_speed_preferring", "high-speed preferring"),
    "small_scale_preferring": ("low_speed_preferring", "low-speed preferring"),
}
GROUP_ORDER = ["high_speed_preferring", "low_speed_preferring"]
GROUP_LABELS = {
    "high_speed_preferring": "high-speed preferring",
    "low_speed_preferring": "low-speed preferring",
}
GROUP_COLORS = {
    "high_speed_preferring": "#1f77b4",
    "low_speed_preferring": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--groups-csv", type=Path, default=DEFAULT_GROUPS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--speed-family", type=str, default="cycle_valid")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def sem(values: pd.Series | np.ndarray) -> float:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size <= 1:
        return 0.0
    return float(np.std(vals, ddof=1) / math.sqrt(vals.size))


def welch(a: pd.Series, b: pd.Series) -> tuple[float, float]:
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


def add_speed_group_labels(groups: pd.DataFrame) -> pd.DataFrame:
    groups = groups.copy()
    mapped = groups["curve_group"].map(GROUP_MAP)
    groups["speed_pref_group"] = [item[0] if isinstance(item, tuple) else None for item in mapped]
    groups["speed_pref_label"] = [item[1] if isinstance(item, tuple) else None for item in mapped]
    return groups[groups["speed_pref_group"].notna()].copy()


def load_points(probe_dir: Path, groups_csv: Path, *, speed_family: str) -> pd.DataFrame:
    surface = pd.read_csv(probe_dir / "dense_sf_tf_unit_surface.csv")
    groups = add_speed_group_labels(pd.read_csv(groups_csv))
    frame = surface[surface["speed_family"].astype(str).eq(str(speed_family))].copy()
    frame["log2_sf"] = np.log2(pd.to_numeric(frame["spatial_cpd"], errors="coerce"))
    frame["log2_tf"] = np.log2(pd.to_numeric(frame["temporal_hz"], errors="coerce"))
    frame = frame.merge(
        groups[
            [
                "unit_index",
                "curve_group",
                "curve_group_label",
                "speed_pref_group",
                "speed_pref_label",
                "sf_group",
                "sf_group_label",
            ]
        ],
        on="unit_index",
        how="inner",
    )
    z_parts: list[pd.DataFrame] = []
    for _, sub in frame.groupby("unit_index", sort=False):
        vals = pd.to_numeric(sub["response_amp_rms_mean"], errors="coerce").to_numpy(dtype=float)
        sd = float(np.nanstd(vals))
        out = sub.copy()
        out["unit_surface_z"] = (vals - float(np.nanmean(vals))) / sd if sd > EPS else np.nan
        z_parts.append(out)
    return pd.concat(z_parts, ignore_index=True)


def gaussian2d(
    coord: tuple[np.ndarray, np.ndarray],
    baseline: float,
    amplitude: float,
    mux: float,
    muy: float,
    sx: float,
    sy: float,
) -> np.ndarray:
    x, y = coord
    sx = max(float(sx), EPS)
    sy = max(float(sy), EPS)
    return baseline + amplitude * np.exp(-0.5 * (np.square((x - mux) / sx) + np.square((y - muy) / sy)))


def fit_unit_surface(sub: pd.DataFrame) -> dict[str, Any]:
    x = sub["log2_sf"].to_numpy(dtype=float)
    y = sub["log2_tf"].to_numpy(dtype=float)
    z = sub["response_amp_rms_mean"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[finite], y[finite], z[finite]
    rec: dict[str, Any] = {
        "fit_ok": False,
        "fit_status": "insufficient_points",
        "fit_r2": float("nan"),
        "fit_rmse": float("nan"),
        "fit_pref_log2_sf": float("nan"),
        "fit_pref_log2_tf": float("nan"),
        "fit_pref_sf_cpd": float("nan"),
        "fit_pref_tf_hz": float("nan"),
        "fit_pref_speed_dps": float("nan"),
        "fit_sigma_sf_octaves": float("nan"),
        "fit_sigma_tf_octaves": float("nan"),
        "fit_fwhm_sf_octaves": float("nan"),
        "fit_fwhm_tf_octaves": float("nan"),
        "fit_fwhm_area_octave2": float("nan"),
        "fit_edge_sf": False,
        "fit_edge_tf": False,
        "observed_peak_sf_cpd": float("nan"),
        "observed_peak_tf_hz": float("nan"),
        "observed_peak_speed_dps": float("nan"),
        "observed_peak_response": float("nan"),
        "n_points": int(x.size),
    }
    if x.size:
        peak_idx = int(np.nanargmax(z))
        rec.update(
            {
                "observed_peak_sf_cpd": float(2.0 ** x[peak_idx]),
                "observed_peak_tf_hz": float(2.0 ** y[peak_idx]),
                "observed_peak_speed_dps": float(2.0 ** (y[peak_idx] - x[peak_idx])),
                "observed_peak_response": float(z[peak_idx]),
            }
        )
    if x.size < 10 or float(np.nanmax(z) - np.nanmin(z)) <= EPS:
        return rec
    try:
        from scipy.optimize import curve_fit
    except Exception:
        rec["fit_status"] = "scipy_unavailable"
        return rec

    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    xrange = max(xmax - xmin, EPS)
    yrange = max(ymax - ymin, EPS)
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    zrange = max(zmax - zmin, EPS)
    baseline_hi = max(zmax * 2.0, zmin + 2.0 * zrange, EPS)
    amplitude_hi = max(zmax * 5.0, zrange * 10.0, EPS)
    bounds = (
        [0.0, 0.0, xmin, ymin, 0.15, 0.15],
        [baseline_hi, amplitude_hi, xmax, ymax, max(0.5, 2.0 * xrange), max(0.5, 2.0 * yrange)],
    )
    peak_idx = int(np.nanargmax(z))
    p0s: list[list[float]] = []
    for sx in [0.5, 1.0, 2.0, 4.0, max(0.5, xrange / 2.0)]:
        for sy in [0.5, 1.0, 2.0, 4.0, max(0.5, yrange / 2.0)]:
            p0s.append(
                [
                    max(zmin, 0.0),
                    zrange,
                    float(np.clip(x[peak_idx], xmin, xmax)),
                    float(np.clip(y[peak_idx], ymin, ymax)),
                    min(float(sx), bounds[1][4]),
                    min(float(sy), bounds[1][5]),
                ]
            )
    best: tuple[float, np.ndarray] | None = None
    for p0 in p0s:
        try:
            params, _ = curve_fit(gaussian2d, (x, y), z, p0=p0, bounds=bounds, maxfev=25000)
        except Exception:
            continue
        pred = gaussian2d((x, y), *params)
        rss = float(np.nansum(np.square(z - pred)))
        if best is None or rss < best[0]:
            best = (rss, params)
    if best is None:
        rec["fit_status"] = "fit_failed"
        return rec

    rss, params = best
    pred = gaussian2d((x, y), *params)
    tss = float(np.nansum(np.square(z - float(np.nanmean(z)))))
    r2 = float(1.0 - rss / tss) if tss > EPS else float("nan")
    baseline, amplitude, mux, muy, sx, sy = [float(v) for v in params]
    del baseline, amplitude
    edge_sf = bool(mux <= xmin + 0.05 * xrange or mux >= xmax - 0.05 * xrange)
    edge_tf = bool(muy <= ymin + 0.05 * yrange or muy >= ymax - 0.05 * yrange)
    if edge_sf and edge_tf:
        status = "sf_tf_edge"
    elif edge_sf:
        status = "sf_edge"
    elif edge_tf:
        status = "tf_edge"
    else:
        status = "interior"
    rec.update(
        {
            "fit_ok": True,
            "fit_status": status,
            "fit_r2": r2,
            "fit_rmse": float(math.sqrt(rss / max(x.size, 1))),
            "fit_pref_log2_sf": mux,
            "fit_pref_log2_tf": muy,
            "fit_pref_sf_cpd": float(2.0**mux),
            "fit_pref_tf_hz": float(2.0**muy),
            "fit_pref_speed_dps": float(2.0 ** (muy - mux)),
            "fit_sigma_sf_octaves": sx,
            "fit_sigma_tf_octaves": sy,
            "fit_fwhm_sf_octaves": FWHM_FACTOR * sx,
            "fit_fwhm_tf_octaves": FWHM_FACTOR * sy,
            "fit_fwhm_area_octave2": float((FWHM_FACTOR * sx) * (FWHM_FACTOR * sy)),
            "fit_edge_sf": edge_sf,
            "fit_edge_tf": edge_tf,
        }
    )
    return rec


def fit_all(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "unit_index",
        "unit_label",
        "speed_family",
        "curve_group",
        "curve_group_label",
        "speed_pref_group",
        "speed_pref_label",
        "sf_group",
        "sf_group_label",
    ]
    for key_values, sub in points.groupby(keys, sort=True):
        rec = dict(zip(keys, key_values, strict=True))
        rec["unit_index"] = int(rec["unit_index"])
        rec.update(fit_unit_surface(sub))
        rows.append(rec)
    return pd.DataFrame(rows)


def group_surface(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["speed_pref_group", "speed_pref_label", "spatial_cpd", "temporal_hz", "log2_sf", "log2_tf"]
    for key_values, sub in points.groupby(keys, sort=True):
        group, label, sf, tf, log_sf, log_tf = key_values
        vals = pd.to_numeric(sub["unit_surface_z"], errors="coerce").dropna()
        rows.append(
            {
                "speed_pref_group": group,
                "speed_pref_label": label,
                "spatial_cpd": float(sf),
                "temporal_hz": float(tf),
                "log2_sf": float(log_sf),
                "log2_tf": float(log_tf),
                "speed_dps": float(tf) / max(float(sf), EPS),
                "log2_speed_dps": float(log_tf) - float(log_sf),
                "mean_unit_z": float(vals.mean()) if not vals.empty else float("nan"),
                "sem_unit_z": sem(vals),
                "n_units": int(vals.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def marginal_curves(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    marg_rows: list[dict[str, Any]] = []
    frame = points.copy()
    frame["speed_bin_log2"] = np.round(pd.to_numeric(frame["log2_speed_dps"], errors="coerce") * 2.0) / 2.0
    frame["speed_bin_dps"] = np.power(2.0, frame["speed_bin_log2"])
    axis_specs = [
        ("sf", "spatial_cpd", "log2_spatial_cpd"),
        ("tf", "temporal_hz", "log2_temporal_hz"),
        ("speed", "speed_bin_dps", "speed_bin_log2"),
    ]
    for axis, xcol, log_col in axis_specs:
        group_cols = ["unit_index", "unit_label", "speed_pref_group", "speed_pref_label", xcol, log_col]
        for keys, sub in frame.groupby(group_cols, sort=True):
            unit, label, group, group_label, xval, log_xval = keys
            zvals = pd.to_numeric(sub["unit_surface_z"], errors="coerce").dropna()
            ampvals = pd.to_numeric(sub["response_amp_rms_mean"], errors="coerce").dropna()
            marg_rows.append(
                {
                    "axis": axis,
                    "unit_index": int(unit),
                    "unit_label": str(label),
                    "speed_pref_group": group,
                    "speed_pref_label": group_label,
                    "x_value": float(xval),
                    "log2_x_value": float(log_xval),
                    "unit_z_mean": float(zvals.mean()) if not zvals.empty else float("nan"),
                    "response_amp_mean": float(ampvals.mean()) if not ampvals.empty else float("nan"),
                    "n_points": int(sub.shape[0]),
                }
            )
    marg = pd.DataFrame(marg_rows)
    summary_rows: list[dict[str, Any]] = []
    for keys, sub in marg.groupby(["axis", "speed_pref_group", "speed_pref_label", "x_value", "log2_x_value"], sort=True):
        axis, group, label, xval, logx = keys
        vals = pd.to_numeric(sub["unit_z_mean"], errors="coerce").dropna()
        amps = pd.to_numeric(sub["response_amp_mean"], errors="coerce").dropna()
        summary_rows.append(
            {
                "axis": axis,
                "speed_pref_group": group,
                "speed_pref_label": label,
                "x_value": float(xval),
                "log2_x_value": float(logx),
                "unit_z_mean": float(vals.mean()) if not vals.empty else float("nan"),
                "unit_z_sem": sem(vals),
                "response_amp_mean": float(amps.mean()) if not amps.empty else float("nan"),
                "response_amp_sem": sem(amps),
                "n_units": int(vals.shape[0]),
            }
        )
    summary = pd.DataFrame(summary_rows)

    metric_rows: list[dict[str, Any]] = []
    for keys, sub in marg.groupby(["axis", "unit_index", "unit_label", "speed_pref_group", "speed_pref_label"], sort=True):
        axis, unit, label, group, group_label = keys
        sub = sub.sort_values("x_value")
        x = sub["x_value"].to_numpy(dtype=float)
        lx = sub["log2_x_value"].to_numpy(dtype=float)
        z = sub["unit_z_mean"].to_numpy(dtype=float)
        amp = sub["response_amp_mean"].to_numpy(dtype=float)
        finite = np.isfinite(lx) & np.isfinite(z) & np.isfinite(amp)
        if finite.sum() < 3:
            continue
        x, lx, z, amp = x[finite], lx[finite], z[finite], amp[finite]
        weights = amp - float(np.nanmin(amp)) + EPS
        peak_idx = int(np.nanargmax(amp))
        z_peak_idx = int(np.nanargmax(z))
        slope = float(np.polyfit(lx, z, 1)[0]) if np.unique(lx).size >= 2 else float("nan")
        metric_rows.append(
            {
                "axis": axis,
                "unit_index": int(unit),
                "unit_label": str(label),
                "speed_pref_group": group,
                "speed_pref_label": group_label,
                "peak_x_by_amp": float(x[peak_idx]),
                "peak_log2_x_by_amp": float(lx[peak_idx]),
                "peak_x_by_z": float(x[z_peak_idx]),
                "peak_log2_x_by_z": float(lx[z_peak_idx]),
                "amp_weighted_log2_x": float(np.nansum(lx * weights) / max(float(np.nansum(weights)), EPS)),
                "z_slope_vs_log2_x": slope,
                "z_dynamic_range": float(np.nanmax(z) - np.nanmin(z)),
                "amp_dynamic_range": float(np.nanmax(amp) - np.nanmin(amp)),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    return marg, summary, metrics


def summarize_fit_tests(fits: pd.DataFrame, metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_metrics = [
        "fit_pref_log2_sf",
        "fit_pref_log2_tf",
        "fit_pref_speed_dps",
        "fit_fwhm_sf_octaves",
        "fit_fwhm_tf_octaves",
        "fit_fwhm_area_octave2",
        "fit_r2",
    ]
    rows: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    for subset, sub in [
        ("all_fit_ok", fits[fits["fit_ok"].astype(bool)]),
        ("interior_only", fits[fits["fit_ok"].astype(bool) & ~(fits["fit_edge_sf"].astype(bool) | fits["fit_edge_tf"].astype(bool))]),
    ]:
        for group in GROUP_ORDER:
            ss = sub[sub["speed_pref_group"].eq(group)]
            rec = {
                "subset": subset,
                "speed_pref_group": group,
                "speed_pref_label": GROUP_LABELS[group],
                "n_units": int(ss.shape[0]),
                "edge_sf_fraction": float(ss["fit_edge_sf"].mean()) if ss.shape[0] else float("nan"),
                "edge_tf_fraction": float(ss["fit_edge_tf"].mean()) if ss.shape[0] else float("nan"),
            }
            for metric in fit_metrics:
                vals = pd.to_numeric(ss[metric], errors="coerce").dropna()
                rec[f"{metric}_mean"] = float(vals.mean()) if not vals.empty else float("nan")
                rec[f"{metric}_sem"] = sem(vals)
                rec[f"{metric}_median"] = float(vals.median()) if not vals.empty else float("nan")
            rows.append(rec)
        high = sub[sub["speed_pref_group"].eq("high_speed_preferring")]
        low = sub[sub["speed_pref_group"].eq("low_speed_preferring")]
        for metric in fit_metrics:
            high_vals = pd.to_numeric(high[metric], errors="coerce").dropna()
            low_vals = pd.to_numeric(low[metric], errors="coerce").dropna()
            t_stat, p_value = welch(high_vals, low_vals)
            tests.append(
                {
                    "source": "2d_fit",
                    "subset": subset,
                    "axis": "",
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

    marginal_metrics = ["peak_log2_x_by_amp", "amp_weighted_log2_x", "z_slope_vs_log2_x", "z_dynamic_range", "amp_dynamic_range"]
    for axis, sub in metrics.groupby("axis", sort=True):
        high = sub[sub["speed_pref_group"].eq("high_speed_preferring")]
        low = sub[sub["speed_pref_group"].eq("low_speed_preferring")]
        for metric in marginal_metrics:
            high_vals = pd.to_numeric(high[metric], errors="coerce").dropna()
            low_vals = pd.to_numeric(low[metric], errors="coerce").dropna()
            t_stat, p_value = welch(high_vals, low_vals)
            tests.append(
                {
                    "source": "marginal",
                    "subset": "",
                    "axis": axis,
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


def design_r2(frame: pd.DataFrame, y_col: str, terms: list[str]) -> float:
    y = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=float)
    mats: list[np.ndarray] = []
    for term in terms:
        if term == "intercept":
            mats.append(np.ones((frame.shape[0], 1), dtype=float))
        elif term == "log2_sf":
            mats.append(pd.to_numeric(frame["log2_sf"], errors="coerce").to_numpy(dtype=float)[:, None])
        elif term == "log2_tf":
            mats.append(pd.to_numeric(frame["log2_tf"], errors="coerce").to_numpy(dtype=float)[:, None])
        elif term == "log2_speed":
            mats.append(pd.to_numeric(frame["log2_speed_dps"], errors="coerce").to_numpy(dtype=float)[:, None])
        elif term == "sf_cat":
            mats.append(pd.get_dummies(frame["spatial_cpd"].astype(str), drop_first=True).to_numpy(dtype=float))
        elif term == "tf_cat":
            mats.append(pd.get_dummies(frame["temporal_hz"].astype(str), drop_first=True).to_numpy(dtype=float))
        elif term == "speed_cat":
            speed_bin = np.round(pd.to_numeric(frame["log2_speed_dps"], errors="coerce") * 2.0) / 2.0
            mats.append(pd.get_dummies(speed_bin.astype(str), drop_first=True).to_numpy(dtype=float))
        else:
            raise ValueError(f"Unknown design term: {term}")
    x = np.hstack(mats)
    finite = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y = y[finite]
    x = x[finite]
    if y.size < 2:
        return float("nan")
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    pred = x @ beta
    ss_res = float(np.nansum(np.square(y - pred)))
    ss_tot = float(np.nansum(np.square(y - float(np.nanmean(y)))))
    return float(1.0 - ss_res / ss_tot) if ss_tot > EPS else float("nan")


def decompose_group_difference(surface: pd.DataFrame) -> pd.DataFrame:
    high = surface[surface["speed_pref_group"].eq("high_speed_preferring")][
        ["spatial_cpd", "temporal_hz", "log2_sf", "log2_tf", "mean_unit_z"]
    ].rename(columns={"mean_unit_z": "high_z"})
    low = surface[surface["speed_pref_group"].eq("low_speed_preferring")][
        ["spatial_cpd", "temporal_hz", "log2_sf", "log2_tf", "mean_unit_z"]
    ].rename(columns={"mean_unit_z": "low_z"})
    diff = high.merge(low, on=["spatial_cpd", "temporal_hz", "log2_sf", "log2_tf"], how="inner")
    diff["high_minus_low_z"] = diff["high_z"] - diff["low_z"]
    diff["log2_speed_dps"] = diff["log2_tf"] - diff["log2_sf"]
    models = [
        ("linear_sf_only", ["intercept", "log2_sf"], "linear SF only"),
        ("linear_tf_only", ["intercept", "log2_tf"], "linear TF only"),
        ("linear_speed_only", ["intercept", "log2_speed"], "linear derived-speed only"),
        ("categorical_sf_only", ["intercept", "sf_cat"], "categorical SF only"),
        ("categorical_tf_only", ["intercept", "tf_cat"], "categorical TF only"),
        ("categorical_speed_only", ["intercept", "speed_cat"], "categorical half-octave speed only"),
        ("linear_sf_plus_tf", ["intercept", "log2_sf", "log2_tf"], "linear SF + TF"),
        ("categorical_sf_plus_tf_additive", ["intercept", "sf_cat", "tf_cat"], "categorical SF + TF additive"),
    ]
    rows = []
    for model_id, terms, label in models:
        rows.append(
            {
                "model_id": model_id,
                "model_label": label,
                "terms": ",".join(terms),
                "r2_high_minus_low_z": design_r2(diff, "high_minus_low_z", terms),
                "n_points": int(diff.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def tick_values(values: pd.Series) -> tuple[list[float], list[str]]:
    unique = sorted(float(v) for v in pd.unique(values.dropna()))
    return [float(np.log2(v)) for v in unique], [f"{v:g}" for v in unique]


def surface_matrix(surface: pd.DataFrame, group: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = surface[surface["speed_pref_group"].eq(group)].copy()
    xs = np.array(sorted(sub["spatial_cpd"].unique()), dtype=float)
    ys = np.array(sorted(sub["temporal_hz"].unique()), dtype=float)
    grid = sub.pivot_table(index="temporal_hz", columns="spatial_cpd", values="mean_unit_z", aggfunc="mean")
    grid = grid.reindex(index=ys, columns=xs)
    return xs, ys, grid.to_numpy(dtype=float)


def add_ellipses(ax: plt.Axes, fits: pd.DataFrame, group: str) -> None:
    sub = fits[fits["fit_ok"].astype(bool) & fits["speed_pref_group"].eq(group)]
    color = GROUP_COLORS[group]
    for _, row in sub.iterrows():
        edge = bool(row["fit_edge_sf"]) or bool(row["fit_edge_tf"])
        ell = Ellipse(
            (float(row["fit_pref_log2_sf"]), float(row["fit_pref_log2_tf"])),
            width=float(row["fit_fwhm_sf_octaves"]),
            height=float(row["fit_fwhm_tf_octaves"]),
            fill=False,
            edgecolor=color,
            linewidth=0.7 if edge else 0.9,
            linestyle="--" if edge else "-",
            alpha=0.13 if edge else 0.26,
        )
        ax.add_patch(ell)
    interior = sub[~(sub["fit_edge_sf"].astype(bool) | sub["fit_edge_tf"].astype(bool))]
    use = interior if not interior.empty else sub
    if not use.empty:
        med = use[["fit_pref_log2_sf", "fit_pref_log2_tf", "fit_fwhm_sf_octaves", "fit_fwhm_tf_octaves"]].median(numeric_only=True)
        ell = Ellipse(
            (float(med["fit_pref_log2_sf"]), float(med["fit_pref_log2_tf"])),
            width=float(med["fit_fwhm_sf_octaves"]),
            height=float(med["fit_fwhm_tf_octaves"]),
            fill=False,
            edgecolor="black",
            linewidth=2.0,
            alpha=0.9,
        )
        ax.add_patch(ell)
        ax.scatter([float(med["fit_pref_log2_sf"])], [float(med["fit_pref_log2_tf"])], color="black", s=22, zorder=5)


def plot_dense_comparison(
    out_dir: Path,
    points: pd.DataFrame,
    surface: pd.DataFrame,
    fits: pd.DataFrame,
    marginal_summary: pd.DataFrame,
    tests: pd.DataFrame,
    *,
    speed_family: str,
    dpi: int,
) -> Path:
    png = out_dir / f"{speed_family}_dense_sf_tf_speed_pref_group_comparison.png"
    pdf = out_dir / f"{speed_family}_dense_sf_tf_speed_pref_group_comparison.pdf"
    fig, axes = plt.subplots(3, 3, figsize=(17.2, 14.2), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.975, bottom=0.06, top=0.89, hspace=0.42, wspace=0.3)
    fig.suptitle("Dense SF/TF grating tuning by microsaccade-derived speed-preference group", y=0.975, fontsize=16)
    fig.text(
        0.5,
        0.94,
        "Groups are fixed from event-scaled microsaccade SSI curves; color maps are group-mean within-unit z of response amplitude",
        ha="center",
        color="0.35",
        fontsize=10.5,
    )
    xticks, xlabels = tick_values(points["spatial_cpd"])
    yticks, ylabels = tick_values(points["temporal_hz"])
    vmax = max(float(np.nanmax(np.abs(surface["mean_unit_z"].to_numpy(dtype=float)))), 0.5)
    ims = []
    for col, group in enumerate(GROUP_ORDER):
        ax = axes[0, col]
        xs, ys, grid = surface_matrix(surface, group)
        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=[np.log2(xs.min()), np.log2(xs.max()), np.log2(ys.min()), np.log2(ys.max())],
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ims.append(im)
        ax.scatter(surface[surface["speed_pref_group"].eq(group)]["log2_sf"], surface[surface["speed_pref_group"].eq(group)]["log2_tf"], s=11, color="black", alpha=0.35)
        ax.set_title(f"{GROUP_LABELS[group]} (n={points[points['speed_pref_group'].eq(group)]['unit_index'].nunique()})")
        ax.set_xlabel("SF (cpd)")
        ax.set_ylabel("TF (Hz)")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.grid(False)
    ax = axes[0, 2]
    high = surface[surface["speed_pref_group"].eq("high_speed_preferring")][["spatial_cpd", "temporal_hz", "log2_sf", "log2_tf", "mean_unit_z"]].rename(columns={"mean_unit_z": "high_z"})
    low = surface[surface["speed_pref_group"].eq("low_speed_preferring")][["spatial_cpd", "temporal_hz", "log2_sf", "log2_tf", "mean_unit_z"]].rename(columns={"mean_unit_z": "low_z"})
    diff = high.merge(low, on=["spatial_cpd", "temporal_hz", "log2_sf", "log2_tf"], how="inner")
    diff["high_minus_low_z"] = diff["high_z"] - diff["low_z"]
    xs = np.array(sorted(diff["spatial_cpd"].unique()), dtype=float)
    ys = np.array(sorted(diff["temporal_hz"].unique()), dtype=float)
    dgrid = diff.pivot_table(index="temporal_hz", columns="spatial_cpd", values="high_minus_low_z", aggfunc="mean").reindex(index=ys, columns=xs).to_numpy(dtype=float)
    dvmax = max(float(np.nanmax(np.abs(dgrid))), 0.5)
    ax.imshow(
        dgrid,
        origin="lower",
        aspect="auto",
        extent=[np.log2(xs.min()), np.log2(xs.max()), np.log2(ys.min()), np.log2(ys.max())],
        cmap="coolwarm",
        vmin=-dvmax,
        vmax=dvmax,
        interpolation="nearest",
    )
    ax.set_title("high minus low")
    ax.set_xlabel("SF (cpd)")
    ax.set_ylabel("TF (Hz)")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)

    cax = fig.add_axes([0.982, 0.675, 0.009, 0.19])
    fig.colorbar(ims[0], cax=cax, label="mean within-unit z")

    for col, group in enumerate(GROUP_ORDER):
        ax = axes[1, col]
        add_ellipses(ax, fits, group)
        ss = surface[surface["speed_pref_group"].eq(group)]
        ax.scatter(ss["log2_sf"], ss["log2_tf"], s=12, color="0.55", alpha=0.45)
        ax.set_title(f"{GROUP_LABELS[group]} fit ellipses")
        ax.set_xlabel("SF (cpd)")
        ax.set_ylabel("TF (Hz)")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_xlim(min(xticks) - 0.5, max(xticks) + 0.5)
        ax.set_ylim(min(yticks) - 0.6, max(yticks) + 0.6)
        ax.grid(True, color="0.9", linewidth=0.55)
    ax = axes[1, 2]
    rng = np.random.default_rng(99)
    for xloc, group in enumerate(GROUP_ORDER):
        ss = fits[fits["fit_ok"].astype(bool) & fits["speed_pref_group"].eq(group)]
        vals = pd.to_numeric(ss["fit_fwhm_area_octave2"], errors="coerce").dropna()
        jitter = rng.uniform(-0.08, 0.08, size=vals.shape[0])
        ax.scatter(np.full(vals.shape[0], xloc) + jitter, vals, color=GROUP_COLORS[group], alpha=0.55, s=24)
        ax.errorbar([xloc], [float(vals.mean())], yerr=[sem(vals)], color="black", marker="o", capsize=4)
    area_test = tests[(tests["source"].eq("2d_fit")) & (tests["subset"].eq("all_fit_ok")) & (tests["metric"].eq("fit_fwhm_area_octave2"))]
    interior_area = tests[(tests["source"].eq("2d_fit")) & (tests["subset"].eq("interior_only")) & (tests["metric"].eq("fit_fwhm_area_octave2"))]
    title = "2D bandwidth area"
    if not area_test.empty:
        title += f"\nall p={float(area_test['welch_p'].iloc[0]):.3g}"
    if not interior_area.empty:
        title += f"; interior p={float(interior_area['welch_p'].iloc[0]):.3g}"
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["high-speed\npref.", "low-speed\npref."])
    ax.set_ylabel("FWHM SF x TF (octave^2)")
    ax.grid(True, axis="y", color="0.9")

    for col, axis_name in enumerate(["sf", "tf", "speed"]):
        ax = axes[2, col]
        for group in GROUP_ORDER:
            sub = marginal_summary[
                marginal_summary["axis"].eq(axis_name) & marginal_summary["speed_pref_group"].eq(group)
            ].sort_values("x_value")
            color = GROUP_COLORS[group]
            ax.plot(sub["x_value"], sub["unit_z_mean"], color=color, marker="o", lw=2.0, ms=4, label=GROUP_LABELS[group])
            ax.fill_between(
                sub["x_value"],
                sub["unit_z_mean"] - sub["unit_z_sem"],
                sub["unit_z_mean"] + sub["unit_z_sem"],
                color=color,
                alpha=0.15,
                linewidth=0,
            )
        axis_tests = tests[(tests["source"].eq("marginal")) & (tests["axis"].eq(axis_name))]
        p_peak = axis_tests[axis_tests["metric"].eq("amp_weighted_log2_x")]
        title = {"sf": "SF marginal", "tf": "TF marginal", "speed": "derived-speed marginal"}[axis_name]
        if not p_peak.empty:
            title += f"\nweighted log2 p={float(p_peak['welch_p'].iloc[0]):.3g}"
        ax.set_title(title)
        ax.axhline(0, color="0.45", lw=1.0, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xlabel({"sf": "SF (cpd)", "tf": "TF (Hz)", "speed": "TF/SF speed (deg/s)"}[axis_name])
        ax.set_ylabel("within-unit z")
        ax.grid(True, color="0.9")
        if col == 0:
            ax.legend(frameon=False, fontsize=8)

    fig.savefig(png, dpi=int(dpi))
    fig.savefig(pdf)
    plt.close(fig)
    return png


def plot_examples(out_dir: Path, points: pd.DataFrame, fits: pd.DataFrame, *, speed_family: str, dpi: int) -> Path:
    png = out_dir / f"{speed_family}_dense_sf_tf_example_unit_surfaces.png"
    chosen: list[int] = []
    for group in GROUP_ORDER:
        sub = fits[fits["fit_ok"].astype(bool) & fits["speed_pref_group"].eq(group)].copy()
        sub["is_edge"] = sub["fit_edge_sf"].astype(bool) | sub["fit_edge_tf"].astype(bool)
        sub["score"] = pd.to_numeric(sub["fit_r2"], errors="coerce") - 0.25 * sub["is_edge"].astype(float)
        chosen.extend(sub.sort_values("score", ascending=False).head(4)["unit_index"].astype(int).to_list())
    n_rows = len(chosen)
    fig, axes = plt.subplots(n_rows, 1, figsize=(7.2, max(2.25 * n_rows, 5.5)), constrained_layout=False)
    if n_rows == 1:
        axes = np.asarray([axes])
    fig.subplots_adjust(left=0.13, right=0.965, bottom=0.055, top=0.915, hspace=0.68)
    fig.suptitle("Example dense SF/TF unit surfaces", y=0.982, fontsize=15)
    fig.text(0.5, 0.948, "Color is within-unit z over dense SF/TF pairs; black ellipse is fitted FWHM", ha="center", fontsize=10, color="0.35")
    xticks, xlabels = tick_values(points["spatial_cpd"])
    yticks, ylabels = tick_values(points["temporal_hz"])
    for ax, unit in zip(axes, chosen, strict=True):
        ss = points[points["unit_index"].eq(unit)].copy()
        fit = fits[fits["unit_index"].eq(unit)].iloc[0]
        group = str(fit["speed_pref_group"])
        x = ss["log2_sf"].to_numpy(dtype=float)
        y = ss["log2_tf"].to_numpy(dtype=float)
        z = ss["unit_surface_z"].to_numpy(dtype=float)
        vmax = max(float(np.nanmax(np.abs(z))), 0.5)
        tri = mtri.Triangulation(x, y)
        ax.tricontourf(tri, z, levels=np.linspace(-vmax, vmax, 17), cmap="coolwarm", vmin=-vmax, vmax=vmax, alpha=0.9)
        ax.scatter(x, y, c=z, cmap="coolwarm", vmin=-vmax, vmax=vmax, s=32, edgecolor="black", linewidth=0.25)
        if bool(fit["fit_ok"]):
            ell = Ellipse(
                (float(fit["fit_pref_log2_sf"]), float(fit["fit_pref_log2_tf"])),
                width=float(fit["fit_fwhm_sf_octaves"]),
                height=float(fit["fit_fwhm_tf_octaves"]),
                fill=False,
                edgecolor="black",
                linewidth=1.7,
                linestyle="--" if bool(fit["fit_edge_sf"]) or bool(fit["fit_edge_tf"]) else "-",
            )
            ax.add_patch(ell)
            ax.scatter([float(fit["fit_pref_log2_sf"])], [float(fit["fit_pref_log2_tf"])], color="black", s=20)
        ax.set_title(
            f"{fit['unit_label']} | {GROUP_LABELS[group]} | pref TF={float(fit['fit_pref_tf_hz']):.2g} Hz, "
            f"speed={float(fit['fit_pref_speed_dps']):.2g} deg/s, area={float(fit['fit_fwhm_area_octave2']):.1f}, {fit['fit_status']}",
            color=GROUP_COLORS[group],
            fontsize=9.3,
        )
        ax.set_xlabel("SF (cpd)")
        ax.set_ylabel("TF (Hz)")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.grid(True, color="0.88", linewidth=0.55)
    fig.savefig(png, dpi=int(dpi))
    fig.savefig(out_dir / f"{speed_family}_dense_sf_tf_example_unit_surfaces.pdf")
    plt.close(fig)
    return png


def plot_bandwidth_summary(out_dir: Path, fits: pd.DataFrame, tests: pd.DataFrame, *, speed_family: str, dpi: int) -> Path:
    png = out_dir / f"{speed_family}_dense_sf_tf_bandwidth_summary.png"
    pdf = out_dir / f"{speed_family}_dense_sf_tf_bandwidth_summary.pdf"
    metrics = [
        ("fit_fwhm_sf_octaves", "SF bandwidth\nFWHM (octaves)"),
        ("fit_fwhm_tf_octaves", "TF bandwidth\nFWHM (octaves)"),
        ("fit_fwhm_area_octave2", "2D area\nSF x TF (octave^2)"),
    ]
    subsets = [
        ("all_fit_ok", "All successful fits"),
        ("interior_only", "Interior fits only"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.6), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.085, top=0.84, hspace=0.42, wspace=0.34)
    fig.suptitle("Dense SF/TF fitted bandwidths by microsaccade speed-preference group", y=0.965, fontsize=15)
    fig.text(
        0.5,
        0.925,
        "FWHM values are from axis-aligned 2D log-Gaussian fits to response amplitude over SF/TF points",
        ha="center",
        color="0.35",
        fontsize=10.2,
    )
    rng = np.random.default_rng(121)
    for row, (subset, subset_label) in enumerate(subsets):
        if subset == "all_fit_ok":
            subfits = fits[fits["fit_ok"].astype(bool)].copy()
        else:
            subfits = fits[fits["fit_ok"].astype(bool) & ~(fits["fit_edge_sf"].astype(bool) | fits["fit_edge_tf"].astype(bool))].copy()
        for col, (metric, label) in enumerate(metrics):
            ax = axes[row, col]
            test = tests[
                tests["source"].eq("2d_fit")
                & tests["subset"].eq(subset)
                & tests["metric"].eq(metric)
            ]
            for xloc, group in enumerate(GROUP_ORDER):
                vals = pd.to_numeric(subfits[subfits["speed_pref_group"].eq(group)][metric], errors="coerce").dropna()
                jitter = rng.uniform(-0.08, 0.08, size=vals.shape[0])
                ax.scatter(
                    np.full(vals.shape[0], xloc) + jitter,
                    vals,
                    color=GROUP_COLORS[group],
                    alpha=0.58,
                    s=24,
                    edgecolors="none",
                )
                if not vals.empty:
                    ax.errorbar([xloc], [float(vals.mean())], yerr=[sem(vals)], color="black", marker="o", capsize=4)
            p_text = ""
            if not test.empty:
                high = float(test["high_mean"].iloc[0])
                low = float(test["low_mean"].iloc[0])
                p_text = f"\nhigh={high:.3g}, low={low:.3g}, p={float(test['welch_p'].iloc[0]):.3g}"
            ax.set_title(f"{subset_label}: {label}{p_text}", fontsize=9.6)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["high-speed\npref.", "low-speed\npref."])
            ax.grid(True, axis="y", color="0.9")
            if col == 0:
                ax.set_ylabel("FWHM")
    fig.savefig(png, dpi=int(dpi))
    fig.savefig(pdf)
    plt.close(fig)
    return png


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = load_points(Path(args.probe_dir), Path(args.groups_csv), speed_family=str(args.speed_family))
    fits = fit_all(points)
    surface = group_surface(points)
    marg, marg_summary, marg_metrics = marginal_curves(points)
    fit_summary, tests = summarize_fit_tests(fits, marg_metrics)
    decomposition = decompose_group_difference(surface)

    points.to_csv(out_dir / f"{args.speed_family}_dense_sf_tf_points.csv", index=False)
    surface.to_csv(out_dir / f"{args.speed_family}_dense_sf_tf_group_surface.csv", index=False)
    fits.to_csv(out_dir / f"{args.speed_family}_dense_sf_tf_fit_unit_summary.csv", index=False)
    fit_summary.to_csv(out_dir / f"{args.speed_family}_dense_sf_tf_fit_group_summary.csv", index=False)
    marg.to_csv(out_dir / f"{args.speed_family}_dense_sf_tf_marginal_unit_curves.csv", index=False)
    marg_summary.to_csv(out_dir / f"{args.speed_family}_dense_sf_tf_marginal_group_curves.csv", index=False)
    marg_metrics.to_csv(out_dir / f"{args.speed_family}_dense_sf_tf_marginal_unit_metrics.csv", index=False)
    tests.to_csv(out_dir / f"{args.speed_family}_dense_sf_tf_group_tests.csv", index=False)
    decomposition.to_csv(out_dir / f"{args.speed_family}_dense_sf_tf_group_difference_decomposition.csv", index=False)

    comparison_png = plot_dense_comparison(
        out_dir,
        points,
        surface,
        fits,
        marg_summary,
        tests,
        speed_family=str(args.speed_family),
        dpi=int(args.dpi),
    )
    examples_png = plot_examples(out_dir, points, fits, speed_family=str(args.speed_family), dpi=int(args.dpi))
    bandwidth_png = plot_bandwidth_summary(out_dir, fits, tests, speed_family=str(args.speed_family), dpi=int(args.dpi))
    print(f"Wrote {comparison_png}")
    print(f"Wrote {examples_png}")
    print(f"Wrote {bandwidth_png}")
    print(decomposition.to_string(index=False))
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
