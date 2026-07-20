#!/usr/bin/env python3
"""Fit and plot 2D SF/TF tuning surfaces from the speed-controlled grating probe."""

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
    "backimage_rr100_speed_controlled_grating_probe_v1"
)
DEFAULT_GROUPS_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_microsaccade_snippet_rr100_spatial_ssi_isotropic_padded_event_scaled_full_amp1sd_n40_v1/"
    "bimodal_unit_curve_groups/bimodal_unit_curve_groups.csv"
)
DEFAULT_OUT_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_speed_controlled_2d_sf_tf_tuning_fits_v1"
)
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
EPS = 1e-12
FWHM_FACTOR = float(2.0 * math.sqrt(2.0 * math.log(2.0)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--groups-csv", type=Path, default=DEFAULT_GROUPS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--speed-family", type=str, default="cycle_valid")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def sem(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
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

        out = stats.ttest_ind(av, bv, equal_var=False, nan_policy="omit")
        return float(out.statistic), float(out.pvalue)
    except Exception:
        return float("nan"), float("nan")


def add_group_labels(groups: pd.DataFrame) -> pd.DataFrame:
    groups = groups.copy()
    mapped = groups["curve_group"].map(GROUP_MAP)
    groups["speed_pref_group"] = [item[0] if isinstance(item, tuple) else None for item in mapped]
    groups["speed_pref_label"] = [item[1] if isinstance(item, tuple) else None for item in mapped]
    return groups[groups["speed_pref_group"].notna()].copy()


def gaussian2d(coord: tuple[np.ndarray, np.ndarray], baseline: float, amplitude: float, mux: float, muy: float, sx: float, sy: float) -> np.ndarray:
    x, y = coord
    sx = np.maximum(float(sx), EPS)
    sy = np.maximum(float(sy), EPS)
    return baseline + amplitude * np.exp(-0.5 * (np.square((x - mux) / sx) + np.square((y - muy) / sy)))


def aggregate_points(grouped: pd.DataFrame, groups: pd.DataFrame, *, speed_family: str) -> pd.DataFrame:
    frame = grouped[grouped["speed_family"].astype(str).eq(str(speed_family))].copy()
    frame["log2_sf"] = np.log2(pd.to_numeric(frame["spatial_cpd"], errors="coerce"))
    frame["log2_tf"] = np.log2(pd.to_numeric(frame["temporal_hz"], errors="coerce"))
    points = (
        frame.groupby(
            ["unit_index", "unit_label", "speed_family", "pair_id", "spatial_cpd", "temporal_hz", "log2_sf", "log2_tf"],
            as_index=False,
            sort=True,
        )
        .agg(
            response_amp_rms=("response_amp_rms", "mean"),
            n_orientation_rows=("probe_orientation_deg", "size"),
            n_orientations=("probe_orientation_deg", "nunique"),
        )
        .merge(
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
    )
    z_parts = []
    for _, sub in points.groupby("unit_index", sort=False):
        vals = sub["response_amp_rms"].to_numpy(dtype=float)
        sd = float(np.nanstd(vals))
        out = sub.copy()
        out["unit_surface_z"] = (vals - float(np.nanmean(vals))) / sd if sd > EPS else np.nan
        z_parts.append(out)
    return pd.concat(z_parts, ignore_index=True)


def fit_unit_surface(sub: pd.DataFrame) -> dict[str, Any]:
    x = sub["log2_sf"].to_numpy(dtype=float)
    y = sub["log2_tf"].to_numpy(dtype=float)
    z = sub["response_amp_rms"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[finite], y[finite], z[finite]
    rec: dict[str, Any] = {
        "fit_ok": False,
        "fit_status": "insufficient_points",
        "fit_r2": float("nan"),
        "fit_rmse": float("nan"),
        "fit_baseline": float("nan"),
        "fit_amplitude": float("nan"),
        "fit_pref_log2_sf": float("nan"),
        "fit_pref_log2_tf": float("nan"),
        "fit_pref_sf_cpd": float("nan"),
        "fit_pref_tf_hz": float("nan"),
        "fit_sigma_sf_octaves": float("nan"),
        "fit_sigma_tf_octaves": float("nan"),
        "fit_fwhm_sf_octaves": float("nan"),
        "fit_fwhm_tf_octaves": float("nan"),
        "fit_fwhm_area_octave2": float("nan"),
        "fit_edge_sf": False,
        "fit_edge_tf": False,
        "observed_peak_sf_cpd": float("nan"),
        "observed_peak_tf_hz": float("nan"),
        "observed_peak_response": float("nan"),
        "n_points": int(x.size),
    }
    if x.size:
        peak_idx = int(np.nanargmax(z))
        rec.update(
            {
                "observed_peak_sf_cpd": float(2.0 ** x[peak_idx]),
                "observed_peak_tf_hz": float(2.0 ** y[peak_idx]),
                "observed_peak_response": float(z[peak_idx]),
            }
        )
    if x.size < 8 or float(np.nanmax(z) - np.nanmin(z)) <= EPS:
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
    p0s = []
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
            params, _ = curve_fit(gaussian2d, (x, y), z, p0=p0, bounds=bounds, maxfev=20000)
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
            "fit_baseline": baseline,
            "fit_amplitude": amplitude,
            "fit_pref_log2_sf": mux,
            "fit_pref_log2_tf": muy,
            "fit_pref_sf_cpd": float(2.0**mux),
            "fit_pref_tf_hz": float(2.0**muy),
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
    rows = []
    for keys, sub in points.groupby(
        [
            "unit_index",
            "unit_label",
            "speed_family",
            "curve_group",
            "curve_group_label",
            "speed_pref_group",
            "speed_pref_label",
            "sf_group",
            "sf_group_label",
        ],
        sort=True,
    ):
        (
            unit,
            label,
            family,
            curve_group,
            curve_group_label,
            speed_pref_group,
            speed_pref_label,
            sf_group,
            sf_group_label,
        ) = keys
        rows.append(
            {
                "unit_index": int(unit),
                "unit_label": str(label),
                "speed_family": str(family),
                "curve_group": str(curve_group),
                "curve_group_label": str(curve_group_label),
                "speed_pref_group": str(speed_pref_group),
                "speed_pref_label": str(speed_pref_label),
                "sf_group": str(sf_group),
                "sf_group_label": str(sf_group_label),
                **fit_unit_surface(sub),
            }
        )
    return pd.DataFrame(rows)


def summarize_fits(fits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    tests = []
    metrics = ["fit_fwhm_sf_octaves", "fit_fwhm_tf_octaves", "fit_fwhm_area_octave2", "fit_pref_log2_sf", "fit_pref_log2_tf"]
    for subset_name, sub in [
        ("all_fit_ok", fits[fits["fit_ok"].astype(bool)]),
        ("interior_only", fits[fits["fit_ok"].astype(bool) & ~(fits["fit_edge_sf"].astype(bool) | fits["fit_edge_tf"].astype(bool))]),
    ]:
        for keys, ss in sub.groupby(["speed_family", "speed_pref_group", "speed_pref_label"], sort=True):
            family, group, label = keys
            rec = {
                "subset": subset_name,
                "speed_family": family,
                "speed_pref_group": group,
                "speed_pref_label": label,
                "n_units": int(ss.shape[0]),
                "edge_sf_fraction": float(ss["fit_edge_sf"].mean()) if ss.shape[0] else float("nan"),
                "edge_tf_fraction": float(ss["fit_edge_tf"].mean()) if ss.shape[0] else float("nan"),
                "median_fit_r2": float(pd.to_numeric(ss["fit_r2"], errors="coerce").median()),
            }
            for metric in metrics:
                vals = pd.to_numeric(ss[metric], errors="coerce").dropna()
                rec[f"{metric}_mean"] = float(vals.mean()) if not vals.empty else float("nan")
                rec[f"{metric}_sem"] = sem(vals)
                rec[f"{metric}_median"] = float(vals.median()) if not vals.empty else float("nan")
            rows.append(rec)
        for family, ss in sub.groupby("speed_family", sort=True):
            high = ss[ss["speed_pref_group"].eq("high_speed_preferring")]
            low = ss[ss["speed_pref_group"].eq("low_speed_preferring")]
            for metric in metrics:
                t_stat, p_value = welch(high[metric], low[metric])
                high_vals = pd.to_numeric(high[metric], errors="coerce").dropna()
                low_vals = pd.to_numeric(low[metric], errors="coerce").dropna()
                tests.append(
                    {
                        "subset": subset_name,
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


def group_surface(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in points.groupby(["speed_pref_group", "speed_pref_label", "spatial_cpd", "temporal_hz", "log2_sf", "log2_tf"], sort=True):
        group, label, sf, tf, log_sf, log_tf = keys
        vals = pd.to_numeric(sub["unit_surface_z"], errors="coerce").dropna()
        rows.append(
            {
                "speed_pref_group": group,
                "speed_pref_label": label,
                "spatial_cpd": float(sf),
                "temporal_hz": float(tf),
                "log2_sf": float(log_sf),
                "log2_tf": float(log_tf),
                "mean_unit_z": float(vals.mean()) if not vals.empty else float("nan"),
                "sem_unit_z": sem(vals),
                "n_units": int(vals.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def tick_values(values: pd.Series) -> tuple[list[float], list[str]]:
    unique = sorted(float(v) for v in pd.unique(values.dropna()))
    return [float(np.log2(v)) for v in unique], [f"{v:g}" for v in unique]


def add_fit_ellipses(ax: plt.Axes, fits: pd.DataFrame, group: str, *, include_edges: bool) -> None:
    sub = fits[fits["speed_pref_group"].eq(group) & fits["fit_ok"].astype(bool)].copy()
    color = GROUP_COLORS[group]
    for _, row in sub.iterrows():
        edge = bool(row["fit_edge_sf"]) or bool(row["fit_edge_tf"])
        if edge and not include_edges:
            continue
        ell = Ellipse(
            (float(row["fit_pref_log2_sf"]), float(row["fit_pref_log2_tf"])),
            width=float(row["fit_fwhm_sf_octaves"]),
            height=float(row["fit_fwhm_tf_octaves"]),
            angle=0.0,
            fill=False,
            edgecolor=color,
            linewidth=0.65 if edge else 0.8,
            alpha=0.14 if edge else 0.26,
            linestyle="--" if edge else "-",
        )
        ax.add_patch(ell)
    interior = sub[~(sub["fit_edge_sf"].astype(bool) | sub["fit_edge_tf"].astype(bool))]
    use = interior if not interior.empty else sub
    if not use.empty:
        med = use[
            [
                "fit_pref_log2_sf",
                "fit_pref_log2_tf",
                "fit_fwhm_sf_octaves",
                "fit_fwhm_tf_octaves",
            ]
        ].median(numeric_only=True)
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
        ax.scatter([float(med["fit_pref_log2_sf"])], [float(med["fit_pref_log2_tf"])], color="black", s=20, zorder=5)


def plot_surface_and_ellipses(out_dir: Path, points: pd.DataFrame, fits: pd.DataFrame, surface: pd.DataFrame, tests: pd.DataFrame, *, dpi: int) -> Path:
    png = out_dir / "cycle_valid_2d_sf_tf_group_surface_and_fit_ellipses.png"
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.0), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.965, bottom=0.08, top=0.86, hspace=0.42, wspace=0.28)
    fig.suptitle("2D SF/TF tuning from speed-controlled grating probe", y=0.965, fontsize=16)
    fig.text(
        0.5,
        0.925,
        "Top: group-mean within-unit z at sampled SF/TF points. Bottom: per-unit log-Gaussian FWHM ellipses; dashed ellipses are edge-preference fits.",
        ha="center",
        fontsize=10.5,
        color="0.35",
    )
    xticks, xlabels = tick_values(points["spatial_cpd"])
    yticks, ylabels = tick_values(points["temporal_hz"])
    vmax = float(np.nanmax(np.abs(surface["mean_unit_z"].to_numpy(dtype=float))))
    vmax = max(vmax, 0.5)
    for col, group in enumerate(GROUP_ORDER):
        ax = axes[0, col]
        ss = surface[surface["speed_pref_group"].eq(group)].sort_values(["log2_sf", "log2_tf"])
        x = ss["log2_sf"].to_numpy(dtype=float)
        y = ss["log2_tf"].to_numpy(dtype=float)
        z = ss["mean_unit_z"].to_numpy(dtype=float)
        if ss.shape[0] >= 4:
            tri = mtri.Triangulation(x, y)
            ax.tricontourf(tri, z, levels=np.linspace(-vmax, vmax, 17), cmap="coolwarm", vmin=-vmax, vmax=vmax, alpha=0.88)
        sc = ax.scatter(x, y, c=z, cmap="coolwarm", vmin=-vmax, vmax=vmax, s=58, edgecolor="black", linewidth=0.35)
        ax.set_title(GROUP_LABELS[group])
        ax.set_xlabel("SF (cpd)")
        ax.set_ylabel("TF (Hz)")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.grid(True, color="0.85", linewidth=0.55)
    ax = axes[0, 2]
    high = surface[surface["speed_pref_group"].eq("high_speed_preferring")][
        ["spatial_cpd", "temporal_hz", "log2_sf", "log2_tf", "mean_unit_z"]
    ].rename(columns={"mean_unit_z": "high_z"})
    low = surface[surface["speed_pref_group"].eq("low_speed_preferring")][
        ["spatial_cpd", "temporal_hz", "log2_sf", "log2_tf", "mean_unit_z"]
    ].rename(columns={"mean_unit_z": "low_z"})
    diff = high.merge(low, on=["spatial_cpd", "temporal_hz", "log2_sf", "log2_tf"], how="inner")
    diff["low_minus_high_z"] = diff["low_z"] - diff["high_z"]
    dvmax = max(float(np.nanmax(np.abs(diff["low_minus_high_z"].to_numpy(dtype=float)))), 0.5)
    x = diff["log2_sf"].to_numpy(dtype=float)
    y = diff["log2_tf"].to_numpy(dtype=float)
    z = diff["low_minus_high_z"].to_numpy(dtype=float)
    if diff.shape[0] >= 4:
        tri = mtri.Triangulation(x, y)
        ax.tricontourf(tri, z, levels=np.linspace(-dvmax, dvmax, 17), cmap="coolwarm", vmin=-dvmax, vmax=dvmax, alpha=0.88)
    ax.scatter(x, y, c=z, cmap="coolwarm", vmin=-dvmax, vmax=dvmax, s=58, edgecolor="black", linewidth=0.35)
    ax.set_title("low minus high")
    ax.set_xlabel("SF (cpd)")
    ax.set_ylabel("TF (Hz)")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.grid(True, color="0.85", linewidth=0.55)

    cax = fig.add_axes([0.975, 0.545, 0.011, 0.295])
    fig.colorbar(sc, cax=cax, label="mean within-unit z")

    for col, group in enumerate(GROUP_ORDER):
        ax = axes[1, col]
        add_fit_ellipses(ax, fits, group, include_edges=True)
        ss = surface[surface["speed_pref_group"].eq(group)]
        ax.scatter(ss["log2_sf"], ss["log2_tf"], s=18, color="0.65", alpha=0.55)
        ax.set_title(f"{GROUP_LABELS[group]} fitted FWHM ellipses")
        ax.set_xlabel("SF (cpd)")
        ax.set_ylabel("TF (Hz)")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_xlim(min(xticks) - 0.7, max(xticks) + 0.7)
        ax.set_ylim(min(yticks) - 0.9, max(yticks) + 0.9)
        ax.grid(True, color="0.9")

    ax = axes[1, 2]
    rng = np.random.default_rng(33)
    for xloc, group in enumerate(GROUP_ORDER):
        ss = fits[fits["fit_ok"].astype(bool) & fits["speed_pref_group"].eq(group)].copy()
        vals = pd.to_numeric(ss["fit_fwhm_area_octave2"], errors="coerce")
        good = vals.notna()
        ss = ss[good]
        vals = vals[good].to_numpy(dtype=float)
        edge = ss["fit_edge_sf"].astype(bool).to_numpy() | ss["fit_edge_tf"].astype(bool).to_numpy()
        jitter = rng.uniform(-0.08, 0.08, size=vals.size)
        color = GROUP_COLORS[group]
        ax.scatter(
            np.full(vals.size, xloc) + jitter,
            vals,
            s=24,
            facecolors=np.where(edge, "none", color),
            edgecolors=color,
            linewidths=np.where(edge, 0.85, 0.0),
            alpha=0.62,
        )
        ax.errorbar([xloc], [float(np.nanmean(vals))], yerr=[float(np.nanstd(vals, ddof=1) / math.sqrt(max(vals.size, 1)))], color="black", marker="o", capsize=4)
    area_test = tests[
        (tests["subset"].eq("all_fit_ok"))
        & (tests["metric"].eq("fit_fwhm_area_octave2"))
    ]
    interior_test = tests[
        (tests["subset"].eq("interior_only"))
        & (tests["metric"].eq("fit_fwhm_area_octave2"))
    ]
    title = "bandwidth area"
    if not area_test.empty:
        title += f"\nall p={float(area_test['welch_p'].iloc[0]):.3g}"
    if not interior_test.empty:
        title += f"; interior p={float(interior_test['welch_p'].iloc[0]):.3g}"
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["high-speed\npref.", "low-speed\npref."])
    ax.set_ylabel("SF FWHM x TF FWHM (octave^2)")
    ax.grid(True, axis="y", color="0.9")

    fig.savefig(png, dpi=int(dpi))
    fig.savefig(out_dir / "cycle_valid_2d_sf_tf_group_surface_and_fit_ellipses.pdf")
    plt.close(fig)
    return png


def plot_example_units(out_dir: Path, points: pd.DataFrame, fits: pd.DataFrame, *, dpi: int, examples_per_group: int = 4) -> Path:
    png = out_dir / "cycle_valid_2d_sf_tf_example_unit_surfaces.png"
    chosen = []
    for group in GROUP_ORDER:
        sub = fits[fits["fit_ok"].astype(bool) & fits["speed_pref_group"].eq(group)].copy()
        sub["is_edge"] = sub["fit_edge_sf"].astype(bool) | sub["fit_edge_tf"].astype(bool)
        sub["score"] = pd.to_numeric(sub["fit_r2"], errors="coerce") - 0.25 * sub["is_edge"].astype(float)
        chosen.extend(sub.sort_values("score", ascending=False).head(examples_per_group)["unit_index"].astype(int).to_list())
    chosen_fits = fits[fits["unit_index"].isin(chosen)].copy()
    n_rows = len(chosen)
    fig, axes = plt.subplots(n_rows, 1, figsize=(6.8, max(2.05 * n_rows, 5.5)), constrained_layout=False)
    if n_rows == 1:
        axes = np.asarray([axes])
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.07, top=0.9, hspace=0.58)
    fig.suptitle("Example unit 2D SF/TF sampled surfaces", y=0.975, fontsize=15)
    fig.text(0.5, 0.94, "Color is within-unit z over sampled SF/TF pairs; black ellipse is the fitted FWHM", ha="center", fontsize=10.2, color="0.35")
    xticks, xlabels = tick_values(points["spatial_cpd"])
    yticks, ylabels = tick_values(points["temporal_hz"])
    for ax, unit in zip(axes, chosen, strict=True):
        ss = points[points["unit_index"].eq(unit)].copy()
        fit = chosen_fits[chosen_fits["unit_index"].eq(unit)].iloc[0]
        group = str(fit["speed_pref_group"])
        color = GROUP_COLORS[group]
        x = ss["log2_sf"].to_numpy(dtype=float)
        y = ss["log2_tf"].to_numpy(dtype=float)
        z = ss["unit_surface_z"].to_numpy(dtype=float)
        vmax = max(float(np.nanmax(np.abs(z))), 0.5)
        tri = mtri.Triangulation(x, y)
        ax.tricontourf(tri, z, levels=np.linspace(-vmax, vmax, 17), cmap="coolwarm", vmin=-vmax, vmax=vmax, alpha=0.88)
        ax.scatter(x, y, c=z, cmap="coolwarm", vmin=-vmax, vmax=vmax, s=44, edgecolor="black", linewidth=0.3)
        ell = Ellipse(
            (float(fit["fit_pref_log2_sf"]), float(fit["fit_pref_log2_tf"])),
            width=float(fit["fit_fwhm_sf_octaves"]),
            height=float(fit["fit_fwhm_tf_octaves"]),
            fill=False,
            edgecolor="black",
            linewidth=1.8,
            linestyle="--" if bool(fit["fit_edge_sf"]) or bool(fit["fit_edge_tf"]) else "-",
        )
        ax.add_patch(ell)
        ax.scatter([float(fit["fit_pref_log2_sf"])], [float(fit["fit_pref_log2_tf"])], color="black", s=22)
        ax.set_title(
            f"{fit['unit_label']} | {GROUP_LABELS[group]} | R2={float(fit['fit_r2']):.2f}, "
            f"area={float(fit['fit_fwhm_area_octave2']):.1f}, {fit['fit_status']}",
            color=color,
            fontsize=9.5,
        )
        ax.set_xlabel("SF (cpd)")
        ax.set_ylabel("TF (Hz)")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.grid(True, color="0.88", linewidth=0.55)
    fig.savefig(png, dpi=int(dpi))
    fig.savefig(out_dir / "cycle_valid_2d_sf_tf_example_unit_surfaces.pdf")
    plt.close(fig)
    return png


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped = pd.read_csv(Path(args.probe_dir) / "speed_controlled_grouped.csv")
    groups = add_group_labels(pd.read_csv(args.groups_csv))
    points = aggregate_points(grouped, groups, speed_family=str(args.speed_family))
    fits = fit_all(points)
    summary, tests = summarize_fits(fits)
    surface = group_surface(points)
    points.to_csv(out_dir / f"{args.speed_family}_2d_sf_tf_surface_points.csv", index=False)
    fits.to_csv(out_dir / f"{args.speed_family}_2d_sf_tf_fit_unit_summary.csv", index=False)
    summary.to_csv(out_dir / f"{args.speed_family}_2d_sf_tf_fit_group_summary.csv", index=False)
    tests.to_csv(out_dir / f"{args.speed_family}_2d_sf_tf_fit_group_tests.csv", index=False)
    surface.to_csv(out_dir / f"{args.speed_family}_2d_sf_tf_group_surface.csv", index=False)
    surface_png = plot_surface_and_ellipses(out_dir, points, fits, surface, tests, dpi=int(args.dpi))
    example_png = plot_example_units(out_dir, points, fits, dpi=int(args.dpi))
    print(f"Wrote {surface_png}")
    print(f"Wrote {example_png}")
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
