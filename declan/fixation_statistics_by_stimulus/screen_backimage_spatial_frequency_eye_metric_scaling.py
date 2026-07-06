#!/usr/bin/env python3
"""Test whether BackImage FEM scale varies with local spatial-frequency content."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .io_utils import parse_csv_list, write_json
except ImportError:  # pragma: no cover
    from declan.fixation_statistics_by_stimulus.io_utils import parse_csv_list, write_json


DEFAULT_INPUT = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_image_structure_reviewed_v2_screenfiltered_yfix_slope_v1"
    / "backimage_image_fem_windows.csv"
)
DEFAULT_OUT_DIR = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_spatial_frequency_eye_metric_scaling_v1"
)
DEFAULT_PHASES = ("mid_fixation", "late_fixation")

BAND_SPECS = (
    ("abs_power_0_2_cpd", "0-2 cpd", 1.0, "image_power_0_2_cpd_fraction"),
    ("abs_power_2_4_cpd", "2-4 cpd", 3.0, "image_power_2_4_cpd_fraction"),
    ("abs_power_4_8_cpd", "4-8 cpd", 6.0, "image_power_4_8_cpd_fraction"),
    ("abs_power_8plus_cpd", "8+ cpd", 12.0, "image_power_8plus_cpd_fraction"),
)

FEATURE_SPECS = BAND_SPECS + (
    ("fine_abs_power_4plus_cpd", "4+ cpd absolute power", 8.0, ""),
    ("sf_centroid_cpd", "SF centroid", 6.0, ""),
)

EYE_METRICS = (
    ("rms_radius_arcmin", "total RMS (arcmin)"),
    ("step_median_arcmin", "median step (arcmin)"),
    ("signed_msd_d_arcmin2_s", "slope D (arcmin^2/s)"),
    ("rms_along_arcmin", "along RMS (arcmin)"),
    ("rms_across_arcmin", "across RMS (arcmin)"),
    ("rms_delta_along_minus_across_arcmin", "along - across RMS (arcmin)"),
    ("drift_edge_cos2", "motion/image-axis cos2"),
)

CONTROL_COLS = (
    "image_patch_mean",
    "image_patch_rms_contrast",
    "image_gradient_energy",
    "image_orientation_coherence",
    "image_spectrum_anisotropy",
    "image_edge_density",
)


@dataclass(frozen=True)
class SpatialFrequencyScreenConfig:
    input_windows: str
    out_dir: str
    phases: list[str]
    pole_quantile: float
    min_windows_per_session_pole: int
    min_windows_per_session_slope: int
    n_bootstrap: int
    seed: int


def _subject_from_session(session: Any) -> str:
    return str(session).split("_", 1)[0]


def _axis_vectors(axis_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.radians(np.asarray(axis_deg, dtype=np.float64))
    along = np.column_stack([np.cos(theta), np.sin(theta)])
    across = np.column_stack([-np.sin(theta), np.cos(theta)])
    return along, across


def _fit_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(ok) < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    return float(slope), float(intercept)


def _bootstrap_ci(values: np.ndarray, *, n_bootstrap: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.median(values))
    if values.size == 1 or n_bootstrap <= 0:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    boots = np.empty(int(n_bootstrap), dtype=np.float64)
    for i in range(int(n_bootstrap)):
        boots[i] = np.median(values[rng.integers(0, values.size, size=values.size)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd <= 0:
        return np.full(arr.shape, np.nan, dtype=np.float64)
    return (arr - mu) / sd


def _required_columns() -> list[str]:
    cols = {
        "session",
        "phase",
        "rms_radius_deg",
        "step_median_deg",
        "cov_xx_deg2",
        "cov_xy_deg2",
        "cov_yy_deg2",
        "diffusion_constant_deg2_s",
        "drift_edge_cos2",
        "image_edge_axis_deg",
        "image_patch_mean",
        "image_patch_std",
        "image_patch_rms_contrast",
        "image_gradient_energy",
        "image_orientation_coherence",
        "image_spectrum_anisotropy",
        "image_edge_density",
        "image_feature_ok",
    }
    cols.update(col for *_prefix, col in BAND_SPECS)
    cols.update(f"msd_lag{lag}_deg2" for lag in [1, 2, 4, 8, 16])
    return sorted(cols)


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["subject"] = out["session"].map(_subject_from_session)
    out["rms_radius_arcmin"] = out["rms_radius_deg"].astype(float) * 60.0
    out["step_median_arcmin"] = out["step_median_deg"].astype(float) * 60.0

    lags = [1, 2, 4, 8, 16]
    times = np.asarray(lags, dtype=np.float64) / 120.0
    signed_d: list[float] = []
    for row in out[[f"msd_lag{lag}_deg2" for lag in lags]].itertuples(index=False, name=None):
        slope, _ = _fit_slope(times, np.asarray(row, dtype=np.float64) * 3600.0)
        signed_d.append(slope / 4.0 if np.isfinite(slope) else float("nan"))
    out["signed_msd_d_arcmin2_s"] = np.asarray(signed_d, dtype=np.float64)

    along, across = _axis_vectors(out["image_edge_axis_deg"].to_numpy(dtype=np.float64))
    cxx = out["cov_xx_deg2"].to_numpy(dtype=np.float64)
    cxy = out["cov_xy_deg2"].to_numpy(dtype=np.float64)
    cyy = out["cov_yy_deg2"].to_numpy(dtype=np.float64)

    def project_var(u: np.ndarray) -> np.ndarray:
        return u[:, 0] * u[:, 0] * cxx + 2.0 * u[:, 0] * u[:, 1] * cxy + u[:, 1] * u[:, 1] * cyy

    out["rms_along_arcmin"] = 60.0 * np.sqrt(np.maximum(project_var(along), 0.0))
    out["rms_across_arcmin"] = 60.0 * np.sqrt(np.maximum(project_var(across), 0.0))
    out["rms_delta_along_minus_across_arcmin"] = out["rms_along_arcmin"] - out["rms_across_arcmin"]

    variance_proxy = out["image_patch_std"].astype(float) * out["image_patch_std"].astype(float)
    abs_cols: list[str] = []
    centers: list[float] = []
    for feature, _label, center, fraction_col in BAND_SPECS:
        out[feature] = out[fraction_col].astype(float) * variance_proxy
        abs_cols.append(feature)
        centers.append(float(center))
    abs_power = out[abs_cols].sum(axis=1)
    out["fine_abs_power_4plus_cpd"] = out["abs_power_4_8_cpd"] + out["abs_power_8plus_cpd"]
    out["sf_centroid_cpd"] = (
        sum(out[col].astype(float) * center for col, center in zip(abs_cols, centers, strict=True))
        / abs_power.replace(0.0, np.nan)
    )
    return out


def load_windows(input_path: Path, *, phases: list[str]) -> pd.DataFrame:
    df = pd.read_csv(input_path, usecols=_required_columns())
    if phases:
        df = df[df["phase"].astype(str).isin(phases)].copy()
    if "image_feature_ok" in df.columns:
        df = df[df["image_feature_ok"].astype(bool)].copy()
    df = _add_derived_columns(df)
    needed = [feature for feature, *_ in FEATURE_SPECS] + [metric for metric, _ in EYE_METRICS] + list(CONTROL_COLS)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=needed + ["session", "subject"]).copy()
    return df


def session_metric_scales(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric, label in EYE_METRICS:
        session_values = df.groupby("session", observed=True)[metric].median().to_numpy(dtype=np.float64)
        session_values = session_values[np.isfinite(session_values)]
        med = float(np.median(session_values)) if session_values.size else float("nan")
        mad = float(np.median(np.abs(session_values - med))) if session_values.size else float("nan")
        iqr = float(np.subtract(*np.quantile(session_values, [0.75, 0.25]))) if session_values.size else float("nan")
        scale = 1.4826 * mad if mad > 0 else iqr / 1.349 if iqr > 0 else np.nan
        rows.append({
            "eye_metric": metric,
            "eye_metric_label": label,
            "session_median": med,
            "session_mad": mad,
            "session_iqr": iqr,
            "robust_scale": float(scale) if np.isfinite(scale) else float("nan"),
        })
    return pd.DataFrame(rows)


def pole_contrasts(
    df: pd.DataFrame,
    *,
    pole_quantile: float,
    min_windows_per_session_pole: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    session_rows: list[dict[str, Any]] = []
    for feature, feature_label, center, _fraction_col in FEATURE_SPECS:
        for session, sub in df.groupby("session", observed=True):
            values = sub[feature].astype(float)
            if values.size < 2 * int(min_windows_per_session_pole):
                continue
            lo = values.quantile(float(pole_quantile))
            hi = values.quantile(1.0 - float(pole_quantile))
            for pole, pole_df in [("low", sub[values <= lo]), ("high", sub[values >= hi])]:
                if pole_df.shape[0] < int(min_windows_per_session_pole):
                    continue
                row: dict[str, Any] = {
                    "feature": feature,
                    "feature_label": feature_label,
                    "band_center_cpd": float(center),
                    "session": str(session),
                    "subject": _subject_from_session(session),
                    "pole": pole,
                    "n_windows": int(pole_df.shape[0]),
                    "feature_median": float(pole_df[feature].median()),
                }
                for metric, _label in EYE_METRICS:
                    row[metric] = float(pole_df[metric].median())
                session_rows.append(row)
    session_poles = pd.DataFrame(session_rows)
    scales = session_metric_scales(df).set_index("eye_metric")
    contrast_rows: list[dict[str, Any]] = []
    for (feature, feature_label, center), sub in session_poles.groupby(["feature", "feature_label", "band_center_cpd"], observed=True):
        for metric, metric_label in EYE_METRICS:
            wide = sub.pivot(index="session", columns="pole", values=metric)
            if "high" not in wide.columns or "low" not in wide.columns:
                continue
            delta = (wide["high"] - wide["low"]).to_numpy(dtype=np.float64)
            point, lo, hi = _bootstrap_ci(delta, n_bootstrap=n_bootstrap, seed=seed + len(contrast_rows))
            scale = float(scales.loc[metric, "robust_scale"]) if metric in scales.index else float("nan")
            contrast_rows.append({
                "feature": feature,
                "feature_label": feature_label,
                "band_center_cpd": float(center),
                "eye_metric": metric,
                "eye_metric_label": metric_label,
                "contrast": "high_minus_low_session_paired",
                "median_delta": point,
                "ci95_low": lo,
                "ci95_high": hi,
                "effect_robust_sd": point / scale if np.isfinite(scale) and scale > 0 else float("nan"),
                "n_sessions": int(np.count_nonzero(np.isfinite(delta))),
            })
    return session_poles, pd.DataFrame(contrast_rows)


def controlled_session_slopes(
    df: pd.DataFrame,
    *,
    min_windows_per_session_slope: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    slope_rows: list[dict[str, Any]] = []
    control_cols = list(CONTROL_COLS)
    for feature, feature_label, center, _fraction_col in FEATURE_SPECS:
        for metric, metric_label in EYE_METRICS:
            for session, sub in df.groupby("session", observed=True):
                if sub.shape[0] < int(min_windows_per_session_slope):
                    continue
                x = _zscore(sub[feature].to_numpy(dtype=np.float64))
                y = _zscore(sub[metric].to_numpy(dtype=np.float64))
                controls = [_zscore(sub[col].to_numpy(dtype=np.float64)) for col in control_cols]
                phase = (sub["phase"].astype(str) == "late_fixation").to_numpy(dtype=float)
                xmat = np.column_stack([np.ones(sub.shape[0]), x, *controls, phase])
                ok = np.isfinite(y) & np.isfinite(xmat).all(axis=1)
                if np.count_nonzero(ok) < int(min_windows_per_session_slope) or np.nanstd(x[ok]) <= 0 or np.nanstd(y[ok]) <= 0:
                    continue
                try:
                    beta = np.linalg.lstsq(xmat[ok], y[ok], rcond=None)[0]
                except np.linalg.LinAlgError:
                    continue
                slope_rows.append({
                    "feature": feature,
                    "feature_label": feature_label,
                    "band_center_cpd": float(center),
                    "eye_metric": metric,
                    "eye_metric_label": metric_label,
                    "session": str(session),
                    "subject": _subject_from_session(session),
                    "controlled_beta_z": float(beta[1]),
                    "n_windows": int(np.count_nonzero(ok)),
                })
    slopes = pd.DataFrame(slope_rows)
    summary_rows: list[dict[str, Any]] = []
    for (feature, feature_label, center, metric, metric_label), sub in slopes.groupby(
        ["feature", "feature_label", "band_center_cpd", "eye_metric", "eye_metric_label"],
        observed=True,
    ):
        point, lo, hi = _bootstrap_ci(
            sub["controlled_beta_z"].to_numpy(dtype=np.float64),
            n_bootstrap=n_bootstrap,
            seed=seed + 5000 + len(summary_rows),
        )
        summary_rows.append({
            "feature": feature,
            "feature_label": feature_label,
            "band_center_cpd": float(center),
            "eye_metric": metric,
            "eye_metric_label": metric_label,
            "controlled_beta_z_median": point,
            "ci95_low": lo,
            "ci95_high": hi,
            "n_sessions": int(sub["session"].nunique()),
        })
    return slopes, pd.DataFrame(summary_rows)


def binned_curves(df: pd.DataFrame, *, n_bins: int = 5) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature, feature_label, center, _fraction_col in FEATURE_SPECS:
        for session, sub in df.groupby("session", observed=True):
            ranks = sub[feature].rank(method="first")
            try:
                bins = pd.qcut(ranks, int(n_bins), labels=False)
            except ValueError:
                continue
            tmp = sub.copy()
            tmp["sf_bin"] = np.asarray(bins, dtype=int) + 1
            for bin_idx, bin_df in tmp.groupby("sf_bin", observed=True):
                row: dict[str, Any] = {
                    "feature": feature,
                    "feature_label": feature_label,
                    "band_center_cpd": float(center),
                    "session": str(session),
                    "subject": _subject_from_session(session),
                    "sf_bin": int(bin_idx),
                    "n_windows": int(bin_df.shape[0]),
                    "feature_median": float(bin_df[feature].median()),
                }
                for metric, _label in EYE_METRICS:
                    row[metric] = float(bin_df[metric].median())
                rows.append(row)
    return pd.DataFrame(rows)


def binned_curve_summary(curves: pd.DataFrame, *, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (feature, feature_label, center, sf_bin), sub in curves.groupby(["feature", "feature_label", "band_center_cpd", "sf_bin"], observed=True):
        for metric, metric_label in EYE_METRICS:
            point, lo, hi = _bootstrap_ci(sub[metric].to_numpy(dtype=np.float64), n_bootstrap=n_bootstrap, seed=seed + 9000 + len(rows))
            rows.append({
                "feature": feature,
                "feature_label": feature_label,
                "band_center_cpd": float(center),
                "sf_bin": int(sf_bin),
                "eye_metric": metric,
                "eye_metric_label": metric_label,
                "session_median": point,
                "ci95_low": lo,
                "ci95_high": hi,
                "n_sessions": int(sub["session"].nunique()),
            })
    return pd.DataFrame(rows)


def _metric_label(metric: str) -> str:
    return dict(EYE_METRICS).get(metric, metric)


def plot_band_pole_effects(contrasts: pd.DataFrame, out_path: Path) -> None:
    metrics = ["rms_radius_arcmin", "step_median_arcmin", "signed_msd_d_arcmin2_s", "rms_along_arcmin", "rms_across_arcmin"]
    band_features = [feature for feature, *_ in BAND_SPECS]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.25 * len(metrics), 3.15), sharex=True)
    for ax, metric in zip(np.atleast_1d(axes), metrics, strict=True):
        sub = contrasts[(contrasts["feature"].isin(band_features)) & (contrasts["eye_metric"].astype(str) == metric)].sort_values("band_center_cpd")
        x = sub["band_center_cpd"].to_numpy(dtype=float)
        y = sub["median_delta"].to_numpy(dtype=float)
        lo = sub["ci95_low"].to_numpy(dtype=float)
        hi = sub["ci95_high"].to_numpy(dtype=float)
        ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), marker="o", color="#284b63", lw=2, capsize=3)
        ax.axhline(0.0, color="#555555", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 6, 12], ["1", "3", "6", "12"])
        ax.set_xlabel("band center (cpd)")
        ax.set_title(_metric_label(metric), loc="left", fontsize=9)
        ax.grid(axis="y", color="#dddddd", lw=0.8)
    axes[0].set_ylabel("high - low absolute band-power pole")
    fig.suptitle("Does FEM scale change with absolute local spatial-frequency power?", x=0.04, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_controlled_band_slopes(summary: pd.DataFrame, out_path: Path) -> None:
    metrics = ["rms_radius_arcmin", "step_median_arcmin", "signed_msd_d_arcmin2_s", "rms_along_arcmin", "rms_across_arcmin"]
    band_features = [feature for feature, *_ in BAND_SPECS]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.25 * len(metrics), 3.15), sharex=True)
    for ax, metric in zip(np.atleast_1d(axes), metrics, strict=True):
        sub = summary[(summary["feature"].isin(band_features)) & (summary["eye_metric"].astype(str) == metric)].sort_values("band_center_cpd")
        x = sub["band_center_cpd"].to_numpy(dtype=float)
        y = sub["controlled_beta_z_median"].to_numpy(dtype=float)
        lo = sub["ci95_low"].to_numpy(dtype=float)
        hi = sub["ci95_high"].to_numpy(dtype=float)
        ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), marker="o", color="#5b6f95", lw=2, capsize=3)
        ax.axhline(0.0, color="#555555", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 6, 12], ["1", "3", "6", "12"])
        ax.set_xlabel("band center (cpd)")
        ax.set_title(_metric_label(metric), loc="left", fontsize=9)
        ax.grid(axis="y", color="#dddddd", lw=0.8)
    axes[0].set_ylabel("within-session controlled beta (z)")
    fig.suptitle("Controlled spatial-frequency slopes", x=0.04, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_feature_heatmap(contrasts: pd.DataFrame, out_path: Path) -> None:
    metrics = [metric for metric, _ in EYE_METRICS]
    features = [feature for feature, *_ in FEATURE_SPECS]
    labels = {feature: label for feature, label, *_ in FEATURE_SPECS}
    matrix = np.full((len(features), len(metrics)), np.nan, dtype=float)
    for i, feature in enumerate(features):
        for j, metric in enumerate(metrics):
            rec = contrasts[(contrasts["feature"].astype(str) == feature) & (contrasts["eye_metric"].astype(str) == metric)]
            if not rec.empty:
                matrix[i, j] = float(rec["effect_robust_sd"].iloc[0])
    vmax = float(np.nanpercentile(np.abs(matrix), 95.0)) if np.isfinite(matrix).any() else 1.0
    vmax = max(vmax, 0.25)
    fig, ax = plt.subplots(figsize=(9.7, 3.9))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(np.arange(len(features)), [labels[f] for f in features])
    ax.set_xticks(np.arange(len(metrics)), [_metric_label(metric) for metric in metrics], rotation=35, ha="right")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, shrink=0.84)
    cbar.set_label("high-low delta / robust session scale")
    ax.set_title("Spatial-frequency feature pole effects", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_report(contrasts: pd.DataFrame, slopes_summary: pd.DataFrame, out_path: Path) -> None:
    key_metrics = ["rms_radius_arcmin", "step_median_arcmin", "signed_msd_d_arcmin2_s", "rms_along_arcmin", "rms_across_arcmin"]
    band = contrasts[(contrasts["feature"].isin([feature for feature, *_ in BAND_SPECS])) & (contrasts["eye_metric"].isin(key_metrics))]
    sig = band[(band["ci95_low"] * band["ci95_high"]) > 0].copy()
    slope_band = slopes_summary[
        (slopes_summary["feature"].isin([feature for feature, *_ in BAND_SPECS]))
        & (slopes_summary["eye_metric"].isin(key_metrics + ["rms_delta_along_minus_across_arcmin", "drift_edge_cos2"]))
    ].copy()
    slope_sig = slope_band[(slope_band["ci95_low"] * slope_band["ci95_high"]) > 0].copy()
    lines = [
        "# BackImage Spatial-Frequency FEM Scaling Screen",
        "",
        "This test asks whether FEM scale changes with local spatial-frequency content.",
        "",
        "Important definitions:",
        "",
        "- Band powers are absolute proxies: `band_fraction * image_patch_std^2`, not fractional power alone.",
        "- Pole contrasts are high minus low absolute band-power values, assigned within session.",
        "- Controlled slopes are within-session z-scored OLS coefficients controlling for mean luminance, RMS contrast, gradient energy, orientation coherence, spectrum anisotropy, edge density, and late-fixation phase.",
        "- The output separates total scale from along/across components; a specific across reduction is not the same claim as global FEM scaling down.",
        "",
        "Interpretation snapshot:",
        "",
        "- Raw high-vs-low absolute band-power poles often show smaller FEM scale, but this is not monotonic with spatial-frequency band and appears in low/mid bands too.",
        "- After image-feature controls, there is no clean evidence that total FEM scale decreases specifically with 8+ cpd content.",
        "- The most stable controlled band effect is component-specific: 4-8 cpd power is associated with lower across RMS and higher along-minus-across allocation.",
        "",
        "## Significant Band-Power Pole Effects",
        "",
        "| band | eye metric | delta | 95% CI | scaled | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if sig.empty:
        lines.append("| none |  |  |  |  |  |")
    else:
        for row in sig.sort_values("effect_robust_sd", key=lambda s: np.abs(s), ascending=False).to_dict("records"):
            lines.append(
                f"| {row['feature_label']} | {row['eye_metric_label']} | {row['median_delta']:.4g} | "
                f"[{row['ci95_low']:.4g}, {row['ci95_high']:.4g}] | {row['effect_robust_sd']:.3f} | {int(row['n_sessions'])} |"
            )
    lines.extend([
        "",
        "## Significant Controlled Band Slopes",
        "",
        "| band | eye metric | beta z | 95% CI | n |",
        "|---|---:|---:|---:|---:|",
    ])
    if slope_sig.empty:
        lines.append("| none |  |  |  |  |")
    else:
        for row in slope_sig.sort_values("controlled_beta_z_median", key=lambda s: np.abs(s), ascending=False).to_dict("records"):
            lines.append(
                f"| {row['feature_label']} | {row['eye_metric_label']} | {row['controlled_beta_z_median']:.4g} | "
                f"[{row['ci95_low']:.4g}, {row['ci95_high']:.4g}] | {int(row['n_sessions'])} |"
            )
    lines.extend([
        "",
        "## Controlled Slopes For 8+ cpd",
        "",
        "| eye metric | beta z | 95% CI | n |",
        "|---|---:|---:|---:|",
    ])
    sf8 = slopes_summary[slopes_summary["feature"].astype(str) == "abs_power_8plus_cpd"]
    for row in sf8[sf8["eye_metric"].isin(key_metrics)].to_dict("records"):
        lines.append(
            f"| {row['eye_metric_label']} | {row['controlled_beta_z_median']:.4g} | "
            f"[{row['ci95_low']:.4g}, {row['ci95_high']:.4g}] | {int(row['n_sessions'])} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-windows", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--phases", default=",".join(DEFAULT_PHASES))
    parser.add_argument("--pole-quantile", type=float, default=0.20)
    parser.add_argument("--min-windows-per-session-pole", type=int, default=8)
    parser.add_argument("--min-windows-per-session-slope", type=int, default=30)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> Path:
    phases = parse_csv_list(str(args.phases))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_windows(Path(args.input_windows), phases=phases)
    session_poles, contrasts = pole_contrasts(
        df,
        pole_quantile=float(args.pole_quantile),
        min_windows_per_session_pole=int(args.min_windows_per_session_pole),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    slopes, slopes_summary = controlled_session_slopes(
        df,
        min_windows_per_session_slope=int(args.min_windows_per_session_slope),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    curves = binned_curves(df)
    curve_summary = binned_curve_summary(curves, n_bootstrap=int(args.n_bootstrap), seed=int(args.seed))
    scales = session_metric_scales(df)

    session_poles.to_csv(out_dir / "sf_pole_session_eye_metrics.csv", index=False)
    contrasts.to_csv(out_dir / "sf_pole_high_low_contrasts.csv", index=False)
    slopes.to_csv(out_dir / "sf_controlled_session_slopes.csv", index=False)
    slopes_summary.to_csv(out_dir / "sf_controlled_slope_summary.csv", index=False)
    curves.to_csv(out_dir / "sf_binned_session_eye_metrics.csv", index=False)
    curve_summary.to_csv(out_dir / "sf_binned_curve_summary.csv", index=False)
    scales.to_csv(out_dir / "eye_metric_session_scales.csv", index=False)
    pd.DataFrame([
        {"feature": feature, "feature_label": label, "band_center_cpd": center, "source_fraction_col": fraction_col}
        for feature, label, center, fraction_col in FEATURE_SPECS
    ]).to_csv(out_dir / "sf_feature_manifest.csv", index=False)

    plot_band_pole_effects(contrasts, out_dir / "band_power_pole_effects_by_spatial_frequency")
    plot_controlled_band_slopes(slopes_summary, out_dir / "controlled_band_power_slopes_by_spatial_frequency")
    plot_feature_heatmap(contrasts, out_dir / "sf_feature_pole_effect_heatmap")
    write_report(contrasts, slopes_summary, out_dir / "summary_report.md")

    cfg = SpatialFrequencyScreenConfig(
        input_windows=str(args.input_windows),
        out_dir=str(out_dir),
        phases=phases,
        pole_quantile=float(args.pole_quantile),
        min_windows_per_session_pole=int(args.min_windows_per_session_pole),
        min_windows_per_session_slope=int(args.min_windows_per_session_slope),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    write_json(out_dir / "run_metadata.json", {
        "config": asdict(cfg),
        "n_windows": int(df.shape[0]),
        "n_sessions": int(df["session"].nunique()),
        "subjects": sorted(df["subject"].dropna().astype(str).unique().tolist()),
        "note": "Absolute band powers are proportional proxies based on band power fraction times patch variance. They are intended for within-session ranking and regression, not calibrated physical image power.",
    })
    print(f"Wrote BackImage spatial-frequency FEM scaling screen to {out_dir}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
