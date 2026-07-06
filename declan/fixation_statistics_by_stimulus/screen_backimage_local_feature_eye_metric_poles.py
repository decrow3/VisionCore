#!/usr/bin/env python3
"""Screen eye-metric differences at low/high poles of local BackImage features."""

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
    from .run_backimage_image_structure_analysis import circular_axis_delta_deg
except ImportError:  # pragma: no cover
    from declan.fixation_statistics_by_stimulus.io_utils import parse_csv_list, write_json
    from declan.fixation_statistics_by_stimulus.run_backimage_image_structure_analysis import circular_axis_delta_deg


DEFAULT_INPUT = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_image_structure_reviewed_v2_screenfiltered_yfix_slope_v1"
    / "backimage_image_fem_windows.csv"
)
DEFAULT_OUT_DIR = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_local_feature_eye_metric_poles_v2_axisfixed"
)
DEFAULT_PHASES = ("mid_fixation", "late_fixation")
DEFAULT_POLE_QUANTILE = 0.20

COLORS = {
    "along": "#1b7f5c",
    "across": "#7a3b9a",
    "total": "#284b63",
    "speed": "#b26b22",
    "alignment": "#5b6f95",
}


FEATURE_SPECS = (
    {
        "feature": "orientation_coherence",
        "column": "image_orientation_coherence",
        "label": "orientation coherence",
        "note": "dominant local image axis reliability from the Sobel structure tensor",
    },
    {
        "feature": "gradient_energy",
        "column": "image_gradient_energy",
        "label": "gradient energy",
        "note": "local edge/contrast strength from Sobel gradients",
    },
    {
        "feature": "rms_contrast",
        "column": "image_patch_rms_contrast",
        "label": "RMS contrast",
        "note": "patch luminance contrast normalized by mean luminance",
    },
    {
        "feature": "oriented_8plus_cpd_power",
        "column": "image_oriented_8plus_power_proxy",
        "label": "oriented 8+ cpd power",
        "note": "absolute 8+ cpd power proxy weighted by spectrum anisotropy and corrected contour-axis agreement",
    },
    {
        "feature": "spectral_slope",
        "column": "image_power_slope_0p5_16_cpd",
        "label": "spectral slope",
        "note": "signed log-log power slope from 0.5-16 cycles/deg; high is flatter",
    },
    {
        "feature": "spectrum_anisotropy",
        "column": "image_spectrum_anisotropy",
        "label": "spectrum anisotropy",
        "note": "oriented texture strength from Fourier power moments",
    },
    {
        "feature": "edge_density",
        "column": "image_edge_density",
        "label": "edge density",
        "note": "fraction of patch pixels with above-local-threshold gradient magnitude",
    },
    {
        "feature": "multi_orientation_energy",
        "column": "image_multi_orientation_energy",
        "label": "multi-orientation energy",
        "note": "gradient energy weighted by 1 - orientation coherence; a junction/texture-complexity proxy",
    },
    {
        "feature": "spectrum_contour_axis_agreement",
        "column": "image_edge_spectrum_contour_axis_agreement",
        "label": "edge/spectrum axis agreement",
        "note": "cos2 agreement between Sobel contour axis and Fourier-derived contour axis; Fourier frequency axis is rotated by 90 deg",
    },
    {
        "feature": "mean_luminance",
        "column": "image_patch_mean",
        "label": "mean luminance",
        "note": "mean patch luminance/intensity",
    },
)

EYE_METRICS = (
    ("rms_radius_arcmin", "total RMS (arcmin)"),
    ("step_median_arcmin", "median step (arcmin)"),
    ("signed_msd_d_arcmin2_s", "slope D (arcmin^2/s)"),
    ("speed_median_deg_s", "median speed (deg/s)"),
    ("anisotropy", "position anisotropy"),
    ("direction_persistence", "direction persistence"),
    ("return_to_center_strength", "return to center"),
    ("rms_along_arcmin", "along RMS (arcmin)"),
    ("rms_across_arcmin", "across RMS (arcmin)"),
    ("rms_delta_along_minus_across_arcmin", "along - across RMS (arcmin)"),
    ("drift_edge_cos2", "motion/image-axis cos2"),
)

PLOT_METRICS = (
    "rms_radius_arcmin",
    "step_median_arcmin",
    "signed_msd_d_arcmin2_s",
    "rms_along_arcmin",
    "rms_across_arcmin",
    "rms_delta_along_minus_across_arcmin",
    "drift_edge_cos2",
)


@dataclass(frozen=True)
class PoleScreenConfig:
    input_windows: str
    out_dir: str
    phases: list[str]
    pole_quantile: float
    min_windows_per_session_pole: int
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


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["subject"] = out["session"].map(_subject_from_session)
    out["rms_radius_arcmin"] = out["rms_radius_deg"].astype(float) * 60.0
    out["step_median_arcmin"] = out["step_median_deg"].astype(float) * 60.0
    out["diffusion_constant_arcmin2_s"] = out["diffusion_constant_deg2_s"].astype(float) * 3600.0

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
    out["rms_ratio_along_over_across"] = out["rms_along_arcmin"] / out["rms_across_arcmin"]

    out["image_multi_orientation_energy"] = (
        out["image_gradient_energy"].astype(float)
        * np.maximum(1.0 - out["image_orientation_coherence"].astype(float), 0.0)
    )
    spectrum_contour_axis = out["image_spectrum_orientation_deg"].to_numpy(dtype=np.float64) + 90.0
    edge_spectrum_contour_delta = circular_axis_delta_deg(
        out["image_edge_axis_deg"].to_numpy(dtype=np.float64),
        spectrum_contour_axis,
    )
    out["image_axis_disagreement_edge_spectrum_deg"] = np.abs(circular_axis_delta_deg(
        out["image_edge_axis_deg"].to_numpy(dtype=np.float64),
        out["image_spectrum_orientation_deg"].to_numpy(dtype=np.float64),
    ))
    out["image_axis_disagreement_edge_spectrum_contour_deg"] = np.abs(edge_spectrum_contour_delta)
    out["image_edge_spectrum_contour_axis_agreement"] = np.cos(2.0 * np.radians(edge_spectrum_contour_delta))
    out["image_abs_8plus_power_proxy"] = (
        out["image_power_8plus_cpd_fraction"].astype(float)
        * out["image_patch_std"].astype(float)
        * out["image_patch_std"].astype(float)
    )
    out["image_oriented_8plus_power_proxy"] = (
        out["image_abs_8plus_power_proxy"].astype(float)
        * np.maximum(out["image_spectrum_anisotropy"].astype(float), 0.0)
        * np.maximum(out["image_edge_spectrum_contour_axis_agreement"].astype(float), 0.0)
    )
    return out


def _required_columns() -> list[str]:
    cols = {
        "session",
        "phase",
        "rms_radius_deg",
        "step_median_deg",
        "speed_median_deg_s",
        "diffusion_constant_deg2_s",
        "anisotropy",
        "direction_persistence",
        "return_to_center_strength",
        "cov_xx_deg2",
        "cov_xy_deg2",
        "cov_yy_deg2",
        "drift_edge_cos2",
        "image_feature_ok",
        "image_orientation_coherence",
        "image_gradient_energy",
        "image_patch_rms_contrast",
        "image_power_8plus_cpd_fraction",
        "image_power_slope_0p5_16_cpd",
        "image_spectrum_anisotropy",
        "image_edge_density",
        "image_patch_mean",
        "image_patch_std",
        "image_edge_axis_deg",
        "image_spectrum_orientation_deg",
    }
    cols.update(f"msd_lag{lag}_deg2" for lag in [1, 2, 4, 8, 16])
    return sorted(cols)


def load_windows(path: Path, *, phases: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=_required_columns())
    if phases:
        df = df[df["phase"].astype(str).isin(phases)].copy()
    if "image_feature_ok" in df.columns:
        df = df[df["image_feature_ok"].astype(bool)].copy()
    df = _add_derived_columns(df)
    needed = [spec["column"] for spec in FEATURE_SPECS] + [metric for metric, _ in EYE_METRICS]
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


def _pole_labels_within_session(values: pd.Series, *, q: float) -> pd.Series:
    labels = pd.Series(index=values.index, dtype=object)
    for _session, idx in values.groupby(values.index.map(lambda x: x[0])).groups.items():
        sub = values.loc[idx].astype(float)
        sub = sub[np.isfinite(sub)]
        if sub.empty:
            continue
        lo = sub.quantile(float(q))
        hi = sub.quantile(1.0 - float(q))
        labels.loc[sub[sub <= lo].index] = "low"
        labels.loc[sub[sub >= hi].index] = "high"
    return labels


def build_pole_tables(
    df: pd.DataFrame,
    *,
    pole_quantile: float,
    min_windows_per_session_pole: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    session_rows: list[pd.DataFrame] = []
    feature_rows: list[dict[str, Any]] = []
    for spec in FEATURE_SPECS:
        feature = str(spec["feature"])
        col = str(spec["column"])
        labels = pd.Series(index=df.index, dtype=object)
        for session, sub in df.groupby("session", observed=True):
            vals = sub[col].astype(float)
            vals = vals[np.isfinite(vals)]
            if vals.size < 2 * int(min_windows_per_session_pole):
                continue
            lo = vals.quantile(float(pole_quantile))
            hi = vals.quantile(1.0 - float(pole_quantile))
            labels.loc[vals[vals <= lo].index] = "low"
            labels.loc[vals[vals >= hi].index] = "high"
            feature_rows.append({
                "feature": feature,
                "session": str(session),
                "subject": _subject_from_session(session),
                "low_threshold": float(lo),
                "high_threshold": float(hi),
                "n_low_windows": int(np.count_nonzero(vals <= lo)),
                "n_high_windows": int(np.count_nonzero(vals >= hi)),
            })
        pole_df = df.loc[labels.notna()].copy()
        pole_df["feature"] = feature
        pole_df["feature_label"] = str(spec["label"])
        pole_df["feature_column"] = col
        pole_df["pole"] = labels.loc[pole_df.index].astype(str)
        metrics = [col] + [metric for metric, _ in EYE_METRICS]
        grouped = (
            pole_df.groupby(["feature", "feature_label", "feature_column", "subject", "session", "pole"], observed=True)[metrics]
            .median(numeric_only=True)
            .reset_index()
        )
        grouped = grouped.rename(columns={col: "feature_value_median"})
        counts = (
            pole_df.groupby(["feature", "session", "pole"], observed=True)
            .size()
            .rename("n_windows")
            .reset_index()
        )
        grouped = grouped.merge(counts, on=["feature", "session", "pole"], how="left")
        complete_sessions = grouped.groupby(["feature", "session"], observed=True)["pole"].nunique()
        complete_index = complete_sessions[complete_sessions == 2].index
        grouped = grouped.set_index(["feature", "session"]).loc[complete_index].reset_index()
        session_rows.append(grouped)

    session_poles = pd.concat(session_rows, ignore_index=True) if session_rows else pd.DataFrame()
    feature_thresholds = pd.DataFrame(feature_rows)
    summary = summarize_poles(session_poles, n_bootstrap=n_bootstrap, seed=seed)
    contrasts = contrast_poles(session_poles, df, n_bootstrap=n_bootstrap, seed=seed)
    return session_poles, summary, contrasts, feature_thresholds


def summarize_poles(session_poles: pd.DataFrame, *, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (feature, feature_label, pole), sub in session_poles.groupby(["feature", "feature_label", "pole"], observed=True):
        for metric, metric_label in EYE_METRICS:
            point, lo, hi = _bootstrap_ci(sub[metric].to_numpy(dtype=np.float64), n_bootstrap=n_bootstrap, seed=seed + len(rows))
            rows.append({
                "feature": feature,
                "feature_label": feature_label,
                "pole": pole,
                "eye_metric": metric,
                "eye_metric_label": metric_label,
                "session_median": point,
                "ci95_low": lo,
                "ci95_high": hi,
                "n_sessions": int(sub["session"].nunique()),
                "n_session_poles": int(sub.shape[0]),
            })
    return pd.DataFrame(rows)


def contrast_poles(session_poles: pd.DataFrame, df: pd.DataFrame, *, n_bootstrap: int, seed: int) -> pd.DataFrame:
    scales = session_metric_scales(df).set_index("eye_metric")
    rows: list[dict[str, Any]] = []
    for (feature, feature_label), sub in session_poles.groupby(["feature", "feature_label"], observed=True):
        for metric, metric_label in EYE_METRICS:
            wide = sub.pivot(index="session", columns="pole", values=metric)
            if "high" not in wide.columns or "low" not in wide.columns:
                continue
            delta = (wide["high"] - wide["low"]).to_numpy(dtype=np.float64)
            point, lo, hi = _bootstrap_ci(delta, n_bootstrap=n_bootstrap, seed=seed + 1000 + len(rows))
            scale = float(scales.loc[metric, "robust_scale"]) if metric in scales.index else float("nan")
            rows.append({
                "feature": feature,
                "feature_label": feature_label,
                "eye_metric": metric,
                "eye_metric_label": metric_label,
                "contrast": "high_minus_low_session_paired",
                "median_delta": point,
                "ci95_low": lo,
                "ci95_high": hi,
                "effect_robust_sd": point / scale if np.isfinite(scale) and scale > 0 else float("nan"),
                "n_sessions": int(np.count_nonzero(np.isfinite(delta))),
            })
    return pd.DataFrame(rows)


def animal_contrasts(session_poles: pd.DataFrame, df: pd.DataFrame, *, n_bootstrap: int, seed: int) -> pd.DataFrame:
    scales = session_metric_scales(df).set_index("eye_metric")
    rows: list[dict[str, Any]] = []
    for (subject, feature, feature_label), sub in session_poles.groupby(["subject", "feature", "feature_label"], observed=True):
        for metric, metric_label in EYE_METRICS:
            wide = sub.pivot(index="session", columns="pole", values=metric)
            if "high" not in wide.columns or "low" not in wide.columns:
                continue
            delta = (wide["high"] - wide["low"]).to_numpy(dtype=np.float64)
            point, lo, hi = _bootstrap_ci(delta, n_bootstrap=n_bootstrap, seed=seed + 2000 + len(rows))
            scale = float(scales.loc[metric, "robust_scale"]) if metric in scales.index else float("nan")
            rows.append({
                "subject": str(subject),
                "feature": feature,
                "feature_label": feature_label,
                "eye_metric": metric,
                "eye_metric_label": metric_label,
                "contrast": "high_minus_low_session_paired",
                "median_delta": point,
                "ci95_low": lo,
                "ci95_high": hi,
                "effect_robust_sd": point / scale if np.isfinite(scale) and scale > 0 else float("nan"),
                "n_sessions": int(np.count_nonzero(np.isfinite(delta))),
            })
    return pd.DataFrame(rows)


def _feature_order() -> list[str]:
    return [str(spec["feature"]) for spec in FEATURE_SPECS]


def _metric_label(metric: str) -> str:
    return dict(EYE_METRICS).get(metric, metric)


def plot_effect_heatmap(contrasts: pd.DataFrame, out_path: Path) -> None:
    metrics = list(PLOT_METRICS)
    features = _feature_order()
    matrix = np.full((len(features), len(metrics)), np.nan, dtype=float)
    for i, feature in enumerate(features):
        for j, metric in enumerate(metrics):
            rec = contrasts[(contrasts["feature"].astype(str) == feature) & (contrasts["eye_metric"].astype(str) == metric)]
            if not rec.empty:
                matrix[i, j] = float(rec["effect_robust_sd"].iloc[0])
    vmax = float(np.nanpercentile(np.abs(matrix), 95.0)) if np.isfinite(matrix).any() else 1.0
    vmax = max(vmax, 0.25)
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(np.arange(len(features)), [str(spec["label"]) for spec in FEATURE_SPECS])
    ax.set_xticks(np.arange(len(metrics)), [_metric_label(metric) for metric in metrics], rotation=35, ha="right")
    ax.set_title("High-low pole effects, scaled by eye-metric session variability", loc="left", fontsize=11)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=7, color="black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label("median high-low delta / robust session scale")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_key_metric_forest(contrasts: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        "rms_radius_arcmin",
        "rms_along_arcmin",
        "rms_across_arcmin",
        "rms_delta_along_minus_across_arcmin",
        "drift_edge_cos2",
    ]
    features = _feature_order()
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 5.1), sharey=True)
    y = np.arange(len(features))
    for ax, metric in zip(np.atleast_1d(axes), metrics, strict=True):
        sub = contrasts[contrasts["eye_metric"].astype(str) == metric].copy()
        vals: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for feature in features:
            rec = sub[sub["feature"].astype(str) == feature]
            if rec.empty:
                vals.append(float("nan"))
                lows.append(float("nan"))
                highs.append(float("nan"))
            else:
                vals.append(float(rec["median_delta"].iloc[0]))
                lows.append(float(rec["ci95_low"].iloc[0]))
                highs.append(float(rec["ci95_high"].iloc[0]))
        vals_arr = np.asarray(vals, dtype=float)
        lo_arr = np.asarray(lows, dtype=float)
        hi_arr = np.asarray(highs, dtype=float)
        color = COLORS["total"]
        if "along" in metric:
            color = COLORS["along"]
        elif "across" in metric:
            color = COLORS["across"]
        elif "cos2" in metric:
            color = COLORS["alignment"]
        ax.errorbar(
            vals_arr,
            y,
            xerr=np.vstack([vals_arr - lo_arr, hi_arr - vals_arr]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.4,
            capsize=2.5,
            ms=4,
        )
        ax.axvline(0.0, color="#555555", lw=0.8)
        ax.grid(axis="x", color="#dddddd", lw=0.8)
        ax.set_title(_metric_label(metric), loc="left", fontsize=9)
    axes[0].set_yticks(y, [str(spec["label"]) for spec in FEATURE_SPECS])
    axes[0].invert_yaxis()
    fig.suptitle("Eye-metric shifts at high vs low image-feature poles", x=0.04, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_animal_effect_heatmaps(contrasts: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        "rms_along_arcmin",
        "rms_across_arcmin",
        "rms_delta_along_minus_across_arcmin",
        "drift_edge_cos2",
    ]
    features = _feature_order()
    subjects = sorted(contrasts["subject"].dropna().astype(str).unique())
    if not subjects:
        return
    fig, axes = plt.subplots(1, len(subjects), figsize=(5.7 * len(subjects), 5.0), sharey=True)
    axes_arr = np.atleast_1d(axes)
    all_values = contrasts[contrasts["eye_metric"].isin(metrics)]["effect_robust_sd"].to_numpy(dtype=float)
    vmax = float(np.nanpercentile(np.abs(all_values), 95.0)) if np.isfinite(all_values).any() else 1.0
    vmax = max(vmax, 0.25)
    im = None
    for ax, subject in zip(axes_arr, subjects, strict=True):
        matrix = np.full((len(features), len(metrics)), np.nan, dtype=float)
        sub = contrasts[contrasts["subject"].astype(str) == subject]
        for i, feature in enumerate(features):
            for j, metric in enumerate(metrics):
                rec = sub[(sub["feature"].astype(str) == feature) & (sub["eye_metric"].astype(str) == metric)]
                if not rec.empty:
                    matrix[i, j] = float(rec["effect_robust_sd"].iloc[0])
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(subject, loc="left", fontsize=10)
        ax.set_xticks(np.arange(len(metrics)), [_metric_label(metric) for metric in metrics], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(features)), [str(spec["label"]) for spec in FEATURE_SPECS])
    fig.suptitle("Animal split: along/across and alignment pole effects", x=0.04, ha="left", fontsize=11)
    fig.subplots_adjust(left=0.29, right=0.90, bottom=0.25, top=0.86, wspace=0.08)
    if im is not None:
        cax = fig.add_axes([0.925, 0.26, 0.018, 0.56])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("scaled high-low delta")
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_report(contrasts: pd.DataFrame, out_path: Path) -> None:
    key = contrasts[contrasts["eye_metric"].isin(PLOT_METRICS)].copy()
    key["abs_effect"] = np.abs(key["effect_robust_sd"].astype(float))
    top = key.sort_values("abs_effect", ascending=False).head(20)
    lines = [
        "# BackImage Local Feature Pole Screen",
        "",
        "Pole contrasts are high minus low feature values, assigned within session using the configured quantile.",
        "Each point is a session-paired delta; CIs bootstrap sessions.",
        "",
        "Feature convention notes:",
        "",
        "- `oriented 8+ cpd power` is not the raw 8+ cpd power fraction. The raw fraction behaves more like a fine-texture fraction and can anti-correlate with orientation coherence.",
        "- Fourier spectrum orientation is a frequency-vector axis, so it is rotated by 90 degrees before comparison with the Sobel contour axis.",
        "- The oriented 8+ cpd feature combines absolute high-frequency power, spectrum anisotropy, and corrected contour-axis agreement.",
        "",
        "## Top scaled effects",
        "",
        "| feature | eye metric | delta | 95% CI | scaled | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in top.to_dict("records"):
        lines.append(
            f"| {row['feature_label']} | {row['eye_metric_label']} | "
            f"{row['median_delta']:.4g} | [{row['ci95_low']:.4g}, {row['ci95_high']:.4g}] | "
            f"{row['effect_robust_sd']:.3f} | {int(row['n_sessions'])} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-windows", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--phases", default=",".join(DEFAULT_PHASES))
    parser.add_argument("--pole-quantile", type=float, default=DEFAULT_POLE_QUANTILE)
    parser.add_argument("--min-windows-per-session-pole", type=int, default=8)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> Path:
    phases = parse_csv_list(str(args.phases))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_windows(Path(args.input_windows), phases=phases)
    session_poles, summary, contrasts, thresholds = build_pole_tables(
        df,
        pole_quantile=float(args.pole_quantile),
        min_windows_per_session_pole=int(args.min_windows_per_session_pole),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    by_animal = animal_contrasts(session_poles, df, n_bootstrap=int(args.n_bootstrap), seed=int(args.seed))
    scales = session_metric_scales(df)

    feature_manifest = pd.DataFrame(FEATURE_SPECS)
    metric_manifest = pd.DataFrame([
        {"eye_metric": metric, "eye_metric_label": label}
        for metric, label in EYE_METRICS
    ])
    session_poles.to_csv(out_dir / "session_pole_eye_metrics.csv", index=False)
    summary.to_csv(out_dir / "pole_eye_metric_summary.csv", index=False)
    contrasts.to_csv(out_dir / "pole_eye_metric_high_low_contrasts.csv", index=False)
    by_animal.to_csv(out_dir / "pole_eye_metric_high_low_contrasts_by_animal.csv", index=False)
    thresholds.to_csv(out_dir / "feature_pole_thresholds_by_session.csv", index=False)
    scales.to_csv(out_dir / "eye_metric_session_scales.csv", index=False)
    feature_manifest.to_csv(out_dir / "feature_manifest.csv", index=False)
    metric_manifest.to_csv(out_dir / "eye_metric_manifest.csv", index=False)

    plot_effect_heatmap(contrasts, out_dir / "pole_effect_scaled_heatmap")
    plot_key_metric_forest(contrasts, out_dir / "key_eye_metric_pole_deltas")
    plot_animal_effect_heatmaps(by_animal, out_dir / "animal_split_scaled_effect_heatmaps")
    write_report(contrasts, out_dir / "summary_report.md")

    cfg = PoleScreenConfig(
        input_windows=str(args.input_windows),
        out_dir=str(out_dir),
        phases=phases,
        pole_quantile=float(args.pole_quantile),
        min_windows_per_session_pole=int(args.min_windows_per_session_pole),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    write_json(out_dir / "run_metadata.json", {
        "config": asdict(cfg),
        "n_windows": int(df.shape[0]),
        "n_sessions": int(df["session"].nunique()),
        "subjects": sorted(df["subject"].dropna().astype(str).unique().tolist()),
        "feature_count": len(FEATURE_SPECS),
        "eye_metric_count": len(EYE_METRICS),
        "notes": (
            "Poles are assigned within each session. The contrast is high minus low feature value, "
            "with session-paired deltas bootstrapped across sessions. Along/across are measured relative "
            "to the local Sobel edge axis already stored in the BackImage image-feature table. "
            "The oriented 8+ cpd power feature is not the raw 8+ cpd power fraction: it is a Parseval-style "
            "absolute high-frequency proxy, weighted by spectrum anisotropy and by corrected contour-axis "
            "agreement. The correction rotates the Fourier frequency axis by 90 degrees before comparing it "
            "to the Sobel contour axis. Multi-orientation energy remains a proxy for junction/texture "
            "complexity and does not replace a dedicated corner detector."
        ),
    })
    print(f"Wrote local-feature pole screen to {out_dir}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
