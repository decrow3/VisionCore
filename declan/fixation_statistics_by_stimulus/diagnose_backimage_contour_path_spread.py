#!/usr/bin/env python3
"""Diagnostic plots for contour-relative path length versus position spread.

The goal is to separate three things that can look contradictory in a single
summary plot:

* the size of the eye-position cloud along an axis (RMS spread);
* the projected cumulative path length along that axis (sum(abs(projected steps)));
* how much of that distance survives as displacement rather than cancellation.

The script uses the same BackImage contour windows and 0.1 coherence bins as the
Panel B position-spread followups. It recomputes component step diagnostics from
the saved eye traces so that reversal fraction and lag-1 step autocorrelation
can be inspected next to RMS and unsigned path length.
"""

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

from . import plot_backimage_contour_motion_components as contour_motion
from .generate_backimage_contour_signed_path_followups import (
    DEFAULT_INPUT_WINDOWS,
    DEFAULT_OUT_DIR,
    load_windows,
)
from .io_utils import write_json


BLUE = "#3366aa"
GRAY = "#7f8b96"
GREEN = "#1b7f5c"
PURPLE = "#7a3b9a"
ORANGE = "#c58b2a"
TEAL = "#2b8c8c"
INK = "#20242a"
GRID = "#d5dce3"
SPARSE_BAND = "#eef1f5"
LOW_COHERENCE_BASELINE_BIN = "0.0-0.1"
FINAL_COHERENCE_BIN = "0.9-1.0"
FINAL_BIN_NEIGHBOR_BINS = ("0.7-0.8", "0.8-0.9", "0.9-1.0")
PATH_LENGTH_METRIC = "unsigned_path_arcmin_equiv"
FEATURE_BIN_COUNT = 10
COHERENCE_X_FEATURE_BIN_COUNT = 5
WINDOW_METADATA_COLUMNS = (
    "stimulus",
    "regime",
    "trial_idx",
    "global_start",
    "global_stop",
    "local_start",
    "local_stop",
    "phase",
)

COMPONENTS = (
    ("parallel", "parallel", "rms_along_arcmin", 0.0, BLUE),
    ("orthogonal", "orthogonal", "rms_across_arcmin", 0.5 * np.pi, GRAY),
)
TRACE_2D_COMPONENT = ("trace_2d", "2D trace", TEAL)
PATH_LENGTH_COMPONENTS = (
    ("parallel", "parallel", BLUE),
    ("orthogonal", "orthogonal", GRAY),
    (TRACE_2D_COMPONENT[0], TRACE_2D_COMPONENT[1], TRACE_2D_COMPONENT[2]),
)
FEATURE_BIN_SPECS = (
    {
        "feature": "orientation_energy",
        "column": "image_orientation_energy",
        "label": "orientation energy",
        "axis_label": "orientation energy quantile",
        "stem": "b_path_spread_scale_diagnostics_by_orientation_energy",
        "note": "Raw local Sobel gradient energy; high values can include strong but multi-orientation texture.",
    },
    {
        "feature": "coherent_orientation_energy",
        "column": "image_coherent_orientation_energy",
        "label": "orientation energy x coherence",
        "axis_label": "coherent orientation energy quantile",
        "stem": "b_path_spread_scale_diagnostics_by_coherent_orientation_energy",
        "note": "Raw Sobel gradient energy multiplied by structure-tensor orientation coherence.",
    },
)
COHERENCE_X_FEATURE_BANDS = (
    (0.0, 0.3, "coh 0.0-0.3", "#6f7b86"),
    (0.3, 0.6, "coh 0.3-0.6", ORANGE),
    (0.6, 0.9, "coh 0.6-0.9", GREEN),
    (0.9, 1.0, "coh 0.9-1.0", PURPLE),
)
MSD_LAGS = (1, 2, 4, 8, 16)
MSD_BANDS = (
    (0.0, 0.3, "0.0-0.3"),
    (0.3, 0.6, "0.3-0.6"),
    (0.6, 0.9, "0.6-0.9"),
    (0.9, 1.0, "0.9-1.0 sparse"),
)


def _diagnostic_metric_cols() -> list[str]:
    return [
        "rms_arcmin",
        PATH_LENGTH_METRIC,
        "net_magnitude_arcmin_equiv",
        "path_per_rms",
        "rms_per_100_path",
        "net_to_path",
        "cancellation_fraction",
        "mean_abs_step_arcmin",
        "rms_step_arcmin",
        "step_reversal_fraction",
        "step_autocorr_lag1",
        "msd_slope_arcmin2_s",
        *[f"sqrt_msd_lag{lag}_arcmin" for lag in MSD_LAGS],
    ]


def _scale_metric_specs() -> list[tuple[str, str, str, str]]:
    return [
        ("rms_arcmin", "position spread\nRMS (arcmin)", "position spread\nRMS (arcmin)", "spread"),
        (
            PATH_LENGTH_METRIC,
            "projected cumulative\npath length\n(arcmin / 0.325 s)",
            "2D cumulative\npath length\n(arcmin / 0.325 s)",
            "path length",
        ),
        ("path_per_rms", "projected path length / spread", "2D path length / spread", "path length per spread"),
        (
            "net_to_path",
            "abs(net displacement)\n/ projected path length",
            "2D net displacement\n/ 2D path length",
            "net fraction",
        ),
    ]


def _bin_edges_from_label(label: str) -> tuple[float, float]:
    lo, hi = str(label).split("-", maxsplit=1)
    return float(lo), float(hi)


def _bin_center_from_label(label: str) -> float:
    lo, hi = _bin_edges_from_label(label)
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class DiagnosticConfig:
    input_windows: str
    out_dir: str
    coherence_bin_width: float
    equivalent_window_s: float
    sample_rate_hz: float
    n_bootstrap: int
    seed: int


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _maybe_shade_sparse_bin(ax: plt.Axes) -> None:
    ax.axvspan(0.9, 1.0, color=SPARSE_BAND, alpha=0.75, zorder=0)


def _bootstrap_median_ci(values: np.ndarray, *, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.median(vals))
    if vals.size == 1 or n_bootstrap <= 0:
        return point, float("nan"), float("nan")
    draws = rng.integers(0, vals.size, size=(int(n_bootstrap), vals.size))
    boot = np.median(vals[draws], axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return point, float(lo), float(hi)


def _finite_median(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float("nan")


def _series_float(series: pd.Series, column: str) -> float:
    if column not in series.index:
        return float("nan")
    try:
        return float(series[column])
    except (TypeError, ValueError):
        return float("nan")


def _step_autocorr_lag1(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    a = x[:-1]
    b = x[1:]
    if np.nanstd(a) <= 0.0 or np.nanstd(b) <= 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _reversal_fraction(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    signs = np.sign(x)
    ok = signs[:-1] != 0.0
    ok &= signs[1:] != 0.0
    if not np.any(ok):
        return float("nan")
    return float(np.mean(signs[:-1][ok] != signs[1:][ok]))


def _component_step_rows(
    row: pd.Series,
    *,
    equivalent_window_s: float,
) -> list[dict[str, Any]]:
    trace = contour_motion._window_trace(row)
    x = np.asarray(trace, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] < 2:
        return []
    steps = np.diff(x, axis=0)
    ok = np.isfinite(steps).all(axis=1)
    steps = steps[ok]
    if steps.size == 0:
        return []

    duration_s = float(row["duration_s"])
    scale = float(equivalent_window_s) / duration_s if duration_s > 0 and equivalent_window_s > 0 else 1.0
    finite_positions = x[np.isfinite(x).all(axis=1)]
    edge_axis_rad = np.radians(float(row["image_edge_axis_deg"]))
    rows: list[dict[str, Any]] = []

    step_norm_arcmin = np.linalg.norm(steps, axis=1) * 60.0
    total_path_2d = float(np.sum(step_norm_arcmin) * scale)
    net_2d = float(np.linalg.norm(np.sum(steps, axis=0)) * 60.0 * scale)
    centered = finite_positions - np.mean(finite_positions, axis=0)
    rms_2d = float(60.0 * np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    net_to_path_2d = net_2d / total_path_2d if total_path_2d > 0 else float("nan")
    rows.append(
        {
            "component": TRACE_2D_COMPONENT[0],
            "component_label": TRACE_2D_COMPONENT[1],
            "rms_arcmin": rms_2d,
            "unsigned_path_arcmin_equiv": total_path_2d,
            "signed_net_arcmin_equiv": float("nan"),
            "net_magnitude_arcmin_equiv": net_2d,
            "path_per_rms": total_path_2d / rms_2d if rms_2d > 0 else float("nan"),
            "rms_per_100_path": 100.0 * rms_2d / total_path_2d if total_path_2d > 0 else float("nan"),
            "net_to_path": net_to_path_2d,
            "cancellation_fraction": 1.0 - net_to_path_2d if np.isfinite(net_to_path_2d) else float("nan"),
            "mean_abs_step_arcmin": float(np.mean(step_norm_arcmin)),
            "rms_step_arcmin": float(np.sqrt(np.mean(step_norm_arcmin * step_norm_arcmin))),
            "step_reversal_fraction": float("nan"),
            "step_autocorr_lag1": float("nan"),
            "msd_slope_arcmin2_s": float("nan"),
            **{
                f"sqrt_msd_lag{lag}_arcmin": float(
                    np.sqrt(max(float(row[f"msd_lag{lag}_deg2"]), 0.0)) * 60.0
                )
                for lag in MSD_LAGS
            },
        }
    )

    for component, label, rms_col, rel_angle, _color in COMPONENTS:
        axis = np.asarray([np.cos(edge_axis_rad + rel_angle), np.sin(edge_axis_rad + rel_angle)], dtype=np.float64)
        projected = steps @ axis
        projected_arcmin = projected * 60.0
        unsigned_path = float(np.sum(np.abs(projected_arcmin)) * scale)
        signed_net = float(np.sum(projected_arcmin) * scale)
        net_magnitude = abs(signed_net)
        rms = float(row[rms_col])
        path_per_rms = unsigned_path / rms if rms > 0 else float("nan")
        rms_per_100_path = 100.0 * rms / unsigned_path if unsigned_path > 0 else float("nan")
        net_to_path = net_magnitude / unsigned_path if unsigned_path > 0 else float("nan")
        rows.append(
            {
                "component": component,
                "component_label": label,
                "rms_arcmin": rms,
                "unsigned_path_arcmin_equiv": unsigned_path,
                "signed_net_arcmin_equiv": signed_net,
                "net_magnitude_arcmin_equiv": net_magnitude,
                "path_per_rms": path_per_rms,
                "rms_per_100_path": rms_per_100_path,
                "net_to_path": net_to_path,
                "cancellation_fraction": 1.0 - net_to_path if np.isfinite(net_to_path) else float("nan"),
                "mean_abs_step_arcmin": float(np.mean(np.abs(projected_arcmin))),
                "rms_step_arcmin": float(np.sqrt(np.mean(projected_arcmin * projected_arcmin))),
                "step_reversal_fraction": _reversal_fraction(projected_arcmin),
                "step_autocorr_lag1": _step_autocorr_lag1(projected_arcmin),
                "msd_slope_arcmin2_s": float(row[f"msd_slope_{'along' if component == 'parallel' else 'across'}_arcmin2_s"]),
                **{
                    f"sqrt_msd_lag{lag}_arcmin": float(
                        np.sqrt(max(float(row[f"msd_lag{lag}_{'along' if component == 'parallel' else 'across'}_arcmin2"]), 0.0))
                    )
                    for lag in MSD_LAGS
                },
            }
        )
    return rows


def compute_component_diagnostics(
    windows: pd.DataFrame,
    *,
    equivalent_window_s: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(windows)
    for idx, row in enumerate(windows.itertuples(index=False), start=1):
        series = pd.Series(row._asdict())
        orientation_coherence = _series_float(series, "image_orientation_coherence")
        orientation_energy = _series_float(series, "image_gradient_energy")
        coherent_orientation_energy = (
            orientation_energy * orientation_coherence
            if np.isfinite(orientation_energy) and np.isfinite(orientation_coherence)
            else float("nan")
        )
        common = {
            "window_index": idx - 1,
            "session": str(series["session"]),
            "subject": str(series.get("subject", "")),
            "coherence_bin": str(series["coherence_bin"]),
            "image_orientation_coherence": orientation_coherence,
            "image_gradient_energy": orientation_energy,
            "image_orientation_energy": orientation_energy,
            "image_coherent_orientation_energy": coherent_orientation_energy,
            "duration_s": float(series["duration_s"]),
            "n_samples": int(series["n_samples"]) if np.isfinite(series["n_samples"]) else -1,
        }
        for metadata_col in WINDOW_METADATA_COLUMNS:
            if metadata_col in series.index:
                common[metadata_col] = series[metadata_col]
        for component_row in _component_step_rows(series, equivalent_window_s=equivalent_window_s):
            rows.append({**common, **component_row})
        if idx % 1000 == 0:
            print(f"computed component step diagnostics for {idx}/{total} windows", flush=True)
    return pd.DataFrame(rows)


def summarize_by_coherence(
    diagnostics: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metric_cols = _diagnostic_metric_cols()
    rows: list[dict[str, Any]] = []
    grouped = diagnostics.groupby(["coherence_bin", "component"], observed=True, sort=False)
    for (coherence_bin, component), sub in grouped:
        bin_center = _bin_center_from_label(str(coherence_bin))
        for metric in metric_cols:
            session_values = sub.groupby("session", observed=True)[metric].median(numeric_only=True).to_numpy(dtype=float)
            point, lo, hi = _bootstrap_median_ci(session_values, rng=rng, n_bootstrap=n_bootstrap)
            rows.append(
                {
                    "coherence_bin": str(coherence_bin),
                    "bin_center": bin_center,
                    "component": str(component),
                    "metric": metric,
                    "session_median": point,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "window_median": _finite_median(sub[metric].to_numpy(dtype=float)),
                    "n_sessions": int(sub["session"].nunique()),
                    "n_windows": int(sub["window_index"].nunique()),
                }
            )
    out = pd.DataFrame(rows)
    bin_order = {label: i for i, label in enumerate(sorted(out["coherence_bin"].unique(), key=lambda x: float(str(x).split("-")[0])))}
    out["bin_order"] = out["coherence_bin"].map(bin_order).astype(int)
    return out.sort_values(["bin_order", "component", "metric"]).reset_index(drop=True)


def summarize_component_ratios(
    diagnostics: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ratio_specs = [
        ("rms_parallel_over_orthogonal", "rms_arcmin", "ratio"),
        ("path_parallel_over_orthogonal", "unsigned_path_arcmin_equiv", "ratio"),
        ("net_parallel_over_orthogonal", "net_magnitude_arcmin_equiv", "ratio"),
        ("path_per_rms_parallel_over_orthogonal", "path_per_rms", "ratio"),
        ("mean_abs_step_parallel_over_orthogonal", "mean_abs_step_arcmin", "ratio"),
        ("reversal_orthogonal_minus_parallel", "step_reversal_fraction", "orthogonal_minus_parallel"),
        ("step_autocorr_orthogonal_minus_parallel", "step_autocorr_lag1", "orthogonal_minus_parallel"),
    ]
    session_component = (
        diagnostics.groupby(["coherence_bin", "session", "component"], observed=True)
        .median(numeric_only=True)
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for coherence_bin, sub in session_component.groupby("coherence_bin", observed=True, sort=False):
        for metric_name, source_col, mode in ratio_specs:
            pivot = sub.pivot(index="session", columns="component", values=source_col)
            if "parallel" not in pivot.columns or "orthogonal" not in pivot.columns:
                continue
            parallel = pivot["parallel"].to_numpy(dtype=float)
            orthogonal = pivot["orthogonal"].to_numpy(dtype=float)
            if mode == "ratio":
                values = parallel / orthogonal
                baseline = 1.0
            else:
                values = orthogonal - parallel
                baseline = 0.0
            point, lo, hi = _bootstrap_median_ci(values, rng=rng, n_bootstrap=n_bootstrap)
            rows.append(
                {
                    "coherence_bin": str(coherence_bin),
                    "bin_center": _bin_center_from_label(str(coherence_bin)),
                    "metric": metric_name,
                    "source_metric": source_col,
                    "mode": mode,
                    "baseline": baseline,
                    "session_median": point,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_sessions": int(pivot.shape[0]),
                }
            )
    out = pd.DataFrame(rows)
    bin_order = {label: i for i, label in enumerate(sorted(out["coherence_bin"].unique(), key=lambda x: float(str(x).split("-")[0])))}
    out["bin_order"] = out["coherence_bin"].map(bin_order).astype(int)
    return out.sort_values(["bin_order", "metric"]).reset_index(drop=True)


def _assign_feature_quantile_bins(
    diagnostics: pd.DataFrame,
    *,
    feature_col: str,
    n_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if feature_col not in diagnostics.columns:
        raise ValueError(f"missing feature column: {feature_col}")
    unique = diagnostics.drop_duplicates("window_index")[
        ["window_index", "session", "subject", feature_col]
    ].copy()
    values = pd.to_numeric(unique[feature_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = values.notna()
    if int(valid.sum()) < 2:
        raise ValueError(f"not enough finite values to bin {feature_col}")

    q = min(int(n_bins), int(valid.sum()))
    codes, edges = pd.qcut(values.loc[valid], q=q, labels=False, retbins=True, duplicates="drop")
    n_actual = len(edges) - 1
    if n_actual < 1:
        raise ValueError(f"could not make quantile bins for {feature_col}")

    unique["feature_bin_order"] = -1
    unique.loc[valid, "feature_bin_order"] = np.asarray(codes, dtype=int)
    unique = unique[unique["feature_bin_order"] >= 0].copy()
    unique["feature_bin_order"] = unique["feature_bin_order"].astype(int)
    unique["feature_bin"] = unique["feature_bin_order"].map(lambda order: f"Q{int(order) + 1}")
    unique["bin_center"] = (unique["feature_bin_order"].astype(float) + 0.5) / float(n_actual)
    unique["feature_bin_low"] = unique["feature_bin_order"].map(lambda order: float(edges[int(order)]))
    unique["feature_bin_high"] = unique["feature_bin_order"].map(lambda order: float(edges[int(order) + 1]))

    bin_info_rows: list[dict[str, Any]] = []
    for order, sub in unique.groupby("feature_bin_order", observed=True, sort=True):
        bin_info_rows.append(
            {
                "feature_bin": f"Q{int(order) + 1}",
                "bin_order": int(order),
                "bin_center": float((float(order) + 0.5) / float(n_actual)),
                "feature_bin_low": float(edges[int(order)]),
                "feature_bin_high": float(edges[int(order) + 1]),
                "feature_median": float(sub[feature_col].median()),
                "n_windows": int(sub["window_index"].nunique()),
                "n_sessions": int(sub["session"].nunique()),
            }
        )
    bin_info = pd.DataFrame(bin_info_rows)
    merge_cols = [
        "window_index",
        "feature_bin",
        "feature_bin_order",
        "bin_center",
        "feature_bin_low",
        "feature_bin_high",
    ]
    out = diagnostics.merge(unique[merge_cols], on="window_index", how="inner")
    out["bin_order"] = out["feature_bin_order"].astype(int)
    return out, bin_info


def summarize_by_feature_quantile(
    diagnostics: pd.DataFrame,
    *,
    feature_col: str,
    feature: str,
    feature_label: str,
    n_bins: int,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    binned, _bin_info = _assign_feature_quantile_bins(
        diagnostics,
        feature_col=feature_col,
        n_bins=n_bins,
    )
    rows: list[dict[str, Any]] = []
    grouped = binned.groupby(["feature_bin_order", "component"], observed=True, sort=True)
    for (bin_order, component), sub in grouped:
        feature_values = (
            sub.drop_duplicates("window_index")[feature_col]
            .to_numpy(dtype=float)
        )
        feature_values = feature_values[np.isfinite(feature_values)]
        for metric in _diagnostic_metric_cols():
            session_values = sub.groupby("session", observed=True)[metric].median(numeric_only=True).to_numpy(dtype=float)
            point, lo, hi = _bootstrap_median_ci(session_values, rng=rng, n_bootstrap=n_bootstrap)
            rows.append(
                {
                    "feature": feature,
                    "feature_col": feature_col,
                    "feature_label": feature_label,
                    "feature_bin": f"Q{int(bin_order) + 1}",
                    "bin_order": int(bin_order),
                    "bin_center": float((float(bin_order) + 0.5) / binned["feature_bin_order"].nunique()),
                    "feature_bin_low": float(sub["feature_bin_low"].iloc[0]),
                    "feature_bin_high": float(sub["feature_bin_high"].iloc[0]),
                    "feature_median": _finite_median(feature_values),
                    "component": str(component),
                    "metric": metric,
                    "session_median": point,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "window_median": _finite_median(sub[metric].to_numpy(dtype=float)),
                    "n_sessions": int(sub["session"].nunique()),
                    "n_windows": int(sub["window_index"].nunique()),
                }
            )
    return pd.DataFrame(rows).sort_values(["bin_order", "component", "metric"]).reset_index(drop=True)


def _add_coherence_band_columns(diagnostics: pd.DataFrame) -> pd.DataFrame:
    out = diagnostics.copy()
    coherence = pd.to_numeric(out["image_orientation_coherence"], errors="coerce").to_numpy(dtype=float)
    band_order = np.full(out.shape[0], -1, dtype=int)
    band_label = np.full(out.shape[0], "", dtype=object)
    band_center = np.full(out.shape[0], np.nan, dtype=float)
    for order, (lo, hi, label, _color) in enumerate(COHERENCE_X_FEATURE_BANDS):
        if hi >= 1.0:
            mask = (coherence >= lo) & (coherence <= hi)
        else:
            mask = (coherence >= lo) & (coherence < hi)
        band_order[mask] = order
        band_label[mask] = label
        band_center[mask] = 0.5 * (lo + hi)
    out["coherence_band_order"] = band_order
    out["coherence_band"] = band_label
    out["coherence_band_center"] = band_center
    return out[out["coherence_band_order"] >= 0].copy()


def summarize_by_coherence_band_and_feature_quantile(
    diagnostics: pd.DataFrame,
    *,
    feature_col: str,
    feature: str,
    feature_label: str,
    n_feature_bins: int,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    binned, _bin_info = _assign_feature_quantile_bins(
        diagnostics,
        feature_col=feature_col,
        n_bins=n_feature_bins,
    )
    binned = _add_coherence_band_columns(binned)
    rows: list[dict[str, Any]] = []
    group_cols = ["coherence_band_order", "feature_bin_order", "component"]
    for (coh_order, feature_order, component), sub in binned.groupby(group_cols, observed=True, sort=True):
        feature_values = sub.drop_duplicates("window_index")[feature_col].to_numpy(dtype=float)
        feature_values = feature_values[np.isfinite(feature_values)]
        for metric, _projected_ylabel, _trace_ylabel, _title in _scale_metric_specs():
            session_values = sub.groupby("session", observed=True)[metric].median(numeric_only=True).to_numpy(dtype=float)
            point, lo, hi = _bootstrap_median_ci(session_values, rng=rng, n_bootstrap=n_bootstrap)
            rows.append(
                {
                    "feature": feature,
                    "feature_col": feature_col,
                    "feature_label": feature_label,
                    "coherence_band": str(sub["coherence_band"].iloc[0]),
                    "coherence_band_order": int(coh_order),
                    "coherence_band_center": float(sub["coherence_band_center"].iloc[0]),
                    "feature_bin": f"Q{int(feature_order) + 1}",
                    "feature_bin_order": int(feature_order),
                    "bin_order": int(feature_order),
                    "bin_center": float((float(feature_order) + 0.5) / binned["feature_bin_order"].nunique()),
                    "feature_bin_low": float(sub["feature_bin_low"].iloc[0]),
                    "feature_bin_high": float(sub["feature_bin_high"].iloc[0]),
                    "feature_median": _finite_median(feature_values),
                    "component": str(component),
                    "metric": metric,
                    "session_median": point,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "window_median": _finite_median(sub[metric].to_numpy(dtype=float)),
                    "n_sessions": int(sub["session"].nunique()),
                    "n_windows": int(sub["window_index"].nunique()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["coherence_band_order", "bin_order", "component", "metric"]
    ).reset_index(drop=True)


def _summary_line(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    component: str,
    metric: str,
    color: str,
    label: str | None = None,
    ci: bool = True,
) -> None:
    sub = summary[
        summary["component"].astype(str).eq(component)
        & summary["metric"].astype(str).eq(metric)
    ].sort_values("bin_center")
    x = sub["bin_center"].to_numpy(dtype=float)
    y = sub["session_median"].to_numpy(dtype=float)
    ax.plot(x, y, marker="o", color=color, linewidth=2.0, markersize=5.0, label=label)
    if ci:
        lo = sub["ci95_low"].to_numpy(dtype=float)
        hi = sub["ci95_high"].to_numpy(dtype=float)
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)


def _ratio_line(ax: plt.Axes, ratios: pd.DataFrame, *, metric: str, color: str, label: str | None = None) -> None:
    sub = ratios[ratios["metric"].astype(str).eq(metric)].sort_values("bin_center")
    x = sub["bin_center"].to_numpy(dtype=float)
    y = sub["session_median"].to_numpy(dtype=float)
    lo = sub["ci95_low"].to_numpy(dtype=float)
    hi = sub["ci95_high"].to_numpy(dtype=float)
    ax.plot(x, y, marker="o", color=color, linewidth=2.0, markersize=5.0, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)


def _coherence_axis(ax: plt.Axes) -> None:
    _maybe_shade_sparse_bin(ax)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("local edge coherence")
    ax.grid(axis="y", color=GRID, linewidth=0.9)
    _clean_axis(ax)


def _low_coherence_baseline(
    summary: pd.DataFrame,
    metric: str,
    *,
    components: tuple[str, ...] = ("parallel", "orthogonal"),
) -> float:
    sub = summary[
        summary["coherence_bin"].astype(str).eq(LOW_COHERENCE_BASELINE_BIN)
        & summary["metric"].astype(str).eq(metric)
        & summary["component"].astype(str).isin(components)
    ]
    values = sub["session_median"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.nanmedian(values)) if values.size else float("nan")


def _draw_low_coherence_baseline(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    *,
    components: tuple[str, ...] = ("parallel", "orthogonal"),
    label: bool = False,
) -> None:
    baseline = _low_coherence_baseline(summary, metric, components=components)
    if not np.isfinite(baseline):
        return
    ax.axhline(
        baseline,
        color="#33383f",
        linewidth=1.15,
        linestyle=(0, (4.0, 2.4)),
        alpha=0.82,
        label=f"{LOW_COHERENCE_BASELINE_BIN} coh baseline" if label else None,
        zorder=1,
    )


def _feature_quantile_axis(ax: plt.Axes, *, axis_label: str, n_bins: int) -> None:
    n_bins = max(int(n_bins), 1)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.5 / n_bins, 0.5, (n_bins - 0.5) / n_bins], ["low", "mid", "high"])
    ax.set_xlabel(f"{axis_label}\n(within-sample quantile)")
    ax.grid(axis="y", color=GRID, linewidth=0.9)
    _clean_axis(ax)


def _lowest_feature_bin_baseline(
    summary: pd.DataFrame,
    metric: str,
    *,
    components: tuple[str, ...] = ("parallel", "orthogonal"),
) -> float:
    if "bin_order" not in summary.columns:
        return float("nan")
    first_bin = int(summary["bin_order"].min())
    sub = summary[
        summary["bin_order"].astype(int).eq(first_bin)
        & summary["metric"].astype(str).eq(metric)
        & summary["component"].astype(str).isin(components)
    ]
    values = sub["session_median"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.nanmedian(values)) if values.size else float("nan")


def _draw_lowest_feature_bin_baseline(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    *,
    components: tuple[str, ...] = ("parallel", "orthogonal"),
    label: str | None = None,
) -> None:
    baseline = _lowest_feature_bin_baseline(summary, metric, components=components)
    if not np.isfinite(baseline):
        return
    ax.axhline(
        baseline,
        color="#33383f",
        linewidth=1.15,
        linestyle=(0, (4.0, 2.4)),
        alpha=0.82,
        label=label,
        zorder=1,
    )


def plot_scale_diagnostics_by_feature(
    summary: pd.DataFrame,
    out_path: Path,
    *,
    feature_label: str,
    axis_label: str,
) -> None:
    n_bins = int(summary["bin_order"].nunique()) if not summary.empty else FEATURE_BIN_COUNT
    fig, axes = plt.subplots(4, 4, figsize=(13.0, 11.0), sharex=True, constrained_layout=True)
    metrics = _scale_metric_specs()
    baseline_label = f"lowest {feature_label} bin"
    for row_idx, (component, label, _rms_col, _angle, color) in enumerate(COMPONENTS):
        for col_idx, (metric, projected_ylabel, _trace_ylabel, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            _draw_lowest_feature_bin_baseline(ax, summary, metric, label=baseline_label if row_idx == 0 and col_idx == 0 else None)
            _summary_line(ax, summary, component=component, metric=metric, color=color, ci=False)
            _feature_quantile_axis(ax, axis_label=axis_label, n_bins=n_bins)
            if row_idx == 0:
                ax.set_title(title)
            ax.set_xlabel("")
            ax.set_ylabel(f"{label}\n{projected_ylabel}")
            if metric == "net_to_path":
                ax.set_ylim(bottom=0.0)

    trace_component, trace_label, trace_color = TRACE_2D_COMPONENT
    for col_idx, (metric, _projected_ylabel, trace_ylabel, _title) in enumerate(metrics):
        ax = axes[2, col_idx]
        _draw_lowest_feature_bin_baseline(ax, summary, metric, components=(trace_component,))
        _summary_line(ax, summary, component=trace_component, metric=metric, color=trace_color, ci=False)
        _feature_quantile_axis(ax, axis_label=axis_label, n_bins=n_bins)
        ax.set_xlabel("")
        ax.set_ylabel(f"{trace_label}\n{trace_ylabel}")
        if metric == "net_to_path":
            ax.set_ylim(bottom=0.0)

    for col_idx, (metric, projected_ylabel, _trace_ylabel, _title) in enumerate(metrics):
        ax = axes[3, col_idx]
        _draw_lowest_feature_bin_baseline(ax, summary, metric, label=baseline_label if col_idx == 0 else None)
        for component, label, _rms_col, _angle, color in COMPONENTS:
            _summary_line(ax, summary, component=component, metric=metric, color=color, label=label, ci=False)
        _feature_quantile_axis(ax, axis_label=axis_label, n_bins=n_bins)
        ax.set_ylabel(f"both\n{projected_ylabel}")
        if col_idx == 0:
            ax.legend(frameon=False, fontsize=8, loc="best")
        if metric == "net_to_path":
            ax.set_ylim(bottom=0.0)

    fig.suptitle(f"Scale diagnostics by {feature_label}: spread, path length, and cancellation", fontsize=15)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _coherence_band_summary_line(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    component: str,
    metric: str,
    coherence_band: str,
    color: str,
    label: str | None = None,
    min_sessions: int = 1,
    min_windows: int = 1,
) -> None:
    sub = summary[
        summary["component"].astype(str).eq(component)
        & summary["metric"].astype(str).eq(metric)
        & summary["coherence_band"].astype(str).eq(coherence_band)
    ].sort_values("bin_center")
    sub = sub[
        (sub["n_sessions"].astype(int) >= int(min_sessions))
        & (sub["n_windows"].astype(int) >= int(min_windows))
    ]
    if sub.empty:
        return
    x = sub["bin_center"].to_numpy(dtype=float)
    y = sub["session_median"].to_numpy(dtype=float)
    ax.plot(x, y, marker="o", color=color, linewidth=1.7, markersize=4.4, label=label)


def plot_scale_diagnostics_by_coherence_x_feature(
    summary: pd.DataFrame,
    out_path: Path,
    *,
    feature_label: str,
    axis_label: str,
) -> None:
    n_bins = int(summary["bin_order"].nunique()) if not summary.empty else COHERENCE_X_FEATURE_BIN_COUNT
    fig, axes = plt.subplots(3, 4, figsize=(13.0, 8.4), sharex=True, constrained_layout=True)
    metrics = _scale_metric_specs()
    component_rows = (
        ("parallel", "parallel", BLUE, False),
        ("orthogonal", "orthogonal", GRAY, False),
        (TRACE_2D_COMPONENT[0], TRACE_2D_COMPONENT[1], TRACE_2D_COMPONENT[2], True),
    )
    for row_idx, (component, component_label, _component_color, is_2d) in enumerate(component_rows):
        for col_idx, (metric, projected_ylabel, trace_ylabel, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for _lo, _hi, band_label, band_color in COHERENCE_X_FEATURE_BANDS:
                _coherence_band_summary_line(
                    ax,
                    summary,
                    component=component,
                    metric=metric,
                    coherence_band=band_label,
                    color=band_color,
                    label=band_label if row_idx == 0 and col_idx == 0 else None,
                    min_sessions=10,
                    min_windows=30,
                )
            _feature_quantile_axis(ax, axis_label=axis_label, n_bins=n_bins)
            if row_idx < len(component_rows) - 1:
                ax.set_xlabel("")
            if row_idx == 0:
                ax.set_title(title)
            ylabel = trace_ylabel if is_2d else projected_ylabel
            ax.set_ylabel(f"{component_label}\n{ylabel}")
            if metric == "net_to_path":
                ax.set_ylim(bottom=0.0)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle(
        f"Scale diagnostics by {feature_label}, split by local edge coherence "
        "(points require >=10 sessions and >=30 windows)",
        fontsize=15,
    )
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _axis_comparison_columns(
    coherence_summary: pd.DataFrame,
    feature_summaries: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    return [
        {
            "title": "local edge coherence",
            "axis_label": "local edge coherence",
            "kind": "coherence",
            "summary": coherence_summary,
            "baseline": f"{LOW_COHERENCE_BASELINE_BIN} coherence bin",
        },
        {
            "title": "orientation energy",
            "axis_label": "orientation energy quantile",
            "kind": "feature",
            "summary": feature_summaries["orientation_energy"],
            "baseline": "lowest orientation-energy bin",
        },
        {
            "title": "orientation energy x coherence",
            "axis_label": "coherent orientation energy quantile",
            "kind": "feature",
            "summary": feature_summaries["coherent_orientation_energy"],
            "baseline": "lowest coherent-energy bin",
        },
    ]


def _axis_comparison_x_axis(ax: plt.Axes, *, column: dict[str, Any], row_idx: int, n_rows: int) -> None:
    summary = column["summary"]
    if column["kind"] == "coherence":
        _coherence_axis(ax)
    else:
        n_bins = int(summary["bin_order"].nunique()) if not summary.empty else FEATURE_BIN_COUNT
        _feature_quantile_axis(ax, axis_label=str(column["axis_label"]), n_bins=n_bins)
    if row_idx < n_rows - 1:
        ax.set_xlabel("")


def _draw_axis_comparison_baseline(
    ax: plt.Axes,
    *,
    column: dict[str, Any],
    metric: str,
    components: tuple[str, ...],
    label: str | None = None,
) -> None:
    summary = column["summary"]
    if column["kind"] == "coherence":
        _draw_low_coherence_baseline(ax, summary, metric, components=components, label=label is not None)
    else:
        _draw_lowest_feature_bin_baseline(ax, summary, metric, components=components, label=label)


def plot_axis_comparison_projection_overlay(
    coherence_summary: pd.DataFrame,
    feature_summaries: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    metrics = _scale_metric_specs()
    columns = _axis_comparison_columns(coherence_summary, feature_summaries)
    fig, axes = plt.subplots(
        len(metrics),
        len(columns),
        figsize=(11.7, 9.0),
        sharey="row",
        constrained_layout=True,
    )
    for col_idx, column in enumerate(columns):
        summary = column["summary"]
        for row_idx, (metric, projected_ylabel, _trace_ylabel, metric_title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            baseline_label = str(column["baseline"]) if row_idx == 0 else None
            _draw_axis_comparison_baseline(
                ax,
                column=column,
                metric=metric,
                components=("parallel", "orthogonal"),
                label=baseline_label,
            )
            for component, label, _rms_col, _angle, color in COMPONENTS:
                _summary_line(
                    ax,
                    summary,
                    component=component,
                    metric=metric,
                    color=color,
                    label=label if row_idx == 0 else None,
                    ci=False,
                )
            _axis_comparison_x_axis(ax, column=column, row_idx=row_idx, n_rows=len(metrics))
            if row_idx == 0:
                ax.set_title(str(column["title"]))
            if col_idx == 0:
                ax.set_ylabel(f"{metric_title}\n{projected_ylabel}")
            else:
                ax.set_ylabel("")
            if metric == "net_to_path":
                ax.set_ylim(bottom=0.0)
            if row_idx == 0:
                ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Projected-component scale diagnostics across visual-feature axes", fontsize=15)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_axis_comparison_trace_2d(
    coherence_summary: pd.DataFrame,
    feature_summaries: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    metrics = _scale_metric_specs()
    columns = _axis_comparison_columns(coherence_summary, feature_summaries)
    trace_component, trace_label, trace_color = TRACE_2D_COMPONENT
    fig, axes = plt.subplots(
        len(metrics),
        len(columns),
        figsize=(11.7, 8.8),
        sharey="row",
        constrained_layout=True,
    )
    for col_idx, column in enumerate(columns):
        summary = column["summary"]
        for row_idx, (metric, _projected_ylabel, trace_ylabel, metric_title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            baseline_label = str(column["baseline"]) if row_idx == 0 else None
            _draw_axis_comparison_baseline(
                ax,
                column=column,
                metric=metric,
                components=(trace_component,),
                label=baseline_label,
            )
            _summary_line(
                ax,
                summary,
                component=trace_component,
                metric=metric,
                color=trace_color,
                label=trace_label if row_idx == 0 else None,
                ci=False,
            )
            _axis_comparison_x_axis(ax, column=column, row_idx=row_idx, n_rows=len(metrics))
            if row_idx == 0:
                ax.set_title(str(column["title"]))
            if col_idx == 0:
                ax.set_ylabel(f"{metric_title}\n{trace_ylabel}")
            else:
                ax.set_ylabel("")
            if metric == "net_to_path":
                ax.set_ylim(bottom=0.0)
            if row_idx == 0:
                ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("2D-trace scale diagnostics across visual-feature axes", fontsize=15)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_scale_diagnostics(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(13.0, 11.0), sharex=True, constrained_layout=True)
    metrics = _scale_metric_specs()
    for row_idx, (component, label, _rms_col, _angle, color) in enumerate(COMPONENTS):
        for col_idx, (metric, projected_ylabel, _trace_ylabel, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            _draw_low_coherence_baseline(ax, summary, metric)
            _summary_line(ax, summary, component=component, metric=metric, color=color, ci=False)
            _coherence_axis(ax)
            if row_idx == 0:
                ax.set_title(title)
            ax.set_xlabel("")
            ax.set_ylabel(f"{label}\n{projected_ylabel}")
            if metric == "net_to_path":
                ax.set_ylim(bottom=0.0)

    trace_component, trace_label, trace_color = TRACE_2D_COMPONENT
    for col_idx, (metric, _projected_ylabel, trace_ylabel, _title) in enumerate(metrics):
        ax = axes[2, col_idx]
        _draw_low_coherence_baseline(ax, summary, metric, components=(trace_component,))
        _summary_line(ax, summary, component=trace_component, metric=metric, color=trace_color, ci=False)
        _coherence_axis(ax)
        ax.set_xlabel("")
        ax.set_ylabel(f"{trace_label}\n{trace_ylabel}")
        if metric == "net_to_path":
            ax.set_ylim(bottom=0.0)

    for col_idx, (metric, projected_ylabel, _trace_ylabel, _title) in enumerate(metrics):
        ax = axes[3, col_idx]
        _draw_low_coherence_baseline(ax, summary, metric, label=col_idx == 0)
        for component, label, _rms_col, _angle, color in COMPONENTS:
            _summary_line(ax, summary, component=component, metric=metric, color=color, label=label, ci=False)
        _coherence_axis(ax)
        ax.set_ylabel(f"both\n{projected_ylabel}")
        if col_idx == 0:
            ax.legend(frameon=False, fontsize=8, loc="best")
        if metric == "net_to_path":
            ax.set_ylim(bottom=0.0)
    fig.suptitle("Scale diagnostics: spread, path length, and cancellation", fontsize=15)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_ratio_diagnostics(ratios: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.0), sharex=True, constrained_layout=True)
    specs = [
        ("rms_parallel_over_orthogonal", "RMS\nparallel / orthogonal", 1.0, BLUE),
        ("path_parallel_over_orthogonal", "unsigned path\nparallel / orthogonal", 1.0, PURPLE),
        ("net_parallel_over_orthogonal", "abs(net)\nparallel / orthogonal", 1.0, GREEN),
        ("path_per_rms_parallel_over_orthogonal", "distance/spread\nparallel / orthogonal", 1.0, ORANGE),
        ("mean_abs_step_parallel_over_orthogonal", "mean abs step\nparallel / orthogonal", 1.0, "#3f7f87"),
        ("reversal_orthogonal_minus_parallel", "reversal fraction\northogonal - parallel", 0.0, "#7b4f9d"),
    ]
    for ax, (metric, ylabel, baseline, color) in zip(axes.flat, specs, strict=True):
        _ratio_line(ax, ratios, metric=metric, color=color)
        ax.axhline(baseline, color="#333333", linewidth=1.0, linestyle="--")
        ax.set_ylabel(ylabel)
        _coherence_axis(ax)
    fig.suptitle("Parallel-orthogonal component diagnostics", fontsize=15)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_step_diagnostics(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), sharex=True, constrained_layout=True)
    metrics = [
        ("mean_abs_step_arcmin", "mean abs step\narcmin/frame", "local step size"),
        ("step_reversal_fraction", "step sign-reversal\nfraction", "back-and-forth"),
        ("step_autocorr_lag1", "projected step\nautocorr lag 1", "step persistence"),
    ]
    for row_idx, (component, label, _rms_col, _angle, color) in enumerate(COMPONENTS):
        for col_idx, (metric, ylabel, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            _summary_line(ax, summary, component=component, metric=metric, color=color, ci=True)
            if metric == "step_autocorr_lag1":
                ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--")
            _coherence_axis(ax)
            if row_idx == 0:
                ax.set_title(title)
            ax.set_ylabel(f"{label}\n{ylabel}")
    fig.suptitle("Step-level diagnostics behind unsigned path length", fontsize=15)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_msd_band_summary(
    diagnostics: pd.DataFrame,
    *,
    sample_rate_hz: float,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for band_idx, (lo, hi, label) in enumerate(MSD_BANDS):
        if hi >= 1.0:
            band = diagnostics[
                (diagnostics["image_orientation_coherence"] >= lo)
                & (diagnostics["image_orientation_coherence"] <= hi)
            ]
        else:
            band = diagnostics[
                (diagnostics["image_orientation_coherence"] >= lo)
                & (diagnostics["image_orientation_coherence"] < hi)
            ]
        for component, sub_component in band.groupby("component", observed=True):
            for lag in MSD_LAGS:
                metric = f"sqrt_msd_lag{lag}_arcmin"
                session_values = (
                    sub_component.groupby("session", observed=True)[metric]
                    .median(numeric_only=True)
                    .to_numpy(dtype=float)
                )
                point, ci_lo, ci_hi = _bootstrap_median_ci(session_values, rng=rng, n_bootstrap=n_bootstrap)
                rows.append(
                    {
                        "band": label,
                        "band_order": band_idx,
                        "bin_low": lo,
                        "bin_high": hi,
                        "component": str(component),
                        "lag_samples": int(lag),
                        "lag_s": float(lag / sample_rate_hz),
                        "sqrt_msd_session_median_arcmin": point,
                        "ci95_low": ci_lo,
                        "ci95_high": ci_hi,
                        "n_sessions": int(sub_component["session"].nunique()),
                        "n_windows": int(sub_component["window_index"].nunique()),
                    }
                )
    return pd.DataFrame(rows)


def plot_msd_growth(msd_summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(MSD_BANDS), figsize=(13.0, 3.6), sharey=True, constrained_layout=True)
    for ax, (_lo, _hi, label) in zip(axes, MSD_BANDS, strict=True):
        for component, comp_label, _rms_col, _angle, color in COMPONENTS:
            sub = msd_summary[
                msd_summary["band"].astype(str).eq(label)
                & msd_summary["component"].astype(str).eq(component)
            ].sort_values("lag_s")
            x = sub["lag_s"].to_numpy(dtype=float) * 1000.0
            y = sub["sqrt_msd_session_median_arcmin"].to_numpy(dtype=float)
            lo = sub["ci95_low"].to_numpy(dtype=float)
            hi = sub["ci95_high"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", color=color, linewidth=1.8, markersize=4.5, label=comp_label)
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.set_title(label)
        ax.set_xlabel("lag (ms)")
        ax.grid(axis="y", color=GRID, linewidth=0.9)
        _clean_axis(ax)
    axes[0].set_ylabel("sqrt(MSD) along component\n(arcmin)")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Displacement growth across lag: short-step motion vs longer-range spread", fontsize=15)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_orthogonal_focus(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.5), constrained_layout=True)
    specs = [
        ("rms_arcmin", "RMS spread", BLUE),
        (PATH_LENGTH_METRIC, "cumulative path length", PURPLE),
        ("path_per_rms", "path length / spread", ORANGE),
        ("net_to_path", "net / path length", GREEN),
    ]
    for metric, label, color in specs:
        sub = summary[
            summary["component"].astype(str).eq("orthogonal")
            & summary["metric"].astype(str).eq(metric)
        ].sort_values("bin_center")
        x = sub["bin_center"].to_numpy(dtype=float)
        y = sub["session_median"].to_numpy(dtype=float)
        finite = y[np.isfinite(y)]
        if finite.size == 0:
            continue
        denom = float(finite[0]) if finite[0] != 0.0 else float(np.nanmedian(finite))
        y_norm = y / denom if denom != 0.0 else y
        ax.plot(x, y_norm, marker="o", color=color, linewidth=2.0, markersize=5.0, label=f"{label} / first bin")
    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--")
    _coherence_axis(ax)
    ax.set_ylabel("orthogonal metric\nnormalized to 0.0-0.1 bin")
    ax.set_title("Orthogonal-axis focus: does path stay while spread shrinks?", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_final_bin_path_length_influence_tables(
    diagnostics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    distribution_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    leave_one_out_rows: list[dict[str, Any]] = []
    top_window_frames: list[pd.DataFrame] = []

    for component, component_label, _color in PATH_LENGTH_COMPONENTS:
        component_data = diagnostics[diagnostics["component"].astype(str).eq(component)].copy()
        final_data = component_data[component_data["coherence_bin"].astype(str).eq(FINAL_COHERENCE_BIN)].copy()
        final_sessions = set(final_data["session"].astype(str).unique())

        for coherence_bin, sub in component_data.groupby("coherence_bin", observed=True, sort=False):
            values = sub[PATH_LENGTH_METRIC].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            total = float(np.sum(values)) if values.size else float("nan")
            sorted_desc = np.sort(values)[::-1] if values.size else np.asarray([], dtype=float)
            matched = sub[sub["session"].astype(str).isin(final_sessions)]
            matched_session_medians = (
                matched.groupby("session", observed=True)[PATH_LENGTH_METRIC]
                .median(numeric_only=True)
                .to_numpy(dtype=float)
            )
            distribution_rows.append(
                {
                    "component": component,
                    "component_label": component_label,
                    "coherence_bin": str(coherence_bin),
                    "bin_center": _bin_center_from_label(str(coherence_bin)),
                    "n_windows": int(sub["window_index"].nunique()),
                    "n_sessions": int(sub["session"].nunique()),
                    "window_median": float(np.median(values)) if values.size else float("nan"),
                    "window_mean": float(np.mean(values)) if values.size else float("nan"),
                    "window_p90": float(np.quantile(values, 0.90)) if values.size else float("nan"),
                    "window_p95": float(np.quantile(values, 0.95)) if values.size else float("nan"),
                    "window_max": float(np.max(values)) if values.size else float("nan"),
                    "session_median": float(
                        sub.groupby("session", observed=True)[PATH_LENGTH_METRIC]
                        .median(numeric_only=True)
                        .median()
                    ),
                    "matched_final_sessions_session_median": _finite_median(matched_session_medians),
                    "top1_window_share": float(np.sum(sorted_desc[:1]) / total) if total > 0 else float("nan"),
                    "top3_window_share": float(np.sum(sorted_desc[:3]) / total) if total > 0 else float("nan"),
                    "top5_window_share": float(np.sum(sorted_desc[:5]) / total) if total > 0 else float("nan"),
                    "top10_window_share": float(np.sum(sorted_desc[:10]) / total) if total > 0 else float("nan"),
                }
            )

        if not final_data.empty:
            session_summary = (
                final_data.groupby(["session", "subject"], observed=True)
                .agg(
                    n_windows=("window_index", "nunique"),
                    window_median=(PATH_LENGTH_METRIC, "median"),
                    window_mean=(PATH_LENGTH_METRIC, "mean"),
                    window_p90=(PATH_LENGTH_METRIC, lambda x: float(np.quantile(x, 0.90))),
                    window_max=(PATH_LENGTH_METRIC, "max"),
                )
                .reset_index()
                .sort_values("window_median", ascending=False)
            )
            session_summary.insert(0, "component_label", component_label)
            session_summary.insert(0, "component", component)
            session_rows.extend(session_summary.to_dict("records"))

            session_medians = final_data.groupby("session", observed=True)[PATH_LENGTH_METRIC].median(numeric_only=True)
            full_median = float(session_medians.median())
            for session, session_median in session_medians.items():
                remaining = session_medians.drop(index=session)
                leave_one_out_median = float(remaining.median()) if not remaining.empty else float("nan")
                leave_one_out_rows.append(
                    {
                        "component": component,
                        "component_label": component_label,
                        "left_out_session": str(session),
                        "left_out_session_median": float(session_median),
                        "left_out_n_windows": int(final_data[final_data["session"].astype(str).eq(str(session))]["window_index"].nunique()),
                        "full_final_bin_session_median": full_median,
                        "leave_one_out_session_median": leave_one_out_median,
                        "leave_one_out_minus_full": leave_one_out_median - full_median,
                    }
                )

            top_cols = [
                "component",
                "component_label",
                "window_index",
                "session",
                "subject",
                "stimulus",
                "regime",
                "trial_idx",
                "global_start",
                "global_stop",
                "local_start",
                "local_stop",
                "phase",
                "coherence_bin",
                "image_orientation_coherence",
                "rms_arcmin",
                PATH_LENGTH_METRIC,
                "path_per_rms",
                "net_to_path",
            ]
            present_cols = [col for col in top_cols if col in final_data.columns]
            top_window_frames.append(
                final_data.sort_values(PATH_LENGTH_METRIC, ascending=False).head(25)[present_cols].copy()
            )

    distribution = pd.DataFrame(distribution_rows)
    session_summary = pd.DataFrame(session_rows)
    leave_one_out = pd.DataFrame(leave_one_out_rows)
    top_windows = pd.concat(top_window_frames, ignore_index=True) if top_window_frames else pd.DataFrame()
    return distribution, session_summary, leave_one_out, top_windows


def _plot_box_with_points(
    ax: plt.Axes,
    values_by_bin: list[np.ndarray],
    *,
    color: str,
    rng: np.random.Generator,
    show_clip_note: bool = True,
) -> None:
    finite_values = np.concatenate([vals[np.isfinite(vals)] for vals in values_by_bin if vals.size])
    if finite_values.size == 0:
        return
    last_values = values_by_bin[-1]
    last_values = last_values[np.isfinite(last_values)]
    final_bin_cap = float(np.max(last_values) * 1.05) if last_values.size else 1.0
    cap = float(max(np.nanpercentile(finite_values, 99.0), final_bin_cap))
    cap = max(cap, 1.0)
    box_values = [np.minimum(vals[np.isfinite(vals)], cap) for vals in values_by_bin]
    bp = ax.boxplot(
        box_values,
        positions=np.arange(1, len(values_by_bin) + 1),
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 1.4},
        whiskerprops={"color": "#69727c", "linewidth": 1.0},
        capprops={"color": "#69727c", "linewidth": 1.0},
    )
    for patch in bp["boxes"]:
        patch.set(facecolor=color, alpha=0.18, edgecolor=color, linewidth=1.0)
    for idx, vals in enumerate(values_by_bin, start=1):
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.13, 0.13, size=vals.size)
        clipped = vals > cap
        ax.scatter(
            np.full(vals.size, idx) + jitter,
            np.minimum(vals, cap),
            s=12,
            color=color,
            alpha=0.28,
            linewidths=0,
            zorder=2,
        )
        if np.any(clipped):
            ax.scatter(
                np.full(int(np.sum(clipped)), idx),
                np.full(int(np.sum(clipped)), cap),
                marker="^",
                s=26,
                color=INK,
                alpha=0.65,
                linewidths=0,
                zorder=3,
            )
    ax.set_ylim(bottom=0.0, top=cap * 1.08)
    if show_clip_note:
        ax.text(
            0.98,
            0.96,
            "triangles: clipped high windows",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#50575f",
        )


def plot_final_bin_path_length_influence(
    diagnostics: pd.DataFrame,
    distribution: pd.DataFrame,
    session_summary: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 8.7), constrained_layout=True)
    rng = np.random.default_rng(12345)

    for col_idx, (component, component_label, color) in enumerate(PATH_LENGTH_COMPONENTS):
        component_data = diagnostics[diagnostics["component"].astype(str).eq(component)]
        values_by_bin = [
            component_data[component_data["coherence_bin"].astype(str).eq(bin_label)][PATH_LENGTH_METRIC].to_numpy(dtype=float)
            for bin_label in FINAL_BIN_NEIGHBOR_BINS
        ]

        ax = axes[0, col_idx]
        _plot_box_with_points(ax, values_by_bin, color=color, rng=rng)
        ax.set_xticks(np.arange(1, len(FINAL_BIN_NEIGHBOR_BINS) + 1), FINAL_BIN_NEIGHBOR_BINS, rotation=25)
        ax.set_title(component_label)
        ax.set_ylabel("window cumulative\npath length\n(arcmin / 0.325 s)")
        ax.grid(axis="y", color=GRID, linewidth=0.9)
        _clean_axis(ax)

        final_sessions = session_summary[session_summary["component"].astype(str).eq(component)].copy()
        final_sessions = final_sessions.sort_values("window_median", ascending=True).reset_index(drop=True)
        ax = axes[1, col_idx]
        if not final_sessions.empty:
            x = np.arange(final_sessions.shape[0])
            y = final_sessions["window_median"].to_numpy(dtype=float)
            sizes = 24.0 + 4.2 * np.sqrt(final_sessions["n_windows"].to_numpy(dtype=float))
            subjects = final_sessions["subject"].astype(str).to_numpy()
            colors = np.where(subjects == "Allen", BLUE, ORANGE)
            ax.scatter(x, y, s=sizes, color=colors, alpha=0.78, edgecolor="white", linewidth=0.6)
            ax.axhline(float(np.median(y)), color="#33383f", linewidth=1.1, linestyle=(0, (4.0, 2.4)))
            for rank, row in final_sessions.tail(3).iterrows():
                ax.text(
                    float(rank) + 0.18,
                    float(row["window_median"]),
                    str(row["session"]).replace("_20", "\n20"),
                    fontsize=6.7,
                    va="center",
                    color="#454b52",
                )
        ax.set_xlabel("final-bin sessions sorted by median")
        ax.set_ylabel("session median\npath length")
        ax.grid(axis="y", color=GRID, linewidth=0.9)
        _clean_axis(ax)

        loo = leave_one_out[leave_one_out["component"].astype(str).eq(component)].copy()
        ax = axes[2, col_idx]
        if not loo.empty:
            loo = loo.sort_values("leave_one_out_minus_full").reset_index(drop=True)
            x = np.arange(loo.shape[0])
            delta = loo["leave_one_out_minus_full"].to_numpy(dtype=float)
            ax.axhline(0.0, color="#33383f", linewidth=1.0, linestyle="--")
            ax.vlines(x, 0.0, delta, color=color, alpha=0.35, linewidth=1.3)
            ax.scatter(x, delta, color=color, s=18, alpha=0.85)
            top_share = distribution[
                distribution["component"].astype(str).eq(component)
                & distribution["coherence_bin"].astype(str).eq(FINAL_COHERENCE_BIN)
            ]
            if not top_share.empty:
                row = top_share.iloc[0]
                ax.text(
                    0.02,
                    0.95,
                    f"top 1/5/10 windows:\n"
                    f"{100.0 * row['top1_window_share']:.1f}% / "
                    f"{100.0 * row['top5_window_share']:.1f}% / "
                    f"{100.0 * row['top10_window_share']:.1f}%",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="#454b52",
                )
        ax.set_xlabel("left-out session rank")
        ax.set_ylabel("change in final-bin\nsession median")
        ax.grid(axis="y", color=GRID, linewidth=0.9)
        _clean_axis(ax)

    axes[0, 0].set_title("parallel\nwindow distribution")
    axes[0, 1].set_title("orthogonal\nwindow distribution")
    axes[0, 2].set_title("2D trace\nwindow distribution")
    fig.suptitle("Final high-coherence bin influence check for cumulative path length", fontsize=15)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_final_bin_vs_overall_path_length_comparison(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    quantiles = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

    def add_row(
        *,
        component: str,
        component_label: str,
        level: str,
        overall_values: np.ndarray,
        final_values: np.ndarray,
    ) -> None:
        overall = np.asarray(overall_values, dtype=np.float64)
        overall = overall[np.isfinite(overall)]
        final = np.asarray(final_values, dtype=np.float64)
        final = final[np.isfinite(final)]
        row: dict[str, Any] = {
            "component": component,
            "component_label": component_label,
            "level": level,
            "n_overall": int(overall.size),
            "n_final_bin": int(final.size),
        }
        for q in quantiles:
            row[f"overall_q{int(q * 100):02d}"] = float(np.quantile(overall, q)) if overall.size else float("nan")
            row[f"final_q{int(q * 100):02d}"] = float(np.quantile(final, q)) if final.size else float("nan")
        row["overall_mean"] = float(np.mean(overall)) if overall.size else float("nan")
        row["final_mean"] = float(np.mean(final)) if final.size else float("nan")
        row["overall_max"] = float(np.max(overall)) if overall.size else float("nan")
        row["final_max"] = float(np.max(final)) if final.size else float("nan")
        final_median = row["final_q50"]
        row["final_median_percentile_in_overall"] = (
            float(100.0 * np.mean(overall <= final_median)) if overall.size and np.isfinite(final_median) else float("nan")
        )
        for q in (0.75, 0.90, 0.95, 0.99):
            threshold = row[f"overall_q{int(q * 100):02d}"]
            row[f"final_fraction_above_overall_q{int(q * 100):02d}"] = (
                float(np.mean(final > threshold)) if final.size and np.isfinite(threshold) else float("nan")
            )
        rows.append(row)

    for component, component_label, _color in PATH_LENGTH_COMPONENTS:
        component_data = diagnostics[diagnostics["component"].astype(str).eq(component)].copy()
        final_data = component_data[component_data["coherence_bin"].astype(str).eq(FINAL_COHERENCE_BIN)].copy()

        add_row(
            component=component,
            component_label=component_label,
            level="window",
            overall_values=component_data[PATH_LENGTH_METRIC].to_numpy(dtype=float),
            final_values=final_data[PATH_LENGTH_METRIC].to_numpy(dtype=float),
        )

        overall_session_bins = (
            component_data.groupby(["coherence_bin", "session"], observed=True)[PATH_LENGTH_METRIC]
            .median(numeric_only=True)
            .to_numpy(dtype=float)
        )
        final_sessions = (
            final_data.groupby("session", observed=True)[PATH_LENGTH_METRIC]
            .median(numeric_only=True)
            .to_numpy(dtype=float)
        )
        add_row(
            component=component,
            component_label=component_label,
            level="session_bin_median",
            overall_values=overall_session_bins,
            final_values=final_sessions,
        )

    return pd.DataFrame(rows)


def _annotate_overall_comparison(ax: plt.Axes, comparison: pd.DataFrame, *, component: str, level: str) -> None:
    row = comparison[
        comparison["component"].astype(str).eq(component)
        & comparison["level"].astype(str).eq(level)
    ]
    if row.empty:
        return
    record = row.iloc[0]
    ax.text(
        0.03,
        0.96,
        f"final median at overall p{record['final_median_percentile_in_overall']:.0f}\n"
        f"final > overall p90: {100.0 * record['final_fraction_above_overall_q90']:.1f}%\n"
        f"final > overall p95: {100.0 * record['final_fraction_above_overall_q95']:.1f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#454b52",
    )


def plot_final_bin_vs_overall_path_length(
    diagnostics: pd.DataFrame,
    comparison: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.8), constrained_layout=True)
    rng = np.random.default_rng(54321)

    for col_idx, (component, component_label, color) in enumerate(PATH_LENGTH_COMPONENTS):
        component_data = diagnostics[diagnostics["component"].astype(str).eq(component)].copy()
        final_data = component_data[component_data["coherence_bin"].astype(str).eq(FINAL_COHERENCE_BIN)].copy()

        ax = axes[0, col_idx]
        window_values = [
            component_data[PATH_LENGTH_METRIC].to_numpy(dtype=float),
            final_data[PATH_LENGTH_METRIC].to_numpy(dtype=float),
        ]
        _plot_box_with_points(ax, window_values, color=color, rng=rng, show_clip_note=False)
        ax.set_xticks([1, 2], ["overall", FINAL_COHERENCE_BIN])
        ax.set_title(f"{component_label}: windows")
        ax.set_ylabel("window cumulative\npath length\n(arcmin / 0.325 s)")
        ax.grid(axis="y", color=GRID, linewidth=0.9)
        _clean_axis(ax)
        _annotate_overall_comparison(ax, comparison, component=component, level="window")

        overall_session_bins = (
            component_data.groupby(["coherence_bin", "session"], observed=True)[PATH_LENGTH_METRIC]
            .median(numeric_only=True)
            .to_numpy(dtype=float)
        )
        final_sessions = (
            final_data.groupby("session", observed=True)[PATH_LENGTH_METRIC]
            .median(numeric_only=True)
            .to_numpy(dtype=float)
        )
        ax = axes[1, col_idx]
        _plot_box_with_points(ax, [overall_session_bins, final_sessions], color=color, rng=rng, show_clip_note=False)
        ax.set_xticks([1, 2], ["overall\nsession-bins", f"{FINAL_COHERENCE_BIN}\nsessions"])
        ax.set_title(f"{component_label}: session medians")
        ax.set_ylabel("session-bin median\npath length")
        ax.grid(axis="y", color=GRID, linewidth=0.9)
        _clean_axis(ax)
        _annotate_overall_comparison(ax, comparison, component=component, level="session_bin_median")

    fig.suptitle("Final high-coherence bin compared with the overall path-length distribution", fontsize=15)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    diagnostics: pd.DataFrame,
    summary: pd.DataFrame,
    ratios: pd.DataFrame,
    msd_summary: pd.DataFrame,
    out_dir: Path,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()
    paths: dict[str, Path] = {}
    paths["windows_csv"] = out_dir / "b_path_spread_step_diagnostics_windows.csv"
    paths["summary_csv"] = out_dir / "b_path_spread_step_diagnostics_summary.csv"
    paths["ratios_csv"] = out_dir / "b_path_spread_step_diagnostics_component_ratios.csv"
    paths["msd_summary_csv"] = out_dir / "b_path_spread_step_diagnostics_msd_growth_summary.csv"
    paths["final_bin_path_length_distribution_csv"] = (
        out_dir / "b_path_spread_final_bin_path_length_distribution_summary.csv"
    )
    paths["final_bin_path_length_sessions_csv"] = out_dir / "b_path_spread_final_bin_path_length_session_medians.csv"
    paths["final_bin_path_length_leave_one_out_csv"] = (
        out_dir / "b_path_spread_final_bin_path_length_leave_one_session_out.csv"
    )
    paths["final_bin_path_length_top_windows_csv"] = out_dir / "b_path_spread_final_bin_path_length_top_windows.csv"
    paths["final_bin_vs_overall_path_length_csv"] = (
        out_dir / "b_path_spread_final_bin_vs_overall_path_length_summary.csv"
    )
    feature_summaries: dict[str, pd.DataFrame] = {}
    for feature_idx, spec in enumerate(FEATURE_BIN_SPECS):
        feature = str(spec["feature"])
        feature_summary = summarize_by_feature_quantile(
            diagnostics,
            feature_col=str(spec["column"]),
            feature=feature,
            feature_label=str(spec["label"]),
            n_bins=FEATURE_BIN_COUNT,
            n_bootstrap=n_bootstrap,
            seed=seed + 400_000 + feature_idx,
        )
        feature_summaries[feature] = feature_summary
        paths[f"{feature}_scale_summary_csv"] = out_dir / f"{spec['stem']}_summary.csv"

    coherence_x_orientation_energy = summarize_by_coherence_band_and_feature_quantile(
        diagnostics,
        feature_col="image_orientation_energy",
        feature="coherence_x_orientation_energy",
        feature_label="orientation energy",
        n_feature_bins=COHERENCE_X_FEATURE_BIN_COUNT,
        n_bootstrap=n_bootstrap,
        seed=seed + 500_000,
    )
    paths["coherence_x_orientation_energy_scale_summary_csv"] = (
        out_dir / "b_path_spread_scale_diagnostics_by_coherence_x_orientation_energy_summary.csv"
    )

    diagnostics.to_csv(paths["windows_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    ratios.to_csv(paths["ratios_csv"], index=False)
    msd_summary.to_csv(paths["msd_summary_csv"], index=False)
    for feature, feature_summary in feature_summaries.items():
        feature_summary.to_csv(paths[f"{feature}_scale_summary_csv"], index=False)
    coherence_x_orientation_energy.to_csv(paths["coherence_x_orientation_energy_scale_summary_csv"], index=False)
    final_distribution, final_sessions, final_leave_one_out, final_top_windows = (
        make_final_bin_path_length_influence_tables(diagnostics)
    )
    final_vs_overall = make_final_bin_vs_overall_path_length_comparison(diagnostics)
    final_distribution.to_csv(paths["final_bin_path_length_distribution_csv"], index=False)
    final_sessions.to_csv(paths["final_bin_path_length_sessions_csv"], index=False)
    final_leave_one_out.to_csv(paths["final_bin_path_length_leave_one_out_csv"], index=False)
    final_top_windows.to_csv(paths["final_bin_path_length_top_windows_csv"], index=False)
    final_vs_overall.to_csv(paths["final_bin_vs_overall_path_length_csv"], index=False)

    figure_specs = [
        ("scale_png", "scale_pdf", "b_path_spread_scale_diagnostics_by_edge_coherence", plot_scale_diagnostics, summary),
        ("ratio_png", "ratio_pdf", "b_path_spread_component_ratio_diagnostics_by_edge_coherence", plot_ratio_diagnostics, ratios),
        ("step_png", "step_pdf", "b_path_spread_step_reversal_diagnostics_by_edge_coherence", plot_step_diagnostics, summary),
        ("msd_png", "msd_pdf", "b_path_spread_msd_growth_by_edge_coherence", plot_msd_growth, msd_summary),
        ("orthogonal_focus_png", "orthogonal_focus_pdf", "b_path_spread_orthogonal_focus_by_edge_coherence", plot_orthogonal_focus, summary),
    ]
    for png_key, pdf_key, stem_name, func, data in figure_specs:
        stem = out_dir / stem_name
        func(data, stem)
        paths[png_key] = stem.with_suffix(".png")
        paths[pdf_key] = stem.with_suffix(".pdf")
    final_influence_stem = out_dir / "b_path_spread_final_bin_path_length_influence"
    plot_final_bin_path_length_influence(
        diagnostics,
        final_distribution,
        final_sessions,
        final_leave_one_out,
        final_influence_stem,
    )
    paths["final_bin_path_length_influence_png"] = final_influence_stem.with_suffix(".png")
    paths["final_bin_path_length_influence_pdf"] = final_influence_stem.with_suffix(".pdf")
    final_vs_overall_stem = out_dir / "b_path_spread_final_bin_vs_overall_path_length"
    plot_final_bin_vs_overall_path_length(diagnostics, final_vs_overall, final_vs_overall_stem)
    paths["final_bin_vs_overall_path_length_png"] = final_vs_overall_stem.with_suffix(".png")
    paths["final_bin_vs_overall_path_length_pdf"] = final_vs_overall_stem.with_suffix(".pdf")
    for spec in FEATURE_BIN_SPECS:
        feature = str(spec["feature"])
        stem = out_dir / str(spec["stem"])
        plot_scale_diagnostics_by_feature(
            feature_summaries[feature],
            stem,
            feature_label=str(spec["label"]),
            axis_label=str(spec["axis_label"]),
        )
        paths[f"{feature}_scale_png"] = stem.with_suffix(".png")
        paths[f"{feature}_scale_pdf"] = stem.with_suffix(".pdf")
    coherence_x_orientation_energy_stem = out_dir / "b_path_spread_scale_diagnostics_by_coherence_x_orientation_energy"
    plot_scale_diagnostics_by_coherence_x_feature(
        coherence_x_orientation_energy,
        coherence_x_orientation_energy_stem,
        feature_label="orientation energy",
        axis_label="orientation energy quantile",
    )
    paths["coherence_x_orientation_energy_scale_png"] = coherence_x_orientation_energy_stem.with_suffix(".png")
    paths["coherence_x_orientation_energy_scale_pdf"] = coherence_x_orientation_energy_stem.with_suffix(".pdf")
    axis_comparison_projection_stem = out_dir / "b_path_spread_scale_axis_comparison_projection_overlay"
    plot_axis_comparison_projection_overlay(summary, feature_summaries, axis_comparison_projection_stem)
    paths["axis_comparison_projection_overlay_png"] = axis_comparison_projection_stem.with_suffix(".png")
    paths["axis_comparison_projection_overlay_pdf"] = axis_comparison_projection_stem.with_suffix(".pdf")
    axis_comparison_2d_stem = out_dir / "b_path_spread_scale_axis_comparison_2d_trace"
    plot_axis_comparison_trace_2d(summary, feature_summaries, axis_comparison_2d_stem)
    paths["axis_comparison_2d_trace_png"] = axis_comparison_2d_stem.with_suffix(".png")
    paths["axis_comparison_2d_trace_pdf"] = axis_comparison_2d_stem.with_suffix(".pdf")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-windows", type=Path, default=DEFAULT_INPUT_WINDOWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--coherence-bin-width", type=float, default=0.1)
    parser.add_argument("--equivalent-window-s", type=float, default=0.325)
    parser.add_argument("--sample-rate-hz", type=float, default=120.0)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> Path:
    windows, _bin_meta = load_windows(Path(args.input_windows), coherence_bin_width=float(args.coherence_bin_width))
    diagnostics = compute_component_diagnostics(windows, equivalent_window_s=float(args.equivalent_window_s))
    summary = summarize_by_coherence(
        diagnostics,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed) + 100_000,
    )
    ratios = summarize_component_ratios(
        diagnostics,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed) + 200_000,
    )
    msd_summary = make_msd_band_summary(
        diagnostics,
        sample_rate_hz=float(args.sample_rate_hz),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed) + 300_000,
    )
    out_dir = Path(args.out_dir)
    paths = write_outputs(
        diagnostics,
        summary,
        ratios,
        msd_summary,
        out_dir,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    cfg = DiagnosticConfig(
        input_windows=str(Path(args.input_windows)),
        out_dir=str(out_dir),
        coherence_bin_width=float(args.coherence_bin_width),
        equivalent_window_s=float(args.equivalent_window_s),
        sample_rate_hz=float(args.sample_rate_hz),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    write_json(
        out_dir / "backimage_contour_path_spread_step_diagnostics_metadata.json",
        {
            "config": asdict(cfg),
            "n_windows": int(windows.shape[0]),
            "n_sessions": int(windows["session"].nunique()),
            "outputs": {key: str(value) for key, value in sorted(paths.items())},
            "metric_notes": {
                "rms_arcmin": "Position-cloud spread along the component axis.",
                "unsigned_path_arcmin_equiv": "Cumulative path length scaled to the equivalent window duration: sum(abs(projected sample-to-sample steps)) for contour-relative components, or sum(Euclidean step lengths) for the 2D trace.",
                "path_per_rms": "Cumulative path length divided by RMS spread; larger values mean more local path length per unit spatial spread.",
                "net_to_path": "Net displacement divided by cumulative path length; smaller values mean more cancellation.",
                "step_reversal_fraction": "Fraction of adjacent projected steps with opposite sign.",
                "msd_growth": "sqrt(MSD) by lag; short-lag similarity with long-lag separation suggests local motion that cancels before becoming broad spread.",
                "image_orientation_energy": "Alias for raw local Sobel gradient energy; it measures local visual signal strength but can be high for multi-orientation texture.",
                "image_coherent_orientation_energy": "image_orientation_energy * image_orientation_coherence; equivalent to the anisotropic part of the local Sobel structure-tensor energy.",
                "feature_quantile_bins": "Energy-axis plots use within-sample quantile bins because image_orientation_energy and image_coherent_orientation_energy are heavy-tailed.",
                "coherence_x_orientation_energy": "Coherence-by-energy plots split orientation-energy quantiles by broad local edge-coherence bands and omit plotted cells with fewer than 10 sessions or 30 windows.",
            },
        },
    )
    print(f"Wrote BackImage path/spread diagnostics to {out_dir}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
