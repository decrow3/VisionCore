#!/usr/bin/env python3
"""Generate BackImage contour-relative signed path-component follow-up plots.

This mirrors the coherence bins and unwrapped plotting convention used by
``generate_backimage_contour_position_spread_followups.py``, but replaces the
position-spread RMS metric with trace-derived step accumulation metrics:

* signed net component: ``sum(projected step)`` along each relative axis;
* net component magnitude: ``abs(sum(projected step))``;
* unsigned component path: ``sum(abs(projected step))``.

The first metric is useful as a cancellation/bias check, but the local contour
axis is axial rather than directional, so its sign should not be interpreted as
a stable biological direction. The magnitude and unsigned-path rows are the
safer diagnostics for asking whether orthogonal-axis path reflects net displacement
or back-and-forth jitter.
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
from .io_utils import write_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_contour_motion_component_plots_v1"
)
DEFAULT_INPUT_WINDOWS = DEFAULT_OUT_DIR / "contour_motion_component_windows.csv"
DEFAULT_RMS_PROGRESSION = DEFAULT_OUT_DIR / "b_position_spread_progression_by_edge_coherence.csv"

BLUE = "#3366aa"
GRAY = "#7f8b96"
PURPLE = "#7a3b9a"
GREEN = "#1b7f5c"
GRID = "#d5dce3"
BAR = "#d7dde4"
RANDOM = "#33383f"


METRIC_SPECS = (
    (
        "signed_net",
        "signed net component",
        "signed_net_component_arcmin_equiv",
        "signed net component\narcmin equiv.",
        BLUE,
    ),
    (
        "net_magnitude",
        "net component magnitude",
        "net_component_magnitude_arcmin_equiv",
        "abs(net component)\narcmin equiv.",
        GREEN,
    ),
    (
        "unsigned_path",
        "unsigned component path",
        "unsigned_component_path_arcmin_equiv",
        "unsigned path\narcmin equiv.",
        PURPLE,
    ),
)


@dataclass(frozen=True)
class SignedPathConfig:
    input_windows: str
    rms_progression: str
    out_dir: str
    coherence_bin_width: float
    angle_step_deg: float
    equivalent_window_s: float
    n_bootstrap: int
    n_random_orientation: int
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


def _coherence_bin_edges(width: float) -> np.ndarray:
    if not np.isfinite(width) or width <= 0.0 or width > 1.0:
        raise ValueError(f"coherence bin width must be in (0, 1], got {width}")
    n_bins = int(round(1.0 / float(width)))
    if not np.isclose(n_bins * float(width), 1.0):
        raise ValueError("coherence bin width must divide 1.0 exactly")
    return np.linspace(0.0, 1.0, n_bins + 1)


def _format_bin_label(lo: float, hi: float) -> str:
    return f"{lo:.1f}-{hi:.1f}"


def _add_coherence_bins(df: pd.DataFrame, *, width: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges = _coherence_bin_edges(width)
    coherence = pd.to_numeric(df["image_orientation_coherence"], errors="coerce")
    labels = []
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        label = _format_bin_label(float(lo), float(hi))
        labels.append(label)
        rows.append(
            {
                "coherence_bin": label,
                "bin_low": float(lo),
                "bin_high": float(hi),
                "bin_center": float(0.5 * (lo + hi)),
            }
        )
    cats = pd.cut(coherence, edges, labels=labels, include_lowest=True, right=False)
    cats = cats.astype(object)
    cats[np.isclose(coherence.to_numpy(dtype=float), 1.0)] = labels[-1]
    out = df.copy()
    out["coherence_bin"] = pd.Categorical(cats, categories=labels, ordered=True)
    out = out[out["coherence_bin"].notna()].copy()
    return out, pd.DataFrame(rows)


def load_windows(path: Path, *, coherence_bin_width: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = [
        "session",
        "trial_idx",
        "global_start",
        "global_stop",
        "local_start",
        "local_stop",
        "duration_s",
        "image_orientation_coherence",
        "image_edge_axis_deg",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    for col in [
        "trial_idx",
        "global_start",
        "global_stop",
        "local_start",
        "local_stop",
        "duration_s",
        "image_orientation_coherence",
        "image_edge_axis_deg",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    ok = df["session"].notna()
    for col in [
        "trial_idx",
        "global_start",
        "global_stop",
        "local_start",
        "local_stop",
        "duration_s",
        "image_orientation_coherence",
        "image_edge_axis_deg",
    ]:
        ok &= np.isfinite(df[col])
    ok &= df["duration_s"] > 0
    df = df[ok].copy()
    return _add_coherence_bins(df, width=coherence_bin_width)


def _angle_grid(angle_step_deg: float) -> np.ndarray:
    if not np.isfinite(angle_step_deg) or angle_step_deg <= 0:
        raise ValueError(f"angle-step-deg must be positive, got {angle_step_deg}")
    return np.arange(0.0, 180.0 + 0.5 * float(angle_step_deg), float(angle_step_deg), dtype=np.float64)


def _component_matrices_for_row(
    row: pd.Series,
    angles_rad: np.ndarray,
    *,
    equivalent_window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    trace = contour_motion._window_trace(row)
    x = np.asarray(trace, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] < 2:
        nan = np.full_like(angles_rad, np.nan, dtype=np.float64)
        return nan, nan
    steps = np.diff(x, axis=0)
    ok = np.isfinite(steps).all(axis=1)
    steps = steps[ok]
    if steps.size == 0:
        nan = np.full_like(angles_rad, np.nan, dtype=np.float64)
        return nan, nan
    edge_axis_rad = np.radians(float(row["image_edge_axis_deg"]))
    axes = np.column_stack(
        [
            np.cos(edge_axis_rad + angles_rad),
            np.sin(edge_axis_rad + angles_rad),
        ]
    )
    projected = steps @ axes.T
    signed_net = np.sum(projected, axis=0) * 60.0
    unsigned_path = np.sum(np.abs(projected), axis=0) * 60.0
    if np.isfinite(equivalent_window_s) and equivalent_window_s > 0:
        duration = float(row["duration_s"])
        scale = float(equivalent_window_s) / duration if duration > 0 else float("nan")
        signed_net = signed_net * scale
        unsigned_path = unsigned_path * scale
    return signed_net, unsigned_path


def compute_window_profiles(
    df: pd.DataFrame,
    *,
    angle_step_deg: float,
    equivalent_window_s: float,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, dict[str, np.ndarray]]]:
    angles_deg = _angle_grid(angle_step_deg)
    angles_rad = np.radians(angles_deg)
    rows: list[dict[str, Any]] = []
    matrices: dict[str, dict[str, list[np.ndarray]]] = {
        str(label): {key: [] for key, *_rest in METRIC_SPECS}
        for label in df["coherence_bin"].cat.categories
    }
    total = len(df)
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        series = pd.Series(row._asdict())
        signed_net, unsigned_path = _component_matrices_for_row(
            series,
            angles_rad,
            equivalent_window_s=equivalent_window_s,
        )
        net_magnitude = np.abs(signed_net)
        band = str(series["coherence_bin"])
        matrices[band]["signed_net"].append(signed_net)
        matrices[band]["net_magnitude"].append(net_magnitude)
        matrices[band]["unsigned_path"].append(unsigned_path)
        parallel_idx = 0
        orthogonal_idx = int(np.argmin(np.abs(angles_deg - 90.0)))
        rows.append(
            {
                "session": str(series["session"]),
                "coherence_bin": band,
                "parallel_signed_net_component_arcmin_equiv": float(signed_net[parallel_idx]),
                "orthogonal_signed_net_component_arcmin_equiv": float(signed_net[orthogonal_idx]),
                "parallel_net_component_magnitude_arcmin_equiv": float(net_magnitude[parallel_idx]),
                "orthogonal_net_component_magnitude_arcmin_equiv": float(net_magnitude[orthogonal_idx]),
                "parallel_unsigned_component_path_arcmin_equiv": float(unsigned_path[parallel_idx]),
                "orthogonal_unsigned_component_path_arcmin_equiv": float(unsigned_path[orthogonal_idx]),
            }
        )
        if idx % 1000 == 0:
            print(f"computed signed path profiles for {idx}/{total} windows", flush=True)

    matrix_out: dict[str, dict[str, np.ndarray]] = {}
    for band, metric_map in matrices.items():
        matrix_out[band] = {}
        for metric, values in metric_map.items():
            matrix_out[band][metric] = np.vstack(values) if values else np.empty((0, len(angles_deg)), dtype=np.float64)
    return pd.DataFrame(rows), angles_deg, matrix_out


def _randomized_axis_profiles(
    matrix: np.ndarray,
    *,
    metric_key: str,
    angles_deg: np.ndarray,
    n_random_orientation: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_unique = max(1, len(angles_deg) - 1)
    row_idx = np.arange(matrix.shape[0])
    null_profiles = np.full((int(n_random_orientation), len(angles_deg)), np.nan, dtype=np.float64)
    for i in range(int(n_random_orientation)):
        offset_idx = rng.integers(0, n_unique, size=matrix.shape[0])
        for angle_idx in range(len(angles_deg)):
            wrapped_idx = (offset_idx + angle_idx) % n_unique
            values = matrix[row_idx, wrapped_idx]
            if metric_key == "signed_net":
                flip = ((offset_idx + angle_idx) // n_unique) % 2
                values = np.where(flip == 0, values, -values)
            null_profiles[i, angle_idx] = float(np.nanmedian(values))
    return null_profiles


def summarize_profiles(
    matrices: dict[str, dict[str, np.ndarray]],
    bin_meta: pd.DataFrame,
    angles_deg: np.ndarray,
    *,
    n_random_orientation: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    profile_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    for meta in bin_meta.itertuples(index=False):
        band = str(meta.coherence_bin)
        for metric_key, metric_label, value_col, _ylabel, _color in METRIC_SPECS:
            matrix = matrices[band][metric_key]
            valid_rows = np.isfinite(matrix).any(axis=1)
            matrix = matrix[valid_rows]
            if matrix.size == 0:
                continue
            observed = np.nanmedian(matrix, axis=0)
            null_profiles = _randomized_axis_profiles(
                matrix,
                metric_key=metric_key,
                angles_deg=angles_deg,
                n_random_orientation=n_random_orientation,
                rng=rng,
            )
            ref_median = np.nanmedian(null_profiles, axis=0)
            ref_lo = np.nanquantile(null_profiles, 0.025, axis=0)
            ref_hi = np.nanquantile(null_profiles, 0.975, axis=0)
            for angle_idx, (angle, value) in enumerate(zip(angles_deg, observed, strict=True)):
                base = {
                    "coherence_bin": band,
                    "bin_low": float(meta.bin_low),
                    "bin_high": float(meta.bin_high),
                    "bin_center": float(meta.bin_center),
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "relative_angle_deg": float(angle),
                    "n_windows": int(matrix.shape[0]),
                }
                profile_rows.append({**base, value_col: float(value)})
                reference_rows.append(
                    {
                        **base,
                        f"observed_{value_col}": float(value),
                        f"random_orientation_median_{value_col}": float(ref_median[angle_idx]),
                        f"random_orientation_ci95_low_{value_col}": float(ref_lo[angle_idx]),
                        f"random_orientation_ci95_high_{value_col}": float(ref_hi[angle_idx]),
                        f"observed_minus_random_median_{value_col}": float(value - ref_median[angle_idx])
                        if np.isfinite(value) and np.isfinite(ref_median[angle_idx])
                        else float("nan"),
                    }
                )
    return pd.DataFrame(profile_rows), pd.DataFrame(reference_rows)


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


def make_progression_summary(
    window_components: pd.DataFrame,
    bin_meta: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metric_cols = [
        ("signed_net", "parallel_signed_net_component_arcmin_equiv", "orthogonal_signed_net_component_arcmin_equiv"),
        (
            "net_magnitude",
            "parallel_net_component_magnitude_arcmin_equiv",
            "orthogonal_net_component_magnitude_arcmin_equiv",
        ),
        (
            "unsigned_path",
            "parallel_unsigned_component_path_arcmin_equiv",
            "orthogonal_unsigned_component_path_arcmin_equiv",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for meta in bin_meta.itertuples(index=False):
        sub = window_components[window_components["coherence_bin"].astype(str).eq(str(meta.coherence_bin))].copy()
        if sub.empty:
            continue
        for metric, parallel_col, orthogonal_col in metric_cols:
            sub[f"{metric}_parallel_minus_orthogonal"] = sub[parallel_col] - sub[orthogonal_col]
            if metric != "signed_net":
                sub[f"{metric}_parallel_over_orthogonal"] = sub[parallel_col] / sub[orthogonal_col]
            for component, col in [
                ("parallel", parallel_col),
                ("orthogonal", orthogonal_col),
                ("parallel_minus_orthogonal", f"{metric}_parallel_minus_orthogonal"),
            ]:
                session_values = sub.groupby("session", observed=True)[col].median(numeric_only=True).to_numpy(dtype=float)
                point, lo, hi = _bootstrap_median_ci(session_values, rng=rng, n_bootstrap=n_bootstrap)
                rows.append(
                    {
                        "coherence_bin": str(meta.coherence_bin),
                        "bin_low": float(meta.bin_low),
                        "bin_high": float(meta.bin_high),
                        "bin_center": float(meta.bin_center),
                        "metric": metric,
                        "component": component,
                        "session_median": point,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "n_sessions": int(sub["session"].nunique()),
                        "n_windows": int(sub.shape[0]),
                        "window_median": float(np.nanmedian(sub[col].to_numpy(dtype=float))),
                    }
                )
            if metric != "signed_net":
                col = f"{metric}_parallel_over_orthogonal"
                session_values = sub.groupby("session", observed=True)[col].median(numeric_only=True).to_numpy(dtype=float)
                point, lo, hi = _bootstrap_median_ci(session_values, rng=rng, n_bootstrap=n_bootstrap)
                rows.append(
                    {
                        "coherence_bin": str(meta.coherence_bin),
                        "bin_low": float(meta.bin_low),
                        "bin_high": float(meta.bin_high),
                        "bin_center": float(meta.bin_center),
                        "metric": metric,
                        "component": "parallel_over_orthogonal",
                        "session_median": point,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "n_sessions": int(sub["session"].nunique()),
                        "n_windows": int(sub.shape[0]),
                        "window_median": float(np.nanmedian(sub[col].to_numpy(dtype=float))),
                    }
                )
    return pd.DataFrame(rows)


def _value_col(metric_key: str) -> str:
    for key, _label, value_col, _ylabel, _color in METRIC_SPECS:
        if key == metric_key:
            return value_col
    raise KeyError(metric_key)


def _profile_colors(labels: list[str]) -> dict[str, Any]:
    cmap = plt.get_cmap("viridis")
    if len(labels) == 1:
        return {labels[0]: cmap(0.75)}
    return {label: cmap(0.12 + 0.78 * i / (len(labels) - 1)) for i, label in enumerate(labels)}


def _set_unwrapped_axis(ax: plt.Axes) -> None:
    ax.axvline(90.0, color="#7d858c", lw=0.8, ls=":")
    ax.set_xlim(0.0, 180.0)
    ax.set_xticks([0.0, 90.0, 180.0])
    ax.set_xticklabels(["parallel", "orthogonal", "parallel"])
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    _clean_axis(ax)


def plot_unwrapped_metric(
    profiles: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    metric_key: str,
    out_path: Path,
) -> None:
    spec = next(item for item in METRIC_SPECS if item[0] == metric_key)
    _metric_key, metric_label, value_col, ylabel, _color = spec
    labels = sorted(profiles["coherence_bin"].astype(str).unique(), key=lambda x: float(x.split("-", maxsplit=1)[0]))
    colors = _profile_colors(labels)
    fig, axes = plt.subplots(2, 5, figsize=(11.4, 5.5), sharex=True, constrained_layout=True)
    value_cols = [
        f"observed_{value_col}",
        f"random_orientation_ci95_low_{value_col}",
        f"random_orientation_ci95_high_{value_col}",
    ]
    finite = reference.loc[reference["metric"].astype(str).eq(metric_key), value_cols].to_numpy(dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    if metric_key == "signed_net":
        yabs = max(0.05, float(np.nanmax(np.abs(finite))) * 1.12)
        ylim = (-yabs, yabs)
    else:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
        pad = max(0.05, 0.08 * (hi - lo))
        ylim = (lo - pad, hi + pad)
    for ax, label in zip(axes.flat, labels, strict=True):
        sub = reference[
            reference["coherence_bin"].astype(str).eq(label)
            & reference["metric"].astype(str).eq(metric_key)
        ].sort_values("relative_angle_deg")
        x = sub["relative_angle_deg"].to_numpy(dtype=float)
        ax.fill_between(
            x,
            sub[f"random_orientation_ci95_low_{value_col}"].to_numpy(dtype=float),
            sub[f"random_orientation_ci95_high_{value_col}"].to_numpy(dtype=float),
            color=RANDOM,
            alpha=0.10,
            lw=0,
        )
        ax.plot(
            x,
            sub[f"random_orientation_median_{value_col}"],
            color=RANDOM,
            linestyle="--",
            linewidth=1.1,
            label="random axis",
        )
        ax.plot(x, sub[f"observed_{value_col}"], color=colors[label], linewidth=1.8, label="observed")
        if metric_key == "signed_net":
            ax.axhline(0.0, color="#333333", lw=0.9, alpha=0.55)
        ax.set_ylim(*ylim)
        ax.set_title(f"coh {label}", fontsize=9)
        _set_unwrapped_axis(ax)
    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(ylabel)
    for ax in axes[-1, :]:
        ax.set_xlabel("axis from local edge")
    axes[0, -1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(f"{metric_label} relative to local edge", fontsize=15)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_delta_vs_random_metric(
    reference: pd.DataFrame,
    *,
    metric_key: str,
    out_path: Path,
) -> None:
    spec = next(item for item in METRIC_SPECS if item[0] == metric_key)
    _metric_key, metric_label, value_col, ylabel, _color = spec
    labels = sorted(reference["coherence_bin"].astype(str).unique(), key=lambda x: float(x.split("-", maxsplit=1)[0]))
    colors = _profile_colors(labels)
    delta_col = f"observed_minus_random_median_{value_col}"
    finite = reference.loc[reference["metric"].astype(str).eq(metric_key), delta_col].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    yabs = max(0.05, float(np.nanmax(np.abs(finite))) * 1.12)
    fig, axes = plt.subplots(2, 5, figsize=(11.4, 5.5), sharex=True, sharey=True, constrained_layout=True)
    for ax, label in zip(axes.flat, labels, strict=True):
        sub = reference[
            reference["coherence_bin"].astype(str).eq(label)
            & reference["metric"].astype(str).eq(metric_key)
        ].sort_values("relative_angle_deg")
        ax.plot(sub["relative_angle_deg"], sub[delta_col], color=colors[label], linewidth=1.8)
        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_ylim(-yabs, yabs)
        ax.set_title(f"coh {label}", fontsize=9)
        _set_unwrapped_axis(ax)
    axes[0, 0].set_ylabel(f"observed - random\n{ylabel}")
    axes[1, 0].set_ylabel(f"observed - random\n{ylabel}")
    for ax in axes[-1, :]:
        ax.set_xlabel("axis from local edge")
    fig.suptitle(f"{metric_label}: deviation from randomized-orientation baseline", fontsize=15)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_unwrapped_stack(reference: pd.DataFrame, out_path: Path) -> None:
    labels = sorted(reference["coherence_bin"].astype(str).unique(), key=lambda x: float(x.split("-", maxsplit=1)[0]))
    colors = _profile_colors(labels)
    fig, axes = plt.subplots(3, len(labels), figsize=(22.0, 7.8), sharex=True, constrained_layout=True)
    for row_idx, (metric_key, metric_label, value_col, ylabel, _color) in enumerate(METRIC_SPECS):
        finite = reference.loc[reference["metric"].astype(str).eq(metric_key), [
            f"observed_{value_col}",
            f"random_orientation_ci95_low_{value_col}",
            f"random_orientation_ci95_high_{value_col}",
        ]].to_numpy(dtype=float).ravel()
        finite = finite[np.isfinite(finite)]
        if metric_key == "signed_net":
            yabs = max(0.05, float(np.nanmax(np.abs(finite))) * 1.12)
            ylim = (-yabs, yabs)
        else:
            lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
            pad = max(0.05, 0.08 * (hi - lo))
            ylim = (lo - pad, hi + pad)
        for col_idx, label in enumerate(labels):
            ax = axes[row_idx, col_idx]
            sub = reference[
                reference["coherence_bin"].astype(str).eq(label)
                & reference["metric"].astype(str).eq(metric_key)
            ].sort_values("relative_angle_deg")
            x = sub["relative_angle_deg"].to_numpy(dtype=float)
            ax.fill_between(
                x,
                sub[f"random_orientation_ci95_low_{value_col}"].to_numpy(dtype=float),
                sub[f"random_orientation_ci95_high_{value_col}"].to_numpy(dtype=float),
                color=RANDOM,
                alpha=0.10,
                lw=0,
            )
            ax.plot(x, sub[f"random_orientation_median_{value_col}"], color=RANDOM, linestyle="--", linewidth=0.9)
            ax.plot(x, sub[f"observed_{value_col}"], color=colors[label], linewidth=1.5)
            if metric_key == "signed_net":
                ax.axhline(0.0, color="#333333", lw=0.8, alpha=0.55)
            ax.set_ylim(*ylim)
            if row_idx == 0:
                ax.set_title(f"coh {label}", fontsize=8.5)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == len(METRIC_SPECS) - 1:
                ax.set_xlabel("axis from edge")
            _set_unwrapped_axis(ax)
    fig.suptitle("Signed/net and unsigned path components relative to local edge", fontsize=15)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _progression_error(ax: plt.Axes, summary: pd.DataFrame, metric: str, component: str, *, color: str, label: str) -> None:
    sub = summary[
        summary["metric"].astype(str).eq(metric)
        & summary["component"].astype(str).eq(component)
    ].sort_values("bin_center")
    x = sub["bin_center"].to_numpy(dtype=float)
    y = sub["session_median"].to_numpy(dtype=float)
    lo = sub["ci95_low"].to_numpy(dtype=float)
    hi = sub["ci95_high"].to_numpy(dtype=float)
    yerr = np.vstack([y - lo, hi - y])
    yerr[~np.isfinite(yerr)] = 0.0
    ax.errorbar(x, y, yerr=yerr, marker="o", color=color, linewidth=2.0, markersize=5.5, label=label)


def plot_progression(summary: pd.DataFrame, out_path: Path) -> None:
    counts = (
        summary[summary["metric"].astype(str).eq("signed_net")]
        .sort_values("bin_center")[["bin_center", "n_windows"]]
        .drop_duplicates()
    )
    x = counts["bin_center"].to_numpy(dtype=float)
    n = counts["n_windows"].to_numpy(dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(7.3, 9.7), sharex=True, constrained_layout=True)
    for ax, (metric_key, _metric_label, _value_col, ylabel, _color) in zip(axes, METRIC_SPECS, strict=True):
        ax2 = ax.twinx()
        ax2.bar(x, n, width=0.075, color=BAR, alpha=0.75, zorder=0)
        ax2.set_ylabel("window count", color="#6b7785")
        ax2.tick_params(axis="y", colors="#6b7785")
        ax2.set_ylim(0, max(n) * 1.35)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        _progression_error(ax, summary, metric_key, "parallel", color=BLUE, label="parallel")
        _progression_error(ax, summary, metric_key, "orthogonal", color=GRAY, label="orthogonal")
        if metric_key == "signed_net":
            ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, linewidth=1.0)
        ax.legend(frameon=False, loc="best")
        _clean_axis(ax)
    axes[-1].set_xlabel("local edge coherence")
    axes[-1].set_xlim(0.0, 1.0)
    fig.suptitle("Path components by edge coherence", fontsize=18)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_unsigned_path_rms_comparison(
    rms_progression: pd.DataFrame,
    path_progression: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specs = [
        ("parallel", "parallel", "parallel_rms_arcmin"),
        ("orthogonal", "orthogonal", "orthogonal_rms_arcmin"),
    ]
    path = path_progression[
        path_progression["metric"].astype(str).eq("unsigned_path")
        & path_progression["component"].astype(str).isin(["parallel", "orthogonal"])
    ].copy()
    for component, label, rms_metric in specs:
        rms_sub = rms_progression[rms_progression["metric"].astype(str).eq(rms_metric)].copy()
        path_sub = path[path["component"].astype(str).eq(component)].copy()
        merged = rms_sub.merge(
            path_sub,
            on=["coherence_bin", "bin_low", "bin_high", "bin_center"],
            suffixes=("_rms", "_path"),
            validate="one_to_one",
        )
        for row in merged.itertuples(index=False):
            rows.append(
                {
                    "component": component,
                    "component_label": label,
                    "coherence_bin": str(row.coherence_bin),
                    "bin_low": float(row.bin_low),
                    "bin_high": float(row.bin_high),
                    "bin_center": float(row.bin_center),
                    "rms_session_median_arcmin": float(row.session_median_rms),
                    "rms_ci95_low_arcmin": float(row.ci95_low_rms),
                    "rms_ci95_high_arcmin": float(row.ci95_high_rms),
                    "rms_window_median_arcmin": float(row.window_median_rms),
                    "unsigned_path_session_median_arcmin_equiv": float(row.session_median_path),
                    "unsigned_path_ci95_low_arcmin_equiv": float(row.ci95_low_path),
                    "unsigned_path_ci95_high_arcmin_equiv": float(row.ci95_high_path),
                    "unsigned_path_window_median_arcmin_equiv": float(row.window_median_path),
                    "n_sessions": int(row.n_sessions_rms),
                    "n_windows": int(row.n_windows_rms),
                }
            )
    return pd.DataFrame(rows)


def _error_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    color: str,
) -> None:
    yerr = np.vstack([y - lo, hi - y])
    yerr[~np.isfinite(yerr)] = 0.0
    ax.errorbar(x, y, yerr=yerr, marker="o", color=color, linewidth=2.0, markersize=5.5, capsize=2.5)


def _median_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray, *, color: str) -> None:
    ax.plot(x, y, marker="o", color=color, linewidth=2.2, markersize=5.8)


def plot_unsigned_path_vs_rms_comparison(comparison: pd.DataFrame, out_path: Path) -> None:
    components = [("parallel", "parallel", BLUE), ("orthogonal", "orthogonal", GRAY)]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.8), constrained_layout=True)
    rms_finite = comparison["rms_session_median_arcmin"].to_numpy(dtype=float)
    path_finite = comparison["unsigned_path_session_median_arcmin_equiv"].to_numpy(dtype=float)
    rms_finite = rms_finite[np.isfinite(rms_finite)]
    path_finite = path_finite[np.isfinite(path_finite)]
    rms_pad = max(0.04, 0.08 * (float(np.nanmax(rms_finite)) - float(np.nanmin(rms_finite))))
    path_pad = max(0.5, 0.08 * (float(np.nanmax(path_finite)) - float(np.nanmin(path_finite))))
    rms_ylim = (float(np.nanmin(rms_finite)) - rms_pad, float(np.nanmax(rms_finite)) + rms_pad)
    path_ylim = (float(np.nanmin(path_finite)) - path_pad, float(np.nanmax(path_finite)) + path_pad)
    coherence_norm = plt.Normalize(0.0, 1.0)
    coherence_cmap = plt.get_cmap("viridis")

    for row_idx, (component, label, color) in enumerate(components):
        sub = comparison[comparison["component"].astype(str).eq(component)].sort_values("bin_center")
        x = sub["bin_center"].to_numpy(dtype=float)
        rms = sub["rms_session_median_arcmin"].to_numpy(dtype=float)
        path = sub["unsigned_path_session_median_arcmin_equiv"].to_numpy(dtype=float)

        ax = axes[row_idx, 0]
        _median_line(ax, x, rms, color=color)
        ax.set_ylim(*rms_ylim)
        ax.set_ylabel(f"{label}\nRMS spread (arcmin)")
        ax.grid(axis="y", color=GRID, linewidth=0.9)
        _clean_axis(ax)

        ax = axes[row_idx, 1]
        _median_line(ax, x, path, color=color)
        ax.set_ylim(*path_ylim)
        ax.set_ylabel(f"{label}\nunsigned path (arcmin equiv.)")
        ax.grid(axis="y", color=GRID, linewidth=0.9)
        _clean_axis(ax)

        ax = axes[row_idx, 2]
        ax.plot(rms, path, color="#6d7680", linewidth=1.0, zorder=1)
        scatter = ax.scatter(
            rms,
            path,
            c=x,
            cmap=coherence_cmap,
            norm=coherence_norm,
            s=42,
            edgecolor="white",
            linewidth=0.7,
            zorder=2,
        )
        ax.set_xlim(*rms_ylim)
        ax.set_ylim(*path_ylim)
        ax.set_xlabel("RMS spread (arcmin)")
        ax.set_ylabel(f"{label}\nunsigned path (arcmin equiv.)")
        ax.grid(color=GRID, linewidth=0.9)
        _clean_axis(ax)

    for ax in axes[-1, :2]:
        ax.set_xlabel("local edge coherence")
        ax.set_xlim(0.0, 1.0)
    axes[0, 0].set_title("position spread RMS", fontsize=10)
    axes[0, 1].set_title("sum(abs(projected steps))", fontsize=10)
    axes[0, 2].set_title("unsigned path vs RMS", fontsize=10)
    fig.suptitle("Component path length compared with position spread", fontsize=15)
    fig.colorbar(scatter, ax=axes[:, 2], label="local edge coherence", shrink=0.82)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_unsigned_path_rms_comparison_outputs(
    rms_progression: pd.DataFrame,
    path_progression: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Path]:
    comparison = make_unsigned_path_rms_comparison(rms_progression, path_progression)
    paths: dict[str, Path] = {}
    paths["unsigned_path_vs_rms_csv"] = out_dir / "b_unsigned_path_vs_position_spread_rms_by_edge_coherence.csv"
    comparison.to_csv(paths["unsigned_path_vs_rms_csv"], index=False)
    stem = out_dir / "b_unsigned_path_vs_position_spread_rms_by_edge_coherence"
    plot_unsigned_path_vs_rms_comparison(comparison, stem)
    paths["unsigned_path_vs_rms_png"] = stem.with_suffix(".png")
    paths["unsigned_path_vs_rms_pdf"] = stem.with_suffix(".pdf")
    return paths


def write_outputs(
    window_components: pd.DataFrame,
    profiles: pd.DataFrame,
    reference: pd.DataFrame,
    progression: pd.DataFrame,
    out_dir: Path,
    *,
    rms_progression: pd.DataFrame | None,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()
    paths: dict[str, Path] = {}
    paths["window_components_csv"] = out_dir / "b_signed_path_component_windows_by_edge_coherence.csv"
    window_components.to_csv(paths["window_components_csv"], index=False)
    paths["profiles_csv"] = out_dir / "b_signed_path_component_unwrapped_profiles_by_edge_coherence.csv"
    profiles.to_csv(paths["profiles_csv"], index=False)
    paths["random_reference_csv"] = out_dir / "b_signed_path_component_random_orientation_baseline_by_edge_coherence.csv"
    reference.to_csv(paths["random_reference_csv"], index=False)
    paths["progression_csv"] = out_dir / "b_signed_path_component_progression_by_edge_coherence.csv"
    progression.to_csv(paths["progression_csv"], index=False)

    stack_stem = out_dir / "b_signed_path_components_unwrapped_stack_by_edge_coherence"
    plot_unwrapped_stack(reference, stack_stem)
    paths["stack_png"] = stack_stem.with_suffix(".png")
    paths["stack_pdf"] = stack_stem.with_suffix(".pdf")

    for metric_key, _metric_label, _value_col, _ylabel, _color in METRIC_SPECS:
        stem = out_dir / f"b_{metric_key}_component_unwrapped_small_multiples_by_edge_coherence"
        plot_unwrapped_metric(profiles, reference, metric_key=metric_key, out_path=stem)
        paths[f"{metric_key}_unwrapped_png"] = stem.with_suffix(".png")
        paths[f"{metric_key}_unwrapped_pdf"] = stem.with_suffix(".pdf")

        delta_stem = out_dir / f"b_{metric_key}_component_unwrapped_delta_vs_random_orientation_by_edge_coherence"
        plot_delta_vs_random_metric(reference, metric_key=metric_key, out_path=delta_stem)
        paths[f"{metric_key}_delta_vs_random_png"] = delta_stem.with_suffix(".png")
        paths[f"{metric_key}_delta_vs_random_pdf"] = delta_stem.with_suffix(".pdf")

    progression_stem = out_dir / "b_signed_path_component_progression_by_edge_coherence"
    plot_progression(progression, progression_stem)
    paths["progression_png"] = progression_stem.with_suffix(".png")
    paths["progression_pdf"] = progression_stem.with_suffix(".pdf")

    if rms_progression is not None:
        paths.update(write_unsigned_path_rms_comparison_outputs(rms_progression, progression, out_dir))
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-windows", type=Path, default=DEFAULT_INPUT_WINDOWS)
    parser.add_argument("--rms-progression", type=Path, default=DEFAULT_RMS_PROGRESSION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--coherence-bin-width", type=float, default=0.1)
    parser.add_argument("--angle-step-deg", type=float, default=3.75)
    parser.add_argument(
        "--equivalent-window-s",
        type=float,
        default=0.325,
        help="Scale path metrics by this/window_duration. Set <=0 to keep raw full-window arcmin.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-random-orientation", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="Only rebuild the unsigned-path-vs-RMS comparison from existing progression CSVs.",
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    rms_progression_path = Path(args.rms_progression)
    rms_progression = pd.read_csv(rms_progression_path) if rms_progression_path.exists() else None
    if args.comparison_only:
        path_progression = pd.read_csv(out_dir / "b_signed_path_component_progression_by_edge_coherence.csv")
        if rms_progression is None:
            raise FileNotFoundError(rms_progression_path)
        paths = write_unsigned_path_rms_comparison_outputs(rms_progression, path_progression, out_dir)
        print(f"Wrote unsigned-path-vs-RMS comparison to {paths['unsigned_path_vs_rms_png']}")
        return out_dir

    df, bin_meta = load_windows(Path(args.input_windows), coherence_bin_width=float(args.coherence_bin_width))
    window_components, angles_deg, matrices = compute_window_profiles(
        df,
        angle_step_deg=float(args.angle_step_deg),
        equivalent_window_s=float(args.equivalent_window_s),
    )
    profiles, reference = summarize_profiles(
        matrices,
        bin_meta,
        angles_deg,
        n_random_orientation=int(args.n_random_orientation),
        seed=int(args.seed) + 200_000,
    )
    progression = make_progression_summary(
        window_components,
        bin_meta,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed) + 300_000,
    )
    paths = write_outputs(
        window_components,
        profiles,
        reference,
        progression,
        out_dir,
        rms_progression=rms_progression,
    )
    cfg = SignedPathConfig(
        input_windows=str(Path(args.input_windows)),
        rms_progression=str(rms_progression_path),
        out_dir=str(out_dir),
        coherence_bin_width=float(args.coherence_bin_width),
        angle_step_deg=float(args.angle_step_deg),
        equivalent_window_s=float(args.equivalent_window_s),
        n_bootstrap=int(args.n_bootstrap),
        n_random_orientation=int(args.n_random_orientation),
        seed=int(args.seed),
    )
    write_json(
        out_dir / "backimage_contour_signed_path_followups_metadata.json",
        {
            "config": asdict(cfg),
            "n_windows": int(df.shape[0]),
            "n_sessions": int(df["session"].nunique()),
            "n_coherence_bins": int(bin_meta.shape[0]),
            "outputs": {key: str(value) for key, value in sorted(paths.items())},
            "metric_note": (
                "signed_net is sum(projected sample-to-sample eye displacement). Because the contour axis is axial, "
                "the sign is a cancellation/bias diagnostic rather than a stable contour direction. "
                "net_magnitude is abs(signed_net). unsigned_path is sum(abs(projected step))."
            ),
            "scaling_note": (
                f"Path metrics are scaled by {float(args.equivalent_window_s):g} s / window_duration_s when "
                "--equivalent-window-s is positive; set it <=0 for raw full-window path."
            ),
        },
    )
    print(f"Wrote BackImage contour signed-path follow-up plots to {out_dir}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
