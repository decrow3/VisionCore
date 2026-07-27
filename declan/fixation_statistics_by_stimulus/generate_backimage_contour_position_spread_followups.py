#!/usr/bin/env python3
"""Regenerate BackImage contour-relative position-spread follow-up plots.

This script promotes the exploratory ``b_position_spread_*`` diagnostics into
reproducible code. It starts from the saved
``contour_motion_component_windows.csv`` table produced by
``plot_backimage_contour_motion_components.py`` and rebuilds:

* coherence-bin progression plots for contour-parallel/orthogonal RMS;
* polar and unwrapped RMS profiles versus angle from the local contour;
* the randomized-orientation baseline used for the panel-H style comparison.

The randomized-orientation baseline keeps each observed eye-position covariance
window intact and replaces only the local contour axis with a random axial
orientation. That preserves the animal's actual FEM spread while breaking the
relationship between FEM geometry and the measured local contour direction.
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

from .io_utils import write_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_contour_motion_component_plots_v1"
)
DEFAULT_INPUT_WINDOWS = DEFAULT_OUT_DIR / "contour_motion_component_windows.csv"

BLUE = "#3366aa"
GRAY = "#7f8b96"
GREEN = "#1b7f5c"
GRID = "#d5dce3"
BAR = "#d7dde4"
INK = "#20242a"
RANDOM = "#33383f"


@dataclass(frozen=True)
class FollowupConfig:
    input_windows: str
    out_dir: str
    coherence_bin_width: float
    angle_step_deg: float
    n_bootstrap: int
    n_random_orientation: int
    seed: int


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
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
    lows = []
    highs = []
    centers = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        labels.append(_format_bin_label(float(lo), float(hi)))
        lows.append(float(lo))
        highs.append(float(hi))
        centers.append(float(0.5 * (lo + hi)))
    cats = pd.cut(
        coherence,
        edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    # pd.cut with right=False drops exactly 1.0; put those in the final bin.
    final_label = labels[-1]
    cats = cats.astype(object)
    cats[np.isclose(coherence.to_numpy(dtype=float), 1.0)] = final_label

    meta = pd.DataFrame(
        {
            "coherence_bin": labels,
            "bin_low": lows,
            "bin_high": highs,
            "bin_center": centers,
        }
    )
    out = df.copy()
    out["coherence_bin"] = pd.Categorical(cats, categories=labels, ordered=True)
    out = out[out["coherence_bin"].notna()].copy()
    return out, meta


def load_windows(path: Path, *, coherence_bin_width: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = [
        "session",
        "image_orientation_coherence",
        "image_edge_axis_deg",
        "cov_xx_deg2",
        "cov_xy_deg2",
        "cov_yy_deg2",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    if "rms_along_arcmin" not in df.columns:
        df["rms_along_arcmin"] = _directional_rms_values(df, 0.0)
    if "rms_across_arcmin" not in df.columns:
        df["rms_across_arcmin"] = _directional_rms_values(df, 0.5 * np.pi)
    df["rms_delta_along_minus_across_arcmin"] = df["rms_along_arcmin"] - df["rms_across_arcmin"]
    df["rms_ratio_along_over_across"] = df["rms_along_arcmin"] / df["rms_across_arcmin"]

    numeric_cols = [
        "image_orientation_coherence",
        "image_edge_axis_deg",
        "cov_xx_deg2",
        "cov_xy_deg2",
        "cov_yy_deg2",
        "rms_along_arcmin",
        "rms_across_arcmin",
        "rms_delta_along_minus_across_arcmin",
        "rms_ratio_along_over_across",
    ]
    ok = df["session"].notna()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        ok &= np.isfinite(df[col])
    df = df[ok].copy()
    df, bin_meta = _add_coherence_bins(df, width=coherence_bin_width)
    return df, bin_meta


def _directional_rms_values(
    df: pd.DataFrame,
    rel_angle_rad: float,
    *,
    base_axis_deg: np.ndarray | None = None,
) -> np.ndarray:
    if base_axis_deg is None:
        base = pd.to_numeric(df["image_edge_axis_deg"], errors="coerce").to_numpy(dtype=np.float64)
        theta = np.radians(base) + float(rel_angle_rad)
    else:
        theta = np.radians(np.asarray(base_axis_deg, dtype=np.float64)) + float(rel_angle_rad)
    cxx = pd.to_numeric(df["cov_xx_deg2"], errors="coerce").to_numpy(dtype=np.float64)
    cxy = pd.to_numeric(df["cov_xy_deg2"], errors="coerce").to_numpy(dtype=np.float64)
    cyy = pd.to_numeric(df["cov_yy_deg2"], errors="coerce").to_numpy(dtype=np.float64)
    ux = np.cos(theta)
    uy = np.sin(theta)
    var = ux * ux * cxx + 2.0 * ux * uy * cxy + uy * uy * cyy
    return 60.0 * np.sqrt(np.maximum(var, 0.0))


def _directional_rms_values_for_angles(
    df: pd.DataFrame,
    rel_angles_rad: np.ndarray,
    *,
    base_axis_rad: np.ndarray | None = None,
) -> np.ndarray:
    if base_axis_rad is None:
        base = np.radians(pd.to_numeric(df["image_edge_axis_deg"], errors="coerce").to_numpy(dtype=np.float64))
    else:
        base = np.asarray(base_axis_rad, dtype=np.float64)
    cxx = pd.to_numeric(df["cov_xx_deg2"], errors="coerce").to_numpy(dtype=np.float64)[:, None]
    cxy = pd.to_numeric(df["cov_xy_deg2"], errors="coerce").to_numpy(dtype=np.float64)[:, None]
    cyy = pd.to_numeric(df["cov_yy_deg2"], errors="coerce").to_numpy(dtype=np.float64)[:, None]
    theta = base[:, None] + np.asarray(rel_angles_rad, dtype=np.float64)[None, :]
    ux = np.cos(theta)
    uy = np.sin(theta)
    var = ux * ux * cxx + 2.0 * ux * uy * cxy + uy * uy * cyy
    return 60.0 * np.sqrt(np.maximum(var, 0.0))


def _bootstrap_median_ci(values: np.ndarray, *, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.median(vals))
    if vals.size == 1 or n_bootstrap <= 0:
        return point, float("nan"), float("nan")
    draws = rng.integers(0, vals.size, size=(int(n_bootstrap), vals.size))
    boots = np.median(vals[draws], axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


def make_progression_summary(
    df: pd.DataFrame,
    bin_meta: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics = [
        ("rms_along_arcmin", "parallel_rms_arcmin"),
        ("rms_across_arcmin", "orthogonal_rms_arcmin"),
        ("rms_delta_along_minus_across_arcmin", "parallel_minus_orthogonal_rms_arcmin"),
        ("rms_ratio_along_over_across", "parallel_over_orthogonal_rms"),
    ]
    rows: list[dict[str, Any]] = []
    for meta in bin_meta.itertuples(index=False):
        sub = df[df["coherence_bin"].astype(str) == str(meta.coherence_bin)]
        if sub.empty:
            continue
        for source_col, metric_name in metrics:
            session_values = sub.groupby("session")[source_col].median(numeric_only=True).to_numpy(dtype=np.float64)
            point, lo, hi = _bootstrap_median_ci(session_values, n_bootstrap=n_bootstrap, rng=rng)
            rows.append(
                {
                    "coherence_bin": str(meta.coherence_bin),
                    "bin_low": float(meta.bin_low),
                    "bin_high": float(meta.bin_high),
                    "bin_center": float(meta.bin_center),
                    "metric": metric_name,
                    "session_median": point,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_sessions": int(sub["session"].nunique()),
                    "n_windows": int(sub.shape[0]),
                    "window_median": float(np.nanmedian(sub[source_col].to_numpy(dtype=np.float64))),
                }
            )
    return pd.DataFrame(rows)


def make_directional_profiles(
    df: pd.DataFrame,
    bin_meta: pd.DataFrame,
    angles_deg: np.ndarray,
) -> pd.DataFrame:
    angles_rad = np.radians(np.asarray(angles_deg, dtype=np.float64))
    rows: list[dict[str, Any]] = []
    for meta in bin_meta.itertuples(index=False):
        sub = df[df["coherence_bin"].astype(str) == str(meta.coherence_bin)]
        if sub.empty:
            continue
        values = _directional_rms_values_for_angles(sub, angles_rad)
        profile = np.nanmedian(values, axis=0)
        for angle, rms in zip(angles_deg, profile, strict=True):
            rows.append(
                {
                    "coherence_bin": str(meta.coherence_bin),
                    "relative_angle_deg": float(angle),
                    "rms_arcmin": float(rms),
                    "n_windows": int(sub.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def make_random_orientation_baseline(
    df: pd.DataFrame,
    bin_meta: pd.DataFrame,
    observed_unwrapped: pd.DataFrame,
    angles_deg: np.ndarray,
    *,
    n_random_orientation: int,
    seed: int,
) -> pd.DataFrame:
    angles_rad = np.radians(np.asarray(angles_deg, dtype=np.float64))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    observed_lookup = {
        (str(row.coherence_bin), float(row.relative_angle_deg)): float(row.rms_arcmin)
        for row in observed_unwrapped.itertuples(index=False)
    }
    for meta in bin_meta.itertuples(index=False):
        sub = df[df["coherence_bin"].astype(str) == str(meta.coherence_bin)]
        if sub.empty:
            continue
        null_profiles = np.empty((int(n_random_orientation), angles_rad.size), dtype=np.float64)
        for i in range(int(n_random_orientation)):
            random_axis = rng.uniform(0.0, np.pi, size=sub.shape[0])
            values = _directional_rms_values_for_angles(sub, angles_rad, base_axis_rad=random_axis)
            null_profiles[i] = np.nanmedian(values, axis=0)
        median = np.nanmedian(null_profiles, axis=0)
        lo = np.nanquantile(null_profiles, 0.025, axis=0)
        hi = np.nanquantile(null_profiles, 0.975, axis=0)
        for j, angle in enumerate(angles_deg):
            observed = observed_lookup.get((str(meta.coherence_bin), float(angle)), float("nan"))
            rows.append(
                {
                    "coherence_bin": str(meta.coherence_bin),
                    "relative_angle_deg": float(angle),
                    "n_windows": int(sub.shape[0]),
                    "observed_rms_arcmin": observed,
                    "random_orientation_median_rms_arcmin": float(median[j]),
                    "random_orientation_ci95_low_arcmin": float(lo[j]),
                    "random_orientation_ci95_high_arcmin": float(hi[j]),
                    "observed_minus_random_median_arcmin": float(observed - median[j])
                    if np.isfinite(observed) and np.isfinite(median[j])
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _line_error(ax: plt.Axes, summary: pd.DataFrame, metric: str, *, color: str, label: str) -> None:
    sub = summary[summary["metric"].astype(str) == metric].sort_values("bin_center")
    x = sub["bin_center"].to_numpy(dtype=np.float64)
    y = sub["session_median"].to_numpy(dtype=np.float64)
    lo = sub["ci95_low"].to_numpy(dtype=np.float64)
    hi = sub["ci95_high"].to_numpy(dtype=np.float64)
    yerr = np.vstack([y - lo, hi - y])
    yerr[~np.isfinite(yerr)] = 0.0
    ax.errorbar(x, y, yerr=yerr, color=color, marker="o", linewidth=2.2, markersize=6.5, label=label, zorder=3)


def plot_progression(summary: pd.DataFrame, out_path: Path) -> None:
    counts = (
        summary[summary["metric"].astype(str) == "parallel_rms_arcmin"]
        .sort_values("bin_center")[["bin_center", "n_windows"]]
        .drop_duplicates()
    )
    x = counts["bin_center"].to_numpy(dtype=np.float64)
    n = counts["n_windows"].to_numpy(dtype=np.float64)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.6), sharex=True, constrained_layout=True)
    fig.suptitle("Contour-relative position spread vs local edge coherence", fontsize=18)
    specs = [
        ("position spread\nRMS (arcmin)", None),
        ("spread allocation\nRMS diff (arcmin)", 0.0),
        ("spread allocation\nRMS ratio", 1.0),
    ]
    for ax, (ylabel, baseline) in zip(axes, specs, strict=True):
        ax2 = ax.twinx()
        ax2.bar(x, n, width=0.075, color=BAR, alpha=0.75, zorder=0)
        ax2.set_ylabel("window count", color="#6b7785")
        ax2.tick_params(axis="y", colors="#6b7785")
        ax2.set_ylim(0, max(n) * 1.35)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=1.1)
        ax.set_ylabel(ylabel)
        if baseline is not None:
            ax.axhline(baseline, color="#333333", linestyle="--", linewidth=1.2)
        _clean_axis(ax)
    _line_error(axes[0], summary, "parallel_rms_arcmin", color=BLUE, label="parallel spread")
    _line_error(axes[0], summary, "orthogonal_rms_arcmin", color=GRAY, label="orthogonal spread")
    _line_error(axes[1], summary, "parallel_minus_orthogonal_rms_arcmin", color=BLUE, label="parallel - orthogonal")
    _line_error(axes[2], summary, "parallel_over_orthogonal_rms", color=GREEN, label="parallel / orthogonal")
    for ax in axes:
        ax.legend(frameon=False, loc="upper left")
    axes[-1].set_xlabel("local edge coherence")
    axes[-1].set_xlim(0.0, 1.0)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _profile_color_map(labels: list[str]) -> dict[str, Any]:
    cmap = plt.get_cmap("viridis")
    if len(labels) == 1:
        return {labels[0]: cmap(0.8)}
    return {label: cmap(0.12 + 0.78 * i / (len(labels) - 1)) for i, label in enumerate(labels)}


def _setup_contour_relative_polar(ax: plt.Axes) -> None:
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetagrids([0, 90, 180, 270], ["parallel", "orthogonal", "parallel", "orthogonal"])
    ax.grid(color=GRID, linewidth=0.8)


def _wrap_polar_profile(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sub = sub.sort_values("relative_angle_deg")
    theta = np.radians(sub["relative_angle_deg"].to_numpy(dtype=np.float64))
    radius = sub["rms_arcmin"].to_numpy(dtype=np.float64)
    return np.r_[theta, theta[0]], np.r_[radius, radius[0]]


def _unwrapped_to_full_polar(sub: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray]:
    sub = sub.sort_values("relative_angle_deg")
    angles = sub["relative_angle_deg"].to_numpy(dtype=np.float64)
    values = sub[value_col].to_numpy(dtype=np.float64)
    extra = (angles[1:-1] + 180.0) if angles.size > 2 else np.asarray([], dtype=np.float64)
    extra_values = values[1:-1] if values.size > 2 else np.asarray([], dtype=np.float64)
    full_angles = np.concatenate([angles, extra])
    full_values = np.concatenate([values, extra_values])
    order = np.argsort(full_angles)
    theta = np.radians(full_angles[order])
    radius = full_values[order]
    return np.r_[theta, theta[0]], np.r_[radius, radius[0]]


def plot_polar_small_multiples(profile: pd.DataFrame, out_path: Path, *, zero_origin: bool) -> None:
    labels = sorted(profile["coherence_bin"].astype(str).unique(), key=lambda x: float(x.split("-", maxsplit=1)[0]))
    colors = _profile_color_map(labels)
    fig, axes = plt.subplots(2, 5, figsize=(11.0, 5.4), subplot_kw={"projection": "polar"}, constrained_layout=True)
    finite = profile["rms_arcmin"].to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    rlo, rhi = float(np.min(finite)), float(np.max(finite))
    pad = max(0.04, 0.08 * (rhi - rlo))
    for ax, label in zip(axes.flat, labels, strict=True):
        sub = profile[profile["coherence_bin"].astype(str) == label]
        theta, radius = _wrap_polar_profile(sub)
        _setup_contour_relative_polar(ax)
        ax.plot(theta, radius, color=colors[label], linewidth=2.0)
        ax.fill(theta, radius, color=colors[label], alpha=0.08)
        ax.set_title(f"coh {label}", fontsize=10)
        ax.set_rlim(0.0 if zero_origin else max(0.0, rlo - pad), rhi + pad)
    fig.suptitle("Position spread relative to local edge", fontsize=15)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_polar_overlay(profile: pd.DataFrame, out_path: Path, *, zero_origin: bool) -> None:
    labels = sorted(profile["coherence_bin"].astype(str).unique(), key=lambda x: float(x.split("-", maxsplit=1)[0]))
    colors = _profile_color_map(labels)
    fig = plt.figure(figsize=(6.2, 5.6))
    ax = fig.add_subplot(111, projection="polar")
    _setup_contour_relative_polar(ax)
    finite = profile["rms_arcmin"].to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    rlo, rhi = float(np.min(finite)), float(np.max(finite))
    pad = max(0.04, 0.08 * (rhi - rlo))
    for label in labels:
        sub = profile[profile["coherence_bin"].astype(str) == label]
        theta, radius = _wrap_polar_profile(sub)
        ax.plot(theta, radius, color=colors[label], linewidth=1.8, label=label)
    ax.set_rlim(0.0 if zero_origin else max(0.0, rlo - pad), rhi + pad)
    ax.set_title("Position spread relative to local edge", va="bottom", fontsize=13)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.50), title="coherence")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _set_unwrapped_axis_labels(ax: plt.Axes) -> None:
    ax.set_xlim(0.0, 180.0)
    ax.set_xticks([0.0, 90.0, 180.0])
    ax.set_xticklabels(["parallel", "orthogonal", "parallel"])
    ax.grid(axis="y", color=GRID, linewidth=0.9)
    _clean_axis(ax)


def plot_unwrapped_small_multiples(profile: pd.DataFrame, out_path: Path, *, zero_origin: bool) -> None:
    labels = sorted(profile["coherence_bin"].astype(str).unique(), key=lambda x: float(x.split("-", maxsplit=1)[0]))
    colors = _profile_color_map(labels)
    fig, axes = plt.subplots(2, 5, figsize=(11.4, 5.5), sharex=True, sharey=zero_origin, constrained_layout=True)
    finite = profile["rms_arcmin"].to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    ylo, yhi = float(np.min(finite)), float(np.max(finite))
    pad = max(0.04, 0.08 * (yhi - ylo))
    for ax, label in zip(axes.flat, labels, strict=True):
        sub = profile[profile["coherence_bin"].astype(str) == label].sort_values("relative_angle_deg")
        ax.plot(sub["relative_angle_deg"], sub["rms_arcmin"], color=colors[label], linewidth=2.0)
        ax.axvline(90.0, color="#7d858c", lw=0.8, ls=":")
        ax.set_title(f"coh {label}", fontsize=10)
        _set_unwrapped_axis_labels(ax)
        ax.set_ylim(0.0 if zero_origin else ylo - pad, yhi + pad)
    axes[0, 0].set_ylabel("RMS (arcmin)")
    axes[1, 0].set_ylabel("RMS (arcmin)")
    for ax in axes[-1, :]:
        ax.set_xlabel("angle from local edge")
    fig.suptitle("Unwrapped position-spread profiles", fontsize=15)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_unwrapped_overlay(profile: pd.DataFrame, out_path: Path, *, zero_origin: bool) -> None:
    labels = sorted(profile["coherence_bin"].astype(str).unique(), key=lambda x: float(x.split("-", maxsplit=1)[0]))
    colors = _profile_color_map(labels)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    finite = profile["rms_arcmin"].to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    ylo, yhi = float(np.min(finite)), float(np.max(finite))
    pad = max(0.04, 0.08 * (yhi - ylo))
    for label in labels:
        sub = profile[profile["coherence_bin"].astype(str) == label].sort_values("relative_angle_deg")
        ax.plot(sub["relative_angle_deg"], sub["rms_arcmin"], color=colors[label], linewidth=1.8, label=label)
    ax.axvline(90.0, color="#7d858c", lw=0.8, ls=":")
    ax.set_ylim(0.0 if zero_origin else ylo - pad, yhi + pad)
    ax.set_ylabel("position spread RMS (arcmin)")
    ax.set_xlabel("angle from local edge")
    _set_unwrapped_axis_labels(ax)
    ax.legend(frameon=False, ncol=2, title="coherence")
    ax.set_title("Unwrapped position spread relative to local edge", loc="left")
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_observed_vs_random_unwrapped(baseline: pd.DataFrame, out_path: Path, *, zero_origin: bool) -> None:
    labels = sorted(baseline["coherence_bin"].astype(str).unique(), key=lambda x: float(x.split("-", maxsplit=1)[0]))
    colors = _profile_color_map(labels)
    fig, axes = plt.subplots(2, 5, figsize=(11.4, 5.5), sharex=True, sharey=zero_origin, constrained_layout=True)
    finite_cols = [
        "observed_rms_arcmin",
        "random_orientation_ci95_low_arcmin",
        "random_orientation_ci95_high_arcmin",
    ]
    finite = baseline[finite_cols].to_numpy(dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    ylo, yhi = float(np.min(finite)), float(np.max(finite))
    pad = max(0.04, 0.08 * (yhi - ylo))
    for ax, label in zip(axes.flat, labels, strict=True):
        sub = baseline[baseline["coherence_bin"].astype(str) == label].sort_values("relative_angle_deg")
        color = colors[label]
        x = sub["relative_angle_deg"].to_numpy(dtype=np.float64)
        ax.fill_between(
            x,
            sub["random_orientation_ci95_low_arcmin"].to_numpy(dtype=np.float64),
            sub["random_orientation_ci95_high_arcmin"].to_numpy(dtype=np.float64),
            color=RANDOM,
            alpha=0.10,
            lw=0,
        )
        ax.plot(x, sub["random_orientation_median_rms_arcmin"], color=RANDOM, linestyle="--", linewidth=1.2)
        ax.plot(x, sub["observed_rms_arcmin"], color=color, linewidth=2.0)
        ax.axvline(90.0, color="#7d858c", lw=0.8, ls=":")
        ax.set_title(f"coh {label}", fontsize=10)
        _set_unwrapped_axis_labels(ax)
        ax.set_ylim(0.0 if zero_origin else ylo - pad, yhi + pad)
    axes[0, 0].set_ylabel("RMS (arcmin)")
    axes[1, 0].set_ylabel("RMS (arcmin)")
    for ax in axes[-1, :]:
        ax.set_xlabel("angle from local edge")
    fig.suptitle("Observed profile vs randomized-orientation baseline", fontsize=15)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_delta_vs_random_unwrapped(baseline: pd.DataFrame, out_path: Path) -> None:
    labels = sorted(baseline["coherence_bin"].astype(str).unique(), key=lambda x: float(x.split("-", maxsplit=1)[0]))
    colors = _profile_color_map(labels)
    fig, axes = plt.subplots(2, 5, figsize=(11.4, 5.5), sharex=True, sharey=True, constrained_layout=True)
    finite = baseline["observed_minus_random_median_arcmin"].to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    yabs = max(0.08, float(np.nanmax(np.abs(finite))) * 1.12)
    for ax, label in zip(axes.flat, labels, strict=True):
        sub = baseline[baseline["coherence_bin"].astype(str) == label].sort_values("relative_angle_deg")
        ax.plot(
            sub["relative_angle_deg"],
            sub["observed_minus_random_median_arcmin"],
            color=colors[label],
            linewidth=2.0,
        )
        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
        ax.axvline(90.0, color="#7d858c", lw=0.8, ls=":")
        ax.set_ylim(-yabs, yabs)
        ax.set_title(f"coh {label}", fontsize=10)
        _set_unwrapped_axis_labels(ax)
    axes[0, 0].set_ylabel("observed - random\nRMS (arcmin)")
    axes[1, 0].set_ylabel("observed - random\nRMS (arcmin)")
    for ax in axes[-1, :]:
        ax.set_xlabel("angle from local edge")
    fig.suptitle("Observed deviation from randomized-orientation baseline", fontsize=15)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_observed_vs_random_polar(baseline: pd.DataFrame, out_path: Path) -> None:
    labels = sorted(baseline["coherence_bin"].astype(str).unique(), key=lambda x: float(x.split("-", maxsplit=1)[0]))
    colors = _profile_color_map(labels)
    fig, axes = plt.subplots(2, 5, figsize=(11.0, 5.4), subplot_kw={"projection": "polar"}, constrained_layout=True)
    finite = baseline[
        [
            "observed_rms_arcmin",
            "random_orientation_ci95_low_arcmin",
            "random_orientation_ci95_high_arcmin",
        ]
    ].to_numpy(dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    rhi = float(np.max(finite)) * 1.08
    for ax, label in zip(axes.flat, labels, strict=True):
        sub = baseline[baseline["coherence_bin"].astype(str) == label]
        _setup_contour_relative_polar(ax)
        theta_obs, radius_obs = _unwrapped_to_full_polar(sub, "observed_rms_arcmin")
        theta_null, radius_null = _unwrapped_to_full_polar(sub, "random_orientation_median_rms_arcmin")
        theta_lo, radius_lo = _unwrapped_to_full_polar(sub, "random_orientation_ci95_low_arcmin")
        _, radius_hi = _unwrapped_to_full_polar(sub, "random_orientation_ci95_high_arcmin")
        ax.fill_between(theta_lo, radius_lo, radius_hi, color=RANDOM, alpha=0.10, lw=0)
        ax.plot(theta_null, radius_null, color=RANDOM, linestyle="--", linewidth=1.1)
        ax.plot(theta_obs, radius_obs, color=colors[label], linewidth=2.0)
        ax.set_rlim(0.0, rhi)
        ax.set_title(f"coh {label}", fontsize=10)
    fig.suptitle("Observed vs randomized-orientation baseline", fontsize=15)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _record_figure(paths: dict[str, Path], key: str, stem: Path) -> None:
    paths[f"{key}_png"] = stem.with_suffix(".png")
    paths[f"{key}_pdf"] = stem.with_suffix(".pdf")


def write_outputs(
    df: pd.DataFrame,
    bin_meta: pd.DataFrame,
    out_dir: Path,
    *,
    angle_step_deg: float,
    n_bootstrap: int,
    n_random_orientation: int,
    seed: int,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    polar_angles = np.arange(0.0, 360.0, float(angle_step_deg), dtype=np.float64)
    unwrapped_angles = np.arange(0.0, 180.0 + 0.5 * float(angle_step_deg), float(angle_step_deg), dtype=np.float64)

    paths: dict[str, Path] = {}
    progression = make_progression_summary(df, bin_meta, n_bootstrap=n_bootstrap, seed=seed)
    paths["progression_csv"] = out_dir / "b_position_spread_progression_by_edge_coherence.csv"
    progression.to_csv(paths["progression_csv"], index=False)
    progression_stem = out_dir / "b_position_spread_progression_by_edge_coherence"
    _record_figure(paths, "progression", progression_stem)
    plot_progression(progression, progression_stem)

    polar_profile = make_directional_profiles(df, bin_meta, polar_angles)
    paths["polar_profile_csv"] = out_dir / "b_position_spread_polar_profiles_by_edge_coherence.csv"
    polar_profile.to_csv(paths["polar_profile_csv"], index=False)
    _record_figure(paths, "polar_small_multiples_zoomed", out_dir / "b_position_spread_polar_small_multiples_by_edge_coherence")
    plot_polar_small_multiples(
        polar_profile,
        out_dir / "b_position_spread_polar_small_multiples_by_edge_coherence",
        zero_origin=False,
    )
    _record_figure(paths, "polar_small_multiples_zero_origin", out_dir / "b_position_spread_polar_small_multiples_by_edge_coherence_zero_origin")
    plot_polar_small_multiples(
        polar_profile,
        out_dir / "b_position_spread_polar_small_multiples_by_edge_coherence_zero_origin",
        zero_origin=True,
    )
    _record_figure(paths, "polar_overlay_zero_origin", out_dir / "b_position_spread_polar_overlay_by_edge_coherence_zero_origin")
    plot_polar_overlay(
        polar_profile,
        out_dir / "b_position_spread_polar_overlay_by_edge_coherence_zero_origin",
        zero_origin=True,
    )

    unwrapped_profile = make_directional_profiles(df, bin_meta, unwrapped_angles)
    paths["unwrapped_profile_csv"] = out_dir / "b_position_spread_unwrapped_profiles_by_edge_coherence.csv"
    unwrapped_profile.to_csv(paths["unwrapped_profile_csv"], index=False)
    _record_figure(paths, "unwrapped_small_multiples_zero_origin", out_dir / "b_position_spread_unwrapped_small_multiples_by_edge_coherence_zero_origin")
    plot_unwrapped_small_multiples(
        unwrapped_profile,
        out_dir / "b_position_spread_unwrapped_small_multiples_by_edge_coherence_zero_origin",
        zero_origin=True,
    )
    _record_figure(paths, "unwrapped_small_multiples_zoomed", out_dir / "b_position_spread_unwrapped_small_multiples_by_edge_coherence_zoomed")
    plot_unwrapped_small_multiples(
        unwrapped_profile,
        out_dir / "b_position_spread_unwrapped_small_multiples_by_edge_coherence_zoomed",
        zero_origin=False,
    )
    _record_figure(paths, "unwrapped_overlay_zero_origin", out_dir / "b_position_spread_unwrapped_overlay_by_edge_coherence_zero_origin")
    plot_unwrapped_overlay(
        unwrapped_profile,
        out_dir / "b_position_spread_unwrapped_overlay_by_edge_coherence_zero_origin",
        zero_origin=True,
    )
    _record_figure(paths, "unwrapped_overlay_zoomed", out_dir / "b_position_spread_unwrapped_overlay_by_edge_coherence_zoomed")
    plot_unwrapped_overlay(
        unwrapped_profile,
        out_dir / "b_position_spread_unwrapped_overlay_by_edge_coherence_zoomed",
        zero_origin=False,
    )

    random_baseline = make_random_orientation_baseline(
        df,
        bin_meta,
        unwrapped_profile,
        unwrapped_angles,
        n_random_orientation=n_random_orientation,
        seed=seed + 100_000,
    )
    paths["random_baseline_csv"] = out_dir / "b_position_spread_random_orientation_baseline_by_edge_coherence.csv"
    random_baseline.to_csv(paths["random_baseline_csv"], index=False)
    _record_figure(paths, "observed_vs_random_unwrapped_zero_origin", out_dir / "b_position_spread_unwrapped_observed_vs_random_orientation_by_edge_coherence_zero_origin")
    plot_observed_vs_random_unwrapped(
        random_baseline,
        out_dir / "b_position_spread_unwrapped_observed_vs_random_orientation_by_edge_coherence_zero_origin",
        zero_origin=True,
    )
    _record_figure(paths, "observed_vs_random_unwrapped_zoomed", out_dir / "b_position_spread_unwrapped_observed_vs_random_orientation_by_edge_coherence_zoomed")
    plot_observed_vs_random_unwrapped(
        random_baseline,
        out_dir / "b_position_spread_unwrapped_observed_vs_random_orientation_by_edge_coherence_zoomed",
        zero_origin=False,
    )
    _record_figure(paths, "observed_minus_random_unwrapped", out_dir / "b_position_spread_unwrapped_delta_vs_random_orientation_by_edge_coherence")
    plot_delta_vs_random_unwrapped(
        random_baseline,
        out_dir / "b_position_spread_unwrapped_delta_vs_random_orientation_by_edge_coherence",
    )
    _record_figure(paths, "observed_vs_random_polar_zero_origin", out_dir / "b_position_spread_polar_small_multiples_observed_vs_random_orientation_zero_origin")
    plot_observed_vs_random_polar(
        random_baseline,
        out_dir / "b_position_spread_polar_small_multiples_observed_vs_random_orientation_zero_origin",
    )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-windows", type=Path, default=DEFAULT_INPUT_WINDOWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--coherence-bin-width", type=float, default=0.1)
    parser.add_argument("--angle-step-deg", type=float, default=3.75)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-random-orientation", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> Path:
    df, bin_meta = load_windows(Path(args.input_windows), coherence_bin_width=float(args.coherence_bin_width))
    out_dir = Path(args.out_dir)
    paths = write_outputs(
        df,
        bin_meta,
        out_dir,
        angle_step_deg=float(args.angle_step_deg),
        n_bootstrap=int(args.n_bootstrap),
        n_random_orientation=int(args.n_random_orientation),
        seed=int(args.seed),
    )
    cfg = FollowupConfig(
        input_windows=str(Path(args.input_windows)),
        out_dir=str(out_dir),
        coherence_bin_width=float(args.coherence_bin_width),
        angle_step_deg=float(args.angle_step_deg),
        n_bootstrap=int(args.n_bootstrap),
        n_random_orientation=int(args.n_random_orientation),
        seed=int(args.seed),
    )
    write_json(
        out_dir / "backimage_contour_position_spread_followups_metadata.json",
        {
            "config": asdict(cfg),
            "n_windows": int(df.shape[0]),
            "n_sessions": int(df["session"].nunique()),
            "n_coherence_bins": int(bin_meta.shape[0]),
            "outputs": {key: str(value) for key, value in sorted(paths.items())},
            "random_orientation_note": (
                "The null preserves each eye-position covariance window but replaces image_edge_axis_deg "
                "with a random axial orientation drawn uniformly on [0, 180) degrees for each window and "
                "randomization draw."
            ),
        },
    )
    print(f"Wrote BackImage contour position-spread follow-up plots to {out_dir}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
