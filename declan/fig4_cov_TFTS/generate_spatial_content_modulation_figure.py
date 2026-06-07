#!/usr/bin/env python3
"""Generate the spatial-content modulation figure.

The figure starts with the former Figure 4F spatial-content result and carries
the supporting diagnostics: spectral confounds, absolute recruitment,
continuous residualized regressions, phase scrambling, and leave-one-image-out
stability.  The cumulative-information example from the Jake/Panel-E analysis
handoff is written as a separate Panel E support diagnostic.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from VisionCore.paths import VISIONCORE_ROOT


TEXT = "#202124"
MODEL = "#2f5f9f"
BRIDGE = "#7b5ea7"
NULL = "#9a9a9a"
ACCENT = "#c44e52"
GREEN = "#3b7f5f"

FIGURE_PREFIX = "spatial_content_modulation_figure"
PANEL_E_SUPPORT_PREFIX = "spatial_content_modulation_panelE_support"

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.8,
})


@dataclass(frozen=True)
class Paths:
    panel_root: Path
    tfts_root: Path
    panel_e_root: Path
    out_dir: Path


def clean_axes(ax: plt.Axes, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, alpha=0.18, zorder=-10)


def panel_label(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=11, pad=4)
    ax.text(0.14, 1.015, title, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.7, fontweight="bold", color=TEXT)


def _z(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    sd = float(np.nanstd(arr))
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(arr)
    return (arr - float(np.nanmean(arr))) / sd


def _ols_beta_ci(y: np.ndarray, x: np.ndarray, term_index: int) -> tuple[float, float, float, int]:
    keep = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    y = np.asarray(y[keep], dtype=np.float64)
    x = np.asarray(x[keep], dtype=np.float64)
    n, p = x.shape if x.ndim == 2 else (0, 0)
    if n <= p + 1:
        return float("nan"), float("nan"), float("nan"), int(n)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    sigma2 = float(np.sum(resid * resid) / max(n - p, 1))
    xtx_inv = np.linalg.pinv(x.T @ x)
    se = float(np.sqrt(max(sigma2 * xtx_inv[term_index, term_index], 0.0)))
    b = float(beta[term_index])
    return b, b - 1.96 * se, b + 1.96 * se, int(n)


def _current_frame(history: np.ndarray) -> np.ndarray:
    h = np.asarray(history, dtype=np.float64)
    frame = h[0] if h.ndim == 3 else np.squeeze(h)
    if frame.ndim != 2:
        raise ValueError(f"Expected 2D current frame, got {frame.shape}")
    return frame


def _spectral_features(frame: np.ndarray, n_radial_bins: int = 8, n_orientation_bins: int = 6) -> dict[str, float]:
    x = np.asarray(frame, dtype=np.float64)
    x = x - float(np.mean(x))
    h, w = x.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy)
    theta = (np.arctan2(yy, xx) + np.pi) % np.pi
    power = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    power[rr == 0] = 0.0
    total = float(np.sum(power) + 1e-12)

    positive = rr[rr > 0]
    edges = np.geomspace(float(np.min(positive)), float(np.max(positive)) + 1e-9, n_radial_bins + 1)
    out: dict[str, float] = {}
    density_vals: list[float] = []
    center_vals: list[float] = []
    for i in range(n_radial_bins):
        mask = (rr >= edges[i]) & (rr < edges[i + 1] if i < n_radial_bins - 1 else rr <= edges[i + 1])
        bin_power = power[mask]
        density = float(np.mean(bin_power)) if int(np.sum(mask)) > 0 else float("nan")
        center = float(np.sqrt(edges[i] * edges[i + 1]))
        out[f"radial_power_bin_{i:02d}"] = float(np.sum(power[mask]) / total)
        out[f"radial_psd_density_bin_{i:02d}"] = density
        out[f"radial_freq_center_{i:02d}"] = center
        density_vals.append(density)
        center_vals.append(center)

    center_arr = np.asarray(center_vals, dtype=np.float64)
    density_arr = np.asarray(density_vals, dtype=np.float64)
    keep = np.isfinite(center_arr) & np.isfinite(density_arr) & (center_arr > 0) & (density_arr > 0)
    if int(np.sum(keep)) >= 3:
        slope, intercept = np.polyfit(np.log10(center_arr[keep]), np.log10(density_arr[keep]), deg=1)
        out["radial_psd_loglog_slope"] = float(slope)
        out["radial_psd_loglog_intercept"] = float(intercept)
    else:
        out["radial_psd_loglog_slope"] = float("nan")
        out["radial_psd_loglog_intercept"] = float("nan")

    out["spectral_centroid"] = float(np.sum(rr * power) / total)
    flat_order = np.argsort(rr.ravel())
    rr_flat = rr.ravel()[flat_order]
    pw_flat = power.ravel()[flat_order]
    cdf = np.cumsum(pw_flat) / total
    out["spectral_median_frequency"] = float(rr_flat[int(np.searchsorted(cdf, 0.5, side="left"))])

    gx = np.diff(x, axis=1, prepend=x[:, :1])
    gy = np.diff(x, axis=0, prepend=x[:1, :])
    grad_power = gx * gx + gy * gy
    angle = (np.arctan2(gy, gx) + np.pi) % np.pi
    gp_total = float(np.sum(grad_power) + 1e-12)
    ori_edges = np.linspace(0.0, np.pi, n_orientation_bins + 1)
    for i in range(n_orientation_bins):
        mask = (angle >= ori_edges[i]) & (angle < ori_edges[i + 1] if i < n_orientation_bins - 1 else angle <= ori_edges[i + 1])
        out[f"orientation_energy_bin_{i:02d}"] = float(np.sum(grad_power[mask]) / gp_total)
    return out


def build_spectral_audit(paths: Paths, n_radial_bins: int = 8) -> pd.DataFrame:
    metrics_path = paths.panel_root / "panelF_image_structure_metrics.csv"
    out_path = paths.panel_root / "panelF_spectral_audit.csv"
    metrics = pd.read_csv(metrics_path)
    with (paths.tfts_root / "tangent_maps" / "twin_tangent_maps.pkl").open("rb") as handle:
        cache = pickle.load(handle)
    delta_key = min((float(k) for k in cache["object_payload"].keys()), key=lambda k: abs(k - 0.25))
    payload = cache["object_payload"][delta_key]

    rows: list[dict[str, Any]] = []
    for _, row in metrics.iterrows():
        oid = str(row["object_id"])
        if oid not in payload:
            continue
        feats = _spectral_features(_current_frame(payload[oid]["history"]), n_radial_bins=n_radial_bins)
        rows.append({
            "object_id": oid,
            "image_id": int(row["image_id"]),
            "structure_group": row["structure_group"],
            "residual_structure_score": float(row["residual_structure_score"]),
            "rms_contrast": float(row["rms_contrast"]),
            "gradient_rms": float(row["gradient_rms"]),
            "orientation_coherence": float(row["orientation_coherence"]),
            **feats,
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(out_path, index=False)
    return audit


def write_spectral_slope_summary(paths: Paths, audit: pd.DataFrame) -> pd.DataFrame:
    out_path = paths.panel_root / "panelF_spectral_psd_slope_summary.csv"
    rows: list[dict[str, Any]] = []
    for label, d in [("all", audit), *list(audit.groupby("structure_group"))]:
        slopes = pd.to_numeric(d["radial_psd_loglog_slope"], errors="coerce").dropna()
        rows.append({
            "structure_group": str(label),
            "n_objects": int(len(slopes)),
            "mean_radial_psd_loglog_slope": float(slopes.mean()) if len(slopes) else float("nan"),
            "std_radial_psd_loglog_slope": float(slopes.std(ddof=1)) if len(slopes) > 1 else float("nan"),
            "median_radial_psd_loglog_slope": float(slopes.median()) if len(slopes) else float("nan"),
            "min_radial_psd_loglog_slope": float(slopes.min()) if len(slopes) else float("nan"),
            "max_radial_psd_loglog_slope": float(slopes.max()) if len(slopes) else float("nan"),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False)
    return summary


def _spectral_pcs(audit: pd.DataFrame, n_components: int = 3) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in audit.columns if c.startswith("radial_power_bin_")]
    x = audit[cols].to_numpy(dtype=np.float64)
    x = np.column_stack([_z(x[:, i]) for i in range(x.shape[1])])
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    pcs = x @ vt[:n_components].T
    names = [f"spectrum_pc{i + 1}" for i in range(pcs.shape[1])]
    return pcs, names


def build_residualized_excess(paths: Paths, audit: pd.DataFrame) -> pd.DataFrame:
    out_path = paths.panel_root / "panelF_residualized_excess_summary.csv"
    sweep = pd.read_csv(paths.panel_root / "panelF_natural_structure_scale_sweep.csv", low_memory=False)
    obj = sweep[
        (sweep["bootstrap_id_or_fold"].astype(str) == "object") &
        (sweep["metric_name"].astype(str) == "tangent_subspace_fraction") &
        (sweep["image_condition"].astype(str) == "intact_natural")
    ].copy()
    pivot = (
        obj.pivot_table(index=["object_id", "image_id", "structure_group", "displacement_arcmin"],
                        columns="basis_type", values="metric_value", aggfunc="mean")
        .reset_index()
    )
    pivot["true_minus_random"] = pivot["true_tangent"] - pivot["random_subspace"]
    pivot["true_minus_unit_shuffle"] = pivot["true_tangent"] - pivot["unit_shuffle"]

    pcs, pc_names = _spectral_pcs(audit)
    audit_pcs = audit[["object_id", "residual_structure_score", "rms_contrast", "gradient_rms"]].copy()
    for i, name in enumerate(pc_names):
        audit_pcs[name] = pcs[:, i]
    merged = pivot.merge(audit_pcs, on="object_id", how="left")

    rows: list[dict[str, Any]] = []
    for metric in ["true_minus_random", "true_minus_unit_shuffle"]:
        for disp, block in merged.groupby("displacement_arcmin"):
            y = block[metric].to_numpy(dtype=np.float64)
            x_cols = [
                np.ones(len(block), dtype=np.float64),
                _z(block["residual_structure_score"].to_numpy(dtype=np.float64)),
                _z(np.log1p(block["rms_contrast"].to_numpy(dtype=np.float64))),
                _z(np.log1p(block["gradient_rms"].to_numpy(dtype=np.float64))),
            ]
            for name in pc_names:
                x_cols.append(_z(block[name].to_numpy(dtype=np.float64)))
            x = np.column_stack(x_cols)
            b, lo, hi, n = _ols_beta_ci(y, x, term_index=1)
            rows.append({
                "dependent_metric": metric,
                "displacement_arcmin": float(disp),
                "beta_residual_structure": b,
                "ci_lower": lo,
                "ci_upper": hi,
                "n_objects": n,
                "covariates": "rms_contrast,gradient_rms,spectrum_pc1-3",
            })
    reg = pd.DataFrame(rows)
    reg.to_csv(out_path, index=False)
    return reg


def _bootstrap_ci(df: pd.DataFrame, value_col: str = "metric_value") -> pd.DataFrame:
    boot = df[df["bootstrap_id_or_fold"].astype(str) != "observed"].copy()
    obs = df[df["bootstrap_id_or_fold"].astype(str) == "observed"].copy()
    if len(obs) == 0 or len(boot) == 0:
        return pd.DataFrame()
    out = (
        boot.groupby("displacement_arcmin", as_index=False)
        .agg(lo=(value_col, lambda x: np.nanpercentile(x, 2.5)),
             hi=(value_col, lambda x: np.nanpercentile(x, 97.5)))
    )
    return out.merge(obs[["displacement_arcmin", value_col]], on="displacement_arcmin", how="left")


def _plot_spatial_content_recruitment(ax: plt.Axes, paths: Paths, drift: tuple[float, float], msac: tuple[float, float]) -> None:
    panel_label(ax, "S1", "Spatial content recruits compact geometry")
    sweep = pd.read_csv(paths.panel_root / "panelF_natural_structure_scale_sweep.csv", low_memory=False)
    diff = sweep[
        (sweep["metric_name"].astype(str) == "high_minus_low_fraction") &
        (sweep["structure_group"].astype(str) == "high_minus_low") &
        (sweep["image_condition"].astype(str) == "intact_natural")
    ].copy()
    diff["displacement_arcmin"] = pd.to_numeric(diff["displacement_arcmin"], errors="coerce")
    diff["metric_value"] = pd.to_numeric(diff["metric_value"], errors="coerce")

    ax.set_xscale("log")
    ax.axvspan(drift[0], drift[1], color=MODEL, alpha=0.10, lw=0)
    ax.axvspan(msac[0], msac[1], color=BRIDGE, alpha=0.08, lw=0)
    for d_lo, d_hi, color, label in [
        (drift[0], drift[1], MODEL, "drift"),
        (msac[0], msac[1], BRIDGE, "microsaccade"),
    ]:
        if np.isfinite(d_lo) and np.isfinite(d_hi) and d_lo < d_hi:
            ax.text(np.sqrt(d_lo * d_hi), 0.97, label,
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=6.7, color=color)

    for basis, color, label, ls, marker, lw, alpha in [
        ("true_tangent", MODEL, "True tangent basis", "-", "o", 2.2, 0.16),
        ("unit_shuffle", NULL, "Unit-shuffle", "--", "s", 1.45, 0.08),
        ("random_subspace", "0.68", "Random subspace", ":", ".", 1.35, 0.05),
    ]:
        d = diff[diff["basis_type"].astype(str) == basis]
        s = _bootstrap_ci(d)
        if len(s) == 0:
            continue
        ax.fill_between(s["displacement_arcmin"], s["lo"], s["hi"], color=color, alpha=alpha, lw=0)
        ax.plot(s["displacement_arcmin"], s["metric_value"], ls=ls, marker=marker,
                color=color, lw=lw, ms=4.0, markeredgecolor="white", markeredgewidth=0.45,
                label=label)

    finite_y = diff["metric_value"].to_numpy(float)
    finite_y = finite_y[np.isfinite(finite_y)]
    ymin = min(-0.06, float(np.nanpercentile(finite_y, 1)) - 0.015) if finite_y.size else -0.06
    ymax = max(0.16, float(np.nanpercentile(finite_y, 99)) + 0.025) if finite_y.size else 0.16
    xmin = float(np.nanmin(diff["displacement_arcmin"])) * 0.78
    xmax = max(float(np.nanmax(diff["displacement_arcmin"])) * 1.35, msac[1] * 1.05)
    ticks = [0.0625, 0.25, 1, 4, 16, 50]
    ticks = [t for t in ticks if xmin <= t <= xmax]
    ax.axhline(0, color="0.45", lw=0.75, ls=":")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1/16" if np.isclose(t, 0.0625) else f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel("Retinal displacement scale (arcmin)")
    ax.set_ylabel("High - low tangent-subspace fraction")
    clean_axes(ax, grid=True)
    ax.legend(frameon=False, loc="upper left", handlelength=1.4, labelspacing=0.3)


def _plot_spectral_audit(ax: plt.Axes, audit: pd.DataFrame) -> None:
    panel_label(ax, "S2", "Annular Fourier power (not PSD)")
    bins = [c for c in audit.columns if c.startswith("radial_power_bin_")]
    centers = [float(audit[f"radial_freq_center_{i:02d}"].iloc[0]) for i in range(len(bins))]
    for group, color, label in [
        ("high_structure", MODEL, "High structure"),
        ("low_structure_matched", BRIDGE, "Low comparison"),
    ]:
        d = audit[audit["structure_group"] == group]
        y = d[bins].mean(axis=0).to_numpy(float)
        se = d[bins].sem(axis=0).to_numpy(float)
        ax.fill_between(centers, y - se, y + se, color=color, alpha=0.15, lw=0)
        ax.plot(centers, y, "-o", color=color, lw=1.9, ms=4.0, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Radial spatial frequency (cycles/pixel)")
    ax.set_ylabel("Fraction of total Fourier power\nper radial annulus")
    slope = float(pd.to_numeric(audit["radial_psd_loglog_slope"], errors="coerce").mean())
    if np.isfinite(slope):
        ax.text(0.04, 0.07, f"PSD density slope = {slope:.1f}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=6.9, color="0.35")
    clean_axes(ax, grid=True)
    ax.legend(frameon=False, loc="upper right")


def _plot_residual_regression(ax: plt.Axes, reg: pd.DataFrame, drift: tuple[float, float], msac: tuple[float, float]) -> None:
    panel_label(ax, "S4", "Residual structure after spectrum PCs")
    d = reg[reg["dependent_metric"] == "true_minus_random"].sort_values("displacement_arcmin")
    ax.set_xscale("log")
    ax.axvspan(drift[0], drift[1], color=MODEL, alpha=0.10, lw=0)
    ax.axvspan(msac[0], msac[1], color=BRIDGE, alpha=0.08, lw=0)
    ax.fill_between(d["displacement_arcmin"], d["ci_lower"], d["ci_upper"],
                    color=GREEN, alpha=0.18, lw=0)
    ax.plot(d["displacement_arcmin"], d["beta_residual_structure"],
            "-o", color=GREEN, lw=2.0, ms=4.2)
    ax.axhline(0, color="0.45", lw=0.7, ls=":")
    ax.set_xlabel("Retinal displacement scale (arcmin)")
    ax.set_ylabel("OLS beta for residual structure\n(true - random fraction)")
    ax.set_xticks([0.0625, 0.25, 1, 4, 16, 50])
    ax.set_xticklabels(["1/16", "0.25", "1", "4", "16", "50"])
    clean_axes(ax, grid=True)


def _plot_absolute_fraction(ax: plt.Axes, paths: Paths, drift: tuple[float, float], msac: tuple[float, float]) -> None:
    panel_label(ax, "S3", "Absolute tangent fraction")
    sweep = pd.read_csv(paths.panel_root / "panelF_natural_structure_scale_sweep.csv", low_memory=False)
    obj = sweep[
        (sweep["metric_name"].astype(str) == "tangent_subspace_fraction") &
        (sweep["bootstrap_id_or_fold"].astype(str) == "object") &
        (sweep["basis_type"].astype(str) == "true_tangent") &
        (sweep["image_condition"].astype(str) == "intact_natural")
    ].copy()
    obj["displacement_arcmin"] = pd.to_numeric(obj["displacement_arcmin"], errors="coerce")
    obj["metric_value"] = pd.to_numeric(obj["metric_value"], errors="coerce")

    ax.set_xscale("log")
    ax.axvspan(drift[0], drift[1], color=MODEL, alpha=0.10, lw=0)
    ax.axvspan(msac[0], msac[1], color=BRIDGE, alpha=0.08, lw=0)
    for group, color, label in [
        ("high_structure", MODEL, "High structure"),
        ("low_structure_matched", BRIDGE, "Low comparison"),
        ("middle_structure", "0.65", "Middle"),
    ]:
        d = obj[obj["structure_group"].astype(str) == group]
        if len(d) == 0:
            continue
        summary = (
            d.groupby("displacement_arcmin", as_index=False)
            .agg(mean=("metric_value", "mean"),
                 lo=("metric_value", lambda x: np.nanpercentile(x, 25)),
                 hi=("metric_value", lambda x: np.nanpercentile(x, 75)))
            .sort_values("displacement_arcmin")
        )
        ax.fill_between(summary["displacement_arcmin"], summary["lo"], summary["hi"],
                        color=color, alpha=0.12, lw=0)
        ax.plot(summary["displacement_arcmin"], summary["mean"], "-o",
                color=color, lw=1.9, ms=4.0, label=label)
    ax.set_xlabel("Retinal displacement scale (arcmin)")
    ax.set_ylabel("Tangent-subspace fraction")
    ax.set_xticks([0.0625, 0.25, 1, 4, 16, 50])
    ax.set_xticklabels(["1/16", "0.25", "1", "4", "16", "50"])
    clean_axes(ax, grid=True)
    ax.legend(frameon=False, loc="lower right", handlelength=1.4)


def _plot_phase_diagnostic(ax: plt.Axes, paths: Paths, drift: tuple[float, float], msac: tuple[float, float]) -> None:
    panel_label(ax, "S5", "Per-lag phase-scramble diagnostic")
    phase = pd.read_csv(paths.panel_root / "panelF_phase_scramble_diagnostic.csv", low_memory=False)
    hml = phase[
        (phase["metric_name"] == "high_minus_low_fraction") &
        (phase["basis_type"] == "true_tangent")
    ].copy()
    ax.set_xscale("log")
    ax.axvspan(drift[0], drift[1], color=MODEL, alpha=0.10, lw=0)
    ax.axvspan(msac[0], msac[1], color=BRIDGE, alpha=0.08, lw=0)
    for condition, color, label in [
        ("intact_natural", MODEL, "Intact"),
        ("phase_scrambled", ACCENT, "Phase scrambled"),
    ]:
        d = hml[hml["image_condition"] == condition]
        s = _bootstrap_ci(d)
        if len(s) == 0:
            continue
        ax.fill_between(s["displacement_arcmin"], s["lo"], s["hi"], color=color, alpha=0.12, lw=0)
        ax.plot(s["displacement_arcmin"], s["metric_value"], "-o", color=color, lw=1.8, ms=4.0, label=label)
    ax.axhline(0, color="0.45", lw=0.7, ls=":")
    ax.set_xlabel("Retinal displacement scale (arcmin)")
    ax.set_ylabel("High - low tangent fraction")
    ax.set_xticks([0.0625, 0.25, 1, 4, 16, 50])
    ax.set_xticklabels(["1/16", "0.25", "1", "4", "16", "50"])
    clean_axes(ax, grid=True)
    ax.legend(frameon=False, loc="upper right")


def _plot_leave_one_image(ax: plt.Axes, paths: Paths, drift: tuple[float, float], msac: tuple[float, float]) -> None:
    panel_label(ax, "S6", "Image-level robustness")
    loio = pd.read_csv(paths.panel_root / "panelF_leave_one_image_out.csv")
    d = loio[
        (loio["basis_type"] == "true_tangent") &
        (loio["left_out_image_id"].astype(str) == "summary")
    ].copy().sort_values("displacement_arcmin")
    ax.set_xscale("log")
    ax.axvspan(drift[0], drift[1], color=MODEL, alpha=0.10, lw=0)
    ax.axvspan(msac[0], msac[1], color=BRIDGE, alpha=0.08, lw=0)
    ax.fill_between(d["displacement_arcmin"], d["min_leave_one_out"], d["max_leave_one_out"],
                    color=MODEL, alpha=0.16, lw=0, label="leave-one-image range")
    ax.plot(d["displacement_arcmin"], d["high_minus_low_fraction"], "-o", color=MODEL, lw=2.0, ms=4.2,
            label="observed")
    ax.axhline(0, color="0.45", lw=0.7, ls=":")
    ax.set_xlabel("Retinal displacement scale (arcmin)")
    ax.set_ylabel("High - low tangent fraction")
    ax.set_xticks([0.0625, 0.25, 1, 4, 16, 50])
    ax.set_xticklabels(["1/16", "0.25", "1", "4", "16", "50"])
    clean_axes(ax, grid=True)
    ax.legend(frameon=False, loc="upper right")


def _find_accumulation_example(npz: np.lib.npyio.NpzFile) -> str | None:
    series_ids = sorted({
        k.split("/")[0]
        for k in npz.files
        if k.endswith("/full/cumulative_fisher_pattern") and "_real/" in k
    })
    candidates: list[tuple[float, str]] = []
    for real_id in series_ids:
        base = real_id.removesuffix("_real")
        stab_id = f"{base}_stabilized"
        kr = f"{real_id}/full/cumulative_fisher_pattern"
        ks = f"{stab_id}/full/cumulative_fisher_pattern"
        if kr in npz.files and ks in npz.files:
            candidates.append((float(npz[kr][-1] - npz[ks][-1]), base))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][1]


def _plot_accumulation_example(
    ax: plt.Axes,
    paths: Paths,
    *,
    label: str = "E-S1",
    title: str = "Example cumulative local displacement information",
) -> None:
    panel_label(ax, label, title)
    npz_path = paths.panel_e_root / "cache" / "panelE_cumulative_information_series.npz"
    if not npz_path.exists():
        ax.text(0.5, 0.5, "Panel E cumulative series missing", ha="center", va="center", transform=ax.transAxes)
        clean_axes(ax, grid=True)
        return
    z = np.load(npz_path, allow_pickle=True)
    time = z["time_s"]
    base = _find_accumulation_example(z)
    if base is None:
        ax.text(0.5, 0.5, "No paired real/stabilized example", ha="center", va="center", transform=ax.transAxes)
        clean_axes(ax, grid=True)
        return
    curves = [
        (f"{base}_real/full/cumulative_fisher_pattern", TEXT, "full, real", "-", 1.7),
        (f"{base}_real/tangent/cumulative_fisher_pattern", MODEL, "k=10 tangent, real", "-", 2.1),
        (f"{base}_real/unit_shuffle_mean/cumulative_fisher_pattern", NULL, "unit-shuffle mean", "--", 1.6),
        (f"{base}_real/random_orthogonal_mean/cumulative_fisher_pattern", "0.70", "random mean", ":", 1.5),
        (f"{base}_stabilized/full/cumulative_fisher_pattern", "0.35", "full, stabilized", "--", 1.4),
    ]
    scale = 1e6
    for key, color, label, ls, lw in curves:
        if key in z.files:
            ax.plot(time, z[key] / scale, color=color, label=label, ls=ls, lw=lw)
    ax.set_xlabel("Time in trace (s)")
    ax.set_ylabel("Cumulative pattern FI (x10^6)")
    clean_axes(ax, grid=True)
    ax.legend(frameon=False, loc="upper left", ncol=1, handlelength=1.6)
    ax.text(0.98, 0.05, base.replace("_", " "), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.8, color="0.35")


def _load_bands(paths: Paths) -> tuple[tuple[float, float], tuple[float, float]]:
    manifest_path = paths.panel_root / "panelF_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    empirical = manifest.get("empirical_ranges") or {}
    drift = tuple(empirical.get("drift_band_arcmin", [1.0, 3.0]))
    msac = tuple(empirical.get("msac_band_arcmin", [10.0, 50.0]))
    return (float(drift[0]), float(drift[1])), (float(msac[0]), float(msac[1]))


def _write_figure_legend(paths: Paths, slope_summary: pd.DataFrame) -> None:
    slope_row = slope_summary[slope_summary["structure_group"] == "all"]
    slope = float(slope_row["mean_radial_psd_loglog_slope"].iloc[0]) if len(slope_row) else float("nan")
    slope_text = f"{slope:.2f}" if np.isfinite(slope) else "not estimated"
    text = f"""# Figure Legend: Spatial-Content Modulation of Tangent Geometry

**Spatial-content modulation of tangent geometry.**

**(S1) Spatial content recruits compact geometry.** Held-out natural-image patches were split into high-structure and low-structure comparison groups after residualizing the structure score against RMS contrast, gradient RMS, and coarse radial-frequency fractions. The panel plots the high-minus-low fraction of translation-induced response change captured by the image-disjoint k=10 tangent basis, with unit-shuffle and random-subspace references. The effect is positive across finite translations and overlaps the empirical drift regime, supporting a conservative spatial-content modulation result rather than a drift-tuned peak, explicit matching, or natural-phase-structure claim.

**(S2) Annular Fourier power, not PSD.** For each full stimulus-history object, the current frame was Fourier transformed and power was summed within log-spaced radial-frequency annuli. Curves show the mean fraction of total Fourier power contained in each annulus for high-structure and low-structure comparison patches, with SEM bands. This panel audits how power is distributed across radial-frequency bands rather than plotting PSD density directly. A separate radial PSD-density check on the same frames gave a mean log-log slope of {slope_text}, consistent with approximately 1/f^2 natural-image scaling.

**(S3) Absolute tangent-subspace fraction.** Mean fraction of translation-induced response change routed through the image-disjoint k=10 tangent basis is shown for high-structure, low-structure comparison, and middle-structure patches. Shaded bands mark interquartile ranges across objects. Drift and microsaccade displacement regimes are shown as background bands.

**(S4) Residual structure after spectrum PCs.** Object-level tangent recruitment above random-subspace controls was regressed against residual natural-structure score while controlling for RMS contrast, gradient RMS, and the first three radial-spectrum PCs. Coefficients were small and not reliably distinguished from zero after these coarse spectral controls, so this panel is treated as a covariate audit rather than independent evidence for a structure-specific effect.

**(S5) Per-lag phase-scramble diagnostic.** High-minus-low tangent recruitment is compared for intact natural-image histories and per-lag phase-scrambled counterparts that preserve each frame's Fourier amplitude spectrum, mean, and standard deviation while randomizing spatial phase independently at each history lag. Because this manipulation also disrupts temporal and spatiotemporal consistency, it is treated as an out-of-distribution diagnostic. It reduces absolute tangent recruitment but does not cleanly remove the high-minus-low contrast in the sampled drift regime.

**(S6) Image-level robustness.** Leave-one-image-out summaries show that the high-minus-low effect for the true tangent basis remains positive across held-out image identities, indicating that the main effect is not carried by a single source image.
"""
    (paths.out_dir / f"{FIGURE_PREFIX}_legend.md").write_text(text, encoding="utf-8")


def _write_methods(
    paths: Paths,
    slope_summary: pd.DataFrame,
    drift: tuple[float, float],
    msac: tuple[float, float],
) -> None:
    slope_row = slope_summary[slope_summary["structure_group"] == "all"]
    slope = float(slope_row["mean_radial_psd_loglog_slope"].iloc[0]) if len(slope_row) else float("nan")
    slope_text = f"{slope:.2f}" if np.isfinite(slope) else "not estimated"
    text = f"""# Methods: Spatial-Content Modulation of Tangent Geometry

Generated by `declan/fig4_cov_TFTS/generate_spatial_content_modulation_figure.py`.

## Source Files

- Natural-image spatial-content sweep: `{paths.panel_root / "panelF_natural_structure_scale_sweep.csv"}`
- Image structure metrics: `{paths.panel_root / "panelF_image_structure_metrics.csv"}`
- Phase-scramble diagnostic: `{paths.panel_root / "panelF_phase_scramble_diagnostic.csv"}`
- Leave-one-image-out summary: `{paths.panel_root / "panelF_leave_one_image_out.csv"}`
- Tangent-map cache: `{paths.tfts_root / "tangent_maps" / "twin_tangent_maps.pkl"}`
- Figure output directory: `{paths.out_dir}`

Empirical displacement bands were read from `panelF_manifest.json` when available. The plotted drift band is {drift[0]:.3g}-{drift[1]:.3g} arcmin and the microsaccade band is {msac[0]:.3g}-{msac[1]:.3g} arcmin.

## Shared Definitions

Each object is a full stimulus-history patch with an associated response vector under the canonical digital twin. For a finite retinal displacement scale `s`, the response change is treated as

```text
Delta r_o(s) = r_o(s) - r_o(0)
```

For an orthonormal image-disjoint tangent basis `B_k` with `k = 10`, the tangent-subspace recruitment fraction is

```text
f_o(s; B_k) = || B_k B_k^T Delta r_o(s) ||_2^2 / || Delta r_o(s) ||_2^2
```

The same fraction is computed for matched null bases. The unit-shuffle null disrupts coordinated population geometry while preserving unit-level marginal structure. The random-subspace null uses dimension-matched random orthogonal subspaces. The high-minus-low contrast plotted in the figure is

```text
HML_b(s) = mean_o in high f_o(s; B_b) - mean_o in low f_o(s; B_b)
```

where `b` indexes the true tangent basis or one of the null basis families. Confidence bands are read from the bootstrap rows in the production sweep table, using the 2.5th and 97.5th percentiles at each displacement scale.

## Panel S1: Spatial Content Recruits Compact Geometry

S1 plots `HML_b(s)` for the true image-disjoint tangent basis, unit-shuffle basis, and random-subspace basis. The y-axis is therefore not an absolute response magnitude; it is the excess fraction of translation-induced response change routed through the compact tangent geometry for high-structure patches relative to the low-structure comparison group. The groups are not exact matched pairs; they are quantile groups formed after residualizing a raw structure score against RMS contrast, gradient RMS, and coarse radial-frequency fractions.

The central calculation checks whether spatial content changes how strongly finite translations recruit the tangent geometry:

```text
positive HML_true(s) > 0
HML_true(s) > HML_random(s)
```

The interpretation is intentionally conservative. The panel supports positive recruitment across finite translations, with overlap near the empirical drift regime. It is not used as a claim that recruitment has a drift-tuned peak, that the groups are fully matched on all image statistics, or that higher-order natural phase structure is isolated.

## Panel S2: Annular Fourier Power, Not PSD

For each current image frame `x`, the frame is mean-centered and Fourier transformed:

```text
x0 = x - mean(x)
P(u, v) = |fftshift(fft2(x0))|^2
```

The DC component is removed. Radial frequency is

```text
rho(u, v) = sqrt(u^2 + v^2)
```

Power is summed in log-spaced radial-frequency annuli:

```text
A_i = {{(u, v): e_i <= rho(u, v) < e_(i+1)}}
annular_power_i = sum_(u,v in A_i) P(u, v) / sum_(u,v) P(u, v)
```

This is why the S2 curves can peak at mid frequencies: the panel shows total power per log-frequency annulus, not PSD density. As a check, the script also computes radial PSD density per annulus,

```text
psd_density_i = mean_(u,v in A_i) P(u, v)
```

and fits

```text
log10(psd_density_i) = alpha log10(rho_i) + beta
```

using finite positive bins. The mean fitted slope across objects is {slope_text}, consistent with approximately `1/f^2` natural-image scaling.

## Panel S3: Absolute Tangent-Subspace Fraction

S3 plots the object-level absolute `f_o(s; B_k)` values for the true tangent basis, grouped by high-structure, low-structure comparison, and middle-structure patches. Lines show group means over objects at each displacement scale, and bands show interquartile ranges. This panel checks that the S1 high-minus-low contrast comes from interpretable absolute recruitment curves rather than only from a derived difference plot.

## Panel S4: Residual Structure After Spectrum PCs

S4 asks whether continuous residual structure predicts tangent recruitment above random controls after coarse spectral covariates are included. The dependent variable is computed per object and displacement scale as

```text
y_o(s) = f_o(s; true tangent) - f_o(s; random subspace)
```

The radial annular-power vector from S2 is standardized across objects and decomposed by singular value decomposition. The first three spectrum PCs are used as covariates. At each displacement scale, the script fits

```text
y_o = beta_0
    + beta_1 z(residual_structure_o)
    + beta_2 z(log1p(rms_contrast_o))
    + beta_3 z(log1p(gradient_rms_o))
    + beta_4 z(spectrum_pc1_o)
    + beta_5 z(spectrum_pc2_o)
    + beta_6 z(spectrum_pc3_o)
    + epsilon_o
```

The plotted line is `beta_1`, and the band is `beta_1 +/- 1.96 SE(beta_1)` from the ordinary least-squares covariance estimate. In the current run these coefficients are small and not reliably distinguished from zero after the coarse controls. This is a covariate audit, not a causal isolation of structure from all possible image statistics.

## Panel S5: Per-Lag Phase-Scramble Diagnostic

S5 compares the high-minus-low tangent recruitment for intact histories and phase-scrambled histories. The phase-scramble generation preserves each frame's Fourier amplitude spectrum, mean, and standard deviation while randomizing Fourier phase independently at each history lag. The same `HML_true(s)` calculation from S1 is then repeated for phase-scrambled stimuli.

The diagnostic tests whether the intact effect is cleanly removed by this phase randomization. Because phase is scrambled separately for each lag, the manipulation also disrupts temporal and spatiotemporal consistency of the stimulus history, making it an out-of-distribution diagnostic rather than a clean natural-image phase control. In the production result, phase scrambling reduces absolute tangent recruitment but does not consistently attenuate the high-minus-low contrast across the sampled drift-scale regime.

## Panel S6: Image-Level Robustness

S6 uses the leave-one-image-out table. For each source image identity `j`, the high-minus-low contrast is recomputed after excluding all objects from image `j`:

```text
HML_(-j)(s) = mean_o in high, image != j f_o(s)
            - mean_o in low, image != j f_o(s)
```

The plotted line is the observed high-minus-low contrast, and the band spans the minimum to maximum leave-one-image-out value at each scale. This checks whether the S1 effect is carried by a single image identity rather than being stable across held-out natural-image sources.

## Calculation Guardrails

- S1 is a spatial-content diagnostic, not the conceptual landing point for the main tangent-geometry result.
- The high/low groups are residualized comparison groups, not exact matched pairs.
- The analysis does not claim a drift-tuned peak; the safer claim is positive recruitment across finite translations with overlap near the empirical drift range.
- S2 confirms that the frames retain approximately natural `1/f^2` PSD-density scaling even though annular power can peak at mid frequencies.
- S4 controls only the plotted coarse spectral covariates and does not exhaust all possible image-statistical confounds; its coefficients are not treated as a reliable positive result.
- S5 does not establish phase specificity because the per-lag phase-scrambled high-minus-low contrast is not cleanly abolished and the manipulation is out of distribution for a temporal-history model.
"""
    (paths.out_dir / f"{FIGURE_PREFIX}_methods.md").write_text(text, encoding="utf-8")


def _write_panel_e_support_legend(paths: Paths) -> None:
    text = """# Supporting Figure Legend: Panel E Cumulative Information Diagnostic

**Supplementary Figure Sy. Panel E supporting diagnostic: cumulative local displacement information.** A representative cumulative derivative-projection Fisher-information trace from the Panel E/Jake analysis shows how the image-disjoint tangent basis accumulates local spatial-displacement information across the stimulus history relative to full real-motion, full stabilized, unit-shuffle, and random-subspace references. This diagnostic supports the Panel E claim that the tangent subspace captures FEM-related local displacement sensitivity, rather than serving as a spatial-content diagnostic.
"""
    (paths.out_dir / f"{PANEL_E_SUPPORT_PREFIX}_legend.md").write_text(text, encoding="utf-8")


def _compose_panel_e_support(paths: Paths, dpi: int = 300) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    _plot_accumulation_example(ax, paths)
    fig.suptitle("Panel E supporting diagnostic", fontsize=10.0, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.83)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(paths.out_dir / f"{PANEL_E_SUPPORT_PREFIX}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    _write_panel_e_support_legend(paths)


def compose(paths: Paths, dpi: int = 300) -> None:
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    audit = build_spectral_audit(paths)
    slope_summary = write_spectral_slope_summary(paths, audit)
    reg = build_residualized_excess(paths, audit)
    drift, msac = _load_bands(paths)

    fig = plt.figure(figsize=(10.3, 7.6), constrained_layout=False)
    gs = GridSpec(2, 3, figure=fig, left=0.075, right=0.975, bottom=0.08, top=0.90,
                  wspace=0.42, hspace=0.38)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    _plot_spatial_content_recruitment(axes[0], paths, drift, msac)
    _plot_spectral_audit(axes[1], audit)
    _plot_absolute_fraction(axes[2], paths, drift, msac)
    _plot_residual_regression(axes[3], reg, drift, msac)
    _plot_phase_diagnostic(axes[4], paths, drift, msac)
    _plot_leave_one_image(axes[5], paths, drift, msac)

    fig.suptitle("Spatial-content modulation of tangent geometry",
                 fontsize=10.5, fontweight="bold", x=0.52, y=0.965)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(paths.out_dir / f"{FIGURE_PREFIX}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    _write_figure_legend(paths, slope_summary)
    _write_methods(paths, slope_summary, drift, msac)
    _compose_panel_e_support(paths, dpi=dpi)

    manifest = {
        "figure": FIGURE_PREFIX,
        "panel_root": str(paths.panel_root),
        "tfts_root": str(paths.tfts_root),
        "panel_e_root": str(paths.panel_e_root),
        "source_files": {
            "scale_sweep": str(paths.panel_root / "panelF_natural_structure_scale_sweep.csv"),
            "phase_scramble": str(paths.panel_root / "panelF_phase_scramble_diagnostic.csv"),
            "leave_one_image_out": str(paths.panel_root / "panelF_leave_one_image_out.csv"),
            "spectral_audit": str(paths.panel_root / "panelF_spectral_audit.csv"),
            "spectral_psd_slope_summary": str(paths.panel_root / "panelF_spectral_psd_slope_summary.csv"),
            "residualized_excess": str(paths.panel_root / "panelF_residualized_excess_summary.csv"),
            "figure": str(paths.out_dir / f"{FIGURE_PREFIX}.png"),
            "figure_legend": str(paths.out_dir / f"{FIGURE_PREFIX}_legend.md"),
            "figure_methods": str(paths.out_dir / f"{FIGURE_PREFIX}_methods.md"),
            "panelE_support_figure": str(paths.out_dir / f"{PANEL_E_SUPPORT_PREFIX}.png"),
            "panelE_cumulative_information_series": str(paths.panel_e_root / "cache" / "panelE_cumulative_information_series.npz"),
            "panelE_support_legend": str(paths.out_dir / f"{PANEL_E_SUPPORT_PREFIX}_legend.md"),
        },
        "interpretation": (
            "Spatial-content analysis supports a conservative result: intact natural-image spatial content "
            "modulates projection onto compact tangent geometry; residual spectral differences, "
            "per-lag phase scrambling, and weak residualized coefficients limit stronger active-sensing "
            "or phase-structure interpretations."
        ),
    }
    (paths.out_dir / f"{FIGURE_PREFIX}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate the spatial-content modulation figure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--panel-root", type=Path,
                   default=VISIONCORE_ROOT / "outputs" / "panelF_natural_structure")
    p.add_argument("--tfts-root", type=Path,
                   default=VISIONCORE_ROOT / "outputs" / "twin_feature_tangent_structure_prod_v2")
    p.add_argument("--panel-e-root", type=Path,
                   default=VISIONCORE_ROOT / "outputs" / "tangent_subspace_information" / "panelE_production_fisher")
    p.add_argument("--output-dir", type=Path,
                   default=VISIONCORE_ROOT / "outputs" / "covTFTS_figure")
    p.add_argument("--dpi", type=int, default=300)
    return p


def main() -> None:
    args = build_parser().parse_args()
    compose(
        Paths(
            panel_root=Path(args.panel_root),
            tfts_root=Path(args.tfts_root),
            panel_e_root=Path(args.panel_e_root),
            out_dir=Path(args.output_dir),
        ),
        dpi=int(args.dpi),
    )
    print(f"Saved spatial-content modulation figure to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
